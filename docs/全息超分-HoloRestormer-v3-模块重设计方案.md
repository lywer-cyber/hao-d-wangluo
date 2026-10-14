# HoloRestormer v3 —— Restormer 物理模块重设计（数字全息超分）

> 2026-10-12 · 配套：`全息超分-实验协议-冻结版-v7.md`、`全息超分-HoloRestormer-v2-改进方案.md`
> **本文取代 v2 方案**（v2 的 M1/M2/M5 作为低成本组件保留在 P2 内）。
> 目标读者：实现者（Codex / 工程师）。每个模块都给**可落地的代码骨架、插入点与验收判据**。

---

## §0 一句话讲清 v3 的物理叙事

> **本任务的退化被 `r̂ = 1`（LR 的 Nyquist）这条物理边界切成两个性质完全不同的子问题：**
> **带内**（`r̂<1`）是**复原**问题 —— 去散斑 + 去模糊 + 去噪，信息还在；
> **带外**（`r̂>1`）是**外推**问题 —— 信息已丢，只能靠先验生成。
>
> **而 Restormer 的归纳偏置来自自然图的"加性高斯噪声复原"任务**，对这两者一视同仁。
> 更根本的是：**本任务的噪声是乘性散斑**（`y = x·n`，空间相关、非高斯、非平稳），
> 与 Restormer 的加性假设 + L1 损失是**错配**的。
>
> **v3 就做三件事**：① 把乘性噪声错配修掉；② 把两个子问题分开处理；
> ③ 把**已知的退化算子 `D`** 从损失里搬进网络内部。

---

## §1 本任务的三个物理事实 vs Restormer 的三个错配

| # | 物理事实（有代码可查） | Restormer 的假设 | 错配的后果 |
|---|---|---|---|
| **F1** | 噪声是**乘性散斑**：`add_speckle` 里 `img * speckle`，`contrast∈[0.3,0.8]`、`corr∈[3,6]px`，非高斯、非平稳、空间相关 | 加性高斯 + L1 | 模型按加性噪声的方式去减，**散斑抑制不干净**；暗区（乘性噪声相对更强）被系统性低估 |
| **F2** | 退化算子 `D` **完全已知且可微**：圆孔径低通 → 抗混叠高斯 → 面积平均×4 → 乘性散斑 → 泊松+读出 → 8bit | 黑箱映射，`D` 只在损失里出现 | 白扔了一大块免费先验；网络要靠数据自己学出"低通截止在哪" |
| **F3** | 信息缺失有**硬边界** `r̂ = 1`；两侧是两个不同性质的子问题 | 对全频带一视同仁 | 复原算子被拿去干外推的活，外推算子又被复原任务干扰 |

**v3 的四个模块，一一对应这三条。**

---

## §2 模块总表

| # | 模块 | 对应 | 改 Restormer 的哪儿 | 风险 | 预期 |
|---|---|---|---|---|---|
| **P1** | `HoloStem` 同态双域前端 | F1 | 替换 `OverlapPatchEmbed`（第一级） | 中低 | +0.05~0.20 |
| **P2** | `HoloBlock` 带内-带外解耦块 | F3 | 替换 `TransformerBlock`（全部） | 中 | +0.15~0.40 |
| **P3** | `HoloUnroll` 物理一致性展开 | F2 | **包裹整个网络**（新增外层） | 中高 | **+0.15~0.45** |
| **P4** | `HoloFuse` 融合与 skip 门控 | F1/F3 | 改 U-Net 的 skip 与输出 | 低 | +0.05~0.15 |
| P5* | 频谱外推头（探索项） | F3 | 并联在输出端 | 高 | ? |

\* P5 借鉴 Fourier ptychography 的"带宽外推"，作为探索项，不默认开启。

**叠加预期：+0.4 ~ +1.1 dB**（相对原版 Restormer，同协议下）。

> ⚠️ 协议 v7 那 +0.5~1.8 dB 对所有模型一视同仁，**不是本方法的优势**，别混算。

---

## §3 逐模块详设

### P1 `HoloStem` —— 同态双域前端【对应 F1：乘性散斑】

**问题**：`add_speckle` 是**乘性**的（`holo_degrade.py:107`，`img * speckle`）。
自然图复原网络按**加性**噪声设计。经典做法是取 log 把乘性变加性（`ln y = ln x + ln n`）。

**为什么不能简单加个 log**：文献里对 log 域有明确争议 ——
SAME（Springer）实测 log 域有 **Jensen 不等式偏差**，且**暗区噪声被放大**；
而 arXiv 2605.24716、arXiv 2601.14334、IEEE 11584834 三条工作都用 homomorphic/log 域取得了提升。
**结论：变换方向对，但强度要可调。**

**做法：可学强度的方差稳定化 + 双域并联**

```python
class HoloStem(nn.Module):
    """同态双域前端。

    不做"log 前处理 + exp 后处理"的硬变换（Jensen 偏差 + 暗区噪声放大），
    而是：
      · Box-Cox 变换 z = (x^p − 1)/p，p **可学**：p→0 退化为 log，p=1 退化为恒等；
      · 强度域与变换域**并联**两个 stem，各出 dim/2，拼接。
    网络自己决定变换强度 —— 既拿到了"乘性→加性"的好处，又不会被 log 的偏差坑死。
    """
    def __init__(self, in_chans, embed_dim, use_homomorphic=True):
        super().__init__()
        self.use_homomorphic = use_homomorphic
        if use_homomorphic:
            # p = softplus(p_raw) + 0.05 保证 p > 0，避开 (x^p−1)/p 在 p→0 的数值不稳定
            self.p_raw = nn.Parameter(torch.tensor(-3.0))       # 初始 ≈ log 域
            d = embed_dim // 2
            self.stem_id = nn.Conv2d(in_chans, d, 3, 1, 1)      # 强度域
            self.stem_hm = nn.Conv2d(in_chans, d, 3, 1, 1)      # 变换域
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

    @property
    def p(self):
        return F.softplus(self.p_raw) + 0.05                    # ∈ (0.05, ∞)

    def boxcox(self, x, eps=1e-3):
        x = x.clamp_min(eps)
        return (x.pow(self.p) - 1.0) / self.p

    def forward(self, x):
        if not self.use_homomorphic:
            return self.proj(x)
        return torch.cat([self.stem_id(x), self.stem_hm(self.boxcox(x))], dim=1)
```

**插入点**：`Restormer.forward` 里第一级 `OverlapPatchEmbed`。
**必报**：训练结束后把学到的 `p` 打印出来 —— 若 `p` 收敛到 ~0.05，说明网络**主动选择了 log 域**，
这本身就是一条可写进论文的证据（"散斑的乘性特征被网络自发捕捉"）。

**开关**：`use_homomorphic`　**验收**：`p` 有限、`boxcox` 对 `x∈[1e-3,1]` 输出有限且单调。

---

### P2 `HoloBlock` —— 带内-带外解耦块【对应 F3】

**问题**：`r̂ = 1` 两侧是两个不同性质的子问题，原 `TransformerBlock` 一视同仁。

**做法**：保留 MDTA 原路径，**并联两条各自的频域支路**，用 `HoloBandSplit` 的 mask 把结果
限制在各自频段内；再叠加方向性、多尺度与非线性 V（吸收 v2 的 M1/M2/M5）。

```python
def spectral_mask(t, m):
    """把特征在频域用 mask 限制到指定频段（软 mask，可微）。"""
    with torch.cuda.amp.autocast(enabled=False):          # ★ 频域必须 fp32
        F = torch.fft.fft2(t.float(), norm='ortho')
        out = torch.fft.ifft2(F * m, norm='ortho').real
    return out.to(t.dtype)


class MultiScaleDW(nn.Module):
    """多尺度深度卷积：散斑相关长度 corr∈[3,6] LR 像素 → 对应不同的膨胀率。
    单一 3×3 只能看一个尺度，散斑粒度是有范围的物理量，这里显式覆盖它。"""
    def __init__(self, dim, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, 1, d, dilation=d, groups=dim)
            for d in dilations])
        self.w = nn.Parameter(torch.zeros(len(dilations)))
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        w = torch.softmax(self.w, dim=0)
        out = sum(w[i] * b(x) for i, b in enumerate(self.branches))
        return self.proj(out)


class HoloBlock(nn.Module):
    """Restormer 的 TransformerBlock 重设计。

    ★ 与原路径的关系：MDTA 一行不改，只是并联了两条频域支路。
      （失败过的 band 版本是把 MDTA **替换**掉 —— 那是净的拟合能力损失。）
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 scale=4, n_orient=6, use_restore_branch=True,
                 use_extrap_branch=True, use_dirconv=True, use_multiscale=True,
                 use_nonlinear_v=True):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, use_nonlinear_v=use_nonlinear_v)

        self.split = HoloBandSplit(scale=scale)
        self.use_restore_branch = use_restore_branch
        self.use_extrap_branch = use_extrap_branch
        if use_restore_branch:                       # 带内：复原（去散斑/去模糊/去噪）
            self.norm_r = LayerNorm(dim, LayerNorm_type)
            self.branch_r = FSASBranch(dim)
        if use_extrap_branch:                        # 带外：外推
            self.norm_e = LayerNorm(dim, LayerNorm_type)
            self.branch_e = FSASBranch(dim)

        self.dirc = DirConv(dim, n_orient) if use_dirconv else nn.Identity()
        self.ms = (MultiScaleDW(dim) if use_multiscale
                   else nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))
        self.gate = DualGate(dim)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = HoloGDFN(dim, ffn_expansion_factor, bias, n_orient=n_orient)

    def forward(self, x):
        m_pass, m_edge, m_stop = self.split.masks(
            x.shape[-2], x.shape[-1], x.device, x.dtype)

        f = 0.0
        if self.use_restore_branch:
            r = self.branch_r(self.norm_r(x))
            f = f + spectral_mask(r, m_pass + m_edge)        # 只在带内生效
        if self.use_extrap_branch:
            e = self.branch_e(self.norm_e(x))
            f = f + spectral_mask(e, m_stop)                 # 只在带外生效
        if not isinstance(f, float):
            f = self.ms(self.dirc(f))

        a = self.attn(self.norm1(x))
        if not isinstance(f, float):
            a = self.gate(a, f)                              # 双向门控（方向勿反）
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x
```

**开关**：`use_restore_branch` / `use_extrap_branch` / `use_dirconv` / `use_multiscale` / `use_nonlinear_v`
**验收**：开关全关时，**输出与 `restormer_arch_v1.py` 逐比特一致**（`torch.equal`）。

---

### P3 `HoloUnroll` —— 物理一致性展开【对应 F2，**最强创新**】

**问题**：`D` 完全已知（`holo_degrade.py`），但经典 SR 网络把它扔了，只学黑箱映射。

**做法**：按 **HQS（半二次分裂）** 把 `argmin_x ½‖D(x) − LR‖² + λR(x)` 展开成 K 个阶段：

```
x_{k+1} = Prox_λ( x_k − η_k · D_detᵀ( D_det(x_k) − LR ) )
          └ 可学先验（HoloBlock）┘  └── 数据保真项：**零可学参数** ──┘
```

对照 USRNet（CVPR 2020）：它的 **Data module** 同样是 closed-form、**无可学参数**，
**Prior module** 用 ResUNet，**Hyper-parameter module** 用 MLP 在两者间滑动。
**但 USRNet 处理的是 bicubic/高斯核退化；这里的 `D` 是完整的全息成像退化算子。**

```python
class HoloUnroll(nn.Module):
    """物理一致性展开。K 个 HQS 阶段包在外面。

    ★ 关键工程决策：数据保真项**只包含 D 的确定性部分**
        D_det = 面积平均降采样 ∘ 抗混叠高斯 ∘ 圆孔径低通
      不含散斑、泊松、读出噪声、量化 —— 这些是随机的，逐次前向变化会引入
      噪声梯度、让训练不稳。它们交给 Prox（HoloBlock）去学。
      这个拆分有物理含义：**确定性部分负责约束解，随机部分负责让先验有用武之地**。

    ★ 为什么这条对"必须有效果"最友好：
      数据保真项零可学参数 → **不增加过拟合风险**；
      K 个阶段带来的增益是"物理"给的，不是"容量"给的。
    """
    def __init__(self, dim, K=3, scale=4, learnable_step=True,
                 block_kwargs=None):
        super().__init__()
        self.K = K
        self.D = HoloForwardDet(scale=scale)        # 只含确定性的可微退化算子
        if learnable_step:
            self.eta = nn.Parameter(torch.full((K,), 0.5))
        else:
            self.register_buffer('eta', torch.full((K,), 0.5))
        self.prox = nn.ModuleList([
            HoloBlock(dim=dim, **(block_kwargs or {})) for _ in range(K)])

    def forward(self, x0, lr):
        x = x0
        for k in range(self.K):
            # 数据保真：∇_x ½‖D(x)−LR‖² = Dᵀ(D(x)−LR)   ← 零可学参数
            x = x - self.eta[k] * self.D.adjoint(self.D.forward(x) - lr)
            x = self.prox[k](x)                     # 可学先验
        return x
```

**必须先写好的依赖：`HoloForwardDet`**

```python
class HoloForwardDet(nn.Module):
    """holo_degrade.degrade_physical 的【确定性部分】可微 torch 版。
    只含：圆形孔径低通 → 抗混叠高斯 → 面积平均降采样。
    严格不含散斑/泊松/读出噪声/量化。
    归一化必须与 numpy 版一致：cutoff = diffraction_cutoff / (2*scale)  [cycles/px, HR 网格]
    """
    def __init__(self, scale=4, cutoff=1.0, aa_sigma=0.5, kernel='circle'):
        ...
    def forward(self, x):   # HR -> LR
        ...
    def adjoint(self, y):   # LR -> HR（转置算子，供梯度回传）
        ...
```

**验收（必须做）**：
1. 与 numpy 版 `holo_degrade.degrade_physical` 的**确定性路径**对比，
   同参数下逐像素 **MSE < 1e-4**（`fft_lowpass` + `gaussian_blur` + 面积平均三步）；
2. `adjoint` 的伴随性检验：`<D(x), y> ≈ <x, Dᵀ(y)>`（`atol=1e-3`）；
3. 尺寸：`forward(HR) → LR 尺寸`；`adjoint(LR) → HR 尺寸`。

**⚠️ 两个实现陷阱**
- **LR/HR 尺寸不同**：`forward` 降 4 倍、`adjoint` 升 4 倍。unroll 的输入 `x0` 必须已是 HR 尺寸，
  `lr` 是原始 LR。最省的做法：`x0 = bicubic(lr)`，整条 unroll 在 HR 分辨率上跑。
  **但这会让 60k iter 的显存与时间都上去** —— 先按 K=1 跑通，再逐步加到 K=3。
- **fp32**：FFT 与降采样必须在 fp32 下算（`autocast(enabled=False)`）。

**开关**：`use_unroll`、`unroll_K`
**预期**：+0.15~+0.45 dB（K=3）。**风险**：中高（工程量大），但**收益/风险比最好** ——
因为它加的是"物理"不是"容量"。

---

### P4 `HoloFuse` —— 融合与 skip 门控【对应 F1/F3】

```python
class BandGatedSkip(nn.Module):
    """用解码器当前层的带外能量，决定 skip 里各通道保留多少。

    动机：编码器浅层 skip 的"高频"大半是**散斑噪声与混叠**（LR 在带外没有真信息），
    原样 concat 给解码器等于喂噪声。文献支持这个判断：全息散斑是
    非高斯、非平稳、空间相关的，高斯噪声上预训练的网络直接失效
    （Swin-UNet, Opt. Express 32(19):33465）。
    """
    def __init__(self, dim, scale=4):
        super().__init__()
        self.split = HoloBandSplit(scale)
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, max(dim // 4, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(dim // 4, 8), dim, bias=False),
            nn.Sigmoid())

    def forward(self, skip, dec):
        B, C = dec.shape[0], dec.shape[1]
        e = self.split.band_energy(dec)                     # (B, C, 3)
        w = self.proj(e.reshape(B, C * 3)).view(B, C, 1, 1)
        return skip * w
```

**插入点**：`Restormer.forward` 的 decoder，在 `torch.cat([...])` **之前**。
**输出头**：强度图非负，输出建议用 `x_hr = x_lr_up + delta`，其中 `delta` 不限制，
最终评估前由协议统一 `clamp(0,1)`（协议 §12.1）。

---

### P5 频谱外推头（**探索项，不默认开启**）

借鉴 Fourier ptychography 的**带宽外推**：
FPMUP（Opt. Express 2022）证明网络**可以预测合成孔径之外的傅里叶谱**，从而消除截止振铃。

简化实现：在输出端并联一个小头，只对 `r̂>1` 区域预测一个**幅度包络**
（例如可学的幂律 `|F(r)| ∝ r^(-α)`），相位由带内相位延拓。

⚠️ **风险高、参考文献少、且我们不是 ptychography（没有多帧合成孔径）** ——
**先不做**。等 P1–P4 都有结论后，如果还需要故事再回来考虑。

---

## §4 实施顺序与门禁

### 4.1 顺序（按"收益/风险比"排）

| 阶段 | 做什么 | 成本 | 为什么这个顺序 |
|---|---|---|---|
| **S1** | 备份 `restormer_arch_v1.py`；实现 `holo_components.py`（P2 的零件） | 半天 | 地基 |
| **S2** | **P2 的零件逐个 A/B**：`use_nonlinear_v` → `use_multiscale` → `use_dirconv` | 3 × 1.5 h | 低风险，先确认"物理先验注入"这条思路成立 |
| **S3** | **P2 两条支路 A/B**：`use_restore_branch` / `use_extrap_branch`（先各单独，再一起） | 3 × 1.7 h | 中风险，核心结构 |
| **S4** | **P1 `HoloStem` A/B**（看学到的 `p`） | 1.7 h | 中低风险，物理理由最硬 |
| **S5** | **`HoloForwardDet` + 一致性校验**（不训练，纯数值） | 半天 | P3 的前置，必须先过 |
| **S6** | **P3 `HoloUnroll` K=1 → K=3** | 2 × 2 h | 高创新，但依赖 S5 |
| **S7** | **P4 skip 门控** | 1.7 h | 低风险 |

**合计门禁成本 ≈ 15 h（不到一天）。** 跑完就知道哪些有用。

### 4.2 A/B 门禁判据（逐项过，不许合并）

```
每项单独开一个 network_name，跑 5000 iter、seed 10，与「同配置但该项关闭」对比：

  通过判据（两条都要满足）：
    ① Δ val PSNR ≥ +0.05 dB
    ② 训练 l_pix 不劣化（同窗口均值相对差 ≤ +2%）
       ★ ② 是 band 那次的直接教训：它 val 低、训练 loss 也高 → 欠拟合而非过拟合

  不通过 → 回退该项，如实记录，不进主表
```

### 4.3 反"调参救场"

首次不涨时：**只允许查一次实现 bug**（形状、fp32、mask 是否非空、门控方向、是否误替换原路径）。
实现无误则**不允许反复调超参直到涨**，标记为"本域无收益"并如实写进消融表。

---

## §5 为什么不会重蹈 `band` 的覆辙

`band` 实测：训练 `l_pix` 全程高 5~10%、val PSNR 低 0.6 dB → **净的拟合能力损失**。

| `band` 的坑 | v3 的做法 |
|---|---|
| **替换**掉 MDTA（C 个通道坍缩到 6 个 token） | **并联**，MDTA 一行不改（P2） |
| 6 个 token 的注意力 → 信息瓶颈 | `FSASBranch` 逐元素乘，**全分辨率、不坍缩** |
| 按等半径分环 → 低频主导 | 按**物理截止频率** `f_LR` 分带，带边单独成带 |
| 没有空间局部算子（纯 FFT + 1×1 conv） | `FSASBranch` 保留 3×3 dwconv；另有 `DirConv` / `MultiScaleDW` |
| 只在权重最高的模块上赌一把 | P1–P4 分四个独立开关，逐个过门禁 |
| 涨不涨只能事后看 | **每项 5k A/B 门禁**（§4.2） |

---

## §6 与已发表工作的边界（**投稿前必须认清**）

| 已发表 | 做了什么 | 我们的差异 |
|---|---|---|
| **USRNet** (CVPR 2020) | HQS 深度展开，Data module（closed-form 零参数）+ Prior(ResUNet) + Hyper-param(MLP)；处理 bicubic/高斯核 | ⚠️ **最需要区分**。差异：我们的数据保真项是**全息成像退化算子**（圆孔径+抗混叠+面积平均），且**只取确定性部分**、随机项交给先验 |
| **DPIR** | Deep PnP（DRUNet）+ 闭式数据子问题 | 同上 |
| **AdaIR** (ICLR 2025) | Restormer + **自适应**频域学习（FFT 掩码学出来的） | 我们的带边界是**算出来的物理量** `f_LR = 1/(2·scale)` |
| **SwinFIR** (arXiv 2208.11247) | SwinIR + Fast Fourier Convolution（**+0.8 dB**） | 通用自然图；我们按物理带解耦做**复原 vs 外推**两条路 |
| **DFAN** (Symmetry 2025) | SWT 小波 + Swin 走低频 + 方向卷积走高频 | ⚠️ **很接近 P2**。差异：它用固定小波基；我们按物理截止频率锚定，且**显式分开"复原"与"外推"两种子问题** |
| **FPMUP** (Opt. Express 2022) | 无训练先验网络预测合成孔径外的傅里叶谱（"spectral extrapolation"） | 它是**多帧** ptychography；我们是单帧强度图 SR。作为 P5 的依据 |
| **同态/log 域去斑** (arXiv 2605.24716 / 2601.14334 / IEEE 11584834) | log 域把乘性散斑变加性 | **它是预处理+后处理**；我们把可学强度的 Box-Cox 做成**网络前端的一部分** |
| ⚠️ **SAME** (Springer) | **反对** log 域：Jensen 偏差 + 暗区噪声放大 | 我们承认这条，所以做的是**可学强度的 Box-Cox**（p→1 退化为恒等），不是硬 log |
| **UDAT** (IEEE 2025) | 给 Restormer 补密集窗口通道注意力（+0.57 dB） | "补局部性"已发表，只作为组件依据 |
| **FFTformer** (CVPR 2023) | FSAS 频域自注意力算子 | **借用**，必须引用 |

**一句话立论**：
> 别人的频带划分是**学出来的**（AdaIR 的自适应掩码）或**固定基分解的**（DFAN 的 SWT）；
> 别人的物理约束**只出现在损失里**（绝大多数 SR 工作）；
> **我们把已知的成像退化算子搬进了网络内部，并按物理截止频率把"复原"与"外推"分成两条路。**

---

## §7 给 Codex 的实现清单

### 7.1 第一步：读文件（**不要急着改代码**）

```
必读，按顺序：
  1. ~/桌面/10.11/全息超分-实验协议-冻结版-v7.md   ← 52 条规则，冻结，有冲突以它为准
  2. ~/桌面/10.11/全息超分-论文出图规范-v7.md       ← 出图规范 + per_sample_metrics.json 契约
  3. ~/桌面/10.11/experiment_config.txt             ← 配置正文
  4. 本文《全息超分-HoloRestormer-v3-模块重设计方案.md》
  5. ~/holo-sr/scripts/holo_degrade.py              ← 退化算子（P3 的 HoloForwardDet 要照着它写）
```

### 7.2 实施清单

- [ ] **S1 备份**：`restormer_arch.py` → `restormer_arch_v1.py`，保证原版仍可注册
      （论文里原版 Restormer 是干净对照，必须能跑）
- [ ] **S1 组件库**：`basicsr/archs/holo_components.py`：
      `HoloBandSplit` / `FSASBranch` / `DirConv` / `DualGate` / `MultiScaleDW` / `spectral_mask`
- [ ] **S2** `Attention` 支持 `use_nonlinear_v`；`FeedForward` → `HoloGDFN`（带 `DirConv` + 带门控）
- [ ] **S3** `HoloBlock`（P2 两条支路）
- [ ] **S4** `HoloStem`（P1，替换第一级 `OverlapPatchEmbed`）
- [ ] **S5** `HoloForwardDet`（只含确定性部分的退化算子 + 伴随）
- [ ] **S6** `HoloUnroll`（P3，K 可调）
- [ ] **S7** `BandGatedSkip`（P4）+ 输出头
- [ ] **S8 组装**：`HoloRestormerV3`，注册进 basicsr，`network_type=HoloRestormerV3`

### 7.3 验收判据（**硬门槛**）

1. **开关全关时，输出与 `restormer_arch_v1.py` 逐比特一致**（`torch.equal`）。
   → **这一条不过，后面都不用做。**
2. 开关全开：`(1,3,64,64) → (1,3,256,256)`，`torch.isfinite(out).all()`
3. `HoloBandSplit.masks` 三个 mask **之和恒为 1**（`atol=1e-5`）
4. `DirConv` / `MultiScaleDW` 的 softmax 权重和为 1；梯度能正常回传
5. **所有 FFT 与降采样在 fp32 下**（`torch.cuda.amp.autocast(enabled=False)` + 显式 `.float()`）
6. `HoloForwardDet`：与 numpy 版确定性路径 **MSE < 1e-4**；`adjoint` 满足 `<D(x),y> ≈ <x,Dᵀ(y)>`（`atol=1e-3`）
7. `HoloStem.p` 训练后打印出来（若收敛到 ~0.05，说明网络选了 log 域 —— 这是论文证据）
8. 报告**参数量与 FLOPs 变化**（×4 输出 `3×512×512`），写进 `run_manifest.json`
9. `DualGate` 门控方向：**C-Map 调 attention、S-Map 调 freq**（写反了会莫名变差）

### 7.4 硬约束

- [ ] 除 `network_name` / `network_type` / `network_params` 外，不改配置文件任何字段
- [ ] 不用预训练权重；不加梯度裁剪；不给单个模型单独调 lr/betas
- [ ] 每个变体一个独立 `network_name`，目录带 `seed{seed}` 层
- [ ] 原版 Restormer 保持可跑
- [ ] 不把 `meta` 喂给网络（协议 §2.3：v7 不启用）
- [ ] P3 的 `HoloForwardDet` **绝不能**把 `holo_degrade.py` 的 `rng` 调用顺序改掉
      （协议 §2.2 的红线；只允许**新增**一个独立的 torch 模块，不动原 numpy 文件）

---

## §8 文献依据

| 用于 | 文献 |
|---|---|
| **P3 深度展开的方法论** | **USRNet** (CVPR 2020) · [arXiv 2003.10428](https://arxiv.org/pdf/2003.10428.pdf) · [code](https://github.com/cszn/USRNet) |
| P3 的 PnP 变体 | **DPIR** · cszn/DPIR；CVPR 2025 Proximal Algorithm Unrolling |
| **P1 乘性散斑的同态处理** | Sonar Despeckling（homomorphic 残差一致性）· [arXiv 2605.24716](https://export.arxiv.org/pdf/2605.24716) |
| P1 的 log 域变体 | ⁴S⁴DM · [arXiv 2601.14334](https://browse-export.arxiv.org/pdf/2601.14334)（log + Log-Yeo-Johnson） |
| P1 的方差稳定化 | VSCPT-CWA · [IEEE 11584834](https://ieeexplore.ieee.org/document/11584834) |
| ⚠️ **P1 的反面证据（必须回应）** | SAME · [Springer](https://link.springer.com/article/10.1007/s44443-026-00789-1)（log 域的 Jensen 偏差 + 暗区噪声放大） |
| **P5 频谱外推的依据** | FPMUP · [Opt. Express 30(22):39597](https://pubmed.ncbi.nlm.nih.gov/36298907/)（网络可预测合成孔径外的傅里叶谱） |
| P2 的频域算子 | FFTformer, CVPR 2023 · [arXiv 2211.12250](https://arxiv.org/abs/2211.12250) |
| P2 的方向性 | DFAN, Symmetry 17:1175, 2025 |
| P2 的先例 | SwinFIR · [arXiv 2208.11247](https://ar5iv.labs.arxiv.org/html/2208.11247)（+0.8 dB） |
| P2 的 value 通路去线性化 | ICLR 2026 · [OpenReview](https://openreview.net/pdf?id=IBzmQVia88) |
| ⚠️ 必须区分（Restormer + 自适应频域） | AdaIR, ICLR 2025 |
| P4 的散斑特性依据 | Swin-UNet 散斑去噪, Opt. Express 32(19):33465 |
| 门控方向（勿反） | DAT · `dat_arch.py:422-435` |
| 域内对照 | SEMNet (IEEE TIM 2026, 75:4503013)、HoloMamba (IEEE TCI 2026, 12:935–945) |

---

## §9 一页速览

```
v3 的物理叙事：
  退化被 r̂=1 切成两个子问题 —— 带内=复原（信息还在），带外=外推（信息已丢）。
  Restormer 的归纳偏置来自"加性高斯噪声的自然图复原"，对本任务有三个错配：
    F1 噪声是【乘性散斑】 y=x·n     ← Restormer 假设加性
    F2 退化算子 D【完全已知且可微】  ← 经典 SR 只把它放进损失，没放进网络
    F3 信息缺失有【硬边界 r̂=1】     ← Restormer 对全频带一视同仁

四个模块一一对应：
  P1 HoloStem    可学强度的 Box-Cox 双域前端     修 F1   +0.05~0.20  中低
  P2 HoloBlock   带内/带外双支路 + 方向 + 多尺度  修 F3   +0.15~0.40  中
  P3 HoloUnroll  HQS 展开，数据保真项零可学参数   修 F2   +0.15~0.45  中高 ★
  P4 HoloFuse    skip 带能量门控                  修 F1/F3 +0.05~0.15  低
  P5 频谱外推头（探索，不默认开）

P3 是最强创新点，理由：加的是【物理】不是【容量】——
  数据保真项 Dᵀ(D(x)−LR) 零可学参数 → 不增加过拟合风险；
  对照 USRNet 的 Data module，但 D 是【全息成像退化算子】而非 bicubic 核。

实施顺序（门禁成本合计 ≈ 15 h，不到一天）：
  S1 备份+组件库 → S2 零件逐个 A/B（非线性V/多尺度/方向）
  → S3 两条支路 A/B → S4 HoloStem → S5 HoloForwardDet 一致性校验
  → S6 HoloUnroll（K=1→3）→ S7 skip 门控

门禁判据（逐项，不许合并）：ΔPSNR ≥ +0.05 且 训练 l_pix 不劣化
  （第二条是 band 的教训：它 val 低、训练 loss 也高 → 欠拟合）

绝对不要：
  替换原模块 / 坍缩 token / 按等半径分环 / 在 fp16 下算 FFT /
  把散斑等随机项放进数据保真项（会引入噪声梯度）/
  改动 holo_degrade.py 里的 rng 调用顺序 / 把协议层增益算成自己的优势
```
