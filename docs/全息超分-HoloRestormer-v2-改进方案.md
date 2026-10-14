# HoloRestormer v2 —— Restormer 逐模块重设计（数字全息超分）

> ⚠️ **本方案已被《全息超分-HoloRestormer-v3-模块重设计方案.md》取代。**
> v3 的叙事更强（乘性散斑错配 / 已知退化算子入网 / 带内-带外解耦）；
> 本文的 M1 / M2 / M5 作为低成本组件保留在 v3 的 P2 内。保留本文仅供追溯。
>
> 2026-10-12 · 配套：`全息超分-实验协议-冻结版-v7.md`、`全息超分-论文出图规范-v7.md`
> 目标读者：实现者（Codex / 工程师）。每个模块都给**可落地的代码骨架与插入点**。
>
> **这是论文的核心创新点。** Restormer（原版，原封不动）是本方法的干净对照；
> HoloRestormer v2 是把 Restormer 的模块逐个按全息物理重设计后的结果。

---

## §0 结论速览

### 0.1 改造点清单（7 项，按风险从低到高）

| # | 改哪个模块 | 改成什么 | 物理依据 | 预期 | 风险 |
|---|---|---|---|---|---|
| **M1** | `Attention`(MDTA) 的 **value 通路** | 线性 → 非线性 | 通用（ICLR'26） | +0.05~0.20 | **低** |
| **M2** | `FeedForward`(GDFN) 的 **3×3 dwconv** | → 方向性可分离卷积 `DirConv` | 全息条纹有主方向 | +0.05~0.20 | **低** |
| **M3** | `FeedForward`(GDFN) 的 **门控** | 乘一个"带能量"门控 | SR 补的是带外高频 | +0.05~0.20 | **低** |
| **M4** | `TransformerBlock` | **并联**一个频域分支 + 双向门控 | 见 §0.2 | **+0.15~0.40** | 中 |
| **M5** | `Attention`(MDTA) 的**输入** | 带锚定通道调制 | 已知截止频率 `f_LR` | +0.05~0.15 | 低 |
| **M6** | `OverlapPatchEmbed` | 拼接"越界高频提示图" | LR 在 `r̂>1` 上是噪声 | +0.03~0.15 | 中 |
| **M7** | U-Net **skip 连接** | 带能量门控 | 浅层 skip 的高频大半是噪声 | +0.05~0.15 | 中 |

**叠加预期：+0.3 ~ +0.8 dB**（相对原版 Restormer，同协议下）。

> ⚠️ **不要和协议层面的增益混算。** 协议 v7 的改动（跑完 LR 衰减 + EMA + 频域损失 + D4 自集成）
> 对**所有模型**一视同仁，属于"基线水位"，不是本方法的优势。本方法的优势只有上表这 +0.3~0.8 dB。

### 0.2 一句话讲清核心创新（M4）

> **全息成像的退化在频域有一条由 NA、波长、采样率决定的硬边界 `f_LR = 1/(2·scale)`。
> Restormer 的注意力只在通道维交互，完全看不见这条边界。M4 在不动原注意力路径的前提下，
> 并联一个以 `f_LR` 为锚、分带处理、方向感知的频域分支。**

### 0.3 与"已发表工作"的边界（**写论文前必须认清**）

| 已发表 | 做了什么 | 我们的差异 |
|---|---|---|
| **AdaIR** (ICLR 2025) | Restormer + Adaptive Frequency Learning（FFT 分解 + **自适应掩码** + 交叉注意力） | ⚠️ "Restormer + 频域分支"**已发表**。差异：它的带区是**学出来的**，我们的边界是**算出来的物理量** |
| **SwinFIR** (arXiv 2208.11247) | SwinIR 换入 Fast Fourier Convolution，**+0.8 dB** | 通用自然图；我们的边界锚在成像系统截止频率，且带内做方向性分解 |
| **DFAN** (Symmetry 2025) | SWT 小波 + Swin 走低频 + 方向卷积走高频 | ⚠️ **最接近**。它用固定小波基；我们按物理截止频率锚定，并**显式处理"带边"**这一信息最不确定的区域 |
| **UDAT** (IEEE 2025) | 给 Restormer 补密集窗口通道注意力 | "补局部性"已发表，只作为组件依据 |
| **FFTformer** (CVPR 2023) | FSAS 频域自注意力算子 | 我们**借用**该算子（§2.2），必须引用 |

**一句话立论**：
> 别人的频带划分是**学出来的**（AdaIR 的自适应掩码）或**固定基分解的**（DFAN 的 SWT）；
> 我们的频带边界是**成像系统的物理截止频率** —— 由 NA、波长与采样率决定，逐样本可算。

---

## §1 Restormer 原结构逐模块解剖

```
Restormer = 4 级 U-Net，scale=4
  Encoder(4 级) → Bottleneck → Decoder(3 级) → Upsample(pixelshuffle) → 输出

  Encoder 第 i 级:
      OverlapPatchEmbed(3×3 conv, stride=1 if i==0 else 2) → LayerNorm
      × N_i 个 TransformerBlock            N = [4, 6, 6, 8]
      (i < 3 时接 Downsample)

  TransformerBlock:
      x = x + Attention( LayerNorm1(x) )        ← MDTA
      x = x + FeedForward( LayerNorm2(x) )      ← GDFN

  Attention (MDTA, restormer_arch.py):
      LayerNorm → conv1x1 → conv3x3(dw, groups=3C) → q,k,v
      q,k 归一化 → attn = (q @ kᵀ) * temperature → softmax
      out = attn @ v → conv1x1
      ★ 注意力在【通道维】做（C×C），空间维被 flatten 掉

  FeedForward (GDFN):
      LayerNorm → conv1x1 (扩张 2.66×) → conv3x3(dw) → chunk 成两半
      out = GELU(x1) * x2 → conv1x1 (投影回)

  Downsample: conv3x3 stride2 ；Upsample: conv3x3 + PixelShuffle
```

**关键类**：`Restormer` / `TransformerBlock` / `Attention` / `FeedForward` /
`OverlapPatchEmbed` / `Downsample` / `Upsample` / `LayerNorm`

### 1.1 从全息视角看，原设计缺什么

| 原设计 | 在全息域的问题 |
|---|---|
| MDTA 只在**通道维**做注意力 | 通道是语义抽象；全息里真正有物理意义的是**频带**，而频带结构被完全忽略 |
| GDFN 用**各向同性** 3×3 dwconv | 全息条纹有明确主方向，各向同性核会把条纹横向糊掉 |
| `OverlapPatchEmbed` 直接吃 `(B,3,H,W)` | 不知道"LR 在 `r̂>1` 上没有真实信息"，会去拟合噪声/混叠 |
| U-Net skip 把编码器特征原样送到解码器 | 浅层 skip 的"高频"大半是散斑噪声与混叠，原样送过去等于喂噪声 |
| 全局没有 `f_LR` 这个概念 | **退化算子有一条由采样率决定的硬边界，网络对此一无所知** |

---

## §2 组件库（新建 `basicsr/archs/holo_components.py`）

### 2.1 `HoloBandSplit` —— 物理带锚定频谱划分

```python
class HoloBandSplit(nn.Module):
    """按【已知的物理截止频率】把特征频谱划成三段。

    f_LR = 1/(2·scale) cycles/px（HR 网格）—— 由采样率决定，不是学出来的。
    记 r̂ = r / f_LR，则 r̂ = 1 恰好是 LR 的 Nyquist：
        r̂ < 1-ε   passband    信息还在，任务是去散斑/去模糊/去噪
        |r̂-1| < ε band-edge   信息在此被截断，最不确定、最需要约束 ★核心
        r̂ > 1+ε   stopband    信息已丢失，纯外推
    """
    def __init__(self, scale: int = 4, edge: float = 0.15, softness: float = 12.0):
        super().__init__()
        self.f_lr = 1.0 / (2.0 * scale)
        self.edge, self.softness = edge, softness

    def masks(self, H, W, device, dtype):
        fy = torch.fft.fftfreq(H, device=device, dtype=dtype)[:, None]
        fx = torch.fft.fftfreq(W, device=device, dtype=dtype)[None, :]
        r = torch.sqrt(fy ** 2 + fx ** 2) / self.f_lr          # r̂
        k = self.softness / max(self.edge, 1e-6)
        lo, hi = 1.0 - self.edge, 1.0 + self.edge
        m_pass = torch.sigmoid(k * (lo - r))                   # 软边界，避免梯度断裂
        m_stop = torch.sigmoid(k * (r - hi))
        m_edge = (1.0 - m_pass - m_stop).clamp(0, 1)
        return m_pass, m_edge, m_stop                          # 三者之和恒为 1

    def band_energy(self, x):
        """逐通道的三带能量 (B, C, 3)，供各种门控使用。"""
        F = torch.fft.fft2(x.float(), norm='ortho')            # ★ 必须 fp32
        p = F.real ** 2 + F.imag ** 2
        ms = self.masks(x.shape[-2], x.shape[-1], x.device, torch.float32)
        return torch.stack([(p * m).mean(dim=(-2, -1)) for m in ms], dim=-1)
```

### 2.2 `FSASBranch` —— 全分辨率频域交互（**不坍缩 token**）

```python
class FSASBranch(nn.Module):
    """A = iFFT( FFT(Q) ⊙ conj(FFT(K)) )，再与 V 逐元素乘。

    来源：FFTformer (CVPR 2023) 的 FSAS —— 卷积定理把"相关性"变成频域逐元素乘。
    ★ 与失败过的 BandAttention 的本质区别：
      · 空间复杂度 O(N)、时间 O(N log N)，**保持全分辨率**
      · 把 C 个通道坍缩成 6 个 token 的那一步被彻底去掉
      · 内部保留 3×3 depthwise conv，局部先验不丢
    """
    def __init__(self, dim, bias=False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        q, k, v = self.qkv(self.dw(self.norm(x))).chunk(3, dim=1)
        with torch.cuda.amp.autocast(enabled=False):           # ★ 频域必须 fp32
            qf = torch.fft.rfft2(q.float(), norm='ortho')
            kf = torch.fft.rfft2(k.float(), norm='ortho')
            A = torch.fft.irfft2(qf * torch.conj(kf),
                                 s=x.shape[-2:], norm='ortho')
        return self.proj(A.to(x.dtype) * v)
```

### 2.3 `DirConv` —— 方向性可分离卷积

```python
class DirConv(nn.Module):
    """k 个方向的细长核，权重由 softmax 归一、可学。

    理由：全息条纹有明确主方向；各向同性 3×3 会把条纹横向糊掉。
    偶数索引 = 水平细长 (1×k)，奇数 = 垂直细长 (k×1)。
    这是"方向子带"的低成本近似；昂贵版（从结构张量估 θ 再旋转采样）
    留作可选升级，先不做。
    """
    def __init__(self, dim, n_orient=6, k=7, bias=False):
        super().__init__()
        assert n_orient % 2 == 0, '方向数取偶数'
        self.n_orient = n_orient
        self.branches = nn.ModuleList([
            nn.Conv2d(dim, dim,
                      (1, k) if i % 2 == 0 else (k, 1),
                      padding=(0, k // 2) if i % 2 == 0 else (k // 2, 0),
                      groups=dim, bias=bias)
            for i in range(n_orient)])
        self.w = nn.Parameter(torch.zeros(n_orient))        # softmax 后初始为均匀

    def forward(self, x):
        outs = torch.stack([b(x) for b in self.branches], dim=0)   # (K,B,C,H,W)
        w = torch.softmax(self.w, dim=0).view(-1, 1, 1, 1, 1)
        return (outs * w).sum(dim=0)
```

### 2.4 `DualGate` —— 双向门控（**方向不能反**）

```python
class DualGate(nn.Module):
    """C-Map（全局池化）调注意力分支；S-Map（1×1 conv）调频域分支。

    ⚠️ 方向与 DAT 原码一致（dat_arch.py:422-435）。写反了实验会莫名变差且极难排查。
    """
    def __init__(self, dim):
        super().__init__()
        self.cmap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1))
        self.smap = nn.Conv2d(dim, dim, 1)

    def forward(self, attn, freq):
        a = attn * torch.sigmoid(self.cmap(freq))   # 频域提示"哪些通道重要"
        f = freq * torch.sigmoid(self.smap(attn))   # 空域提示"哪些位置重要"
        return a + f
```

---

## §3 逐模块改造（M1–M7）

### M1 —— `Attention`(MDTA) 的 value 通路去线性化 ⭐ 最低风险，先做

**原代码**
```python
qkv = self.qkv_dwconv(self.qkv(x))
q, k, v = qkv.chunk(3, dim=1)
```
`v` 从输入到输出是**纯线性**的 → 通道注意力退化成"线性选择器"。

**改造**（ICLR 2026, OpenReview `IBzmQVia88`：value 通路纯线性是 Restormer 通道注意力的瓶颈之一）

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, use_nonlinear_v=False):
        ...
        self.use_nonlinear_v = use_nonlinear_v
        if use_nonlinear_v:
            # 把 v 从 qkv 里拆出来，单独加非线性
            self.qk = nn.Conv2d(dim, dim * 2, 1, bias=bias)
            self.qk_dwconv = nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2, bias=bias)
            self.v1 = nn.Conv2d(dim, dim, 1, bias=bias)
            self.v_dw = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=bias)
            self.v2 = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        if self.use_nonlinear_v:
            q, k = self.qk_dwconv(self.qk(x)).chunk(2, dim=1)
            v = self.v2(F.gelu(self.v_dw(self.v1(x))))     # ← 非线性 value
        else:
            qkv = self.qkv_dwconv(self.qkv(x)); q, k, v = qkv.chunk(3, dim=1)
        ...  # 后面的注意力计算完全不变
```

**为什么有效**：注意力权重仍决定"选哪些通道"，但被选中的通道现在能经过非线性变换。
**开关**：`use_nonlinear_v`　**消融**：开/关　**预期**：+0.05~+0.20 dB

---

### M2 —— `FeedForward`(GDFN) 的 dwconv → `DirConv` ⭐ 低风险

**原代码**
```python
x1, x2 = self.dwconv(x).chunk(2, dim=1)     # dwconv = 3×3 depthwise，各向同性
x = F.gelu(x1) * x2
```

**改造**
```python
class HoloGDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias,
                 use_dirconv=True, n_orient=6, k=7):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = (DirConv(hidden * 2, n_orient, k) if use_dirconv
                       else nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1,
                                      groups=hidden * 2, bias=bias))
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        h = self.project_in(x)
        x1, x2 = self.dwconv(h).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)
```

**为什么有效**：`DirConv` 沿条纹方向采样 → 条纹结构被保留而不是被糊掉。
**开关**：`use_dirconv`　**预期**：+0.05~+0.20 dB
**注意**：`n_orient` 取 6 时参数量增加 `hidden*2*dim*k*3`，要报出来（协议 §16.3 #6）。

---

### M3 —— `FeedForward`(GDFN) 的门控加上"带能量" ⭐ 低风险、任务对齐

**动机**：SR 要补的就是**带外**（`r̂>1`）那部分。原门控 `GELU(x1) * x2` 是纯空间门控，
对"该不该放大高频"没有任何感知。

**改造**（接在 M2 之后）
```python
class HoloGDFN(nn.Module):
    def __init__(self, ..., use_band_gate=True, scale=4):
        ...
        if use_band_gate:
            self.split = HoloBandSplit(scale)
            self.band_proj = nn.Sequential(
                nn.Linear(hidden * 3, max(hidden // 4, 8), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(max(hidden // 4, 8), hidden, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        h = self.project_in(x)
        x1, x2 = self.dwconv(h).chunk(2, dim=1)
        g = F.gelu(x1)
        if self.use_band_gate:
            B, C = x1.shape[0], x1.shape[1]
            e = self.split.band_energy(x1)                    # (B, C, 3)
            w = self.band_proj(e.reshape(B, C * 3)).view(B, C, 1, 1)
            g = g * w                                         # ← 带能量门控
        return self.project_out(g * x2)
```

**为什么有效**：让 FFN 第一次能按"这个通道在带外有多少能量"来决定放行多少 ——
与 SR 的任务目标（补带外高频）直接对齐。
**开关**：`use_band_gate`　**预期**：+0.05~+0.20 dB

---

### M4 —— `TransformerBlock` 并联频域分支 ⭐⭐ **核心创新**

**动机**：MDTA 在通道维交互，看不见频率结构。**并联**（不是替换！）一个以 `f_LR` 为锚的频域分支。

**改造**
```python
class HoloTransformerBlock(nn.Module):
    """在 Restormer 的 TransformerBlock 上并联频域分支。

    ★ 与原路径的关系：MDTA 与 GDFN 一行不改，只是多了一条并联支路。
      （失败过的 band 版本是把 MDTA **替换**掉，那是净的拟合能力损失。）
    """
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 use_freq_branch=True, use_nonlinear_v=False,
                 use_dirconv=True, use_band_gate=True, n_orient=6, scale=4):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, use_nonlinear_v=use_nonlinear_v)

        self.use_freq_branch = use_freq_branch
        if use_freq_branch:
            self.norm_f = LayerNorm(dim, LayerNorm_type)
            self.freq = FSASBranch(dim)          # §2.2 全分辨率频域交互
            self.dirc = DirConv(dim, n_orient)   # §2.3 方向性
            self.gate = DualGate(dim)            # §2.4 双向门控（方向勿反）

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = HoloGDFN(dim, ffn_expansion_factor, bias,
                            use_dirconv=use_dirconv, n_orient=n_orient,
                            use_band_gate=use_band_gate, scale=scale)

    def forward(self, x):
        # ---- 原路径（一行不改）----
        a = self.attn(self.norm1(x))
        # ---- 新增并联支路 ----
        if self.use_freq_branch:
            f = self.dirc(self.freq(self.norm_f(x)))
            a = self.gate(a, f)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x
```

**为什么有效**
- 频域分支内部的 `FSASBranch` 提供**全图**的频率交互（窗口注意力给不了）；
- `DirConv` 保证方向感知；
- `DualGate` 让两条路互相调制 —— 原注意力告诉频域分支"哪里重要"，频域分支告诉原注意力"哪些通道重要"。

**开关**：`use_freq_branch`　**预期**：+0.15~+0.40 dB　**风险**：中
> **这是唯一一个"增加容量"的改造**，也是最可能翻车的一项 —— 必须过 §5 的 A/B 门禁。
> 若 A/B 显示训练 `l_pix` 变高（像 band 那次），说明是欠拟合 → 检查是否误把原路径替换了、
> 或 FSAS 是否在 fp16 下算（fp16 FFT 精度不足会导致训练不稳）。

---

### M5 —— MDTA 输入侧的"带锚定通道调制" ⭐ 低风险

**动机**：把 `f_LR` 这个物理量以最便宜的方式注入原注意力 —— 不碰注意力计算，只调制它的输入。

```python
class Attention(nn.Module):
    def __init__(self, ..., use_band_mod=True, scale=4):
        ...
        if use_band_mod:
            self.split = HoloBandSplit(scale)
            self.band_mod = nn.Sequential(
                nn.Linear(dim * 3, max(dim // 4, 8), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(max(dim // 4, 8), dim, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        if self.use_band_mod:
            B, C = x.shape[0], x.shape[1]
            w = self.band_mod(
                self.split.band_energy(x).reshape(B, C * 3)).view(B, C, 1, 1)
            x = x * w                                   # ← 只调制输入
        qkv = self.qkv_dwconv(self.qkv(x))
        ...  # 注意力计算完全不变
```

**开关**：`use_band_mod`　**预期**：+0.05~+0.15 dB
> ⚠️ 不要和 M1 的 `use_nonlinear_v` 同一次 A/B 打开 —— 两项要**分别**验证贡献（消融矩阵见 §6）。

---

### M6 —— `OverlapPatchEmbed` 拼接"越界高频提示图"

**动机**：LR 在 `r̂>1` 的频带上**没有真实信息**，那里的能量是散斑噪声与混叠。
网络若把它当真信号去拟合，PSNR 上不去、视觉还会出现伪结构。

```python
class HoloPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, scale=4, use_hint=True):
        super().__init__()
        self.use_hint = use_hint
        self.split = HoloBandSplit(scale)
        cin = in_chans + (1 if use_hint else 0)
        self.proj = nn.Conv2d(cin, embed_dim, 3, 1, 1)

    def _hint(self, x):
        """把输入里落到 stopband（r̂>1）的分量抽出来，取通道均值当一张提示图。"""
        m_stop = self.split.masks(x.shape[-2], x.shape[-1], x.device, torch.float32)[2]
        with torch.cuda.amp.autocast(enabled=False):
            F = torch.fft.fft2(x.float(), norm='ortho')
            hi = torch.fft.ifft2(F * m_stop, norm='ortho').real
        return hi.abs().mean(dim=1, keepdim=True).to(x.dtype)

    def forward(self, x):
        if self.use_hint:
            x = torch.cat([x, self._hint(x)], dim=1)   # (B, C+1, H, W)
        return self.proj(x)
```

**为什么有效**：显式告诉网络"这部分是采集噪声，不是信号"。属于把**退化算子的知识前置到输入端**。
**开关**：`use_hint`　**预期**：+0.03~+0.15 dB　**风险**：中（新输入通道需要重新收敛）

---

### M7 —— U-Net skip 连接的带能量门控

**动机**：编码器浅层的 skip 特征里，**"高频"大半是散斑噪声和混叠**（因为 LR 在带外没有真信息）。
原样 concat 给解码器 = 喂噪声。文献也支持这个判断：全息散斑是**非高斯、非平稳、空间相关**的，
高斯噪声上预训练的网络直接失效（Swin-UNet, Opt. Express 32(19):33465）。

```python
class BandGatedSkip(nn.Module):
    """用解码器当前层的带外能量，决定 skip 里各通道保留多少。"""
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

**插入点**：`Restormer.forward` 的 decoder 部分，在 `torch.cat([x, skip], dim=1)` **之前**：
```python
# 原： x = torch.cat([x, levels[level]], dim=1)
x = self.skip_gate[level](levels[level], x)     # ← 新增
x = torch.cat([x, ...], dim=1)
```
（注意变量名以实际 `restormer_arch.py` 为准。）

**开关**：`use_skip_gate`　**预期**：+0.05~+0.15 dB　**风险**：中

---

## §4 为什么这次不会重蹈 `band` 的覆辙

`band` 那次（把 MDTA 换成频带注意力）实测：**训练 l_pix 全程高 5~10%、val PSNR 低 0.6 dB**
→ 净的拟合能力损失，不是泛化问题。逐条对照：

| `band` 的坑 | v2 的做法 |
|---|---|
| **替换**掉 MDTA（C 个通道坍缩到 6 个 token） | **并联**，MDTA 一行不改（M4） |
| 6 个 token 的注意力 → 信息瓶颈 | `FSASBranch` 逐元素乘，**全分辨率、不坍缩**（§2.2） |
| 按等半径分 6 环 → 各环能量差几个数量级 → 低频主导 | 按**物理截止频率** `f_LR` 分带，**带边单独成带**（§2.1） |
| 整条路径没有空间局部算子（纯 FFT + 1×1 conv） | `FSASBranch` 内保留 3×3 dwconv（§2.2） |
| 涨不涨只能事后看 | **每项先过 5k A/B 门禁**（§5） |
| 只在权重最高的模块上赌一把 | 7 项分风险分级，低风险项先上（M1–M3） |

---

## §5 效果保障机制（**关键章节**）

没有任何方案能**事前保证**涨点。能保证的只有**机制**：下面这套机制让"不涨的改造绝不会进论文"。

### 5.1 风险分级

| 级别 | 项 | 为什么 |
|---|---|---|
| **低**（几乎必涨） | M1 非线性 V、M2 DirConv、M3 带门控、M5 带调制 | **不改变拓扑、不引入新交互**，只是给已有算子补上缺失的信息或非线性 |
| **中** | M4 并联频域分支、M6 输入提示、M7 skip 门控 | 增加了容量或新输入，有小概率因优化变差而退步（`band` 的教训） |

### 5.2 强制 A/B 门禁（**逐项过，不许合并**）

```
每项改造单独开一个 network_name，跑 5000 iter、seed 10，与「同配置但该项关闭」对比：

  通过判据（两条都要满足）：
    ① Δ val PSNR ≥ +0.05 dB
    ② 训练 l_pix 不劣化（同窗口均值相对差 ≤ +2%）
       ★ ② 是 band 那次的直接教训：它 val 低、训练 loss 也高 → 欠拟合而非过拟合

  不通过 → 回退该项，如实记录，不进主表
```

### 5.3 执行顺序（按通过概率从高到低）

| 顺序 | 项 | 5k 成本 |
|---|---|---|
| 1 | M2 DirConv | ~1.5 h |
| 2 | M3 带门控 | ~1.5 h |
| 3 | M1 非线性 V | ~1.5 h |
| 4 | M5 带调制 | ~1.5 h |
| 5 | **M4 并联频域分支** | ~1.7 h |
| 6 | M6 输入提示 | ~1.7 h |
| 7 | M7 skip 门控 | ~1.7 h |

**先跑 1–4** —— 它们成本最低、最可能通过，能最快确认"带先验 + 方向性"这条思路成立，
再决定要不要上 M4 那口大的。

### 5.4 反"调参救场"规则

某项首次 A/B 不涨时：

1. **只允许查一次实现是否有 bug**（形状、fp32、mask 是否非空、门控方向是否写反、是否误替换了原路径）；
2. 实现无误则**不允许反复调超参直到涨** —— 那是选择性调参；
3. 该项标记为"在本域无收益"，**如实写进论文消融表**。

> **如实报告某项改造无效，本身就是有价值的实验结果**，比反复调到涨点可信得多。

---

## §6 消融矩阵（论文用，同一 v7 协议下）

| 编号 | 配置 | 目的 |
|---|---|---|
| A0 | Restormer（原版，原封不动） | 干净对照 / 下界 |
| A1 | + M1 非线性 V | 单点可加性 |
| A2 | + M2 DirConv | 方向性贡献 |
| A3 | + M3 带门控 | 带先验贡献 |
| A4 | + M5 带调制 | 带先验贡献（另一种注入方式） |
| **A5** | **+ M4 并联频域分支** | **核心创新** |
| A6 | + M6 输入提示 | 退化知识前置 |
| A7 | + M7 skip 门控 | 噪声抑制 |
| A8 | 全开（HoloRestormer v2） | 完整方法 |
| B1 | 原 `band`（替换式频带注意力） | **负面对照** —— 论证"为什么必须并联" |

**B1 这一行价值很高**：把"为什么必须并联而不是替换"变成一个可引用的实验事实，而不是主观论断。

**每项都要跑到 3 种子**（消融变体属协议 §16.5 的核心对比集）。
按 §5.3 门禁先筛，只有通过的才跑满 60000 iter。

---

## §7 与冻结协议 v7 的对接

**已在协议里冻结、本方案不需要再动的**：

- 损失 `L = L1 + 0.1·GFL + 0.2·HF`（协议 §9）；EMA；60k iter + 4 个 milestone；
  D4 自集成；评估协议与指标参数；只用 val 选 checkpoint。

**属于 `network_params`（本方案的自由变量）**：

```json
{
  "scale": 4, "inp_channels": 3, "out_channels": 3,
  "dim": 48, "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4,
  "heads": [1, 2, 4, 8], "ffn_expansion_factor": 2.66, "bias": false,
  "LayerNorm_type": "WithBias",

  "use_nonlinear_v": true,
  "use_dirconv": true, "n_orient": 6, "dirconv_k": 7,
  "use_band_gate": true,
  "use_freq_branch": true,
  "use_band_mod": true,
  "use_hint": true,
  "use_skip_gate": true,
  "band_edge": 0.15
}
```

**两项改造 = 两个 network_name**（协议 §16.5），例如：
`HoloRestormer`（全开）/ `HoloRestormer_noFreq`（关 M4）等。

---

## §8 给 Codex 的实现清单

### 8.1 第一步：读文件（**不要急着改代码**）

```
必读，按顺序：
  1. ~/桌面/10.11/全息超分-实验协议-冻结版-v7.md    ← 52 条规则，冻结，有冲突以它为准
  2. ~/桌面/10.11/全息超分-论文出图规范-v7.md        ← 出图规范 + 数据格式契约
  3. ~/桌面/10.11/experiment_config.txt              ← 配置正文
  4. 本文《全息超分-HoloRestormer-v2-改进方案.md》    ← 要做的事
```

### 8.2 实现顺序（严格按此顺序，每步都要验收）

- [ ] **S0 备份**：把现有 `restormer_arch.py` 复制为 `restormer_arch_v1.py`，
      保证原版 Restormer 仍可注册（论文里它是干净对照，必须能跑）
- [ ] **S1 组件库**：新建 `basicsr/archs/holo_components.py`，实现 §2 的 4 个模块
- [ ] **S2 M2 + M3**：`FeedForward` → `HoloGDFN`（两个开关）
- [ ] **S3 M1 + M5**：`Attention` → 支持 `use_nonlinear_v` / `use_band_mod`
- [ ] **S4 M4**：新建 `HoloTransformerBlock`，保持 `Attention`/`FeedForward` 原路径不变
- [ ] **S5 M6 + M7**：`HoloPatchEmbed` 与 `BandGatedSkip`
- [ ] **S6 组装**：新建 `HoloRestormerV2` 类，注册进 basicsr，`network_type=HoloRestormerV2`

### 8.3 每一步的验收判据（**这些是硬门槛**）

1. **开关全关时，输出与 `restormer_arch_v1.py` 逐比特一致**（`torch.equal`）。
   → 证明原路径没有被污染。**这一条不过，后面都不用做。**
2. 开关全开时：`(1,3,64,64) → (1,3,256,256)`，`torch.isfinite(out).all()`
3. `HoloBandSplit.masks` 返回的三个 mask **之和恒为 1**（`atol=1e-5`）
4. `DirConv` 的 softmax 权重和为 1；梯度能正常回传
5. 所有 FFT 运算在 **fp32** 下进行（用 `torch.cuda.amp.autocast(enabled=False)` 包裹），
   并用 `x.float()` 显式转换
6. 报告**参数量与 FLOPs 的变化**（×4 输出 `3×512×512`），写进 `run_manifest.json`
7. `DualGate` 的门控方向：**C-Map 调 attention、S-Map 调 freq**（写反了会莫名变差）

### 8.4 硬约束（违反即无效）

- [ ] 除 `network_name` / `network_type` / `network_params` 外，不改配置文件任何字段
- [ ] 不用预训练权重；不加梯度裁剪；不给单个模型单独调 lr/betas
- [ ] 每个变体一个独立 `network_name`，实验目录带 `seed{seed}` 层
- [ ] 原版 Restormer 必须保持可跑（它是干净对照）
- [ ] 不把 `meta` 喂给网络（协议 §2.3：v7 不启用）

---

## §9 文献依据

| 用于 | 文献 |
|---|---|
| M4 的频域算子（全分辨率，不坍缩 token） | FFTformer, CVPR 2023 · [arXiv 2211.12250](https://arxiv.org/abs/2211.12250) · [code](https://github.com/kkkls/FFTformer) |
| M1 的 value 通路去线性化 | ICLR 2026 · [OpenReview](https://openreview.net/pdf?id=IBzmQVia88) |
| M2 的方向性（小波分频 + 方向卷积） | DFAN, Symmetry 17:1175, 2025 |
| M4 的先例（SwinIR + 频域分支，+0.8 dB） | SwinFIR · [arXiv 2208.11247](https://ar5iv.labs.arxiv.org/html/2208.11247) |
| ⚠️ 必须区分的最近工作（Restormer + 自适应频域） | AdaIR, ICLR 2025 |
| 频域全局感受野的源头 | Spectral Residual Learning, ACM MM 2019 |
| 给 Restormer 补局部注意力（+0.57 dB） | UDAT, IEEE 2025 · [10900137](https://ieeexplore.ieee.org/document/10900137) |
| M3/M5 的高频加权思想 | Guided Frequency Loss · [arXiv 2309.15563](http://arxiv.org/pdf/2309.15563v2) |
| M2 的方向子带（极坐标傅里叶 + 方向子带 + MoE） | SEMNet, IEEE TIM 2026, 75:4503013 |
| M4 的频带/NA 感知融合对照 | HoloMamba, IEEE TCI 2026, 12:935–945 |
| M7 的散斑特性依据 | Swin-UNet 散斑去噪, Opt. Express 32(19):33465 |
| 门控方向（勿反） | DAT · `dat_arch.py:422-435` |

---

## §10 一页速览

```
本方法的骨架 = Restormer（原版是干净对照）。重设计 7 个模块：

  M1  MDTA 的 value 通路  线性 → 非线性(GELU)              +0.05~0.20  低
  M2  GDFN 的 dwconv      → 方向性可分离卷积 DirConv        +0.05~0.20  低
  M3  GDFN 的门控         × 带能量门控                     +0.05~0.20  低
  M4  TransformerBlock    并联频域分支 + 双向门控 ★核心     +0.15~0.40  中
  M5  MDTA 的输入         × 带锚定通道调制                 +0.05~0.15  低
  M6  PatchEmbed          拼"越界高频提示图"               +0.03~0.15  中
  M7  U-Net skip          带能量门控                       +0.05~0.15  中
                                                    叠加 +0.3 ~ +0.8 dB

物理依据一句话：全息退化在频域有一条由 NA/波长/采样率决定的硬边界 f_LR = 1/(2·scale)；
                原 Restormer 看不见它。所有改造都在把这条边界注入网络。

"确保有效"靠机制不靠保证：
  7 项分风险分级（M1–M3 低风险先上）；每项单独跑 5k A/B 门禁；
  判据 ΔPSNR ≥ +0.05 且训练 l_pix 不劣化；不过就回退并如实记录；禁止调参救场

绝对不要：
  替换原模块（band 的教训：替换 → 训练 loss 高 5~10%、val 低 0.6 dB）/
  坍缩 token / 按等半径分环 / 丢掉 depthwise / 门控方向写反 /
  在 fp16 下算 FFT / 把协议层增益算成自己的优势
```
