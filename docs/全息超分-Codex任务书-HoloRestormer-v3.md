# Codex 任务书 —— HoloRestormer v3 模块实现

> 项目：数字全息强度图 ×4 超分
> 任务：把 Restormer 的模块逐个按全息物理重设计，形成论文的方法部分
> 配套文档：`全息超分-HoloRestormer-v3-模块重设计方案.md`（含代码骨架）、
> `全息超分-实验协议-冻结版-v7.md`（冻结规范）、`全息超分-论文出图规范-v7.md`、`experiment_config.txt`
> 日期：2026-10-12

---

## 第一步：先把这五份文件完整读一遍，**不要急着改代码**

| # | 文件 | 说明 |
|---|---|---|
| 1 | `~/桌面/10.11/全息超分-HoloRestormer-v3-模块重设计方案.md` | **你要做的事**（最重要，含代码骨架） |
| 2 | `~/桌面/10.11/全息超分-实验协议-冻结版-v7.md` | 52 条规则，**冻结**；有冲突以它为准 |
| 3 | `~/桌面/10.11/全息超分-论文出图规范-v7.md` | 出图规范；`per_sample_metrics.json` 的 schema |
| 4 | `~/桌面/10.11/experiment_config.txt` | 配置正文 |
| 5 | `~/holo-sr/scripts/holo_degrade.py` | 退化算子；写 `HoloForwardDet` 时照着它 |

---

## 第二步：按顺序实现，**不要一口气全写完**

| 步骤 | 做什么 | 关键要求 |
|---|---|---|
| **S1** | 把现有 `restormer_arch.py` 备份成 `restormer_arch_v1.py`，确认原版 Restormer **仍可注册可跑** | ★ 原版 Restormer 在论文里是「干净对照」，必须一直能跑 |
| **S1** | 新建 `basicsr/archs/holo_components.py`，实现：`HoloBandSplit` / `FSASBranch` / `DirConv` / `DualGate` / `MultiScaleDW` / `spectral_mask` | 代码骨架见设计方案 §2、§3 |
| **S2** | `Attention` 加 `use_nonlinear_v`；`FeedForward` → `HoloGDFN`（含 `DirConv`） | **然后停下**，先跑这三项的 5k iter A/B（见第四步） |
| **S3** | `HoloBlock`（带内 restoration 支路 + 带外 extrapolation 支路，各自用 mask 限制在自己频段） | 见设计方案 §3 P2 |
| **S4** | `HoloStem`（可学强度的 Box-Cox 双域前端，替换第一级 `OverlapPatchEmbed`） | 见设计方案 §3 P1 |
| **S5** | `HoloForwardDet`：`holo_degrade.degrade_physical` 的**【确定性部分】**可微 torch 版（只含 圆孔径低通 → 抗混叠高斯 → 面积平均降采样；**不含**散斑 / 泊松 / 读出噪声 / 量化），同时实现 `adjoint`（转置算子） | 见设计方案 §3 P3；过第三步门槛 3 |
| **S6** | `HoloUnroll`：HQS 展开 K 个阶段：`x = x − η_k·Dᵀ(D(x)−LR)`；`x = Prox_k(x)` | **先跑通 K=1，再加到 K=3** |
| **S7** | `BandGatedSkip` + 输出头 | 见设计方案 §3 P4 |
| **S8** | 组装 `HoloRestormerV3`，注册进 basicsr | — |

---

## 第三步：三条硬门槛（任何一条不过就**停下来告诉我**，不要绕过）

### 门槛 1 —— 原路径未被污染 ★ 最重要

**所有开关关闭时，`HoloRestormerV3` 的输出必须与 `restormer_arch_v1.py` 逐比特一致**（`torch.equal`）。

> **这条不过，后面全部不用做。**

### 门槛 2 —— 频域运算必须在 fp32

所有 FFT 与降采样必须在 fp32 下算：
```python
with torch.cuda.amp.autocast(enabled=False):
    ...torch.fft.fft2(x.float(), norm='ortho')...
```
fp16 下 FFT 精度不够，会让训练不稳。

### 门槛 3 —— 数值校验

| 项 | 判据 |
|---|---|
| (a) `HoloForwardDet` vs numpy 版 `holo_degrade` 的确定性路径，同参数逐像素对比 | **MSE < 1e-4** |
| (b) `HoloForwardDet.adjoint` 的伴随性：`<D(x), y> ≈ <x, Dᵀ(y)>` | `atol = 1e-3` |
| (c) `HoloBandSplit.masks` 的三个 mask 之和 | 恒为 1，`atol = 1e-5` |

---

## 第四步：每项改动单独做 5k iter A/B 门禁

每项**单独**开一个 `network_name`，与「同配置但该项关闭」对比。

**通过判据（两条都要满足）：**

| # | 判据 |
|---|---|
| ① | **Δ val PSNR ≥ +0.05 dB** |
| ② | **训练 `l_pix` 不劣化**（同窗口均值相对差 ≤ +2%） |

> **② 是之前一次失败的直接教训。** 那次改造 val 低、训练 loss 也高 ——
> 说明是**欠拟合**（净的拟合能力损失），不是过拟合。只看 val 会误判。

**不通过 → 回退该项，如实记录。**

**不要试图调超参救场** —— 那是选择性调参。首次不涨时只允许查一次实现 bug
（形状、fp32、mask 是否非空、门控方向、是否误替换了原路径）；实现无误就标记为
"本域无收益"并如实写进消融表。

---

## 红线（违反即无效）

1. 除 `network_name` / `network_type` / `network_params` 三个字段外，**不改配置文件任何其它字段**。
2. **绝对不能改 `holo_degrade.py` 里任何 `rng` 的调用顺序或次数**（协议 §2.2）。
   你只能**新增**一个独立的 torch 模块，**不许动那个 numpy 文件**。
3. 不用预训练权重；不加梯度裁剪；不给单个模型单独调 lr / betas。
4. 每个变体一个独立 `network_name`，实验目录必须带 `seed{seed}` 层。
5. **`DualGate` 的门控方向**：C-Map（全局池化）调 attention、S-Map（1×1 conv）调 freq。
   **写反了会莫名变差且极难排查。**
6. 数据保真项里**不能**放散斑 / 泊松 / 量化这些随机项 ——
   每次前向都会变，会引入噪声梯度、让训练不稳。

---

## 第五步：汇报

1. 每项改动**落在哪个文件哪一行**。
2. **参数量与 FLOPs**（×4 输出 `3×512×512`）相对原版的变化。
3. **`HoloStem` 训练后学到的 `p` 值**是多少。
   （若收敛到 ~0.05，说明网络自己选择了 log 域 —— 这是可写进论文的证据。）
4. **5k A/B 结果表**：

   | 改造项 | ΔPSNR | Δ训练 l_pix | 通过? |
   |---|---|---|---|
   | | | | |

5. 哪些条目**你实现不了或认为有问题** —— 直接说，**不要默默跳过**。

---

## 附：为什么这些设计是"物理驱动"的（供你理解，不必写进代码）

本任务的退化被 `r̂ = 1`（LR 的 Nyquist）切成两个性质不同的子问题：
**带内 = 复原**（信息还在，去散斑 + 去模糊 + 去噪）、**带外 = 外推**（信息已丢，靠先验生成）。

Restormer 的归纳偏置来自自然图的"加性高斯噪声复原"任务，对本任务有三个错配：

| | 物理事实（`holo_degrade.py` 可查） | Restormer 的假设 | 对应模块 |
|---|---|---|---|
| **F1** | 噪声是**乘性散斑**（`img * speckle`），相关、非高斯、非平稳 | 加性 | **P1** `HoloStem` |
| **F2** | 退化算子 `D` **完全已知且可微** | 黑箱，`D` 只放在损失里 | **P3** `HoloUnroll` |
| **F3** | 信息缺失有**硬边界 `r̂ = 1`** | 全频带一视同仁 | **P2** `HoloBlock` |

**P3 是最强的创新点**，理由：它加的是**物理**不是**容量** ——
数据保真项 `Dᵀ(D(x)−LR)` **零可学参数**，不增加过拟合风险。
对照 USRNet (CVPR 2020) 的 Data module，但我们的 `D` 是**全息成像退化算子**（圆孔径 + 抗混叠 + 面积平均），
不是 bicubic / 高斯核 —— 这是必须守住的差异点。
