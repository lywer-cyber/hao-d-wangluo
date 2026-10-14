> 历史快照：包含当时的进度与部分已纠正表述。当前复现与论文口径以仓库根目录README为准。

# 数字全息重建强度图 ×4 超分

数字全息重建所得**强度图**的单图超分：LR（256×320，3 通道 8 bit BMP）→ HR（1024×1280）。

本仓库是论文的**实验复现包**：训练/评估代码、全部实验配置、数据生成管线、出图脚本、
以及已经跑出的结果与图片。

---

## 1 任务定义

| 项 | 值 |
|---|---|
| 输入 | LR 强度图，3 通道 BMP，8 bit，256×320 |
| 输出 | HR 强度图，3 通道，×4，1024×1280 |
| 数据 | 5000 对（HR / LR4），**预生成**；禁止训练时在线退化 |
| 评价 | PSNR / SSIM / LPIPS / RMSE / MAE（RGB、[0,1]、`crop_border=4`） |
| 数据划分 | train 501–4500（4000）／val 1–250+4751–5000（500）／test 251–500+4501–4750（500），三方零重叠 |
| 划分种子 | 42（显式区间，不随机） |
| 唯一自由变量 | **网络结构**（`network_name` / `network_type` / `network_params`） |

**核心物理前提**：退化被 `r̂ = 1`（LR 的 Nyquist，`f_LR = 1/(2·scale) = 0.125` cycles/px）
切成两个性质不同的子问题 —— **带内（r̂<1）= 复原**（信息还在），**带外（r̂>1）= 外推**（信息已丢）。

### 退化链

```
圆孔径衍射低通 → 抗混叠高斯 → 面积平均 ×4 → 乘性散斑 → 泊松散粒 + 读出噪声 → 8bit 量化
```

训练集逐样本抖动退化参数；**val 与 test 使用同一组固定值**，保证所有模型面对完全相同的退化。

---

## 2 仓库结构

```
.
├── docs/                  协议、设计、交接、出图规范（唯一规范来源）
├── pipeline/              数据生成管线 + 候选基线配置
│   ├── scripts/           切分 / 生成 LR / 校验 / 退化算子
│   └── configs/           各候选基线（train + test）
├── code/
│   ├── holo_v7_baseline/  Restormer 基线代码树（BasicSR）
│   └── holo_project/      SwinIR 基线代码树（BasicSR）
├── configs/
│   ├── holo_v7_baseline/  该机跑过的全部 options.yml
│   └── holo_project/      该机跑过的全部 options.yml
├── splits/                两台机器的 train/val/test 清单与 visual_samples
├── figures/scripts/       出图脚本（14 个）
├── logs/                  训练日志（含 v6/v7 早期原始日志）
└── results/               已跑出的结果、图片、指标
```

### 两份代码树的关系

`code/holo_v7_baseline` 与 `code/holo_project` 是**同源但各自演进过的两份 BasicSR 树**，
不是同一份拷贝（`basicsr/models/archs/holo_restormer_arch.py` 两者 sha256 不同）。
差异集中在：

| | holo_v7_baseline | holo_project |
|---|---|---|
| 网络 | `HoloRestormer`（Restormer 基线） | `HoloSwinIR`（SwinIR 基线） |
| `archs/` 独有 | — | `swinir_holo_arch.py` |
| 备注 | 含 `use_staged_head` 开关 | 无 |

两者的 `basicsr/metrics/holo_metrics.py`、`basicsr/utils/holo_protocol.py`、
数据集与损失实现按协议必须一致，改动请同步。

---

## 3 环境

训练与评估实测环境（与 `configs/*/options.yml` 的 `runtime:` 段一致）：

```
Python            3.8
PyTorch           2.0.0+cu118
TorchVision       0.15.1+cu118
CUDA runtime      11.8
NumPy             1.24.4
OpenCV            4.8.1.78
scikit-image      0.19.3
lpips             0.1.4
einops            0.6.1
PyYAML            6.0.1
```

硬件：单卡 RTX 4090 24G。60000 iter / micro-batch 2 × accum 4 约 17.5 小时。

---

## 4 数据

### 4.1 数据集不在本仓库

原始数据集共 **13.2 G**（HR 12 G + LR4 1.2 G），超出仓库容量，**未随仓库分发**。
两种获取方式：

1. 向作者索取原图；
2. 用 `pipeline/scripts/` 从 HR 原图重新生成 LR（见 4.3）。

### 4.2 目录约定

```
<DATA_ROOT>/
├── HR/    5000 张 1024×1280 3 通道 8 bit BMP，命名 <序号>.bmp
└── LR4/   5000 张 256×320  3 通道 8 bit BMP，命名 <序号>.bmp
          + degrade_params.json（每张的退化参数，1.25 MB）
```

代码通过环境变量定位数据：

```bash
export HOLO_PROJECT_ROOT=<项目根>
export HOLO_DATA_ROOT=<DATA_ROOT>          # 里面有 HR/ 与 LR4/
```

### 4.3 重新生成数据

```bash
cd pipeline/scripts
python 01_split_dataset.py     # 按 seed 42 的显式区间切分
python 02_generate_lr.py       # 跑退化链生成 LR4/ 与 degrade_params.json
python 03_verify_dataset.py    # 校验 5000 对齐全、尺寸/位深正确
```

> ⚠️ `holo_degrade.py` 里 **rng 的调用顺序一旦改变，LR 图就会变，全部已跑实验作废**。
> 复现请勿修改该文件的随机数调用顺序。

### 4.4 划分清单

`splits/` 下每个项目各有一份，两者**集合完全相同**，仅换行符不同（CRLF vs LF）：

- `train_meta.txt` / `val_meta.txt` / `test_meta.txt`
- `split_seed42.json`（划分的冻结记录）
- `visual_samples.txt`（10 张定性对比样本，来自测试集，seed 42）

> ⚠️ `visual_samples.txt` 的条目必须**带 `.bmp` 后缀**。评估脚本用
> `name in visual["save_names"]` 与测试集清单（`251.bmp` 形式）比对，
> 写成裸数字会**静默一张图都导不出来**。

---

## 5 复现步骤

### 5.1 训练

```bash
cd code/holo_v7_baseline/restormer          # 或 code/holo_project/restormer
export HOLO_PROJECT_ROOT=<项目根>
export HOLO_DATA_ROOT=<DATA_ROOT>
export PYTHONPATH=$PWD

python -u basicsr/train.py \
  -opt <项目根>/configs/<对应目录>/<run名>_options.yml --launcher none
```

- 验证每 5000 iter，500 张整图；**checkpoint 按 val PSNR 选**（并列用 SSIM 破平）。
- EMA decay 0.999（warmup 2000）；**验证与选点一律用 EMA 权重**。
- 同名重启会自动归档旧目录（`..._archived_<时间戳>`），不会覆盖。

### 5.2 评估（测试集）

```bash
cd code/<项目>/restormer
export HOLO_PROJECT_ROOT=<项目根> HOLO_DATA_ROOT=<DATA_ROOT> PYTHONPATH=$PWD

python evaluate_holo.py \
  --option     <项目根>/configs/<目录>/<run名>_options.yml \
  --checkpoint <项目根>/experiments/<run名>/models/net_g_best_<iter>.pth \
  --split test --require-lpips
```

产物：`experiments/<run名>/results/test/`
- `<序号>_LR.bmp` / `_SR.bmp` / `_HR.bmp`（10 张可视样本，归档母版，不可修改）
- `metrics.json`（聚合指标）

> 脚本会打印 `[eval] 网络类型=... 载入权重=...`，请核对载入的是 **EMA** 权重。

### 5.3 出图

见 `figures/scripts/README.md` 与 `docs/全息超分-出图运行手册-20261013.md`。

---

## 6 已跑出的结果

### 6.1 Restormer 基线（`holo_v7_baseline`）

run：`holo_restormer_mdta_stockhead_60000_seed10_mb2`
网络：`HoloRestormer`（8 个重构开关全关 + 官方 Restormer SR 重建头），**26,165,524** 参数
训练：60000 iter，seed 10，17 h 45 m

| iter | 5000 | 10000 | 20000 | 30000 | 40000 | 50000 | 60000 |
|---|---|---|---|---|---|---|---|
| val PSNR | 30.2174 | 31.3383 | 32.5506 | 33.9286 | 34.4757 | 34.8053 | **34.9497** |
| val SSIM | 0.9006 | 0.9166 | 0.9263 | 0.9320 | 0.9340 | 0.9351 | **0.9355** |

**测试集（500 张，EMA 权重）：**

| PSNR | SSIM | LPIPS | RMSE | MAE |
|---|---|---|---|---|
| **35.2892** | **0.9367** | **0.18609** | 0.01854 | 0.01108 |

产物在 `results/restormer_stockhead_60000/`（含 `metrics.json`、`val_metrics.csv`、
10 张可视样本的 LR/SR/HR BMP）。

### 6.2 SwinIR 基线（`holo_project`）

run：`holo_swinir_60000_seed10_mb2_retry`，`HoloSwinIR`，**3,487,347** 参数。
训练中（撰写时约 55000/60000 iter），结果稍后补入本仓库。

### 6.3 历史 run（作废，结论仍有效）

`logs/v6_v7_raw/` 保留 v6 时代三次与 v7 早期两次的原始日志。

v6 三次 run 的 `total_iter=25500` 与 LR 调度不匹配（milestones 一个都没触发，
lr 全程恒为 `1.000e-04`），模型停在未收敛点，**数字不可用**；但其中一条结论仍成立：
**同等 iter 下 Restormer 明显强于 SwinIR**，本仓库 6.1/6.2 的新数据再次证实。

---

## 7 论文口径：能写与不能写

依据**源码**核对，不依据设计文档。

```python
class HoloRestormerV3(nn.Module):       # 研究版：8 个开关【默认全开】→ 88.7 M 参数
class HoloRestormer(HoloRestormerV3):   # 基线版：8 个开关【强制全关】
```

| 想写的东西 | 能否成立 |
|---|---|
| "基线：官方 Restormer" | ✅ 成立（`..._stockhead_...` run，26,165,524） |
| "基线：Restormer + 两级重建头（变体）" | ✅ 也可，指 `use_staged_head: true` 的 run |
| "本文方法 = 条纹尺度 × 频带、带内复原 / 带外外推" | ❌ **暂不能** —— 这些模块在已训练的网络里全是关的 |
| "HoloBandSplit 按物理截止频率分带" | ❌ 基线版里不参与计算，是纯装饰 |
| "本文做了 D4 自集成" | ❌ 代码里未实现（配置里的 `self_ensemble` 是死字段） |

**方法本体 `HoloRestormerV3`（默认全开）至今没有跑过**，对比表里 "Ours" 那一列是空的。
开跑前需先定两件事：**参数量公平性**（全开 88.7 M vs 基线 26 M）、
**P3 物理展开是否启用**（`HoloForwardDet` / `HoloUnroll` 的数值校验尚未做过）。

---

## 8 已知问题

| 现象 | 说明 |
|---|---|
| 评估（浮点）与出图（8bit BMP）口径差 | `evaluate_holo.py` 的指标算在浮点 SR 上；BMP 是 8bit 量化后另存，PSNR 差约 0.007 dB。**论文数字用 `metrics.json`，出图用 BMP**，出图链路需加 `--no-check` |
| `visual_images_saved` 字段 | `metrics.json` 里该字段在 test 时恒为 500（全测试集张数），实际只导出 10 张。**元数据误报，不影响指标数值** |
| `use_dirconv` 参数臃肿 | 实测 +53.1 M 参数（+203%）。实现是把方向卷积**包在** `dwconv` 外面（`self.dirconv(self.dwconv(x))`），而非设计文档所称的"替换"。做消融前需先改成参数中性版本 |
| 两台机器无 git 历史 | 代码从工作区直接收进本仓库，`run_manifest.json` 的 `code_git_commit` 无法填写 |

---

## 9 引用与联系

（待补：论文标题、作者、联系方式）
