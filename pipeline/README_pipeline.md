# 数字全息超分辨率 · 统一实验协议

强度图 LR×4 → HR 的完整实验管线。**所有模型走同一套数据、同一套退化、同一套超参**，
唯一变量是网络结构 —— 这是论文可比性的前提。

```
holo-sr/
├── scripts/
│   ├── holo_degrade.py        ★ 退化算子（两套协议，核心）
│   ├── holo_common.py           读写/归一化
│   ├── smoke_test.py          ★ 管线自检（不需要 torch，先跑这个）
│   ├── 01_split_dataset.py      5000 张 → 4000/500/500（按组划分）
│   ├── 02_generate_lr.py        批量生成 LR（train 随机抖动 / val·test 固定）
│   ├── 03_verify_dataset.py     数据集自检 + 论文用统计表
│   ├── gen_configs.py         ★ 生成所有模型的 yml（只换网络结构）
│   ├── 04_train.sh            ★ 统一训练入口
│   └── 05_evaluate.py           指标 + 论文对比表
├── configs/{train,test}/        自动生成，勿手改
└── data/holo/                   自动生成
    ├── HR/{train,val,test}/
    ├── LR_bicubic/X4/{train,val,test}/
    ├── LR_physical/X4/{train,val,test}/
    └── degrade_config.json      退化参数（论文里直接引用）
```

---

## 快速开始

**所有实验都读同一个「实验总配置」**（默认 `~/桌面/全息超分-实验总配置.yaml`，
放到项目根或 `$HOLO_EXP_CONFIG` 也行）。路径全用相对路径 + 环境变量覆盖，
**不绑定机器，拷到别的服务器只要设两个环境变量就能跑**。

```bash
cd holo-sr

# 0) 先自检
python3 scripts/smoke_test.py

# 0.5) 校验总配置（会检查 hr_root 是否存在、划分是否自洽、有无写死绝对路径）
python3 scripts/exp_config.py --check

# 1~4) 全部零参数，所有参数从总配置读
export HOLO_HR_ROOT=/path/to/你的5000张HR   # 数据在哪（覆盖配置里的 ${HOLO_HR_ROOT}）
export BASICSR_ROOT=/path/to/DAT           # BasicSR 骨架在哪

python3 scripts/01_split_dataset.py        # 划分 5000 → 4000/500/500
python3 scripts/02_generate_lr.py          # 生成两套退化协议的 LR
python3 scripts/03_verify_dataset.py       # 校验 + 统计
python3 scripts/gen_configs.py             # 生成所有模型的 yml

# 5) 训练（所有模型同一条命令）
./scripts/04_train.sh --list               # 看有哪些配置
./scripts/04_train.sh DAT physical 4
./scripts/04_train.sh --matrix 4           # 一次跑完整对比矩阵

# 6) 出论文表
python3 scripts/05_evaluate.py metrics --results_root $BASICSR_ROOT/results --out_md results_table.md
python3 scripts/05_evaluate.py params
```

**注意**：脚本 01–03 和环境无关（只用 numpy/cv2），04–05 需要 torch + BasicSR。
在正式跑 5000 张前，务必先 `smoke_test.py` 全绿。

---

## 实验总配置（唯一事实来源）

```yaml
dataset:    5000 张，train 4000 / val 500 / test 500，split_seed 42，按组划分
degradation: bicubic + physical 两套；train 抖动、val/test 固定
training:   patch 256, batch 8, 250k iter, Adam lr 1e-4, L1, 种子 [10,20,30]
evaluation: psnr/ssim/lpips, crop_border=4, 只用 test 报数
models:     comparison(8) + ablation(1) + ours(1)
fairness_rules: 9 条公平性自查项
```

**换服务器三步**：

```bash
git clone <你的代码> /opt/holo-sr && cd /opt/holo-sr
cp 全息超分-实验总配置.yaml .          # 或 export HOLO_EXP_CONFIG=<路径>
export HOLO_HR_ROOT=/data/hr BASICSR_ROOT=/opt/BasicSR
python3 scripts/exp_config.py --check  # 通过即可开跑
```

`project_root` 默认取 `scripts/` 的上一级，所以只要保持
`<项目根>/scripts/exp_config.py` 这个结构，任何机器上路径都能自动解析对。
生成的 yml 里 dataroot 是**本机绝对路径**（BasicSR 按 CWD 解析），
所以换机器后要重新跑一次 `gen_configs.py`。

---

## 一、数据集划分

| 划分 | 张数 | 用途 |
|---|---|---|
| train | 4000 | 训练 |
| val | 500 | 选超参 / 早停（**不参与最终报数**） |
| test | 500 | **只跑一次**，最终指标 |

**关键：默认按「组」划分，不是按张。**
如果 5000 张里存在同源图（同一病人/切片/场景/采集批次的多个视角），
按张随机划分会让同源图同时进训练和测试，指标虚高——这是审稿人必问的点。

- 文件名有分组信息：`--group-regex '^(S\d+)'`
- 无分组信息：不加该参数（每张独立），但论文里要写清楚

脚本会自动做泄漏自检（`01_split_dataset.py` 末尾断言 + `03_verify_dataset.py` 复核）。

---

## 二、退化协议（两套，都必须做）

### D1 · bicubic —— 经典 SR 协议
LR = bicubic 降采样（含抗混叠）。**用途：与文献（RCAN/SwinIR 基准）可比。**

### D2 · physical —— 物理退化（域贡献）
按光学链路顺序施加，参数见 `holo_degrade.py:DEFAULT_PHYS`：

| # | 环节 | 参数 | 默认值 |
|---|---|---|---|
| 1 | 衍射受限低通（孔径截止） | `diffraction_kernel` / `diffraction_cutoff` | circle / 1.0 |
| 2 | 抗混叠低通 | `aa_sigma` | 0.50 |
| 3 | 降采样（面积平均） | — | — |
| 4 | 乘性散斑 | `speckle_contrast` / `speckle_corr` | 0.60 / 4.0 |
| 5 | 泊松散粒 + 高斯读出噪声 | `poisson_gain` / `read_std` | 2000 / 0.005 |
| 6 | 量化 | `bit_depth` | 8 |

- **train 集**：参数逐样本随机抖动（`RANDOM_RANGES`），提升泛化
- **val/test 集**：参数固定、种子固定 —— **保证所有模型面对完全相同的退化**

⚠️ 论文中需如实说明：这里的散斑是在**重建强度图上**施加的退化模型，
不等价于全息图层面散斑的完整物理仿真。

**为什么必须做 D2**：CV-HoloSR 明确指出 naive bicubic "完全无法在目标距离上聚焦，
因为深度畸变"；而 HoloADMM 的对比显示，**纯仿真训练的模型在真实数据上"输出无意义的重建"**。
只用 D1 会被质疑"不真实"，只用 D2 又无法与文献对齐。

---

## 三、统一训练超参（所有模型完全一致）

| 项 | 值 |
|---|---|
| GT patch | 256 |
| batch size / GPU | 8 |
| 优化器 | Adam (β=0.9, 0.99), weight_decay=0 |
| 初始 lr | 1e-4 |
| scheduler | MultiStepLR, milestones [125k, 200k, 225k, 237.5k], γ=0.5 |
| 总迭代 | 250,000 |
| 损失 | L1 |
| 验证频率 | 每 5k iter |
| 数据增强 | hflip + rot90 |
| 随机种子 | manual_seed=10 |

**公平性要求**：同一模型至少跑 **3 个种子**，报均值 ± 标准差（这个域数据少，方差大）。

---

## 四、基线与来源

| 层 | 模型 | arch 文件来源 | 状态 |
|---|---|---|---|
| 下界 | Bicubic | — | 无需训练 |
| 经典 CNN | SRResNet | BasicSR 原生 | ✅ 可直接跑 |
| 经典 CNN | **RCAN** | github.com/yulunzhang/RCAN → `models/rcan.py` | 需接入 |
| 轻量高效 | **NAFNet** | BasicSR 原生 | ✅ 可直接跑 |
| 注意力 | **SwinIR** | github.com/JingyunLiang/SwinIR → `models/network_swinir.py` | 需接入 |
| 注意力 | **HAT** | github.com/XPixelGroup/HAT → `basicsr/archs/hat_arch.py` | 需接入 |
| 状态空间 | **MambaIR** | github.com/csguoh/MambaIR | 需接入 + `pip install mamba-ssm causal-conv1d` |
| 对照 | **DAT** | 本仓库自带 | ✅ 可直接跑 |

### 为什么必须有 MambaIR
HoloMamba（IEEE TCI 12:935, 2026）明确指出 CNN 与 Transformer
"inadequately capture long-range diffraction behavior and essential physical constraints"，
并用双向 SSM + 谱/数值孔径感知融合打赢 Transformer。
**如果 Mamba 也输给你的方法，你的论点就非常硬。**

### 为什么 DAT 放在对照而不是主基线
DAT 在全息领域是**真空**（多轮检索无任何应用），没有前人用它，
审稿人不会要求对比。把它放进**消融表**当"直接搬通用 SOTA 会怎样"的对照更有说服力。

### 接入新模型（3 步）
1. 把 arch 文件放到 `$BASICSR_ROOT/basicsr/archs/<name>_arch.py`
2. 在 `$BASICSR_ROOT/basicsr/archs/__init__.py` 里 `from .<name>_arch import *`
3. 在 `gen_configs.py` 的 `MODELS` 里加一条，重新生成配置

---

## 五、论文里要报的表

1. **主对比表**：×4，两套退化协议，PSNR / SSIM / LPIPS
2. **参数量 + FLOPs 表**：主张效率必须有
3. **多种子表**：3 seed 均值 ± 标准差
4. **距离鲁棒性曲线**：对标 HoloPASWIN 的失败点（z ±0.5mm / ±1mm）
   *——只有做复值/相位输出才需要，纯强度图可省*
5. **消融表**：见桌面《数字全息超分辨率-方案与论文基线.md》第 9 节

---

## 六、已知限制

- 退化是**离线生成**的，不是 BasicSR 训练时的在线退化。好处：所有模型面对字节级相同的 LR。
- 散斑模型是在强度图上施加的近似，非全息图层面的物理仿真。
- `05_evaluate.py` 的 LPIPS 需要 `pip install lpips`；未安装时该列显示 `—`。
- FLOPs 目前靠外部工具（如 `thop` / `fvcore`），脚本只算了参数量。
- 本机**未安装 torch**，因此 04/05 的 torch 部分**未经实机验证**；
  01–03 和 `smoke_test.py` 已实机验证全绿。
