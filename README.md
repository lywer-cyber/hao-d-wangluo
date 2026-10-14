# 数字全息重建强度图 ×4 超分：基线复现与论文素材

本仓库提供**实际运行过的基线代码、原始配置、固定划分、训练日志、结果数据和少量可视化样本**。
完整5000对HR/LR数据及模型权重不入库；复现训练请使用你已有的同一份RGB BMP数据。

> **范围与时间：**2026-10-14基线快照。Restormer-SR、SwinIR已完成训练与测试；RCAN正在训练，RRDBNet、EDSR尚未启动。状态会随实验推进改变，详见`metadata/baseline_status_20261014.json`。未完成实验不填造结果，未定型主方法不列入主表。

## 1. 已完成结果

所有下列数字来自保存的`metrics.json`，在浮点SR预测上计算；均为**单个训练种子10**、500张测试图的均值，不是三种子统计。

| 基线（实际配置） | 参数量 | PSNR↑ | SSIM↑ | LPIPS↓ | RMSE↓ | MAE↓ |
|---|---:|---:|---:|---:|---:|---:|
| Restormer-SR：主干＋×4重建头 | 26,165,524 | 35.2892 | 0.9367 | 0.18609 | 0.01854 | 0.01108 |
| SwinIR：embed_dim=96，4组，每组6层 | 3,487,347 | 31.9922 | 0.9201 | 0.20823 | 0.02590 | 0.01399 |

待完成基线：RCAN（64通道、10组×20块）、RRDBNet（23块）、EDSR（64通道、16块）。
RRDBNet在这里使用统一重建损失，**不是完整ESRGAN对抗训练**。
Restormer-SR是本项目的超分适配版，不称为官方原生×4架构。

## 2. 仓库内容

```text
code/                       两台机器对应的训练/评估代码，保留上游来源
configs/baseline_catalog.json  五个基线的精确配置、代码树、状态、结果路径索引
configs/holo_*/             服务器原始options.yml；保留原绝对路径，作为审计快照
splits/holo_*/              train/val/test清单、冻结划分和10张可视化清单
results/restormer_stockhead_60000/  指标、最佳点记录、完整日志、30张LR/SR/HR BMP
results/swinir_60000/       指标、最佳点记录、完整日志、10张SR BMP
results/sensitivity/        Restormer的27个单参数扫描点和扫描清单
metadata/                  逐图退化参数、文件清单、哈希和服务器导出快照
paper/                     可直接用于整理稿件的CSV、PNG、PDF及来源哈希
scripts/                   可移植配置生成、论文素材重建、发布包校验
reproducibility/            历史评估脚本原件及评估行为回归测试
environment/               运行环境快照与安装依赖
figures/scripts/           原完整论文出图工具；部分输入需后续正式评估补齐
docs/                      复现说明、限制、历史实验协议与设计文档
LICENSES/                  上游许可证
```

SwinIR使用与Restormer**SHA256完全相同**的10组HR/LR参考图，因此只另存SR，避免重复；映射在`results/swinir_60000/visual_sources.json`。
这里仅含论文展示的10张参考样本，**不是完整数据集，也不足以代替训练数据**。

## 3. 环境准备

推荐Linux、Python **3.8**、CUDA 11.8兼容GPU环境。不要用Python 3.12启动正式训练，协议会检查版本。

```bash
conda create -n holo-repro python=3.8 -y
conda activate holo-repro

# REPO是你克隆后的绝对目录
export REPO=/absolute/path/to/hao-d-wangluo
python -m pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$REPO/environment/requirements-reproduce.txt"
```

本仓库通过`PYTHONPATH`使用随附的BasicSR树；不要另外安装其他版本BasicSR来替换它。
本地同名`timm`安装可能改变SwinIR使用的辅助函数；记录的服务器环境没有timm，代码使用自带兼容实现。
完整包版本记录见`environment/server_A_packages.txt`和`server_B_packages.txt`。

首次LPIPS计算还需要torchvision的AlexNet预训练骨干缓存，可能触发下载；离线机器请提前准备PyTorch模型缓存。
LPIPS线性校准权重由`lpips==0.1.4`包提供，并由现有代码核对SHA256；它们都不是SR模型的预训练权重。

## 4. 使用已有数据生成可移植配置

数据目录应为：

```text
/absolute/path/to/data/
  HR/1.bmp ... 5000.bmp
  LR4/1.bmp ... 5000.bmp
```

生成配置不会复制数据，也不会启动训练：

```bash
python "$REPO/scripts/prepare_reproduction.py" \
  --model restormer \
  --data-root /absolute/path/to/data \
  --work-root /absolute/path/to/reproduction_runs
```

`--model`可选`restormer`、`swinir`、`rcan`、`rrdbnet`、`edsr`。
脚本会复制划分清单、重定位路径，生成`launch_train.sh`、`evaluate.sh`和`reproduction_manifest.json`。
默认保留原始seed和所有训练超参；`--seed 20`/`--seed 30`只在需要重复实验时显式使用。
已有输出配置内容不同则拒绝覆盖，避免误混实验。

**不要直接把原始配置中的`/root/autodl-tmp/...`当作自己机器的目录。** 环境变量本身不会自动改写原始YAML里的全部路径，使用上述脚本重定位。

## 5. 训练与评估

```bash
# 例：Restormer
bash /absolute/path/to/reproduction_runs/restormer/launch_train.sh

# 训练后查看best_checkpoint.json，使用验证集选择出的最佳点
bash /absolute/path/to/reproduction_runs/restormer/evaluate.sh \
  /absolute/path/to/reproduction_runs/restormer/experiments/RUN_NAME/models/net_g_best_ITER.pth
```

- 60,000次优化器更新；micro-batch 2 × accumulation 4，有效batch 8。
- 相同L1＋0.1 GFL＋0.2 HF损失、EMA与评价口径。
- 每5000步验证，按验证集PSNR选最佳checkpoint；不按测试结果选点。
- 使用RGB、[0,1]、crop_border=4，保持原生尺寸。
- **历史差异保留：SwinIR的warmup_init_ratio=0.01，其他基线为0.0。** 为复现已有数字，不将它静默改成统一值。
- OOM不会由训练器自动改变batch；归档中有部分模型的mb1配置。如需要换配置，应明确记录，不能覆盖原run冒充同一次训练。

完整数据和权重不在Git中。权重文件名、大小及哈希见各结果目录的`checkpoint_manifest.json`；可以使用自己已有的匹配权重，或从头训练。

### 评估导出改进

本次发布的评估器在不改变预测、指标数值表达式的前提下，额外保存：

- `metrics.json`：聚合指标；
- `per_sample_metrics.json` / `.csv`：**量化前浮点预测**的逐图指标；
- 固定10组LR/SR/HR BMP；
- 实际保存的样本/文件数量。

服务器历史评估器原件在`reproducibility/evaluate_holo_server_snapshot.py`。
历史两组已完成结果只留有聚合指标，**本次没有重跑GPU评估，也没有伪造其500张逐图指标**。
新导出功能仅在本仓库代码中，未覆盖正在训练的服务器代码。

## 6. 论文用图和表格

已生成：

- `paper/tables/baseline_test_results.csv`：两组完成基线的原始精度指标；
- `paper/tables/*_validation_curve.csv`：每组12个正常训练期验证点；
- `paper/tables/restormer_sensitivity.csv`：27个扫描点；
- `paper/figures/baseline_validation_curves.png/.pdf`；
- `paper/figures/baseline_test_summary.png/.pdf`；
- `paper/figures/restormer_sensitivity.png/.pdf`；
- `paper/figures/fixed_samples_overview.png`：固定10张全图预览。

这些是**基线阶段素材**，不是包含最终主方法的完整论文主图。

```bash
python -m pip install -r "$REPO/environment/requirements-paper.txt"
python "$REPO/scripts/build_paper_assets.py"
```

重建现有图表不需要GPU、完整数据或模型权重。
数值取自保存的浮点指标，**不从8bit BMP重新计算后替换原表格**。图片预览经过显示缩放，原始BMP仍保留。
27点扫描每点仅50张固定图，不能与500张主测试均值混为一谈。
图表源文件及哈希见`paper/provenance.json`。

## 7. 复现边界

- 本仓库支持在你已有的同一份HR/LR4数据上复现训练与评估；没有数据和权重时只能复核公开结果与重建已有图表。
- `pipeline/scripts/02_generate_lr.py`等早期管线含灰度/PNG路径，**不能无验证地用来替换已训练实验的RGB BMP输入**。优先使用你保留的原数据；逐图退化元数据已归档，但它不等于完整原始数据。
- `docs/`内有历史方案与进度快照；当前执行入口以本README和`baseline_catalog.json`为准。不能把旧方案里的预测增益当作已完成结果。
- 原HoloRestormerV3研究模块仍随主干代码保留，但不代表这些模块已用于本次基线。已发现旧MSDW在关闭频带支路时不被调用，不能用其短跑差异宣称模块收益。
- 当前两组结果只有seed10，不提供伪造的均值±种子标准差或显著性结论。

更多说明见`docs/REPRODUCTION_NOTES.md`、`paper/README.md`和`THIRD_PARTY_NOTICES.md`。

## 8. 校验发布包

```bash
python "$REPO/scripts/verify_package.py"
python -m unittest discover -s "$REPO/tests/reproduction" -v
# 安装复现环境后可选运行；仅CPU，不启动训练：
python "$REPO/scripts/smoke_baselines.py"
python "$REPO/reproducibility/test_evaluator_exports.py"
```

评估器回归测试使用合成小图、简单插值网络和确定性LPIPS测试替身，验证导出行为不改原指标/图片；它不是新性能结果，也不替代真实LPIPS模型核验。
