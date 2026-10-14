# 复现包核对说明（2026-10-14）

## 原始结果与导出更新分开保存

1. `results/*/metrics.json`按服务器原件保留，没有改写已有指标。
2. 旧评估器的`visual_images_saved=500`是元数据错误：实际导出10组、30张BMP，不是500组。
3. 两组基线各自500张测试图的聚合指标是浮点预测计算值；量化BMP是展示资产。
4. 旧完整验证日志在`results/*/logs/training.log`与`console.log`。训练结束后可能额外再验证一次；新论文曲线只提取正常训练期的12个验证点，避免同一步重复计入。
5. 历史500张逐图浮点指标没有存盘。不能从只有10张的BMP推导500张统计，也不能把8bit重算值冒充原浮点结果。
6. 发布版评估器新增逐样本JSON/CSV和正确导出数量，保留原数值计算表达式；原服务器脚本另存供核对。

## 路径和环境

- 两份训练代码对应不同服务器快照。`image_restoration_model.py`的EMA开关写法略不同，但记录的配置均开启EMA。
- 五个基线的索引由`configs/baseline_catalog.json`管理。
- 原始YAML保留服务器绝对路径，不直接拿来在新机器运行；使用`prepare_reproduction.py`。
- 默认运行只重定位路径，不修改损失、训练时长或历史warmup差异。
- 首次LPIPS可能需要下载torchvision AlexNet骨干；正式评估仍检查LPIPS校准权重版本与哈希。
- 新准备脚本不安装环境、不复制数据、不自动启动任何任务。

## 不包含的材料

- 完整HR/LR4、扫描集输入图片；
- SR训练权重、优化器状态、其他大二进制压缩包；
- SSH私钥、服务器密码、GitHub Token；
- 未完成模型的测试数字；
- 尚未定型的新MSRes方法代码与运行队列（本次发布仅基线范围）。

## 元数据的不同用途

- `server_A/B_snapshot_manifest.json`：本次从服务器收集的文件原始哈希，描述的是导出时快照。
- `checkpoint_manifest.json`：未上传权重的文件名、字节数、哈希，供自有副本核对。
- `hr/lr4_file_manifest.json`：完整数据文件名与大小，**没有读取并计算所有数据图像的内容哈希**。
- `visual_reference_hashes_A/B.json`：10张共享HR/LR展示图的内容哈希，两机已经比较一致。
- `package_manifest.json`：最终发布文件内容哈希，优先用于检验这次Git包。

服务器快照评估器与发布版评估器之间的已声明差异，会使它们的源代码哈希不同。这是导出功能更新，不是对历史结果的改写。

## 许可证

保留Restormer的MIT许可证，以及BasicSR和SwinIR上游许可证。具体来源和适配范围见仓库根目录`THIRD_PARTY_NOTICES.md`。
