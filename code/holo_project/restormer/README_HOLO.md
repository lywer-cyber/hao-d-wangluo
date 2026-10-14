# 全息 LR×4 Restormer 决策实验

`run_holo_trials.py` 从项目根目录的 `experiment_config.txt` 读取统一协议，生成固定
train/val/test 清单和 BasicSR 选项，然后可顺序训练：

- `mdta`：Restormer 原始 MDTA 注意力，改成 LR×4→HR 重建头；
- `band`：同一骨架、同一训练设置，仅将 MDTA 替换为径向 Fourier 频带注意力。

公平性规则已固定在项目根目录的 `experiment_config.txt`（version 4）：验证集每
5000 iter 评估，按最高 PSNR 选 checkpoint，PSNR 并列时用 SSIM 破平，绝不使用
test 集选模型。指标固定为 RGB、[0,1]、crop=4；SSIM 使用固定版本的
scikit-image，`channel_axis=2`，逐 RGB 通道计算后取 mean；LPIPS 为 AlexNet、
[-1,1]、crop=4、spatial=false，并在最终评估启动时校验 `alex.pth` SHA256。
全息数据的水平翻转已关闭，所有模型均关闭梯度裁剪。val/test 固定 batch=1、
整图、原生尺寸、无 resize/padding，DataLoader 固定不使用 pin memory、
persistent workers 和 prefetch；val 只算 PSNR/SSIM/RMSE/MAE，LPIPS 仅在最终
test 计算。训练启动还会校验 Python/PyTorch/CUDA、AMP 梯度累积和 cuDNN 确定性设置；NVIDIA 驱动版本仅记录，不作为启动阻断条件。

训练的 iteration 是 optimizer update。所有模型默认 AMP float16、effective batch=8。
OOM 时统一把 `--micro-batch-size` 改为 4 或 2，脚本自动设置梯度累积为 2 或 4，
不能只给某个网络降低有效 batch。

在 AutoDL 上：

```bash
cd "$HOLO_PROJECT_ROOT/restormer"
pip install pyyaml opencv-python scikit-image  # torch/torchvision 使用 AutoDL 现有环境
export HOLO_MICRO_BATCH_SIZE=2
nohup ./watch_holo_trials.sh >/tmp/holo_watcher.out 2>&1 &
```

该 watcher 可安全承受 SSH 断开（它从不通过 grep 日志文本推断完成状态）。

训练过程不保存每次验证的 500 张图。两组训练成功结束后，可按固定的 10 张
`visual_samples` 清单，从验证集选出的最佳 checkpoint 导出对应的 LR/SR/HR BMP；
该视觉导出只处理固定样本，不计算测试指标，完整测试报告仍按最终报告流程单独运行：

```bash
./export_holo_paper_images.sh RUNNER_PID
```

图片保存到每个实验目录的 `results/test/`，每张包含 `_LR.bmp`、`_SR.bmp`、
`_HR.bmp`。

The watcher writes `experiments/serial_run.pid`, `serial_run.status`, and
`serial_run.log`. It starts Band only when the MDTA subprocess exits with code
0; any nonzero exit is recorded as `FAILED exit_code=...`.

如需只生成配置、不启动训练：

```bash
python run_holo_trials.py --models both --seeds 10
```

例如统一 OOM 恢复：

```bash
python run_holo_trials.py --run --seeds 10 --micro-batch-size 4
```

输出位于 `experiments/holo_restormer_{mdta,band}_25500_seed10_mb8/`（OOM 时
`mb4` 等后缀会随实际 micro batch 改变）。两组实验的
`options.yml`、checkpoint 和 log 均独立保存。25500 iter 仅用于阶段 3 的快速决策；
正式论文结果仍需将 `--iters` 设为总配置的 250000，并分别跑种子 10、20、30。

训练完成后，在两个 options/checkpoint 上分别执行：

```bash
python evaluate_holo.py \
  --option "$HOLO_PROJECT_ROOT/experiments/holo_restormer_mdta_25500_seed10_mb8/options.yml" \
  --checkpoint "$HOLO_PROJECT_ROOT/experiments/holo_restormer_mdta_25500_seed10_mb8/models/net_g_latest.pth"
```

将 `mdta` 替换为 `band` 即可得到验证集 `metrics.json`（默认不触碰 test 集）；
最终报告时加上 `--split test --require-lpips`，并确保 AutoDL 已安装 LPIPS 权重。

