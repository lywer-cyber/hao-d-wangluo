# 重跑说明：Restormer 基线改回官方重建头（2026-10-13）

**原因**：旧 run `holo_restormer_mdta_60000_seed10_mb2` 的重建头不是官方 Restormer：

| | 旧 run（两级头） | 官方 Restormer SR |
|---|---|---|
| output | `Conv2d(96 → 3)` | `Conv2d(96 → 3×4²=48)` |
| 上采样 | `SRUpsample`：Conv(3,12)+PS2 → Conv(3,12)+PS2 | `PixelShuffle(4)` |
| 参数量 | 26,127,292 | **26,165,524**（= v6 那次，完全一致） |

两者差 38,232 参数，全在头上。旧 run 在 30k iter 时 val PSNR 31.19；
同一个身体用官方头时（v6 那次）是 32.64@25500 —— 头是主要原因。

**改了什么**（只改代码 + 新 run 的 `network_params`，未动任何其它配置字段）：
`basicsr/models/archs/holo_restormer_arch.py`
- `HoloRestormerV3.__init__` 新增 `use_staged_head`（默认 `True`，保持旧行为，
  **不改变任何已跑 run 的可复现性**）
- `use_staged_head=False` 时用官方头：`output = Conv2d(96, out_ch*scale²)` + `nn.PixelShuffle(scale)`
- forward 不变（`bicubic(LR) + head(x)` 的全局残差官方本来就有）

备份：`holo_restormer_arch.py.bak_before_stockhead`

**新 run**：
- 名称 `holo_restormer_mdta_stockhead_60000_seed10_mb2`
- options：`experiments/holo_restormer_mdta_stockhead_60000_seed10_mb2_options.yml`
- 与旧 run 的**唯一配置差异**：`network_g.use_staged_head: false`
- 结构核对（启动日志）：`with parameters: 26,165,524`、
  `(output): Conv2d(96, 48, 3,1,1)`、`(reconstruction): PixelShuffle(upscale_factor=4)`
- 前向核对：(1,3,64,64) → (1,3,256,256)，有限值

**旧 run 未删**，仍在
`experiments/holo_restormer_mdta_60000_seed10_mb2/`（跑到 iter ~36000 时手动停止）。
注意：**评估旧 run 时必须显式传 `use_staged_head: true`**，否则头不匹配、`strict=True` 加载会报错。
