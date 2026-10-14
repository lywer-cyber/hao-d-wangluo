# 2026-10-14 修复：holo_v7_baseline 的 visual_samples.txt 是坏的

## 现象
`evaluate_holo.py --split test` 直接报错退出：
```
RuntimeError: Fixed visual samples are not in the test split:
[373,422,4502,4532,459,465,4668,473,477,479]
```

## 根因（两处，且互相独立）
`holo_v7_baseline/data/splits/visual_samples.txt`（43 B，mtime 2026-10-11 01:11）与权威版本不符：

| | 内容 | 格式 |
|---|---|---|
| 权威（holo_project / swinir_holo_stage 共 3 份一致，83 B，10-11 04:40） | 263/303/308/322/365/376/391/4578/4628/4630 | `NNN.bmp` |
| holo_v7_baseline（坏） | 479/459/373/465/4668/4502/473/477/4532/422 | 裸数字 `NNN` |

1. **内容不同** —— 是另一组 10 张（也都在 test 内，但违反协议 §3「所有模型用同一份清单」）。
2. **格式不同** —— `evaluate_holo.py` 用 `line.strip().split()[0]` 原样读，
   与 split 里的名字（`251.bmp`）比对 `name in visual["save_names"]`。
   裸数字永远不命中 → 即使内容对了也**一张图都导不出来（静默失败）**。
   （`run_holo_trials.py::read_manifest` 有 `allow_plain_numbers=True` 兼容，所以训练期校验没拦住。）

## 改法
备份 `visual_samples.txt.bak_before_fix_20261014`，用权威内容覆盖（写入时 newline="" 保证 LF）。
核验：sha256 = b8a8cad57c3b98300379d33e3b52004945889b0ebdb320bbf319d130eeec1f1e，83 B，与 holo_project 逐字节一致。

## 影响
- 训练不受影响（该 run 的 yml 是 `export_after_all_comparison_runs_complete: true` / `visual_only: true`，
  训练期间不导出）。已训好的 net_g_best_60000.pth 未受影响。
- 修复后评估一次通过。

## 顺带发现（未改，仅记录）
`evaluate_holo.py` 末尾：
```python
summary["visual_images_saved"] = len(names) if a.split == "test" else 0
```
test 时恒等于 500（= 全测试集张数），而实际只导出 10 张可视样本。
该字段是**元数据误报**，不影响 PSNR/SSIM/LPIPS 数值。
如要在论文/清单里引用这个字段，先把它改成 `len(visual["save_names"])`。
