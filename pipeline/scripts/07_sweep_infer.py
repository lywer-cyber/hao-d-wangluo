#!/usr/bin/env python3
"""Fig.8 扫描集推理 —— 在扫描 LR 上跑一个模型，落 SR 图与指标

对应 `06_generate_sweep.py` 产出的目录结构：

    <sweep-root>/<param>=<value>/LR/<序号>.bmp
    产出 -> <sweep-root>/<param>=<value>/<model>/results/test/<序号>_SR.bmp
                                                      .../metrics.json

`fig8_sensitivity.py build` 认得这个布局（它列出的真机候选路径之一）。

与 evaluate_holo.py 保持同一口径：
  - 权重**优先 params_ema**（checkpoint 是按 EMA 的 val PSNR 选的）
  - BGR 读入 -> RGB -> /255 -> 网络 -> clamp(0,1) -> BMP 落盘
  - 指标算在**浮点** SR 上（论文口径），BMP 只是可视母版

用法
  python 07_sweep_infer.py \
      --option <models/xxx_options.yml> --checkpoint <net_g_best_N.pth> \
      --sweep-root <sweep> --hr-dir <HR> --model <模型名> \
      --params speckle_contrast diffraction_cutoff poisson_gain read_std aa_sigma
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basicsr.models.archs import define_network  # noqa: E402
from basicsr.metrics.holo_metrics import calculate_holo_metrics  # noqa: E402


def metric_values(sr, hr, evaluation):
    return calculate_holo_metrics(
        sr, hr,
        crop_border=int(evaluation.get("crop_border", 4)),
        psnr_data_range=float(evaluation.get("psnr_data_range", 1.0)),
        ssim_data_range=float(evaluation.get("ssim_data_range", 1.0)),
        ssim_window_size=int(evaluation.get("ssim_window_size", 11)),
        ssim_sigma=float(evaluation.get("ssim_sigma", 1.5)),
        ssim_gaussian_weights=bool(evaluation.get("ssim_gaussian_weights", True)),
        ssim_use_sample_covariance=bool(
            evaluation.get("ssim_use_sample_covariance", False)),
        ssim_channel_axis=int(evaluation.get("ssim_channel_axis", 2)),
        ssim_channel_reduction=evaluation.get("ssim_channel_reduction", "mean"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--option", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--hr-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--params", nargs="+", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write-sr", action="store_true",
                    help="另外落 SR BMP。**默认不写**：1350 张 x 3.9MB x N 个模型会占几十 GB，"
                         "而 Fig.8 只需要指标曲线，用 08_sweep_contract.py 汇总 metrics.json 即可。")
    args = ap.parse_args()

    option = yaml.safe_load(args.option.read_text(encoding="utf-8"))
    evaluation = option.get("evaluation", {})
    scale = int(option.get("scale", 4))

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = define_network(dict(option["network_g"])).to(device)

    state = torch.load(args.checkpoint, map_location="cpu")
    if "params_ema" in state:
        weights, which = state["params_ema"], "params_ema (EMA)"
    elif "params" in state:
        weights, which = state["params"], "params (raw)"
    else:
        weights, which = state, "raw state_dict"
    model.load_state_dict(weights, strict=True)
    model.eval()
    print(f"[sweep] 模型={args.model} 网络={option['network_g'].get('type')} "
          f"权重={which} ckpt={args.checkpoint.name}")

    n_done = 0
    with torch.no_grad():
        for param in args.params:
            for vdir in sorted(args.sweep_root.glob(f"{param}=*")):
                lr_dir = vdir / "LR"
                if not lr_dir.is_dir():
                    continue
                out_dir = vdir / args.model / "results" / "test"
                out_dir.mkdir(parents=True, exist_ok=True)
                accum = {"PSNR": [], "SSIM": [], "RMSE": [], "MAE": []}
                files = sorted(p for p in lr_dir.iterdir()
                               if p.suffix.lower() in (".bmp", ".png", ".tif", ".tiff"))
                for p in files:
                    lr_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                    hr_bgr = cv2.imread(str(args.hr_dir / p.name), cv2.IMREAD_COLOR)
                    if lr_bgr is None or hr_bgr is None:
                        print(f"  跳过（读不到）{p.name}")
                        continue
                    lr = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    hr = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    if hr.shape[:2] != (lr.shape[0] * scale, lr.shape[1] * scale):
                        print(f"  跳过（尺寸不符）{p.name} HR={hr.shape[:2]} LR={lr.shape[:2]}")
                        continue
                    inp = torch.from_numpy(lr.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    sr = model(inp).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                    if sr.shape != hr.shape:
                        print(f"  跳过（SR 尺寸不符）{p.name} SR={sr.shape} HR={hr.shape}")
                        continue
                    m = metric_values(sr, hr, evaluation)
                    for k in accum:
                        accum[k].append(m[k])
                    if args.write_sr:
                        cv2.imwrite(str(out_dir / f"{p.stem}_SR.bmp"),
                                    np.clip(cv2.cvtColor(sr, cv2.COLOR_RGB2BGR) * 255.0,
                                            0, 255).astype(np.uint8))
                    n_done += 1
                summary = {k: (float(np.mean(v)) if v else None) for k, v in accum.items()}
                summary["count"] = len(accum["PSNR"])
                summary["model"] = args.model
                summary["sweep_dir"] = vdir.name
                (out_dir / "metrics.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8")
                ps = summary["PSNR"]
                print(f"  {vdir.name:<28} n={summary['count']:<4} "
                      f"PSNR={ps:.4f}" if ps is not None else f"  {vdir.name}: 无样本")
    print(f"[sweep] {args.model} 完成，共 {n_done} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
