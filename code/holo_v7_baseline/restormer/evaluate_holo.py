"""Evaluate a trained HoloRestormer checkpoint on the fixed test split.

The evaluator writes uncompressed BMP LR/SR/HR triplets and mean metrics.
LPIPS is mandatory for the final test report.  The evaluator validates the
package version and the pinned AlexNet weight hash before inference.
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from basicsr.models.archs import define_network
from basicsr.metrics.holo_metrics import calculate_holo_metrics
from basicsr.utils.holo_protocol import (configure_determinism,
                                         validate_determinism,
                                         validate_experiment_options,
                                         validate_runtime_versions,
                                         locate_lpips_weights)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--option", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--require-lpips", action="store_true")
    p.add_argument("--visual-only", action="store_true",
                   help="Export the fixed visual_samples without calculating test metrics")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


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


def visual_output_options(root: Path, option: dict) -> dict:
    """Load the fixed paper-visualization list from options or shared config."""
    visual = option.get("visual_output")
    if visual is None:
        shared_config = root / "experiment_config.txt"
        if not shared_config.is_file():
            raise FileNotFoundError(f"Missing shared experiment config: {shared_config}")
        parser = configparser.ConfigParser(interpolation=None, delimiters=("="))
        parser.optionxform = str
        with shared_config.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
        visual = dict(parser["visual_output"])

    manifest = Path(visual["save_list"])
    if not manifest.is_absolute():
        manifest = root / manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing fixed visual sample list: {manifest}")
    names = {line.strip().split()[0] for line in manifest.read_text(encoding="utf-8").splitlines()
             if line.strip()}
    expected = int(visual.get("save_count", len(names)))
    if len(names) != expected:
        raise RuntimeError(
            f"Visual sample count mismatch: configured {expected}, manifest has {len(names)}")
    image_format = str(visual.get("format", "BMP")).upper()
    compression = str(visual.get("compression", "none")).lower()
    if image_format != "BMP" or compression != "none":
        raise ValueError("Holographic visual outputs must be uncompressed BMP")
    visual["save_names"] = names
    return visual


def main():
    a = args()
    if a.visual_only and a.split != "test":
        raise ValueError("--visual-only is reserved for the fixed test visual samples")
    if a.visual_only and a.require_lpips:
        raise ValueError("--visual-only does not compute LPIPS or test metrics")
    if a.split == "val" and a.require_lpips:
        raise ValueError("Validation is fixed to PSNR/SSIM/RMSE/MAE; LPIPS is test-only")
    option = yaml.safe_load(a.option.read_text(encoding="utf-8"))
    deterministic = option.get("determinism", {
        "enabled": True, "cudnn_deterministic": True,
        "cudnn_benchmark": False, "use_deterministic_algorithms": True})
    configure_determinism(int(option.get("manual_seed", 0)), deterministic)
    validate_determinism(deterministic)
    validate_experiment_options(option)
    runtime = option.get("runtime", {})
    validate_runtime_versions(
        runtime, require_lpips=(a.split == "test" and not a.visual_only))
    root = Path(option["path"]["root"]).resolve()
    visual = visual_output_options(root, option)
    split = root / "data" / "splits" / f"{a.split}_meta.txt"
    if not split.exists():
        raise FileNotFoundError(f"Missing {a.split} split manifest: {split}")
    # The generated option stores the two folders separately; use those exact
    # paths for either the validation decision or the final test report.
    hr_dir = Path(option["datasets"]["val"]["dataroot_gt"])
    lr_dir = Path(option["datasets"]["val"]["dataroot_lq"])
    names = [x.strip().split()[0] for x in split.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.split == "test":
        missing_visuals = visual["save_names"] - set(names)
        if missing_visuals:
            raise RuntimeError(
                f"Fixed visual samples are not in the test split: {sorted(missing_visuals)}")
        if a.visual_only:
            names = [name for name in names if name in visual["save_names"]]
    evaluation = option.get("evaluation", {})
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")

    # 按 network_g.type 动态建网络（原来是写死 HoloRestormer，评不了 HoloSwinIR）
    net_opt = dict(option["network_g"])
    model = define_network(net_opt).to(device)
    state = torch.load(a.checkpoint, map_location="cpu")
    # ★ 必须优先用 EMA：checkpoint 是按 EMA 权重的 val PSNR 选出来的
    #   （协议 §7.4 / 规则 34：验证与选点一律用 EMA）。原来只读 "params"（raw），
    #   会出现"选点用 EMA、报数用 raw"的错配。
    if "params_ema" in state:
        weights, which = state["params_ema"], "params_ema (EMA)"
    elif "params" in state:
        weights, which = state["params"], "params (raw)"
    else:
        weights, which = state, "raw state_dict"
    model.load_state_dict(weights, strict=True)
    model.eval()
    print(f"[eval] 网络类型={option['network_g'].get('type')}  载入权重={which}")

    lpips_fn = None
    if a.split == "test" and not a.visual_only:
        try:
            import lpips  # type: ignore
            lpips_fn = lpips.LPIPS(
                net=evaluation.get("lpips_network", "alex"),
                spatial=bool(evaluation.get("lpips_spatial", False)),
                model_path=str(locate_lpips_weights()),
            ).to(device).eval()
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("LPIPS is required for the final test report") from exc

    out_root = root / "experiments" / option["name"] / "results" / a.split
    out_root.mkdir(parents=True, exist_ok=True)
    accum = {"PSNR": [], "SSIM": [], "RMSE": [], "MAE": [], "LPIPS": []}
    with torch.no_grad():
        for name in names:
            hr_bgr = cv2.imread(str(hr_dir / name), cv2.IMREAD_COLOR)
            lr_bgr = cv2.imread(str(lr_dir / name), cv2.IMREAD_COLOR)
            if hr_bgr is None or lr_bgr is None:
                raise FileNotFoundError(name)
            hr = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            lr = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if hr.ndim != 3 or lr.ndim != 3 or hr.shape[2] != 3 or lr.shape[2] != 3:
                raise RuntimeError(f"Holographic evaluation requires RGB images: {name}")
            expected_hr = (lr.shape[0] * int(option.get("scale", 4)),
                           lr.shape[1] * int(option.get("scale", 4)))
            if tuple(hr.shape[:2]) != expected_hr:
                raise RuntimeError(
                    f"Native HR/LR size mismatch for {name}: HR={hr.shape[:2]}, "
                    f"LR={lr.shape[:2]}, scale={option.get('scale', 4)}")
            inp = torch.from_numpy(lr.transpose(2, 0, 1)).unsqueeze(0).to(device)
            sr = model(inp).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
            if sr.shape != hr.shape:
                raise RuntimeError(f"Shape mismatch for {name}: SR={sr.shape}, HR={hr.shape}")
            if not a.visual_only:
                values = metric_values(sr, hr, evaluation)
                for key in ("PSNR", "SSIM", "RMSE", "MAE"):
                    accum[key].append(values[key])
            if lpips_fn is not None:
                lpips_crop = int(evaluation.get("lpips_crop_border", evaluation.get("crop_border", 4)))
                sr_lp = sr[lpips_crop:-lpips_crop, lpips_crop:-lpips_crop]
                hr_lp = hr[lpips_crop:-lpips_crop, lpips_crop:-lpips_crop]
                x = torch.from_numpy(sr_lp.transpose(2, 0, 1)).unsqueeze(0).to(device) * 2 - 1
                y = torch.from_numpy(hr_lp.transpose(2, 0, 1)).unsqueeze(0).to(device) * 2 - 1
                accum["LPIPS"].append(float(lpips_fn(x, y).mean().item()))
            if a.split == "test" and name in visual["save_names"]:
                base = out_root / Path(name).stem
                lr_u8 = np.clip(cv2.cvtColor(lr, cv2.COLOR_RGB2BGR) * 255.0, 0, 255).astype(np.uint8)
                sr_u8 = np.clip(cv2.cvtColor(sr, cv2.COLOR_RGB2BGR) * 255.0, 0, 255).astype(np.uint8)
                hr_u8 = np.clip(cv2.cvtColor(hr, cv2.COLOR_RGB2BGR) * 255.0, 0, 255).astype(np.uint8)
                outputs = (
                    ("save_lr", base.with_name(base.name + "_LR.bmp"), lr_u8),
                    ("save_sr", base.with_name(base.name + "_SR.bmp"), sr_u8),
                    ("save_hr", base.with_name(base.name + "_HR.bmp"), hr_u8),
                )
                for enabled_key, output_path, image in outputs:
                    if str(visual.get(enabled_key, "true")).lower() == "true":
                        if not cv2.imwrite(str(output_path), image):
                            raise IOError(f"Could not save visual output: {output_path}")

    summary = {k: (float(np.mean(v)) if v else None) for k, v in accum.items()}
    summary["count"] = len(names)
    summary["visual_images_saved"] = len(names) if a.split == "test" else 0
    summary["visual_only"] = bool(a.visual_only)
    summary_name = "visual_export.json" if a.visual_only else "metrics.json"
    (out_root / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

