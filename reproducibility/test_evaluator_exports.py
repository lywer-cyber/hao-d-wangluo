#!/usr/bin/env python3
"""Check new export serialization against the original evaluator on CPU fixtures.

The SR network and LPIPS module are deterministic test doubles. This checks
export plumbing, metric-expression preservation and BMP identity, not a new
benchmark or a validation of LPIPS model accuracy. No real training is run.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/holo_v7_baseline/restormer"))
import cv2
import numpy as np
import torch
import yaml

torch.set_num_threads(1)


class TinySR(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)


class FakeLPIPS(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return (x - y).square().mean().reshape(1)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def check():
    fake = types.ModuleType("lpips")
    fake.LPIPS = FakeLPIPS
    sys.modules["lpips"] = fake
    rows = []
    frozen_determinism = yaml.safe_load((ROOT / "configs/holo_v7_baseline/holo_restormer_mdta_stockhead_60000_seed10_mb2_options.yml").read_text())["determinism"]
    with tempfile.TemporaryDirectory(prefix="holo_eval_fixture_") as td:
        t = Path(td)
        for folder in ["HR", "LR4"]:
            (t / folder).mkdir()
        rng = np.random.default_rng(20)
        for name in ["1.bmp", "2.bmp"]:
            lr = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
            hr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            assert cv2.imwrite(str(t / "LR4" / name), lr)
            assert cv2.imwrite(str(t / "HR" / name), hr)
        checkpoint = t / "fixture.pth"
        torch.save({"params": {}, "params_ema": {}}, checkpoint)
        for mode in ["test", "val", "visual_only"]:
            outputs = {}
            for label, source in [
                ("original", ROOT / "reproducibility/evaluate_holo_server_snapshot.py"),
                ("published", ROOT / "code/holo_v7_baseline/restormer/evaluate_holo.py"),
            ]:
                root = t / (mode + "_" + label)
                split = root / "data/splits"
                split.mkdir(parents=True)
                for phase in ["val", "test"]:
                    (split / (phase + "_meta.txt")).write_text("1.bmp\n2.bmp\n")
                visual = split / "visual_samples.txt"
                visual.write_text("1.bmp\n")
                cfg = {"name": "fixture", "manual_seed": 10, "scale": 4,
                       "determinism": frozen_determinism,
                       "path": {"root": str(root)}, "network_g": {"type": "TinySR"},
                       "datasets": {"val": {"dataroot_gt": str(t / "HR"), "dataroot_lq": str(t / "LR4")}},
                       "visual_output": {"save_list": str(visual), "save_count": 1,
                                         "format": "BMP", "compression": "none"},
                       "evaluation": {"crop_border": 4}}
                option = root / "options.yml"
                option.write_text(yaml.safe_dump(cfg))
                m = load_module(source, "evaluator_" + mode + "_" + label)
                selected_split = "val" if mode == "val" else "test"
                m.args = lambda option=option, selected_split=selected_split: types.SimpleNamespace(
                    option=option, checkpoint=checkpoint, split=selected_split,
                    visual_only=(mode == "visual_only"), require_lpips=(mode == "test"), device="cpu")
                m.define_network = lambda _: TinySR()
                m.validate_experiment_options = lambda _: None  # tiny fixture, not a full training config
                m.validate_runtime_versions = lambda *args, **kwargs: None
                m.locate_lpips_weights = lambda: t / "fixture_lpips.pth"
                m.main()
                folder = root / "experiments/fixture/results" / selected_split
                summary_name = "visual_export.json" if mode == "visual_only" else "metrics.json"
                outputs[label] = (folder, json.loads((folder / summary_name).read_text()))
            old_dir, old = outputs["original"]
            new_dir, new = outputs["published"]
            for key in ["PSNR", "SSIM", "RMSE", "MAE", "LPIPS", "count", "visual_only"]:
                assert old[key] == new[key], (mode, key, old[key], new[key])
            old_images = {p.name: p.read_bytes() for p in old_dir.glob("*.bmp")}
            new_images = {p.name: p.read_bytes() for p in new_dir.glob("*.bmp")}
            assert old_images == new_images
            if mode != "visual_only":
                per = json.loads((new_dir / "per_sample_metrics.json").read_text())
                assert len(per["samples"]) == 2
                for key in ["PSNR", "SSIM", "RMSE", "MAE", "LPIPS"]:
                    values = [v[key] for v in per["samples"].values() if v[key] is not None]
                    assert (float(np.mean(values)) if values else None) == new[key]
                assert (new_dir / "per_sample_metrics.csv").is_file()
            else:
                assert not (new_dir / "per_sample_metrics.json").exists()
            assert new["visual_images_saved"] == (0 if mode == "val" else 1)
            assert new["visual_files_saved"] == (0 if mode == "val" else 3)
            rows.append({"mode": mode, "aggregate_metrics_unchanged": True,
                         "bmp_bytes_unchanged": True, "new_exports_verified": True})
    return {"passed": True, "device": "cpu", "uses_test_doubles": True,
            "not_a_new_benchmark": True, "cases": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    result = check()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("EXPORT_TEST_JSON=" + json.dumps(result))
