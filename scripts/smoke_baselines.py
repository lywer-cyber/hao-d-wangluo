#!/usr/bin/env python3
"""CPU-only construction/forward checks for the five archived configurations."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def one(name):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import yaml
    entry = json.loads((ROOT / "configs/baseline_catalog.json").read_text())["models"][name]
    sys.path.insert(0, str(ROOT / entry["source_tree"]))
    import torch
    from basicsr.models.archs import define_network
    from basicsr.utils.holo_protocol import configure_determinism, validate_experiment_options
    cfg = yaml.safe_load((ROOT / entry["config"]).read_text())
    torch.set_num_threads(1)
    configure_determinism(cfg["manual_seed"], cfg["determinism"])
    validate_experiment_options(cfg)
    model = define_network(dict(cfg["network_g"])).cpu().eval()
    params = sum(p.numel() for p in model.parameters())
    if params != entry["params"]:
        raise RuntimeError(f"{name}: parameter count {params}, expected {entry['params']}")
    with torch.no_grad():
        y = model(torch.rand(1, 3, 16, 16))
    if tuple(y.shape) != (1, 3, 64, 64) or not torch.isfinite(y).all():
        raise RuntimeError(f"Invalid output for {name}")
    return {"model": name, "parameters": params, "output_shape": list(y.shape),
            "finite": True, "device": "cpu", "training_started": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--one", default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    if a.one:
        print("SMOKE_JSON=" + json.dumps(one(a.one)))
        return
    names = json.loads((ROOT / "configs/baseline_catalog.json").read_text())["models"]
    rows = []
    for name in names:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", name],
                              capture_output=True, text=True, timeout=90)
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
        line = next(l for l in proc.stdout.splitlines() if l.startswith("SMOKE_JSON="))
        rows.append(json.loads(line.split("=", 1)[1]))
    result = {"passed": True, "gpu_used": False, "models": rows}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
