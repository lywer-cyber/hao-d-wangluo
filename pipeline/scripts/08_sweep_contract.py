#!/usr/bin/env python3
"""把 07_sweep_infer.py 产出的 per-扫描点 metrics.json 汇总成 Fig.8 的 contract

`fig8_sensitivity.py` 的 `build` 子命令是从 SR 图重算指标的，会要求落
1350 x N 张 3.9MB 的 BMP（几十 GB）。本脚本改用 07 已经算好的 metrics.json，
**不需要 SR 图**，直接产出同样结构的 contract，交给 `fig8_sensitivity.py plot`。

产出
  <out-dir>/sensitivity_<param>.json      每个扫描参数一份

用法
  python 08_sweep_contract.py --sweep-root <sweep> --out-dir figures/data \
      [--only-params speckle_contrast diffraction_cutoff]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL_VERSION = "v7"


def sort_key(v: str):
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[10])
    ap.add_argument("--only-params", nargs="+", default=None)
    args = ap.parse_args()

    manifest = {}
    mf = args.sweep_root / "sweep_manifest.json"
    if mf.is_file():
        manifest = json.loads(mf.read_text(encoding="utf-8"))

    by_param: dict[str, set] = {}
    for d in sorted(args.sweep_root.iterdir()):
        if d.is_dir() and "=" in d.name:
            param, val = d.name.split("=", 1)
            by_param.setdefault(param, set()).add(val)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for param in sorted(by_param):
        if args.only_params and param not in args.only_params:
            continue
        values = sorted(by_param[param], key=sort_key)
        results: dict[str, dict[str, dict]] = {}
        n_images = 0
        for val in values:
            d = args.sweep_root / f"{param}={val}"
            if not d.is_dir():
                continue
            for model_dir in sorted(p for p in d.iterdir() if p.is_dir()):
                mj = model_dir / "results" / "test" / "metrics.json"
                if not mj.is_file():
                    continue
                r = json.loads(mj.read_text(encoding="utf-8"))
                results.setdefault(model_dir.name, {})[val] = {
                    "psnr": r.get("PSNR"), "ssim": r.get("SSIM"),
                    "n": r.get("count"),
                }
                n_images = max(n_images, int(r.get("count") or 0))
        if not results:
            print(f"  跳过 {param}：没有任何模型的 metrics.json")
            continue

        fixed = dict(manifest.get("fixed", {}))
        fixed.pop(param, None)
        contract = {
            "protocol_version": PROTOCOL_VERSION,
            "sweep_param": param,
            "sweep_values": [float(v) for v in values],
            "fixed_params": fixed,
            "reference_set": (
                "independent analysis set; not used for checkpoint selection; "
                "same HR images as test split but degraded at swept parameters"),
            "n_images": n_images,
            "seeds": list(args.seeds),
            "results": results,
        }
        out = args.out_dir / f"sensitivity_{param}.json"
        out.write_text(json.dumps(contract, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        n_written += 1

        models = sorted(results)
        print(f"\n== {param} ==  ({n_images} 张/点, 模型 {len(models)} 个)")
        print("  " + "值".ljust(10) + "".join(m[:14].rjust(15) for m in models))
        for val in values:
            row = f"  {val:<10}"
            for m in models:
                rec = results[m].get(val)
                row += (f"{rec['psnr']:>15.4f}" if rec and rec.get("psnr") is not None
                        else f"{'--':>15}")
            print(row)
        print(f"  -> 已写出 {out}")

    print(f"\n[08] 共 {n_written} 份 contract")
    return 0 if n_written else 2


if __name__ == "__main__":
    raise SystemExit(main())
