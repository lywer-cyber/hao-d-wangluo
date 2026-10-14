#!/usr/bin/env python3
"""Fig.8 退化参数敏感性扫描 —— 生成分析集

对 HR 逐通道施加 `holo_degrade.degrade_physical` 的**确定性物理退化**，
其余参数固定在测试集的标称值，只扫**一个**参数。

产出
  <out>/<param>=<value>/LR/<序号>.bmp      3 通道 BMP，与主实验 LR4 同规格

约定
  - 所有模型使用**同一批**扫描图；本集**不用于选 checkpoint**
  - 每张图、每个通道的随机种子由「文件名 + 通道号」派生，保证可复现
  - 扫描点的标称值应与测试集固定值一致，曲线才能与主表对齐

用法
  python 06_generate_sweep.py --hr-dir <HR> --list <test_meta.txt> \
      --limit 50 --out <sweep_root> [--params speckle_contrast diffraction_cutoff]
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holo_degrade import degrade_physical  # noqa: E402

# 测试集/验证集的固定退化参数（协议 §4；与 LR4/degrade_params.json 一致）
FIXED = dict(
    diffraction_kernel="circle",
    diffraction_cutoff=1.0,
    aa_sigma=0.50,
    speckle_enable=True,
    speckle_contrast=0.60,
    speckle_corr=4.0,
    poisson_gain=2000.0,
    read_std=0.005,
    bit_depth=8,
)

# 扫描点：跨过训练抖动范围，两端外推到训练分布之外（OOD）
SWEEPS: dict[str, list[float]] = {
    "speckle_contrast":   [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "diffraction_cutoff": [0.4, 0.6, 0.8, 1.0, 1.2],
    "poisson_gain":       [200.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
    "read_std":           [0.0, 0.005, 0.010, 0.020, 0.040],
    "aa_sigma":           [0.2, 0.35, 0.5, 0.7, 0.9],
}

SCALE = 4


def image_seed(name: str, channel: int) -> int:
    """由文件名与通道号派生稳定种子。"""
    return (zlib.crc32(name.encode("utf-8")) + 1000 * channel) % (2 ** 31)


def fmt(v: float) -> str:
    """目录名里的取值：整数不带小数点，其余用最短表示。"""
    return str(int(v)) if float(v).is_integer() else ("%g" % v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr-dir", required=True)
    ap.add_argument("--list", required=True, help="用哪些图，如 test_meta.txt")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", nargs="+", default=list(SWEEPS))
    args = ap.parse_args()

    hr_dir = Path(args.hr_dir)
    out = Path(args.out)
    names = [l.strip().split()[0] for l in
             Path(args.list).read_text(encoding="utf-8").splitlines() if l.strip()]
    names = names[:args.limit]
    if not names:
        print("错误: 清单为空", file=sys.stderr)
        return 2

    unknown = [p for p in args.params if p not in SWEEPS]
    if unknown:
        print(f"错误: 未知扫描参数 {unknown}；可选 {sorted(SWEEPS)}", file=sys.stderr)
        return 2

    manifest = {"scale": SCALE, "fixed": FIXED, "nominal_reference": 1.0,
                "sweeps": {p: SWEEPS[p] for p in args.params},
                "images": [Path(n).stem for n in names], "n_images": len(names)}

    total = 0
    for param in args.params:
        for v in SWEEPS[param]:
            cfg = dict(FIXED)
            cfg[param] = v
            dst = out / f"{param}={fmt(v)}" / "LR"
            dst.mkdir(parents=True, exist_ok=True)
            for n in names:
                stem = Path(n).stem
                src = hr_dir / n
                bgr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                if bgr is None:
                    print(f"错误: 读不到 {src}", file=sys.stderr)
                    return 2
                if bgr.ndim != 3 or bgr.shape[2] != 3:
                    print(f"错误: 需要 3 通道，{src} 是 {bgr.shape}", file=sys.stderr)
                    return 2
                hr01 = bgr.astype(np.float32) / 255.0
                chans = []
                for c in range(3):
                    rng = np.random.default_rng(image_seed(n, c))
                    lr = degrade_physical(hr01[:, :, c], SCALE, cfg, rng,
                                          randomize=False)
                    chans.append(np.clip(np.round(lr * 255.0), 0, 255).astype(np.uint8))
                cv2.imwrite(str(dst / f"{stem}.bmp"), np.stack(chans, axis=2))
                total += 1
            print(f"  {param}={fmt(v)}  已出 {len(names)} 张")

    (out / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[06] 完成：{total} 张，清单 {out / 'sweep_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
