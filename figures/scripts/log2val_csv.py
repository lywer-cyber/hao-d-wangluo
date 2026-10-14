#!/usr/bin/env python3
"""把训练日志解析成 val_metrics.csv（协议 §11.9 要求的逐验证点指标表）。

训练代码（holo_project / holo_v7_baseline 两份都是）只把验证结果打进日志，
**不写 val_metrics.csv**；而 Fig.5 的 `load_val_curves` 读的正是这个文件。本脚本补上这一环。

日志里验证行不含 iter，只能靠"上一条带 iter 的进度行"推断：
    2026-01-01 00:00:00 INFO: [holo_..][epoch: 5, iter: 25,000, lr:(4.998e-05,)] ... 
    2026-01-01 00:00:00 INFO: Saving models and training states.
    2026-01-01 00:07:00 INFO: Validation HoloVal,  # psnr: 31.0239  # ssim: 0.9008 ...
因此取"验证行之前最后一次见到的 iter / lr"。

用法
  python log2val_csv.py --log train.log --out-dir experiments/HoloRestormer/seed10/logs
  python log2val_csv.py --log train.log --index -1 --out val_metrics.csv   # 多段日志取最后一段
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ITER_RE = re.compile(r"iter:\s*([0-9,]+),\s*lr:\(([0-9.eE+-]+),\)")
VAL_RE = re.compile(
    r"Validation\s+\S+,\s*#\s*psnr:\s*([0-9.]+)\s*#\s*ssim:\s*([0-9.]+)"
    r"(?:\s*#\s*rmse:\s*([0-9.]+))?(?:\s*#\s*mae:\s*([0-9.]+))?")
START_RE = re.compile(r"Start training from epoch:\s*\d+,\s*iter:\s*([0-9,]+)")


def split_runs(lines: List[str]) -> List[Tuple[int, List[str]]]:
    """按 'Start training' 把日志切成若干段，返回 (起始行号, 行列表)。"""
    starts = [i for i, l in enumerate(lines) if START_RE.search(l)]
    if not starts:
        return [(0, lines)]
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(lines)
        out.append((s, lines[s:e]))
    return out


def parse(lines: List[str]) -> List[dict]:
    rows, last_iter, last_lr = [], None, None
    for line in lines:
        m = ITER_RE.search(line)
        if m:
            last_iter = int(m.group(1).replace(',', ''))
            last_lr = float(m.group(2))
            continue
        v = VAL_RE.search(line)
        if v:
            psnr, ssim, rmse, mae = v.group(1), v.group(2), v.group(3), v.group(4)
            rows.append({
                'iter': '' if last_iter is None else last_iter,
                'psnr': psnr, 'ssim': ssim,
                'rmse': rmse or '', 'mae': mae or '',
                'lr': '' if last_lr is None else ('%.6e' % last_lr),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description='训练日志 -> val_metrics.csv')
    ap.add_argument('--log', required=True, help='训练日志（可含多段 run）')
    ap.add_argument('--index', type=int, default=-1,
                    help='第几段 run（从 0 起，默认 -1 = 最后一段）')
    ap.add_argument('--out', default=None, help='输出 csv 路径')
    ap.add_argument('--out-dir', default=None,
                    help='输出目录（与它二选一），写 {out-dir}/val_metrics.csv')
    ap.add_argument('--append', action='store_true', help='已有文件时追加而不是覆盖')
    args = ap.parse_args()

    lines = Path(args.log).read_text(encoding='utf-8', errors='ignore').splitlines()
    runs = split_runs(lines)
    if not runs:
        print('错误：没找到 Start training 标记', file=sys.stderr)
        return 2
    try:
        seg = runs[args.index]
    except IndexError:
        print(f'错误：--index {args.index} 超出范围（共 {len(runs)} 段）', file=sys.stderr)
        return 2

    rows = parse(seg[1])
    if not rows:
        print('错误：这一段里没有验证记录', file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else (Path(args.out_dir) / 'val_metrics.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if (args.append and out.exists()) else 'w'
    with open(out, mode, newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['iter', 'psnr', 'ssim', 'rmse', 'mae', 'lr'])
        if mode == 'w':
            w.writeheader()
        w.writerows(rows)

    print(f'{out} 写入 {len(rows)} 个验证点（第 {args.index} 段，共 {len(runs)} 段）')
    for r in rows:
        print(f"  iter {str(r['iter']):>6s}  psnr={r['psnr']:>8s}  ssim={r['ssim']:>6s}  lr={r['lr']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
