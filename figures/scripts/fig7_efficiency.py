#!/usr/bin/env python3
"""
Fig.7 —— 精度-效率散点（协议 §C Fig.7，主张效率时必需）。

横轴：参数量 (M) 或 FLOPs (G) @ 3×512×512
纵轴：测试集 PSNR (dB)。该模型跑了几个种子就用几个种子的均值；
      多于 1 个种子时画标准差误差棒，只有单种子则不画（图注需如实说明）。

理想位置是**左上角**（低代价 + 高精度），可在图注点明。
坐标轴是否 log 由 --xlog / --ylog 控制，**必须在图注说明**。

用法
  python fig7_efficiency.py \
      --metrics figures/data/per_sample_metrics_all.json \
      --cost    figures/data/model_cost.csv \
      --x params_m --out-stem figures/fig7_efficiency --xlog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIG_FONT_SIZE, FIG_SEED, SEEDS, figure_size, info, load_json,
    save_paper_figure, setup_matplotlib, style_map, warn,
)
from fighelpers import read_cost_csv  # noqa: E402

X_LABELS = {
    'params_m': 'Parameters (M)',
    'flops_g': 'FLOPs (G) @ $3\\times512\\times512$',
    'infer_ms': 'Inference time (ms)',
}


def collect(metrics_path, cost_path, seed: int, ours_pattern: str
            ) -> List[Dict]:
    data = load_json(metrics_path)
    res = data.get('results', {})
    costs = read_cost_csv(cost_path)
    rows = []
    for model, by_seed in res.items():
        if model not in costs:
            continue
        # 逐图先算均值，再跨种子统计：有几个种子用几个（不再只看 seed 参数）
        per_seed, seeds_used = [], []
        for s in sorted(by_seed, key=lambda x: int(str(x))):
            per = by_seed.get(s) or {}
            vals = [float(r['psnr']) for r in per.values()
                    if r.get('psnr') is not None]
            if vals:
                per_seed.append(float(np.mean(vals)))
                seeds_used.append(int(str(s)))
        if not per_seed:
            continue
        rows.append({
            'model': model,
            'is_ours': model.startswith(ours_pattern),
            'psnr': float(np.mean(per_seed)),
            'psnr_std': (float(np.std(per_seed, ddof=1))
                         if len(per_seed) > 1 else 0.0),
            'n_seeds': len(per_seed),
            'seeds': seeds_used,
            **costs[model],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description='Fig.7 精度-效率散点')
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--cost', required=True)
    ap.add_argument('--x', choices=sorted(X_LABELS), default='params_m')
    ap.add_argument('--seed', type=int, default=FIG_SEED)
    ap.add_argument('--ours-pattern', default='HoloRestormer')
    ap.add_argument('--xlog', action='store_true')
    ap.add_argument('--out-stem', default='figures/fig7_efficiency')
    args = ap.parse_args()

    rows = collect(args.metrics, args.cost, args.seed, args.ours_pattern)
    rows = [r for r in rows if r.get(args.x) is not None]
    if not rows:
        print(f'错误: 没有一个模型同时具备 {args.x} 与 PSNR；'
              f'检查 --cost 的列名', file=sys.stderr)
        return 2

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=figure_size('single', 2.5))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.95, bottom=0.17)

    styles = style_map([r['model'] for r in rows])
    for r in rows:
        st = styles[r['model']]
        ours = r['is_ours']
        ax.scatter(r[args.x], r['psnr'],
                   s=34 if ours else 20,
                   marker='*' if ours else st['marker'],
                   facecolor=st['color'], edgecolor='black',
                   linewidth=0.5 if not ours else 0.9, zorder=5 if ours else 4)
        if r.get('n_seeds', 1) > 1:
            ax.errorbar(r[args.x], r['psnr'], yerr=r['psnr_std'],
                        fmt='none', ecolor=st['color'], elinewidth=0.8,
                        capsize=1.5, zorder=3)
        ax.annotate(r['model'], (r[args.x], r['psnr']),
                    textcoords='offset points', xytext=(4, 3),
                    fontsize=FIG_FONT_SIZE - 2)

    if args.xlog:
        ax.set_xscale('log')
    ax.set_xlabel(X_LABELS[args.x] + (' (log scale)' if args.xlog else ''))
    ax.set_ylabel('PSNR (dB)')
    ax.grid(True, linewidth=0.3, alpha=0.30)
    ax.margins(x=0.12, y=0.10)

    for p in save_paper_figure(fig, Path(args.out_stem)):
        info(f'已写出 {p}')
    plt.close(fig)

    _ns = sorted({int(s) for r in rows for s in r.get('seeds', [])})
    info(f'图注需写明: x/y 轴、是否 log（xlog={args.xlog}）、'
         f'FLOPs 的输入规格 3×512×512、PSNR 由种子 {_ns} 平均得到'
         f'（共 {len(_ns)} 个种子；单种子时无误差棒）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
