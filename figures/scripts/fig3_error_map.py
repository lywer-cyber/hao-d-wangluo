#!/usr/bin/env python3
"""
Fig.3 —— 误差图 / 残差热力图（协议 §C Fig.3）。

内容：|SR − HR|，逐方法 × 逐场景。
作用：直观回答"谁数值上更接近 GT"，与 Fig.2 的视觉印象互为验证。

冻结规则
  - **colorbar 全局统一**：跨所有方法、所有场景取同一 vmax（99 分位）    §C Fig.3 第 2 条
    自适应色标会把"误差大"画成"误差小"，是最容易被审稿人抓到的操作
  - colormap 用 viridis（感知均匀 + 色盲友好）；**禁止 jet**            §C Fig.3 第 3 条
  - 每格标该样本该方法的 MAE，数值来自 per_sample_metrics.json        §C Fig.3 第 4 条
  - 布局与 Fig.2 对齐（同场景、同方法顺序），便于读者对照              §C Fig.3 第 6 条

用法
  python fig3_error_map.py \
      --metrics figures/data/per_sample_metrics_all.json \
      --roi data/splits/roi.json \
      --hr-dir $HOLO_DATA_ROOT/HR --exp-root experiments \
      --out-stem figures/fig3_error_map
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIG_FONT_SIZE, FIG_SEED, info, load_json, save_paper_figure,
    setup_matplotlib, to_display_gray, warn,
)
from fighelpers import (  # noqa: E402
    annotate_metric, global_vmax, load_roi, ordered_methods, panel_grid,
    read_hr, read_sr, tight_panel,
)

CMAP = 'viridis'          # 协议 §C Fig.3 第 3 条：感知均匀 + 色盲友好


def get_mae(metrics: Dict, model: str, seed: int, number: str,
            variant: str = 'plain') -> Optional[float]:
    per = (metrics.get('results', {}).get(model) or {})
    per = per.get(str(seed)) or per.get(seed) or {}
    rec = per.get(str(number)) or per.get(number)
    if not rec:
        return None
    v = rec.get('mae' + ('_plus' if variant == 'plus' else ''))
    return None if v is None else float(v)


def main() -> int:
    ap = argparse.ArgumentParser(description='Fig.3 误差图')
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--roi', required=True)
    ap.add_argument('--hr-dir', required=True)
    ap.add_argument('--exp-root', required=True)
    ap.add_argument('--seed', type=int, default=FIG_SEED)
    ap.add_argument('--ours-pattern', default='HoloRestormer')
    ap.add_argument('--with-plus', action='store_true')
    ap.add_argument('--gain', type=float, default=1.0,
                    help='统一增益（看清暗部差异用），会写进图注；默认 1.0')
    ap.add_argument('--vmax-percentile', type=float, default=99.0)
    ap.add_argument('--scenes', nargs='*', default=None,
                    help='场景号；默认用 roi.json 的 fig2_samples')
    ap.add_argument('--cell', type=float, default=1.05)
    ap.add_argument('--width', choices=['single', 'double'], default='double')
    ap.add_argument('--out-stem', default='figures/fig3_error_map')
    args = ap.parse_args()

    metrics = load_json(args.metrics)
    roi = load_roi(args.roi)
    scenes = [str(s) for s in (args.scenes or roi.get('fig2_samples', []))]
    if not scenes:
        print('错误: 没有场景（roi.json 缺 fig2_samples 或未给 --scenes）',
              file=sys.stderr)
        return 2

    variants = ('plain', 'plus') if args.with_plus else ('plain',)
    methods = ordered_methods(args.metrics, args.ours_pattern, variants,
                              exp_root=args.exp_root, seed=args.seed)
    if not methods:
        print('错误: 没有可用的方法', file=sys.stderr)
        return 2

    # ---- 先算全部误差图，才能取全局 vmax ----
    errs: List[Dict] = []
    hr_cache = {}
    for m in methods:
        row = {'display': str(m['display']), 'maps': {}, 'meta': m}
        for n in scenes:
            try:
                sr = to_display_gray(read_sr(args.exp_root, str(m['model']),
                                             args.seed, n, str(m['variant'])))
            except FileNotFoundError as e:
                warn(str(e))
                continue
            if n not in hr_cache:
                hr_cache[n] = to_display_gray(read_hr(args.hr_dir, n))
            hr = hr_cache[n]
            if sr.shape != hr.shape:
                warn(f'{m["display"]}/{n}: SR {sr.shape} != HR {hr.shape}，跳过')
                continue
            row['maps'][n] = np.abs(sr - hr) * args.gain
        if row['maps']:
            errs.append(row)

    vmax = global_vmax([mp for r in errs for mp in r['maps'].values()],
                       args.vmax_percentile)
    info(f'全局 colorbar vmax = {vmax:.4f}（{args.vmax_percentile:g} 分位，所有面板共用）')
    if args.gain != 1.0:
        info(f'统一增益 ×{args.gain:g}（必须写进图注）')

    # ---- 画 ----
    plt = setup_matplotlib()
    n_rows, n_cols = len(errs), len(scenes)
    fig, axes = panel_grid(n_rows, n_cols, width=args.width, cell_in=args.cell)

    im = None
    for ri, row in enumerate(errs):
        for ci, n in enumerate(scenes):
            ax = axes[ri][ci]
            tight_panel(ax)
            mp = row['maps'].get(n)
            if mp is None:
                ax.axis('off')
                continue
            im = ax.imshow(mp, cmap=CMAP, vmin=0.0, vmax=vmax,
                           interpolation='nearest', aspect='equal')
            mae = get_mae(metrics, str(row['meta']['model']), args.seed, n,
                          str(row['meta']['variant']))
            if mae is not None:
                annotate_metric(ax, f'MAE {mae:.4f}', loc='tl',
                                fontsize=FIG_FONT_SIZE - 2)

    for ri, row in enumerate(errs):
        axes[ri][0].set_ylabel(row['display'], rotation=0, ha='right',
                               va='center', labelpad=8, fontsize=FIG_FONT_SIZE)
    for ci, n in enumerate(scenes):
        axes[0][ci].set_title(f'sample {n}', fontsize=FIG_FONT_SIZE - 1, pad=2)

    # 全局共用一条 colorbar（不是每格一条）
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02,
                            pad=0.01, aspect=30)
        label = r'$|\mathrm{SR}-\mathrm{HR}|$'
        if args.gain != 1.0:
            label += f' $\\times$ {args.gain:g}'
        cbar.set_label(label, fontsize=FIG_FONT_SIZE - 1)
        cbar.ax.tick_params(labelsize=FIG_FONT_SIZE - 2)

    paths = save_paper_figure(fig, Path(args.out_stem))
    plt.close(fig)
    for p in paths:
        info(f'已写出 {p}')
    info(f'图注需写明: 误差定义 |SR−HR|，colormap={CMAP}，'
         f'全局 vmax={vmax:.4f}（{args.vmax_percentile:g} 分位）'
         + (f'，统一增益 ×{args.gain:g}' if args.gain != 1.0 else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
