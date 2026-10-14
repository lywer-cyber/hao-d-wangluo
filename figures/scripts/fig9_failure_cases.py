#!/usr/bin/env python3
"""
Fig.9 —— 失败案例分析（协议 §C Fig.9）。

作用：把"局限"讲清楚。只展示成功案例会被审稿人当成选择性报告；
给出失败案例 + 该样本的退化参数，是拿正面评价的加分项。

冻结规则
  - **样本选取规则必须写进图注**（协议 §L），且要说明依据是 Ours 还是基线  §C Fig.9
  - 必须附该样本的**退化参数**（来自 degrade_params.json，协议 §2.2）
  - 禁止只展示成功案例而不给失败案例                                      §C Fig.9

用法
  python fig9_failure_cases.py \
      --metrics figures/data/per_sample_metrics_all.json \
      --hr-dir $HOLO_DATA_ROOT/HR --lr-dir $HOLO_DATA_ROOT/LR4 \
      --exp-root experiments \
      --degrade-params $HOLO_DATA_ROOT/LR4/degrade_params.json \
      --rank-by HoloRestormer --n 3 --with-error \
      --out-stem figures/fig9_failure_cases
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIG_FONT_SIZE, FIG_SEED, info, load_json, save_paper_figure,
    setup_matplotlib, to_display_gray, warn,
)
from fighelpers import (  # noqa: E402
    annotate_metric, global_display_range, global_vmax, panel_grid, read_hr,
    read_lr, read_sr, tight_panel, upsample_lr,
)

PARAM_ORDER = ('diffraction_cutoff', 'aa_sigma', 'speckle_contrast',
               'speckle_corr', 'poisson_gain', 'read_std')
PARAM_SHORT = {
    'diffraction_cutoff': 'cutoff',
    'aa_sigma': r'$\sigma_{aa}$',
    'speckle_contrast': 'speckle',
    'speckle_corr': 'corr',
    'poisson_gain': 'gain',
    'read_std': r'$\sigma_{read}$',
}


def rank_samples(metrics: Dict, model: str, seed: int, n: int,
                 worst: bool = True, variant: str = 'plain',
                 quantile: Optional[float] = None) -> Tuple[List[str], str]:
    """按逐图 PSNR 排序取样本。返回 (样本号, 规则描述)。"""
    key = 'psnr' + ('_plus' if variant == 'plus' else '')
    per = (metrics.get('results', {}).get(model) or {})
    per = per.get(str(seed)) or per.get(seed) or {}
    items = [(str(k), float(v[key])) for k, v in per.items()
             if v.get(key) is not None and np.isfinite(float(v[key]))]
    if not items:
        raise ValueError(f'{model}/seed{seed} 没有逐图 PSNR')
    items.sort(key=lambda kv: kv[1], reverse=not worst)

    if quantile is not None:
        thr = float(np.percentile([p for _, p in items], 100 * quantile))
        sel = [k for k, p in items if p <= thr][:n]
        rule = (f'lowest {100*quantile:.0f}% of per-image PSNR of {model} '
                f'(threshold {thr:.2f} dB), first {len(sel)}')
        return sel or [k for k, _ in items[:n]], rule

    rule = (f'{n} worst samples ranked by per-image PSNR of {model} '
            f'(seed {seed})')
    return [k for k, _ in items[:n]], rule


def get_rec(metrics: Dict, model: str, seed: int, number: str,
            variant: str = 'plain') -> Optional[Dict]:
    per = (metrics.get('results', {}).get(model) or {})
    per = per.get(str(seed)) or per.get(seed) or {}
    return per.get(str(number)) or per.get(number)


def param_line(params: Optional[Dict], number: str) -> str:
    if not params:
        return ''
    rec = params.get(str(number)) or params.get(f'{number}.bmp')
    if not isinstance(rec, dict):
        return ''
    parts = []
    for k in PARAM_ORDER:
        if k in rec:
            v = float(rec[k])
            parts.append(f'{PARAM_SHORT[k]}={v:.3g}')
    return '  '.join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description='Fig.9 失败案例分析')
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--hr-dir', required=True)
    ap.add_argument('--lr-dir', required=True)
    ap.add_argument('--exp-root', required=True)
    ap.add_argument('--degrade-params', default=None,
                    help='degrade_params.json（协议 §2.2），用于图注')
    ap.add_argument('--rank-by', required=True,
                    help='按哪个模型的 PSNR 挑样本（写进图注，必须如实说明）')
    ap.add_argument('--ours-pattern', default='HoloRestormer',
                    help='本方法的名字前缀；只有当 rank-by 命中它时列标题才标 (ours)')
    ap.add_argument('--seed', type=int, default=FIG_SEED)
    ap.add_argument('--variant', choices=['plain', 'plus'], default='plain')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--worst', action='store_true', default=True)
    ap.add_argument('--best', dest='worst', action='store_false',
                    help='改成展示最好的样本（默认展示最差）')
    ap.add_argument('--quantile', type=float, default=None,
                    help='取低于该分位的样本，如 0.05 表示最低 5%%')
    ap.add_argument('--with-error', action='store_true',
                    help='增加一列 |Ours−HR| 误差图')
    ap.add_argument('--cell', type=float, default=1.5)
    ap.add_argument('--out-stem', default='figures/fig9_failure_cases')
    args = ap.parse_args()

    metrics = load_json(args.metrics)
    params = load_json(args.degrade_params) if args.degrade_params else None
    if args.degrade_params and params is None:
        warn(f'读不到退化参数 {args.degrade_params}，图注将不含参数')

    samples, rule = rank_samples(metrics, args.rank_by, args.seed, args.n,
                                 args.worst, args.variant, args.quantile)
    info(f'样本选取规则: {rule}')
    info(f'选中样本: {samples}')

    # ---- 载入图像 ----
    # 按谁排序就标谁；只有当它确实是本方法时才写 (ours)（协议 13.B.2 第 9 条）
    _tag = ' (ours)' if str(args.rank_by).startswith(args.ours_pattern) else ''
    cols: List[str] = ['LR (bicubic$\\times$4)', args.rank_by + _tag, 'HR (GT)']
    if args.with_error:
        cols.append(r'$|$ours$-$HR$|$')

    data: List[Dict] = []
    for n in samples:
        lr_up = to_display_gray(upsample_lr(read_lr(args.lr_dir, n)))
        hr = to_display_gray(read_hr(args.hr_dir, n))
        try:
            sr = to_display_gray(read_sr(args.exp_root, args.rank_by, args.seed, n,
                                         args.variant))
        except FileNotFoundError as e:
            warn(str(e))
            continue
        if sr.shape != hr.shape:
            warn(f'{n}: SR {sr.shape} != HR {hr.shape}，跳过')
            continue
        data.append({'number': n, 'lr': lr_up, 'sr': sr, 'hr': hr,
                     'err': np.abs(sr - hr)})
    if not data:
        print('错误: 一个样本都没能载入', file=sys.stderr)
        return 2

    drange = global_display_range([d['lr'] for d in data] +
                                  [d['sr'] for d in data] +
                                  [d['hr'] for d in data])
    evmax = global_vmax([d['err'] for d in data]) if args.with_error else 1.0

    # ---- 画：行=样本，列=LR / ours / HR [/ 误差] ----
    plt = setup_matplotlib()
    fig, axes = panel_grid(len(data), len(cols), width='double',
                           cell_in=args.cell, row_label_extra=0.0)
    vmin, vmax = drange
    em = None
    for ri, d in enumerate(data):
        imgs = [d['lr'], d['sr'], d['hr']] + ([d['err']] if args.with_error else [])
        for ci, img in enumerate(imgs):
            ax = axes[ri][ci]
            tight_panel(ax)
            if ci < 3:
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax,
                          interpolation='nearest', aspect='equal')
            else:
                em = ax.imshow(img, cmap='viridis', vmin=0.0, vmax=evmax,
                               interpolation='nearest', aspect='equal')
            if ci == 1:
                rec = get_rec(metrics, args.rank_by, args.seed, d['number'],
                              args.variant)
                if rec and rec.get('psnr') is not None:
                    annotate_metric(
                        ax, f"{float(rec['psnr']):.2f} / "
                            f"{float(rec.get('ssim') or float('nan')):.4f}",
                        loc='br', fontsize=FIG_FONT_SIZE - 2)
        axes[ri][0].set_ylabel(f"sample {d['number']}\n{param_line(params, d['number'])}",
                               rotation=0, ha='right', va='center',
                               labelpad=10, fontsize=FIG_FONT_SIZE - 2)
    for ci, c in enumerate(cols):
        axes[0][ci].set_title(c, fontsize=FIG_FONT_SIZE - 1, pad=2)
    if args.with_error and em is not None:
        cbar = fig.colorbar(em, ax=axes.ravel().tolist(), fraction=0.02,
                            pad=0.01, aspect=30)
        cbar.set_label(r'$|$SR$-$HR$|$', fontsize=FIG_FONT_SIZE - 2)
        cbar.ax.tick_params(labelsize=FIG_FONT_SIZE - 3)

    for p in save_paper_figure(fig, Path(args.out_stem)):
        info(f'已写出 {p}')
    plt.close(fig)

    info(f'图注必须写明: 选取规则「{rule}」；'
         f'若依据的是本方法（{args.rank_by}）必须如实说明；'
         f'anchor=seed {args.seed}；每个样本附退化参数'
         + (f'；误差图全局 vmax={evmax:.4f}' if args.with_error else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
