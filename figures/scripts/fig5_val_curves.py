#!/usr/bin/env python3
"""
Fig.5 —— 验证曲线 + LR 衰减标记（协议 §C Fig.5）。

为什么必须有
  这张图直接回答审稿人"你们是不是在未收敛点上比"。
  milestone 竖线之后曲线**必须出现平台**；若没有，说明迭代数不足，
  触发协议 §0.3（升版本 + 全部重跑）。

冻结规则（协议 §C Fig.5）
  - 横轴 iteration (0 → total_iterations)，纵轴 val PSNR
  - 每方法一条线，**3 个种子的均值 ± 标准差阴影带**
  - **必须画 4 条竖直虚线**在 30000 / 48000 / 54000 / 57000（LR milestone ×0.5）
  - 可加小图显示 LR 曲线本身

用法
  python fig5_val_curves.py \
      --exp-root experiments \
      --models HoloRestormer SwinIR RCAN NAFNet DAT HAT SRResNet \
      --out-stem figures/fig5_val_curves \
      --plateau-check          # 顺带做平台判定，不通过则非零退出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIG_FONT_SIZE, LR_MILESTONES, SEEDS, TOTAL_ITERATIONS, figure_size, info,
    save_paper_figure, setup_matplotlib, style_map, warn,
)
from fighelpers import load_val_curves, mean_std_curve  # noqa: E402

PLATEAU_WINDOW = (0.85, 1.0)      # 末段 15% 区间
PLATEAU_MAX_GAIN_DB = 0.05        # 该区间内 PSNR 增幅上限（dB）


def plateau_report(curves: Dict, models: Sequence[str]) -> Tuple[bool, List[str]]:
    """判定末段是否平台化：末 15% 区间内 PSNR 增幅 < PLATEAU_MAX_GAIN_DB 即通过。"""
    msgs, ok = [], True
    lo_frac, hi_frac = PLATEAU_WINDOW
    for m in models:
        if m not in curves:
            msgs.append(f'{m}: 无验证曲线')
            ok = False
            continue
        grid, mean, _std = mean_std_curve(curves[m])
        if grid.size == 0:
            msgs.append(f'{m}: 曲线为空')
            ok = False
            continue
        lo, hi = lo_frac * grid[-1], hi_frac * grid[-1]
        sel = (grid >= lo) & (grid <= hi)
        if sel.sum() < 2:
            msgs.append(f'{m}: 末段样本点不足')
            ok = False
            continue
        gain = float(mean[sel][-1] - mean[sel][0])
        flag = 'OK ' if gain < PLATEAU_MAX_GAIN_DB else '未平台'
        if gain >= PLATEAU_MAX_GAIN_DB:
            ok = False
        msgs.append(f'{m}: 末段 [{lo:.0f},{hi:.0f}] 增幅 {gain:+.3f} dB -> {flag}')
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser(description='Fig.5 验证曲线 + LR milestone')
    ap.add_argument('--exp-root', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    ap.add_argument('--fname', default='val_metrics.csv')
    ap.add_argument('--out-stem', default='figures/fig5_val_curves')
    ap.add_argument('--xmax', type=float, default=float(TOTAL_ITERATIONS))
    ap.add_argument('--show-lr', action='store_true', help='下方加一个 LR 小图')
    ap.add_argument('--plateau-check', action='store_true',
                    help='做末段平台判定；不通过则退出码 1（触发协议 §0.3）')
    args = ap.parse_args()

    curves = load_val_curves(args.exp_root, args.models, args.seeds, args.fname)
    if not curves:
        print('错误: 一条验证曲线都没读到', file=sys.stderr)
        return 2

    plt = setup_matplotlib()
    styles = style_map(list(curves.keys()))
    nrow = 2 if args.show_lr else 1
    h = (2.55, 2.9) if args.show_lr else (2.55,)
    fig, axes = plt.subplots(nrow, 1, figsize=figure_size('double', h[0]),
                             squeeze=False, sharex=True,
                             gridspec_kw={'height_ratios': ([3, 1] if args.show_lr
                                                            else [1])})
    ax = axes[0][0]

    for m in sorted(curves):
        grid, mean, std = mean_std_curve(curves[m])
        if grid.size == 0:
            continue
        st = styles[m]
        n_seeds = len(curves[m])
        ax.plot(grid, mean, label=f'{m} ({n_seeds} seed{"s" if n_seeds > 1 else ""})',
                **st)
        if n_seeds > 1:
            ax.fill_between(grid, mean - std, mean + std, alpha=0.18,
                            color=st['color'], linewidth=0)

    for k, ms in enumerate(LR_MILESTONES):
        ax.axvline(ms, color='0.72', linestyle=':', linewidth=0.8, zorder=0)
        if k == 0:
            ax.text(ms, 0.98, 'LR milestones ($\\times$0.5)',
                    transform=ax.get_xaxis_transform(), rotation=90,
                    va='top', ha='right', fontsize=FIG_FONT_SIZE - 2,
                    color='0.45')

    ax.set_xlim(0, args.xmax)
    ax.set_ylabel('validation PSNR (dB)')
    ax.legend(loc='lower right', fontsize=FIG_FONT_SIZE - 2, ncol=2)
    ax.grid(True, linewidth=0.3, alpha=0.30)
    if not args.show_lr:
        ax.set_xlabel('iteration')

    if args.show_lr:
        import csv as _csv
        axl = axes[1][0]
        _lrs_all = []
        for m in sorted(curves):
            cands = [Path(args.exp_root) / m / f'seed{args.seeds[0]}' / 'logs' / args.fname,
                     Path(args.exp_root) / m / 'logs' / args.fname]
            p = next((c for c in cands if c.exists()), None)
            if p is None:
                continue
            its, lrs = [], []
            try:
                with p.open('r', encoding='utf-8', newline='') as f:
                    for row in _csv.DictReader(f):
                        if row.get('lr'):
                            its.append(float(row['iter']))
                            lrs.append(float(row['lr']))
            except (OSError, KeyError, ValueError):
                continue
            # 有数据就画：LR 恒定也是有效信息（"milestone 没生效"正是协议 §8 要抓的）
            if its:
                axl.plot(its, lrs, label=m, linewidth=1.0)
                _lrs_all.extend(lrs)
        for ms in LR_MILESTONES:
            axl.axvline(ms, color='0.72', linestyle=':', linewidth=0.8, zorder=0)
        axl.set_yscale('log')
        if _lrs_all:
            _lo, _hi = min(_lrs_all), max(_lrs_all)
            if _lo == _hi:          # 全程恒定：给个对称范围，别让 log 轴退化
                _lo, _hi = _lo / 3.0, _hi * 3.0
            axl.set_ylim(_lo / 1.5, _hi * 1.5)
        axl.set_ylabel('learning rate')
        axl.set_xlabel('iteration')
        axl.grid(True, linewidth=0.3, alpha=0.30)

    for p in save_paper_figure(fig, Path(args.out_stem)):
        info(f'已写出 {p}')
    plt.close(fig)

    if args.plateau_check:
        ok, msgs = plateau_report(curves, args.models)
        print('\n末段平台判定（阈值 %.2f dB / 末 %.0f%% 区间）:'
              % (PLATEAU_MAX_GAIN_DB, 100 * (1 - PLATEAU_WINDOW[0])), file=sys.stderr)
        for m in msgs:
            print('  ' + m, file=sys.stderr)
        if not ok:
            print('\n判定未通过 -> 迭代数可能不足，触发协议 §0.3'
                  '（升版本号 + 全部实验重跑）', file=sys.stderr)
            return 1
        print('\n判定通过：末段已平台化 ✓', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
