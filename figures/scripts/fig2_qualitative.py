#!/usr/bin/env python3
"""
Fig.2 —— 定性对比 + ROI 局部放大（协议 §C Fig.2）。

产出两张图
  {stem}.pdf/.png        主网格：行=方法，列=场景；整图上画 ROI 框
  {stem}_roi.pdf/.png    ROI 放大网格：行=方法，列=场景 × {flat, detail}

冻结规则
  - 行序：LR(bicubic↑4) → 各基线（PSNR 升序）→ 本方法 → HR          §C Fig.2 第 2 条
  - ROI 坐标取自 roi.json，所有方法/场景同一坐标                     §C Fig.2 第 3 条
  - 每格右下角标 PSNR/SSIM，数字来自 per_sample_metrics.json         §C Fig.2 第 4 条
  - LR 必须 bicubic 上采样，禁止最近邻/锐化                          §E.5
  - 所有面板共用同一显示范围（全局分位），禁止逐格自适应             §C Fig.2 陷阱

用法
  python fig2_qualitative.py \
      --metrics figures/data/per_sample_metrics_all.json \
      --roi data/splits/roi.json \
      --hr-dir $HOLO_DATA_ROOT/HR --lr-dir $HOLO_DATA_ROOT/LR4 \
      --exp-root experiments \
      --out-stem figures/fig2_qualitative
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
    setup_matplotlib, style_map, to_display_gray, warn,
)
from fighelpers import (  # noqa: E402
    annotate_metric, crop_box, draw_roi_box, global_display_range, load_roi, ordered_methods,
    panel_grid, read_hr, read_lr, read_sr, roi_box, tight_panel, upsample_lr,
)

# ROI 两类各有固定颜色（整图框与放大块用同色，协议 §C Fig.2 第 3 条）
ROI_KINDS = (('flat', 'ROI-A (flat: speckle suppression)', '#D55E00'),
             ('detail', 'ROI-B (detail: resolution recovery)', '#0072B2'))


# --------------------------------------------------------------------------
def get_metrics(metrics: Dict, model: str, seed: int, number: str,
                variant: str = 'plain') -> Optional[Tuple[float, float]]:
    per = (metrics.get('results', {}).get(model) or {})
    per = per.get(str(seed)) or per.get(seed) or {}
    rec = per.get(str(number)) or per.get(number)
    if not rec:
        return None
    sfx = '_plus' if variant == 'plus' else ''
    p, s = rec.get('psnr' + sfx), rec.get('ssim' + sfx)
    if p is None or s is None:
        return None
    return float(p), float(s)


def build_stack(args, methods: Sequence[Dict]) -> Tuple[List[Dict], List[str]]:
    """返回 (rows, scenes)。每行 = {'display', 'images': {number: gray}}。"""
    roi = load_roi(args.roi)
    scenes = [str(s) for s in roi.get('fig2_samples', [])]
    if not scenes:
        raise ValueError(f'{args.roi} 里没有 fig2_samples')
    info(f'场景（来自 roi.json）: {scenes}')

    rows: List[Dict] = []
    grids: Dict[str, np.ndarray] = {}

    # LR 行：bicubic 上采样，且**不做任何后处理**
    lr_imgs = {}
    for n in scenes:
        lr_imgs[n] = to_display_gray(upsample_lr(read_lr(args.lr_dir, n)))
    rows.append({'display': 'LR (bicubic$\\times$4)', 'images': lr_imgs,
                 'kind': 'lr'})

    for m in methods:
        imgs = {}
        for n in scenes:
            try:
                sr = read_sr(args.exp_root, str(m['model']), args.seed, n,
                             str(m['variant']))
            except FileNotFoundError as e:
                warn(str(e))
                continue
            imgs[n] = to_display_gray(sr)
        if imgs:
            rows.append({'display': str(m['display']), 'images': imgs,
                         'kind': 'method', 'meta': m})

    hr_imgs = {n: to_display_gray(read_hr(args.hr_dir, n)) for n in scenes}
    rows.append({'display': 'HR (GT)', 'images': hr_imgs, 'kind': 'hr'})

    # 全局显示范围（所有行、所有场景一起算），协议 §C Fig.2 陷阱
    all_imgs = [im for r in rows for im in r['images'].values()]
    vmin, vmax = global_display_range(all_imgs)
    info(f'全局显示范围: [{vmin:.4f}, {vmax:.4f}]（所有面板共用）')
    return rows, scenes, roi, (vmin, vmax)


# --------------------------------------------------------------------------
def draw_grid(rows, scenes, roi, drange, args, out_stem: Path, zoom: bool):
    plt = setup_matplotlib()
    n_rows = len(rows)
    if zoom:
        n_cols = len(scenes) * len(ROI_KINDS)
        cell = args.zoom_cell
    else:
        n_cols = len(scenes)
        cell = args.cell

    fig, axes = panel_grid(n_rows, n_cols, width=args.width, cell_in=cell,
                           row_label_extra=0.0)
    vmin, vmax = drange

    for ri, row in enumerate(rows):
        for ci in range(n_cols):
            ax = axes[ri][ci]
            tight_panel(ax)
            if not zoom:
                n = scenes[ci]
                img = row['images'].get(n)
                if img is None:
                    ax.axis('off')
                    continue
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax,
                          interpolation='nearest', aspect='equal')
                if row['kind'] == 'method':
                    for kind, _lbl, color in ROI_KINDS:
                        box = roi_box(roi, n, kind)
                        if box:
                            draw_roi_box(ax, box, color,
                                         label=('A' if kind == 'flat' else 'B'))
                if row['kind'] == 'method':
                    mt = get_metrics(args.metrics_data, str(row['meta']['model']),
                                     args.seed, n, str(row['meta']['variant']))
                    if mt:
                        annotate(ax, mt)
                elif row['kind'] == 'lr':
                    annotate_metric(ax, 'input', loc='br',
                                    fontsize=FIG_FONT_SIZE - 2)
            else:
                scene_i, kind_i = divmod(ci, len(ROI_KINDS))
                n = scenes[scene_i]
                kind, _lbl, color = ROI_KINDS[kind_i]
                img = row['images'].get(n)
                box = roi_box(roi, n, kind)
                if img is None or not box:
                    ax.axis('off')
                    continue
                patch = crop_box(img, box)
                ax.imshow(patch, cmap='gray', vmin=vmin, vmax=vmax,
                          interpolation='nearest', aspect='equal')
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_color(color)
                    s.set_linewidth(1.1)
                if row['kind'] == 'method':
                    mt = get_metrics(args.metrics_data, str(row['meta']['model']),
                                     args.seed, n, str(row['meta']['variant']))
                    if mt:
                        annotate(ax, mt, fontsize=FIG_FONT_SIZE - 2)

    # 行标签（左侧）
    for ri, row in enumerate(rows):
        axes[ri][0].set_ylabel(row['display'], rotation=0, ha='right',
                               va='center', labelpad=8,
                               fontsize=FIG_FONT_SIZE)
    # 列标题
    if zoom:
        for ci in range(n_cols):
            scene_i, kind_i = divmod(ci, len(ROI_KINDS))
            _k, lbl, color = ROI_KINDS[kind_i]
            axes[0][ci].set_title(f"{scenes[scene_i]}\n{lbl.split(':')[0]}",
                                  fontsize=FIG_FONT_SIZE - 2, color=color, pad=2)
    else:
        for ci, n in enumerate(scenes):
            axes[0][ci].set_title(f'sample {n}', fontsize=FIG_FONT_SIZE - 1, pad=2)

    suffix = '_roi' if zoom else ''
    paths = save_paper_figure(fig, Path(str(out_stem) + suffix))
    plt.close(fig)
    for p in paths:
        info(f'已写出 {p}')


def annotate(ax, mt: Tuple[float, float], note: str = '',
             fontsize: Optional[float] = None):
    """指标标注：PSNR(2 位) / SSIM(4 位)，从 per_sample_metrics 读取。"""
    txt = f'{mt[0]:.2f} / {mt[1]:.4f}'
    if note:
        txt = f'{note}: {txt}'
    from fighelpers import annotate_metric
    annotate_metric(ax, txt, loc='br', fontsize=fontsize)


# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description='Fig.2 定性对比 + ROI 放大')
    ap.add_argument('--metrics', required=True, help='per_sample_metrics_all.json')
    ap.add_argument('--roi', required=True, help='roi.json（决定场景与 ROI 坐标）')
    ap.add_argument('--hr-dir', required=True)
    ap.add_argument('--lr-dir', required=True)
    ap.add_argument('--exp-root', required=True)
    ap.add_argument('--seed', type=int, default=FIG_SEED,
                    help=f'协议 §13.B.2 第 2 条：正文图统一取 seed 10')
    ap.add_argument('--ours-pattern', default='HoloRestormer')
    ap.add_argument('--with-plus', action='store_true', help='同时画自集成行')
    ap.add_argument('--width', choices=['single', 'double'], default='double')
    ap.add_argument('--cell', type=float, default=1.05, help='主网格单格边长 (inch)')
    ap.add_argument('--zoom-cell', type=float, default=0.95)
    ap.add_argument('--out-stem', default='figures/fig2_qualitative')
    ap.add_argument('--no-zoom', action='store_true', help='只出主网格')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.metrics_data = load_json(args.metrics)

    variants = ('plain', 'plus') if args.with_plus else ('plain',)
    methods = ordered_methods(args.metrics, args.ours_pattern, variants,
                              exp_root=args.exp_root, seed=args.seed)
    if not methods:
        print('错误: 没有可用的方法（检查 --metrics / --exp-root）', file=sys.stderr)
        return 2
    info('行序（PSNR 升序 → 本方法）: ' + ', '.join(str(m['display'])
                                                   for m in methods))

    rows, scenes, roi, drange = build_stack(args, methods)
    draw_grid(rows, scenes, roi, drange, args, Path(args.out_stem), zoom=False)
    if not args.no_zoom:
        draw_grid(rows, scenes, roi, drange, args, Path(args.out_stem), zoom=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
