#!/usr/bin/env python3
"""
Fig.4 —— 频谱对比 + 物理截止线（协议 §C Fig.4）★ **核心证据图**

为什么必须有
  本方法的主张是"恢复了带外高频"。`r̂ > 1` 区域的曲线能否抬起来、抬多少，
  是唯一能把该主张**量化且可视化**的东西。没有它，"频带"就只是叙述。

产出两张图（协议要求两种形式各出一张）
  {stem}a_spec2d.pdf/.png   2D 频谱：log(1+|fftshift(FFT2)|)，共用 colorbar + r̂=1 圆
  {stem}b_radial.pdf/.png   1D 径向平均功率谱：横轴 r̂ ∈ [0,2]，r̂=1 竖直虚线

冻结规则（协议 §C Fig.4）
  - 横轴**必须归一化**到 LR Nyquist：r̂ = r / f_LR，f_LR = 1/(2·scale) = 0.125 cyc/px
  - **必须画 r̂ = 1 的截止线**（2D 圆 / 1D 竖虚线）
  - 曲线：HR / LR / 各方法 / Ours 都要有（只给两条说不清相对基线好多少）
  - **用同一批样本的平均谱**，并在图注写明样本数与是否取平均

用法
  python fig4_spectrum.py \
      --roi data/splits/roi.json \
      --hr-dir $HOLO_DATA_ROOT/HR --lr-dir $HOLO_DATA_ROOT/LR4 \
      --exp-root experiments \
      --metrics figures/data/per_sample_metrics_all.json \
      --out-stem figures/fig4_spectrum
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIG_FONT_SIZE, FIG_SEED, LR_NYQUIST, OKABE_ITO, figure_size, info,
    load_json, save_paper_figure, setup_matplotlib, style_map, to_display_gray,
    warn,
)
from fighelpers import (  # noqa: E402
    add_cutoff_circle, add_cutoff_line, load_roi, log_magnitude_spectrum,
    ordered_methods, read_hr, read_lr, read_sr, radial_psd, tight_panel,
    upsample_lr,
)

N_BINS = 128
R_MAX = 2.0


def stack_spectra(args, methods, scenes) -> Dict[str, Dict[str, np.ndarray]]:
    """为每个条目收集所有场景的 (2D 对数幅度谱, 径向 PSD)。"""
    out: Dict[str, Dict[str, np.ndarray]] = {}

    def _add(name: str, imgs: List[np.ndarray]):
        if not imgs:
            return
        out[name] = {
            'spec2d': np.mean([log_magnitude_spectrum(g) for g in imgs], axis=0),
            'radial': np.mean([radial_psd(g, N_BINS, R_MAX)[1] for g in imgs],
                              axis=0),
        }

    lr_imgs = [to_display_gray(upsample_lr(read_lr(args.lr_dir, n)))
               for n in scenes]
    _add('LR (bicubic$\\times$4)', lr_imgs)

    for m in methods:
        imgs = []
        for n in scenes:
            try:
                imgs.append(to_display_gray(
                    read_sr(args.exp_root, str(m['model']), args.seed, n,
                            str(m['variant']))))
            except FileNotFoundError as e:
                warn(str(e))
        _add(str(m['display']), imgs)

    _add('HR (GT)', [to_display_gray(read_hr(args.hr_dir, n)) for n in scenes])
    return out


def _stem(out_stem: Path, tag: str) -> Path:
    s = str(out_stem)
    return Path(s + ('_' if not s.endswith('_') else '') + tag)


def draw_2d(data, order, out_stem: Path, args):
    plt = setup_matplotlib()
    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=figure_size('single', 1.9 * n),
                             squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.86, top=0.96, bottom=0.02,
                        hspace=0.10)

    vmax = max(float(np.percentile(d['spec2d'], 99.0)) for d in data.values())
    im = None
    for i, name in enumerate(order):
        ax = axes[i][0]
        im = ax.imshow(data[name]['spec2d'], cmap='viridis', vmin=0.0, vmax=vmax,
                       interpolation='nearest')
        tight_panel(ax)
        ax.set_title(name, fontsize=FIG_FONT_SIZE - 1, pad=1)
        add_cutoff_circle(ax, color='white', lw=1.0,
                          label=('$\\hat{r}=1$' if i == 0 else None))
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cbar.set_label(r'$\log(1+|\mathcal{F}|)$', fontsize=FIG_FONT_SIZE - 2)
        cbar.ax.tick_params(labelsize=FIG_FONT_SIZE - 3)

    for p in save_paper_figure(fig, _stem(out_stem, 'a_spec2d')):
        info(f'已写出 {p}')
    plt.close(fig)


def draw_radial(data, order, out_stem: Path, args):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=figure_size('single', 2.3))
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.16)

    styles = style_map(order)
    r = None
    for name in order:
        rad = np.asarray(data[name]['radial'], dtype=np.float64)
        floor = np.nanmax(rad) * 1e-8 + 1e-30
        rad = np.where(np.isfinite(rad) & (rad > 0), rad, floor)
        x = np.linspace(0, R_MAX, rad.size)
        r = x
        st = styles[name]
        lw = 1.6 if ('HR' in name or name.startswith(args.ours_pattern)) else 1.0
        ax.semilogy(x, rad, label=name, linewidth=lw,
                    markevery=max(1, rad.size // 10), **st)

    add_cutoff_line(ax, y_text=0.04)
    ax.set_xlim(0, R_MAX)
    ax.set_xlabel(r'normalized radial frequency  $\hat{r} = r / f_{\mathrm{LR}}$')
    ax.set_ylabel('radial average PSD')
    ax.legend(loc='upper right', fontsize=FIG_FONT_SIZE - 2, ncol=1,
              handlelength=1.8)
    ax.grid(True, which='major', linewidth=0.3, alpha=0.35)

    for p in save_paper_figure(fig, _stem(out_stem, 'b_radial')):
        info(f'已写出 {p}')
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description='Fig.4 频谱对比（核心证据图）')
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--roi', required=True)
    ap.add_argument('--hr-dir', required=True)
    ap.add_argument('--lr-dir', required=True)
    ap.add_argument('--exp-root', required=True)
    ap.add_argument('--seed', type=int, default=FIG_SEED)
    ap.add_argument('--ours-pattern', default='HoloRestormer')
    ap.add_argument('--with-plus', action='store_true')
    ap.add_argument('--scenes', nargs='*', default=None,
                    help='参与平均的场景；默认用 roi.json 的 fig2_samples')
    ap.add_argument('--out-stem', default='figures/fig4_spectrum')
    args = ap.parse_args()

    roi = load_roi(args.roi)
    scenes = [str(s) for s in (args.scenes or roi.get('fig2_samples', []))]
    if not scenes:
        print('错误: 没有场景', file=sys.stderr)
        return 2
    info(f'参与平均的场景 ({len(scenes)} 张): {scenes}')

    variants = ('plain', 'plus') if args.with_plus else ('plain',)
    methods = ordered_methods(args.metrics, args.ours_pattern, variants,
                              exp_root=args.exp_root, seed=args.seed)
    if not methods:
        print('错误: 没有可用的方法', file=sys.stderr)
        return 2

    data = stack_spectra(args, methods, scenes)
    order = [k for k in (['LR (bicubic$\\times$4)'] +
                         [str(m['display']) for m in methods] + ['HR (GT)'])
             if k in data]
    info(f'曲线顺序: {order}')

    draw_2d(data, order, Path(args.out_stem), args)
    draw_radial(data, order, Path(args.out_stem), args)

    info(f'图注必须写明: 频谱定义、样本数={len(scenes)}、是否取平均（是）、'
         f'横轴归一化基准 f_LR={LR_NYQUIST:.4f} cyc/px = 1/(2·scale)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
