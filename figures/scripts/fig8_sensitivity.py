#!/usr/bin/env python3
"""
Fig.8 —— 退化参数敏感性（协议 §C Fig.8）。

固定其余退化参数，**单参数扫描**：
  speckle_contrast（本任务最相关）、diffraction_cutoff、poisson_gain、read_std

⚠️ 协议要求：这是**额外生成的分析集**，不用于选 checkpoint，与主实验相互独立。
   扫描用的退化必须另外生成并冻结，所有模型用同一批，**不得复用训练集或测试集**。

两种用法
  # 1) 从扫描结果目录算指标（推荐，自洽）
  python fig8_sensitivity.py build \
      --sweep-root sweeps/speckle_contrast \
      --hr-dir $HOLO_DATA_ROOT/HR_analysis \
      --models HoloRestormer SwinIR RCAN --seeds 10 20 30 \
      --out figures/data/sensitivity_speckle.json

  # 2) 直接画（用已经算好的 contract）
  python fig8_sensitivity.py plot \
      --contract figures/data/sensitivity_speckle.json \
      --out-stem figures/fig8_sensitivity

contract 结构（build 子命令会生成）
  {"protocol_version","sweep_param","sweep_values","fixed_params",
   "reference_set","n_images","results": {model: {value_str: {"psnr","ssim"}}}}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIG_FONT_SIZE, FIG_SEED, PROTOCOL_VERSION, SEEDS, compute_metrics,
    dump_json_strict, figure_size, imread_rgb01, info, load_json,
    save_paper_figure, setup_matplotlib, style_map, warn,
)
from fighelpers import find_image  # noqa: E402

PARAM_LABELS = {
    'speckle_contrast': 'speckle contrast  $\\sigma_I/\\langle I\\rangle$',
    'diffraction_cutoff': 'diffraction cutoff (normalized to LR Nyquist)',
    'poisson_gain': 'Poisson gain (photons at full scale)',
    'read_std': 'read noise std (normalized)',
}


# --------------------------------------------------------------------------
# build：从扫描目录算指标
# --------------------------------------------------------------------------
def discover_values(sweep_root: Path) -> List[str]:
    vals = []
    for d in sorted(sweep_root.iterdir()):
        if d.is_dir() and '=' in d.name:
            vals.append(d.name.split('=', 1)[1])

    def _key(x):
        try:                       # 能转成数就按数值排序，x 轴才单调
            return (0, float(x), '')
        except (TypeError, ValueError):
            return (1, 0.0, str(x))

    return sorted(set(vals), key=_key)


def _norm_key(v) -> str:
    """sweep_values 数列会被转成 float（20.0），而 results 的键是目录名（'20'）。"""
    try:
        f = float(v)
        return str(int(f)) if float(f).is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v)


def _lookup(d, v):
    """按 原样 / 归一化 两种键去取，避免整数取值点被静默丢弃。"""
    for k in (str(v), _norm_key(v)):
        if k in d:
            return d[k]
    return None


def build(args) -> int:
    sweep_root = Path(args.sweep_root)
    if not sweep_root.exists():
        print(f'错误: 扫描目录不存在 {sweep_root}', file=sys.stderr)
        return 2
    values = discover_values(sweep_root)
    if not values:
        print(f'错误: {sweep_root} 下没有 "param=value" 形式的子目录', file=sys.stderr)
        return 2
    param = args.param or sweep_root.name
    info(f'参数 {param}，取值 {values}')

    hr_dir = Path(args.hr_dir)
    numbers = [p.stem for p in sorted(hr_dir.iterdir())
               if p.suffix.lower() in ('.bmp', '.png', '.tif', '.tiff')]
    if args.limit:
        numbers = numbers[:args.limit]
    if not numbers:
        print(f'错误: {hr_dir} 下没有 HR 图', file=sys.stderr)
        return 2
    info(f'分析集 {len(numbers)} 张')

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in args.models:
        results[model] = {}
        for v in values:
            psnr_list, ssim_list = [], []
            for seed in args.seeds:
                # 优先 {model}/seed{seed}/{param}={v}/test_sr/，退化到 {model}/{param}={v}/
                cands = [
                    sweep_root / f'{param}={v}' / model / f'seed{seed}' / 'test_sr',
                    sweep_root / model / f'seed{seed}' / f'{param}={v}' / 'test_sr',
                    sweep_root / model / f'{param}={v}' / 'test_sr',
                    # 真机布局：evaluate_holo.py 写到 {model}/results/{split}/
                    sweep_root / f'{param}={v}' / model / 'results' / 'test',
                    sweep_root / model / f'seed{seed}' / f'{param}={v}' / 'results' / 'test',
                    sweep_root / model / f'{param}={v}' / 'results' / 'test',
                ]
                sr_dir = next((c for c in cands if c.exists()), None)
                if sr_dir is None:
                    continue
                for n in numbers:
                    sp = find_image(sr_dir, n, '_SR')
                    if sp is None:
                        continue
                    hr_p = find_image(hr_dir, n, '')
                    if hr_p is None:
                        continue
                    sr, hr = imread_rgb01(sp), imread_rgb01(hr_p)
                    if sr.shape != hr.shape:
                        warn(f'{model}/{v}/{n}: 尺寸不符 {sr.shape} vs {hr.shape}')
                        continue
                    m = compute_metrics(sr, hr, with_lpips=False)
                    psnr_list.append(m['psnr'])
                    ssim_list.append(m['ssim'])
            if psnr_list:
                results[model][v] = {
                    'psnr': float(np.mean(psnr_list)),
                    'ssim': float(np.mean(ssim_list)),
                    'n': len(psnr_list),
                }
            else:
                warn(f'{model} @ {param}={v}: 没有可用样本')

    if not any(results.values()):
        print('错误: 一个指标都没算出来，检查目录结构', file=sys.stderr)
        return 2

    out = {
        'protocol_version': PROTOCOL_VERSION,
        'sweep_param': param,
        'sweep_values': [float(v) if _isnum(v) else v for v in values],
        'fixed_params': {},
        'reference_set': ('independent analysis set; not used for checkpoint '
                          'selection; disjoint from train/val/test'),
        'n_images': len(numbers),
        'seeds': list(args.seeds),
        'results': results,
    }
    dump_json_strict(out, args.out)
    info(f'已写出 {args.out}')
    return 0


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------
def plot(args) -> int:
    c = load_json(args.contract)
    param = c.get('sweep_param', 'param')
    values = c.get('sweep_values', [])
    results = c.get('results', {})
    if not values or not results:
        print('错误: contract 缺 sweep_values 或 results', file=sys.stderr)
        return 2

    vr = [v for v in values if _isnum(v)]
    numeric = bool(vr) and len(vr) == len(values)
    x = np.asarray([float(v) for v in values]) if numeric else np.arange(len(values))

    models = sorted(results)
    styles = style_map(models)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=figure_size('single', 2.4))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.20)

    for m in models:
        ys, xs = [], []
        for i, v in enumerate(values):
            rec = _lookup(results[m], v)
            if rec and rec.get('psnr') is not None:
                xs.append(x[i])
                ys.append(float(rec['psnr']))
        if not ys:
            continue
        ax.plot(xs, ys, label=m, **styles[m])

    if numeric:
        ax.set_xlabel(PARAM_LABELS.get(param, param))
    else:
        ax.set_xlabel(PARAM_LABELS.get(param, param))
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in values], rotation=45, ha='right')
    ax.set_ylabel('PSNR (dB)')
    ax.legend(loc='best', fontsize=FIG_FONT_SIZE - 2)
    ax.grid(True, linewidth=0.3, alpha=0.30)

    for p in save_paper_figure(fig, Path(args.out_stem)):
        info(f'已写出 {p}')
    plt.close(fig)

    info(f"图注必须写明: 扫描参数={param}，范围={values}，"
         f"分析集 {c.get('n_images')} 张，"
         f"**该分析集独立于训练/测试集**，"
         f"每点为 {len(c.get('seeds', [])) or 1} 个种子的均值")
    return 0


# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description='Fig.8 退化参数敏感性')
    sub = ap.add_subparsers(dest='mode')

    b = sub.add_parser('build', help='从扫描目录算指标')
    b.add_argument('--sweep-root', required=True,
                   help='扫描根目录，内含 "param=value" 子目录')
    b.add_argument('--param', default=None, help='参数名；默认取 sweep-root 的目录名')
    b.add_argument('--hr-dir', required=True, help='分析集 HR 目录（独立于主实验）')
    b.add_argument('--models', nargs='+', required=True)
    b.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    b.add_argument('--limit', type=int, default=None, help='只用前 N 张（调试用）')
    b.add_argument('--out', default='figures/data/sensitivity.json')

    p = sub.add_parser('plot', help='按 contract 画图')
    p.add_argument('--contract', required=True)
    p.add_argument('--out-stem', default='figures/fig8_sensitivity')

    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == 'build':
        return build(args)
    if args.mode == 'plot':
        return plot(args)
    print('请指定子命令: build 或 plot（-h 看帮助）', file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
