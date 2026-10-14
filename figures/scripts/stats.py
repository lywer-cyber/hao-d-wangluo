#!/usr/bin/env python3
"""
显著性检验与胜负统计 —— Tab.1（协议 §J）。

为什么必须有
  只报 mean ± std 时，0.2 dB 的差距会被质疑"是否只是种子波动"。
  本脚本按冻结方案做配对检验，并把"差异是否真实"变成一个可报告的数。

冻结方案（协议 §J.2，勿改）
  检验      Wilcoxon signed-rank，双侧
  配对单位  逐张测试图（同一张图的 Ours vs 基线）
  配对数量  全部测试图（禁止抽子集）
  种子处理  每个种子单独做一次检验，**3 个 p 值全部报告**
  多重比较  Holm-Bonferroni 校正（跨所有基线）
  效应量    median ΔPSNR + 95% bootstrap CI（10000 次重采样，固定随机种子）
  打平定义  |ΔPSNR| < 0.01 dB

用法
  python stats.py \
      --metrics figures/data/per_sample_metrics_all.json \
      --ours HoloRestormer \
      --baselines SwinIR RCAN NAFNet DAT HAT SRResNet \
      --out-stem figures/data/stats_table \
      --fig figures/fig_tab1_delta_distribution.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    OKABE_ITO, SEEDS, dump_json_strict, figure_size, info, load_json,
    save_paper_figure, setup_matplotlib, style_map, warn,
)

TIE_DB = 0.01               # 打平阈值（dB），协议 §J.4
N_BOOT = 10000              # bootstrap 次数
BOOT_SEED = 20261012        # 固定随机种子，必须写进图注


# --------------------------------------------------------------------------
# 取配对数据
# --------------------------------------------------------------------------
def paired_psnr(metrics: Dict, model: str, seed: int, variant: str
                ) -> Dict[str, float]:
    """取某模型某个种子的逐图 PSNR。variant: '' 或 '_plus'。"""
    by_seed = metrics.get('results', {}).get(model)
    if not by_seed:
        raise KeyError(f'找不到模型 {model}')
    per_sample = by_seed.get(str(seed)) or by_seed.get(seed)
    if not per_sample:
        raise KeyError(f'{model} 在 seed {seed} 下无数据')
    out = {}
    for n, rec in per_sample.items():
        key = 'psnr' + variant
        v = rec.get(key, rec.get('psnr') if variant == '' else None)
        if v is not None and np.isfinite(float(v)):
            out[str(n)] = float(v)
    return out


def align(ours: Dict[str, float], base: Dict[str, float]
          ) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """按样本号取交集并**按数值序**排列（保证配对顺序确定）。"""
    common = sorted(set(ours) & set(base),
                    key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    if not common:
        raise ValueError('两个模型没有共同样本')
    a = np.array([ours[s] for s in common], dtype=np.float64)
    b = np.array([base[s] for s in common], dtype=np.float64)
    return common, a, b


# --------------------------------------------------------------------------
# 检验与效应量
# --------------------------------------------------------------------------
def wilcoxon_p(ours: np.ndarray, base: np.ndarray) -> Tuple[float, str]:
    """双侧 Wilcoxon signed-rank。返回 (p, 说明)。"""
    from scipy.stats import wilcoxon
    d = ours - base
    if np.allclose(d, 0.0):
        return 1.0, 'all-zero'
    if np.all(d == 0):
        return 1.0, 'all-zero'
    try:
        res = wilcoxon(ours, base, zero_method='wilcox', alternative='two-sided')
        p = float(res.pvalue)
        if not np.isfinite(p):
            return 1.0, 'nan'
        return p, 'ok'
    except ValueError as e:
        # 例如全为 0 或样本退化；按最保守处理
        return 1.0, f'exception:{e}'


def bootstrap_median_ci(d: np.ndarray, n_boot: int = N_BOOT,
                        seed: int = BOOT_SEED, alpha: float = 0.05
                        ) -> Tuple[float, float]:
    """median(d) 的 95% 百分位 bootstrap 区间。"""
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.size
    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(d[idx], axis=1)
    lo = float(np.percentile(meds, 100 * alpha / 2))
    hi = float(np.percentile(meds, 100 * (1 - alpha / 2)))
    return lo, hi


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni：返回校正后 p（保持原顺序），单调不减且截断在 1。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def win_loss_tie(d: np.ndarray, tie: float = TIE_DB
                 ) -> Tuple[int, int, int]:
    wins = int(np.sum(d >= tie))
    losses = int(np.sum(d <= -tie))
    ties = int(d.size - wins - losses)
    return wins, losses, ties


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return 'n/a'
    return '< 0.001' if p < 0.001 else f'{p:.3g}'


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def compute_all(metrics: Dict, ours: str, baselines: Sequence[str],
                seeds: Sequence[int], variant: str
                ) -> Tuple[List[Dict], Dict[int, Dict], int]:
    records: List[Dict] = []
    n_images = 0

    for seed in seeds:
        try:
            ours_ps = paired_psnr(metrics, ours, seed, variant)
        except KeyError as e:
            warn(str(e))
            continue

        # 先算原始 p，再统一做 Holm
        seed_rows: List[Dict] = []
        for b in baselines:
            if b == ours:
                continue
            try:
                base_ps = paired_psnr(metrics, b, seed, '')
            except KeyError as e:
                warn(str(e))
                continue
            common, a, bb = align(ours_ps, base_ps)
            n_images = max(n_images, len(common))
            d = a - bb
            p, note = wilcoxon_p(a, bb)
            lo, hi = bootstrap_median_ci(d)
            wins, losses, ties = win_loss_tie(d)
            seed_rows.append({
                'ours': ours,
                'ours_variant': variant or 'plain',
                'baseline': b,
                'seed': int(seed),
                'n_pairs': len(common),
                'p_raw': p,
                'median_delta_psnr': float(np.median(d)),
                'ci_low': lo,
                'ci_high': hi,
                'mean_delta_psnr': float(np.mean(d)),
                'wins': wins, 'losses': losses, 'ties': ties,
                'note': note,
            })

        # Holm 校正（跨基线，本种子内）
        ps = [r['p_raw'] for r in seed_rows]
        for r, pa in zip(seed_rows, holm(ps)):
            r['p_holm'] = pa
        records.extend(seed_rows)

    # 哪个基线最强（按 seed 10 的均值 PSNR）
    best_baseline, best_score = None, -1e18
    for b in baselines:
        if b == ours:
            continue
        try:
            v = np.mean(list(paired_psnr(metrics, b, seeds[0], '').values()))
        except (KeyError, ValueError):
            continue
        if v > best_score:
            best_baseline, best_score = b, float(v)

    info(f'最佳基线 = {best_baseline}（seed {seeds[0]} 平均 PSNR {best_score:.4f}）')
    return records, {'best_baseline': best_baseline,
                     'best_baseline_psnr': best_score}, n_images


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def write_long_csv(records: Sequence[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['ours', 'ours_variant', 'baseline', 'seed', 'n_pairs', 'p_raw', 'p_holm',
            'median_delta_psnr', 'ci_low', 'ci_high', 'mean_delta_psnr',
            'wins', 'losses', 'ties', 'note']
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in cols})


def pivot(records: Sequence[Dict], seeds: Sequence[int]) -> List[Dict]:
    """宽表：一行一个基线，列为各种子 p、medianΔ、CI、是否 Holm。"""
    by_base: Dict[str, Dict] = {}
    for r in records:
        d = by_base.setdefault(r['baseline'], {'baseline': r['baseline']})
        d[f'p_seed{r["seed"]}'] = r['p_holm']
        if r['seed'] == seeds[0]:
            d['median_delta'] = r['median_delta_psnr']
            d['ci_low'] = r['ci_low']
            d['ci_high'] = r['ci_high']
            d['wins'] = r['wins']
            d['losses'] = r['losses']
            d['ties'] = r['ties']
            d['n_pairs'] = r['n_pairs']
    rows = list(by_base.values())
    rows.sort(key=lambda d: -(d.get('median_delta') or -1e9))
    for d in rows:
        d['holm_applied'] = 'yes'
    return rows


def write_wide_csv(rows: Sequence[Dict], seeds: Sequence[int], path: Path):
    cols = ['baseline'] + [f'p_seed{s}' for s in seeds] + \
           ['median_delta', 'ci_low', 'ci_high', 'wins', 'losses', 'ties',
            'n_pairs', 'holm_applied']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})


def write_wide_md(rows: Sequence[Dict], seeds: Sequence[int], path: Path,
                  ours: str):
    head = ['Comparison'] + [f'seed {s}' for s in seeds] + \
           ['median ΔPSNR', '95% CI', 'W/L/T', 'Holm']
    lines = ['| ' + ' | '.join(head) + ' |',
             '|' + '|'.join(['---'] * len(head)) + '|']
    for r in rows:
        pcell = [fmt_p(r.get(f'p_seed{s}')) for s in seeds]
        ci = f"[{r.get('ci_low', float('nan')):+.2f}, {r.get('ci_high', float('nan')):+.2f}]"
        wlt = f"{r.get('wins','-')}/{r.get('losses','-')}/{r.get('ties','-')}"
        lines.append('| ' + ' | '.join(
            [f"{ours} vs {r['baseline']}"] + pcell +
            [f"{r.get('median_delta', float('nan')):+.2f}", ci, wlt, 'yes']) + ' |')
    lines += ['', (f'*Two-sided Wilcoxon signed-rank test over paired test images; '
                   f'Holm-Bonferroni corrected across baselines. '
                   f'95% CI of the median ΔPSNR from {N_BOOT} bootstrap resamples '
                   f'(fixed seed {BOOT_SEED}). '
                   f'Win/loss/tie defined as ΔPSNR ≥ {TIE_DB} dB / ≤ -{TIE_DB} dB / otherwise.*'),
              '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_wide_tex(rows: Sequence[Dict], seeds: Sequence[int], path: Path,
                   ours: str, label: str):
    # 列 = Comparison + 每种子一列 + median dPSNR + 95% CI + W/L/T
    ncol = 1 + len(seeds) + 3
    lines = [
        '% 需要宏包: booktabs',
        '% 由 figures/scripts/stats.py 自动生成，请勿手工编辑',
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Statistical significance of the improvement over each baseline. '
        r'Two-sided Wilcoxon signed-rank test on paired per-image PSNR over the full '
        rf'test set; $p$-values are Holm--Bonferroni corrected across baselines. '
        rf'95\% CI of the median $\Delta$PSNR obtained from {N_BOOT} bootstrap '
        rf'resamples (fixed seed {BOOT_SEED}). W/L/T: images where '
        rf'$\Delta$PSNR $\geq$ {TIE_DB} dB / $\leq$ $-${TIE_DB} dB / otherwise.}}',
        f'\\label{{{label}}}',
        r'\begin{tabular}{l' + 'c' * (ncol - 1) + '}',
        r'\toprule',
        ' & '.join(['Comparison'] + [rf'\textbf{{seed {s}}}' for s in seeds] +
                   [r'\textbf{median $\Delta$PSNR}', r'\textbf{95\% CI}',
                    r'\textbf{W/L/T}']) + r' \\',
        r'\midrule',
    ]
    for r in rows:
        cells = [f"{ours} vs {r['baseline']}"]
        cells += [fmt_p(r.get(f'p_seed{s}')) for s in seeds]
        cells.append(f"{r.get('median_delta', float('nan')):+.2f}")
        cells.append(f"[{r.get('ci_low', float('nan')):+.2f}, "
                     f"{r.get('ci_high', float('nan')):+.2f}]")
        cells.append(f"{r.get('wins','-')}/{r.get('losses','-')}/{r.get('ties','-')}")
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_wide_typ(rows: Sequence[Dict], seeds: Sequence[int], path: Path,
                   ours: str, label: str):
    cs = 'auto, ' + ', '.join(['center'] * (1 + len(seeds) + 3))
    lines = ['// 由 figures/scripts/stats.py 自动生成，请勿手工编辑',
             '#figure(',
             '  table(',
             f'    columns: ({cs}),',
             '    stroke: none,',
             '    table.header([' +
             '], ['.join([f'Comparison'] + [f'seed {s}' for s in seeds] +
                        ['median ΔPSNR', '95% CI', 'W/L/T']) + ']),']
    for r in rows:
        cells = [f"{ours} vs {r['baseline']}"] + \
                [fmt_p(r.get(f'p_seed{s}')) for s in seeds] + [
                    f"{r.get('median_delta', float('nan')):+.2f}",
                    f"[{r.get('ci_low', float('nan')):+.2f}, "
                    f"{r.get('ci_high', float('nan')):+.2f}]",
                    f"{r.get('wins','-')}/{r.get('losses','-')}/{r.get('ties','-')}"]
        lines.append('    ' + ', '.join(cells) + ',')
    lines += ['  ),',
              f'  caption: [Statistical significance vs each baseline. Two-sided Wilcoxon '
              f'signed-rank on paired per-image PSNR, Holm--Bonferroni corrected. '
              f'95% CI of median ΔPSNR from {N_BOOT} bootstrap resamples '
              f'(seed {BOOT_SEED}).],',
              f'  <{label}>,',
              ')',
              '']
    path.write_text('\n'.join(lines), encoding='utf-8')


# --------------------------------------------------------------------------
# 分布图（协议 §J.4）
# --------------------------------------------------------------------------
def plot_distribution(records: Sequence[Dict], ours: str, seed: int, out: Path):
    plt = setup_matplotlib()
    rows = [r for r in records if r['seed'] == seed]
    if not rows:
        warn('没有可用于画分布图的数据')
        return
    rows.sort(key=lambda r: -r['median_delta_psnr'])

    # 需要真实差值分布，这里从汇总统计重建是不可行的 —— 由调用方传入更为妥当。
    # 为保持脚本自洽，本函数仅画 median + CI（森林图样式），差值的完整分布
    # 由 --dump-deltas 导出的 csv 供外部绘制。
    fig, ax = plt.subplots(figsize=figure_size('single', 1.6 + 0.28 * len(rows)))
    ys = np.arange(len(rows))
    med = np.array([r['median_delta_psnr'] for r in rows])
    lo = np.array([r['ci_low'] for r in rows])
    hi = np.array([r['ci_high'] for r in rows])
    ax.errorbar(med, ys, xerr=[med - lo, hi - med], fmt='o', capsize=2,
                color=OKABE_ITO[1], ecolor=OKABE_ITO[0], elinewidth=0.9,
                markersize=4)
    ax.axvline(0.0, color='0.4', linewidth=0.8, linestyle='--')
    ax.set_yticks(ys)
    ax.set_yticklabels([r['baseline'] for r in rows])
    ax.set_xlabel(r'median $\Delta$PSNR (dB), ' + f'{ours} $-$ baseline')
    ax.invert_yaxis()
    save_paper_figure(fig, out)
    plt.close(fig)
    info(f'已写出 {out}（森林图，median + 95% CI）')


def dump_deltas(metrics: Dict, ours: str, baselines: Sequence[str],
                seeds: Sequence[int], variant: str, path: Path):
    """导出逐图 ΔPSNR，供画小提琴/箱线或审稿人复核。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['baseline', 'seed', 'number', 'psnr_ours', 'psnr_base', 'delta'])
        for seed in seeds:
            try:
                ours_ps = paired_psnr(metrics, ours, seed, variant)
            except KeyError:
                continue
            for b in baselines:
                if b == ours:
                    continue
                try:
                    base_ps = paired_psnr(metrics, b, seed, '')
                except KeyError:
                    continue
                common, a, bb = align(ours_ps, base_ps)
                for n, av, bv in zip(common, a, bb):
                    w.writerow([b, seed, n, f'{av:.6f}', f'{bv:.6f}',
                                f'{av - bv:+.6f}'])
    info(f'已写出逐图 ΔPSNR: {path}')


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description='显著性检验与胜负统计（协议 §J）')
    ap.add_argument('--metrics', required=True, help='per_sample_metrics_all.json')
    ap.add_argument('--ours', required=True)
    ap.add_argument('--baselines', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    ap.add_argument('--ours-variant', choices=['plain', 'plus'], default='plain',
                    help='用单体还是自集成结果参与检验')
    ap.add_argument('--out-stem', default='figures/data/stats_table')
    ap.add_argument('--p-json', default='figures/data/p_vs_best_baseline.json')
    ap.add_argument('--fig', default='figures/fig_tab1_delta_ci.pdf')
    ap.add_argument('--dump-deltas', default='figures/data/delta_psnr.csv')
    ap.add_argument('--no-fig', action='store_true')
    ap.add_argument('--label', default='tab:significance')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    metrics = load_json(args.metrics)
    variant = '_plus' if args.ours_variant == 'plus' else ''

    records, meta, n_images = compute_all(
        metrics, args.ours, args.baselines, args.seeds, variant)
    if not records:
        print('错误: 没有算出任何检验结果，检查模型名与数据', file=sys.stderr)
        return 2

    stem = Path(args.out_stem)
    write_long_csv(records, stem.with_suffix('.csv'))
    rows = pivot(records, args.seeds)
    write_wide_csv(rows, args.seeds, stem.parent / (stem.stem + '_wide.csv'))
    write_wide_md(rows, args.seeds, stem.with_suffix('.md'), args.ours)
    write_wide_tex(rows, args.seeds, stem.with_suffix('.tex'), args.ours, args.label)
    write_wide_typ(rows, args.seeds, stem.with_suffix('.typ'), args.ours, args.label)

    dump_deltas(metrics, args.ours, args.baselines, args.seeds, variant,
                Path(args.dump_deltas))

    # 主表要用的 p（vs 最佳基线，Holm 校正后，逐种子）
    p_json = {
        'ours': args.ours,
        'ours_variant': args.ours_variant,
        'best_baseline': meta['best_baseline'],
        'best_baseline_psnr': meta['best_baseline_psnr'],
        'p_seed': int(args.seeds[0]),
        'n_pairs': n_images,
        'n_boot': N_BOOT,
        'boot_seed': BOOT_SEED,
        'tie_db': TIE_DB,
        'p_by_seed': {},
        'p_holm_by_seed': {},
        'median_delta_by_seed': {},
    }
    for r in records:
        if r['baseline'] != meta['best_baseline']:
            continue
        s = str(r['seed'])
        p_json['p_by_seed'][s] = r['p_raw']
        p_json['p_holm_by_seed'][s] = r['p_holm']
        p_json['median_delta_by_seed'][s] = r['median_delta_psnr']
    dump_json_strict(p_json, args.p_json)

    if not args.no_fig:
        plot_distribution(records, args.ours, int(args.seeds[0]), Path(args.fig))

    info(f'已写出 {stem}.csv / _wide.csv / .md / .tex / .typ 与 {args.p_json}')
    for r in records:
        info(f"  vs {r['baseline']:12s} seed{r['seed']}: "
             f"p_raw={fmt_p(r['p_raw'])} p_holm={fmt_p(r['p_holm'])} "
             f"medianΔ={r['median_delta_psnr']:+.2f} "
             f"CI=[{r['ci_low']:+.2f},{r['ci_high']:+.2f}] "
             f"W/L/T={r['wins']}/{r['losses']}/{r['ties']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
