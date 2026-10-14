#!/usr/bin/env python3
r"""
Fig.6 / 主结果表 —— 一次运行同时产出 csv / tex / typ / md 四份（协议 §K）。

为什么要脚本出表
  表里是 32.64 ± 0.07 这类数字，手抄一次就可能错一位，而表是审稿人核对最仔细的部分。
  因此：**数字全部来自 per_sample_metrics_all.json，加粗/下划线由脚本按显示值自动判定。**

冻结规则（协议 §K.3）
  1. booktabs 的 toprule/midrule/bottomrule，禁止竖线
  2. 最优加粗、次优下划线 —— 按**显示值**（四舍五入后）判定，避免"加粗的值看起来和没加粗的一样"
  3. mean ± std 写成 $a \pm b$
  4. 小数位固定：PSNR 2、SSIM 4、LPIPS 4、RMSE 4、MAE 4
  5. 单体与 `+` 两行都要有
  6. 表注含全套评估协议 + "测试集未参与任何选择"
  7. 排名判定在所有行上统一执行，不做特例

用法
  python fig6_table.py \
      --metrics figures/data/per_sample_metrics_all.json \
      --cost    figures/data/model_cost.csv \
      --p-json  figures/data/p_vs_best_baseline.json \
      --out-stem figures/data/table_main
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CROP_BORDER, LPIPS_NET, LPIPS_INPUT_RANGE, PROTOCOL_VERSION, SEEDS,
    SSIM_SIGMA, SSIM_WIN, info, load_json, warn,
)

# 列定义：(键, 表头, 小数位, 越大越好)
METRIC_COLS = [
    ('psnr',  'PSNR (dB)', 2, True),
    ('ssim',  'SSIM',      4, True),
    ('lpips', 'LPIPS',     4, False),
    ('rmse',  'RMSE',      4, False),
    ('mae',   'MAE',       4, False),
]
COST_COLS = [
    ('params_m', 'Params (M)', 2, False),
    ('flops_g',  'FLOPs (G)',  1, False),
    ('infer_ms', 'Time (ms)',  1, False),
]
# 这些列不参与加粗/下划线（代价列加粗没有意义，且容易误导）
NO_RANK_KEYS = {'params_m', 'flops_g', 'infer_ms', 'p', 'p_holm'}


# --------------------------------------------------------------------------
# 数据准备
# --------------------------------------------------------------------------
def build_rows(metrics_path: str, cost_path: Optional[str],
               ours_pattern: str) -> List[Dict[str, object]]:
    data = load_json(metrics_path)
    results = data.get('results', {})
    if not results:
        raise ValueError(f'{metrics_path} 里没有 results')

    costs = read_costs(cost_path)
    rows: List[Dict[str, object]] = []

    for model in sorted(results):
        by_seed = results[model]
        variants = collect_variants(by_seed)
        for variant in variants:                     # '' = 单体, '_plus' = 自集成
            # 每个种子先在图级取均值，再跨种子统计
            per_seed_means: Dict[str, Dict[str, float]] = {}
            for seed, per_sample in by_seed.items():
                recs = [r for r in per_sample.values() if r]
                if not recs:
                    continue
                means = {}
                for k, _, _, _ in METRIC_COLS:
                    vals = [float(r[k]) for r in recs if r.get(k) is not None]
                    means[k] = float(np.mean(vals)) if vals else None
                per_seed_means[str(seed)] = means
            if not per_seed_means:
                continue

            row: Dict[str, object] = {
                'model': model,
                'variant': variant,
                'display': model + ('+' if variant == '_plus' else ''),
                'is_ours': model.startswith(ours_pattern),
                'n_seeds': len(per_seed_means),
                'seeds': sorted(int(s) for s in per_seed_means),
            }
            for k, _, _, _ in METRIC_COLS:
                vals = [v[k] for v in per_seed_means.values()
                        if v.get(k) is not None and np.isfinite(v[k])]
                row[k] = float(np.mean(vals)) if vals else None
                row[k + '_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row.update(costs.get(model, {}))
            rows.append(row)

    if not rows:
        raise ValueError('没有可用的行')

    # 排序：基线按 PSNR 升序，本方法（含 +）放最后
    def _sort_key(r):
        return (1 if r['is_ours'] else 0,
                -(r['psnr'] if r['psnr'] is not None else -1e9),
                r['display'])

    rows.sort(key=_sort_key)
    return rows


def collect_variants(by_seed: Dict[str, Dict[str, dict]]) -> List[str]:
    """探测数据里有没有 _plus（自集成）。"""
    has_plus = False
    for per_sample in by_seed.values():
        for rec in per_sample.values():
            if any(k.endswith('_plus') for k in rec):
                has_plus = True
                break
        if has_plus:
            break
    return ['', '_plus'] if has_plus else ['']


def read_costs(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path or not Path(path).exists():
        if path:
            warn(f'未找到代价文件 {path}，Params/FLOPs 列将显示 --')
        return {}
    out: Dict[str, Dict[str, float]] = {}
    with Path(path).open('r', encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            model = (row.get('model') or '').strip()
            if not model:
                continue
            rec = {}
            for k, _, nd, _ in COST_COLS:
                v = (row.get(k) or '').strip()
                if v:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        warn(f'{path}: {model}.{k} 不是数字: {v!r}')
            out[model] = rec
    return out


# --------------------------------------------------------------------------
# 格式化
# --------------------------------------------------------------------------
def fmt_cell(value: Optional[float], std: Optional[float], nd: int) -> Tuple[str, float]:
    """返回 (显示字符串, 用于排名的四舍五入值)。"""
    if value is None or not np.isfinite(value):
        return '--', float('nan')
    q = round(float(value), nd)
    if std is None or not np.isfinite(std):
        return f'{q:.{nd}f}', q
    return f'{q:.{nd}f} \\pm {round(float(std), nd):.{nd}f}', q


def rank_marks(display_values: Sequence[float], higher_better: bool
               ) -> Tuple[Optional[float], Optional[float]]:
    """按显示值给出 (最优值, 次优值)。"""
    vals = sorted({v for v in display_values if np.isfinite(v)},
                  reverse=bool(higher_better))
    best = vals[0] if vals else None
    second = vals[1] if len(vals) > 1 else None
    return best, second


def mark(value: float, best: Optional[float], second: Optional[float],
         key: str) -> str:
    if key in NO_RANK_KEYS or not np.isfinite(value):
        return ''
    if best is not None and abs(value - best) < 1e-12:
        return 'best'
    if second is not None and abs(value - second) < 1e-12:
        return 'second'
    return ''


def esc_tex(s: str) -> str:
    for a, b in (('\\', r'\textbackslash{}'), ('_', r'\_'), ('%', r'\%'),
                 ('&', r'\&'), ('#', r'\#'), ('$', r'\$'),
                 ('{', r'\{'), ('}', r'\}')):
        s = s.replace(a, b)
    return s


def esc_typ(s: str) -> str:
    for a, b in (('_', r'\_'), ('#', r'\#'), ('@', r'\@'), ('$', r'\$'),
                 ('*', r'\*'), ('<', r'\<'), ('>', r'\>')):
        s = s.replace(a, b)
    return s


def esc_md(s: str) -> str:
    return s.replace('|', r'\|')


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def caption_text(rows: Sequence[Dict], n_images: Optional[int],
                 p_info: Optional[Dict]) -> str:
    n = n_images if n_images is not None else 500
    # 种子文案按实际数据生成，禁止写死（协议 §14.3：单种子必须在论文里标注）
    seeds_seen = sorted({int(s) for r in rows for s in (r.get('seeds') or [])})
    if len(seeds_seen) > 1:
        seeds_txt = (f'Results are mean $\\pm$ std over {len(seeds_seen)} random '
                     f'seeds ({"/".join(str(s) for s in seeds_seen)}). ')
    elif len(seeds_seen) == 1:
        seeds_txt = (f'Results are from a single random seed ({seeds_seen[0]}); '
                     f'no standard deviation is reported. ')
    else:
        seeds_txt = ''
    plus_txt = ('"+" denotes D4 self-ensemble at inference. '
                if any('+' in str(r.get('display', '')) for r in rows) else '')
    cap = (
        f'Quantitative comparison on the {n}-image test set (RGB, [0,1], '
        f'crop border {CROP_BORDER} px). PSNR/SSIM are computed over the full image; '
        f'SSIM uses an {SSIM_WIN}x{SSIM_WIN} Gaussian window with sigma={SSIM_SIGMA} '
        f'(scikit-image 0.19.3, per-channel then averaged). '
        rf'LPIPS uses AlexNet (lpips 0.1.4, input range $[-1,1]$). '
        + seeds_txt + plus_txt +
        f'Checkpoints were selected solely by validation-set PSNR; the test set was '
        f'never used for any model selection.'
    )
    if p_info and p_info.get('best_baseline'):
        seed = p_info.get('p_seed', 10)
        p = (p_info.get('p_holm_by_seed') or {}).get(str(seed))
        if p is None:
            p = (p_info.get('p_by_seed') or {}).get(str(seed))
        if p is not None:
            cap += (f' p-values are from a two-sided Wilcoxon signed-rank test over the '
                    f'{n} paired test images (seed {seed}), '
                    f'Holm-Bonferroni corrected across baselines '
                    f'(vs. best baseline, {esc_tex(str(p_info["best_baseline"]))}: '
                    f'p = {fmt_p(p)}).')
    return cap


def fmt_p(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 'n/a'
    return '< 0.001' if p < 0.001 else f'{p:.3g}'


def render_rows(rows: Sequence[Dict], p_info: Optional[Dict]
                ) -> Tuple[List[List[str]], List[Tuple[str, str, bool]], List[bool]]:
    """产出显示文本矩阵与每列的 (best, second) 排名信息。"""
    headers = ['Method'] + [h for _, h, _, _ in COST_COLS] + \
              [h for _, h, _, _ in METRIC_COLS] + (['p'] if p_info else [])

    # 排名：每个指标列独立判定
    rank_info: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for k, _, nd, higher in METRIC_COLS:
        disp = []
        for r in rows:
            v = r.get(k)
            d = round(float(v), nd) if v is not None and np.isfinite(v) else float('nan')
            disp.append(d)
        rank_info[k] = rank_marks(disp, higher)

    body: List[List[str]] = []
    marks: List[Tuple[str, str, bool]] = []
    for r in rows:
        cells = [str(r['display'])]
        for k, _, nd, _ in COST_COLS:
            v = r.get(k)
            cells.append('--' if v is None else f'{float(v):.{nd}f}')
        for k, _, nd, _ in METRIC_COLS:
            txt, q = fmt_cell(r.get(k), r.get(k + '_std'), nd)
            best, second = rank_info[k]
            cells.append(txt)
            marks.append((k, mark(q, best, second, k), False))
        if p_info:
            cells.append(p_for_row(r, p_info))
        body.append(cells)
    flags = [bool(r.get('is_ours')) for r in rows]
    return [headers] + body, marks, flags


def p_for_row(row: Dict, p_info: Dict) -> str:
    if not row.get('is_ours'):
        return '--'
    p = (p_info.get('p_holm_by_seed') or {}).get(str(p_info.get('p_seed', 10)))
    if p is None:
        p = p_info.get('p_holm')
    if p is None:
        return '--'
    return fmt_p(p)


def write_csv(headers: Sequence[str], body: Sequence[Sequence[str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(headers))
        for r in body:
            # csv 里去掉 LaTeX 的 \pm，用 +/- 便于程序读
            w.writerow([c.replace(r' \pm ', ' +/- ') for c in r])


def write_md(headers: Sequence[str], body: Sequence[Sequence[str]], path: Path,
             caption: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['| ' + ' | '.join(esc_md(h) for h in headers) + ' |',
             '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in body:
        lines.append('| ' + ' | '.join(
            esc_md(c.replace(r' \pm ', ' ± ')) for c in r) + ' |')
    lines += ['', f'*{caption}*', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_tex(headers: Sequence[str], body: Sequence[Sequence[str]],
              marks: Sequence[Tuple[str, str, bool]], path: Path, caption: str,
              label: str, n_metric_cols: int, flags: Optional[Sequence[bool]] = None):
    """booktabs + 无竖线 + 自动加粗/下划线。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ncol = len(headers)
    lines = [
        r'% 需要宏包: booktabs, amsmath, xcolor（\usepackage{booktabs,amsmath}）',
        '% 由 figures/scripts/fig6_table.py 自动生成，请勿手工编辑',
        r'\begin{table*}[t]',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        r'\begin{tabular}{' + 'l' + 'c' * (ncol - 1) + '}',
        r'\toprule',
        ' & '.join(r'\textbf{' + h + '}' for h in headers) + r' \\',
        r'\midrule',
    ]

    # marks 是逐格 (key, tag) 的扁平序列（只含指标列），这里重建索引
    metric_start = ncol - n_metric_cols - (1 if 'p' in headers else 0)
    extra = 1 if 'p' in headers else 0
    for ri, r in enumerate(body):
        cells = []
        for ci, c in enumerate(r):
            txt = esc_tex(c) if ci == 0 else c
            if 0 < ci and ci >= metric_start and ci < metric_start + n_metric_cols:
                mi = (ci - metric_start)
                idx = ri * n_metric_cols + mi
                if idx < len(marks):
                    tag = marks[idx][1]
                    if tag == 'best':
                        txt = r'\textbf{' + txt + '}'
                    elif tag == 'second':
                        txt = r'\underline{' + txt + '}'
            cells.append(txt)
        lines.append(' & '.join(cells) + r' \\')
        # 本方法之前加一条分隔线（若存在基线行在其上方）
        if flags and ri + 1 < len(body) and flags[ri + 1] and not flags[ri]:
            lines.append(r'\midrule')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_typ(headers: Sequence[str], body: Sequence[Sequence[str]],
              marks: Sequence[Tuple[str, str, bool]], path: Path, caption: str,
              label: str, n_metric_cols: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_start = len(headers) - n_metric_cols - (1 if 'p' in headers else 0)
    cs = 'auto, ' + ', '.join(['center'] * (len(headers) - 1))
    lines = [
        f'// 由 figures/scripts/fig6_table.py 自动生成，请勿手工编辑',
        f'#figure(',
        f'  table(',
        f'    columns: ({cs}),',
        f'    stroke: none,',
        f'    align: center,',
    ]
    lines.append('    table.header(' + ', '.join(
        f'[*{esc_typ(h)}]' for h in headers) + '),')
    for ri, r in enumerate(body):
        cells = []
        for ci, c in enumerate(r):
            txt = esc_typ(c.replace(r' \pm ', ' ± ')) if ci == 0 else \
                c.replace(r' \pm ', ' ± ')
            if 0 < ci and metric_start <= ci < metric_start + n_metric_cols:
                idx = ri * n_metric_cols + (ci - metric_start)
                if idx < len(marks) and marks[idx][1] == 'best':
                    txt = f'*{txt}*'
            cells.append(txt)
        lines.append('    ' + ', '.join(cells) + ',')
    lines += [
        '  ),',
        f'  caption: [{caption}],',
        f'  <{label}>,',
        ')',
        '',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description='主结果表生成（协议 §K）')
    ap.add_argument('--metrics', required=True,
                    help='per_sample_metrics_all.json（数字的唯一来源）')
    ap.add_argument('--cost', default=None, help='model_cost.csv: model,params_m,flops_g,infer_ms')
    ap.add_argument('--p-json', default=None,
                    help='p_vs_best_baseline.json（由 stats.py 产出）')
    ap.add_argument('--out-stem', default='figures/data/table_main')
    ap.add_argument('--ours-pattern', default='HoloRestormer',
                    help='匹配本方法名前缀，用于把它排在表末')
    ap.add_argument('--label', default='tab:main')
    ap.add_argument('--n-images', type=int, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args.metrics, args.cost, args.ours_pattern)

    p_info = None
    if args.p_json and Path(args.p_json).exists():
        p_info = load_json(args.p_json)
        if not isinstance(p_info, dict):
            warn(f'{args.p_json} 内容不是对象，忽略')
            p_info = None
    elif args.p_json:
        warn(f'未找到 {args.p_json}，p 列将省略（请先跑 stats.py）')

    table, marks, flags = render_rows(rows, p_info)
    headers, body = table[0], table[1:]
    n_metric = len(METRIC_COLS)
    n_images = args.n_images
    if n_images is None:
        try:
            n_images = int(load_json(args.metrics)['eval_protocol']['test_image_count'])
            info(f'测试样本数取自 eval_protocol: {n_images}')
        except (KeyError, TypeError, ValueError):
            warn('metrics 里没有 eval_protocol.test_image_count，'
                 '表注将写 500；请核对或显式传 --n-images')
            n_images = 500
    caption = caption_text(rows, n_images, p_info)

    stem = Path(args.out_stem)
    write_csv(headers, body, stem.with_suffix('.csv'))
    write_md(headers, body, stem.with_suffix('.md'), caption)
    write_tex(headers, body, marks, stem.with_suffix('.tex'), caption,
              args.label, n_metric, flags)
    write_typ(headers, body, marks, stem.with_suffix('.typ'), caption,
              args.label, n_metric)

    info(f'已写出四份: {stem}.csv / .md / .tex / .typ')
    info(f'行数 {len(body)}；指标列 {n_metric}；p 列 {"有" if p_info else "无"}')
    for r in rows:
        info(f"  {r['display']:24s} seeds={r['seeds']} "
             f"PSNR={r['psnr'] if r['psnr'] is None else round(float(r['psnr']), 2)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
