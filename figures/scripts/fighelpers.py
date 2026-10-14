"""
出图辅助层 —— fig2/3/4/5/7/8/9 的共用底座。

分工
  common.py     协议常量、统一样式、色板、图像 IO、指标实现
  fighelpers.py 出图操作：方法排序、面板网格、ROI 框、径向谱、曲线读取

所有出图脚本都必须 import 本模块 + common.py，以保证全篇图件风格与规则一致。
"""

from __future__ import annotations

import csv as _csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIG_FONT_SIZE, FIG_SEED, LR_NYQUIST, SCALE, clamp01, imread_rgb01,
    load_json, to_display_gray, warn,
)

IMG_EXTS = ('.bmp', '.png', '.tif', '.tiff')
LR_UPSAMPLE_INTERPOLATION = 'bicubic'          # 协议 §E.5 冻结


# --------------------------------------------------------------------------
# 图像读取
# --------------------------------------------------------------------------
def find_image(folder, number: str, suffix: str = '') -> Optional[Path]:
    """在目录里找 {number}{suffix}.<ext>，找不到返回 None。"""
    folder = Path(folder)
    if not folder.exists():
        return None
    for ext in IMG_EXTS:
        p = folder / f'{number}{suffix}{ext}'
        if p.exists():
            return p
    hits = sorted(p for p in folder.glob(f'{number}{suffix}.*')
                  if p.suffix.lower() in IMG_EXTS)
    return hits[0] if hits else None


def read_hr(hr_dir, number: str) -> np.ndarray:
    p = find_image(hr_dir, number, '')
    if p is None:
        raise FileNotFoundError(f'{hr_dir} 下找不到 HR {number}')
    return imread_rgb01(p)


def read_lr(lr_dir, number: str) -> np.ndarray:
    p = find_image(lr_dir, number, '')
    if p is None:
        raise FileNotFoundError(f'{lr_dir} 下找不到 LR {number}')
    return imread_rgb01(p)


def sr_dir_candidates(exp_root, model: str, seed: int,
                      variant: str = 'plain') -> List[Path]:
    """按优先级列出 SR 图可能所在的目录。

    两套布局都要认：
      A) 本仓库早期约定  {exp_root}/{model}/seed{seed}/test_sr[/_plus]/
      B) 真机实际        {exp_root}/{model}/results/{test|test_plus}/
         （holo_project/evaluate_holo.py 写到 experiments/{name}/results/{split}/，
           名字里已含 seed，所以没有 seed 层）
    """
    base = Path(exp_root) / model
    sub = 'test_sr_plus' if variant == 'plus' else 'test_sr'
    split = 'test_plus' if variant == 'plus' else 'test'
    cands = [
        base / f'seed{seed}' / sub,
        base / 'results' / split,
        base / f'seed{seed}' / 'results' / split,
        base / 'results' / sub,
        base / f'seed{seed}' / 'results' / sub,
    ]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_sr_dir(exp_root, model: str, seed: int, variant: str = 'plain'):
    """返回第一个存在的 SR 目录；都不存在返回 None。"""
    for c in sr_dir_candidates(exp_root, model, seed, variant):
        if c.is_dir():
            return c
    return None


def read_sr(exp_root, model: str, seed: int, number: str,
            variant: str = 'plain') -> np.ndarray:
    """读某个样本的 SR 图。搜索顺序见 sr_dir_candidates。"""
    d = resolve_sr_dir(exp_root, model, seed, variant)
    p = find_image(d, number, '_SR') if d is not None else None
    if p is None:
        tried = ', '.join(str(c) for c in
                          sr_dir_candidates(exp_root, model, seed, variant))
        raise FileNotFoundError(f'找不到 {number}_SR（试过: {tried}）')
    return imread_rgb01(p)


def upsample_lr(lr01: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """LR 面板必须 bicubic 上采样到 HR 尺寸（协议 §E.5）。

    禁止最近邻（显得输入过差）、禁止任何锐化或神经上采样（显得输入过好）。
    """
    import cv2
    if lr01.ndim == 2:
        lr01 = np.stack([lr01] * 3, axis=-1)
    H, W = lr01.shape[:2]
    up = cv2.resize(lr01.astype(np.float32), (W * scale, H * scale),
                    interpolation=cv2.INTER_CUBIC)
    return np.clip(up, 0.0, 1.0)


# --------------------------------------------------------------------------
# 方法排序（协议 §C Fig.2 固定行序）
# --------------------------------------------------------------------------
def ordered_methods(metrics_path, ours_pattern: str = 'HoloRestormer',
                    variants: Sequence[str] = ('plain',),
                    exp_root=None, seed: int = FIG_SEED
                    ) -> List[Dict[str, object]]:
    """行序：基线按 PSNR **升序**（最差在前）→ 本方法（最后，紧邻 HR）。

    只保留在 metrics 里真的有数据的模型；给了 exp_root 时还会检查 SR 目录是否存在。
    """
    data = load_json(metrics_path)
    res = data.get('results', {})
    rows: List[Dict[str, object]] = []

    for model in sorted(res):
        by_seed = res[model].get(str(seed)) or res[model].get(seed)
        if not by_seed:
            continue
        for variant in variants:
            key = 'psnr' + ('_plus' if variant == 'plus' else '')
            vals = [float(rec[key]) for rec in by_seed.values()
                    if rec.get(key) is not None and np.isfinite(float(rec[key]))]
            if not vals:
                continue
            if exp_root is not None and resolve_sr_dir(
                    exp_root, model, seed, variant) is None:
                tried = [str(c) for c in
                         sr_dir_candidates(exp_root, model, seed, variant)]
                warn(f"跳过 {model}/{variant}：找不到 SR 目录（试过 {tried}）")
                continue
            rows.append({
                'model': model,
                'variant': variant,
                'display': model + ('+' if variant == 'plus' else ''),
                'is_ours': model.startswith(ours_pattern),
                'psnr': float(np.mean(vals)),
            })

    baselines = [r for r in rows if not r['is_ours']]
    ours = [r for r in rows if r['is_ours']]
    baselines.sort(key=lambda r: (float(r['psnr']), str(r['display'])))
    ours.sort(key=lambda r: str(r['display']))
    return baselines + ours


def methods_from_cost(cost: Dict[str, Dict[str, float]],
                      metrics: Dict) -> List[str]:
    """取既有代价数据、又有指标数据的模型名（用于效率散点）。"""
    res = metrics.get('results', {})
    return sorted(m for m in cost if m in res)


# --------------------------------------------------------------------------
# 显示范围与色标（全局统一，禁止逐格自适应）
# --------------------------------------------------------------------------
def global_display_range(arrays: Sequence[np.ndarray],
                         lo_pct: float = 0.5, hi_pct: float = 99.5
                         ) -> Tuple[float, float]:
    """所有面板共用的显示范围（协议 §C Fig.2 陷阱）。"""
    vals = [np.asarray(a, dtype=np.float32).ravel() for a in arrays
            if a is not None and np.size(a)]
    if not vals:
        return 0.0, 1.0
    cat = np.concatenate(vals)
    vmin = float(np.percentile(cat, lo_pct))
    vmax = float(np.percentile(cat, hi_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def global_vmax(arrays: Sequence[np.ndarray], percentile: float = 99.0) -> float:
    """误差图/频谱图共用的全局色标上限（协议 §C Fig.3）。"""
    vals = [np.asarray(a, dtype=np.float32).ravel() for a in arrays
            if a is not None and np.size(a)]
    if not vals:
        return 1.0
    v = float(np.percentile(np.concatenate(vals), percentile))
    return v if np.isfinite(v) and v > 0 else 1.0


# --------------------------------------------------------------------------
# 面板装饰
# --------------------------------------------------------------------------
def _pe_outline(width: float = 1.2):
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground='black')]


def annotate_metric(ax, text: str, loc: str = 'br',
                    color: str = 'white', fontsize: Optional[float] = None):
    """面板角上写指标。白字黑描边，任意背景下都可读（协议 §C Fig.2 第 4 条）。"""
    xy = {'br': (0.98, 0.02), 'tl': (0.02, 0.98),
          'tr': (0.98, 0.98), 'bl': (0.02, 0.02)}[loc]
    ha = 'right' if loc in ('br', 'tr') else 'left'
    va = 'bottom' if loc in ('br', 'bl') else 'top'
    ax.text(xy[0], xy[1], text, transform=ax.transAxes,
            ha=ha, va=va, color=color,
            fontsize=fontsize if fontsize else FIG_FONT_SIZE - 1, zorder=6,
            bbox=dict(facecolor='black', alpha=0.42, edgecolor='none',
                      boxstyle='square,pad=1.6'),
            path_effects=_pe_outline())


def draw_roi_box(ax, box: Sequence[int], color: str, lw: float = 1.1,
                 label: Optional[str] = None):
    """整图上画 ROI 框（协议 §C Fig.2 第 3 条）。box = [x, y, w, h]。"""
    from matplotlib.patches import Rectangle
    x, y, w, h = [int(v) for v in box]
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                           linewidth=lw, zorder=5))
    if label:
        ax.text(x, max(y - 2, 0), label, color=color,
                fontsize=FIG_FONT_SIZE - 2, ha='left', va='bottom', zorder=6,
                path_effects=_pe_outline(1.0))


def tight_panel(ax, border_color: Optional[str] = None, lw: float = 1.0):
    """图象面板：去刻度、去坐标轴；给了颜色则只留一圈细边框。"""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(border_color is not None)
        if border_color is not None:
            s.set_color(border_color)
            s.set_linewidth(lw)


def panel_grid(n_rows: int, n_cols: int, width: str = 'double',
               cell_in: float = 1.05, row_label_extra: float = 0.0):
    """建一个 n_rows × n_cols 的面板网格，返回 (fig, axes)。

    行标签靠左留白由 row_label_extra 控制（单位 inch）。
    """
    import matplotlib.pyplot as plt
    from common import FIG_WIDTH_DOUBLE_IN, FIG_WIDTH_SINGLE_IN
    W = {'single': FIG_WIDTH_SINGLE_IN, 'double': FIG_WIDTH_DOUBLE_IN}[width]
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(W, cell_in * n_rows + row_label_extra),
        squeeze=False)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.965, bottom=0.02,
                        wspace=0.03, hspace=0.03)
    return fig, axes


# --------------------------------------------------------------------------
# ROI
# --------------------------------------------------------------------------
def load_roi(roi_path) -> Dict:
    return load_json(roi_path)


def roi_box(roi: Dict, number: str, kind: str) -> Optional[List[int]]:
    return (roi.get('rois', {}).get(str(number)) or {}).get(kind)


def crop_box(img: np.ndarray, box: Sequence[int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    return img[y:y + h, x:x + w]


# --------------------------------------------------------------------------
# 频谱
# --------------------------------------------------------------------------
def radial_psd(img01: np.ndarray, n_bins: int = 128, r_max: float = 2.0
               ) -> Tuple[np.ndarray, np.ndarray]:
    """径向平均功率谱，横轴归一化到 LR Nyquist（协议 §C Fig.4b）。

    r̂ = sqrt(fy² + fx²) / f_LR，f_LR = 1/(2·scale) = 0.125 cycles/px（HR 网格）。
    返回 (r̂ 中心数组, 平均 PSD 数组)；r̂ > r_max 的样本点丢弃。
    """
    g = to_display_gray(img01).astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(g))
    P = F.real ** 2 + F.imag ** 2
    H, W = g.shape
    fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2) / LR_NYQUIST

    edges = np.linspace(0.0, r_max, n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=P.ravel(), minlength=n_bins)
    cnts = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_psd = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), mean_psd


def log_magnitude_spectrum(img01: np.ndarray) -> np.ndarray:
    """2D 频谱幅度：log(1 + |fftshift(FFT2)|)。所有面板共用同一 colorbar。"""
    g = to_display_gray(img01).astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(g))
    return np.log1p(np.abs(F))


def add_cutoff_circle(ax, color: str = '0.45', lw: float = 1.0,
                      label: Optional[str] = None):
    """2D 频谱上叠加 r̂ = 1 的圆（协议 §C Fig.4a）。"""
    from matplotlib.patches import Circle
    H, W = ax.images[0].get_array().shape[:2]
    # r̂ = 1 在 HR 网格上对应半径 f_LR * (H 或 W) 个像素
    rx, ry = LR_NYQUIST * W, LR_NYQUIST * H
    ax.add_patch(Circle((W / 2 - 0.5, H / 2 - 0.5), min(rx, ry),
                        fill=False, edgecolor=color, linewidth=lw, zorder=5))
    if label:
        ax.text(W / 2, H / 2 - min(ry, rx) - 3, label, color=color,
                fontsize=FIG_FONT_SIZE - 2, ha='center', va='bottom', zorder=6,
                path_effects=_pe_outline(1.0))


def add_cutoff_line(ax, color: str = '0.35', label: bool = True,
                    y_text: float = 0.05):
    """1D 径向谱上画 r̂ = 1 竖线（协议 §C Fig.4b 必需）。"""
    ax.axvline(1.0, color=color, linestyle='--', linewidth=0.9, zorder=1)
    if label:
        ax.text(1.04, y_text, 'LR Nyquist\n(physical cutoff)',
                transform=ax.get_xaxis_transform(), rotation=90,
                va='bottom', ha='left', fontsize=FIG_FONT_SIZE - 2, color=color)


# --------------------------------------------------------------------------
# 曲线 / 代价
# --------------------------------------------------------------------------
def load_val_curves(exp_root, models: Sequence[str], seeds: Sequence[int],
                    fname: str = 'val_metrics.csv'
                    ) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """读各 run 的 val_metrics.csv（协议 §11.1 列: iter,psnr,ssim,rmse,mae,lr）。"""
    out: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for m in models:
        for s in seeds:
            # 两种布局：{m}/seed{s}/logs/（协议 §15）与 {m}/logs/（真机，seed 在 run 名里）
            cands = [Path(exp_root) / m / f'seed{s}' / 'logs' / fname,
                     Path(exp_root) / m / 'logs' / fname]
            p = next((c for c in cands if c.exists()), None)
            if p is None:
                warn('缺验证曲线 ' + ' / '.join(str(c) for c in cands))
                continue
            its, ps = [], []
            with p.open('r', encoding='utf-8', newline='') as f:
                for row in _csv.DictReader(f):
                    try:
                        its.append(float(row['iter']))
                        ps.append(float(row['psnr']))
                    except (KeyError, TypeError, ValueError):
                        continue
            if its:
                out.setdefault(m, {})[int(s)] = (np.asarray(its), np.asarray(ps))
    return out


def mean_std_curve(curves: Dict[int, Tuple[np.ndarray, np.ndarray]]
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """多种子曲线对齐到公共 iter 网格，返回 (iters, mean, std)。"""
    empty = (np.array([]), np.array([]), np.array([]))
    if not curves:
        return empty
    common = None
    for it, _ in curves.values():
        s = set(it.tolist())
        common = s if common is None else (common & s)
    if not common:
        warn('各种子的 iter 网格没有交集')
        return empty
    grid = np.array(sorted(common))
    stack = []
    for it, ps in curves.values():
        m = dict(zip(it.tolist(), ps.tolist()))
        stack.append([m[t] for t in grid.tolist()])
    arr = np.asarray(stack, dtype=np.float64)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(grid)
    return grid, arr.mean(axis=0), std


def read_cost_csv(path) -> Dict[str, Dict[str, float]]:
    """读 model_cost.csv: model,params_m,flops_g,infer_ms。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'找不到 {path}（格式: model,params_m,flops_g,infer_ms）')
    out: Dict[str, Dict[str, float]] = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        for row in _csv.DictReader(f):
            model = (row.get('model') or '').strip()
            if not model:
                continue
            rec = {}
            for k in ('params_m', 'flops_g', 'infer_ms'):
                v = (row.get(k) or '').strip()
                if v:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        warn(f'{path}: {model}.{k} 不是数字: {v!r}')
            out[model] = rec
    return out


def pick_numbers_by_metric(metrics_path, model: str, seed: int,
                           n: int = 3, worst: bool = True,
                           variant: str = 'plain') -> List[str]:
    """按某模型某种子的逐图 PSNR 挑样本（Fig.9 失败案例）。

    **选取规则必须在图注里写明**（协议 §L）：按谁的、取最好还是最差。
    """
    key = 'psnr' + ('_plus' if variant == 'plus' else '')
    per = load_json(metrics_path).get('results', {}).get(model, {})
    per = per.get(str(seed)) or per.get(seed) or {}
    items = [(str(k), float(v[key])) for k, v in per.items()
             if v.get(key) is not None and np.isfinite(float(v[key]))]
    if not items:
        raise ValueError(f'{model}/seed{seed} 没有可用的逐图 PSNR')
    items.sort(key=lambda kv: kv[1], reverse=not worst)
    return [k for k, _ in items[:n]]
