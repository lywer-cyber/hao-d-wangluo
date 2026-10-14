#!/usr/bin/env python3
"""
生成 roi.json —— Fig.2 / Fig.3 的局部放大区域坐标（协议 §I.2，冻结）。

核心原则
  坐标**由算法决定，禁止人工框选**。规则写死在 selection_rule 里并随文件落盘，
  这样"为什么选这个区域"是可复核的，而不是作者的主观偏好。

两类 ROI
  flat   : 局部梯度**最小**的窗 —— 看**散斑抑制**（平坦区最能体现噪声水平）
  detail : 局部梯度**最大**的窗 —— 看**分辨率恢复**（纹理区最能体现细节）

用法
  python build_roi.py \
      --hr-dir   $HOLO_DATA_ROOT/HR \
      --samples  data/splits/visual_samples.txt \
      --out      data/splits/roi.json \
      --metrics  figures/data/per_sample_metrics_all.json \
      --baseline SwinIR

  # 只看算法选出来的框对不对，不写文件：
  python build_roi.py --hr-dir ... --samples ... --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FIG_SEED, PROTOCOL_VERSION, dump_json_strict, imread_rgb01, info, load_json,
    to_display_gray, warn,
)

ROI_SIZE_DEFAULT = (64, 64)      # HR 网格
MARGIN_DEFAULT = 64              # 候选窗离图像边界至少这么远，避开边缘伪影
DETAIL_CENTRAL_FRAC = 0.60       # detail 只在中心 60% 区域内找，避免贴边
IMG_EXT = ('.bmp', '.png', '.tif', '.tiff')


# --------------------------------------------------------------------------
# 局部梯度与积分图
# --------------------------------------------------------------------------
def local_gradient(gray: np.ndarray) -> np.ndarray:
    """|Sobel_x| + |Sobel_y|，与 selection_rule 的措辞一致。"""
    import cv2
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy)


def integral_image(x: np.ndarray) -> np.ndarray:
    """(H+1, W+1) 的二维前缀和，便于 O(1) 求任意窗的和。"""
    return np.pad(np.cumsum(np.cumsum(x, axis=0), axis=1), ((1, 0), (1, 0)))


def window_sums(integ: np.ndarray, wh: int, ww: int) -> np.ndarray:
    """返回 (H-wh+1, W-ww+1) 的窗内和。"""
    return (integ[wh:, ww:] - integ[:-wh, ww:]
            - integ[wh:, :-ww] + integ[:-wh, :-ww])


# --------------------------------------------------------------------------
# 选点
# --------------------------------------------------------------------------
def _valid_mask(H: int, W: int, wh: int, ww: int, margin: int,
                central_only: bool) -> np.ndarray:
    """标出所有合法的窗左上角位置。"""
    rows = H - wh + 1
    cols = W - ww + 1
    if rows <= 0 or cols <= 0:
        raise ValueError(
            f'图太小 ({H}x{W})，放不下 {wh}x{ww} 的 ROI 窗')
    ys = np.arange(rows)[:, None] * np.ones((1, cols), dtype=int)
    xs = np.ones((rows, 1), dtype=int) * np.arange(cols)[None, :]
    m = (ys >= margin) & (xs >= margin) & (ys + wh <= H - margin) & (xs + ww <= W - margin)

    if central_only:
        cy0, cy1 = int(H * (1 - DETAIL_CENTRAL_FRAC) / 2), int(H * (1 + DETAIL_CENTRAL_FRAC) / 2)
        cx0, cx1 = int(W * (1 - DETAIL_CENTRAL_FRAC) / 2), int(W * (1 + DETAIL_CENTRAL_FRAC) / 2)
        m &= (ys >= cy0) & (ys + wh <= cy1) & (xs >= cx0) & (xs + ww <= cx1)

    if not m.any():
        # 退一步：放宽 margin，保证算法仍能给出结果，但要明确警告
        warn(f'在 margin={margin} 下无合法窗（图 {H}x{W}，ROI {wh}x{ww}），已放宽到 0')
        return np.ones((rows, cols), dtype=bool)
    return m


def _argmin_lex(sums: np.ndarray, valid: np.ndarray) -> Tuple[int, int]:
    """取最小值的窗；并列时按 (y, x) 字典序取第一个。"""
    masked = np.where(valid, sums, np.inf)
    flat = np.argmin(masked)
    y, x = np.unravel_index(flat, masked.shape)
    return int(y), int(x)


def _argmax_lex(sums: np.ndarray, valid: np.ndarray) -> Tuple[int, int]:
    masked = np.where(valid, sums, -np.inf)
    flat = np.argmax(masked)
    y, x = np.unravel_index(flat, masked.shape)
    return int(y), int(x)


def _overlap(a: Sequence[int], b: Sequence[int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def select_rois(gray: np.ndarray, roi_wh: Tuple[int, int], margin: int
                ) -> Tuple[List[int], List[int]]:
    """返回 (flat_box, detail_box)，格式 [x, y, w, h]。"""
    wh, ww = roi_wh
    g = local_gradient(gray.astype(np.float32))
    integ = integral_image(g)
    sums = window_sums(integ, wh, ww)

    # flat：梯度最小的合法窗（刻意允许贴边，平坦区在哪都行；但仍受 margin 约束）
    v_flat = _valid_mask(gray.shape[0], gray.shape[1], wh, ww, margin, central_only=False)
    fy, fx = _argmin_lex(sums, v_flat)
    flat = [fx, fy, ww, wh]

    # detail：中心区域内梯度最大的窗；若与 flat 重叠则取次优，直到不重叠
    v_det = _valid_mask(gray.shape[0], gray.shape[1], wh, ww, margin, central_only=True)
    masked = np.where(v_det, sums, -np.inf).copy()

    for _ in range(64):
        dy, dx = _argmax_lex(masked, v_det)
        detail = [dx, dy, ww, wh]
        if not _overlap(flat, detail):
            return flat, detail
        masked[dy, dx] = -np.inf          # 排除这个位置，取下一个
        if not np.isfinite(masked).any() or masked.max() == -np.inf:
            break

    warn('未找到与 flat 不重叠的 detail 窗，已放宽非重叠约束')
    return flat, detail


# --------------------------------------------------------------------------
# 样本清单与分层
# --------------------------------------------------------------------------
def read_samples(path: Optional[str], hr_dir: Path) -> List[str]:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'找不到样本清单: {p}')
        out = []
        for line in p.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            out.append(Path(s).stem)
        if out:
            return out
    nums = sorted([p.stem for p in hr_dir.iterdir()
                   if p.suffix.lower() in IMG_EXT], key=lambda s: (len(s), s))
    return nums


def find_hr(hr_dir: Path, number: str) -> Path:
    for ext in IMG_EXT:
        c = hr_dir / f'{number}{ext}'
        if c.exists():
            return c
    raise FileNotFoundError(f'{hr_dir} 下找不到 {number} 对应的 HR 图')


def stratify_by_baseline(metrics_path: Optional[str], baseline: str,
                         samples: Sequence[str], seed: int) -> Dict[str, List[str]]:
    """按基线在测试集上的 PSNR 把候选样本分成 easy/medium/hard 三档。

    协议要求分层依据**只能来自基线**，不能用本方法的结果，否则是选择性报告。
    """
    empty = {'easy': [], 'medium': [], 'hard': []}
    if not metrics_path or not Path(metrics_path).exists():
        warn(f'未提供基线指标（{metrics_path}），跳过分层；'
             f'fig2_samples 将按数值序取前 4 个，请人工确认覆盖了难度')
        return empty

    data = load_json(metrics_path)
    res = data.get('results', {})
    if baseline not in res:
        warn(f'{metrics_path} 里没有基线 {baseline}，可用: {sorted(res.keys())}')
        return empty

    per_seed = res[baseline].get(str(seed)) or res[baseline].get(seed) or {}
    if not per_seed:
        warn(f'基线 {baseline} 在 seed {seed} 下无数据')
        return empty

    psnr = {}
    for s in samples:
        rec = per_seed.get(str(s))
        if rec and rec.get('psnr') is not None:
            psnr[str(s)] = float(rec['psnr'])

    if len(psnr) < 3:
        warn(f'可用于分层的样本太少（{len(psnr)}），跳过分层')
        return empty

    items = sorted(psnr.items(), key=lambda kv: kv[1])       # 由难到易
    n = len(items)
    k1, k2 = max(1, n // 3), max(2, 2 * n // 3)
    return {
        'easy': [s for s, _ in items[k2:]],      # PSNR 高 = 易
        'medium': [s for s, _ in items[k1:k2]],
        'hard': [s for s, _ in items[:k1]],      # PSNR 低 = 难
    }


def pick_fig2_samples(cohort: Sequence[str], strata: Dict[str, List[str]],
                      need: int = 4) -> List[str]:
    """每档至少 1 张，其余按数值序补齐（保证可复现）。"""
    def _k(s):
        return (0, int(s)) if str(s).isdigit() else (1, str(s))

    picked: List[str] = []
    if strata.get('easy') and strata.get('medium') and strata.get('hard'):
        for tier in ('easy', 'medium', 'hard'):
            if strata[tier] and len(picked) < need:
                picked.append(sorted(strata[tier], key=_k)[0])
    for s in sorted(cohort, key=_k):
        if len(picked) >= need:
            break
        if s not in picked:
            picked.append(s)
    return sorted(picked, key=_k)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description='生成 roi.json（协议 §I.2）')
    ap.add_argument('--hr-dir', required=True, help='HR 图目录')
    ap.add_argument('--samples', default=None,
                    help='样本清单（每行一个文件名）；默认用 hr-dir 下全部')
    ap.add_argument('--out', default='data/splits/roi.json')
    ap.add_argument('--roi-size', nargs=2, type=int, default=list(ROI_SIZE_DEFAULT),
                    metavar=('H', 'W'), help='ROI 尺寸（HR 网格）')
    ap.add_argument('--margin', type=int, default=MARGIN_DEFAULT,
                    help='候选窗离边界的距离')
    ap.add_argument('--metrics', default=None,
                    help='per_sample_metrics_all.json（用于按基线分层）')
    ap.add_argument('--baseline', default='SwinIR', help='分层依据的基线名')
    ap.add_argument('--seed', type=int, default=FIG_SEED, help='分层用的种子')
    ap.add_argument('--fig2-need', type=int, default=4, help='Fig.2 场景数')
    ap.add_argument('--dry-run', action='store_true', help='只打印，不写文件')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    hr_dir = Path(args.hr_dir)
    if not hr_dir.exists():
        print(f'错误: HR 目录不存在 {hr_dir}', file=sys.stderr)
        return 2

    roi_wh = (int(args.roi_size[0]), int(args.roi_size[1]))
    samples = read_samples(args.samples, hr_dir)
    if not samples:
        print('错误: 没有可用的样本', file=sys.stderr)
        return 2

    # ---- 前提检查：所有 HR 必须同尺寸，否则绝对坐标失效 ----
    sizes = set()
    for s in samples:
        g = to_display_gray(imread_rgb01(find_hr(hr_dir, s)))
        sizes.add(g.shape)
    if len(sizes) > 1:
        print('错误: HR 图尺寸不一致 -> ' + str(sorted(sizes)) +
              '\n      绝对坐标的 ROI 无法跨尺寸复用。'
              '\n      请先把数据统一尺寸，或改协议为归一化坐标后再生成。',
              file=sys.stderr)
        return 3
    H, W = sizes.pop()
    min_side = min(H, W)
    if min_side < max(roi_wh) + 2 * args.margin:
        warn(f'图 {H}x{W} 偏小（ROI {roi_wh} + margin {args.margin}），'
             f'候选窗可能退化，建议调小 --margin')

    info(f'HR 尺寸统一为 {H}x{W}；样本 {len(samples)} 张；ROI={roi_wh}；margin={args.margin}')

    rois: Dict[str, Dict[str, List[int]]] = {}
    for s in samples:
        gray = to_display_gray(imread_rgb01(find_hr(hr_dir, s)))
        flat, detail = select_rois(gray, roi_wh, args.margin)
        rois[s] = {'flat': flat, 'detail': detail}
        info(f'  {s}: flat={flat}  detail={detail}')

    strata = stratify_by_baseline(args.metrics, args.baseline, samples, args.seed)
    fig2 = pick_fig2_samples(samples, strata, need=args.fig2_need)
    info(f'fig2_samples = {fig2}')
    if strata.get('easy'):
        info(f'分层: easy={strata["easy"]} medium={strata["medium"]} hard={strata["hard"]}')

    out = {
        'protocol_version': PROTOCOL_VERSION,
        'coordinate_convention': 'HR grid, [x, y, w, h], 0-based, top-left origin, inclusive-exclusive',
        'roi_size_hr': [int(roi_wh[1]), int(roi_wh[0])],   # 写成 [w, h]，与 box 一致
        'selection_rule': {
            'flat': 'minimize local_gradient in valid window',
            'detail': 'maximize local_gradient among candidates within central '
                      f'{DETAIL_CENTRAL_FRAC:.2f} region',
            'local_gradient': 'mean(|Sobel_x(HR)| + |Sobel_y(HR)|) over the window',
            'min_border_margin': int(args.margin),
            'non_overlap': True,
            'tie_break': 'lexicographic (y, x)',
            'gray_definition': 'mean over RGB channels',
        },
        'stratification': {
            'rule': 'baseline_psnr_tercile',
            'baseline': args.baseline,
            'seed': int(args.seed),
            'easy': strata.get('easy', []),
            'medium': strata.get('medium', []),
            'hard': strata.get('hard', []),
        },
        'fig2_samples': fig2,
        'rois': rois,
    }

    if args.dry_run:
        info('dry-run：未写文件')
        return 0

    p = dump_json_strict(out, args.out)
    info(f'已写出 {p}（生成后即冻结，后续改动算协议变更）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
