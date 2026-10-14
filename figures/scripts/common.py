"""
出图公共层：协议读取、统一样式、色板、保存、图像 IO。

所有出图脚本 **必须** import 本模块并使用它提供的样式与色板，
以保证全篇图件风格一致（见《全息超分-论文出图规范-v7.md》§M）。

设计约束
  - 目标运行环境是 Python 3.8（训练服务器），所以只用 3.8 可用的语法。
  - 不引入除 numpy / matplotlib / opencv 之外的重依赖（LPIPS 单独可选）。
  - 一切"影响图上数字"的参数都集中在文件顶部，便于对照协议核查。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# 路径：若同级存在 scripts/holo_common（训练管线用），一并加进 sys.path
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # figures/scripts
_PROJ = _HERE.parent.parent                      # 规范文档所在目录（~/桌面/10.11）
for _cand in (_PROJ / 'scripts', _PROJ):
    if _cand.exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))


# --------------------------------------------------------------------------
# 协议常量（与《全息超分-实验协议-冻结版-v7.md》逐条对应）
# --------------------------------------------------------------------------
PROTOCOL_VERSION = 'v7'
METRIC_IMPL_VERSION = 2

CROP_BORDER = 4                 # 协议 §12.1 第 1 条
SSIM_WIN = 11                   # 协议 §12.1 第 5 条
SSIM_SIGMA = 1.5
SSIM_DATA_RANGE = 1.0
SSIM_CHANNEL_AXIS = 2
LPIPS_NET = 'alex'
LPIPS_INPUT_RANGE = '-1_1'
SCALE = 4
SEEDS = (10, 20, 30)            # 协议 §14
FIG_SEED = 10                   # 协议 §13.B.2 第 2 条：论文正文图统一取 seed 10
FIG_SAMPLE_COUNT = 10
TOTAL_ITERATIONS = 60000
LR_MILESTONES = (30000, 48000, 54000, 57000)   # 协议 §8
LR_NYQUIST = 1.0 / (2.0 * SCALE)               # 0.125 cycles/px（HR 网格）
HF_SIGMA_PX = 1.8006                           # 协议 §9.4
GFL_RAMP = 30000
LOSS_WEIGHTS = {'l1': 1.0, 'gfl': 0.1, 'hf': 0.2}

DEFAULT_CONFIG_NAME = 'experiment_config.txt'


# --------------------------------------------------------------------------
# 极简 INI 读取（对照 experiment_config.txt 的格式：section + key=value）
# --------------------------------------------------------------------------
def _expand_env(s: str) -> str:
    """展开 ${VAR}；未设置的变量原样保留并计数。"""
    missing: List[str] = []

    def _sub(m):
        name = m.group(1)
        if name in os.environ:
            return os.environ[name]
        missing.append(name)
        return m.group(0)

    out = re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', _sub, s)
    if missing:
        print(f'[common] 警告: 环境变量未设置 -> {sorted(set(missing))}',
              file=sys.stderr)
    return out


def load_protocol(path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """读 experiment_config.txt，返回 {section: {key: value}}。

    只做最朴素的解析：`#`/`;` 起始为注释，`[name]` 开节，`key=value` 赋值。
    找不到文件时返回空 dict（调用方自行决定是否必须存在）。
    """
    if path is None:
        cand = [Path.cwd() / DEFAULT_CONFIG_NAME,
                _PROJ / DEFAULT_CONFIG_NAME,
                _PROJ / 'configs' / DEFAULT_CONFIG_NAME]
        path = next((str(p) for p in cand if p.exists()), None)
        if path is None:
            return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'找不到配置文件: {p}')

    cfg: Dict[str, Dict[str, str]] = {}
    cur = 'default'
    for raw in p.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line[0] in '#;':
            continue
        if line.startswith('[') and line.endswith(']'):
            cur = line[1:-1].strip()
            cfg.setdefault(cur, {})
            continue
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        cfg.setdefault(cur, {})[k.strip()] = _expand_env(v.strip())
    return cfg


def cfg_get(cfg: Dict[str, Dict[str, str]], section: str, key: str,
            default=None):
    return cfg.get(section, {}).get(key, default)


def cfg_int(cfg, section, key, default=None) -> Optional[int]:
    v = cfg_get(cfg, section, key, None)
    return default if v is None else int(str(v).strip())


def cfg_float(cfg, section, key, default=None) -> Optional[float]:
    v = cfg_get(cfg, section, key, None)
    return default if v is None else float(str(v).strip())


def cfg_list(cfg, section, key, default=None, cast=str) -> List:
    v = cfg_get(cfg, section, key, None)
    if v is None:
        return list(default or [])
    return [cast(x.strip()) for x in str(v).split(',') if x.strip()]


# --------------------------------------------------------------------------
# 样式：Okabe-Ito 色盲友好色板 + 灰度可区分的线型/标记
# --------------------------------------------------------------------------
# 参考 Okabe & Ito (2008)，色盲友好、灰度打印下也基本可分辨
OKABE_ITO = [
    '#000000',   # black
    '#0072B2',   # blue
    '#D55E00',   # vermillion
    '#009E73',   # bluish green
    '#CC79A7',   # reddish purple
    '#E69F00',   # orange
    '#56B4E9',   # sky blue
    '#F0E442',   # yellow
]
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

# 图件尺寸（协议 §D.2）
FIG_WIDTH_SINGLE_IN = 3.35      # 8.5 cm
FIG_WIDTH_DOUBLE_IN = 6.89      # 17.5 cm
FIG_FONT_SIZE = 8               # 最终尺寸下 >= 7 pt
FIG_DPI = 600                   # 位图导出分辨率
EXPORT_FORMATS = ('pdf', 'png')


def setup_matplotlib():
    """统一 matplotlib 样式。所有出图脚本必须先调它。

    返回 matplotlib.pyplot。
    """
    import matplotlib
    matplotlib.use('Agg')           # 无显示环境
    import matplotlib.pyplot as plt

    # 字体回退链：优先 Arial（协议指定），无则用 DejaVu Sans（matplotlib 自带）
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for fam in ('Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'):
        if fam in avail:
            font_family = fam
            break
    else:
        font_family = 'DejaVu Sans'

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [font_family, 'DejaVu Sans'],
        'font.size': FIG_FONT_SIZE,
        'axes.labelsize': FIG_FONT_SIZE,
        'axes.titlesize': FIG_FONT_SIZE,
        'xtick.labelsize': FIG_FONT_SIZE - 1,
        'ytick.labelsize': FIG_FONT_SIZE - 1,
        'legend.fontsize': FIG_FONT_SIZE - 1,
        'legend.frameon': False,
        'figure.dpi': 150,
        'savefig.dpi': FIG_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'pdf.fonttype': 42,          # TrueType 嵌入，避免 Type3 被期刊退回
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
    })
    return plt


def method_style(index: int) -> Dict[str, object]:
    """按索引给出一套 (颜色, 线型, 标记)，保证灰度下仍可区分。"""
    return {
        'color': OKABE_ITO[index % len(OKABE_ITO)],
        'linestyle': LINESTYLES[index % len(LINESTYLES)],
        'marker': MARKERS[index % len(MARKERS)],
    }


def style_map(models: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """把方法名映射到固定样式（按排序后的顺序分配，保证可复现）。"""
    return {m: method_style(i) for i, m in enumerate(sorted(models))}


def panel_label(ax, label: str, loc: str = 'left', pad: float = 0.02):
    """统一的面板编号 (a)(b)(c)。"""
    xy = {'left': (pad, 1 - pad), 'right': (1 - pad, 1 - pad)}[loc]
    ha = {'left': 'left', 'right': 'right'}[loc]
    ax.text(xy[0], xy[1], f'({label})', transform=ax.transAxes,
            ha=ha, va='top', fontsize=FIG_FONT_SIZE,
            color='black', zorder=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=1.5))


# --------------------------------------------------------------------------
# 保存
# --------------------------------------------------------------------------
def save_paper_figure(fig, out_stem, formats: Sequence[str] = EXPORT_FORMATS,
                      dpi: int = FIG_DPI) -> List[Path]:
    """保存论文图件。强制禁止 JPEG（协议 §D.1）。

    formats: 子集 of ('pdf', 'png')；'png' 一律按 dpi=600 导出。
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    bad = [f for f in formats if f.lower() in ('jpg', 'jpeg')]
    if bad:
        raise ValueError(f'协议禁止 JPEG: {bad}')

    written = []
    for fmt in formats:
        p = out_stem.with_suffix('.' + fmt)
        if fmt == 'png':
            fig.savefig(p, dpi=dpi, facecolor='white')
        else:
            fig.savefig(p, facecolor='white')
        written.append(p)
    return written


def figure_size(width: str = 'double', height_in: float = 2.2):
    """按单栏 / 双栏给出 figsize。"""
    w = {'single': FIG_WIDTH_SINGLE_IN, 'double': FIG_WIDTH_DOUBLE_IN}[width]
    return (w, height_in)


# --------------------------------------------------------------------------
# 图像 IO（统一 [0,1] float32，RGB 或灰度）
# --------------------------------------------------------------------------
def imread_rgb01(path) -> np.ndarray:
    """读图为 RGB float32 [0,1]，形状 (H,W,3)。"""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'读不到: {path}')
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0
    return np.clip(img, 0.0, 1.0)


def to_display_gray(img01: np.ndarray) -> np.ndarray:
    """显示用灰度。数据集是 3 通道复制品，取平均即可（不引入色偏）。"""
    if img01.ndim == 3:
        return img01.mean(axis=-1)
    return img01


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def assert_finite(arr: np.ndarray, name: str) -> None:
    if not np.isfinite(np.asarray(arr)).all():
        raise ValueError(f'{name} 含 NaN/Inf')


# --------------------------------------------------------------------------
# 指标（严格照协议 §12.1；本实现与 basicsr.holo_metrics 的一致性
#       由 export_per_sample_metrics.py 的 1e-9 交叉校验保证）
# --------------------------------------------------------------------------
def _crop(a: np.ndarray, b: np.ndarray, border: int = CROP_BORDER):
    if border <= 0:
        return a, b
    return a[border:-border, border:-border], b[border:-border, border:-border]


def calc_psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    a, b = _crop(sr.astype(np.float64), hr.astype(np.float64))
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float('inf')
    return 10.0 * np.log10(SSIM_DATA_RANGE ** 2 / mse)


def calc_rmse(sr: np.ndarray, hr: np.ndarray) -> float:
    a, b = _crop(sr.astype(np.float64), hr.astype(np.float64))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def calc_mae(sr: np.ndarray, hr: np.ndarray) -> float:
    a, b = _crop(sr.astype(np.float64), hr.astype(np.float64))
    return float(np.mean(np.abs(a - b)))


# 仅供**冒烟测试**的 numpy SSIM 回退。默认关闭：协议 §12.1 第 5 条要求
# scikit-image==0.19.3，缺失时应硬失败，而不是悄悄换实现。
_ALLOW_NUMPY_SSIM = False


def enable_numpy_ssim_fallback(enable: bool = True) -> None:
    """打开 numpy SSIM 回退。**只用于没有 skimage 的机器上跑通管路**，
    产出结果不得进论文。"""
    global _ALLOW_NUMPY_SSIM
    _ALLOW_NUMPY_SSIM = bool(enable)
    if enable:
        warn('已启用 numpy SSIM 回退 —— 结果与 skimage 0.19.3 不保证逐位一致，'
             '禁止用于论文报告。')


def _ssim_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """复刻 skimage 的 gaussian_weights 路径（σ=1.5, truncate=3.5,
    use_sample_covariance=False, data_range=1.0），逐通道取均值。"""
    from scipy.ndimage import gaussian_filter
    K1, K2 = 0.01, 0.03
    L = SSIM_DATA_RANGE
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    pad = (SSIM_WIN - 1) // 2

    def _f(x):
        return gaussian_filter(x.astype(np.float64), sigma=SSIM_SIGMA,
                               truncate=3.5, mode='nearest')

    if a.ndim == 2:
        a = a[:, :, None]
        b = b[:, :, None]
    vals = []
    cov_norm = 1.0                       # use_sample_covariance=False
    for c in range(a.shape[2]):
        x, y = a[:, :, c], b[:, :, c]
        ux, uy = _f(x), _f(y)
        uxx, uyy, uxy = _f(x * x), _f(y * y), _f(x * y)
        vx = cov_norm * (uxx - ux * ux)
        vy = cov_norm * (uyy - uy * uy)
        vxy = cov_norm * (uxy - ux * uy)
        num = (2 * ux * uy + C1) * (2 * vxy + C2)
        den = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
        ssim_map = num / den
        if pad > 0:
            ssim_map = ssim_map[pad:-pad, pad:-pad]
        vals.append(ssim_map.mean())
    return float(np.mean(vals))


def calc_ssim(sr: np.ndarray, hr: np.ndarray) -> float:
    """scikit-image，channel_axis=2，逐通道算完取均值（协议 §12.1 第 5 条）。"""
    a, b = _crop(sr, hr)
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        if not _ALLOW_NUMPY_SSIM:
            raise RuntimeError(
                '协议 §12.1 第 5 条要求 scikit-image==0.19.3 计算 SSIM，'
                '当前环境缺失。\n'
                '  - 正式实验：装上 scikit-image==0.19.3 后重跑；\n'
                '  - 仅跑通管路：加 --dev-numpy-ssim（结果禁止进论文）。')
        return _ssim_numpy(a, b)
    return float(ssim(
        a, b,
        win_size=SSIM_WIN,
        data_range=SSIM_DATA_RANGE,
        channel_axis=SSIM_CHANNEL_AXIS,
        gaussian_weights=True,
        sigma=SSIM_SIGMA,
        use_sample_covariance=False,
    ))


_LPIPS_FN = None


def _get_lpips():
    """惰性构造 LPIPS(AlexNet)。权重与版本必须与协议一致。"""
    global _LPIPS_FN
    if _LPIPS_FN is None:
        import torch
        import lpips as lpips_pkg
        ver = getattr(lpips_pkg, '__version__', 'unknown')
        if ver != '0.1.4':
            raise RuntimeError(
                f'协议要求 lpips==0.1.4，实际 {ver}；禁止在未通过校验时出图')
        fn = lpips_pkg.LPIPS(net=LPIPS_NET, verbose=False)
        fn.eval()
        _LPIPS_FN = fn
    return _LPIPS_FN


def calc_lpips(sr: np.ndarray, hr: np.ndarray) -> float:
    """LPIPS：RGB、[-1,1]、crop_border=4、spatial=False（协议 §12.1 第 6 条）。"""
    import torch
    a, b = _crop(sr, hr)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fn = _get_lpips().to(dev)

    def _t(x):
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        t = t.unsqueeze(0).float().to(dev) * 2.0 - 1.0
        return t

    with torch.no_grad():
        v = fn(_t(a), _t(b))
    return float(v.item())


def compute_metrics(sr: np.ndarray, hr: np.ndarray,
                    with_lpips: bool = True) -> Dict[str, float]:
    """一次算完 5 项。sr/hr 必须是 (H,W,3) float32 [0,1]。"""
    assert sr.shape == hr.shape, f'尺寸不符: {sr.shape} vs {hr.shape}'
    out = {
        'psnr': calc_psnr(sr, hr),
        'ssim': calc_ssim(sr, hr),
        'rmse': calc_rmse(sr, hr),
        'mae': calc_mae(sr, hr),
    }
    # 未算 LPIPS 时写 None（JSON null），与协议 §I.1 「有限值」自检区分开：
    # null = 本次没算；NaN/Inf = 算错了。两者性质不同。
    out['lpips'] = calc_lpips(sr, hr) if with_lpips else None
    return out


def metrics_available() -> Tuple[bool, str]:
    """探测 skimage / torch / lpips 是否可用，返回 (是否全部可用, 说明)。"""
    msgs = []
    ok = True
    try:
        import skimage
        if not skimage.__version__.startswith('0.19'):
            msgs.append(f'scikit-image={skimage.__version__}（协议要求 0.19.3）')
            ok = False
    except ImportError:
        msgs.append('缺 scikit-image')
        ok = False
    try:
        import torch  # noqa: F401
    except ImportError:
        msgs.append('缺 torch（LPIPS 不可用）')
        ok = False
    try:
        import lpips
        if getattr(lpips, '__version__', None) != '0.1.4':
            msgs.append(f'lpips={getattr(lpips, "__version__", None)}（要求 0.1.4）')
            ok = False
    except ImportError:
        msgs.append('缺 lpips（LPIPS 不可用）')
        ok = False
    return ok, '; '.join(msgs) if msgs else '全部可用'


def eval_protocol_block(test_image_count: int) -> Dict[str, object]:
    """写进 per_sample_metrics.json 的 eval_protocol 块（schema 冻结，勿改字段名）。"""
    return {
        'crop_border': CROP_BORDER,
        'pixel_range': '0_1',
        'clamp_sr_to_01': True,
        'ssim_window': SSIM_WIN,
        'ssim_sigma': SSIM_SIGMA,
        'ssim_channel_axis': SSIM_CHANNEL_AXIS,
        'lpips_network': LPIPS_NET,
        'lpips_version': '0.1.4',
        'lpips_input_range': LPIPS_INPUT_RANGE,
        'test_image_count': int(test_image_count),
    }


# --------------------------------------------------------------------------
# 通用 IO
# --------------------------------------------------------------------------
def dump_json_strict(obj, path) -> Path:
    """按协议 §I.1 第 6 条：sort_keys=True，UTF-8 不转义。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    return path


def load_json(path) -> object:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def parse_number_stem(path) -> str:
    """从 `251_SR.bmp` 取出 `251`（协议 §I.1 第 2 条：样本号 = 去掉扩展名与原文件同源）。

    这里只剥掉已知的后缀（_SR/_LR/_HR），保留原文件名主干，
    以免把带下划线的真实文件名切坏。
    """
    stem = Path(path).stem
    for suf in ('_SR', '_LR', '_HR', '_sr', '_lr', '_hr'):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem


def discover_numbers(folder, pattern: str = '*.bmp') -> List[str]:
    """列出目录里的样本号，按数值排序（数值序，不是字典序）。"""
    folder = Path(folder)
    if not folder.exists():
        return []
    nums = [parse_number_stem(p) for p in folder.glob(pattern)]
    def _key(s):
        return (0, int(s)) if s.isdigit() else (1, s)
    return sorted(set(nums), key=_key)


def fmt_float(v, nd: int = 6) -> str:
    """固定有效位输出，避免不同 JSON 解析器/平台产生差异。"""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 'null'
    return f'{float(v):.{nd}f}'


def warn(msg: str) -> None:
    print(f'[warn] {msg}', file=sys.stderr)


def info(msg: str) -> None:
    print(f'[info] {msg}', file=sys.stderr)
