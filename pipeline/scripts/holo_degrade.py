"""
数字全息强度图 —— LR 退化算子

两套协议，所有模型共用同一套（关键：测试集退化必须对所有模型完全一致）：

  D1  bicubic      经典 SR 协议，与文献（RCAN/SwinIR 基准）可比
  D2  physical     物理退化，域内贡献，含衍射受限 + 抗混叠 + 散斑 + 探测器噪声

约定
  - 输入输出均为 float32, 单通道, 范围 [0, 1]
  - HR 是重建后的强度图
  - scale 为整数放大倍数 (2/3/4)
"""

from __future__ import annotations

import numpy as np
import cv2


# --------------------------------------------------------------------------
# D2 物理退化的默认参数（可全部消融）
# --------------------------------------------------------------------------
DEFAULT_PHYS = dict(
    # --- 光学部分 ---
    diffraction_kernel='circle',   # 'circle' | 'square' | 'none'  孔径形状
    diffraction_cutoff=1.0,        # 截止频率，单位 = LR 的 Nyquist（1.0 = 恰好 LR 带宽）
    aa_sigma=0.50,                 # 抗混叠高斯 σ，单位 = LR 像素
    # --- 散斑 ---
    speckle_enable=True,
    speckle_contrast=0.60,         # 0 = 无散斑，1 = 完全发展散斑 (σ/⟨I⟩ = 1)
    speckle_corr=4.0,              # 散斑粒度（相关长度），单位 = LR 像素
    # --- 探测器 ---
    poisson_gain=2000.0,           # 光子数尺度，越大越干净；None = 关闭
    read_std=0.005,                # 归一化读出噪声 σ；0 = 关闭
    # --- 量化 ---
    bit_depth=8,
)

# 训练集用的随机化范围（每个样本独立抖动，提升泛化）
RANDOM_RANGES = dict(
    diffraction_cutoff=(0.85, 1.00),
    aa_sigma=(0.30, 0.70),
    speckle_contrast=(0.30, 0.80),
    speckle_corr=(3.0, 6.0),
    poisson_gain=(800.0, 4000.0),
    read_std=(0.003, 0.010),
)


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _check(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    assert img.ndim == 2, f'期望单通道 (H, W)，得到 {img.shape}'
    return img


def _to_uint(img01: np.ndarray, bit_depth: int) -> np.ndarray:
    """[0,1] float -> 量化整数（返回 float 仍是 [0,1]，但取值已离散）"""
    levels = (1 << bit_depth) - 1
    return np.clip(np.round(img01 * levels), 0, levels) / levels


# --------------------------------------------------------------------------
# 退化算子
# --------------------------------------------------------------------------
def fft_lowpass(img: np.ndarray, cutoff: float, kernel: str = 'circle') -> np.ndarray:
    """
    衍射受限低通。cutoff 单位 = cycles/pixel（HR 网格）。

    物理含义：成像系统的截止频率由 NA / 波长决定。低分辨采集等价于
    把物场带宽截断到 LR 的 Nyquist。
    """
    if kernel == 'none' or cutoff >= 0.5:
        return img
    H, W = img.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    if kernel == 'circle':
        mask = np.sqrt(fy ** 2 + fx ** 2) <= cutoff
    elif kernel == 'square':
        mask = (np.abs(fy) <= cutoff) & (np.abs(fx) <= cutoff)
    else:
        raise ValueError(f'未知孔径形状: {kernel}')
    F = np.fft.fft2(img.astype(np.float32)) * mask
    return np.real(np.fft.ifft2(F)).astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """抗混叠低通"""
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)


def downsample(img: np.ndarray, scale: int) -> np.ndarray:
    """面积平均降采样（等价于盒式积分，保能量）"""
    H, W = img.shape
    return cv2.resize(img, (W // scale, H // scale), interpolation=cv2.INTER_AREA)


def add_speckle(img: np.ndarray, contrast: float, corr_px: float,
                rng: np.random.Generator) -> np.ndarray:
    """
    乘性散斑噪声。

    contrast = σ_I / ⟨I⟩。完全发展散斑该值为 1，对应强度服从负指数分布。
    实现：复高斯随机场 → 相关化 → |·|² 取强度 → 按 contrast 混合。

    注：这里是退化模型意义上的散斑（在重建强度图上施加），
    不等价于全息图层面散斑的完整物理仿真，论文中需如实说明。
    """
    if contrast <= 0:
        return img
    H, W = img.shape
    z = (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W))).astype(np.complex64)

    # 相关化：对实部虚部分别做高斯低通，相关长度 ~ corr_px
    sigma = max(corr_px, 0.5) / 2.0
    zr = gaussian_blur(z.real.astype(np.float32), sigma)
    zi = gaussian_blur(z.imag.astype(np.float32), sigma)
    z = zr + 1j * zi

    inten = (z.real ** 2 + z.imag ** 2).astype(np.float32)
    m = float(inten.mean())
    if m <= 0:
        return img
    inten /= m                       # 均值 1，完全发展时 std ≈ 1
    speckle = 1.0 + contrast * (inten - 1.0)
    return (img * speckle).astype(np.float32)


def add_poisson_gaussian(img: np.ndarray, gain: float, read_std: float,
                         rng: np.random.Generator) -> np.ndarray:
    """
    泊松散粒噪声 + 高斯读出噪声。
    gain 为满量程对应的光子数，越小噪声越大。
    """
    out = img
    if gain is not None and gain > 0:
        out = rng.poisson(np.clip(out, 0, None) * gain).astype(np.float32) / gain
    if read_std and read_std > 0:
        out = out + rng.normal(0.0, read_std, out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------
# 两套协议
# --------------------------------------------------------------------------
def degrade_bicubic(hr: np.ndarray, scale: int) -> np.ndarray:
    """D1：MATLAB 风格 bicubic（含抗混叠）——经典 SR 协议"""
    hr = _check(hr)
    H, W = hr.shape
    lr = cv2.resize(hr, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)
    return np.clip(lr, 0.0, 1.0).astype(np.float32)


def degrade_physical(hr: np.ndarray, scale: int,
                     cfg: dict | None = None,
                     rng: np.random.Generator | None = None,
                     randomize: bool = False,
                     ranges: dict | None = None) -> np.ndarray:
    """
    D2：物理退化。
    randomize=True 时按 ranges（默认 RANDOM_RANGES）逐样本抖动参数（训练集用）；
    randomize=False 时严格按 cfg 固定（验证/测试集用，保证所有模型一致）。
    ranges 由实验总配置的 degradation.randomize_ranges 传入。
    """
    hr = _check(hr)
    cfg = dict(DEFAULT_PHYS if cfg is None else {**DEFAULT_PHYS, **cfg})
    rng = np.random.default_rng(0) if rng is None else rng

    if randomize:
        ranges = RANDOM_RANGES if ranges is None else ranges
        for k, (lo, hi) in ranges.items():
            cfg[k] = float(rng.uniform(lo, hi))

    # 1. 衍射受限低通：截止频率 = LR Nyquist × diffraction_cutoff
    #    LR 的 Nyquist 在 HR 网格上对应 1/(2·scale) cycles/pixel
    cutoff = cfg['diffraction_cutoff'] / (2.0 * scale)
    x = fft_lowpass(hr, cutoff, cfg['diffraction_kernel'])

    # 2. 抗混叠低通
    x = gaussian_blur(x, cfg['aa_sigma'])

    # 3. 降采样
    x = downsample(x, scale)

    # 4. 散斑（LR 尺度上施加）
    if cfg['speckle_enable']:
        x = add_speckle(x, cfg['speckle_contrast'], cfg['speckle_corr'], rng)

    # 5. 探测器噪声
    x = add_poisson_gaussian(x, cfg['poisson_gain'], cfg['read_std'], rng)

    # 6. 量化
    x = _to_uint(x, cfg['bit_depth'])
    return x.astype(np.float32)


PROTOCOLS = {
    'bicubic': degrade_bicubic,
    'physical': degrade_physical,
}


def degrade(hr: np.ndarray, scale: int, protocol: str = 'bicubic',
            cfg: dict | None = None, rng: np.random.Generator | None = None,
            randomize: bool = False, ranges: dict | None = None) -> np.ndarray:
    """统一入口"""
    if protocol == 'bicubic':
        return degrade_bicubic(hr, scale)
    if protocol == 'physical':
        return degrade_physical(hr, scale, cfg, rng, randomize, ranges)
    raise ValueError(f'未知协议: {protocol}（可选 {list(PROTOCOLS)}）')
