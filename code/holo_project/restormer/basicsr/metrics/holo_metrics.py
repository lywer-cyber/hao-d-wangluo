"""Deterministic RGB [0,1] metrics for the holographic experiment.

SSIM is deliberately evaluated once per RGB channel with ``channel_axis=2``
(a singleton HWC channel slice) and then averaged.  This makes both the
channel-axis contract and the RGB reduction explicit and independent of any
version-specific multichannel aggregation behavior in scikit-image.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _to_hwc(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 4:
            image = image[0]
        image = image.numpy().transpose(1, 2, 0)
    raw = np.asarray(image)
    was_integer = np.issubdtype(raw.dtype, np.integer)
    image = raw.astype(np.float64, copy=False)
    if was_integer:
        image = image / 255.0
    if image.min(initial=0.0) < -1e-6 or image.max(initial=0.0) > 1.0 + 1e-6:
        raise ValueError("Holographic metrics require RGB pixels in [0,1]")
    return image


def _crop_pair(img1, img2, crop_border):
    img1, img2 = _to_hwc(img1), _to_hwc(img2)
    if crop_border:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
    return img1, img2


def calculate_holo_psnr(img1, img2, crop_border=4, test_y_channel=False,
                        data_range=1.0):
    if test_y_channel:
        raise ValueError('Holographic metrics are fixed to RGB; Y-channel is disabled')
    img1, img2 = _crop_pair(img1, img2, int(crop_border))
    mse = np.mean((img1 - img2) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(float(data_range) ** 2 / mse))


def calculate_holo_ssim(img1, img2, crop_border=4, test_y_channel=False,
                        data_range=1.0, win_size=11, sigma=1.5,
                        gaussian_weights=True, use_sample_covariance=False,
                        channel_axis=2, channel_reduction="mean"):
    if test_y_channel:
        raise ValueError('Holographic metrics are fixed to RGB; Y-channel is disabled')
    img1, img2 = _crop_pair(img1, img2, int(crop_border))
    if int(channel_axis) != 2:
        raise ValueError("Holographic SSIM requires channel_axis=2 for HWC RGB")
    if channel_reduction != "mean":
        raise ValueError("Holographic SSIM requires channel_reduction=mean")
    if img1.ndim != 3 or img1.shape[2] != 3:
        raise ValueError("Holographic SSIM requires an HWC RGB image")
    per_channel = [
        structural_similarity(
            img1[..., channel:channel + 1], img2[..., channel:channel + 1],
            channel_axis=2, data_range=float(data_range),
            win_size=int(win_size), sigma=float(sigma),
            gaussian_weights=bool(gaussian_weights),
            use_sample_covariance=bool(use_sample_covariance))
        for channel in range(3)
    ]
    return float(np.mean(per_channel))


def calculate_holo_rmse(img1, img2, crop_border=4, data_range=1.0):
    """Calculate RGB RMSE using the same crop/range normalization as PSNR."""
    del data_range  # The images are already normalized to [0,1].
    img1, img2 = _crop_pair(img1, img2, int(crop_border))
    return float(np.sqrt(np.mean((img1 - img2) ** 2)))


def calculate_holo_mae(img1, img2, crop_border=4, data_range=1.0):
    """Calculate RGB MAE using the same crop/range normalization as PSNR."""
    del data_range
    img1, img2 = _crop_pair(img1, img2, int(crop_border))
    return float(np.mean(np.abs(img1 - img2)))


def calculate_holo_metrics(img1, img2, *, crop_border=4, psnr_data_range=1.0,
                           ssim_data_range=1.0, ssim_window_size=11,
                           ssim_sigma=1.5, ssim_gaussian_weights=True,
                           ssim_use_sample_covariance=False,
                           ssim_channel_axis=2,
                           ssim_channel_reduction="mean"):
    """Return the shared PSNR/SSIM/RMSE/MAE implementation for val and test."""
    return {
        "PSNR": calculate_holo_psnr(
            img1, img2, crop_border=crop_border, data_range=psnr_data_range),
        "SSIM": calculate_holo_ssim(
            img1, img2, crop_border=crop_border, data_range=ssim_data_range,
            win_size=ssim_window_size, sigma=ssim_sigma,
            gaussian_weights=ssim_gaussian_weights,
            use_sample_covariance=ssim_use_sample_covariance,
            channel_axis=ssim_channel_axis,
            channel_reduction=ssim_channel_reduction),
        "RMSE": calculate_holo_rmse(img1, img2, crop_border=crop_border),
        "MAE": calculate_holo_mae(img1, img2, crop_border=crop_border),
    }

