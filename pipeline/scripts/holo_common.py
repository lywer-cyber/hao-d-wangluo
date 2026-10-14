"""共用工具：读写、归一化、文件遍历"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import cv2

IMG_EXT = ('.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg')
# 支持的最大位深写入（>8bit 用 16bit PNG 保存，避免再次量化）
PNG_16BIT = ('.png', '.tif', '.tiff')


def imread_gray(path: str | Path) -> np.ndarray:
    """读单通道图，返回 float32。8bit -> [0,1]；16bit -> [0,1]"""
    p = str(path)
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f'读不到: {p}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.5:                       # 疑似未归一化
            img = img / max(float(img.max()), 1e-6)
    return img


def imwrite_gray(path: str | Path, img01: np.ndarray, bit_depth: int = 8) -> None:
    """写单通道图。bit_depth>8 时写 16bit，避免二次量化损失"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img01 = np.clip(np.asarray(img01, dtype=np.float32), 0.0, 1.0)
    if bit_depth > 8:
        arr = np.round(img01 * 65535.0).astype(np.uint16)
    else:
        arr = np.round(img01 * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), arr):
        raise IOError(f'写不了: {path}')


def list_images(folder: str | Path) -> list[Path]:
    """递归列出所有图片，按路径排序（保证可复现）"""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f'目录不存在: {folder}')
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXT) and not f.startswith('.'):
                out.append(Path(root) / f)
    return sorted(out)


def stem_of(path: Path) -> str:
    return path.stem
