#!/usr/bin/env python3
"""
管线自检（不需要 torch / GPU）

造一批仿全息强度图 -> 跑 01 划分 -> 跑 02 退化 -> 校验尺寸/统计/可复现性。
在正式跑 5000 张之前先跑这个，能挡住 90% 的低级错误。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holo_common import imread_gray, imwrite_gray  # noqa: E402
from holo_degrade import degrade, DEFAULT_PHYS, fft_lowpass  # noqa: E402

HERE = Path(__file__).resolve().parent


def synth_intensity(H=256, W=256, seed=0):
    """合成一张"仿全息强度图"：低频结构 + 高频条纹 + 散斑"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] / float(max(H, W))
    base = 0.5 + 0.25 * np.sin(2 * np.pi * 3 * xx) * np.cos(2 * np.pi * 2 * yy)
    # 高频细节（SR 要恢复的就是这部分）
    detail = 0.08 * np.sin(2 * np.pi * 24 * xx + 1.1) * np.sin(2 * np.pi * 19 * yy)
    grain = rng.normal(0, 0.03, (H, W))
    img = np.clip(base + detail + grain, 0, 1).astype(np.float32)
    return img


def make_fake_dataset(root: Path, n: int = 24, size: int = 256):
    for i in range(n):
        # 两两同组：S000 有 S000_a / S000_b，用来验证分组划分
        g = f'S{i // 2:03d}'
        imwrite_gray(root / f'{g}_v{i % 2}.png', synth_intensity(size, size, seed=i))


def check(name, cond, extra=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}{(" | " + extra) if extra else ""}')
    return cond


def main():
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix='holo_smoke_'))
    try:
        hr_root = tmp / 'HR'
        splits = tmp / 'splits'
        data = tmp / 'holo'
        make_fake_dataset(hr_root, n=24, size=256)

        print('[smoke] 1) 退化算子单元检查')
        hr = synth_intensity(256, 256, seed=1)
        for scale in (2, 3, 4):
            for proto in ('bicubic', 'physical'):
                rng = np.random.default_rng(0)
                lr = degrade(hr, scale, proto, DEFAULT_PHYS, rng)
                expect = (256 // scale, 256 // scale)
                ok &= check(f'{proto} X{scale} 尺寸', lr.shape == expect,
                            f'{lr.shape} vs {expect}')
                ok &= check(f'{proto} X{scale} 值域',
                            float(lr.min()) >= 0 and float(lr.max()) <= 1,
                            f'[{lr.min():.3f}, {lr.max():.3f}]')

        print('[smoke] 2) 可复现性（同种子两次结果必须逐像素相同）')
        for proto in ('bicubic', 'physical'):
            a = degrade(hr, 4, proto, DEFAULT_PHYS, np.random.default_rng(7))
            b = degrade(hr, 4, proto, DEFAULT_PHYS, np.random.default_rng(7))
            ok &= check(f'{proto} 确定性', np.array_equal(a, b))

        print('[smoke] 3) 衍射低通必须真的压掉高于截止频率的成分')
        yy, xx = np.mgrid[0:256, 0:256]
        hi = (0.5 + 0.5 * np.sin(2 * np.pi * 40 * xx / 256)).astype(np.float32)  # 高频
        lo = (0.5 + 0.5 * np.sin(2 * np.pi * 3 * xx / 256)).astype(np.float32)   # 低频
        cutoff = 1.0 / (2 * 4)          # X4 的 LR Nyquist
        hi_f = fft_lowpass(hi, cutoff)
        lo_f = fft_lowpass(lo, cutoff)
        ok &= check('高频被显著衰减',
                    float(hi_f.std()) < 0.5 * float(hi.std()),
                    f'std {hi.std():.4f} -> {hi_f.std():.4f}')
        ok &= check('低频基本保留',
                    float(lo_f.std()) > 0.9 * float(lo.std()),
                    f'std {lo.std():.4f} -> {lo_f.std():.4f}')

        print('[smoke] 3b) 关掉散斑/噪声后，物理退化的高频应低于 bicubic')
        clean = dict(DEFAULT_PHYS, speckle_enable=False,
                     poisson_gain=None, read_std=0.0)
        bc = degrade(hr, 4, 'bicubic')
        ph = degrade(hr, 4, 'physical', clean, np.random.default_rng(0))

        def hf_energy(x):
            gx = np.diff(x, axis=1)
            gy = np.diff(x, axis=0)
            return float(np.mean(gx ** 2) + np.mean(gy ** 2))

        ok &= check('physical 高频能量 < bicubic',
                    hf_energy(ph) < hf_energy(bc),
                    f'{hf_energy(ph):.6f} vs {hf_energy(bc):.6f}')

        print('[smoke] 3c) 开启散斑/噪声后，两套协议结果必须不同')
        ph_n = degrade(hr, 4, 'physical', DEFAULT_PHYS, np.random.default_rng(0))
        ok &= check('physical != bicubic', not np.allclose(ph_n, bc))

        print('[smoke] 4) 随机化模式（训练集）参数应逐样本变化')
        l1 = degrade(hr, 4, 'physical', DEFAULT_PHYS, np.random.default_rng(1), randomize=True)
        l2 = degrade(hr, 4, 'physical', DEFAULT_PHYS, np.random.default_rng(2), randomize=True)
        ok &= check('randomize 两次结果不同', not np.array_equal(l1, l2))

        print('[smoke] 5) 跑 01 划分脚本（按组划分）')
        r = subprocess.run([sys.executable, str(HERE / '01_split_dataset.py'),
                            '--hr_root', str(hr_root), '--out', str(splits),
                            '--total', '24', '--n_train', '16', '--n_val', '4', '--n_test', '4',
                            '--group-regex', r'^(S\d+)', '--seed', '42'],
                           capture_output=True, text=True)
        ok &= check('01 退出码为 0', r.returncode == 0, r.stderr[-300:] if r.returncode else '')
        for f in ('train.txt', 'val.txt', 'test.txt', 'split_report.json'):
            ok &= check(f'产出 {f}', (splits / f).exists())

        print('[smoke] 6) 跑 02 生成 LR 脚本')
        r = subprocess.run([sys.executable, str(HERE / '02_generate_lr.py'),
                            '--hr_root', str(hr_root), '--splits', str(splits),
                            '--out', str(data), '--scale', '4',
                            '--protocols', 'bicubic', 'physical'],
                           capture_output=True, text=True)
        ok &= check('02 退出码为 0', r.returncode == 0, r.stderr[-300:] if r.returncode else '')

        for proto in ('bicubic', 'physical'):
            for split in ('train', 'val', 'test'):
                d = data / f'LR_{proto}' / 'X4' / split
                n = len(list(d.glob('*.png'))) if d.exists() else -1
                ok &= check(f'LR_{proto}/X4/{split} 有文件', n > 0, f'{n} 张')

        print('[smoke] 7) LR 尺寸必须正好是 HR 的 1/4')
        sample = sorted((data / 'LR_physical' / 'X4' / 'test').glob('*.png'))[0]
        lr = imread_gray(sample)
        ok &= check('LR 尺寸 = 64x64', lr.shape == (64, 64), str(lr.shape))

        print('[smoke] 8) 同一样本的两张图必须落在同一划分（分组划分生效）')
        import json
        rep = json.loads((splits / 'split_report.json').read_text(encoding='utf-8'))
        ok &= check('按组划分', rep['split_by'] == 'group', rep['split_by'])
        tr = set((splits / 'train.txt').read_text().split())
        te = set((splits / 'test.txt').read_text().split())
        ok &= check('无图片交叉', not (tr & te))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print('[smoke] 全部通过' if ok else '[smoke] 存在失败项，请修好再跑正式数据')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
