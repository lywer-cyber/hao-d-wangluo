#!/usr/bin/env python3
"""
02 生成 LR

所有模型共用同一套 LR —— 这是"统一实验协议"的核心：
  train 集：物理参数逐样本随机抖动（提升泛化）
  val/test 集：参数固定、随机种子固定（保证所有模型面对的退化完全一致）

用法
  python 02_generate_lr.py --hr_root data/HR --splits data/splits \
      --out data/holo --scale 4 --protocols bicubic physical

产出
  data/holo/LR_bicubic/X4/{train,val,test}/*.png
  data/holo/LR_physical/X4/{train,val,test}/*.png
  data/holo/degrade_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holo_common import imread_gray, imwrite_gray  # noqa: E402
from holo_degrade import DEFAULT_PHYS, RANDOM_RANGES, degrade  # noqa: E402

try:
    from exp_config import load as load_exp_cfg
    _HAS_EXP_CFG = True
except Exception:
    _HAS_EXP_CFG = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='实验总配置文件路径（默认自动查找）')
    p.add_argument('--hr_root', default=None, help='默认取总配置的 dataset.hr_root')
    p.add_argument('--splits', default=None, help='默认取总配置的 paths.splits')
    p.add_argument('--out', default=None, help='默认取总配置的 paths.data')
    p.add_argument('--scale', type=int, nargs='+', default=None)
    p.add_argument('--protocols', nargs='+', default=None)
    p.add_argument('--bit_depth', type=int, default=None)
    p.add_argument('--limit', type=int, default=None, help='调试：每个划分只处理 N 张')
    return p.parse_args()


def image_seed(name: str, base: int = 0) -> int:
    """由文件名派生稳定种子，保证同一张图每次抖动结果一致（可复现）"""
    return (zlib.crc32(name.encode('utf-8')) + base) % (2 ** 31)


def main():
    args = parse_args()

    # ---- 读实验总配置（唯一事实来源）----
    cfg = None
    if _HAS_EXP_CFG:
        try:
            cfg = load_exp_cfg(args.config)
            print(f"[02] 读取总配置: {cfg['_config_path']}")
        except FileNotFoundError as e:
            print(f'[02] 警告: {e}')
            print('[02] 回落到脚本内置默认退化参数 —— 这可能与其它实验不一致！')

    if cfg:
        dg = cfg['degradation']
        args.hr_root = args.hr_root or cfg['dataset']['hr_root']
        args.splits = args.splits or cfg['paths']['splits']
        args.out = args.out or cfg['paths']['data']
        args.scale = args.scale or list(dg['scale'])
        args.protocols = args.protocols or list(dg['protocols'])
        args.bit_depth = args.bit_depth or dg.get('bit_depth', 8)
    args.hr_root = args.hr_root or 'data/HR'
    args.splits = args.splits or 'data/splits'
    args.out = args.out or 'data/holo'
    args.scale = args.scale or [4]
    args.protocols = args.protocols or ['bicubic', 'physical']
    args.bit_depth = args.bit_depth or 8

    # val/test 固定参数；train 抖动范围 —— 均来自总配置
    if cfg:
        fixed_cfg = {**DEFAULT_PHYS, **cfg['degradation']['fixed']}
        ranges = dict(cfg['degradation']['randomize_ranges'])
    else:
        fixed_cfg = dict(DEFAULT_PHYS)
        ranges = dict(RANDOM_RANGES)

    hr_root = Path(args.hr_root).expanduser().resolve()
    splits = Path(args.splits)
    out_root = Path(args.out)

    # 写进 json，论文里直接引用
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'degrade_config.json').write_text(
        json.dumps({'source_config': cfg['_config_path'] if cfg else None,
                    'protocols': args.protocols, 'scale': args.scale,
                    'bit_depth': args.bit_depth,
                    'fixed': fixed_cfg,
                    'randomize_ranges': ranges},
                   indent=2, ensure_ascii=False), encoding='utf-8')

    for split in ('train', 'val', 'test'):
        list_file = splits / f'{split}.txt'
        if not list_file.exists():
            print(f'[02] 跳过 {split}：找不到 {list_file}')
            continue
        names = [l.strip() for l in list_file.read_text(encoding='utf-8').splitlines() if l.strip()]
        if args.limit:
            names = names[:args.limit]
        is_train = (split == 'train')

        # 把 HR 归拢到按划分的目录，便于 BasicSR 用 paired_paths_from_folder 直接配对
        hr_dst = out_root / 'HR' / split
        hr_dst.mkdir(parents=True, exist_ok=True)
        for rel in names:
            src, dst = hr_root / rel, hr_dst / f'{Path(rel).stem}.png'
            if dst.exists():
                continue
            try:
                dst.hardlink_to(src)          # 硬链接：不占额外空间
            except OSError:
                import shutil
                shutil.copy2(src, dst)        # 跨设备时退回复制


        for protocol in args.protocols:
            for scale in args.scale:
                dst = out_root / f'LR_{protocol}' / f'X{scale}' / split
                dst.mkdir(parents=True, exist_ok=True)
                print(f'[02] {protocol} X{scale} {split}: {len(names)} 张 -> {dst}')

                for i, rel in enumerate(names):
                    src = hr_root / rel
                    hr = imread_gray(src)
                    rng = np.random.default_rng(image_seed(rel, scale if is_train else 0))
                    lr = degrade(hr, scale, protocol, fixed_cfg, rng, randomize=is_train)
                    imwrite_gray(dst / f'{Path(rel).stem}.png', lr, args.bit_depth)
                    if (i + 1) % 500 == 0:
                        print(f'      {i + 1}/{len(names)}')

    print('[02] 完成')


if __name__ == '__main__':
    main()
