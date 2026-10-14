#!/usr/bin/env python3
"""
03 数据集自检与统计

在开跑训练前跑一遍，确认：
  - HR / LR 一一配对，无缺漏、无多余
  - LR 尺寸 = HR / scale
  - 三个划分之间无重叠（防泄漏）
  - 输出论文可直接用的数据统计表 + 退化参数表

用法
  python 03_verify_dataset.py --data data/holo --scale 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holo_common import imread_gray  # noqa: E402

try:
    from exp_config import load as load_exp_cfg, paths_abs
    _HAS_EXP_CFG = True
except Exception:
    _HAS_EXP_CFG = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='实验总配置文件路径（默认自动查找）')
    p.add_argument('--data', default=None, help='默认取总配置的 paths.data')
    p.add_argument('--scale', type=int, default=None)
    p.add_argument('--protocols', nargs='+', default=None)
    p.add_argument('--stats_sample', type=int, default=50, help='统计强度分布时抽样的张数')
    p.add_argument('--out_json', default=None, help='把统计写到 json')
    return p.parse_args()


def main():
    args = parse_args()

    if _HAS_EXP_CFG:
        try:
            cfg = load_exp_cfg(args.config)
            print(f"[03] 读取总配置: {cfg['_config_path']}")
            args.data = args.data or paths_abs(cfg)['data']
            args.scale = args.scale or cfg['degradation']['scale'][0]
            args.protocols = args.protocols or list(cfg['degradation']['protocols'])
        except FileNotFoundError as e:
            print(f'[03] 警告: {e}')

    args.data = args.data or 'data/holo'
    args.scale = args.scale or 4
    args.protocols = args.protocols or ['bicubic', 'physical']

    data = Path(args.data)
    ok = True

    def chk(name, cond, extra=''):
        nonlocal ok
        print(f'  {"PASS" if cond else "FAIL"}  {name}{(" | " + extra) if extra else ""}')
        ok &= bool(cond)

    summary = {'scale': args.scale, 'splits': {}}

    for split in ('train', 'val', 'test'):
        gt_dir = data / 'HR' / split
        print(f'\n[{split}]')
        if not gt_dir.exists():
            chk(f'HR/{split} 存在', False)
            continue
        gts = sorted(p.stem for p in gt_dir.glob('*.png'))
        chk(f'HR/{split} 有图', len(gts) > 0, f'{len(gts)} 张')

        rec = {'n_gt': len(gts), 'protocols': {}}
        for proto in args.protocols:
            lq_dir = data / f'LR_{proto}' / f'X{args.scale}' / split
            if not lq_dir.exists():
                chk(f'LR_{proto}/X{args.scale}/{split} 存在', False)
                continue
            lqs = sorted(p.stem for p in lq_dir.glob('*.png'))
            chk(f'{proto} 配对完整', set(lqs) == set(gts),
                f'LR {len(lqs)} / HR {len(gts)}，'
                f'缺 {len(set(gts) - set(lqs))}，多 {len(set(lqs) - set(gts))}')

            # 尺寸与统计（抽样）
            idx = np.linspace(0, len(lqs) - 1, min(args.stats_sample, len(lqs))).astype(int)
            shapes, means, stds = set(), [], []
            for i in idx:
                hr = imread_gray(gt_dir / f'{gts[i]}.png')
                lr = imread_gray(lq_dir / f'{lqs[i]}.png')
                shapes.add((lr.shape[0] * args.scale, lr.shape[1] * args.scale))
                means.append(float(lr.mean()))
                stds.append(float(lr.std()))
            chk(f'{proto} LR 尺寸 = HR/scale', all(s == hr.shape for s in shapes),
                f'LR 上采样后 {sorted(shapes)[0]} vs HR {hr.shape}')
            rec['protocols'][proto] = {
                'n_lq': len(lqs),
                'mean': round(float(np.mean(means)), 4),
                'std': round(float(np.mean(stds)), 4),
            }
            print(f'        {proto}: 均值 {np.mean(means):.4f}  标准差 {np.mean(stds):.4f}')

        summary['splits'][split] = rec

    # 划分泄漏检查
    print('\n[划分]')
    stems = {}
    for split in ('train', 'val', 'test'):
        d = data / 'HR' / split
        stems[split] = set(p.stem for p in d.glob('*.png')) if d.exists() else set()
    chk('train ∩ test 为空', not (stems['train'] & stems['test']),
        f'{len(stems["train"] & stems["test"])} 个重叠')
    chk('train ∩ val 为空', not (stems['train'] & stems['val']))
    chk('val ∩ test 为空', not (stems['val'] & stems['test']))

    # 退化参数表（论文里要写清楚）
    cfg_file = data / 'degrade_config.json'
    if cfg_file.exists():
        print('\n[退化参数]（论文中直接引用）')
        cfg = json.loads(cfg_file.read_text(encoding='utf-8'))
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        summary['degrade_config'] = cfg

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'\n[03] 统计已写入 {args.out_json}')

    print()
    print('[03] 全部通过，可以开跑训练' if ok else '[03] 存在失败项')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
