#!/usr/bin/env python3
"""
01 数据集划分：5000 张 HR -> train 4000 / val 500 / test 500

关键：默认按 **组** 划分，不是按张。若 5000 张里有部分来自同一样本/切片/场景，
按张随机划分会把同源图同时放进训练和测试，指标虚高 —— 这是审稿人必问的点。

用法
  # 无分组信息（每张独立）
  python 01_split_dataset.py --hr_root data/HR --out data/splits --total 5000

  # 文件名形如 S012_img03.png，按 S012 分组划分
  python 01_split_dataset.py --hr_root data/HR --out data/splits \
      --group-regex '^(S\\d+)'

产出
  data/splits/train.txt / val.txt / test.txt   （每行一个相对 HR 路径）
  data/splits/split_report.json                （数量、组数、校验信息）
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holo_common import list_images  # noqa: E402

try:
    from exp_config import load as load_exp_cfg, paths_abs, dataset_paths_abs
    _HAS_EXP_CFG = True
except Exception:
    _HAS_EXP_CFG = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='实验总配置文件路径（默认自动查找）')
    p.add_argument('--hr_root', default=None, help='默认取总配置的 dataset.hr_root')
    p.add_argument('--out', default=None, help='默认取总配置的 paths.splits')
    p.add_argument('--total', type=int, default=None)
    p.add_argument('--n_train', type=int, default=None)
    p.add_argument('--n_val', type=int, default=None)
    p.add_argument('--n_test', type=int, default=None)
    p.add_argument('--group-regex', default=None,
                   help="从文件名提取组 ID 的正则（含一个捕获组）。不给则每张独立")
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--dry_run', action='store_true', help='只报告不写文件')
    return p.parse_args()


def main():
    args = parse_args()

    # ---- 读实验总配置（唯一事实来源）----
    cfg = None
    if _HAS_EXP_CFG:
        try:
            cfg = load_exp_cfg(args.config)
            print(f"[01] 读取总配置: {cfg['_config_path']}")
        except FileNotFoundError as e:
            print(f'[01] 警告: {e}')

    if cfg:
        ds = cfg['dataset']
        args.hr_root = args.hr_root or dataset_paths_abs(cfg)['hr_root']
        args.out = args.out or paths_abs(cfg)['splits']
        args.total = args.total or ds['total']
        args.n_train = args.n_train or ds['split']['train']
        args.n_val = args.n_val or ds['split']['val']
        args.n_test = args.n_test or ds['split']['test']
        args.seed = ds['split_seed'] if args.seed is None else args.seed
        if args.group_regex is None:
            args.group_regex = ds.get('group_regex')

    args.hr_root = args.hr_root or 'data/HR'
    args.out = args.out or 'data/splits'
    args.total = args.total or 5000
    args.n_train = args.n_train or 4000
    args.n_val = args.n_val if args.n_val is not None else 500
    args.n_test = args.n_test if args.n_test is not None else 500
    args.seed = 42 if args.seed is None else args.seed

    root = Path(args.hr_root).expanduser().resolve()
    imgs = list_images(root)
    print(f'[01] 在 {root} 找到 {len(imgs)} 张图')

    # 图不够时按比例缩放划分（支持拿小子集调试，避免 train 被切成 0 张）
    if len(imgs) < args.total:
        k = len(imgs) / args.total
        args.n_train = max(1, int(round(args.n_train * k)))
        args.n_val = max(1, int(round(args.n_val * k)))
        args.n_test = max(1, int(round(args.n_test * k)))
        args.total = args.n_train + args.n_val + args.n_test
        print(f'[01] 警告：图少于预期，已等比缩放为 '
              f'train {args.n_train} / val {args.n_val} / test {args.n_test}')

    assert args.n_train + args.n_val + args.n_test == args.total, \
        f'划分之和 {args.n_train + args.n_val + args.n_test} != total {args.total}'

    # 组划分
    groups: dict[str, list[Path]] = defaultdict(list)
    pat = re.compile(args.group_regex) if args.group_regex else None
    n_unmatched = 0
    for p in imgs:
        if pat:
            m = pat.search(p.stem)
            if m:
                gid = m.group(1)
            else:
                n_unmatched += 1
                gid = p.stem            # 兜底：自成一组
        else:
            gid = p.stem
        groups[gid].append(p)

    if pat and n_unmatched:
        print(f'[01] 注意：{n_unmatched} 个文件名未匹配分组正则，已各自单独成组')

    gids = sorted(groups)
    random.Random(args.seed).shuffle(gids)

    print(f'[01] 共 {len(gids)} 个组，'
          f'每组平均 {len(imgs) / len(gids):.2f} 张')

    # 先定 val/test 的组数（按张数比例反推），再切
    n_val_pat = args.n_val / args.total
    n_test_pat = args.n_test / args.total
    n_val_g = max(1, round(len(gids) * n_val_pat))
    n_test_g = max(1, round(len(gids) * n_test_pat))

    val_g = gids[:n_val_g]
    test_g = gids[n_val_g:n_val_g + n_test_g]
    train_g = gids[n_val_g + n_test_g:]

    def collect(gs):
        out = [p for g in gs for p in groups[g]]
        return sorted(out)

    train, val, test = collect(train_g), collect(val_g), collect(test_g)

    def rel(ps):
        return [str(p.relative_to(root)) for p in ps]

    report = {
        'hr_root': str(root),
        'total_images': len(imgs),
        'total_groups': len(gids),
        'split_by': 'group' if pat else 'image',
        'seed': args.seed,
        'train': {'n_images': len(train), 'n_groups': len(train_g)},
        'val': {'n_images': len(val), 'n_groups': len(val_g)},
        'test': {'n_images': len(test), 'n_groups': len(test_g)},
    }

    print(f"[01] train {len(train)} 张 / {len(train_g)} 组")
    print(f"[01] val   {len(val)} 张 / {len(val_g)} 组")
    print(f"[01] test  {len(test)} 张 / {len(test_g)} 组")
    print(f"[01] 合计 {len(train) + len(val) + len(test)} 张")

    # 泄漏自检
    s_tr, s_va, s_te = set(train_g), set(val_g), set(test_g)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), '组划分发生泄漏！'
    assert len(set(train) & set(test)) == 0, '训练/测试图片重叠！'
    print('[01] 泄漏自检通过')

    if args.dry_run:
        print('[01] dry_run，未写文件')
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in (('train', train), ('val', val), ('test', test)):
        (out / f'{name}.txt').write_text('\n'.join(rel(items)) + '\n', encoding='utf-8')
    (out / 'split_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[01] 已写入 {out}/{{train,val,test}}.txt + split_report.json')


if __name__ == '__main__':
    main()
