#!/usr/bin/env python3
"""
05 评估：生成论文用的对比表

指标
  PSNR / SSIM   灰度强度图，crop_border = scale（与 BasicSR 一致）
  LPIPS         感知质量，散斑重的图上 PSNR 会饱和，必须配一个感知指标

用法
  # 1) 先在各模型 configs/test/*.yml 上跑推理（BasicSR 会把结果存到 results/）
  #    cd $BASICSR_ROOT && python -m basicsr.test -opt configs/test/DAT_physical_x4.yml
  # 2) 汇总成表
  python 05_evaluate.py metrics --data data/holo --scale 4 \
      --results_root /home/zero/refs/DAT/results \
      --out_md results_table.md

  # 参数量/FLOPs 表（需要 torch）
  python 05_evaluate.py params --config_dir configs/test --model DAT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from holo_common import imread_gray  # noqa: E402


# --------------------------------------------------------------------------
# 指标实现（无 torch 也能跑的部分）
# --------------------------------------------------------------------------
def calc_psnr(img, gt, crop_border=0):
    if crop_border > 0:
        img, gt = img[crop_border:-crop_border, crop_border:-crop_border], \
                  gt[crop_border:-crop_border, crop_border:-crop_border]
    mse = float(np.mean((img.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def _gaussian_kernel(size=11, sigma=1.5):
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def calc_ssim(img, gt, crop_border=0):
    """标准 SSIM（高斯窗 11x11, σ=1.5），与 MATLAB/BasicSR 口径一致"""
    from scipy.ndimage import gaussian_filter
    if crop_border > 0:
        img, gt = img[crop_border:-crop_border, crop_border:-crop_border], \
                  gt[crop_border:-crop_border, crop_border:-crop_border]
    a, b = img.astype(np.float64), gt.astype(np.float64)
    C1, C2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    s = 1.5
    mu_a = gaussian_filter(a, s, mode='reflect')
    mu_b = gaussian_filter(b, s, mode='reflect')
    saa = gaussian_filter(a * a, s, mode='reflect') - mu_a ** 2
    sbb = gaussian_filter(b * b, s, mode='reflect') - mu_b ** 2
    sab = gaussian_filter(a * b, s, mode='reflect') - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(ssim_map.mean())


def calc_lpips_batch(pairs, device='cuda'):
    """LPIPS（需要 pip install lpips）。返回均值，或 None 表示不可用。"""
    try:
        import torch
        import lpips
    except Exception:
        print('  [warn] 没装 lpips，跳过 LPIPS（pip install lpips）')
        return None
    net = lpips.LPIPS(net='alex').to(device).eval()
    vals = []
    with torch.no_grad():
        for pred, gt in pairs:
            def to_t(x):
                t = torch.from_numpy(x.astype(np.float32))[None, None]
                return t.repeat(1, 3, 1, 1).to(device) * 2 - 1
            vals.append(float(net(to_t(pred), to_t(gt)).item()))
    return float(np.mean(vals)) if vals else None


# --------------------------------------------------------------------------
# 结果目录扫描
# --------------------------------------------------------------------------
def find_visualization_dirs(results_root: Path):
    """
    BasicSR 测试会产出 results/<name>/visualization/<dataset>/*.png
    这里把所有 <name> 找出来。
    """
    found = {}
    if not results_root.exists():
        return found
    for viz in results_root.rglob('visualization'):
        name = viz.parent.name
        imgs = [p for p in viz.rglob('*.png')]
        if imgs:
            found[name] = imgs
    return found


def parse_name(name: str):
    """从实验名里解析 模型/协议/倍数，形如 test_DAT_physical_x4"""
    m = re.match(r'(?:test_)?(.+?)_(bicubic|physical)_x(\d+)$', name)
    if not m:
        return name, '?', '?'
    return m.group(1), m.group(2), m.group(3)


def cmd_metrics(args):
    data = Path(args.data)
    hr_dir = data / 'HR' / 'test'
    results_root = Path(args.results_root)

    hr_cache = {}
    for p in hr_dir.glob('*.png'):
        hr_cache[p.stem] = imread_gray(p)
    print(f'[05] HR test: {len(hr_cache)} 张')

    viz_dirs = find_visualization_dirs(results_root)
    if not viz_dirs:
        print(f'[05] 在 {results_root} 没找到 visualization 目录。')
        print('     先在 basicsr 里跑 test，或用 --results_root 指到正确位置。')
        return 1

    rows = []
    for name, imgs in sorted(viz_dirs.items()):
        model, protocol, scale = parse_name(name)
        if args.scale and scale != str(args.scale):
            continue
        psnrs, ssims, pairs = [], [], []
        for ip in imgs:
            key = ip.stem
            # BasicSR 保存名可能带后缀，做一次宽松匹配
            if key not in hr_cache:
                cand = [k for k in hr_cache if k.startswith(key) or key.startswith(k)]
                if not cand:
                    continue
                key = cand[0]
            pred, gt = imread_gray(ip), hr_cache[key]
            if pred.shape != gt.shape:
                print(f'  [warn] {ip.name} 尺寸 {pred.shape} != GT {gt.shape}，跳过')
                continue
            psnrs.append(calc_psnr(pred, gt, args.crop_border))
            ssims.append(calc_ssim(pred, gt, args.crop_border))
            pairs.append((pred, gt))
        if not psnrs:
            continue
        lp = calc_lpips_batch(pairs, args.device) if args.lpips else None
        rows.append(dict(model=model, protocol=protocol, scale=scale,
                         n=len(psnrs),
                         psnr=float(np.mean(psnrs)), psnr_std=float(np.std(psnrs)),
                         ssim=float(np.mean(ssims)), ssim_std=float(np.std(ssims)),
                         lpips=lp))
        print(f'  {model:10s} {protocol:9s} X{scale}  '
              f'n={len(psnrs):4d}  PSNR {np.mean(psnrs):.3f}  SSIM {np.mean(ssims):.4f}'
              + (f'  LPIPS {lp:.4f}' if lp is not None else ''))

    if not rows:
        print('[05] 没有可汇总的结果')
        return 1

    # 输出 markdown 表（论文直接贴）
    lines = ['| 方法 | 退化协议 | 倍数 | PSNR (dB) | SSIM | LPIPS |',
             '|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda x: (-x['psnr'],)):
        lp = f"{r['lpips']:.4f}" if r['lpips'] is not None else '—'
        lines.append(f"| {r['model']} | {r['protocol']} | X{r['scale']} | "
                     f"{r['psnr']:.3f} ± {r['psnr_std']:.3f} | "
                     f"{r['ssim']:.4f} ± {r['ssim_std']:.4f} | {lp} |")
    table = '\n'.join(lines)
    print()
    print(table)

    if args.out_md:
        Path(args.out_md).write_text(table + '\n', encoding='utf-8')
        print(f'\n[05] 表格已写入 {args.out_md}')
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    return 0


def cmd_params(args):
    """参数量 + FLOPs 表（需要 torch）"""
    try:
        import torch
        from basicsr.utils.registry import ARCH_REGISTRY
        import yaml
        import importlib
    except Exception as e:
        print(f'[05] 需要 torch/basicsr/yaml: {e}')
        return 1

    cfg_dir = Path(args.config_dir)
    if args.basicsr_root:
        sys.path.insert(0, str(Path(args.basicsr_root)))

    print('| 模型 | 参数量 (M) | 倍数 |')
    print('|---|---|---|')
    for cfg in sorted(cfg_dir.glob('*.yml')):
        model, protocol, scale = parse_name(cfg.stem)
        if args.model and model not in args.model:
            continue
        if args.protocol and protocol != args.protocol:
            continue
        try:
            opt = yaml.safe_load(cfg.read_text(encoding='utf-8'))
            net = ARCH_REGISTRY.get(opt['network_g']['type'])(**{
                k: v for k, v in opt['network_g'].items() if k != 'type'})
            n = sum(p.numel() for p in net.parameters())
            print(f'| {model} | {n / 1e6:.3f} | X{scale} |')
        except Exception as e:
            print(f'| {model} | 加载失败: {e} | X{scale} |')
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    m = sub.add_parser('metrics')
    m.add_argument('--data', default='data/holo')
    m.add_argument('--results_root', required=True)
    m.add_argument('--scale', type=int, default=4)
    m.add_argument('--crop_border', type=int, default=None,
                   help='默认 = scale，与 BasicSR 一致')
    m.add_argument('--no_lpips', dest='lpips', action='store_false')
    m.add_argument('--device', default='cuda')
    m.add_argument('--out_md', default=None)
    m.add_argument('--out_json', default=None)
    m.set_defaults(lpips=True)

    q = sub.add_parser('params')
    q.add_argument('--config_dir', default='configs/test')
    q.add_argument('--model', nargs='+', default=None)
    q.add_argument('--protocol', default=None)
    q.add_argument('--basicsr_root', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cmd == 'metrics':
        if args.crop_border is None:
            args.crop_border = args.scale
        return cmd_metrics(args)
    return cmd_params(args)


if __name__ == '__main__':
    sys.exit(main())
