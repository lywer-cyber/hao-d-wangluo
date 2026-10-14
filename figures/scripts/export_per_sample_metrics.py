#!/usr/bin/env python3
"""
逐样本指标落盘 —— per_sample_metrics.json（协议 §I.1，schema 冻结）。

为什么需要它
  协议 §12 只产出**聚合均值**。但论文图（Fig.2 / Fig.3）上每个面板都要标该样本的
  PSNR/SSIM/LPIPS。没有逐样本文件，只能手抄 —— 违反出图规范 §A.1（禁止手抄数字）。

模式
  compute  从 SR 图目录现算（默认）
  merge    把已有的 per-run 文件合并成全局文件（--merge-from）

自检（任一不符即非零退出，协议 §I.1 硬要求）
  1. 每个模型的样本集合完全相同，且等于测试集全量
  2. 种子 10/20/30 齐全
  3. 所有数值有限（无 NaN/Inf）
  4. 逐样本聚合出的均值 == test_metrics.json 的聚合值（容差 1e-9）

用法
  python export_per_sample_metrics.py compute \
      --hr-dir   $HOLO_DATA_ROOT/HR \
      --results-root experiments \
      --models HoloRestormer SwinIR RCAN DAT HAT NAFNet SRResNet \
      --seeds 10 20 30 \
      --sr-template   "{root}/{model}/seed{seed}/test_sr/{number}_SR.bmp" \
      --plus-template "{root}/{model}/seed{seed}/test_sr_plus/{number}_SR.bmp" \
      --test-list data/splits/split_seed42.json \
      --write-per-run \
      --out figures/data/per_sample_metrics_all.json

  # 只标定 LPIPS 一致性、临时跳过 LPIPS：
  python export_per_sample_metrics.py compute ... --no-lpips
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    METRIC_IMPL_VERSION, PROTOCOL_VERSION, SEEDS, compute_metrics,
    enable_numpy_ssim_fallback,
    dump_json_strict, eval_protocol_block, imread_rgb01, info, load_json,
    metrics_available, warn,
)

AGG_TOL = 1e-6          # 协议 §I.1：聚合交叉校验容差（1e-9 会因浮点累加顺序误报）


# --------------------------------------------------------------------------
# 测试集清单
# --------------------------------------------------------------------------
def load_test_numbers(test_list: Optional[str], hr_dir: Optional[Path]) -> List[str]:
    """按优先级取测试样本号：显式清单 -> split_seed42.json 的 test_ranges -> HR 全量。"""
    if test_list:
        p = Path(test_list)
        if not p.exists():
            raise FileNotFoundError(f'找不到测试清单: {p}')
        if p.suffix.lower() == '.json':
            data = load_json(p)
            # 支持两种：{"test": [...]} 或 {"test_ranges": "251..500,4501..4750"}
            if isinstance(data, dict) and isinstance(data.get('test'), list):
                nums = [str(x) for x in data['test']]
            elif isinstance(data, dict) and data.get('test_ranges'):
                nums = _expand_ranges(str(data['test_ranges']))
            else:
                raise ValueError(f'{p} 里没有 test / test_ranges 字段')
        else:
            nums = [Path(l.strip()).stem for l in
                    p.read_text(encoding='utf-8').splitlines()
                    if l.strip() and not l.strip().startswith('#')]
        return _sorted_nums(nums)

    if hr_dir and hr_dir.exists():
        nums = [q.stem for q in hr_dir.iterdir()
                if q.suffix.lower() in ('.bmp', '.png', '.tif', '.tiff')]
        warn('未提供 --test-list，退化为使用 HR 目录全量；'
             '正式实验必须显式提供测试清单')
        return _sorted_nums(nums)
    raise ValueError('既没有 --test-list 也没有可用的 --hr-dir')


def _expand_ranges(spec: str) -> List[str]:
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '..' in part:
            a, b = part.split('..')
            out.extend(str(i) for i in range(int(a), int(b) + 1))
        else:
            out.append(part)
    return out


def _sorted_nums(nums: Sequence[str]) -> List[str]:
    def _k(s):
        return (0, int(s)) if str(s).isdigit() else (1, str(s))
    return sorted(set(str(n) for n in nums), key=_k)


def find_hr(hr_dir: Path, number: str) -> Path:
    for ext in ('.bmp', '.png', '.tif', '.tiff'):
        c = hr_dir / f'{number}{ext}'
        if c.exists():
            return c
    raise FileNotFoundError(f'{hr_dir} 下找不到 {number}')


# --------------------------------------------------------------------------
# 计算
# --------------------------------------------------------------------------
def compute_one_run(hr_dir: Path, sr_dir: Path, numbers: Sequence[str],
                    with_lpips: bool, tag: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    missing = []
    for i, n in enumerate(numbers, 1):
        sr_p = sr_dir / f'{n}_SR.bmp'
        if not sr_p.exists():
            # 容忍其它扩展名
            cand = [p for p in sr_dir.glob(f'{n}_SR.*')]
            if not cand:
                missing.append(n)
                continue
            sr_p = cand[0]
        sr = imread_rgb01(sr_p)
        hr = imread_rgb01(find_hr(hr_dir, n))
        if sr.shape != hr.shape:
            raise ValueError(
                f'[{tag}] {n}: SR {sr.shape} != HR {hr.shape}\n'
                f'  协议 §12.1 第 8 条要求测试几何与验证一致、保持原生尺寸')
        out[str(n)] = compute_metrics(sr, hr, with_lpips=with_lpips)
        if i % 100 == 0:
            info(f'  [{tag}] {i}/{len(numbers)}')
    if missing:
        warn(f'[{tag}] 缺少 {len(missing)} 张 SR: {missing[:8]}{"..." if len(missing) > 8 else ""}')
    return out


def aggregate(per_sample: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = ('psnr', 'ssim', 'lpips', 'rmse', 'mae')
    vals = {k: [] for k in keys}
    for rec in per_sample.values():
        for k in keys:
            v = rec.get(k)
            if v is not None and np.isfinite(v):
                vals[k].append(float(v))
    return {k: (float(np.mean(v)) if v else float('nan')) for k, v in vals.items()}


# --------------------------------------------------------------------------
# 交叉校验
# --------------------------------------------------------------------------
def read_ref_aggregate(path: Path, variant: str) -> Optional[Dict[str, float]]:
    """从 test_metrics.json 里读聚合值（容忍几种常见嵌套）。"""
    if not path.exists():
        return None
    data = load_json(path)
    node = data
    for key in (variant, variant.rstrip('_plus') if variant else '', 'overall', 'metrics'):
        if isinstance(node, dict) and isinstance(node.get(key), dict):
            if any(k in node[key] for k in ('psnr', 'ssim')):
                node = node[key]
                break
    if not isinstance(node, dict):
        return None
    out = {}
    for k in ('psnr', 'ssim', 'lpips', 'rmse', 'mae'):
        if isinstance(node.get(k), (int, float)):
            out[k] = float(node[k])
    return out or None


def check_against_ref(tag: str, mine: Dict[str, float], ref: Dict[str, float]) -> List[str]:
    errs = []
    for k, v in ref.items():
        if k not in mine or not np.isfinite(mine[k]):
            continue
        d = abs(mine[k] - float(v))
        if d > AGG_TOL:
            errs.append(f'{tag}.{k}: 逐样本聚合={mine[k]!r} vs '
                        f'test_metrics={v!r} (差 {d:.3e} > {AGG_TOL:g})')
    return errs


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description='逐样本指标落盘（协议 §I.1）')
    sub = ap.add_subparsers(dest='mode')

    c = sub.add_parser('compute', help='从 SR 图目录现算')
    c.add_argument('--hr-dir', required=True)
    c.add_argument('--results-root', required=True)
    c.add_argument('--models', nargs='+', required=True)
    c.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    c.add_argument('--sr-template',
                   default='{root}/{model}/seed{seed}/test_sr/{number}_SR.bmp')
    c.add_argument('--plus-template', default=None,
                   help='自集成（+）的 SR 目录模板；不提供则不算 _plus')
    c.add_argument('--test-list', default=None)
    c.add_argument('--out', default='figures/data/per_sample_metrics_all.json')
    c.add_argument('--write-per-run', action='store_true',
                   help='同时按 run 写 {root}/{model}/seed{seed}/logs/per_sample_metrics.json')
    c.add_argument('--ref-template',
                   default='{root}/{model}/seed{seed}/logs/test_metrics.json',
                   help='交叉校验用的聚合指标文件模板')
    c.add_argument('--no-check', action='store_true', help='跳过聚合交叉校验')
    c.add_argument('--no-lpips', action='store_true')
    c.add_argument('--dev-numpy-ssim', action='store_true',
                   help='仅用于没有 skimage 的机器上跑通管路；结果禁止进论文')
    c.add_argument('--token', nargs='*', default=[],
                   help='额外替换 {token0} {token1} ... 之类的占位符，形如 k=v')

    m = sub.add_parser('merge', help='把已有 per-run 文件合并成全局文件')
    m.add_argument('--from', dest='src', nargs='+', required=True,
                   help='多个 per-run per_sample_metrics.json')
    m.add_argument('--out', default='figures/data/per_sample_metrics_all.json')
    m.add_argument('--no-check', action='store_true')

    return ap.parse_args()


def _render(tmpl: str, root, model, seed, number, extras: Dict[str, str]) -> str:
    s = tmpl.replace('{root}', str(root)).replace('{model}', str(model)) \
            .replace('{seed}', str(seed)).replace('{number}', str(number))
    for k, v in extras.items():
        s = s.replace('{' + k + '}', v)
    return s


def run_compute(args) -> int:
    if args.dev_numpy_ssim:
        enable_numpy_ssim_fallback(True)
    ok, why = metrics_available()
    with_lpips = not args.no_lpips
    if not ok:
        warn(f'环境检查: {why}')
    if not with_lpips:
        warn('已指定 --no-lpips：LPIPS 将写为 null。'
             '协议 §12.1 第 6 条要求最终报告必须含 LPIPS，此模式仅供中间校验。')

    hr_dir = Path(args.hr_dir)
    root = Path(args.results_root)
    extras = {}
    for kv in args.token:
        k, _, v = kv.partition('=')
        extras[k] = v

    numbers = load_test_numbers(args.test_list, hr_dir)
    info(f'测试样本 {len(numbers)} 张')

    results: Dict[str, Dict[str, Dict[str, dict]]] = {}
    all_errs: List[str] = []

    for model in args.models:
        results[model] = {}
        for seed in args.seeds:
            results[model][str(seed)] = {}
            tag = f'{model}/seed{seed}'

            # ---- 单体 ----
            sr_dir = Path(_render(args.sr_template, root, model, seed, '0', extras)).parent
            if not sr_dir.exists():
                warn(f'[{tag}] SR 目录不存在，跳过: {sr_dir}')
                continue
            info(f'[{tag}] 单体 <- {sr_dir}')
            plain = compute_one_run(hr_dir, sr_dir, numbers, with_lpips, tag)
            if not plain:
                warn(f'[{tag}] 一张都没算出来，跳过')
                continue
            for n, rec in plain.items():
                results[model][str(seed)][str(n)] = dict(rec)

            # ---- 自集成 ----
            if args.plus_template:
                plus_dir = Path(_render(args.plus_template, root, model, seed, '0', extras)).parent
                if plus_dir.exists():
                    info(f'[{tag}] 自集成 <- {plus_dir}')
                    plus = compute_one_run(hr_dir, plus_dir, numbers, with_lpips, f'{tag}(+)')
                    for n, rec in plus.items():
                        if str(n) in results[model][str(seed)]:
                            target = results[model][str(seed)][str(n)]
                            for k, v in rec.items():
                                # None 也照写：保证单体与自集成的键集一致
                                target[k + '_plus'] = v
                else:
                    warn(f'[{tag}] 自集成目录不存在，跳过: {plus_dir}')

            # ---- 交叉校验 ----
            if not args.no_check:
                ref_path = Path(_render(args.ref_template, root, model, seed, '0', extras))
                ref = read_ref_aggregate(ref_path, 'plain')
                if ref is None:
                    warn(f'[{tag}] 未找到可用的聚合参考 {ref_path}，跳过校验')
                else:
                    all_errs += check_against_ref(tag, aggregate(plain), ref)

            # ---- per-run 落盘 ----
            if args.write_per_run:
                per_run = {
                    'protocol_version': PROTOCOL_VERSION,
                    'metric_implementation_version': METRIC_IMPL_VERSION,
                    'eval_protocol': eval_protocol_block(len(numbers)),
                    'results': {model: {str(seed): results[model][str(seed)]}},
                }
                p = Path(_render(
                    '{root}/{model}/seed{seed}/logs/per_sample_metrics.json',
                    root, model, seed, '0', extras))
                dump_json_strict(per_run, p)
                info(f'[{tag}] 已写 {p}')

    # ---- 结构自检 ----
    all_errs += structural_checks(results, numbers, args.seeds)

    if all_errs:
        print('\n自检未通过：', file=sys.stderr)
        for e in all_errs:
            print(f'  ✗ {e}', file=sys.stderr)
        return 1

    out = {
        'protocol_version': PROTOCOL_VERSION,
        'metric_implementation_version': METRIC_IMPL_VERSION,
        'eval_protocol': eval_protocol_block(len(numbers)),
        'results': results,
    }
    p = dump_json_strict(out, args.out)
    info(f'自检通过 ✓  已写出 {p}')
    return 0


def structural_checks(results: Dict, numbers: Sequence[str],
                      required_seeds: Optional[Sequence[int]] = None) -> List[str]:
    """协议 §I.1 的硬自检：样本集一致 / 种子齐全 / 数值有限。

    required_seeds 是**本次请求**的种子集合：
      - 请求了但没跑出来 -> 错误（数据缺失）
      - 请求集合不等于协议要求的 {10,20,30} -> 警告（产物不得进论文）
    """
    if required_seeds is None:
        required_seeds = list(SEEDS)
    errs: List[str] = []
    missing_vals: set = set()
    want = set(str(n) for n in numbers)

    for model, by_seed in results.items():
        for seed, per_sample in by_seed.items():
            got = set(per_sample.keys())
            if got != want:
                miss = sorted(want - got)
                extra = sorted(got - want)
                errs.append(f'{model}/seed{seed}: 样本集不符 '
                            f'(缺 {len(miss)}, 多 {len(extra)})'
                            f'{(" 缺例:" + str(miss[:5])) if miss else ""}'
                            f'{(" 多例:" + str(extra[:5])) if extra else ""}')
            for n, rec in per_sample.items():
                for k, v in rec.items():
                    if v is None:
                        # null 表示"本次未计算"（如 --no-lpips）；只警告，不算错误
                        missing_vals.add(k)
                        continue
                    if not np.isfinite(float(v)):
                        errs.append(f'{model}/seed{seed}/{n}.{k} = {v} (NaN/Inf)')

    if set(int(x) for x in required_seeds) != set(SEEDS):
        warn(f'本次只请求了种子 {sorted(int(x) for x in required_seeds)}，'
             f'协议 §14 要求 {list(SEEDS)}。'
             f'该产物属于中间结果，**不得用于论文报告**。')

    for model, by_seed in results.items():
        if not by_seed:
            errs.append(f'{model}: 没有任何可用数据')
            continue
        present = {int(s) for s in by_seed.keys()}
        missing_seeds = sorted(set(int(x) for x in required_seeds) - present)
        if missing_seeds:
            errs.append(f'{model}: 请求了但缺种子 {missing_seeds}')

    if missing_vals:
        warn('以下指标存在 null（本次未计算）：' + ', '.join(sorted(missing_vals)) +
             '\n       协议 §12.1 第 6 条要求最终报告必须含 LPIPS；'
             '正式出图前请用完整环境重跑（去掉 --no-lpips）。')

    # 跨模型：样本集必须一致（取交集比对）
    sets = {}
    for model, by_seed in results.items():
        for seed, per_sample in by_seed.items():
            sets[f'{model}/{seed}'] = set(per_sample.keys())
    if len({frozenset(v) for v in sets.values()}) > 1:
        base_k = next(iter(sets))
        base = sets[base_k]
        for k, v in sets.items():
            if v != base:
                errs.append(f'{k} 的样本集与 {base_k} 不一致')
    return errs


def run_merge(args) -> int:
    merged: Dict[str, Dict[str, Dict[str, dict]]] = {}
    n_proto = PROTOCOL_VERSION
    eval_block = None
    count = None
    for src in args.src:
        data = load_json(src)
        if data.get('protocol_version') != PROTOCOL_VERSION:
            warn(f'{src}: protocol_version={data.get("protocol_version")} '
                 f'!= {PROTOCOL_VERSION}')
        if eval_block is None:
            eval_block = data.get('eval_protocol')
        elif data.get('eval_protocol') != eval_block:
            warn(f'{src}: eval_protocol 与首个文件不一致 —— 数字可能不可比')
        c = (eval_block or {}).get('test_image_count')
        if c is not None:
            if count is None:
                count = c
            elif count != c:
                warn(f'{src}: test_image_count={c} != {count}')
        for model, by_seed in data.get('results', {}).items():
            for seed, per_sample in by_seed.items():
                merged.setdefault(model, {}).setdefault(str(seed), {}).update(per_sample)

    if not args.no_check:
        errs = structural_checks(merged, list(
            next(iter(next(iter(merged.values())).values())).keys()
        ) if merged else [])
        if errs:
            print('\n自检未通过：', file=sys.stderr)
            for e in errs:
                print(f'  ✗ {e}', file=sys.stderr)
            return 1

    out = {
        'protocol_version': n_proto,
        'metric_implementation_version': METRIC_IMPL_VERSION,
        'eval_protocol': eval_block,
        'results': merged,
    }
    p = dump_json_strict(out, args.out)
    info(f'已写出 {p}')
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == 'merge':
        return run_merge(args)
    if args.mode == 'compute':
        return run_compute(args)
    print('请指定子命令：compute 或 merge（-h 看帮助）', file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
