#!/usr/bin/env python3
"""
实验总配置的加载、环境变量展开与路径解析（唯一事实来源）

【可移植】本模块不绑定任何机器：
  - 配置里所有路径都是相对路径，相对 project_root 解析
  - 需要机器相关的值用 ${VAR:-默认值} 覆盖
  - project_root 默认 = 本脚本的上两级目录（即项目根），所以只要目录结构
    <项目根>/scripts/exp_config.py 保持标准，拷到任何服务器都能直接跑

配置文件查找顺序：
  1. 环境变量 $HOLO_EXP_CONFIG
  2. <项目根>/experiment.yaml 或 <项目根>/全息超分-实验总配置.yaml
  3. 当前工作目录下同名文件
  4. ~/桌面/ 或 ~/Desktop/ 下同名文件（仅本机便利，其它机器上不存在会跳过）

用法
  python3 exp_config.py --check      # 校验 + 打印摘要（含解析后的绝对路径）
  python3 exp_config.py --path       # 打印配置文件路径
  python3 exp_config.py --paths      # 打印解析后的路径

在别的脚本里：
  from exp_config import load, paths_abs
  cfg = load()
  P = paths_abs(cfg)
  P['data'], P['basicsr_root'], cfg['training']['total_iter']
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

CONFIG_FILENAMES = ('实验总配置.yaml', 'experiment.yaml',
                    '全息超分-实验总配置.yaml')
REQUIRED_SECTIONS = ('meta', 'dataset', 'degradation', 'training',
                     'evaluation', 'models', 'paths')

_ENV_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}')


# --------------------------------------------------------------------------
# 环境变量展开
# --------------------------------------------------------------------------
def expand_env(s: str) -> str:
    """把 ${VAR} 或 ${VAR:-默认值} 展开为环境变量的值"""
    def repl(m):
        name, default = m.group(1), m.group(2)
        val = os.environ.get(name)
        if val is None or val == '':
            if default is None:
                raise KeyError(
                    f'配置引用了未设置的环境变量 ${{{name}}}，'
                    f'且没有默认值。请 export {name}=... 或在配置里给默认值')
            return default
        return val
    return _ENV_RE.sub(repl, s)


def _expand_walk(obj):
    if isinstance(obj, str):
        return expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand_walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_walk(v) for v in obj]
    return obj


# --------------------------------------------------------------------------
# 查找与加载
# --------------------------------------------------------------------------
def project_root_default() -> Path:
    """默认项目根 = 本脚本的上两级目录"""
    return Path(__file__).resolve().parent.parent


def find_config(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f'指定的配置不存在: {p}')
        return p

    env = os.environ.get('HOLO_EXP_CONFIG')
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
        print(f'[cfg] 警告: $HOLO_EXP_CONFIG 指向的 {p} 不存在，继续查找', file=sys.stderr)

    roots = [project_root_default(), Path.cwd()]
    for base in roots:
        for name in CONFIG_FILENAMES:
            p = base / name
            if p.exists():
                return p

    # 仅本机便利：桌面
    for base in (Path.home() / '桌面', Path.home() / 'Desktop'):
        for name in CONFIG_FILENAMES:
            p = base / name
            if p.exists():
                return p

    raise FileNotFoundError(
        '找不到实验总配置文件。三种解决方式任选：\n'
        f'  1) 把它放到项目根 {project_root_default()}/\n'
        '  2) export HOLO_EXP_CONFIG=<配置文件绝对路径>\n'
        '  3) 放到当前工作目录')


def load(path: str | None = None) -> dict:
    p = find_config(path)
    with open(p, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = _expand_walk(cfg)          # 展开 ${VAR:-默认}
    cfg['_config_path'] = str(p)
    return cfg


# --------------------------------------------------------------------------
# 路径解析
# --------------------------------------------------------------------------
def resolve_project_root(cfg: dict) -> Path:
    """
    项目根解析顺序：
      1. $HOLO_PROJECT_ROOT（已由 expand_env 展开进 cfg['paths']['project_root']）
      2. cfg['paths']['project_root'] 若为显式路径（不是 '.'）
      3. 本脚本的上两级目录
      4. 当前工作目录
    """
    raw = (cfg.get('paths') or {}).get('project_root')
    if raw and str(raw).strip() not in ('.', './', ''):
        p = Path(str(raw)).expanduser()
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    if os.environ.get('HOLO_PROJECT_ROOT'):
        return Path(os.environ['HOLO_PROJECT_ROOT']).expanduser().resolve()
    d = project_root_default()
    return d if d.exists() else Path.cwd()


def paths_abs(cfg: dict) -> dict:
    """把 paths 段全部解析成绝对路径"""
    root = resolve_project_root(cfg)
    out = {'project_root': str(root)}
    for k, v in (cfg.get('paths') or {}).items():
        if k == 'project_root':
            continue
        p = Path(str(v)).expanduser()
        out[k] = str(p if p.is_absolute() else (root / p).resolve())
    return out


def dataset_paths_abs(cfg: dict) -> dict:
    """dataset.hr_root 也解析成绝对路径"""
    root = resolve_project_root(cfg)
    p = Path(str(cfg['dataset']['hr_root'])).expanduser()
    return {'hr_root': str(p if p.is_absolute() else (root / p).resolve())}


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def validate(cfg: dict) -> list[str]:
    errs = []
    for s in REQUIRED_SECTIONS:
        if s not in cfg:
            errs.append(f'缺配置段: {s}')
    if errs:
        return errs

    ds, sp = cfg['dataset'], cfg['dataset']['split']
    if sp['train'] + sp['val'] + sp['test'] != ds['total']:
        errs.append(f"划分之和 {sp['train']}+{sp['val']}+{sp['test']} != total {ds['total']}")

    hr = Path(dataset_paths_abs(cfg)['hr_root'])
    if not hr.is_dir():
        errs.append(f"dataset.hr_root 不是有效目录: {hr}\n"
                    f"      → 把数据放过去，或 export HOLO_HR_ROOT=<HR 目录>")

    bad = set(cfg['degradation']['protocols']) - {'bicubic', 'physical'}
    if bad:
        errs.append(f'未知退化协议: {bad}（可选 bicubic / physical）')

    tr = cfg['training']
    if len(tr['seeds']) < 3:
        errs.append(f"training.seeds 只有 {len(tr['seeds'])} 个，论文建议至少 3 个")
    ms = tr['scheduler'].get('milestones') or []
    if ms and ms[-1] >= tr['total_iter']:
        errs.append(f"scheduler.milestones 最后一个 {ms[-1]} >= total_iter {tr['total_iter']}")
    if cfg['evaluation']['crop_border'] not in cfg['degradation']['scale']:
        errs.append(f"evaluation.crop_border={cfg['evaluation']['crop_border']} 与 "
                    f"degradation.scale={cfg['degradation']['scale']} 不一致")

    all_m = (cfg['models']['comparison'] + cfg['models']['ablation'] + cfg['models']['ours'])
    dup = sorted({m for m in all_m if all_m.count(m) > 1})
    if dup:
        errs.append(f'模型清单有重复: {dup}')

    # 基本可移植性检查：不应该出现写死的家目录
    raw = Path(cfg['_config_path']).read_text(encoding='utf-8')
    for bad_prefix in ('/home/', '/Users/', 'C:\\'):
        if bad_prefix in raw:
            errs.append(f'配置里出现机器相关绝对路径 "{bad_prefix}"，'
                        f'换服务器会失效；请改成相对路径或 ${{环境变量}}')

    return errs


# --------------------------------------------------------------------------
# 摘要
# --------------------------------------------------------------------------
def summarize(cfg: dict) -> str:
    P = paths_abs(cfg)
    D = dataset_paths_abs(cfg)
    ds, dg, tr, ev, mo = (cfg['dataset'], cfg['degradation'],
                          cfg['training'], cfg['evaluation'], cfg['models'])
    L = []
    L.append(f"配置文件   : {cfg['_config_path']}")
    L.append(f"项目根     : {P['project_root']}")
    L.append(f"实验名     : {cfg['meta']['name']}  v{cfg['meta']['version']}  ({cfg['meta']['date']})")
    L.append(f"数据集     : {ds['name']}  共 {ds['total']} 张  "
             f"train {ds['split']['train']} / val {ds['split']['val']} / test {ds['split']['test']}"
             f"   seed={ds['split_seed']}")
    L.append(f"  HR 目录  : {D['hr_root']}")
    L.append(f"组划分正则 : {ds['group_regex'] or '（无，每张独立）'}")
    L.append(f"退化协议   : {', '.join(dg['protocols'])}   scale={dg['scale']}   "
             f"bit_depth={dg['bit_depth']}")
    L.append(f"训练       : patch {tr['gt_size']}  batch {tr['batch_size_per_gpu']}  "
             f"iter {tr['total_iter']}  lr {tr['optimizer']['lr']}  "
             f"{tr['optimizer']['type']} + {tr['scheduler']['type']}")
    L.append(f"损失       : {tr['loss']['type']}")
    L.append(f"种子       : {tr['seeds']}")
    L.append(f"评估       : {', '.join(ev['metrics'])}  crop_border={ev['crop_border']}  "
             f"报数用 {ev['report_on']}")
    L.append(f"对比模型   : {', '.join(mo['comparison'])}  ({len(mo['comparison'])} 个)")
    L.append(f"消融模型   : {', '.join(mo['ablation'])}")
    L.append(f"我们的方法 : {', '.join(mo['ours'])}")
    for k in ('basicsr_root', 'data', 'splits', 'configs', 'logs'):
        if k in P:
            L.append(f"  {k:14s}: {P[k]}")
    runs = len(mo['comparison'] + mo['ablation'] + mo['ours']) \
        * len(dg['protocols']) * len(dg['scale'])
    L.append(f"实验总数   : {runs} 组（模型 × 协议 × 倍数）；"
             f"若每个跑 {len(tr['seeds'])} 个种子 = {runs * len(tr['seeds'])} 次训练")
    return '\n'.join(L)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None)
    p.add_argument('--check', action='store_true')
    p.add_argument('--path', action='store_true')
    p.add_argument('--paths', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    try:
        cfg = load(args.config)
    except (FileNotFoundError, KeyError) as e:
        print(f'错误: {e}', file=sys.stderr)
        return 2

    if args.path:
        print(cfg['_config_path'])
        return 0
    if args.paths:
        for k, v in {**paths_abs(cfg), **dataset_paths_abs(cfg)}.items():
            print(f'{k:14s} {v}')
        return 0

    print(summarize(cfg))
    errs = validate(cfg)
    print()
    if errs:
        print('校验未通过：')
        for e in errs:
            print(f'  ✗ {e}')
        return 1
    print('校验通过 ✓')
    return 0


if __name__ == '__main__':
    sys.exit(main())
