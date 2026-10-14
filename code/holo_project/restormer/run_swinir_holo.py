"""Prepare/run the v7 SwinIR digital-holographic comparison experiment.

All non-network settings come from ``10.11/experiment_config_swinir_v7.txt``.
This thin runner reuses the existing BasicSR option generator, replacing only
the network name/type/parameters declared in that config.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from run_holo_trials import (make_option, prepare_splits, read_shared_config)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--project-root', type=Path, default=None)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--iters', type=int, default=None,
                        help='override only with the frozen v7 total_iterations')
    parser.add_argument('--seeds', default='10')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--num-gpu', type=int, default=1)
    parser.add_argument('--micro-batch-size', type=int, default=None)
    return parser.parse_args()


def network_params(cfg):
    network = cfg['network']
    network_type = network['network_type'].strip()
    if network_type != 'HoloSwinIR':
        raise ValueError(
            'experiment_config.txt [network].network_type must be HoloSwinIR')
    try:
        params = json.loads(network['network_params'])
    except json.JSONDecodeError as exc:
        raise ValueError('[network].network_params must be a JSON object') from exc
    if not isinstance(params, dict):
        raise ValueError('[network].network_params must decode to an object')
    return network_type, params


def make_swinir_option(project_root, data_root, split_dir, cfg, seed,
                       total_iter, num_gpu, micro_batch_size):
    network_name = cfg['network']['network_name'].strip().lower()
    effective = int(cfg['training'].get('effective_batch_size_per_gpu',
                                        cfg['training']['batch_size_per_gpu']))
    batch = int(micro_batch_size or cfg['training'].get(
        'micro_batch_size_per_gpu', effective))
    if batch <= 0 or effective % batch:
        raise ValueError('micro batch must divide the fixed effective batch')
    name = f'holo_{network_name}_{total_iter}_seed{seed}_mb{batch}'
    # Reuse the shared option template so every dataset, optimizer, scheduler,
    # AMP, validation, and metric field remains identical to Restormer.
    option_path = make_option(
        project_root, data_root, split_dir, cfg, 'swinir', seed, total_iter,
        num_gpu, micro_batch_size, experiment_name=name)
    option = yaml.safe_load(option_path.read_text(encoding='utf-8'))
    network_type, params = network_params(cfg)
    option['name'] = name
    option['network_g'] = {'type': network_type, **params}
    option_path.write_text(
        yaml.safe_dump(option, sort_keys=False, allow_unicode=True),
        encoding='utf-8')
    return option_path


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    project_root = (args.project_root or Path(
        os.environ.get('HOLO_PROJECT_ROOT', repo_root.parent))).resolve()
    if args.config:
        config_path = args.config.resolve()
    else:
        candidates = (
            project_root / '10.11' / 'experiment_config_swinir_v7.txt',
            project_root / 'experiment_config_swinir_v7.txt',
        )
        config_path = next((path for path in candidates if path.is_file()), candidates[0])
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing unified config: {config_path}')
    cfg = read_shared_config(config_path)
    configured_iters = int(cfg['training']['total_iterations'])
    total_iter = configured_iters if args.iters is None else int(args.iters)
    if total_iter != configured_iters:
        raise ValueError(
            f'Frozen v7 SwinIR requires total_iterations={configured_iters}; '
            f'got --iters {total_iter}')
    data_root = (args.data_root or Path(
        os.environ.get('HOLO_DATA_ROOT', project_root / 'five'))).resolve()
    for sub in ('HR', 'LR4'):
        if not (data_root / sub).is_dir():
            raise FileNotFoundError(f'Missing {sub} directory under {data_root}')
    split_dir, _, _, _ = prepare_splits(project_root, cfg)
    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    for seed in seeds:
        option = make_swinir_option(
            project_root, data_root, split_dir, cfg, seed, total_iter,
            args.num_gpu, args.micro_batch_size)
        print(f'[SwinIR, seed={seed}] option: {option}')
        if args.run:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(repo_root) + os.pathsep + env.get(
                'PYTHONPATH', '')
            subprocess.run(
                [sys.executable, 'basicsr/train.py', '-opt', str(option)],
                cwd=repo_root, env=env, check=True)
    print('Prepared SwinIR holographic trial(s).')


if __name__ == '__main__':
    main()
