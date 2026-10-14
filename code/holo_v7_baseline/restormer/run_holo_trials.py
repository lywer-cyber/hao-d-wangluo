"""Prepare and optionally run the two 25,500-iteration decision trials.

Usage on AutoDL (the only machine-specific values are environment variables)::

    export HOLO_PROJECT_ROOT=/root/autodl-tmp/your_project
    export HOLO_DATA_ROOT=/root/autodl-tmp/your_dataset
    cd "$HOLO_PROJECT_ROOT/restormer"
    python run_holo_trials.py --run --seeds 10

The script reads ``experiment_config.txt`` for the shared protocol, writes
split manifests and BasicSR options under ``experiments/``, then runs the
original MDTA and frequency-band variants with identical data and seed.
The 25,500-iteration run is a decision/prototyping run; the final paper run
must use the configured 250,000 iterations and seeds 10,20,30.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--seeds", default="10", help="comma-separated seeds")
    p.add_argument("--models", default="both", choices=("mdta", "band", "both"))
    p.add_argument("--run", action="store_true", help="launch BasicSR training")
    p.add_argument("--num-gpu", type=int, default=1)
    p.add_argument("--micro-batch-size", type=int, default=None,
                   help="only for OOM recovery; effective batch stays fixed")
    return p.parse_args()


def read_shared_config(path: Path):
    parser = configparser.ConfigParser(interpolation=None, delimiters=("="))
    parser.optionxform = str
    with path.open("r", encoding="utf-8") as f:
        parser.read_file(f)
    return parser


def ranges(spec: str):
    values = []
    for part in spec.split(","):
        part = part.strip()
        if ".." in part:
            left, right = (int(x) for x in part.split("..", 1))
            values.extend(range(left, right + 1))
        else:
            values.append(int(part))
    return values


def read_manifest(path: Path, *, allow_plain_numbers: bool = False):
    if not path.is_file():
        raise FileNotFoundError(f"Frozen split manifest is missing: {path}")
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        if value.lower().endswith(".bmp"):
            values.append(int(value[:-4]))
            continue
        if not allow_plain_numbers:
            raise ValueError(f"Invalid manifest entry in {path}: {value!r}")
        values.append(int(value))
    return values


def prepare_splits(project_root: Path, cfg):
    split_dir = project_root / cfg["split"]["split_dir"]
    train = ranges(cfg["split"]["train_ranges"])
    val = ranges(cfg["split"]["val_ranges"])
    test = ranges(cfg["split"]["test_ranges"])
    if (len(train), len(val), len(test)) != (
        int(cfg["split"]["train_count"]),
        int(cfg["split"]["val_count"]),
        int(cfg["split"]["test_count"]),
    ):
        raise ValueError("Configured split ranges do not match their counts")
    # v7 freezes the split and visual lists.  Never regenerate or overwrite
    # them from ranges: doing so could silently change an already-started
    # comparison experiment.
    manifests = {
        "train": split_dir / "train_meta.txt",
        "val": split_dir / "val_meta.txt",
        "test": split_dir / "test_meta.txt",
    }
    for name, expected in (("train", train), ("val", val), ("test", test)):
        actual = read_manifest(manifests[name])
        if actual != expected:
            raise ValueError(
                f"Frozen {name} split differs from config ranges: {manifests[name]}")
    split_manifest = split_dir / Path(cfg["split"]["split_manifest"]).name
    if not split_manifest.is_file():
        raise FileNotFoundError(f"Frozen split manifest is missing: {split_manifest}")
    try:
        recorded = json.loads(split_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid frozen split manifest: {split_manifest}") from exc
    expected_manifest = {
        "seed": int(cfg["split"]["split_seed"]),
        "train": train,
        "val": val,
        "test": test,
    }
    if recorded != expected_manifest:
        raise ValueError(f"Frozen split manifest differs from v7: {split_manifest}")
    visual_name = Path(cfg["split"].get("visual_sample_manifest", "visual_samples.txt")).name
    visual = split_dir / visual_name
    chosen = read_manifest(visual, allow_plain_numbers=True)
    if len(chosen) != int(cfg["split"]["visual_sample_count"]):
        raise ValueError("Frozen visual sample count differs from v7")
    if len(set(chosen)) != len(chosen) or not set(chosen).issubset(set(test)):
        raise ValueError("Frozen visual samples must be unique test-set images")
    return split_dir, train, val, test


def yaml_quote(path: Path):
    # BasicSR's option parser reads ordinary YAML strings; use POSIX paths so
    # generated files are valid on Linux AutoDL even when prepared on Windows.
    return path.resolve().as_posix()


def make_option(project_root: Path, data_root: Path, split_dir: Path,
                cfg, model: str, seed: int, total_iter: int, num_gpu: int,
                micro_batch_size: int | None = None,
                experiment_name: str | None = None):
    training = cfg["training"]
    lr = float(training["learning_rate"])
    beta1, beta2 = (float(x) for x in training["optimizer_betas"].split(","))
    milestones = [int(x) for x in cfg["scheduler"]["milestones"].split(",")]
    scale = int(cfg["dataset"]["scale"])
    patch = int(training["patch_size"])
    effective_batch = int(training.get("effective_batch_size_per_gpu",
                                      training["batch_size_per_gpu"]))
    batch = int(micro_batch_size or training.get("micro_batch_size_per_gpu", effective_batch))
    if batch <= 0 or effective_batch % batch != 0:
        raise ValueError("micro batch must be a positive divisor of the fixed effective batch")
    accumulation_steps = effective_batch // batch
    name = experiment_name or f"holo_restormer_{model}_{total_iter}_seed{seed}_mb{batch}"
    exp_dir = project_root / "experiments" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    workers = int(training["num_workers_per_gpu"])
    runtime = cfg["runtime"]
    deterministic = cfg["determinism"]
    use_hflip = training.get("use_hflip", "false").lower()
    if use_hflip != "false":
        raise ValueError("The holographic protocol requires use_hflip=false")
    option = f"""# Generated from experiment_config.txt; do not edit by hand.
name: {name}
model_type: ImageCleanModel
scale: {scale}
num_gpu: {num_gpu}
manual_seed: {seed}
holo_protocol: true

runtime:
  validate_on_startup: {runtime.get('validate_on_startup', 'true')}
  python_version: '{runtime['python_version']}'
  pytorch_version: '{runtime['pytorch_version']}'
  pytorch_cuda_build: {runtime['pytorch_cuda_build']}
  cuda_runtime_version: '{runtime['cuda_runtime_version']}'
  nvidia_driver_version: {runtime['nvidia_driver_version']}
  validate_nvidia_driver: {runtime.get('validate_nvidia_driver', 'false')}
  torchvision_version: '{runtime['torchvision_version']}'
  scikit_image_version: '{runtime['scikit_image_version']}'
  lpips_version: '{runtime['lpips_version']}'
  numpy_version: '{runtime['numpy_version']}'
  opencv_python_version: '{runtime['opencv_python_version']}'
  einops_version: '{runtime['einops_version']}'
  pyyaml_version: '{runtime['pyyaml_version']}'
  lpips_weights_relpath: {runtime['lpips_weights_relpath']}
  lpips_weights_sha256: {runtime['lpips_weights_sha256']}
  strict_lpips_weights_check: {runtime.get('strict_lpips_weights_check', 'true')}

determinism:
  enabled: {deterministic.get('enabled', 'true')}
  cudnn_deterministic: {deterministic.get('cudnn_deterministic', 'true')}
  cudnn_benchmark: {deterministic.get('cudnn_benchmark', 'false')}
  use_deterministic_algorithms: {deterministic.get('use_deterministic_algorithms', 'true')}
  data_order: {deterministic.get('data_order', 'fixed_sampler_seed_plus_epoch')}
  worker_seed_formula: {deterministic.get('worker_seed_formula', 'base_seed_plus_rank_times_num_workers_plus_worker_id')}

datasets:
  train:
    name: HoloTrain
    type: HoloPairedDataset
    dataroot_gt: {yaml_quote(data_root / 'HR')}
    dataroot_lq: {yaml_quote(data_root / 'LR4')}
    meta_info_file: {yaml_quote(split_dir / 'train_meta.txt')}
    io_backend: {{type: disk}}
    batch_size_per_gpu: {batch}
    num_worker_per_gpu: {workers}
    gt_size: {patch}
    use_hflip: {use_hflip}
    use_rot: false
    dataset_enlarge_ratio: 1
    prefetch_mode: null
    pin_memory: {training.get('pin_memory', 'false')}
    persistent_workers: {training.get('persistent_workers', 'false')}
  val:
    name: HoloVal
    type: HoloPairedDataset
    dataroot_gt: {yaml_quote(data_root / 'HR')}
    dataroot_lq: {yaml_quote(data_root / 'LR4')}
    meta_info_file: {yaml_quote(split_dir / 'val_meta.txt')}
    io_backend: {{type: disk}}
    batch_size_per_gpu: {cfg['validation'].get('batch_size_per_gpu', '1')}
    num_worker_per_gpu: {cfg['validation'].get('num_workers', '0')}
    use_hflip: false
    use_rot: false
    pin_memory: {cfg['validation'].get('pin_memory', 'false')}
    persistent_workers: {cfg['validation'].get('persistent_workers', 'false')}
    prefetch_mode: null
    whole_image: {cfg['validation'].get('whole_image', 'true')}
    resize: {cfg['validation'].get('resize', 'false')}
    padding: {cfg['validation'].get('padding', 'false')}
    preserve_native_resolution: {cfg['validation'].get('preserve_native_resolution', 'true')}
    compute_lpips: {cfg['validation'].get('compute_lpips', 'false')}

network_g:
  type: HoloRestormer
  inp_channels: 3
  out_channels: 3
  scale: {scale}
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4
  heads: [1, 2, 4, 8]
  ffn_expansion_factor: 2.66
  bias: false
  LayerNorm_type: WithBias
  attention_mode: {'mdta' if model == 'mdta' else 'band'}
  n_bands: 6

path:
  pretrain_network_g: null
  strict_load_g: true
  resume_state: null
  root: {yaml_quote(project_root)}

train:
  total_iter: {total_iter}
  warmup_iter: {training.get('warmup_iterations', training.get('warmup_iter', '-1'))}
  warmup_init_ratio: {training.get('warmup_init_ratio', '0.0')}
  use_grad_clip: false
  gradient_clip_enabled: {training.get('gradient_clip_enabled', 'false')}
  gradient_clip_norm: {training.get('gradient_clip_norm', 'none')}
  gradient_accumulation_steps: {accumulation_steps}
  effective_batch_size_per_gpu: {effective_batch}
  amp_enabled: {training.get('amp_enabled', 'true')}
  amp_dtype: {training.get('amp_dtype', 'float16')}
  oom_policy: {training.get('oom_policy', 'reduce_micro_batch_preserve_effective_batch')}
  loss_divide_by_accumulation_steps: {training.get('loss_divide_by_accumulation_steps', 'true')}
  amp_grad_scaler: {training.get('amp_grad_scaler', 'true')}
  amp_init_scale: {training.get('amp_init_scale', '65536')}
  amp_growth_factor: {training.get('amp_growth_factor', '2.0')}
  amp_backoff_factor: {training.get('amp_backoff_factor', '0.5')}
  amp_growth_interval: {training.get('amp_growth_interval', '2000')}
  ema_enabled: {training.get('ema_enabled', 'false')}
  ema_decay: {training.get('ema_decay', '0')}
  ema_warmup_iters: {training.get('ema_warmup_iters', '0')}
  loss_l1_weight: {training.get('loss_l1_weight', training.get('loss_weight', '1.0'))}
  loss_gfl_weight: {training.get('loss_gfl_weight', '0.0')}
  loss_hf_weight: {training.get('loss_hf_weight', '0.0')}
  gfl_ramp_iterations: {training.get('gfl_ramp_iterations', '0')}
  gfl_radius_ref: {training.get('gfl_radius_ref', '0.125')}
  hf_sigma_px: {training.get('hf_sigma_px', '1.8006')}
  mixing_augs: {{mixup: false, use_identity: true}}
  scheduler:
    type: MultiStepLR
    milestones: {milestones}
    gamma: {cfg['scheduler']['gamma']}
  optim_g:
    type: Adam
    lr: {lr}
    weight_decay: {cfg['training']['weight_decay']}
    betas: [{beta1}, {beta2}]
  pixel_opt:
    type: L1Loss
    loss_weight: {training.get('loss_l1_weight', training.get('loss_weight', '1.0'))}
    reduction: mean

val:
  val_freq: {cfg['validation']['frequency_iterations']}
  save_img: {cfg['validation']['save_img']}
  rgb2bgr: false
  use_image: false
  metrics:
    psnr:
      type: calculate_holo_psnr
      crop_border: {cfg['evaluation']['psnr_crop_border']}
      test_y_channel: {cfg['evaluation']['test_y_channel']}
      data_range: {cfg['evaluation']['psnr_data_range']}
    ssim:
      type: calculate_holo_ssim
      crop_border: {cfg['evaluation']['psnr_crop_border']}
      test_y_channel: {cfg['evaluation']['test_y_channel']}
      data_range: {cfg['evaluation']['ssim_data_range']}
      win_size: {cfg['evaluation']['ssim_window_size']}
      sigma: {cfg['evaluation']['ssim_sigma']}
      gaussian_weights: {cfg['evaluation']['ssim_gaussian_weights']}
      use_sample_covariance: {cfg['evaluation']['ssim_use_sample_covariance']}
      channel_axis: {cfg['evaluation']['ssim_channel_axis']}
      channel_reduction: {cfg['evaluation']['ssim_channel_reduction']}
    rmse:
      type: calculate_holo_rmse
      crop_border: {cfg['evaluation']['crop_border']}
      data_range: {cfg['evaluation']['psnr_data_range']}
    mae:
      type: calculate_holo_mae
      crop_border: {cfg['evaluation']['crop_border']}
      data_range: {cfg['evaluation']['psnr_data_range']}
  checkpoint_selection_metric: {cfg['validation']['checkpoint_selection_metric']}
  checkpoint_selection_mode: {cfg['validation']['checkpoint_selection_mode']}
  checkpoint_selection_tie_breaker: {cfg['validation']['checkpoint_selection_tie_breaker']}
  checkpoint_selection_tie_tolerance: {cfg['validation']['checkpoint_selection_tie_tolerance']}
  checkpoint_selection_split: {cfg['validation']['checkpoint_selection_split']}

logger:
  print_freq: 100
  save_checkpoint_freq: {cfg['validation']['save_checkpoint_frequency_iterations']}
  use_tb_logger: false
  wandb: {{project: null, resume_id: null}}

evaluation:
  metrics: [PSNR, SSIM, LPIPS, RMSE, MAE]
  crop_border: {cfg['evaluation']['crop_border']}
  test_y_channel: {cfg['evaluation']['test_y_channel']}
  rgb_to_y: {cfg['evaluation'].get('rgb_to_y', 'false')}
  metric_pixel_range: '{cfg['evaluation']['metric_pixel_range']}'
  psnr_data_range: {cfg['evaluation']['psnr_data_range']}
  ssim_data_range: {cfg['evaluation']['ssim_data_range']}
  ssim_window_size: {cfg['evaluation']['ssim_window_size']}
  ssim_sigma: {cfg['evaluation']['ssim_sigma']}
  ssim_gaussian_weights: {cfg['evaluation']['ssim_gaussian_weights']}
  ssim_use_sample_covariance: {cfg['evaluation']['ssim_use_sample_covariance']}
  ssim_channel_axis: {cfg['evaluation']['ssim_channel_axis']}
  ssim_channel_reduction: {cfg['evaluation']['ssim_channel_reduction']}
  ssim_library: {cfg['evaluation']['ssim_library']}
  ssim_library_version: '{cfg['evaluation']['ssim_library_version']}'
  lpips_network: {cfg['evaluation']['lpips_network']}
  lpips_input_range: '{cfg['evaluation']['lpips_input_range']}'
  lpips_crop_border: {cfg['evaluation']['lpips_crop_border']}
  lpips_spatial: {cfg['evaluation']['lpips_spatial']}
  metric_implementation_version: {cfg['evaluation']['metric_implementation_version']}
  val_compute_lpips: {cfg['validation'].get('compute_lpips', 'false')}
  test_batch_size_per_gpu: {cfg['evaluation'].get('test_batch_size_per_gpu', '1')}
  test_num_workers: {cfg['evaluation'].get('test_num_workers', '0')}
  test_pin_memory: {cfg['evaluation'].get('test_pin_memory', 'false')}
  test_persistent_workers: {cfg['evaluation'].get('test_persistent_workers', 'false')}
  test_prefetch_mode: null
  test_whole_image: {cfg['evaluation'].get('test_whole_image', 'true')}
  test_resize: {cfg['evaluation'].get('test_resize', 'false')}
  test_padding: {cfg['evaluation'].get('test_padding', 'false')}
  test_preserve_native_resolution: {cfg['evaluation'].get('test_preserve_native_resolution', 'true')}
  test_compute_lpips: {cfg['evaluation'].get('test_compute_lpips', 'true')}

visual_output:
  format: {cfg['visual_output']['format']}
  compression: {cfg['visual_output']['compression']}
  save_count: {cfg['visual_output']['save_count']}
  save_list: {yaml_quote(split_dir / Path(cfg['visual_output']['save_list']).name)}
  export_after_all_comparison_runs_complete: {cfg['visual_output']['export_after_all_comparison_runs_complete']}
  export_split: {cfg['visual_output']['export_split']}
  checkpoint_source: {cfg['visual_output']['checkpoint_source']}
  visual_only: {cfg['visual_output']['visual_only']}
  compute_metrics: {cfg['visual_output']['compute_metrics']}
  save_lr: {cfg['visual_output']['save_lr']}
  save_sr: {cfg['visual_output']['save_sr']}
  save_hr: {cfg['visual_output']['save_hr']}
  save_comparison_image: {cfg['visual_output']['save_comparison_image']}
"""
    option_path = exp_dir / "options.yml"
    option_path.write_text(option, encoding="utf-8")
    return option_path


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    project_root = (args.project_root or
                    Path(os.environ.get("HOLO_PROJECT_ROOT", repo_root.parent))).resolve()
    config_path = (args.config or project_root / "experiment_config.txt").resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing unified config: {config_path}")
    cfg = read_shared_config(config_path)
    data_root = (args.data_root or
                 Path(os.environ.get("HOLO_DATA_ROOT", project_root / "five"))).resolve()
    for sub in ("HR", "LR4"):
        if not (data_root / sub).is_dir():
            raise FileNotFoundError(f"Missing {sub} directory under {data_root}")
    split_dir, _, _, _ = prepare_splits(project_root, cfg)
    models = ["mdta", "band"] if args.models == "both" else [args.models]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for seed in seeds:
        for model in models:
            total_iter = int(cfg["training"].get("total_iterations", 0)) if args.iters is None else int(args.iters)
            option = make_option(project_root, data_root, split_dir, cfg, model,
                                 seed, total_iter, args.num_gpu,
                                 args.micro_batch_size)
            print(f"[{model}, seed={seed}] option: {option}")
            if args.run:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
                subprocess.run([sys.executable, "basicsr/train.py", "-opt", str(option)],
                               cwd=repo_root, env=env, check=True)
    print("Prepared trials. Compare the final validation PSNR/SSIM in each experiment log.")


if __name__ == "__main__":
    main()

