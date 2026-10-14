"""Runtime checks for the fixed holographic experiment protocol.

The project deliberately fails early when the reproducibility contract is not
available.  This keeps an experiment from silently using a different metric
implementation, random-number setup, or LPIPS checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch


def configure_determinism(seed: int, settings: Optional[Mapping] = None) -> None:
    """Set every RNG and backend switch used by training/evaluation."""
    settings = settings or {}
    enabled = bool(settings.get("enabled", True))
    if not enabled:
        raise ValueError("The holographic protocol requires deterministic=true")

    # Must be set before the first CUDA matmul.  Keeping an existing explicit
    # value respects a scheduler/container that already configured it.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    torch.backends.cudnn.deterministic = bool(
        settings.get("cudnn_deterministic", True))
    torch.backends.cudnn.benchmark = bool(settings.get("cudnn_benchmark", False))
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(
            bool(settings.get("use_deterministic_algorithms", True)),
            warn_only=False)


def validate_determinism(settings: Mapping) -> None:
    """Verify backend flags after configuration (not just the config text)."""
    if not bool(settings.get("enabled", True)):
        raise RuntimeError("Determinism is disabled in the experiment options")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if not torch.backends.cudnn.deterministic:
        raise RuntimeError("torch.backends.cudnn.deterministic must be True")
    if torch.backends.cudnn.benchmark:
        raise RuntimeError("torch.backends.cudnn.benchmark must be False")
    if hasattr(torch, "are_deterministic_algorithms_enabled") and bool(
            settings.get("use_deterministic_algorithms", True)):
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("PyTorch deterministic algorithms are not enabled")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("TF32 matmul must be disabled for deterministic runs")
    if torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 cuDNN must be disabled for deterministic runs")
    if settings.get("data_order") != "fixed_sampler_seed_plus_epoch":
        raise RuntimeError("data_order must be fixed_sampler_seed_plus_epoch")
    if settings.get("worker_seed_formula") != (
            "base_seed_plus_rank_times_num_workers_plus_worker_id"):
        raise RuntimeError("worker_seed_formula does not match the fixed protocol")


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Required package is not installed: {distribution}") from exc


def _check_version(distribution: str, expected: str, errors: list[str]) -> None:
    if not expected or expected.startswith("TO_BE_"):
        errors.append(f"版本尚未固定: {distribution}={expected!r}")
        return
    try:
        actual = _installed_version(distribution)
    except RuntimeError as exc:
        errors.append(str(exc))
        return
    # CUDA-enabled wheels (notably torchvision) append a local build suffix,
    # e.g. ``0.15.1+cu118``.  The protocol pins the public package version;
    # the CUDA build itself is checked separately for torch.
    actual_public = actual.split("+", 1)[0]
    if actual_public != expected:
        errors.append(f"{distribution} version mismatch: expected {expected}, got {actual}")


def locate_lpips_weights() -> Path:
    import lpips  # type: ignore
    return Path(lpips.__file__).resolve().parent / "weights" / "v0.1" / "alex.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_runtime_versions(runtime: Mapping, *, require_lpips: bool = False) -> None:
    """Validate pinned versions and, for final evaluation, LPIPS weights."""
    errors: list[str] = []
    expected_python = str(runtime.get("python_version", "3.8"))
    if expected_python and not platform.python_version().startswith(expected_python + "."):
        errors.append(
            f"python version mismatch: expected {expected_python}.x, got {platform.python_version()}")

    expected_torch = str(runtime.get("pytorch_version", "2.0.0"))
    actual_torch = torch.__version__.split("+")[0]
    if expected_torch and actual_torch != expected_torch:
        errors.append(f"torch version mismatch: expected {expected_torch}, got {torch.__version__}")
    expected_build = str(runtime.get("pytorch_cuda_build", "cu118"))
    if expected_build and expected_build not in torch.__version__:
        errors.append(
            f"torch CUDA package build mismatch: expected {expected_build}, got {torch.__version__}")
    actual_cuda = torch.version.cuda or "cpu"
    configured_cuda = str(runtime.get("cuda_runtime_version", "11.8"))
    if actual_cuda != configured_cuda:
        errors.append(f"torch CUDA build mismatch: expected {configured_cuda}, got {actual_cuda}")

    expected_driver = str(runtime.get("nvidia_driver_version", ""))
    active_driver = driver_version()
    if bool(runtime.get("validate_nvidia_driver", True)):
        if torch.cuda.is_available() and active_driver is None:
            errors.append("nvidia-smi is unavailable; NVIDIA driver version cannot be verified")
        if active_driver is not None:
            if not expected_driver or expected_driver.startswith("TO_BE_"):
                errors.append(
                    "nvidia_driver_version must be filled with the exact nvidia-smi version")
            elif active_driver != expected_driver:
                errors.append(
                    f"NVIDIA driver mismatch: expected {expected_driver}, got {active_driver}")

    packages = {
        "torchvision": runtime.get("torchvision_version"),
        "scikit-image": runtime.get("scikit_image_version"),
        "lpips": runtime.get("lpips_version"),
        "numpy": runtime.get("numpy_version"),
        "opencv-python": runtime.get("opencv_python_version"),
        "einops": runtime.get("einops_version"),
        "PyYAML": runtime.get("pyyaml_version"),
    }
    for distribution, expected in packages.items():
        if expected is not None:
            _check_version(distribution, str(expected), errors)

    if require_lpips:
        if not bool(runtime.get("strict_lpips_weights_check", True)):
            errors.append("strict_lpips_weights_check must be true for final evaluation")
        expected_hash = str(runtime.get("lpips_weights_sha256", ""))
        if not expected_hash or expected_hash.startswith("TO_BE_"):
            errors.append("LPIPS AlexNet weight SHA256 is not pinned")
        else:
            try:
                weight_path = locate_lpips_weights()
                if not weight_path.is_file():
                    errors.append(f"LPIPS weight file missing: {weight_path}")
                else:
                    actual_hash = sha256_file(weight_path)
                    if actual_hash != expected_hash.upper():
                        errors.append(
                            f"LPIPS weight SHA256 mismatch: expected {expected_hash}, got {actual_hash}")
            except Exception as exc:  # pragma: no cover - environment-specific
                errors.append(f"Cannot validate LPIPS weights: {exc}")

    if errors:
        raise RuntimeError("Holographic runtime validation failed:\n- " + "\n- ".join(errors))


def validate_experiment_options(opt: Mapping) -> None:
    """Validate invariants that must hold for every generated BasicSR option."""
    train = opt.get("train", {})
    datasets = opt.get("datasets", {})
    deterministic = opt.get("determinism", {})
    if bool(opt.get("val", {}).get("save_img", False)):
        raise ValueError("Holographic validation must set val.save_img=false")
    visual = opt.get("visual_output", {})
    for key, expected in {
            "export_after_all_comparison_runs_complete": True,
            "export_split": "test",
            "checkpoint_source": "best_checkpoint_selected_on_val",
            "visual_only": True,
            "compute_metrics": False,
            "save_count": 10,
            "save_lr": True,
            "save_sr": True,
            "save_hr": True}.items():
        if visual.get(key) != expected:
            raise ValueError(f"visual_output.{key} must be {expected!r}")
    if bool(datasets.get("train", {}).get("use_hflip", True)):
        raise ValueError("Holographic protocol requires train.use_hflip=false")
    if bool(datasets.get("train", {}).get("use_rot", True)):
        raise ValueError("Holographic protocol requires train.use_rot=false")
    accumulation = int(train.get("gradient_accumulation_steps", 1))
    effective = int(train.get("effective_batch_size_per_gpu", 0))
    micro = int(datasets.get("train", {}).get("batch_size_per_gpu", 0))
    if accumulation < 1 or effective < 1 or micro < 1 or accumulation * micro != effective:
        raise ValueError("effective_batch_size_per_gpu must equal micro_batch * accumulation_steps")
    if not bool(train.get("loss_divide_by_accumulation_steps", False)):
        raise ValueError("loss_divide_by_accumulation_steps must be true")
    if not bool(train.get("amp_grad_scaler", False)):
        raise ValueError("amp_grad_scaler must be true")
    if bool(train.get("use_grad_clip", False)) or bool(
            train.get("gradient_clip_enabled", False)):
        raise ValueError("The fair holographic protocol disables gradient clipping for every network")
    if str(train.get("gradient_clip_norm", "none")).lower() not in ("none", "null"):
        raise ValueError("gradient_clip_norm must be none when gradient clipping is disabled")

    train_dataset = datasets.get("train", {})
    if bool(train_dataset.get("pin_memory", False)):
        raise ValueError("pin_memory must be false for every holographic dataloader")
    if bool(train_dataset.get("persistent_workers", False)):
        raise ValueError("persistent_workers must be false for every holographic dataloader")
    if train_dataset.get("prefetch_mode") is not None:
        raise ValueError("prefetch_mode must be null for every holographic dataloader")

    for phase in ("val", "test"):
        dataset = datasets.get(phase)
        if not dataset:
            continue
        if int(dataset.get("batch_size_per_gpu", 1)) != 1:
            raise ValueError(f"{phase} batch_size_per_gpu must be 1")
        if int(dataset.get("num_worker_per_gpu", 0)) != 0:
            raise ValueError(f"{phase} num_worker_per_gpu must be 0")
        if bool(dataset.get("pin_memory", False)):
            raise ValueError(f"{phase} pin_memory must be false")
        if bool(dataset.get("persistent_workers", False)):
            raise ValueError(f"{phase} persistent_workers must be false")
        if dataset.get("prefetch_mode") is not None:
            raise ValueError(f"{phase} prefetch_mode must be null")
        for key in ("whole_image", "preserve_native_resolution"):
            if dataset.get(key) is not True:
                raise ValueError(f"{phase} {key} must be true")
        for key in ("resize", "padding"):
            if dataset.get(key) is not False:
                raise ValueError(f"{phase} {key} must be false")
        if phase == "val" and bool(dataset.get("compute_lpips", False)):
            raise ValueError("Validation must not compute LPIPS")

    evaluation = opt.get("evaluation", {})
    if evaluation.get("metric_pixel_range") != "0_1":
        raise ValueError("metric_pixel_range must be 0_1")
    if bool(evaluation.get("test_y_channel", False)) or bool(
            evaluation.get("rgb_to_y", False)):
        raise ValueError("Holographic evaluation is fixed to RGB; Y-channel conversion is disabled")
    if int(evaluation.get("crop_border", -1)) != 4:
        raise ValueError("Holographic validation/test crop_border must be 4")
    if int(evaluation.get("ssim_channel_axis", -1)) != 2:
        raise ValueError("Holographic SSIM requires channel_axis=2")
    if evaluation.get("ssim_channel_reduction") != "mean":
        raise ValueError("Holographic SSIM requires RGB channel reduction=mean")
    if evaluation.get("ssim_library") != "scikit-image":
        raise ValueError("Holographic SSIM must use scikit-image")
    if evaluation.get("lpips_network") != "alex":
        raise ValueError("LPIPS network must be alex")
    if evaluation.get("lpips_input_range") != "-1_1":
        raise ValueError("LPIPS input range must be -1_1")
    if bool(evaluation.get("lpips_spatial", True)):
        raise ValueError("LPIPS spatial must be false")
    for key, expected in {
            "test_batch_size_per_gpu": 1,
            "test_num_workers": 0,
            "test_pin_memory": False,
            "test_persistent_workers": False,
            "test_whole_image": True,
            "test_resize": False,
            "test_padding": False,
            "test_preserve_native_resolution": True,
            "test_compute_lpips": True}.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"evaluation.{key} must be {expected!r}")
    if opt.get("evaluation", {}).get("val_compute_lpips", False):
        raise ValueError("Validation LPIPS is disabled by the fixed protocol")
    val_metrics = opt.get("val", {}).get("metrics", {})
    if set(val_metrics) != {"psnr", "ssim", "rmse", "mae"}:
        raise ValueError("Validation metrics must be exactly PSNR, SSIM, RMSE and MAE; LPIPS is test-only")
    psnr = val_metrics.get("psnr", {})
    ssim = val_metrics.get("ssim", {})
    if int(psnr.get("crop_border", -1)) != int(evaluation.get("crop_border", -2)):
        raise ValueError("val/test crop_border must be identical")
    if float(psnr.get("data_range", -1)) != float(
            evaluation.get("psnr_data_range", -2)):
        raise ValueError("val/test PSNR data_range must be identical")
    if bool(psnr.get("test_y_channel", True)):
        raise ValueError("Validation PSNR must use RGB, not Y channel")
    if int(ssim.get("channel_axis", -1)) != 2:
        raise ValueError("Validation SSIM requires channel_axis=2")
    ssim_pairs = {
        "crop_border": (ssim.get("crop_border"), evaluation.get("crop_border")),
        "data_range": (ssim.get("data_range"), evaluation.get("ssim_data_range")),
        "win_size": (ssim.get("win_size"), evaluation.get("ssim_window_size")),
        "sigma": (ssim.get("sigma"), evaluation.get("ssim_sigma")),
        "gaussian_weights": (ssim.get("gaussian_weights"), evaluation.get("ssim_gaussian_weights")),
        "use_sample_covariance": (ssim.get("use_sample_covariance"), evaluation.get("ssim_use_sample_covariance")),
        "channel_axis": (ssim.get("channel_axis"), evaluation.get("ssim_channel_axis")),
        "channel_reduction": (ssim.get("channel_reduction"), evaluation.get("ssim_channel_reduction")),
    }
    for key, (val_value, test_value) in ssim_pairs.items():
        if str(val_value) != str(test_value):
            raise ValueError(f"val/test SSIM {key} must be identical")
    validate_determinism(deterministic)


def driver_version() -> Optional[str]:
    """Return the active NVIDIA driver version, if nvidia-smi is available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    values = [line.strip() for line in output.splitlines() if line.strip()]
    return values[0] if values else None
