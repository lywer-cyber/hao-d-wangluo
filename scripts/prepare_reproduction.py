#!/usr/bin/env python3
"""Prepare portable configs from unmodified, archived baseline options.

This utility only relocates paths and (when requested) the seed. It never starts
training, rewrites the scientific hyperparameters, or copies the full dataset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "configs" / "baseline_catalog.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_idempotent(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"Refusing to overwrite different content: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def prepare(model: str, data_root: Path, work_root: Path, seed=None,
            skip_data_check: bool = False) -> dict:
    entries = json.loads(CATALOG.read_text(encoding="utf-8"))["models"]
    if model not in entries:
        raise ValueError(f"Unknown baseline: {model}; choices: {sorted(entries)}")
    entry = entries[model]
    source = REPO / entry["config"]
    original = yaml.safe_load(source.read_text(encoding="utf-8"))
    opt = copy.deepcopy(original)
    data_root = data_root.expanduser().resolve()
    run_root = work_root.expanduser().resolve() / model
    source_tree = (REPO / entry["source_tree"]).resolve()
    split_source = REPO / entry["splits"]
    split_target = run_root / "data" / "splits"
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        opt["manual_seed"] = int(seed)
        opt["name"] = re.sub(r"_seed\d+_", f"_seed{seed}_", opt["name"])
    expected = set()
    for phase in ("train", "val", "test"):
        p = split_source / f"{phase}_meta.txt"
        names = [line.split()[0] for line in p.read_text().splitlines() if line.strip()]
        expected.update(names)
    if not skip_data_check:
        missing = [str(data_root / folder / name)
                   for folder in ("HR", "LR4") for name in sorted(expected)
                   if not (data_root / folder / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} input files; first examples: {missing[:5]}. "
                "Use your existing 5000-pair RGB BMP corpus, not the 10 visual samples.")
    for p in sorted(split_source.iterdir()):
        if p.is_file() and p.suffix in (".txt", ".json"):
            write_idempotent(split_target / p.name, p.read_bytes())
    for phase, dataset in opt["datasets"].items():
        base_phase = phase.split("_", 1)[0]
        dataset["dataroot_gt"] = str(data_root / "HR")
        dataset["dataroot_lq"] = str(data_root / "LR4")
        dataset["meta_info_file"] = str(split_target / f"{base_phase}_meta.txt")
    opt["path"]["root"] = str(run_root)
    opt["visual_output"]["save_list"] = str(split_target / "visual_samples.txt")
    option_path = run_root / "options" / (opt["name"] + ".yml")
    write_idempotent(option_path, yaml.safe_dump(
        opt, sort_keys=False, allow_unicode=True).encode("utf-8"))
    q = shlex.quote
    common = ("#!/usr/bin/env bash\nset -euo pipefail\n"
              f"cd {q(str(source_tree))}\n"
              f"export PYTHONPATH={q(str(source_tree))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
              f"export HOLO_PROJECT_ROOT={q(str(run_root))}\n"
              f"export HOLO_DATA_ROOT={q(str(data_root))}\n"
              'PYTHON_BIN="${PYTHON_BIN:-python}"\n')
    train = common + (f'"$PYTHON_BIN" -u basicsr/train.py -opt '
                      f'{q(str(option_path))} --launcher none\n')
    evaluate = common + ('CHECKPOINT="${1:?Usage: evaluate.sh /absolute/path/net_g_best_ITER.pth}"\n'
                         f'"$PYTHON_BIN" evaluate_holo.py --option {q(str(option_path))} '
                         '--checkpoint "$CHECKPOINT" --split test --require-lpips\n')
    train_path = run_root / "launch_train.sh"
    eval_path = run_root / "evaluate.sh"
    write_idempotent(train_path, train.encode())
    write_idempotent(eval_path, evaluate.encode())
    record = {
        "model": model, "archived_config": entry["config"],
        "archived_config_sha256": sha256(source), "generated_option": str(option_path),
        "generated_option_sha256": sha256(option_path), "source_tree": str(source_tree),
        "data_root": str(data_root), "work_root": str(run_root),
        "manual_seed": opt["manual_seed"], "training_hyperparameters_changed": False,
        "data_files_checked": not skip_data_check,
        "launch_training": ["bash", str(train_path)],
        "evaluate": ["bash", str(eval_path), "/absolute/path/to/checkpoint.pth"],
    }
    write_idempotent(run_root / "reproduction_manifest.json",
                     (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Existing directory containing HR/ and LR4/")
    parser.add_argument("--work-root", type=Path, required=True,
                        help="Output directory outside the source tree is recommended")
    parser.add_argument("--seed", type=int, default=None,
                        help="Default preserves the archived seed exactly")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="Generate configs only; not a successful dataset validation")
    a = parser.parse_args()
    print(json.dumps(prepare(a.model, a.data_root, a.work_root, a.seed,
                             a.skip_data_check), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
