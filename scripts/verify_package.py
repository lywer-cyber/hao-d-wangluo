#!/usr/bin/env python3
"""Verify the staged/tracked baseline publication, sizes, exclusions and hashes.

Untracked MSRes research files may remain in a developer's working tree. They
are intentionally outside this baseline release and must not enter its index.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "metadata/package_manifest.json"
SENSITIVE_PATTERNS = [
    re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"https?://[^/\s:@]+:[^/\s@]+@"),
]


def tracked_paths():
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                         capture_output=True, text=True)
    if top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT:
        p = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
        return sorted(x.decode("utf-8") for x in p.stdout.split(b"\0") if x)
    # A downloaded Git archive has no .git; the Git-index file list is
    # replaced by the explicit publication manifest for offline validation.
    manifest = json.loads((ROOT / MANIFEST).read_text())
    return sorted([*manifest["files"], MANIFEST])


def verify(paths, check_hashes=True):
    errors = []
    total = 0
    for relative in paths:
        p = ROOT / relative
        if not p.is_file():
            errors.append(f"Missing published file: {relative}")
            continue
        if p.is_symlink():
            errors.append(f"Unexpected published symlink: {relative}")
        size = p.stat().st_size
        total += size
        if size > 20 * 1024 ** 2:
            errors.append(f"File exceeds release limit (20MiB): {relative}")
        if p.suffix.lower() in (".pth", ".pt", ".state", ".ckpt", ".safetensors", ".7z", ".zip", ".bundle", ".pem", ".key", ".pyc"):
            errors.append(f"Excluded binary/credential type: {relative}")
        if any(part in ("HR", "LR4", ".ssh", "training_states", "__pycache__") for part in p.relative_to(ROOT).parts):
            errors.append(f"Excluded dataset/credential/runtime folder: {relative}")
        if "/msres/" in relative or relative.startswith("configs/msres_v1/") or p.name == "holo_restormer_msres_arch.py":
            errors.append(f"Unfinished MSRes material outside baseline release scope: {relative}")
        if p.suffix.lower() not in (".bmp", ".png", ".jpg", ".jpeg", ".pdf"):
            data = p.read_bytes()
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(data):
                    errors.append(f"Possible credential in {relative} (value not printed)")
                    break
    if check_hashes:
        manifest = json.loads((ROOT / MANIFEST).read_text())
        actual_paths = set(paths) - {MANIFEST}
        if actual_paths != set(manifest["files"]):
            errors.append("Manifest file set differs from Git index/tracked files")
        for relative, item in manifest["files"].items():
            p = ROOT / relative
            if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                errors.append(f"Hash mismatch: {relative}")
    return errors, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write-manifest", action="store_true")
    a = p.parse_args()
    paths = tracked_paths()
    if a.write_manifest:
        paths = [p for p in paths if p != MANIFEST]
        errors, total = verify(paths, check_hashes=False)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            raise SystemExit(1)
        records = {p: {"bytes": (ROOT / p).stat().st_size,
                       "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}
                   for p in paths}
        manifest = {"schema_version": 1, "release": "baseline-snapshot-20261014",
                    "scope": "baseline code, configs, metadata, completed results and paper assets",
                    "full_dataset_included": False, "model_weights_included": False,
                    "file_count_excluding_this_manifest": len(records),
                    "total_bytes_excluding_this_manifest": total, "files": records}
        (ROOT / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"Manifest written for {len(records)} files. Stage it, then run verification again.")
        return
    errors, total = verify(paths)
    print(json.dumps({"passed": not errors, "tracked_files": len(paths),
                      "total_bytes": total, "errors": errors}, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
