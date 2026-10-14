#!/usr/bin/env python3
"""Build baseline-only paper tables/figures from archived results, without a GPU.

No training data, model weights, or reconstructed 8-bit image metrics are used
for the numerical tables. Only completed_test entries are plotted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
METRICS = ("PSNR", "SSIM", "LPIPS", "RMSE", "MAE")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_validation_log(path):
    rows = {}
    current = None
    ended = False
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r"iter:\s*([\d,]+),", line)
        if match:
            current = int(match.group(1).replace(",", ""))
        if "End of training." in line:
            ended = True
        if "Validation HoloVal" not in line or ended or current is None:
            continue
        values = dict(re.findall(r"#\s*(psnr|ssim|rmse|mae):\s*([0-9.eE+-]+)", line))
        if set(values) != {"psnr", "ssim", "rmse", "mae"}:
            raise ValueError(f"Cannot parse validation line: {line}")
        if current in rows:
            raise ValueError(f"Duplicate in-training validation iteration {current}")
        rows[current] = {"iteration": current, **{k.upper(): float(v) for k, v in values.items()}}
    return [rows[i] for i in sorted(rows)]


def save_figure(fig, stem):
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"
    tables = out / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    catalog_path = REPO / "configs/baseline_catalog.json"
    catalog = json.loads(catalog_path.read_text())["models"]
    completed = [(k, v) for k, v in catalog.items() if v["status"] == "completed_test"]
    sources = {catalog_path, Path(__file__).resolve()}
    results, curves = [], {}
    for key, entry in completed:
        folder = REPO / entry["result_dir"]
        metric_path = folder / "metrics.json"
        metrics = json.loads(metric_path.read_text())
        sources.add(metric_path)
        results.append({"model": key, "label": entry["label"], "parameters": entry["params"],
                        "training_seed": 10, "test_images": metrics["count"],
                        **{m: metrics[m] for m in METRICS}})
        log = folder / "logs/training.log"
        sources.add(log)
        rows = parse_validation_log(log)
        if len(rows) != 12 or rows[-1]["iteration"] != 60000:
            raise ValueError(f"Expected 12 scheduled validation points for {key}")
        curves[key] = rows
        write_csv(tables / f"{key}_validation_curve.csv", rows,
                  ["iteration", "PSNR", "SSIM", "RMSE", "MAE"])
    write_csv(tables / "baseline_test_results.csv", results,
              ["model", "label", "parameters", "training_seed", "test_images", *METRICS])
    inventory = [{"model": k, "label": e["label"], "status": e["status"],
                  "parameters": e["params"], "config": e["config"]}
                 for k, e in catalog.items()]
    write_csv(tables / "baseline_inventory.csv", inventory,
              ["model", "label", "status", "parameters", "config"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, metric in zip(axes, ["PSNR", "SSIM"]):
        for key, entry in completed:
            rows = curves[key]
            ax.plot([r["iteration"] / 1000 for r in rows], [r[metric] for r in rows],
                    marker="o", markersize=3, label=key)
        ax.set(xlabel="Optimizer updates (thousands)", ylabel=f"Validation {metric}" + (" (dB)" if metric == "PSNR" else ""))
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Scheduled validation, training seed 10")
    save_figure(fig, figures / "baseline_validation_curves")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    for ax, metric in zip(axes, ["PSNR", "SSIM", "LPIPS"]):
        ys = [r[metric] for r in results]
        bars = ax.bar([r["model"] for r in results], ys,
                      color=["#3b6fb6", "#d78032"][:len(results)])
        ax.set_title(metric + (" (lower is better)" if metric == "LPIPS" else " (higher is better)"))
        ax.set_ylim(0, 1 if metric == "SSIM" else max(ys) * 1.17)
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in ys], padding=3, fontsize=9)
    fig.suptitle("Test-set means: 500 images; one training seed (10)")
    save_figure(fig, figures / "baseline_test_summary")

    sweep_manifest = REPO / "results/sensitivity/sweep_manifest.json"
    sweep = json.loads(sweep_manifest.read_text())
    sources.add(sweep_manifest)
    sweep_rows = []
    for p in sorted((REPO / "results/sensitivity/raw").glob("*/metrics.json")):
        parameter, value = p.parent.name.split("=", 1)
        m = json.loads(p.read_text())
        sweep_rows.append({"model": "restormer", "parameter": parameter,
                           "value": float(value), "images": m["count"],
                           **{k: m.get(k) for k in METRICS}})
        sources.add(p)
    write_csv(tables / "restormer_sensitivity.csv", sweep_rows,
              ["model", "parameter", "value", "images", *METRICS])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    for ax, parameter in zip(axes.flat, sweep["sweeps"]):
        rows = sorted((r for r in sweep_rows if r["parameter"] == parameter), key=lambda r: r["value"])
        if len(rows) != len(sweep["sweeps"][parameter]):
            raise ValueError(f"Incomplete sweep: {parameter}")
        ax.plot([r["value"] for r in rows], [r["PSNR"] for r in rows], "o-", color="#3b6fb6")
        ax.set(xlabel=parameter, ylabel="PSNR (dB)")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    fig.suptitle("Restormer sensitivity: 50 fixed images per condition (not the full test mean)")
    save_figure(fig, figures / "restormer_sensitivity")

    sample_manifest = REPO / "splits/holo_v7_baseline/visual_samples.txt"
    names = [Path(l.split()[0]).stem for l in sample_manifest.read_text().splitlines() if l.strip()]
    sources.add(sample_manifest)
    shared = REPO / "results/restormer_stockhead_60000/visual"
    cell_w, cell_h, header = 270, 240, 32
    columns = [("HR reference", shared, "HR"), ("LR preview", shared, "LR"),
               ("Restormer-SR", shared, "SR"),
               ("SwinIR", REPO / "results/swinir_60000/visual", "SR")]
    canvas = Image.new("RGB", (cell_w * 4, header + cell_h * len(names)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (title, _, _) in enumerate(columns):
        draw.text((col * cell_w + 6, 9), title, fill="black")
    visual_rows = []
    for row, name in enumerate(names):
        for col, (_, directory, tag) in enumerate(columns):
            p = directory / f"{name}_{tag}.bmp"
            sources.add(p)
            with Image.open(p) as im:
                original_size = im.size
                im = im.convert("RGB")
                im.thumbnail((cell_w - 10, cell_h - 28), Image.Resampling.BICUBIC)
                x = col * cell_w + (cell_w - im.width) // 2
                y = header + row * cell_h + 3
                canvas.paste(im, (x, y))
            draw.text((col * cell_w + 6, header + row * cell_h + cell_h - 19),
                      f"Sample {name}", fill="black")
            visual_rows.append({"sample": name, "column": col, "source": str(p.relative_to(REPO)),
                                "width": original_size[0], "height": original_size[1]})
    canvas.save(figures / "fixed_samples_overview.png")
    write_csv(tables / "visual_sources.csv", visual_rows,
              ["sample", "column", "source", "width", "height"])
    provenance = {
        "scope": "Completed baseline snapshots only; not a final proposed-method comparison",
        "test_metrics_source": "archived metrics.json computed from float predictions",
        "validation_curve_source": "normal in-training validation log lines (rounded logging precision)",
        "excluded": "post-training duplicate validation and pending baseline numerical results",
        "visualization": "full-frame previews of all 10 frozen samples; BMP inputs were already quantized",
        "historical_float_per_sample_metrics_available": False,
        "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(sources)},
        "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"test_rows": len(results), "validation_points_per_model":
                      {k: len(v) for k, v in curves.items()}, "sensitivity_points": len(sweep_rows),
                      "visual_samples": len(names), "output": str(out)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=REPO / "paper")
    a = p.parse_args()
    build(a.out.resolve())


if __name__ == "__main__":
    main()
