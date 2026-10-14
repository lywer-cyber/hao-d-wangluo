import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("prepare", ROOT / "scripts/prepare_reproduction.py")
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


class ReproductionPackage(unittest.TestCase):
    def test_all_baseline_configs_relocate_without_training_changes(self):
        catalog = json.loads((ROOT / "configs/baseline_catalog.json").read_text())["models"]
        with tempfile.TemporaryDirectory(prefix="holo repro ") as d:
            for name, entry in catalog.items():
                result = PREP.prepare(name, Path(d) / "external data", Path(d) / "runs",
                                      skip_data_check=True)
                actual = yaml.safe_load(Path(result["generated_option"]).read_text())
                original = yaml.safe_load((ROOT / entry["config"]).read_text())
                self.assertEqual(actual["train"], original["train"])
                self.assertEqual(actual["network_g"], original["network_g"])
                self.assertEqual(actual["runtime"], original["runtime"])
                self.assertEqual(actual["evaluation"], original["evaluation"])
                self.assertEqual(actual["manual_seed"], original["manual_seed"])
                self.assertEqual(actual["name"], original["name"])
                self.assertTrue(Path(actual["visual_output"]["save_list"]).exists())
                for path in [result["launch_training"][1], result["evaluate"][1]]:
                    subprocess.run(["bash", "-n", path], check=True)
                again = PREP.prepare(name, Path(d) / "external data", Path(d) / "runs",
                                     skip_data_check=True)
                self.assertEqual(result, again)

    def test_seed_override_is_explicit_and_preserves_training(self):
        with tempfile.TemporaryDirectory() as d:
            result = PREP.prepare("restormer", Path(d) / "data", Path(d) / "runs",
                                  seed=20, skip_data_check=True)
            actual = yaml.safe_load(Path(result["generated_option"]).read_text())
            self.assertEqual(actual["manual_seed"], 20)
            self.assertIn("_seed20_", actual["name"])
            self.assertEqual(actual["train"]["total_iter"], 60000)

    def test_missing_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                PREP.prepare("restormer", Path(d) / "missing", Path(d) / "runs")
            self.assertFalse((Path(d) / "runs").exists())

    def test_config_overwrite_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            result = PREP.prepare("restormer", Path(d) / "data", Path(d) / "runs",
                                  skip_data_check=True)
            p = Path(result["generated_option"])
            p.write_text(p.read_text() + "# changed\n")
            with self.assertRaises(FileExistsError):
                PREP.prepare("restormer", Path(d) / "data", Path(d) / "runs",
                             skip_data_check=True)

    def test_splits_and_shared_visuals(self):
        for project in ["holo_v7_baseline", "holo_project"]:
            split = ROOT / "splits" / project
            groups = {phase: [l.split()[0] for l in (split / (phase + "_meta.txt")).read_text().splitlines()
                              if l.strip()] for phase in ["train", "val", "test"]}
            self.assertEqual([len(groups[p]) for p in ["train", "val", "test"]], [4000, 500, 500])
            self.assertEqual(len(set(sum(groups.values(), []))), 5000)
            visual = [l.split()[0] for l in (split / "visual_samples.txt").read_text().splitlines() if l.strip()]
            self.assertEqual(len(visual), 10)
            self.assertTrue(set(visual) <= set(groups["test"]))
        a = json.loads((ROOT / "metadata/visual_reference_hashes_A.json").read_text())
        b = json.loads((ROOT / "metadata/visual_reference_hashes_B.json").read_text())
        self.assertEqual({r["filename"]: r["sha256"] for r in a},
                         {r["filename"]: r["sha256"] for r in b})

    def test_completed_metrics_and_images(self):
        catalog = json.loads((ROOT / "configs/baseline_catalog.json").read_text())["models"]
        shared = ROOT / "results/restormer_stockhead_60000/visual"
        for name, entry in catalog.items():
            if entry["status"] != "completed_test":
                self.assertIsNone(entry["result_dir"])
                continue
            folder = ROOT / entry["result_dir"]
            metrics = json.loads((folder / "metrics.json").read_text())
            self.assertEqual(metrics["count"], 500)
            for m in ["PSNR", "SSIM", "LPIPS", "RMSE", "MAE"]:
                self.assertTrue(math.isfinite(metrics[m]))
            images = sorted((folder / "visual").glob("*_SR.bmp"))
            self.assertEqual(len(images), 10)
            for p in images:
                stem = p.name.removesuffix("_SR.bmp") if hasattr(str, "removesuffix") else p.name[:-7]
                with Image.open(p) as sr, Image.open(shared / (stem + "_HR.bmp")) as hr, Image.open(shared / (stem + "_LR.bmp")) as lr:
                    sr.load(); hr.load(); lr.load()
                    self.assertEqual(sr.mode, "RGB")
                    self.assertEqual(sr.size, hr.size)
                    self.assertEqual(hr.size, (4 * lr.width, 4 * lr.height))

    def test_paper_provenance(self):
        import hashlib
        provenance = json.loads((ROOT / "paper/provenance.json").read_text())
        for path, digest in provenance["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        self.assertFalse(provenance["historical_float_per_sample_metrics_available"])
        self.assertEqual(len(list((ROOT / "results/sensitivity/raw").glob("*/metrics.json"))), 27)


if __name__ == "__main__":
    unittest.main()
