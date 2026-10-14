"""Paired BMP dataset for the fixed holographic SR protocol.

Unlike the upstream restoration dataset, this loader keeps the experiment's
augmentation contract explicit: horizontal flip can be enabled independently
from rotation (rotation is disabled by the shared experiment configuration).
"""

from __future__ import annotations

import os
import os.path as osp
import random

import cv2
import numpy as np
from torch.utils import data

from basicsr.data.transforms import paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor


class HoloPairedDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = int(opt["scale"])
        self.gt_size = int(opt.get("gt_size", 256))
        # A mirror changes the handedness of holographic fringes/diffraction.
        # It is opt-in only after a physical validation, and the shared
        # experiment protocol keeps it disabled.
        self.use_hflip = bool(opt.get("use_hflip", False))
        self.use_rot = bool(opt.get("use_rot", False))
        if self.opt.get("phase") in ("val", "test"):
            if opt.get("whole_image") is not True:
                raise ValueError("Holographic val/test must use whole_image=true")
            if opt.get("resize") is not False or opt.get("padding") is not False:
                raise ValueError("Holographic val/test forbid resize and padding")
            if opt.get("preserve_native_resolution") is not True:
                raise ValueError("Holographic val/test must preserve native resolution")
            if self.opt.get("phase") == "val" and opt.get("compute_lpips", False):
                raise ValueError("Validation must not compute LPIPS")
        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt["dataroot_lq"]
        self.io_backend_opt = dict(opt.get("io_backend", {"type": "disk"}))
        meta = opt.get("meta_info_file")
        if meta:
            with open(meta, "r", encoding="utf-8") as fin:
                names = [line.strip().split()[0] for line in fin if line.strip()]
        else:
            names = sorted(
                n for n in os.listdir(self.gt_folder)
                if n.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))
            )
        self.paths = [
            {"gt_path": osp.join(self.gt_folder, n),
             "lq_path": osp.join(self.lq_folder, n)}
            for n in names
        ]
        if not self.paths:
            raise RuntimeError(f"No paired images found under {self.gt_folder} and {self.lq_folder}")
        self.file_client = None

    def __len__(self):
        return len(self.paths)

    def _read(self, path, key):
        if self.file_client is None:
            backend = dict(self.io_backend_opt)
            backend_type = backend.pop("type", "disk")
            self.file_client = FileClient(backend_type, **backend)
        img = imfrombytes(self.file_client.get(path, key), float32=True)
        if img is None:
            raise RuntimeError(f"Unable to decode image: {path}")
        return img

    def __getitem__(self, index):
        pair = self.paths[index % len(self.paths)]
        gt = self._read(pair["gt_path"], "gt")
        lq = self._read(pair["lq_path"], "lq")
        if self.opt["phase"] in ("val", "test"):
            if gt.ndim != 3 or lq.ndim != 3 or gt.shape[2] != 3 or lq.shape[2] != 3:
                raise ValueError(
                    f"Holographic val/test images must be RGB HWC: {pair['gt_path']}")
            expected = (lq.shape[0] * self.scale, lq.shape[1] * self.scale)
            if tuple(gt.shape[:2]) != expected:
                raise ValueError(
                    f"Native HR/LR scale mismatch for {pair['gt_path']}: "
                    f"HR={gt.shape[:2]}, LR={lq.shape[:2]}, scale={self.scale}")
        if self.opt["phase"] == "train":
            gt, lq = paired_random_crop(gt, lq, self.gt_size, self.scale,
                                         pair["gt_path"])
            if self.use_hflip and random.random() < 0.5:
                gt = gt[:, ::-1, :]
                lq = lq[:, ::-1, :]
            if self.use_rot:
                if random.random() < 0.5:
                    gt = gt[::-1, :, :]
                    lq = lq[::-1, :, :]
                if random.random() < 0.5:
                    gt = gt.transpose(1, 0, 2)
                    lq = lq.transpose(1, 0, 2)
        gt, lq = img2tensor([gt.copy(), lq.copy()], bgr2rgb=True,
                            float32=True)
        return {"lq": lq, "gt": gt,
                "lq_path": pair["lq_path"],
                "gt_path": pair["gt_path"]}

