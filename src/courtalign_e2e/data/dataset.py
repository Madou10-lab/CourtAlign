"""Unified CourtAlign-E2E dataset for tennis and badminton.

Yields, per image (all in MODEL-INPUT coordinates, transform-consistent):
  image      : (3,H,W) float, ImageNet-normalized RGB
  seg        : (H,W) int64 class-index mask
  lattice_uv : (K,2) supervision landmark coords
  vis        : (K,) float {0,1}
  H_sup      : (3,3) supervision homography (metric -> input px)
  weight     : () float label confidence
  meta       : image_id, sport, group

Geometry bookkeeping:
  native -> input resize:  S = diag(sx, sy, 1);  uv' = S uv;  H' = S H_native.
  augmentation warp A (small random homography + flip-free):  image warped by A,
  uv'' = A ∘ uv', H'' = A H', visibility recomputed. Photometric aug is RGB-space.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..geometry.homography_np import project
from .. import paths

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
NATIVE_W, NATIVE_H = 1920, 1080


def load_supervision(sport: str) -> list[dict]:
    f = paths.supervision_path(sport)
    return json.load(open(f))["records"]


def load_e2e_split(sport: str) -> dict:
    """image_id -> (split, group). Tennis uses the rally-group split; badminton
    uses the frozen video-based split with video group ids from metadata."""
    import csv as _csv
    if sport == "tennis":
        f = paths.tennis_group_split()
        return {r["image_id"]: (r["split"], r["group"]) for r in _csv.DictReader(open(f))}
    meta = paths.badminton_metadata()
    vid = {r["image_id"] + ".png": f"vid{r['video_id']}" for r in _csv.DictReader(open(meta))}
    out = {}
    for r in _csv.DictReader(open(paths.split_manifest("badminton_zones"))):
        out[r["image_id"]] = (r["split"], vid.get(r["image_id"], "vid_unknown"))
    return out


def _rand_aug_homography(rng: np.random.RandomState, w: int, h: int,
                         max_shift: float, max_persp: float, max_scale: float) -> np.ndarray:
    """Small random homography mapping input frame onto itself (camera jitter sim)."""
    src = np.array([(0, 0), (w, 0), (0, h), (w, h)], np.float32)
    s = 1.0 + rng.uniform(-max_scale, max_scale)
    cx, cy = w / 2, h / 2
    dst = (src - [cx, cy]) * s + [cx, cy]
    dst += rng.uniform(-max_shift, max_shift, size=(1, 2)).astype(np.float32)
    dst += rng.uniform(-max_persp, max_persp, size=(4, 2)).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32)).astype(np.float64)


class CourtAlignE2EDataset(Dataset):
    def __init__(self, sport: str, split: str, input_hw=(448, 784),
                 augment: bool = False, aug_cfg: dict | None = None, seed: int = 0,
                 lattice_metric: np.ndarray | None = None,
                 include_negatives: bool = False):
        """lattice_metric: optional override of the supervision landmark set
        for an optional centroid-based experiment. When given,
        lattice_uv is derived on the fly as project(H_sup, lattice_metric) —
        mathematically identical to how the stored lattice_uv was built.
        include_negatives: include court_visible=0 records for learned
        rejection."""
        self.sport = sport
        self.split = split
        self.h, self.w = input_hw
        self.augment = augment
        self.aug = {"max_shift_px": 24, "max_persp_px": 14, "max_scale": 0.06,
                    "color_jitter": 0.25, "blur_p": 0.15, **(aug_cfg or {})}
        self.rng = np.random.RandomState(seed)
        self.lattice_metric = (np.asarray(lattice_metric, np.float64)
                               if lattice_metric is not None else None)

        sp = load_e2e_split(sport)
        manifest = paths.manifest_index(paths.SOURCES[sport]["dataset"])
        recs = []
        for r in load_supervision(sport):
            if not include_negatives and r.get("court_visible", 1) == 0:
                continue
            s, g = sp.get(r["image_id"], (r["split"], "g_none"))
            if s == split:
                r = dict(r)
                r["group"] = g
                row = manifest.get(r["image_id"])
                if row is None:
                    raise KeyError(f"{r['image_id']} is missing from the frozen dataset manifest")
                r["image_path"] = str(paths.data_path(row["image_path"]))
                r["mask_path"] = str(paths.data_path(row["mask_path"]))
                recs.append(r)
        self.records = recs
        self.S = np.diag([self.w / NATIVE_W, self.h / NATIVE_H, 1.0])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i: int):
        r = self.records[i]
        img = cv2.imread(r["image_path"])
        if img is None:
            raise FileNotFoundError(f"Missing dataset image: {r['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((self.h, self.w), np.uint8)
        else:
            if self.sport == "tennis":
                mask = (mask > 0).astype(np.uint8)
            mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        H_ref = np.array(r["H_sup"], np.float64)
        if self.lattice_metric is not None:
            uv_ref = project(H_ref, self.lattice_metric)          # reference 1920x1080 px
            margin_px = 8.0
            vis = ((uv_ref[:, 0] >= -margin_px) & (uv_ref[:, 0] < NATIVE_W + margin_px)
                   & (uv_ref[:, 1] >= -margin_px) & (uv_ref[:, 1] < NATIVE_H + margin_px)).astype(np.float64)
            uv = uv_ref * [self.w / NATIVE_W, self.h / NATIVE_H]
        else:
            uv = (np.array(r["lattice_uv"], np.float64)
                  * [self.w / NATIVE_W, self.h / NATIVE_H])
            vis = np.array(r["vis"], np.float64)
        H = self.S @ H_ref

        if self.augment:
            A = _rand_aug_homography(self.rng, self.w, self.h,
                                     self.aug["max_shift_px"], self.aug["max_persp_px"],
                                     self.aug["max_scale"])
            img = cv2.warpPerspective(img, A.astype(np.float32), (self.w, self.h),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.warpPerspective(mask, A.astype(np.float32), (self.w, self.h),
                                       flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            uv = project(A, uv)
            H = A @ H
            # photometric (RGB)
            cj = self.aug["color_jitter"]
            gain = 1.0 + self.rng.uniform(-cj, cj, size=3)
            bias = self.rng.uniform(-18, 18, size=3)
            img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            if self.rng.rand() < self.aug["blur_p"]:
                img = cv2.GaussianBlur(img, (5, 5), 0)

        m = 8.0
        vis = vis * ((uv[:, 0] >= -m) & (uv[:, 0] < self.w + m)
                     & (uv[:, 1] >= -m) & (uv[:, 1] < self.h + m)).astype(np.float64)

        cv = int(r.get("court_visible", 1))
        if cv == 0:
            mask = np.zeros_like(mask)   # negatives: seg target unused (loss masked)
            vis = vis * 0.0
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return dict(
            image=torch.from_numpy(x.transpose(2, 0, 1)),
            seg=torch.from_numpy(mask.astype(np.int64)),
            lattice_uv=torch.from_numpy(uv.astype(np.float32)),
            vis=torch.from_numpy(vis.astype(np.float32)),
            H_sup=torch.from_numpy(H.astype(np.float32)),
            weight=torch.tensor(float(r["weight"]), dtype=torch.float32),
            court_visible=torch.tensor(float(cv), dtype=torch.float32),
            image_id=r["image_id"], group=r.get("group", "g_none"),
        )
