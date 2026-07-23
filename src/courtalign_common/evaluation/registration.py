from __future__ import annotations

import ast
import csv
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def load_corner_ground_truth(path: str | Path) -> dict[str, np.ndarray]:
    corners: dict[str, np.ndarray] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = row.get("file_suffix")
            coordinates = row.get("coordinates")
            if not image_id or not coordinates:
                continue
            corners[image_id] = np.asarray(ast.literal_eval(coordinates), dtype=np.float64)
    return corners


def sort_corners_tl_tr_bl_br(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    return np.vstack([top, bottom])


def project_points(reference_points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    src = np.asarray(reference_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, np.asarray(homography, dtype=np.float64))
    return projected.reshape(-1, 2).astype(np.float64)


def corner_distances(gt_corners: np.ndarray, predicted_corners: np.ndarray) -> np.ndarray:
    gt = sort_corners_tl_tr_bl_br(gt_corners)
    pred = sort_corners_tl_tr_bl_br(predicted_corners)
    if gt.shape != pred.shape:
        raise ValueError(f"Corner shape mismatch: {gt.shape} vs {pred.shape}")
    return np.linalg.norm(gt - pred, axis=1)


def summarize_distances(distances: Iterable[float], thresholds: list[float] | None = None) -> dict:
    thresholds = thresholds or [5.0, 10.0, 20.0]
    values = np.asarray(list(distances), dtype=np.float64)
    if values.size == 0:
        return {
            "n_points": 0,
            "mean_error_px": None,
            "median_error_px": None,
            "rmse_px": None,
            **{f"pck_{int(t)}px": None for t in thresholds},
        }
    return {
        "n_points": int(values.size),
        "mean_error_px": float(np.mean(values)),
        "median_error_px": float(np.median(values)),
        "rmse_px": float(math.sqrt(np.mean(values**2))),
        **{f"pck_{int(t)}px": float(np.mean(values <= t)) for t in thresholds},
    }


def polygon_mask_from_corners(corners: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = sort_corners_tl_tr_bl_br(corners)
    tl, tr, bl, br = pts
    polygon = np.asarray([tl, tr, br, bl], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def binary_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float | None:
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: {pred.shape} vs {gt.shape}")
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return None
    intersection = np.logical_and(pred, gt).sum()
    return float(intersection / union)


def projected_footprint_iou(
    predicted_corners_fullres: np.ndarray,
    image_shape_hw: tuple[int, int],
    gt_mask: np.ndarray,
) -> float | None:
    fullres_mask = polygon_mask_from_corners(predicted_corners_fullres, image_shape_hw)
    if fullres_mask.shape != gt_mask.shape:
        fullres_mask = cv2.resize(
            fullres_mask,
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return binary_iou(fullres_mask, gt_mask > 0)


def summarize_values(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
