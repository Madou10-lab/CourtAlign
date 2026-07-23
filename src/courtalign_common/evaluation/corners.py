from __future__ import annotations

import ast
import csv
import math
from pathlib import Path

import numpy as np


def parse_points(cell: str) -> np.ndarray:
    points = ast.literal_eval(cell)
    return np.asarray(points, dtype=np.float64)


def evaluate_corner_csv(path: str | Path, thresholds: list[float] | None = None) -> dict:
    thresholds = thresholds or [5.0, 10.0, 20.0]
    distances: list[float] = []
    per_image_rmse: list[float] = []
    rows = 0

    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = parse_points(row["ground_truth"])
            pred = parse_points(row["predicted"])
            if gt.shape != pred.shape:
                raise ValueError(f"Shape mismatch for {row.get('file_suffix')}: {gt.shape} vs {pred.shape}")
            d = np.linalg.norm(gt - pred, axis=1)
            distances.extend(d.tolist())
            per_image_rmse.append(float(math.sqrt(np.mean(d**2))))
            rows += 1

    d_arr = np.asarray(distances, dtype=np.float64)
    pck = {f"pck_{int(t)}px": float(np.mean(d_arr <= t)) for t in thresholds}
    return {
        "n_images": rows,
        "n_points": int(d_arr.size),
        "mean_error_px": float(np.mean(d_arr)),
        "median_error_px": float(np.median(d_arr)),
        "rmse_px": float(math.sqrt(np.mean(d_arr**2))),
        "mean_image_rmse_px": float(np.mean(per_image_rmse)),
        **pck,
    }
