from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source, np.asarray(homography, dtype=np.float64))
    return projected.reshape(-1, 2).astype(np.float64)


@lru_cache(maxsize=2)
def official_line_segments(sport: str) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    specification = json.loads(
        (ROOT / "data/benchmark_gt/official/court_spec_tables.json").read_text()
    )
    key = "tennis" if sport == "tennis" else "badminton"
    return tuple(
        (tuple(map(float, start)), tuple(map(float, end)))
        for start, end in specification[key]["axis_line_segments_m"]
    )
