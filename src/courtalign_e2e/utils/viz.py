"""Qualitative overlays: input | seg | landmarks | projected court (pred vs sup)."""

from __future__ import annotations

import numpy as np
import cv2

from ..geometry.homography_np import project


def draw_court(img, H, lines_metric, color, thickness=2):
    out = img
    for (a, b) in lines_metric:
        p = project(H, np.array([a, b], np.float64))
        cv2.line(out, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)),
                 color, thickness)
    return out


def overlay_panel(image_rgb, seg_idx, coords, conf, H_pred, H_sup, lines_metric,
                  n_classes: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    base = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    rng = np.random.RandomState(0)
    palette = np.vstack([[0, 0, 0], rng.randint(40, 255, size=(max(n_classes - 1, 1), 3))]).astype(np.uint8)
    seg_vis = cv2.addWeighted(base, 0.6, palette[np.clip(seg_idx, 0, n_classes - 1)], 0.4, 0)

    lm = base.copy()
    for (x, y), c in zip(coords, conf):
        col = (0, int(255 * float(c)), int(255 * (1 - float(c))))
        cv2.circle(lm, (int(x), int(y)), 4, col, -1)

    proj = base.copy()
    if H_sup is not None:
        proj = draw_court(proj, H_sup, lines_metric, (0, 200, 0), 2)      # green = supervision
    if H_pred is not None:
        proj = draw_court(proj, H_pred, lines_metric, (0, 215, 255), 2)   # yellow = predicted

    top = np.hstack([base, seg_vis])
    bot = np.hstack([lm, proj])
    return np.vstack([top, bot])
