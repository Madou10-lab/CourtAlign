from __future__ import annotations

import cv2
import numpy as np


TENNIS_REGION_COLORS_RGB = np.asarray(
    [
        [0, 0, 0],
        [255, 190, 170],
    ],
    dtype=np.uint8,
)

BADMINTON_REGION_COLORS_RGB = np.asarray(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
    ],
    dtype=np.uint8,
)


def overlay_registration(
    image_bgr: np.ndarray,
    projected_region_mask: np.ndarray | None,
    palette_rgb: np.ndarray,
    *,
    projected_line_mask: np.ndarray | None = None,
    region_alpha: float = 0.45,
    display_line_radius_px: int = 1,
    line_color_bgr: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Create a native-resolution qualitative overlay without changing metric masks."""
    output = image_bgr.copy()
    image_h, image_w = output.shape[:2]

    if projected_region_mask is not None:
        labels = np.asarray(projected_region_mask, dtype=np.uint8)
        if labels.shape[:2] != (image_h, image_w):
            labels = cv2.resize(labels, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        colors_rgb = palette_rgb[np.clip(labels.astype(np.int64), 0, len(palette_rgb) - 1)]
        colors_bgr = cv2.cvtColor(colors_rgb, cv2.COLOR_RGB2BGR)
        visible = labels > 0
        if np.any(visible):
            output[visible] = cv2.addWeighted(
                image_bgr[visible],
                1.0 - float(region_alpha),
                colors_bgr[visible],
                float(region_alpha),
                0.0,
            )

    if projected_line_mask is not None:
        lines = np.asarray(projected_line_mask).astype(np.uint8)
        if lines.shape[:2] != (image_h, image_w):
            lines = cv2.resize(lines, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        if display_line_radius_px > 0:
            radius = int(display_line_radius_px)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            lines = cv2.dilate(lines, kernel, iterations=1)
        output[lines > 0] = np.asarray(line_color_bgr, dtype=np.uint8)

    return output
