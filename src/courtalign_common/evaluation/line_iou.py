from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from courtalign_common.evaluation.metric_homography import project_points
from courtalign_common.evaluation.metric_templates import MetricTemplate
from courtalign_common.evaluation.registration import binary_iou


def parse_tolerances(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        tolerances = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        tolerances = [int(v) for v in value]
    if any(v < 0 for v in tolerances):
        raise ValueError(f"Line IoU tolerances must be non-negative: {tolerances}")
    return sorted(set(tolerances))


def line_segments_from_flat_points(points: Iterable[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts = [tuple(map(float, point)) for point in points]
    if len(pts) % 2 != 0:
        raise ValueError(f"Expected an even number of line endpoints, got {len(pts)}")
    return [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]


def line_segments_from_metric_template(template: MetricTemplate) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    edges: dict[tuple[tuple[float, float], tuple[float, float]], tuple[tuple[float, float], tuple[float, float]]] = {}
    for points in template.zone_polygons_tl_tr_bl_br.values():
        tl, tr, bl, br = [tuple(map(float, point)) for point in points]
        for start, end in ((tl, tr), (tr, br), (bl, br), (tl, bl)):
            key = tuple(sorted((start, end)))
            edges[key] = (start, end)
    return list(edges.values())


def rasterize_projected_lines(
    line_segments: Iterable[tuple[tuple[float, float], tuple[float, float]]],
    homography: np.ndarray,
    shape_hw: tuple[int, int],
    thickness_px: int = 1,
) -> np.ndarray:
    h, w = shape_hw
    out = np.zeros((int(h), int(w)), dtype=np.uint8)
    thickness_px = max(1, int(thickness_px))
    for start, end in line_segments:
        projected = project_points(np.asarray([start, end], dtype=np.float32), homography)
        p1 = tuple(np.round(projected[0]).astype(int).tolist())
        p2 = tuple(np.round(projected[1]).astype(int).tolist())
        # Binary metric masks require deterministic, connected pixels. Drawing
        # anti-aliased intensity 1 into uint8 rounds partial-coverage pixels to
        # zero and creates artificial gaps, especially along shallow lines.
        cv2.line(out, p1, p2, 255, thickness=thickness_px, lineType=cv2.LINE_8)
    return out > 0


def dilate_binary(mask: np.ndarray, radius_px: int) -> np.ndarray:
    binary = np.asarray(mask).astype(bool)
    radius_px = int(radius_px)
    if radius_px <= 0:
        return binary
    kernel_size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)


def line_iou_by_tolerance(
    predicted_line_mask: np.ndarray,
    gt_line_mask: np.ndarray,
    tolerances_px: Iterable[int],
) -> dict[str, float | None]:
    pred = np.asarray(predicted_line_mask).astype(bool)
    gt = np.asarray(gt_line_mask).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Line mask shape mismatch: {pred.shape} vs {gt.shape}")
    values: dict[str, float | None] = {}
    for tolerance in tolerances_px:
        pred_tol = dilate_binary(pred, int(tolerance))
        gt_tol = dilate_binary(gt, int(tolerance))
        values[str(int(tolerance))] = binary_iou(pred_tol, gt_tol)
    return values
