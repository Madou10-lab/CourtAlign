from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from courtalign_common.evaluation.line_iou import line_iou_by_tolerance, rasterize_projected_lines
from courtalign_common.evaluation.metric_homography import (
    extract_zone_quad_tl_tr_bl_br,
    multiclass_iou,
    project_points,
    scale_homography_to_image,
)
from courtalign_common.evaluation.registration import binary_iou
from courtalign_common.utils.io import load_json


@dataclass(frozen=True)
class OfficialCourtSpec:
    dataset_id: str
    sport: str
    width_m: float
    length_m: float
    zones: dict[int, np.ndarray]
    landmarks: np.ndarray
    line_segments: list[tuple[tuple[float, float], tuple[float, float]]]

    @property
    def outer_corners(self) -> np.ndarray:
        return np.asarray(
            [(0.0, 0.0), (self.width_m, 0.0), (0.0, self.length_m), (self.width_m, self.length_m)],
            dtype=np.float64,
        )


def handling_outcome(gt_visible: bool, prediction_valid: bool) -> dict[str, int]:
    """Return the protocol counters for one complete-manifest prediction."""
    return {
        "visible": int(gt_visible),
        "nonvisible": int(not gt_visible),
        "visible_success": int(gt_visible and prediction_valid),
        "false_registration": int(not gt_visible and prediction_valid),
        "correct_handling": int(
            (gt_visible and prediction_valid) or (not gt_visible and not prediction_valid)
        ),
    }


def load_official_spec(path: str | Path, dataset_id: str) -> OfficialCourtSpec:
    raw = load_json(path)
    key = "tennis" if dataset_id == "tennis_fullcourt" else "badminton"
    sport = raw[key]
    width_m, length_m = map(float, sport["frame_m"])
    if dataset_id == "tennis_fullcourt":
        zones = {1: np.asarray([(0, 0), (width_m, 0), (0, length_m), (width_m, length_m)], dtype=np.float64)}
    else:
        zones = {int(k): np.asarray(v, dtype=np.float64) for k, v in sport["zones_inner_edges_m"].items()}
    return OfficialCourtSpec(
        dataset_id=dataset_id,
        sport=key,
        width_m=width_m,
        length_m=length_m,
        zones=zones,
        landmarks=np.asarray(sport["axis_landmarks_m"], dtype=np.float64),
        line_segments=[(tuple(a), tuple(b)) for a, b in sport["axis_line_segments_m"]],
    )


def normalize_homography(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Homography must be a finite 3x3 matrix")
    scale = matrix[2, 2]
    if abs(scale) < 1e-12:
        scale = np.linalg.norm(matrix)
    if abs(scale) < 1e-12:
        raise ValueError("Degenerate homography")
    return matrix / scale


def matrix_to_list(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return normalize_homography(matrix).tolist()


def rasterize_zones(spec: OfficialCourtSpec, metric_to_target: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    for class_id, quad in spec.zones.items():
        projected = project_points(quad, metric_to_target)
        polygon = np.round(projected[[0, 1, 3, 2]]).astype(np.int32)
        cv2.fillPoly(out, [polygon], int(class_id))
    return out


def rasterize_fullcourt(
    spec: OfficialCourtSpec,
    metric_to_target: np.ndarray,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape_hw, dtype=np.uint8)
    projected = project_points(spec.outer_corners, metric_to_target)
    polygon = np.round(projected[[0, 1, 3, 2]]).astype(np.int32)
    cv2.fillPoly(out, [polygon], 1)
    return out


def estimate_badminton_homography(
    label_mask: np.ndarray,
    spec: OfficialCourtSpec,
    *,
    image_width: int,
    image_height: int,
    ransac_reprojection_threshold_px: float,
    min_classes: int,
    min_inlier_ratio: float,
    min_zone_miou: float,
    min_fullcourt_iou: float,
) -> dict:
    src_points: list[list[float]] = []
    dst_points: list[list[float]] = []
    used_classes: list[int] = []
    class_status: dict[str, dict] = {}
    for class_id, metric_quad in sorted(spec.zones.items()):
        image_quad, status = extract_zone_quad_tl_tr_bl_br(label_mask, class_id)
        class_status[str(class_id)] = status
        if image_quad is None:
            continue
        src_points.extend(metric_quad.tolist())
        dst_points.extend(image_quad.tolist())
        used_classes.append(class_id)

    if len(used_classes) < int(min_classes):
        return {
            "status": "skipped",
            "skipped_reason": "not_enough_valid_zone_quadrilaterals",
            "metric_to_mask": None,
            "metric_to_image": None,
            "validation": {"used_classes": used_classes, "class_status": class_status},
        }

    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points, dtype=np.float32)
    metric_to_mask, inliers = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        ransacReprojThreshold=float(ransac_reprojection_threshold_px),
        maxIters=5000,
    )
    if metric_to_mask is None or inliers is None:
        return {
            "status": "failed",
            "skipped_reason": "findHomography_failed",
            "metric_to_mask": None,
            "metric_to_image": None,
            "validation": {"used_classes": used_classes, "class_status": class_status},
        }

    inlier_mask = inliers.reshape(-1).astype(bool)
    rendered = rasterize_zones(spec, metric_to_mask, label_mask.shape)
    per_class, zone_miou = multiclass_iou(rendered, label_mask, spec.zones)
    fullcourt_iou = binary_iou(rendered > 0, label_mask > 0)
    inlier_ratio = float(np.mean(inlier_mask))
    projected = project_points(src, metric_to_mask)
    residuals = np.linalg.norm(projected - dst, axis=1)
    passed = (
        inlier_ratio >= float(min_inlier_ratio)
        and zone_miou is not None
        and zone_miou >= float(min_zone_miou)
        and fullcourt_iou is not None
        and fullcourt_iou >= float(min_fullcourt_iou)
    )
    metric_to_image = scale_homography_to_image(metric_to_mask, image_width, image_height, label_mask.shape)
    return {
        "status": "valid" if passed else "rejected",
        "skipped_reason": None if passed else "validation_thresholds_not_met",
        "metric_to_mask": matrix_to_list(metric_to_mask),
        "metric_to_image": matrix_to_list(metric_to_image),
        "validation": {
            "used_classes": used_classes,
            "n_correspondences": int(len(src)),
            "n_inliers": int(inlier_mask.sum()),
            "inlier_ratio": inlier_ratio,
            "residual_rmse_mask_px": float(np.sqrt(np.mean(residuals[inlier_mask] ** 2))),
            "fullcourt_iou": fullcourt_iou,
            "zone_miou": zone_miou,
            "per_class_iou": {str(k): v for k, v in per_class.items()},
            "thresholds": {
                "ransac_reprojection_threshold_px": float(ransac_reprojection_threshold_px),
                "min_classes": int(min_classes),
                "min_inlier_ratio": float(min_inlier_ratio),
                "min_zone_miou": float(min_zone_miou),
                "min_fullcourt_iou": float(min_fullcourt_iou),
            },
        },
    }


def summarize(values: Iterable[float]) -> dict:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {
            "n_images": 0,
            "mean": None,
            "median": None,
            "rmse": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "n_images": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def deterministic_mask_sample(mask: np.ndarray, n_points: int) -> np.ndarray:
    yx = np.argwhere(np.asarray(mask).astype(bool))
    if len(yx) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(yx) > n_points:
        indices = np.linspace(0, len(yx) - 1, n_points).round().astype(int)
        yx = yx[indices]
    return np.column_stack((yx[:, 1], yx[:, 0])).astype(np.float64)


def dense_metric_grid(spec: OfficialCourtSpec, shape: tuple[int, int]) -> np.ndarray:
    nx, ny = map(int, shape)
    xs = np.linspace(0.0, spec.width_m, nx)
    ys = np.linspace(0.0, spec.length_m, ny)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)


def template_canvas(spec: OfficialCourtSpec, pixels_per_meter: int) -> tuple[tuple[int, int], np.ndarray]:
    width = max(2, int(round(spec.width_m * pixels_per_meter)) + 1)
    height = max(2, int(round(spec.length_m * pixels_per_meter)) + 1)
    metric_to_canvas = np.asarray(
        [[pixels_per_meter, 0, 0], [0, pixels_per_meter, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return (height, width), metric_to_canvas


def iou_part(
    predicted_metric_to_mask: np.ndarray,
    gt_metric_to_mask: np.ndarray,
    visible_mask: np.ndarray,
    spec: OfficialCourtSpec,
    pixels_per_meter: int,
) -> float | None:
    shape, metric_to_canvas = template_canvas(spec, pixels_per_meter)
    pred_mask_to_canvas = metric_to_canvas @ np.linalg.inv(predicted_metric_to_mask)
    gt_mask_to_canvas = metric_to_canvas @ np.linalg.inv(gt_metric_to_mask)
    src = np.asarray(visible_mask).astype(np.uint8)
    size = (shape[1], shape[0])
    pred_support = cv2.warpPerspective(src, pred_mask_to_canvas, size, flags=cv2.INTER_NEAREST) > 0
    gt_support = cv2.warpPerspective(src, gt_mask_to_canvas, size, flags=cv2.INTER_NEAREST) > 0
    return binary_iou(pred_support, gt_support)


def iou_whole(
    predicted_metric_to_image: np.ndarray,
    gt_metric_to_image: np.ndarray,
    spec: OfficialCourtSpec,
    pixels_per_meter: int,
) -> float | None:
    shape, metric_to_canvas = template_canvas(spec, pixels_per_meter)
    reference = rasterize_fullcourt(spec, metric_to_canvas, shape) > 0
    gt_to_pred_metric = np.linalg.inv(predicted_metric_to_image) @ gt_metric_to_image
    warped = rasterize_fullcourt(spec, metric_to_canvas @ gt_to_pred_metric, shape) > 0
    return binary_iou(warped, reference)


def geometric_errors(
    predicted_metric_to_mask: np.ndarray,
    gt_metric_to_mask: np.ndarray,
    predicted_metric_to_image: np.ndarray,
    gt_metric_to_image: np.ndarray,
    visible_mask: np.ndarray,
    spec: OfficialCourtSpec,
    projection_samples: int,
    reprojection_grid: tuple[int, int],
    image_height: int,
) -> dict:
    mask_points = deterministic_mask_sample(visible_mask, projection_samples)
    pred_metric = project_points(mask_points, np.linalg.inv(predicted_metric_to_mask))
    gt_metric = project_points(mask_points, np.linalg.inv(gt_metric_to_mask))
    projection = np.linalg.norm(pred_metric - gt_metric, axis=1)

    metric_points = dense_metric_grid(spec, reprojection_grid)
    pred_image = project_points(metric_points, predicted_metric_to_image)
    gt_image = project_points(metric_points, gt_metric_to_image)
    reprojection = np.linalg.norm(pred_image - gt_image, axis=1)
    return {
        "projection_mean_m": float(np.mean(projection)),
        "projection_rmse_m": float(np.sqrt(np.mean(projection**2))),
        "reprojection_mean_px": float(np.mean(reprojection)),
        "reprojection_rmse_px": float(np.sqrt(np.mean(reprojection**2))),
        "reprojection_mean_height_normalized": float(np.mean(reprojection) / image_height),
        "reprojection_rmse_height_normalized": float(np.sqrt(np.mean(reprojection**2)) / image_height),
        "n_projection_points": int(len(projection)),
        "n_reprojection_points": int(len(reprojection)),
    }


def pck_h(
    predicted_metric_to_image: np.ndarray,
    gt_metric_to_image: np.ndarray,
    landmarks: np.ndarray,
    native_wh: tuple[int, int],
    pixel_thresholds: Iterable[float],
    diagonal_thresholds: Iterable[float],
) -> dict:
    predicted = project_points(landmarks, predicted_metric_to_image)
    ground_truth = project_points(landmarks, gt_metric_to_image)
    distances = np.linalg.norm(predicted - ground_truth, axis=1)
    diagonal = math.hypot(*native_wh)
    result = {
        "mean_error_px": float(np.mean(distances)),
        "rmse_error_px": float(np.sqrt(np.mean(distances**2))),
        "n_landmarks": int(len(distances)),
    }
    for threshold in pixel_thresholds:
        result[f"pck_{int(threshold)}px"] = float(np.mean(distances <= threshold))
    for threshold in diagonal_thresholds:
        result[f"pck_{int(round(threshold * 100))}pct_diag"] = float(np.mean(distances <= threshold * diagonal))
    return result


def line_iou(
    predicted_metric_to_image: np.ndarray,
    gt_line_mask: np.ndarray,
    spec: OfficialCourtSpec,
    native_wh: tuple[int, int],
    tolerances: Iterable[int],
    thickness_px: int,
) -> dict[str, float | None]:
    width, height = native_wh
    predicted = rasterize_projected_lines(
        spec.line_segments,
        predicted_metric_to_image,
        (height, width),
        thickness_px=thickness_px,
    )
    ground_truth = np.asarray(gt_line_mask)
    if ground_truth.shape != (height, width):
        ground_truth = cv2.resize(ground_truth, (width, height), interpolation=cv2.INTER_NEAREST)
    return line_iou_by_tolerance(predicted, ground_truth > 0, tolerances)
