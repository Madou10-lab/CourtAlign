from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from courtalign_common.evaluation.metric_templates import MetricTemplate
from courtalign_common.evaluation.registration import binary_iou, sort_corners_tl_tr_bl_br


@dataclass(frozen=True)
class DerivedHomography:
    status: str
    skipped_reason: str | None
    metric_to_mask: np.ndarray | None
    metric_to_image: np.ndarray | None
    validation: dict
    correspondences: dict


def matrix_to_list(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return [[float(v) for v in row] for row in np.asarray(matrix)]


def scale_homography_to_image(metric_to_mask: np.ndarray, image_width: int, image_height: int, mask_shape: tuple[int, int]) -> np.ndarray:
    mask_h, mask_w = mask_shape
    scale = np.asarray(
        [
            [float(image_width) / float(mask_w), 0.0, 0.0],
            [0.0, float(image_height) / float(mask_h), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return scale @ np.asarray(metric_to_mask, dtype=np.float64)


def scale_homography_to_mask(metric_to_image: np.ndarray, image_width: int, image_height: int, mask_shape: tuple[int, int]) -> np.ndarray:
    mask_h, mask_w = mask_shape
    scale = np.asarray(
        [
            [float(mask_w) / float(image_width), 0.0, 0.0],
            [0.0, float(mask_h) / float(image_height), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return scale @ np.asarray(metric_to_image, dtype=np.float64)


def homography_from_corners(metric_corners_tl_tr_bl_br: np.ndarray, image_corners_tl_tr_bl_br: np.ndarray) -> np.ndarray:
    src = np.asarray(metric_corners_tl_tr_bl_br, dtype=np.float32).reshape(4, 2)
    dst = sort_corners_tl_tr_bl_br(np.asarray(image_corners_tl_tr_bl_br, dtype=np.float32)).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, np.asarray(homography, dtype=np.float64))
    return projected.reshape(-1, 2).astype(np.float64)


def rasterize_template(
    template: MetricTemplate,
    metric_to_mask: np.ndarray,
    shape: tuple[int, int],
    class_ids: Iterable[int] | None = None,
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    selected = set(class_ids) if class_ids is not None else set(template.zone_polygons_tl_tr_bl_br)
    for class_id, points in template.zone_polygons_tl_tr_bl_br.items():
        if class_id not in selected:
            continue
        metric_polygon = np.asarray(points, dtype=np.float32)
        projected = project_points(metric_polygon, metric_to_mask)
        fill_order = projected[[0, 1, 3, 2]]
        cv2.fillPoly(out, [np.round(fill_order).astype(np.int32)], int(class_id))
    return out


def template_canvas(template: MetricTemplate, max_side_px: int = 1000) -> tuple[tuple[int, int], np.ndarray]:
    width_m = float(template.width_m)
    length_m = float(template.length_m)
    if width_m <= 0.0 or length_m <= 0.0:
        raise ValueError("Template dimensions must be positive")
    if length_m >= width_m:
        height = int(max_side_px)
        width = max(2, int(round(max_side_px * width_m / length_m)))
    else:
        width = int(max_side_px)
        height = max(2, int(round(max_side_px * length_m / width_m)))
    metric_to_canvas = np.asarray(
        [
            [(width - 1) / width_m, 0.0, 0.0],
            [0.0, (height - 1) / length_m, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return (height, width), metric_to_canvas


def template_space_iou_from_homographies(
    predicted_metric_to_mask: np.ndarray,
    gt_metric_to_mask: np.ndarray,
    template: MetricTemplate,
    class_ids: Iterable[int] | None = None,
    max_side_px: int = 1000,
) -> dict:
    class_ids = sorted(set(class_ids) if class_ids is not None else set(template.zone_polygons_tl_tr_bl_br))
    shape, metric_to_canvas = template_canvas(template, max_side_px=max_side_px)
    gt_metric_to_pred_metric = np.linalg.inv(np.asarray(predicted_metric_to_mask, dtype=np.float64)) @ np.asarray(
        gt_metric_to_mask, dtype=np.float64
    )
    reference_mask = rasterize_template(template, metric_to_canvas, shape, class_ids=class_ids)
    warped_gt_mask = rasterize_template(
        template,
        metric_to_canvas @ gt_metric_to_pred_metric,
        shape,
        class_ids=class_ids,
    )
    per_class_iou, zone_miou = multiclass_iou(warped_gt_mask, reference_mask, class_ids)
    return {
        "canvas_shape": [int(shape[0]), int(shape[1])],
        "fullcourt_occupancy_iou": binary_iou(warped_gt_mask > 0, reference_mask > 0),
        "zone_miou": zone_miou,
        "per_class_iou": {str(k): v for k, v in per_class_iou.items()},
        "definition": (
            "IoU in canonical template space. The GT template is transformed by "
            "inverse(predicted_metric_to_mask) @ gt_metric_to_mask and compared with the unwarped template."
        ),
    }


def multiclass_iou(prediction: np.ndarray, target: np.ndarray, class_ids: Iterable[int]) -> tuple[dict[int, float | None], float | None]:
    per_class: dict[int, float | None] = {}
    values: list[float] = []
    for class_id in class_ids:
        pred = prediction == class_id
        gt = target == class_id
        union = np.logical_or(pred, gt).sum()
        if union == 0:
            per_class[int(class_id)] = None
            continue
        value = float(np.logical_and(pred, gt).sum() / union)
        per_class[int(class_id)] = value
        values.append(value)
    return per_class, float(np.mean(values)) if values else None


def summarize_error(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "rmse": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(math.sqrt(np.mean(values**2))),
        "max": float(np.max(values)),
    }


def sample_mask_points(mask: np.ndarray, max_points: int = 2500) -> np.ndarray:
    ys_xs = np.argwhere(mask.astype(bool))
    if ys_xs.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(ys_xs) > max_points:
        indices = np.linspace(0, len(ys_xs) - 1, int(max_points)).round().astype(int)
        ys_xs = ys_xs[indices]
    return np.column_stack([ys_xs[:, 1], ys_xs[:, 0]]).astype(np.float64)


def sample_metric_grid(width_m: float, length_m: float, max_points: int = 2500) -> np.ndarray:
    aspect = max(float(length_m) / max(float(width_m), 1e-9), 1e-9)
    nx = max(2, int(round(math.sqrt(max_points / aspect))))
    ny = max(2, int(round(nx * aspect)))
    while nx * ny > max_points and ny > 2:
        ny -= 1
    xs = np.linspace(0.0, float(width_m), nx)
    ys = np.linspace(0.0, float(length_m), ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)]).astype(np.float64)


def projection_error_meters_from_mask(
    predicted_metric_to_mask: np.ndarray,
    gt_metric_to_mask: np.ndarray,
    gt_visible_mask: np.ndarray,
    max_points: int = 2500,
) -> dict:
    points_mask = sample_mask_points(gt_visible_mask, max_points=max_points)
    if points_mask.size == 0:
        return {"distances_m": [], "summary": summarize_error(np.asarray([]))}
    pred_mask_to_metric = np.linalg.inv(np.asarray(predicted_metric_to_mask, dtype=np.float64))
    gt_mask_to_metric = np.linalg.inv(np.asarray(gt_metric_to_mask, dtype=np.float64))
    pred_metric = project_points(points_mask, pred_mask_to_metric)
    gt_metric = project_points(points_mask, gt_mask_to_metric)
    distances = np.linalg.norm(pred_metric - gt_metric, axis=1)
    return {"distances_m": distances.astype(float).tolist(), "summary": summarize_error(distances)}


def reprojection_error_pixels_from_template(
    predicted_metric_to_mask: np.ndarray,
    gt_metric_to_mask: np.ndarray,
    template: MetricTemplate,
    mask_height: int,
    max_points: int = 2500,
) -> dict:
    points_metric = sample_metric_grid(template.width_m, template.length_m, max_points=max_points)
    pred_mask = project_points(points_metric, predicted_metric_to_mask)
    gt_mask = project_points(points_metric, gt_metric_to_mask)
    distances = np.linalg.norm(pred_mask - gt_mask, axis=1)
    normalized = distances / float(mask_height)
    return {
        "distances_px": distances.astype(float).tolist(),
        "distances_normalized_by_mask_height": normalized.astype(float).tolist(),
        "summary_px": summarize_error(distances),
        "summary_normalized": summarize_error(normalized),
    }


def extract_zone_quad_tl_tr_bl_br(
    label_mask: np.ndarray,
    class_id: int,
    min_area_px: float = 50.0,
    closing_iterations: int = 2,
    erosion_iterations: int = 1,
) -> tuple[np.ndarray | None, dict]:
    from courtalign_2s.registration.corner_extraction import extract_corners_from_binary_mask

    binary = (label_mask == int(class_id)).astype(np.uint8)
    extraction = extract_corners_from_binary_mask(
        binary,
        frame_shape_hw=label_mask.shape,
        closing_iterations=closing_iterations,
        erosion_iterations=erosion_iterations,
        min_area_px=min_area_px,
    )
    if extraction.corners_tl_tr_bl_br is None:
        return None, {"status": extraction.status, **extraction.details}
    return extraction.corners_tl_tr_bl_br.astype(np.float64), {"status": "ok", **extraction.details}


def derive_badminton_homography_from_zone_mask(
    label_mask: np.ndarray,
    template: MetricTemplate,
    image_width: int,
    image_height: int,
    ransac_reprojection_threshold_px: float = 8.0,
    min_classes: int = 8,
    min_inlier_ratio: float = 0.75,
    min_zone_miou: float = 0.88,
    min_fullcourt_iou: float = 0.94,
) -> DerivedHomography:
    class_ids = sorted(template.zone_polygons_tl_tr_bl_br)
    present_classes = sorted(int(v) for v in np.unique(label_mask) if int(v) in class_ids)
    if len(present_classes) == 0:
        return DerivedHomography(
            status="skipped",
            skipped_reason="mask_has_no_badminton_zone_labels",
            metric_to_mask=None,
            metric_to_image=None,
            validation={"present_classes": present_classes},
            correspondences={"n_points": 0, "class_status": {}},
        )

    src_points: list[list[float]] = []
    dst_points: list[list[float]] = []
    class_status: dict[int, dict] = {}
    template_points_by_class: dict[str, list[list[float]]] = {}
    image_points_by_class: dict[str, list[list[float]]] = {}
    used_classes: list[int] = []

    for class_id in class_ids:
        quad, status = extract_zone_quad_tl_tr_bl_br(label_mask, class_id)
        class_status[class_id] = status
        if quad is None:
            continue
        template_quad = np.asarray(template.zone_polygons_tl_tr_bl_br[class_id], dtype=np.float64)
        src_points.extend(template_quad.tolist())
        dst_points.extend(quad.tolist())
        template_points_by_class[str(class_id)] = template_quad.astype(float).tolist()
        image_points_by_class[str(class_id)] = quad.astype(float).tolist()
        used_classes.append(class_id)

    if len(used_classes) < min_classes or len(src_points) < 4:
        return DerivedHomography(
            status="skipped",
            skipped_reason="not_enough_valid_zone_quadrilaterals",
            metric_to_mask=None,
            metric_to_image=None,
            validation={"present_classes": present_classes, "used_classes": used_classes},
            correspondences={"n_points": len(src_points), "class_status": class_status},
        )

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
        return DerivedHomography(
            status="failed",
            skipped_reason="findHomography_failed",
            metric_to_mask=None,
            metric_to_image=None,
            validation={"present_classes": present_classes, "used_classes": used_classes},
            correspondences={"n_points": len(src_points), "class_status": class_status},
        )

    inlier_mask = inliers.reshape(-1).astype(bool)
    projected = project_points(src, metric_to_mask)
    residuals = np.linalg.norm(projected - dst.astype(np.float64), axis=1)
    inlier_residuals = residuals[inlier_mask]

    rendered = rasterize_template(template, metric_to_mask, label_mask.shape, class_ids=class_ids)
    per_class_iou, zone_miou = multiclass_iou(rendered, label_mask, class_ids)
    fullcourt_iou = binary_iou(rendered > 0, label_mask > 0)
    inlier_ratio = float(np.mean(inlier_mask))

    metric_to_image = scale_homography_to_image(metric_to_mask, image_width, image_height, label_mask.shape)
    passed = (
        inlier_ratio >= min_inlier_ratio
        and zone_miou is not None
        and zone_miou >= min_zone_miou
        and fullcourt_iou is not None
        and fullcourt_iou >= min_fullcourt_iou
    )

    validation = {
        "present_classes": present_classes,
        "used_classes": used_classes,
        "n_correspondence_points": int(len(src_points)),
        "n_inlier_points": int(inlier_mask.sum()),
        "inlier_ratio": inlier_ratio,
        "residual_px_all": summarize_error(residuals),
        "residual_px_inliers": summarize_error(inlier_residuals),
        "zone_miou": zone_miou,
        "fullcourt_iou": fullcourt_iou,
        "per_class_iou": {str(k): v for k, v in per_class_iou.items()},
        "thresholds": {
            "ransac_reprojection_threshold_px": float(ransac_reprojection_threshold_px),
            "min_classes": int(min_classes),
            "min_inlier_ratio": float(min_inlier_ratio),
            "min_zone_miou": float(min_zone_miou),
            "min_fullcourt_iou": float(min_fullcourt_iou),
        },
    }

    return DerivedHomography(
        status="valid" if passed else "failed_validation",
        skipped_reason=None if passed else "validation_threshold_not_met",
        metric_to_mask=metric_to_mask.astype(np.float64),
        metric_to_image=metric_to_image.astype(np.float64),
        validation=validation,
        correspondences={
            "n_points": len(src_points),
            "class_status": class_status,
            "template_points_by_class": template_points_by_class,
            "image_points_by_class": image_points_by_class,
            "source_points_metric": src.astype(float).tolist(),
            "destination_points_mask": dst.astype(float).tolist(),
            "inlier_mask": inlier_mask.tolist(),
        },
    )
