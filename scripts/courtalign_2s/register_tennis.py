#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import numpy as np

from courtalign_common.data.manifest import load_manifest
from courtalign_common.evaluation.metric_homography import (
    homography_from_corners,
    projection_error_meters_from_mask,
    reprojection_error_pixels_from_template,
    scale_homography_to_mask,
    template_space_iou_from_homographies,
)
from courtalign_common.evaluation.line_iou import (
    line_iou_by_tolerance,
    line_segments_from_flat_points,
    parse_tolerances,
    rasterize_projected_lines,
)
from courtalign_common.evaluation.metric_templates import template_from_config
from courtalign_common.evaluation.registration import (
    corner_distances,
    load_corner_ground_truth,
    project_points,
    projected_footprint_iou,
    summarize_distances,
    summarize_values,
)
from courtalign_2s.experiment import environment_snapshot
from courtalign_2s.registration import CourtAlign2STennisFullCourtRegistration
from courtalign_common.evaluation.visual_outputs import TENNIS_REGION_COLORS_RGB, overlay_registration
from courtalign_common.utils.io import load_json, save_json
from courtalign_common.utils.paths import resolve_repo_path


def matrix_to_list(matrix):
    if matrix is None:
        return None
    return [[float(v) for v in row] for row in matrix]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CourtAlign-2S tennis segmentation and homography registration.")
    parser.add_argument("--config", default="configs/registration/courtalign_2s_tennis.json")
    parser.add_argument("--checkpoint", default=None, help="Override segmentation checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--corner-gt", default="data/tennis_fullcourt/corners/ground_truth_corners.csv")
    parser.add_argument("--metric-template-config", default="configs/evaluation/metric_templates.json")
    parser.add_argument("--metric-max-samples", type=int, default=2500)
    parser.add_argument("--save-projected-footprints", action="store_true")
    parser.add_argument(
        "--save-visual-outputs",
        action="store_true",
        help="Save the canonical predicted_masks, projected_masks, projected_line_masks, and RGB overlays.",
    )
    parser.add_argument("--line-gt-dir", default="data/tennis_fullcourt/line_masks/test")
    parser.add_argument("--line-iou-tolerances", default="0,3,5")
    parser.add_argument("--line-raster-thickness-px", type=int, default=1)
    parser.add_argument("--save-projected-line-masks", action="store_true")
    args = parser.parse_args()

    config = load_json(resolve_repo_path(args.config))
    model_dataset_config = load_json(resolve_repo_path(config["dataset_config"]))
    if model_dataset_config.get("sport") != "tennis":
        raise SystemExit("CourtAlign-2S tennis registration requires a tennis segmentation dataset.")
    evaluation_dataset_config = load_json(
        resolve_repo_path(config.get("evaluation_dataset_config", "configs/datasets/tennis_fullcourt.json"))
    )
    if evaluation_dataset_config["dataset_id"] != "tennis_fullcourt":
        raise SystemExit("Tennis registration must be evaluated against the official full-court dataset.")

    checkpoint = args.checkpoint or config["checkpoint"]
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else resolve_repo_path(config["output_dir"]) / args.split
    overlay_dir = output_dir / "overlays"
    pred_mask_dir = output_dir / "predicted_masks"
    projected_mask_dir = output_dir / "projected_masks"
    footprint_dir = output_dir / "projected_footprints"
    projected_line_dir = output_dir / "projected_line_masks"
    save_projected_lines = args.save_projected_line_masks or args.save_visual_outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visual_outputs:
        pred_mask_dir.mkdir(parents=True, exist_ok=True)
        projected_mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_projected_footprints:
        footprint_dir.mkdir(parents=True, exist_ok=True)
    if save_projected_lines:
        projected_line_dir.mkdir(parents=True, exist_ok=True)

    model = CourtAlign2STennisFullCourtRegistration(
        weight_path=checkpoint,
        asset_dir=config["asset_dir"],
        width=int(config["registration"]["width"]),
        height=int(config["registration"]["height"]),
        collapse_foreground_to_fullcourt=bool(
            config["registration"].get("collapse_foreground_to_fullcourt", False)
        ),
    )

    rows = load_manifest(evaluation_dataset_config["manifest"], split=args.split)
    model_rows = load_manifest(model_dataset_config["manifest"], split=args.split)
    if {row.image_id for row in rows} != {row.image_id for row in model_rows}:
        raise SystemExit(
            "The model and evaluation manifests do not contain the same image IDs for "
            f"the {args.split} split."
        )
    if args.limit is not None:
        rows = rows[: args.limit]

    corner_gt = load_corner_ground_truth(resolve_repo_path(args.corner_gt))
    reference_corners = np.asarray(model.court_reference.court_conf1[1], dtype=np.float64)
    metric_template_config = load_json(resolve_repo_path(args.metric_template_config))
    metric_template = template_from_config("tennis_fullcourt", metric_template_config)
    metric_to_reference = homography_from_corners(
        metric_template.fullcourt_corners_tl_tr_bl_br,
        reference_corners,
    )
    line_gt_dir = resolve_repo_path(args.line_gt_dir)
    line_tolerances = parse_tolerances(args.line_iou_tolerances)
    tennis_line_segments = line_segments_from_flat_points(model.court_reference.get_important_lines())

    records = []
    all_corner_distances = []
    per_image_corner_rmse = []
    successful_footprint_ious = []
    footprint_ious_failed_as_zero = []
    all_projection_errors_m = []
    per_image_projection_rmse_m = []
    all_reprojection_errors_px = []
    all_reprojection_errors_normalized = []
    per_image_reprojection_rmse_px = []
    whole_template_fullcourt_ious = []
    line_iou_successful_by_tolerance = {str(t): [] for t in line_tolerances}
    line_iou_failed_as_zero_by_tolerance = {str(t): [] for t in line_tolerances}
    line_gt_available_count = 0
    line_gt_nonempty_count = 0
    total_wall_start = time.perf_counter()
    for row in rows:
        model.reset_run_state()
        image = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(row.mask_path), cv2.IMREAD_GRAYSCALE)
        model.inference(row.image_path)
        pred_mask = model.last_pred_mask

        predicted_corners = None
        projected_mask = None
        predicted_line_mask = None
        corner_metrics = None
        footprint_iou = None
        metric_metrics = None
        line_metrics = None
        line_gt_path = line_gt_dir / row.image_id
        line_gt = cv2.imread(str(line_gt_path), cv2.IMREAD_GRAYSCALE)
        line_gt_available = line_gt is not None
        line_gt_nonempty = False
        if line_gt_available:
            line_gt_available_count += 1
            if image is not None and line_gt.shape[:2] != image.shape[:2]:
                line_gt = cv2.resize(line_gt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            line_gt_nonempty = bool(np.asarray(line_gt > 0).sum() > 0)
            if line_gt_nonempty:
                line_gt_nonempty_count += 1
        if model.is_visible and model.ref_to_frame_matrix is not None:
            predicted_corners = project_points(reference_corners, model.ref_to_frame_matrix)
            if image is not None:
                predicted_line_mask = rasterize_projected_lines(
                    tennis_line_segments,
                    model.ref_to_frame_matrix,
                    image.shape[:2],
                    thickness_px=args.line_raster_thickness_px,
                )
                if save_projected_lines:
                    cv2.imwrite(str(projected_line_dir / row.image_id), predicted_line_mask.astype(np.uint8) * 255)

            if predicted_line_mask is not None and line_gt is not None and line_gt_nonempty:
                line_values = line_iou_by_tolerance(predicted_line_mask, line_gt > 0, line_tolerances)
                for tolerance_key, value in line_values.items():
                    if value is not None:
                        line_iou_successful_by_tolerance[tolerance_key].append(value)
                        line_iou_failed_as_zero_by_tolerance[tolerance_key].append(value)
                line_metrics = {
                    "gt_available": True,
                    "gt_nonempty": True,
                    "source": str(line_gt_path),
                    "iou_by_tolerance_px": line_values,
                }
            if row.image_id in corner_gt:
                distances = corner_distances(corner_gt[row.image_id], predicted_corners)
                all_corner_distances.extend(distances.tolist())
                rmse = float(np.sqrt(np.mean(distances**2)))
                per_image_corner_rmse.append(rmse)
                corner_metrics = {
                    "mean_error_px": float(np.mean(distances)),
                    "median_error_px": float(np.median(distances)),
                    "rmse_px": rmse,
                    "distances_px": distances.tolist(),
                }
                if image is not None and gt_mask is not None:
                    gt_metric_to_image = homography_from_corners(
                        metric_template.fullcourt_corners_tl_tr_bl_br,
                        corner_gt[row.image_id],
                    )
                    predicted_metric_to_image = np.asarray(model.ref_to_frame_matrix, dtype=np.float64) @ metric_to_reference
                    predicted_metric_to_mask = scale_homography_to_mask(
                        predicted_metric_to_image,
                        image_width=image.shape[1],
                        image_height=image.shape[0],
                        mask_shape=gt_mask.shape,
                    )
                    gt_metric_to_mask = scale_homography_to_mask(
                        gt_metric_to_image,
                        image_width=image.shape[1],
                        image_height=image.shape[0],
                        mask_shape=gt_mask.shape,
                    )
                    projection_error = projection_error_meters_from_mask(
                        predicted_metric_to_mask,
                        gt_metric_to_mask,
                        gt_mask > 0,
                        max_points=args.metric_max_samples,
                    )
                    reprojection_error = reprojection_error_pixels_from_template(
                        predicted_metric_to_mask,
                        gt_metric_to_mask,
                        metric_template,
                        mask_height=gt_mask.shape[0],
                        max_points=args.metric_max_samples,
                    )
                    whole_template_iou = template_space_iou_from_homographies(
                        predicted_metric_to_mask,
                        gt_metric_to_mask,
                        metric_template,
                        class_ids=[1],
                    )
                    projection_distances = projection_error["distances_m"]
                    reprojection_distances_px = reprojection_error["distances_px"]
                    reprojection_distances_norm = reprojection_error["distances_normalized_by_mask_height"]
                    all_projection_errors_m.extend(projection_distances)
                    all_reprojection_errors_px.extend(reprojection_distances_px)
                    all_reprojection_errors_normalized.extend(reprojection_distances_norm)
                    if whole_template_iou["fullcourt_occupancy_iou"] is not None:
                        whole_template_fullcourt_ious.append(whole_template_iou["fullcourt_occupancy_iou"])
                    if projection_distances:
                        per_image_projection_rmse_m.append(float(np.sqrt(np.mean(np.asarray(projection_distances) ** 2))))
                    if reprojection_distances_px:
                        per_image_reprojection_rmse_px.append(float(np.sqrt(np.mean(np.asarray(reprojection_distances_px) ** 2))))
                    metric_metrics = {
                        "metric_template": {
                            "name": metric_template.name,
                            "width_m": metric_template.width_m,
                            "length_m": metric_template.length_m,
                            "source_note": metric_template.source_note,
                        },
                        "predicted_metric_to_image": matrix_to_list(predicted_metric_to_image),
                        "gt_metric_to_image": matrix_to_list(gt_metric_to_image),
                        "projection_error_m": projection_error["summary"],
                        "reprojection_error_px": reprojection_error["summary_px"],
                        "reprojection_error_normalized_by_mask_height": reprojection_error["summary_normalized"],
                        "whole_template_occupancy_iou": whole_template_iou,
                    }
            if image is not None and gt_mask is not None:
                fullres_projected_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                ordered = predicted_corners[[0, 1, 3, 2]].astype(np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(fullres_projected_mask, [ordered], 1)
                projected_mask = cv2.resize(
                    fullres_projected_mask,
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                if args.save_visual_outputs:
                    cv2.imwrite(str(projected_mask_dir / row.image_id), projected_mask)

                footprint_iou = projected_footprint_iou(
                    predicted_corners,
                    image_shape_hw=image.shape[:2],
                    gt_mask=gt_mask,
                )
                if footprint_iou is not None:
                    successful_footprint_ious.append(footprint_iou)
                    footprint_ious_failed_as_zero.append(footprint_iou)
                    if args.save_projected_footprints:
                        cv2.imwrite(str(footprint_dir / row.image_id), projected_mask * 255)
        else:
            footprint_ious_failed_as_zero.append(0.0)
            if line_gt_nonempty:
                for tolerance_key in line_iou_failed_as_zero_by_tolerance:
                    line_iou_failed_as_zero_by_tolerance[tolerance_key].append(0.0)
                line_metrics = {
                    "gt_available": line_gt_available,
                    "gt_nonempty": True,
                    "source": str(line_gt_path),
                    "iou_by_tolerance_px": {str(t): None for t in line_tolerances},
                    "skipped_reason": "homography_not_visible",
                }
        if line_metrics is None:
            line_metrics = {
                "gt_available": line_gt_available,
                "gt_nonempty": line_gt_nonempty,
                "source": str(line_gt_path),
                "iou_by_tolerance_px": {str(t): None for t in line_tolerances},
                "skipped_reason": "line_gt_missing_or_empty" if not line_gt_nonempty else None,
            }

        if args.save_visual_outputs and projected_mask is None and gt_mask is not None:
            projected_mask = np.zeros(gt_mask.shape, dtype=np.uint8)
            cv2.imwrite(str(projected_mask_dir / row.image_id), projected_mask)
        if save_projected_lines and predicted_line_mask is None and image is not None:
            cv2.imwrite(
                str(projected_line_dir / row.image_id),
                np.zeros(image.shape[:2], dtype=np.uint8),
            )
        if args.save_visual_outputs and pred_mask is not None:
            cv2.imwrite(str(pred_mask_dir / row.image_id), pred_mask.astype(np.uint8))
        if image is not None:
            overlay = overlay_registration(
                image,
                projected_mask,
                TENNIS_REGION_COLORS_RGB,
                projected_line_mask=predicted_line_mask,
            )
            cv2.imwrite(str(overlay_dir / row.image_id), overlay)

        records.append(
            {
                "image_id": row.image_id,
                "success": bool(model.is_visible),
                "ref_to_frame_matrix": matrix_to_list(model.ref_to_frame_matrix),
                "frame_to_ref_matrix": matrix_to_list(model.frame_to_ref_matrix),
                "court_conf": {str(k): v for k, v in model.court_conf.items()},
                "predicted_fullcourt_corners": predicted_corners.tolist() if predicted_corners is not None else None,
                "corner_gt_available": row.image_id in corner_gt,
                "corner_metrics": corner_metrics,
                "metric_registration_metrics": metric_metrics,
                "projected_fullcourt_footprint_iou": footprint_iou,
                "line_projection_iou": line_metrics,
                "timings": getattr(model, "last_timings", None),
            }
        )
    total_wall_time = time.perf_counter() - total_wall_start

    save_json(output_dir / "registration_results.json", records)
    with (output_dir / "registration_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "success",
                "corner_gt_available",
                "corner_rmse_px",
                "projection_rmse_m",
                "reprojection_rmse_px",
                "projected_fullcourt_footprint_iou",
                "whole_template_fullcourt_iou",
                *[f"line_iou_{t}px" for t in line_tolerances],
                "total_sec",
                "forward_sec",
                "homography_sec",
            ],
        )
        writer.writeheader()
        for record in records:
            timings = record.get("timings") or {}
            corner = record.get("corner_metrics") or {}
            metric = record.get("metric_registration_metrics") or {}
            line_iou_values = (record.get("line_projection_iou") or {}).get("iou_by_tolerance_px") or {}
            writer.writerow(
                {
                    "image_id": record["image_id"],
                    "success": record["success"],
                    "corner_gt_available": record["corner_gt_available"],
                    "corner_rmse_px": corner.get("rmse_px"),
                    "projection_rmse_m": (metric.get("projection_error_m") or {}).get("rmse"),
                    "reprojection_rmse_px": (metric.get("reprojection_error_px") or {}).get("rmse"),
                    "projected_fullcourt_footprint_iou": record["projected_fullcourt_footprint_iou"],
                    "whole_template_fullcourt_iou": (
                        (metric.get("whole_template_occupancy_iou") or {}).get("fullcourt_occupancy_iou")
                    ),
                    **{f"line_iou_{t}px": line_iou_values.get(str(t)) for t in line_tolerances},
                    "total_sec": timings.get("total_sec"),
                    "forward_sec": timings.get("forward_sec"),
                    "homography_sec": timings.get("homography_sec"),
                }
            )

    timing_values = [r["timings"]["total_sec"] for r in records if r.get("timings")]
    timing_values_excluding_first = timing_values[1:] if len(timing_values) > 1 else []
    forward_values = [r["timings"]["forward_sec"] for r in records if r.get("timings")]
    homography_values = [r["timings"]["homography_sec"] for r in records if r.get("timings")]
    success_count = sum(1 for record in records if record["success"])

    summary = {
        "method": "CourtAlign-2S",
        "dataset_id": evaluation_dataset_config["dataset_id"],
        "model_dataset_id": model_dataset_config["dataset_id"],
        "split": args.split,
        "checkpoint": str(resolve_repo_path(checkpoint)),
        "n_images": len(records),
        "n_success": success_count,
        "success_rate": (success_count / len(records)) if records else None,
        "corner_gt": {
            "source": str(resolve_repo_path(args.corner_gt)),
            "n_available_in_split": sum(1 for row in rows if row.image_id in corner_gt),
            "n_evaluated_successful": len(per_image_corner_rmse),
            "point_error": summarize_distances(all_corner_distances),
            "per_image_rmse_px": summarize_values(per_image_corner_rmse),
            "definition": "Euclidean error between GT full-court corners and projected template corners in 1920x1080 image coordinates.",
        },
        "metric_registration": {
            "template": {
                "name": metric_template.name,
                "width_m": metric_template.width_m,
                "length_m": metric_template.length_m,
                "source_note": metric_template.source_note,
            },
            "n_evaluated_successful": len(per_image_projection_rmse_m),
            "projection_error_m": summarize_values(all_projection_errors_m),
            "per_image_projection_rmse_m": summarize_values(per_image_projection_rmse_m),
            "reprojection_error_px": summarize_values(all_reprojection_errors_px),
            "reprojection_error_normalized_by_mask_height": summarize_values(all_reprojection_errors_normalized),
            "per_image_reprojection_rmse_px": summarize_values(per_image_reprojection_rmse_px),
            "definition": (
                "Evaluation-only metric homographies. GT metric homography is derived from tennis GT corners; "
                "predicted metric homography composes the template-to-image homography with a metric-to-template transform. "
                "Projection error samples GT full-court mask pixels and compares image-to-court coordinates in meters."
            ),
        },
        "projected_fullcourt_footprint_iou": {
            "successful_only": summarize_values(successful_footprint_ious),
            "all_images_failed_as_zero": summarize_values(footprint_ious_failed_as_zero),
            "definition": "IoU between rasterized projected full-court polygon and binary full-court GT mask.",
        },
        "visible_template_occupancy_iou": {
            "fullcourt": summarize_values(successful_footprint_ious),
            "definition": (
                "Visible-region occupancy IoU: projected tennis full-court template footprint "
                "compared against the annotated full-court mask in image/mask space."
            ),
        },
        "whole_template_occupancy_iou": {
            "fullcourt": summarize_values(whole_template_fullcourt_ious),
            "definition": (
                "Whole-template occupancy IoU in canonical tennis template space, using the predicted homography "
                "and a GT homography derived from manual full-court corners."
            ),
        },
        "line_projection_iou": {
            "gt_source": str(line_gt_dir),
            "n_gt_available": line_gt_available_count,
            "n_gt_nonempty": line_gt_nonempty_count,
            "n_evaluated_successful": len(line_iou_successful_by_tolerance[str(line_tolerances[0])])
            if line_tolerances
            else 0,
            "template_line_source": "CourtAlign-2S tennis reference lines",
            "base_line_raster_thickness_px": int(args.line_raster_thickness_px),
            "rasterization_protocol": "Connected binary OpenCV LINE_8 rasterization at the configured thickness.",
            "tolerances_px": line_tolerances,
            "dilation_protocol": (
                "For tolerance r, both predicted and GT binary line masks are dilated with an elliptical "
                "(2r+1)x(2r+1) kernel before IoU."
            ),
            "successful_visible_lines_only": {
                str(t): summarize_values(line_iou_successful_by_tolerance[str(t)]) for t in line_tolerances
            },
            "visible_gt_lines_failed_as_zero": {
                str(t): summarize_values(line_iou_failed_as_zero_by_tolerance[str(t)]) for t in line_tolerances
            },
            "definition": (
                "IoU between the projected tennis line template and the annotated line mask. "
                "This is an image-space line-alignment metric and is separate from IoUpart/IoUwhole template metrics."
            ),
        },
        "runtime": {
            "total_wall_time_sec": total_wall_time,
            "per_image_total_sec": summarize_values(timing_values),
            "per_image_total_sec_excluding_first": summarize_values(timing_values_excluding_first),
            "forward_sec": summarize_values(forward_values),
            "homography_sec": summarize_values(homography_values),
            "fps_all_images_wall_time": (len(records) / total_wall_time) if total_wall_time > 0 else None,
            "fps_excluding_first_mean": (1.0 / float(np.mean(timing_values_excluding_first))) if timing_values_excluding_first else None,
            "note": "First image includes model-load cold start. Excluding-first FPS is reported separately.",
        },
        "visual_outputs": {
            "overlays": {
                "path": str(overlay_dir),
                "definition": "Native-resolution RGB frame with projected full-court occupancy and display-only yellow projected lines.",
            },
            "predicted_masks": {
                "path": str(pred_mask_dir) if args.save_visual_outputs else None,
                "definition": "Raw 1280x720 binary semantic class-index predictions.",
            },
            "projected_masks": {
                "path": str(projected_mask_dir) if args.save_visual_outputs else None,
                "definition": "Raw 1280x720 projected full-court occupancy masks for accepted registrations; zero otherwise.",
            },
            "projected_line_masks": {
                "path": str(projected_line_dir) if save_projected_lines else None,
                "definition": "Native-resolution connected binary projected CourtAlign-2S tennis template lines for accepted registrations, and zero otherwise.",
            },
            "note": "Yellow lines in RGB overlays are dilated only for visibility; saved line masks and line-IoU remain unchanged.",
        },
        "note": "Tennis registration metrics are computed only from available GT: corners, full-court masks, success/failure, and runtime.",
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "registration_metrics.json", summary)
    save_json(output_dir / "environment.json", environment_snapshot())
    print(summary)


if __name__ == "__main__":
    main()
