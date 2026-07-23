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
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

from courtalign_common.data.manifest import load_manifest
from courtalign_common.evaluation.metric_homography import (
    derive_badminton_homography_from_zone_mask,
    matrix_to_list,
    multiclass_iou,
    projection_error_meters_from_mask,
    rasterize_template,
    reprojection_error_pixels_from_template,
    template_space_iou_from_homographies,
)
from courtalign_common.evaluation.line_iou import (
    line_iou_by_tolerance,
    line_segments_from_metric_template,
    parse_tolerances,
    rasterize_projected_lines,
)
from courtalign_common.evaluation.metric_templates import template_from_config
from courtalign_common.evaluation.registration import binary_iou, summarize_values
from courtalign_2s.experiment import build_dataset, environment_snapshot, freeze_run_metadata, load_experiment
from courtalign_2s.segmentation import model_utils as mu
from courtalign_2s.segmentation import utils as courtalign_2s_utils
from courtalign_common.utils.io import load_json, save_json
from courtalign_common.utils.paths import resolve_repo_path
from courtalign_common.evaluation.visual_outputs import BADMINTON_REGION_COLORS_RGB, overlay_registration


def split_dataset(dataset, split: str):
    if split == "train":
        return dataset.train_dataset
    if split == "val":
        return dataset.valid_dataset
    if split == "test":
        return dataset.test_dataset
    raise ValueError(split)


def decode_prediction(pred_tensor) -> np.ndarray:
    return courtalign_2s_utils.transpose_reverse_one_hot(pred_tensor.detach().squeeze().cpu().numpy()).astype(np.uint8)


def decode_target(target) -> np.ndarray:
    return courtalign_2s_utils.reverse_one_hot(np.transpose(target, (1, 2, 0))).astype(np.uint8)


def zone_corner_error_px(predicted, ground_truth) -> dict | None:
    pred_by_class = predicted.correspondences.get("image_points_by_class") or {}
    gt_by_class = ground_truth.correspondences.get("image_points_by_class") or {}
    shared_classes = sorted(set(pred_by_class) & set(gt_by_class), key=int)
    if not shared_classes:
        return None

    distances = []
    per_class = {}
    for class_id in shared_classes:
        pred = np.asarray(pred_by_class[class_id], dtype=np.float64)
        gt = np.asarray(gt_by_class[class_id], dtype=np.float64)
        if pred.shape != (4, 2) or gt.shape != (4, 2):
            continue
        class_distances = np.linalg.norm(pred - gt, axis=1)
        distances.extend(class_distances.tolist())
        per_class[class_id] = {
            "mean": float(np.mean(class_distances)),
            "rmse": float(np.sqrt(np.mean(class_distances**2))),
            "distances_px": class_distances.tolist(),
        }

    if not distances:
        return None
    values = np.asarray(distances, dtype=np.float64)
    return {
        "n_classes": len(per_class),
        "n_corners": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "max": float(np.max(values)),
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CourtAlign-2S badminton segmentation and metric homography registration.")
    parser.add_argument("--config", default="configs/registration/courtalign_2s_badminton.json")
    parser.add_argument("--checkpoint", default=None, help="Override segmentation checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-predicted-masks", action="store_true")
    parser.add_argument("--save-projected-zone-masks", action="store_true")
    parser.add_argument(
        "--save-visual-outputs",
        action="store_true",
        help="Save the canonical predicted_masks, projected_masks, projected_line_masks, and RGB overlays.",
    )
    parser.add_argument("--line-gt-dir", default="data/badminton_zones/line_masks/all")
    parser.add_argument("--line-iou-tolerances", default="0,3,5")
    parser.add_argument("--line-raster-thickness-px", type=int, default=1)
    parser.add_argument("--save-projected-line-masks", action="store_true")
    args = parser.parse_args()

    registration_config = load_json(resolve_repo_path(args.config))
    dataset_config = load_json(resolve_repo_path(registration_config["dataset_config"]))
    if dataset_config["dataset_id"] != "badminton_zones":
        raise SystemExit("Badminton registration runner requires badminton_zones dataset config.")

    experiment_config_path = registration_config["experiment_config"]
    experiment_config, experiment_dataset_config, run_dir = load_experiment(experiment_config_path)
    if experiment_dataset_config["dataset_id"] != dataset_config["dataset_id"]:
        raise SystemExit("Registration dataset_config and experiment_config point to different datasets.")

    output_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir
        else resolve_repo_path(registration_config["output_dir"]) / args.split
    )
    overlay_dir = output_dir / "overlays"
    pred_mask_dir = output_dir / "predicted_masks"
    canonical_projected_mask_dir = output_dir / "projected_masks"
    projected_mask_dir = output_dir / "projected_zone_masks"
    projected_line_dir = output_dir / "projected_line_masks"
    save_predicted_masks = args.save_predicted_masks or args.save_visual_outputs
    save_canonical_projected_masks = args.save_visual_outputs
    save_projected_lines = args.save_projected_line_masks or args.save_visual_outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if save_predicted_masks:
        pred_mask_dir.mkdir(parents=True, exist_ok=True)
    if save_canonical_projected_masks:
        canonical_projected_mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_projected_zone_masks:
        projected_mask_dir.mkdir(parents=True, exist_ok=True)
    if save_projected_lines:
        projected_line_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_repo_path(args.checkpoint or registration_config["checkpoint"])
    if not checkpoint.exists():
        raise SystemExit(
            f"Checkpoint does not exist: {checkpoint}. Train first or pass --checkpoint runs/.../checkpoints/best_model.pth"
        )

    freeze_run_metadata(experiment_config_path, experiment_config, dataset_config, run_dir)
    dataset = build_dataset(experiment_config, dataset_config, run_dir)
    segmentation = experiment_config["segmentation"]
    preprocessing_fn = mu.get_preprocessing(
        smp.encoders.get_preprocessing_fn(segmentation["encoder"], segmentation["encoder_weights"])
    )
    dataset.build_train(preprocessing_fn)
    eval_dataset = split_dataset(dataset, args.split)

    manifest_rows = {row.image_id: row for row in load_manifest(dataset_config["manifest"], split=args.split)}
    image_ids = [eval_dataset.get_image_filename(i) for i in range(len(eval_dataset))]
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    template_config = load_json(resolve_repo_path(registration_config["metric_template_config"]))
    template = template_from_config(dataset_config["dataset_id"], template_config)
    template_line_segments = line_segments_from_metric_template(template)
    line_tolerances = parse_tolerances(args.line_iou_tolerances)
    line_gt_dir = resolve_repo_path(args.line_gt_dir)
    split_line_gt_dir = line_gt_dir / args.split
    if split_line_gt_dir.exists():
        line_gt_dir = split_line_gt_dir
    reg = registration_config["registration"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_start = time.perf_counter()
    model = torch.load(checkpoint, map_location=device)
    if not hasattr(model, "eval"):
        raise SystemExit(f"Checkpoint did not load as a full torch model: {checkpoint}")
    model.to(device)
    model.eval()
    load_time = time.perf_counter() - load_start

    records = []
    success_flags = []
    evaluable_flags = []
    projected_zone_mious = []
    projected_fullcourt_ious = []
    projection_errors_m = []
    reprojection_errors_px = []
    reprojection_errors_norm = []
    zone_corner_errors_px = []
    whole_template_fullcourt_ious = []
    whole_template_zone_mious = []
    line_iou_successful_by_tolerance = {str(t): [] for t in line_tolerances}
    line_iou_failed_as_zero_by_tolerance = {str(t): [] for t in line_tolerances}
    line_gt_available_count = 0
    line_gt_nonempty_count = 0
    total_times = []
    forward_times = []
    homography_times = []

    total_wall_start = time.perf_counter()
    with torch.no_grad():
        for idx in tqdm(range(len(image_ids)), desc=f"courtalign-2s-badminton-register-{args.split}"):
            image_id = image_ids[idx]
            row = manifest_rows[image_id]
            original_image = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)

            image, target = eval_dataset[idx]
            target_mask = decode_target(target)
            x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

            start_total = time.perf_counter()
            start_forward = time.perf_counter()
            pred_tensor = model(x_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_sec = time.perf_counter() - start_forward
            pred_mask = decode_prediction(pred_tensor)

            start_homography = time.perf_counter()
            pred_derived = derive_badminton_homography_from_zone_mask(
                pred_mask,
                template,
                image_width=row.width,
                image_height=row.height,
                ransac_reprojection_threshold_px=reg["ransac_reproj_threshold_px"],
                min_classes=reg["min_classes"],
                min_inlier_ratio=reg["min_inlier_ratio"],
                min_zone_miou=reg["min_zone_miou"],
                min_fullcourt_iou=reg["min_fullcourt_iou"],
            )
            gt_derived = derive_badminton_homography_from_zone_mask(
                target_mask,
                template,
                image_width=row.width,
                image_height=row.height,
                ransac_reprojection_threshold_px=reg["ransac_reproj_threshold_px"],
                min_classes=reg["min_classes"],
                min_inlier_ratio=reg["min_inlier_ratio"],
                min_zone_miou=reg["min_zone_miou"],
                min_fullcourt_iou=reg["min_fullcourt_iou"],
            )
            homography_sec = time.perf_counter() - start_homography
            total_sec = time.perf_counter() - start_total

            projected_mask = None
            zone_miou = None
            fullcourt_iou = None
            projection_error = None
            reprojection_error = None
            whole_template_iou = None
            zone_corner_error = zone_corner_error_px(pred_derived, gt_derived)
            line_metrics = None
            predicted_line_mask = None
            evaluation_status = "not_evaluable"
            evaluation_reason = None
            line_gt_path = line_gt_dir / image_id
            line_gt = cv2.imread(str(line_gt_path), cv2.IMREAD_GRAYSCALE)
            line_gt_available = line_gt is not None
            line_gt_nonempty = False
            line_shape = original_image.shape[:2] if original_image is not None else (row.height, row.width)
            if line_gt_available:
                line_gt_available_count += 1
                if line_gt.shape[:2] != line_shape:
                    line_gt = cv2.resize(line_gt, (line_shape[1], line_shape[0]), interpolation=cv2.INTER_NEAREST)
                line_gt_nonempty = bool(np.asarray(line_gt > 0).sum() > 0)
                if line_gt_nonempty:
                    line_gt_nonempty_count += 1

            pred_valid = pred_derived.status == "valid"
            gt_valid = gt_derived.status == "valid"

            if pred_derived.metric_to_mask is not None:
                projected_mask = rasterize_template(template, pred_derived.metric_to_mask, target_mask.shape)
                _, zone_miou = multiclass_iou(projected_mask, target_mask, range(1, 14))
                fullcourt_iou = binary_iou(projected_mask > 0, target_mask > 0)
                if save_canonical_projected_masks and pred_valid:
                    cv2.imwrite(str(canonical_projected_mask_dir / image_id), projected_mask)
                if args.save_projected_zone_masks:
                    cv2.imwrite(str(projected_mask_dir / image_id), projected_mask)

            success_flags.append(pred_valid)
            evaluable_flags.append(pred_valid and gt_valid)

            if pred_valid and pred_derived.metric_to_image is not None:
                predicted_line_mask = rasterize_projected_lines(
                    template_line_segments,
                    pred_derived.metric_to_image,
                    line_shape,
                    thickness_px=args.line_raster_thickness_px,
                )
                if save_projected_lines:
                    cv2.imwrite(str(projected_line_dir / image_id), predicted_line_mask.astype(np.uint8) * 255)

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
            elif line_gt_nonempty and not pred_valid:
                for tolerance_key in line_iou_failed_as_zero_by_tolerance:
                    line_iou_failed_as_zero_by_tolerance[tolerance_key].append(0.0)
                line_metrics = {
                    "gt_available": line_gt_available,
                    "gt_nonempty": True,
                    "source": str(line_gt_path),
                    "iou_by_tolerance_px": {str(t): None for t in line_tolerances},
                    "skipped_reason": "homography_not_valid",
                }
            if line_metrics is None:
                line_metrics = {
                    "gt_available": line_gt_available,
                    "gt_nonempty": line_gt_nonempty,
                    "source": str(line_gt_path),
                    "iou_by_tolerance_px": {str(t): None for t in line_tolerances},
                    "skipped_reason": "line_gt_missing_or_empty" if not line_gt_nonempty else None,
                }

            if pred_valid and gt_valid:
                evaluation_status = "evaluated"
                projection_error = projection_error_meters_from_mask(
                    pred_derived.metric_to_mask,
                    gt_derived.metric_to_mask,
                    target_mask > 0,
                    max_points=reg["metric_max_samples"],
                )
                reprojection_error = reprojection_error_pixels_from_template(
                    pred_derived.metric_to_mask,
                    gt_derived.metric_to_mask,
                    template,
                    mask_height=target_mask.shape[0],
                    max_points=reg["metric_max_samples"],
                )
                whole_template_iou = template_space_iou_from_homographies(
                    pred_derived.metric_to_mask,
                    gt_derived.metric_to_mask,
                    template,
                    class_ids=range(1, 14),
                )
                projected_zone_mious.append(zone_miou)
                projected_fullcourt_ious.append(fullcourt_iou)
                if whole_template_iou["fullcourt_occupancy_iou"] is not None:
                    whole_template_fullcourt_ious.append(whole_template_iou["fullcourt_occupancy_iou"])
                if whole_template_iou["zone_miou"] is not None:
                    whole_template_zone_mious.append(whole_template_iou["zone_miou"])
                projection_errors_m.extend(projection_error["distances_m"])
                reprojection_errors_px.extend(reprojection_error["distances_px"])
                reprojection_errors_norm.extend(reprojection_error["distances_normalized_by_mask_height"])
                if zone_corner_error is not None:
                    zone_corner_errors_px.extend(
                        distance
                        for class_metrics in zone_corner_error["per_class"].values()
                        for distance in class_metrics["distances_px"]
                    )
            elif not gt_valid:
                evaluation_reason = f"derived_gt_{gt_derived.status}:{gt_derived.skipped_reason}"
            elif not pred_valid:
                evaluation_reason = f"prediction_{pred_derived.status}:{pred_derived.skipped_reason}"

            if save_canonical_projected_masks and not pred_valid:
                cv2.imwrite(
                    str(canonical_projected_mask_dir / image_id),
                    np.zeros(target_mask.shape, dtype=np.uint8),
                )
            if save_projected_lines and predicted_line_mask is None:
                cv2.imwrite(
                    str(projected_line_dir / image_id),
                    np.zeros(line_shape, dtype=np.uint8),
                )
            if save_predicted_masks:
                cv2.imwrite(str(pred_mask_dir / image_id), pred_mask)
            if original_image is not None:
                overlay = overlay_registration(
                    original_image,
                    projected_mask if pred_valid else None,
                    BADMINTON_REGION_COLORS_RGB,
                    projected_line_mask=predicted_line_mask,
                )
                cv2.imwrite(str(overlay_dir / image_id), overlay)

            total_times.append(total_sec)
            forward_times.append(forward_sec)
            homography_times.append(homography_sec)
            records.append(
                {
                    "image_id": image_id,
                    "split": args.split,
                    "prediction_status": pred_derived.status,
                    "prediction_skipped_reason": pred_derived.skipped_reason,
                    "gt_status": gt_derived.status,
                    "gt_skipped_reason": gt_derived.skipped_reason,
                    "evaluation_status": evaluation_status,
                    "evaluation_reason": evaluation_reason,
                    "metric_to_image": matrix_to_list(pred_derived.metric_to_image),
                    "metric_to_mask": matrix_to_list(pred_derived.metric_to_mask),
                    "gt_metric_to_mask": matrix_to_list(gt_derived.metric_to_mask),
                    "prediction_validation": pred_derived.validation,
                    "gt_validation": gt_derived.validation,
                    "projected_zone_miou": zone_miou,
                    "projected_fullcourt_iou": fullcourt_iou,
                    "whole_template_occupancy_iou": whole_template_iou,
                    "zone_corner_error_px": zone_corner_error,
                    "line_projection_iou": line_metrics,
                    "projection_error_m": projection_error["summary"] if projection_error else None,
                    "reprojection_error_px": reprojection_error["summary_px"] if reprojection_error else None,
                    "reprojection_error_normalized_by_mask_height": (
                        reprojection_error["summary_normalized"] if reprojection_error else None
                    ),
                    "timings": {
                        "forward_sec": forward_sec,
                        "homography_sec": homography_sec,
                        "total_sec": total_sec,
                    },
                }
            )

    total_wall_time = time.perf_counter() - total_wall_start

    save_json(output_dir / "registration_results.json", records)
    with (output_dir / "registration_per_image.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id",
            "prediction_status",
            "gt_status",
            "evaluation_status",
            "evaluation_reason",
            "projected_zone_miou",
            "projected_fullcourt_iou",
            "whole_template_fullcourt_iou",
            "whole_template_zone_miou",
            "zone_corner_rmse_px",
            *[f"line_iou_{t}px" for t in line_tolerances],
            "projection_rmse_m",
            "reprojection_rmse_px",
            "total_sec",
            "forward_sec",
            "homography_sec",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            line_iou_values = (record.get("line_projection_iou") or {}).get("iou_by_tolerance_px") or {}
            writer.writerow(
                {
                    "image_id": record["image_id"],
                    "prediction_status": record["prediction_status"],
                    "gt_status": record["gt_status"],
                    "evaluation_status": record["evaluation_status"],
                    "evaluation_reason": record["evaluation_reason"],
                    "projected_zone_miou": record["projected_zone_miou"],
                    "projected_fullcourt_iou": record["projected_fullcourt_iou"],
                    "whole_template_fullcourt_iou": (
                        (record.get("whole_template_occupancy_iou") or {}).get("fullcourt_occupancy_iou")
                    ),
                    "whole_template_zone_miou": (record.get("whole_template_occupancy_iou") or {}).get("zone_miou"),
                    "zone_corner_rmse_px": (record.get("zone_corner_error_px") or {}).get("rmse"),
                    **{f"line_iou_{t}px": line_iou_values.get(str(t)) for t in line_tolerances},
                    "projection_rmse_m": (record.get("projection_error_m") or {}).get("rmse"),
                    "reprojection_rmse_px": (record.get("reprojection_error_px") or {}).get("rmse"),
                    "total_sec": record["timings"]["total_sec"],
                    "forward_sec": record["timings"]["forward_sec"],
                    "homography_sec": record["timings"]["homography_sec"],
                }
            )

    summary = {
        "method": "CourtAlign-2S",
        "dataset_id": dataset_config["dataset_id"],
        "split": args.split,
        "checkpoint": str(checkpoint),
        "n_images": len(records),
        "n_prediction_success": int(sum(success_flags)),
        "prediction_success_rate_all_images": float(np.mean(success_flags)) if success_flags else None,
        "n_evaluable_against_derived_gt": int(sum(evaluable_flags)),
        "evaluable_rate_all_images": float(np.mean(evaluable_flags)) if evaluable_flags else None,
        "projected_zone_miou": summarize_values(v for v in projected_zone_mious if v is not None),
        "projected_fullcourt_iou": summarize_values(v for v in projected_fullcourt_ious if v is not None),
        "visible_template_occupancy_iou": {
            "fullcourt": summarize_values(v for v in projected_fullcourt_ious if v is not None),
            "zones": summarize_values(v for v in projected_zone_mious if v is not None),
            "definition": (
                "Visible-region template IoU in image/mask space: the predicted-homography badminton template "
                "is rasterized and compared with annotated 13-zone masks."
            ),
        },
        "whole_template_occupancy_iou": {
            "fullcourt": summarize_values(whole_template_fullcourt_ious),
            "zones": summarize_values(whole_template_zone_mious),
            "definition": (
                "Whole-template IoU in canonical badminton metric-template space, using the predicted homography "
                "and a GT homography derived from the annotated zone mask."
            ),
        },
        "zone_corner_error_px": summarize_values(zone_corner_errors_px),
        "line_projection_iou": {
            "gt_source": str(line_gt_dir),
            "n_gt_available": line_gt_available_count,
            "n_gt_nonempty": line_gt_nonempty_count,
            "n_evaluated_successful": len(line_iou_successful_by_tolerance[str(line_tolerances[0])])
            if line_tolerances
            else 0,
            "template_line_source": "boundaries of the metric 13-zone badminton court template",
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
                "IoU between the predicted-homography badminton metric-template line boundaries and the "
                "ground-truth court-line mask. This image-space line-alignment metric is separate from "
                "zone and court-footprint IoU."
            ),
        },
        "projection_error_m": summarize_values(projection_errors_m),
        "reprojection_error_px": summarize_values(reprojection_errors_px),
        "reprojection_error_normalized_by_mask_height": summarize_values(reprojection_errors_norm),
        "runtime": {
            "model_load_sec": load_time,
            "total_wall_time_sec": total_wall_time,
            "per_image_total_sec": summarize_values(total_times),
            "per_image_total_sec_excluding_first": summarize_values(total_times[1:]),
            "forward_sec": summarize_values(forward_times),
            "homography_sec": summarize_values(homography_times),
            "fps_all_images_wall_time": (len(records) / total_wall_time) if total_wall_time > 0 else None,
            "fps_excluding_first_mean": (1.0 / float(np.mean(total_times[1:]))) if len(total_times) > 1 else None,
        },
        "template": {
            "name": template.name,
            "width_m": template.width_m,
            "length_m": template.length_m,
            "source_note": template.source_note,
        },
        "visual_outputs": {
            "overlays": {
                "path": str(overlay_dir),
                "definition": "Native-resolution RGB frame with projected regions and display-only yellow projected lines.",
            },
            "predicted_masks": {
                "path": str(pred_mask_dir) if save_predicted_masks else None,
                "definition": "Raw 1280x720 semantic class-index predictions.",
            },
            "projected_masks": {
                "path": str(canonical_projected_mask_dir) if save_canonical_projected_masks else None,
                "definition": "Raw 1280x720 metric-template zone labels for accepted registrations; zero otherwise.",
            },
            "projected_line_masks": {
                "path": str(projected_line_dir) if save_projected_lines else None,
                "definition": "Native-resolution connected binary projected template lines for accepted registrations; zero otherwise.",
            },
            "note": "Yellow lines in RGB overlays are dilated only for visibility; saved line masks and line-IoU remain unchanged.",
        },
        "definitions": {
            "prediction_success": "Predicted 13-zone segmentation mask yielded a homography that passed validation gates.",
            "derived_gt": "Evaluation-only GT homography derived from the GT 13-zone mask; images with all-background GT masks are not evaluable.",
            "projected_zone_miou": "mIoU between GT 13-zone mask and metric court template rasterized through the predicted homography.",
            "projected_fullcourt_iou": "Binary IoU between GT court footprint and metric court template footprint rasterized through the predicted homography.",
            "whole_template_occupancy_iou": "Predicted-vs-derived-GT homography IoU in canonical metric-template space.",
            "zone_corner_error_px": "Euclidean error between class-matched TL/TR/BL/BR zone corners extracted from predicted and ground-truth zone masks.",
        },
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "registration_metrics.json", summary)
    save_json(output_dir / "environment.json", environment_snapshot())
    print(summary)


if __name__ == "__main__":
    main()
