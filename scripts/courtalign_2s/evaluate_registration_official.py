#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import numpy as np

from courtalign_common.data.manifest import load_manifest
from courtalign_common.evaluation.metric_homography import multiclass_iou, scale_homography_to_mask
from courtalign_common.evaluation.official_protocol import (
    geometric_errors,
    iou_part,
    iou_whole,
    handling_outcome,
    line_iou,
    load_official_spec,
    normalize_homography,
    pck_h,
    rasterize_zones,
    summarize,
)
from courtalign_common.evaluation.registration import binary_iou
from courtalign_common.utils.io import load_json, save_json, sha256_file, write_csv
from courtalign_common.utils.paths import resolve_repo_path


def ensure_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty official result directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def index_complete_records(
    document: dict,
    expected_rows: list,
    artifact_name: str,
    *,
    require_native_wh_for_all: bool,
) -> dict[str, dict]:
    records = document.get("records")
    if not isinstance(records, list):
        raise SystemExit(f"{artifact_name} must contain a top-level records list.")

    image_ids = [record.get("image_id") for record in records]
    duplicate_ids = sorted({image_id for image_id in image_ids if image_ids.count(image_id) > 1})
    if duplicate_ids:
        raise SystemExit(f"{artifact_name} contains duplicate image IDs: {duplicate_ids}")

    expected_ids = {row.image_id for row in expected_rows}
    actual_ids = set(image_ids)
    missing_ids = sorted(expected_ids - actual_ids)
    extra_ids = sorted(actual_ids - expected_ids)
    if missing_ids or extra_ids:
        raise SystemExit(
            f"{artifact_name} must contain exactly one record for every frozen test image. "
            f"Missing={missing_ids}; extra={extra_ids}"
        )

    indexed = {record["image_id"]: record for record in records}
    allowed_statuses = {"valid", "failed", "skipped"}
    for row in expected_rows:
        record = indexed[row.image_id]
        status = record.get("status")
        if status not in allowed_statuses:
            raise SystemExit(
                f"{artifact_name} record {row.image_id} has unsupported status {status!r}; "
                f"expected one of {sorted(allowed_statuses)}."
            )
        native_wh = record.get("native_wh")
        native_wh_required = require_native_wh_for_all or status == "valid"
        if native_wh_required and native_wh != [row.width, row.height]:
            raise SystemExit(
                f"{artifact_name} record {row.image_id} has native_wh={native_wh}; "
                f"frozen manifest requires {[row.width, row.height]}."
            )
        if native_wh is not None and native_wh != [row.width, row.height]:
            raise SystemExit(
                f"{artifact_name} record {row.image_id} has inconsistent native_wh={native_wh}; "
                f"frozen manifest requires {[row.width, row.height]}."
            )
        has_homography = record.get("metric_to_image") is not None
        if status == "valid" and not has_homography:
            raise SystemExit(f"{artifact_name} record {row.image_id} is valid but has no metric_to_image.")
        if status != "valid" and has_homography:
            raise SystemExit(
                f"{artifact_name} record {row.image_id} has status={status!r} but includes metric_to_image."
            )
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical evaluator for CourtAlign homographies.")
    parser.add_argument("--dataset", required=True, choices=["tennis_fullcourt", "badminton_zones"])
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--protocol", default="configs/evaluation/official_registration_protocol.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    protocol = load_json(resolve_repo_path(args.protocol))
    freeze = load_json(resolve_repo_path(protocol["freeze_manifest"]))
    if freeze.get("status") != "FROZEN":
        raise SystemExit("Official GT is not frozen.")
    output_dir = resolve_repo_path(args.output_dir)

    dataset_config = load_json(resolve_repo_path(f"configs/datasets/{args.dataset}.json"))
    rows = load_manifest(dataset_config["manifest"], split="test")
    gt_path = resolve_repo_path(protocol["ground_truth_root"]) / args.dataset / "test/homographies.json"
    gt_document = load_json(gt_path)
    pred_path = resolve_repo_path(args.predictions)
    pred_document = load_json(pred_path)
    gt_records = index_complete_records(
        gt_document,
        rows,
        "Official GT artifact",
        require_native_wh_for_all=False,
    )
    pred_records = index_complete_records(
        pred_document,
        rows,
        "Prediction artifact",
        require_native_wh_for_all=True,
    )
    ensure_empty_output(output_dir)
    spec = load_official_spec(resolve_repo_path(protocol["court_spec"]), args.dataset)
    metric_config = protocol["metrics"]

    line_dir = resolve_repo_path(dataset_config["line_label_source"])
    per_image = []
    metric_values: dict[str, list[float]] = {}

    def collect(name: str, value: float | None) -> None:
        if value is not None and np.isfinite(value):
            metric_values.setdefault(name, []).append(float(value))

    visible_count = 0
    nonvisible_count = 0
    visible_success = 0
    false_registrations = 0
    correct_handling = 0

    for row in rows:
        gt = gt_records[row.image_id]
        pred = pred_records[row.image_id]
        gt_visible = gt["status"] == "valid"
        pred_valid = pred.get("status") == "valid" and pred.get("metric_to_image") is not None
        handling = handling_outcome(gt_visible, pred_valid)
        visible_count += handling["visible"]
        nonvisible_count += handling["nonvisible"]
        visible_success += handling["visible_success"]
        false_registrations += handling["false_registration"]
        correct_handling += handling["correct_handling"]

        record = {
            "image_id": row.image_id,
            "gt_visible": gt_visible,
            "prediction_valid": pred_valid,
            "gt_status": gt["status"],
            "prediction_status": pred.get("status", "missing"),
        }
        if not gt_visible:
            per_image.append(record)
            continue

        gt_mask = cv2.imread(str(row.mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise SystemExit(f"Missing GT mask: {row.mask_path}")
        if not pred_valid:
            for bounded in (
                "iou_part",
                "iou_whole",
                "region_iou",
                "zone_miou",
                "pck_5px",
                "pck_10px",
                "pck_20px",
                "pck_5pct_diag",
                "pck_10pct_diag",
                "line_iou_0px",
                "line_iou_3px",
                "line_iou_5px",
            ):
                record[bounded] = 0.0
                collect(f"{bounded}_failure_aware", 0.0)
            per_image.append(record)
            continue

        predicted_h = normalize_homography(np.asarray(pred["metric_to_image"], dtype=np.float64))
        gt_h = normalize_homography(np.asarray(gt["metric_to_image"], dtype=np.float64))
        predicted_mask_h = scale_homography_to_mask(predicted_h, row.width, row.height, gt_mask.shape)
        gt_mask_h = scale_homography_to_mask(gt_h, row.width, row.height, gt_mask.shape)

        record["iou_part"] = iou_part(
            predicted_mask_h,
            gt_mask_h,
            gt_mask > 0,
            spec,
            int(metric_config["template_pixels_per_meter"]),
        )
        record["iou_whole"] = iou_whole(
            predicted_h,
            gt_h,
            spec,
            int(metric_config["template_pixels_per_meter"]),
        )
        errors = geometric_errors(
            predicted_mask_h,
            gt_mask_h,
            predicted_h,
            gt_h,
            gt_mask > 0,
            spec,
            int(metric_config["visible_projection_samples"]),
            tuple(metric_config["dense_reprojection_grid"]),
            row.height,
        )
        record.update(errors)
        landmark_metrics = pck_h(
            predicted_h,
            gt_h,
            spec.landmarks,
            (row.width, row.height),
            metric_config["pck_pixel_thresholds"],
            metric_config["pck_diagonal_thresholds"],
        )
        record.update(landmark_metrics)

        projected = rasterize_zones(spec, predicted_mask_h, gt_mask.shape)
        record["region_iou"] = binary_iou(projected > 0, gt_mask > 0)
        if args.dataset == "badminton_zones":
            per_class, zone_miou = multiclass_iou(projected, gt_mask, range(1, 14))
            record["zone_miou"] = zone_miou
            record["per_zone_iou"] = json.dumps({str(k): v for k, v in per_class.items()}, sort_keys=True)

        line_path = line_dir / row.image_id
        gt_lines = cv2.imread(str(line_path), cv2.IMREAD_GRAYSCALE)
        if gt_lines is not None and np.any(gt_lines > 0):
            line_values = line_iou(
                predicted_h,
                gt_lines,
                spec,
                (row.width, row.height),
                metric_config["line_iou_tolerances_px"],
                int(metric_config["line_raster_thickness_px"]),
            )
            for tolerance, value in line_values.items():
                record[f"line_iou_{tolerance}px"] = value

        for name, value in record.items():
            if isinstance(value, (int, float)) and name not in {"gt_visible", "prediction_valid"}:
                collect(name, value)
                if name in {
                    "iou_part",
                    "iou_whole",
                    "region_iou",
                    "zone_miou",
                    "pck_5px",
                    "pck_10px",
                    "pck_20px",
                    "pck_5pct_diag",
                    "pck_10pct_diag",
                    "line_iou_0px",
                    "line_iou_3px",
                    "line_iou_5px",
                }:
                    collect(f"{name}_failure_aware", value)
        per_image.append(record)

    summary = {
        "protocol_id": protocol["protocol_id"],
        "method": args.method,
        "dataset_id": args.dataset,
        "split": "test",
        "ground_truth": {
            "path": str(gt_path),
            "sha256": sha256_file(gt_path),
            "court_spec_sha256": sha256_file(resolve_repo_path(protocol["court_spec"])),
            "n_visible": visible_count,
            "n_nonvisible": nonvisible_count,
        },
        "predictions": {
            "path": str(pred_path),
            "sha256": sha256_file(pred_path),
        },
        "handling_rates": {
            "registration_completeness_visible": visible_success / visible_count if visible_count else None,
            "false_registration_rate_nonvisible": false_registrations / nonvisible_count if nonvisible_count else None,
            "overall_correct_handling": correct_handling / len(rows) if rows else None,
            "n_visible_success": visible_success,
            "n_false_registrations": false_registrations,
            "n_images": len(rows),
        },
        "metrics_successful_visible_only": {
            key: summarize(values) for key, values in sorted(metric_values.items()) if not key.endswith("_failure_aware")
        },
        "metrics_visible_failures_as_zero": {
            key.removesuffix("_failure_aware"): summarize(values)
            for key, values in sorted(metric_values.items())
            if key.endswith("_failure_aware")
        },
        "metric_definitions": {
            "iou_part": "The same visible GT court support is warped to canonical space using predicted and GT homographies; binary IoU is computed between the two warped masks.",
            "iou_whole": "The complete canonical court is transformed by inverse(H_pred) @ H_gt and compared with the unwarped canonical court.",
            "projection_error": "2,500 visible-mask pixels per image are mapped through inverse predicted and GT homographies; metric-space distances are reported.",
            "reprojection_error": "A fixed 50x50 canonical metric grid is projected through predicted and GT homographies; image-space distances are reported.",
            "pck_h": "Homography-projected official line-axis landmarks: 14 tennis or 30 badminton landmarks.",
            "line_iou": "Official projected line axes versus polygon-derived line rasters at 0, 3, and 5 px tolerance.",
            "region_zone_iou": "Projected official court occupancy and badminton inner-edge zones versus manual region masks; same-annotation-family caveat applies.",
        },
    }

    save_json(output_dir / "official_metrics.json", summary)
    save_json(output_dir / "official_per_image.json", per_image)
    fields = sorted({key for record in per_image for key in record if key != "per_zone_iou"})
    if any("per_zone_iou" in record for record in per_image):
        fields.append("per_zone_iou")
    write_csv(output_dir / "official_per_image.csv", per_image, fields)
    print(json.dumps({"output": str(output_dir), "handling_rates": summary["handling_rates"]}, indent=2))


if __name__ == "__main__":
    main()
