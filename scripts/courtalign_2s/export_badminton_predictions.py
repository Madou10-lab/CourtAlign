#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2

from courtalign_common.data.manifest import load_manifest
from courtalign_common.evaluation.official_protocol import estimate_badminton_homography, load_official_spec
from courtalign_common.utils.io import load_json, save_json, sha256_file
from courtalign_common.utils.paths import resolve_repo_path


def ensure_new_file(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing prediction artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def homography_kwargs(registration_config: dict) -> dict:
    reg = registration_config["registration"]
    return {
        "ransac_reprojection_threshold_px": float(
            reg.get("ransac_reprojection_threshold_px", reg.get("ransac_reproj_threshold_px"))
        ),
        "min_classes": int(reg["min_classes"]),
        "min_inlier_ratio": float(reg["min_inlier_ratio"]),
        "min_zone_miou": float(reg["min_zone_miou"]),
        "min_fullcourt_iou": float(reg["min_fullcourt_iou"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export CourtAlign-2S badminton predictions to the canonical official "
            "predicted_homographies.json schema."
        )
    )
    parser.add_argument("--registration-config", required=True)
    parser.add_argument("--protocol", default="configs/evaluation/official_registration_protocol.json")
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--predicted-mask-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocol_path = resolve_repo_path(args.protocol)
    protocol = load_json(protocol_path)
    freeze = load_json(resolve_repo_path(protocol["freeze_manifest"]))
    if freeze.get("status") != "FROZEN":
        raise SystemExit(
            "Official ground truth is not marked as frozen. Run python scripts/verify_setup.py."
        )

    registration_config_path = resolve_repo_path(args.registration_config)
    registration_config = load_json(registration_config_path)
    dataset_config = load_json(resolve_repo_path(registration_config["dataset_config"]))
    if dataset_config["dataset_id"] != "badminton_zones":
        raise SystemExit("This exporter is only for the CourtAlign-2S badminton protocol.")

    experiment_config = load_json(resolve_repo_path(registration_config["experiment_config"]))
    method = args.method or experiment_config["experiment_id"]
    registration_output_dir = resolve_repo_path(registration_config["output_dir"]) / args.split
    pred_dir = (
        resolve_repo_path(args.predicted_mask_dir)
        if args.predicted_mask_dir
        else registration_output_dir / "predicted_masks"
    )
    output = resolve_repo_path(args.output)
    ensure_new_file(output, overwrite=args.overwrite)

    spec = load_official_spec(resolve_repo_path(protocol["court_spec"]), "badminton_zones")
    rows = load_manifest(dataset_config["manifest"], split=args.split)
    checkpoint = resolve_repo_path(args.checkpoint or registration_config["checkpoint"])
    records = []
    prediction_hashes = {}
    for row in rows:
        mask_path = pred_dir / row.image_id
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            records.append(
                {
                    "image_id": row.image_id,
                    "split": args.split,
                    "sport": "badminton",
                    "status": "skipped",
                    "skipped_reason": "predicted_mask_missing",
                    "native_wh": [row.width, row.height],
                    "metric_to_image": None,
                    "method": method,
                }
            )
            continue

        prediction_hashes[row.image_id] = sha256_file(mask_path)
        derived = estimate_badminton_homography(
            mask,
            spec,
            image_width=row.width,
            image_height=row.height,
            **homography_kwargs(registration_config),
        )
        derived_status = derived["status"]
        if derived_status == "rejected":
            official_status = "failed"
        elif derived_status in {"valid", "failed", "skipped"}:
            official_status = derived_status
        else:
            official_status = "failed"
        is_valid = official_status == "valid"
        records.append(
            {
                "image_id": row.image_id,
                "split": args.split,
                "sport": "badminton",
                "status": official_status,
                "skipped_reason": derived["skipped_reason"],
                "native_wh": [row.width, row.height],
                "metric_to_image": derived["metric_to_image"] if is_valid else None,
                "method": method,
                "checkpoint": str(checkpoint),
                "derivation": (
                    "CourtAlign-2S badminton export: per-zone four-corner extraction "
                    "from the predicted 13-zone mask, followed by the official RANSAC-DLT "
                    "metric-template homography conversion."
                ),
                "validation": derived["validation"],
                "registration_status": derived_status,
            }
        )

    document = {
        "schema": "official_predicted_homographies_v1",
        "protocol_id": protocol["protocol_id"],
        "method": method,
        "sport": "badminton",
        "records": records,
        "summary": {
            "n_images": len(records),
            "n_valid": sum(record["status"] == "valid" for record in records),
            "n_not_valid": sum(record["status"] != "valid" for record in records),
        },
        "provenance": {
            "registration_config": str(registration_config_path),
            "registration_config_sha256": sha256_file(registration_config_path),
            "experiment_config": str(resolve_repo_path(registration_config["experiment_config"])),
            "experiment_config_sha256": sha256_file(resolve_repo_path(registration_config["experiment_config"])),
            "predicted_mask_dir": str(pred_dir),
            "prediction_mask_sha256": prediction_hashes,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint)
            if checkpoint.exists()
            else None,
        },
    }
    save_json(output, document)
    print(json.dumps({"output": str(output), "summary": document["summary"]}, indent=2))


if __name__ == "__main__":
    main()
