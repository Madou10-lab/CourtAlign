#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from courtalign_common.data.manifest import load_manifest
from courtalign_common.evaluation.metric_homography import homography_from_corners
from courtalign_common.evaluation.official_protocol import load_official_spec, matrix_to_list
from courtalign_common.utils.io import load_json, save_json, sha256_file
from courtalign_common.utils.paths import resolve_repo_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CourtAlign-2S tennis predictions to the official schema.")
    parser.add_argument("--registration-config", required=True)
    parser.add_argument("--protocol", default="configs/evaluation/official_registration_protocol.json")
    parser.add_argument("--registration-results", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocol = load_json(resolve_repo_path(args.protocol))
    freeze = load_json(resolve_repo_path(protocol["freeze_manifest"]))
    if freeze.get("status") != "FROZEN":
        raise SystemExit("Official GT is not frozen.")

    config = load_json(resolve_repo_path(args.registration_config))
    evaluation_config = load_json(
        resolve_repo_path(config.get("evaluation_dataset_config", "configs/datasets/tennis_fullcourt.json"))
    )
    rows = load_manifest(evaluation_config["manifest"], split="test")
    rows_by_id = {row.image_id: row for row in rows}
    results_path = (
        resolve_repo_path(args.registration_results)
        if args.registration_results
        else resolve_repo_path(config["output_dir"]) / "test/registration_results.json"
    )
    registration_records = load_json(results_path)
    registration_by_id = {record["image_id"]: record for record in registration_records}
    if set(rows_by_id) != set(registration_by_id):
        raise SystemExit("Registration results do not cover the complete official tennis test split.")

    spec = load_official_spec(resolve_repo_path(protocol["court_spec"]), "tennis_fullcourt")
    checkpoint = resolve_repo_path(args.checkpoint or config["checkpoint"])
    records = []
    for row in rows:
        result = registration_by_id[row.image_id]
        corners = result.get("predicted_fullcourt_corners")
        valid = bool(result.get("success") and corners is not None)
        metric_to_image = None
        if valid:
            metric_to_image = homography_from_corners(
                spec.outer_corners,
                np.asarray(corners, dtype=np.float64),
            )
        records.append(
            {
                "image_id": row.image_id,
                "split": "test",
                "sport": "tennis",
                "status": "valid" if valid else "skipped",
                "skipped_reason": None if valid else "courtalign_2s_registration_failed",
                "native_wh": [row.width, row.height],
                "metric_to_image": matrix_to_list(metric_to_image),
                "method": args.method,
                "checkpoint": str(checkpoint),
                "derivation": (
                    "The official metric court corners are paired with the four corners returned by the "
                    "shared CourtAlign-2S tennis contour and homography stage."
                ),
            }
        )

    output = resolve_repo_path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing prediction artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "official_predicted_homographies_v1",
        "protocol_id": protocol["protocol_id"],
        "method": args.method,
        "sport": "tennis",
        "records": records,
        "summary": {
            "n_images": len(records),
            "n_valid": sum(record["status"] == "valid" for record in records),
            "n_not_valid": sum(record["status"] != "valid" for record in records),
        },
        "provenance": {
            "registration_results": str(results_path),
            "registration_results_sha256": sha256_file(results_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "registration_config": str(resolve_repo_path(args.registration_config)),
        },
    }
    save_json(output, document)
    print(json.dumps({"output": str(output), "summary": document["summary"]}, indent=2))


if __name__ == "__main__":
    main()
