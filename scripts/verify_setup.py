#!/usr/bin/env python3
"""Verify frozen protocol files, dataset placement, and downloaded checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SPLIT_COLUMNS = [
    "dataset_id",
    "sport",
    "task",
    "split",
    "image_id",
    "image_path",
    "mask_path",
    "image_sha256",
    "mask_sha256",
    "width",
    "height",
    "n_classes",
    "label_schema",
]

FROZEN_ARTIFACT_HASHES = {
    "data/benchmark_gt/official/FREEZE_MANIFEST.json": "3c18b784a1b62866ddc8093d95e3d57a04faf5bf2dc9f7a9967ecda75980a601",
    "data/benchmark_gt/official/HASHES.json": "93ac4a879fd972cb1b42abe11b3f5c7d3a9e6a86f09df2382948bfa41eacd76f",
    "data/splits/tennis_fullcourt.csv": "f9525d43abeb799ca788caee38835fb8481d38e4d69e2d94d01da28ebd841049",
    "data/splits/badminton_zones.csv": "0af0a4a1147ac7b4ca01d3121b5967173468f2e7da4e1b2c9b18b94a686c4253",
    "data/courtalign_e2e/splits/tennis_groups.csv": "aac5ac14df6396a6efc4d93ccc973c104440a120196349fcc7c89989297de4d9",
    "data/courtalign_e2e/supervision/tennis.json": "4562f98a1e6d30a9cd1c5c7e267d1a7bc87440156bc98299ddb0f0da56cfe31e",
    "data/courtalign_e2e/supervision/badminton.json": "5bd12473833f5a5841e8f2c1dc9265e5bd93504e3d174a5fe79932faead53419",
}

CHECKPOINTS = {
    "weights/courtalign_2s/tennis/best_model.pth": (
        89_963_091,
        "616fca950f2fbf3fc7e6e9818148511e2479dbc5fbc7ecedc83888e279afd35b",
    ),
    "weights/courtalign_2s/badminton/best_model.pth": (
        89_975_379,
        "5bdae8e774cf6d9276791b5a152f2aa9958a76bc905510ba6a9274a174541633",
    ),
    "weights/courtalign_e2e/tennis/best_model.pt": (
        1_819_646_017,
        "509d2816de9771ed53c07e0dae4a0c0e824f59c18661b82016a27fc78fda9601",
    ),
    "weights/courtalign_e2e/badminton/best_model.pt": (
        1_819_674_113,
        "fd664fae2bb54c890d03941414baa9f8af1802062025df32570ff658ba94b303",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--require-weights", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    for relative_path, expected in FROZEN_ARTIFACT_HASHES.items():
        target = ROOT / relative_path
        if not target.is_file():
            failures.append(f"Missing frozen artifact: {target}")
        elif sha256(target) != expected:
            failures.append(f"Frozen artifact hash mismatch: {target}")

    hashes = json.loads((ROOT / "data/benchmark_gt/official/HASHES.json").read_text())
    if hashes.get("status") != "FROZEN":
        failures.append("Official ground-truth hash manifest is not marked FROZEN")
    checked_hashes = 0
    for source_name, expected in hashes["files"].items():
        target = ROOT / "data/benchmark_gt/official" / source_name
        if not target.is_file() or sha256(target) != expected:
            failures.append(f"Frozen GT hash mismatch: {target}")
        checked_hashes += 1

    split_summary = {}
    line_mask_roots = {
        "tennis_fullcourt": ROOT / "data/tennis_fullcourt/line_masks/test",
        "badminton_zones": ROOT / "data/badminton_zones/line_masks/all",
    }
    expected_splits = {
        "tennis_fullcourt": {"train": 904, "val": 160, "test": 100},
        "badminton_zones": {"train": 436, "val": 95, "test": 33},
    }
    for dataset, expected_counts in expected_splits.items():
        manifest = ROOT / "data/splits" / f"{dataset}.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != SPLIT_COLUMNS:
                failures.append(f"Unexpected manifest schema in {manifest}: {reader.fieldnames}")
            rows = list(reader)
        counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
        split_summary[dataset] = counts
        if counts != expected_counts:
            failures.append(f"Unexpected split counts in {manifest}: {counts}")
        if len(rows) != sum(expected_counts.values()):
            failures.append(f"Unexpected row count in {manifest}: {len(rows)}")
        for row in rows:
            for key in ("image_path", "mask_path"):
                if Path(row[key]).is_absolute():
                    failures.append(f"Manifest path must be repository-relative: {row[key]}")
        if args.require_data:
            for row in rows:
                for key, hash_key in (("image_path", "image_sha256"), ("mask_path", "mask_sha256")):
                    target = ROOT / row[key]
                    if not target.is_file():
                        failures.append(f"Missing dataset file: {target}")
                    elif sha256(target) != row[hash_key]:
                        failures.append(f"Dataset hash mismatch: {target}")
                if row["split"] == "test":
                    line_mask = line_mask_roots[dataset] / row["image_id"]
                    if not line_mask.is_file():
                        failures.append(f"Missing test line mask: {line_mask}")

    checked_checkpoints = 0
    for relative_path, (expected_size, expected_hash) in CHECKPOINTS.items():
        path = ROOT / relative_path
        if not path.is_file():
            if args.require_weights:
                failures.append(f"Missing checkpoint: {path}")
            continue
        checked_checkpoints += 1
        if path.stat().st_size != expected_size:
            failures.append(f"Checkpoint size mismatch: {path}")
        if sha256(path) != expected_hash:
            failures.append(f"Checkpoint hash mismatch: {path}")

    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "frozen_gt_files_verified": checked_hashes,
        "frozen_training_artifacts_verified": len(FROZEN_ARTIFACT_HASHES),
        "downloaded_checkpoints_verified": checked_checkpoints,
        "splits": split_summary,
        "dataset_files_required": args.require_data,
        "weights_required": args.require_weights,
        "failures": failures[:50],
        "n_failures": len(failures),
    }, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
