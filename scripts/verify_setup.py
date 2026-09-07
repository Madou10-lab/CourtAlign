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
    "data/benchmark_gt/official/FREEZE_MANIFEST.json": "5eff9eaa12dfc67963e871d7a73a49770d8542abe474062e5b149f53ead1c868",
    "data/benchmark_gt/official/HASHES.json": "ce2a4af1f97954a619ec94ddf832674d4e700eb686c0e68521093af99e62818b",
    "data/splits/tennis_fullcourt.csv": "fe40f8bc69d5e902a98fa33146b8b6fbd3e44bf6ef107c0635e85d45a7b40609",
    "data/splits/badminton_zones.csv": "d67bb370418c80fdd5f90b7aa5198c6f511ca8cea8d7cf1f7a4c199e21cd93f3",
    "data/courtalign_e2e/splits/tennis_groups.csv": "aac5ac14df6396a6efc4d93ccc973c104440a120196349fcc7c89989297de4d9",
    "data/courtalign_e2e/supervision/tennis.json": "e506fefbea31c9ac4295daf8c4e15cf042f1982aa6db6ff693f46355698e560a",
    "data/courtalign_e2e/supervision/badminton.json": "76579b82c2bfea345eb137de8d9554d6333d050c910634964fc07e12c0edc303",
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
        1_819_646_593,
        "77607584497e3bce018b08cf92dfd2f8a9a8bd76c5c5b580a8e626217fcfc706",
    ),
    "weights/courtalign_e2e/badminton/best_model.pt": (
        1_819_674_625,
        "c0689a660201d06d0edb8d9a7f365382c9c8491f92a8b4e96dc1bf09ce3615f2",
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
        "tennis_fullcourt": {"train": 904, "val": 160, "test": 119},
        "badminton_zones": {"train": 436, "val": 95, "test": 44},
    }
    for dataset, expected_counts in expected_splits.items():
        manifest = ROOT / "data/splits" / f"{dataset}.csv"
        official_test = json.loads(
            (ROOT / "data/benchmark_gt/official" / dataset / "test/homographies.json").read_text()
        )
        registrable_test_ids = {
            record["image_id"]
            for record in official_test["records"]
            if record.get("status") == "valid"
        }
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
                if row["split"] == "test" and row["image_id"] in registrable_test_ids:
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
