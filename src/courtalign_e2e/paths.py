"""Repository-relative paths for CourtAlign-E2E."""

from __future__ import annotations

import csv
import os
from pathlib import Path


def repo_root() -> Path:
    configured = os.environ.get("COURTALIGN_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).resolve().parents[2]


def official_gt_dir() -> Path:
    return repo_root() / "data" / "benchmark_gt" / "official"


def split_manifest(dataset: str) -> Path:
    return repo_root() / "data" / "splits" / f"{dataset}.csv"


def supervision_path(sport: str) -> Path:
    return repo_root() / "data" / "courtalign_e2e" / "supervision" / f"{sport}.json"


def tennis_group_split() -> Path:
    return repo_root() / "data" / "courtalign_e2e" / "splits" / "tennis_groups.csv"


def badminton_metadata() -> Path:
    return repo_root() / "data" / "courtalign_e2e" / "splits" / "badminton_metadata.csv"


SOURCES = {
    "tennis": {"dataset": "tennis_fullcourt"},
    "badminton": {"dataset": "badminton_zones"},
}


def manifest_rows(dataset: str) -> list[dict[str, str]]:
    with split_manifest(dataset).open(newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_index(dataset: str) -> dict[str, dict[str, str]]:
    return {row["image_id"]: row for row in manifest_rows(dataset)}


def data_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else repo_root() / path


def line_mask_directory(sport: str, split: str) -> Path:
    dataset = SOURCES[sport]["dataset"]
    return repo_root() / "data" / dataset / "line_masks" / split
