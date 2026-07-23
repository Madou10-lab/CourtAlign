from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from courtalign_common.utils.io import read_csv
from courtalign_common.utils.paths import resolve_repo_path


@dataclass(frozen=True)
class ManifestRow:
    dataset_id: str
    sport: str
    task: str
    split: str
    image_id: str
    image_path: Path
    mask_path: Path
    image_sha256: str
    mask_sha256: str
    width: int
    height: int
    n_classes: int
    label_schema: str


def load_manifest(path: str | Path, split: str | None = None) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for row in read_csv(resolve_repo_path(path)):
        if split is not None and row["split"] != split:
            continue
        rows.append(
            ManifestRow(
                dataset_id=row["dataset_id"],
                sport=row["sport"],
                task=row["task"],
                split=row["split"],
                image_id=row["image_id"],
                image_path=resolve_repo_path(row["image_path"]),
                mask_path=resolve_repo_path(row["mask_path"]),
                image_sha256=row["image_sha256"],
                mask_sha256=row["mask_sha256"],
                width=int(row["width"]),
                height=int(row["height"]),
                n_classes=int(row["n_classes"]),
                label_schema=row["label_schema"],
            )
        )
    return rows
