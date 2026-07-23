"""Rally-group-aware tennis train and validation split for CourtAlign-E2E.

Why: the project's tennis train/val splits are frame-random; consecutive frames
of one rally (near-duplicates with identical camera geometry) straddle train and
val, which inflates validation for any camera-geometry-learning method. Here:
  * frames are grouped into rallies (vlcsnap timestamp adjacency / consecutive
    numeric ids, both gated by click-corner proximity);
  * whole groups are assigned to train or val (seeded, by group);
  * the frozen 100-frame test split is never touched.

Badminton keeps the existing video-based split (already group-safe).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np


def _ts(iid: str):
    m = re.match(r"vlcsnap-(\d{4})-(\d{2})-(\d{2})-(\d{2})h(\d{2})m(\d{2})s", iid)
    if not m:
        return None
    return ((m.group(1), m.group(2), m.group(3)),
            int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6)))


def build_groups(records: list[dict], gap_s: int = 30, prox_px: float = 15.0) -> dict:
    """records: supervision records (need image_id + lattice_uv[0:4] ~ corners)."""
    corners = {r["image_id"]: np.array(r["lattice_uv"][:4], float) for r in records}
    ids = sorted(corners)
    group_of: dict[str, str] = {}
    gid = 0

    vl = sorted([i for i in ids if i.startswith("vlcsnap")], key=lambda i: (_ts(i)[0], _ts(i)[1]))
    prev = None
    for iid in vl:
        if prev is not None:
            (d0, s0), (d1, s1) = _ts(prev), _ts(iid)
            same = (d0 == d1 and s1 - s0 <= gap_s
                    and np.abs(corners[iid] - corners[prev]).mean() < prox_px)
        else:
            same = False
        if not same:
            gid += 1
        group_of[iid] = f"g{gid:04d}"
        prev = iid

    num = sorted([i for i in ids if i[0].isdigit()], key=lambda i: int(i.split(".")[0]))
    prev = None
    for iid in num:
        if prev is not None:
            same = (int(iid.split(".")[0]) - int(prev.split(".")[0]) == 1
                    and np.abs(corners[iid] - corners[prev]).mean() < 10.0)
        else:
            same = False
        if not same:
            gid += 1
        group_of[iid] = f"g{gid:04d}"
        prev = iid
    return group_of


def assign_groups(records: list[dict], group_of: dict, val_ratio: float = 0.12,
                  seed: int = 1337) -> list[dict]:
    """Whole-group train/val assignment over the non-test pool; test untouched."""
    pool = [r for r in records if r["split"] in ("train", "val")]
    groups = sorted({group_of[r["image_id"]] for r in pool})
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    sizes = {g: sum(group_of[r["image_id"]] == g for r in pool) for g in groups}
    target = val_ratio * len(pool)
    val_groups, acc = set(), 0
    for g in groups:
        if acc >= target:
            break
        val_groups.add(g)
        acc += sizes[g]
    rows = []
    for r in records:
        g = group_of.get(r["image_id"], "g_none")
        split = r["split"]
        if split in ("train", "val"):
            split = "val" if g in val_groups else "train"
        rows.append(dict(image_id=r["image_id"], split=split, group=g))
    return rows


def write_split(rows: list[dict], out_csv: Path) -> dict:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "split", "group"])
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    n_groups = {s: len({r["group"] for r in rows if r["split"] == s}) for s in ("train", "val")}
    return {"counts": counts, "n_groups": n_groups}
