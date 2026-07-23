#!/usr/bin/env python3
"""Build the rally-group-aware CourtAlign-E2E tennis train/validation split.

Groups near-duplicate frames (same rally / camera) and assigns whole groups to
train or val. The frozen 100-image test split is untouched. Also writes group
ids for TEST frames (used by the label-free temporal-jitter metric).

Output: data/courtalign_e2e/splits/tennis_groups.csv and its summary JSON.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from courtalign_e2e.data.supervision import build_tennis  # noqa: E402
from courtalign_e2e.data.splits import build_groups, assign_groups, write_split  # noqa: E402

if __name__ == "__main__":
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1337
    recs = build_tennis()
    groups = build_groups(recs)
    rows = assign_groups(recs, groups, val_ratio=0.12, seed=seed)
    output = ROOT / "data" / "courtalign_e2e" / "splits"
    stats = write_split(rows, output / "tennis_groups.csv")
    stats["seed"] = seed
    print(json.dumps(stats, indent=2))
    (output / "tennis_groups_stats.json").write_text(json.dumps(stats, indent=2))
