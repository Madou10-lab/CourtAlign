#!/usr/bin/env python3
"""Build CourtAlign-E2E supervision records from the frozen official ground truth.

Writes data/courtalign_e2e/supervision/{tennis,badminton}.json and prints coverage stats.
The command regenerates the checked-in training artifacts from frozen inputs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from courtalign_e2e.data.supervision import build_all  # noqa: E402

if __name__ == "__main__":
    output = ROOT / "data" / "courtalign_e2e" / "supervision"
    stats = build_all(output)
    print(json.dumps(stats, indent=2))
    (output / "build_stats.json").write_text(json.dumps(stats, indent=2))
