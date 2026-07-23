#!/usr/bin/env python3
"""Evaluate one CourtAlign checkpoint with the frozen evaluation protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    PYTHON,
    ROOT,
    dataset_id,
    default_checkpoint,
    default_config,
    method_name,
    relative_or_absolute,
    require_new_output,
    run,
)


def evaluate_2s(sport: str, checkpoint: Path, output: Path, config: Path) -> None:
    method_scripts = ROOT / "scripts/courtalign_2s"
    registration_config = ROOT / "configs/registration" / f"courtalign_2s_{sport}.json"
    segmentation = output / "segmentation"
    registration = output / "registration"
    predictions = output / "predictions/predicted_homographies.json"
    official = output / "official"

    run([
        PYTHON, str(method_scripts / "evaluate_segmentation.py"), "--config", str(config),
        "--checkpoint", str(checkpoint), "--split", "test", "--output-dir", str(segmentation),
    ])
    runner = method_scripts / f"register_{sport}.py"
    run([
        PYTHON, str(runner), "--config", str(registration_config), "--checkpoint", str(checkpoint),
        "--split", "test", "--output-dir", str(registration), "--save-visual-outputs",
    ])
    if sport == "tennis":
        run([
            PYTHON, str(method_scripts / "export_tennis_predictions.py"),
            "--registration-config", str(registration_config),
            "--registration-results", str(registration / "registration_results.json"),
            "--checkpoint", str(checkpoint), "--output", str(predictions),
            "--method", "CourtAlign-2S",
        ])
    else:
        run([
            PYTHON, str(method_scripts / "export_badminton_predictions.py"),
            "--registration-config", str(registration_config),
            "--predicted-mask-dir", str(registration / "predicted_masks"),
            "--checkpoint", str(checkpoint), "--output", str(predictions),
            "--method", "CourtAlign-2S",
        ])
    run([
        PYTHON, str(method_scripts / "evaluate_registration_official.py"),
        "--dataset", dataset_id(sport), "--predictions", str(predictions),
        "--method", "CourtAlign-2S", "--output-dir", str(official),
    ])


def evaluate_e2e(sport: str, checkpoint: Path, output: Path) -> None:
    inference = output / "predictions"
    run_path = ROOT / "runs" / f"courtalign_e2e_{sport}"
    run([
        PYTHON, str(ROOT / "scripts/courtalign_e2e/evaluate.py"),
        "--run", str(run_path), "--split", "test", "--checkpoint", str(checkpoint),
        "--out-dir", str(inference),
    ])
    run([
        PYTHON, str(ROOT / "scripts/courtalign_2s/evaluate_registration_official.py"),
        "--dataset", dataset_id(sport),
        "--predictions", str(inference / "predicted_homographies.json"),
        "--method", "CourtAlign-E2E", "--output-dir", str(output / "official"),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["courtalign-2s", "courtalign-e2e", "2s", "e2e"])
    parser.add_argument("--sport", required=True, choices=["tennis", "badminton"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None, help="CourtAlign-2S experiment config override.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    method = method_name(args.method)
    checkpoint = relative_or_absolute(args.checkpoint) if args.checkpoint else default_checkpoint(method, args.sport)
    if not checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {checkpoint}. See weights/README.md.")
    output = relative_or_absolute(args.output_dir) if args.output_dir else ROOT / "runs/evaluation" / method / args.sport
    require_new_output(output)

    if method == "courtalign-2s":
        config = relative_or_absolute(args.config) if args.config else default_config(method, args.sport)
        evaluate_2s(args.sport, checkpoint, output, config)
    else:
        evaluate_e2e(args.sport, checkpoint, output)
    print(f"Official metrics: {output / 'official/official_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
