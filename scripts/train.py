#!/usr/bin/env python3
"""Train CourtAlign-2S or CourtAlign-E2E with the released configurations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PYTHON, ROOT, default_config, load_json, method_name, relative_or_absolute, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["courtalign-2s", "courtalign-e2e", "2s", "e2e"])
    parser.add_argument("--sport", required=True, choices=["tennis", "badminton"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume CourtAlign-E2E from last_model.pt.")
    parser.add_argument("--prepare-only", action="store_true", help="Build CourtAlign-2S data and model objects without training.")
    args = parser.parse_args()

    method = method_name(args.method)
    config = relative_or_absolute(args.config) if args.config else default_config(method, args.sport)
    if not config.exists():
        raise SystemExit(f"Missing experiment config: {config}")

    if method == "courtalign-2s":
        if args.resume:
            raise SystemExit("CourtAlign-2S checkpoints do not contain optimizer state and cannot be resumed exactly.")
        cfg = load_json(config)
        run_dir = relative_or_absolute(cfg["run_dir"])
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(
                f"Refusing to overwrite non-empty CourtAlign-2S run directory: {run_dir}. "
                "Set a different run_dir in the configuration file."
            )
        command = [PYTHON, str(ROOT / "scripts/courtalign_2s/train.py"), "--config", str(config)]
        if args.prepare_only:
            command.append("--prepare-only")
    else:
        if args.prepare_only:
            raise SystemExit("Use scripts/verify_setup.py for a lightweight CourtAlign-E2E setup check.")
        cfg = load_json(config)
        command = [
            PYTHON,
            str(ROOT / "scripts/courtalign_e2e/train.py"),
            "--config",
            str(config),
            "--seed",
            "1337",
        ]
        if args.resume:
            command.append("--resume")

    run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
