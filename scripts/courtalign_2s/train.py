#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from courtalign_2s.experiment import (
    build_dataset,
    build_model,
    default_checkpoint_path,
    freeze_run_metadata,
    json_safe,
    load_experiment,
    seed_everything,
)
from courtalign_common.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CourtAlign-2S segmentation from a frozen config.")
    parser.add_argument("--config", required=True, help="CourtAlign-2S experiment JSON config.")
    parser.add_argument("--prepare-only", action="store_true", help="Build data/model objects without training.")
    args = parser.parse_args()

    config, dataset_config, run_dir = load_experiment(args.config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    seed_everything(int(config["training"]["seed"]))
    freeze_run_metadata(args.config, config, dataset_config, run_dir)

    dataset = build_dataset(config, dataset_config, run_dir)
    model = build_model(config, dataset_config, dataset, run_dir)
    model.prepare()

    prepared_summary = {
        "status": "prepared",
        "experiment_id": config["experiment_id"],
        "dataset_id": dataset_config["dataset_id"],
        "train_size": dataset.train_size(),
        "valid_size": dataset.valid_size(),
        "test_size": dataset.test_size(),
        "model": config["segmentation"]["model_name"],
        "encoder": config["segmentation"]["encoder"],
        "activation": config["segmentation"]["activation"],
        "loss_function": config["training"]["loss_function"],
        "checkpoint_policy": "full torch model saved as checkpoints/best_model.pth",
    }
    save_json(run_dir / "metrics" / "prepared_summary.json", prepared_summary)

    if args.prepare_only:
        print(prepared_summary)
        return

    model.train()
    summary: dict = {"experiment_id": config["experiment_id"], "dataset_id": dataset_config["dataset_id"]}
    model.get_results(summary)
    summary["selection"] = {
        "split": "val",
        "criterion": config.get("selection", {}).get("criterion", "lowest validation Dice loss"),
        "checkpoint": str(default_checkpoint_path(run_dir)),
        "saved_epoch": getattr(model, "saved_epoch", None),
        "note": "Validation is run during training. The held-out test split is not used for checkpoint selection.",
    }
    save_json(run_dir / "metrics" / "train_validation_summary.json", json_safe(summary))
    save_json(
        run_dir / "metrics" / "train_validation_epoch_logs.json",
        json_safe(
            {
                "train": getattr(model, "train_logs_list", []),
                "valid": getattr(model, "valid_logs_list", []),
                "train_iterations": getattr(model, "train_logs_iter_list", []),
                "valid_iterations": getattr(model, "valid_logs_iter_list", []),
            }
        ),
    )
    print(json_safe(summary))


if __name__ == "__main__":
    main()
