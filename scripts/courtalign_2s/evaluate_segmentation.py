#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

from courtalign_common.evaluation.segmentation import confusion_matrix_from_masks, metrics_from_confusion
from courtalign_2s.experiment import build_dataset, freeze_run_metadata, load_experiment, resolve_checkpoint
from courtalign_2s.segmentation import model_utils as mu
from courtalign_2s.segmentation import utils as courtalign_2s_utils
from courtalign_common.utils.io import save_json, write_csv
from courtalign_common.utils.paths import resolve_repo_path


def split_dataset(dataset, split: str):
    if split == "train":
        return dataset.train_dataset
    if split == "val":
        return dataset.valid_dataset
    if split == "test":
        return dataset.test_dataset
    raise ValueError(split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CourtAlign-2S segmentation checkpoint.")
    parser.add_argument("--config", required=True, help="CourtAlign-2S experiment JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to run_dir/checkpoints/best_model.pth.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    parser.set_defaults(save_predictions=True)
    args = parser.parse_args()

    config, dataset_config, run_dir = load_experiment(args.config)
    checkpoint = resolve_checkpoint(args.checkpoint, run_dir)
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else run_dir / "evaluation" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    freeze_run_metadata(args.config, config, dataset_config, run_dir)

    dataset = build_dataset(config, dataset_config, run_dir)
    segmentation = config["segmentation"]
    preprocessing_fn = mu.get_preprocessing(
        smp.encoders.get_preprocessing_fn(segmentation["encoder"], segmentation["encoder_weights"])
    )
    dataset.build_train(preprocessing_fn)
    eval_dataset = split_dataset(dataset, args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(checkpoint, map_location=device)
    model.to(device)
    model.eval()

    n_classes = len(dataset_config["class_names"])
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    records = []

    with torch.no_grad():
        for idx in tqdm(range(len(eval_dataset)), desc=f"courtalign-2s-eval-{args.split}"):
            image, target = eval_dataset[idx]
            image_id = eval_dataset.get_image_filename(idx)
            x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
            pred_tensor = model(x_tensor)
            pred_mask = courtalign_2s_utils.transpose_reverse_one_hot(pred_tensor.detach().squeeze().cpu().numpy())
            gt_mask = courtalign_2s_utils.reverse_one_hot(np.transpose(target, (1, 2, 0))).astype(np.uint8)

            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_mask = pred_mask.astype(np.uint8)

            if args.save_predictions:
                cv2.imwrite(str(pred_dir / image_id), pred_mask)

            image_confusion = confusion_matrix_from_masks(gt_mask, pred_mask, n_classes)
            confusion += image_confusion
            image_metrics = metrics_from_confusion(image_confusion, dataset_config["class_names"])
            records.append(
                {
                    "image_id": image_id,
                    "mean_iou": image_metrics["mean_iou"],
                    "mean_dice": image_metrics["mean_dice"],
                    "pixel_accuracy": image_metrics["pixel_accuracy"],
                }
            )

    metrics = metrics_from_confusion(confusion, dataset_config["class_names"])
    metrics.update(
        {
            "experiment_id": config["experiment_id"],
            "dataset_id": dataset_config["dataset_id"],
            "split": args.split,
            "evaluation_role": "held_out_test" if args.split == "test" else "inspection_only_not_checkpoint_selection",
            "checkpoint": str(checkpoint),
            "prediction_dir": str(pred_dir),
            "n_images": len(eval_dataset),
            "confusion_matrix": confusion.tolist(),
            "method": "CourtAlign-2S segmentation",
        }
    )

    save_json(output_dir / "segmentation_metrics.json", metrics)
    write_csv(
        output_dir / "per_image_metrics.csv",
        records,
        fieldnames=["image_id", "mean_iou", "mean_dice", "pixel_accuracy"],
    )
    print(metrics)


if __name__ == "__main__":
    main()
