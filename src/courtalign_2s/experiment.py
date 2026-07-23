from __future__ import annotations

import json
import platform
import random
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from courtalign_common.data.manifest import load_manifest
from courtalign_2s.segmentation import (
    DeepLabV3PlusModel,
    CourtAlign2SManifestDataset,
    ThreePhaseDiceDeepLabV3PlusModel,
)
from courtalign_common.utils.io import load_json, save_json
from courtalign_common.utils.paths import resolve_repo_path


TRACKED_PACKAGES = [
    "albumentations",
    "efficientnet-pytorch",
    "numpy",
    "opencv-python-headless",
    "pandas",
    "pillow",
    "pretrainedmodels",
    "pydantic",
    "scikit-image",
    "scikit-learn",
    "segmentation-models-pytorch",
    "sympy",
    "timm",
    "torch",
    "torchvision",
    "tqdm",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        },
    }
    if torch.cuda.is_available():
        snapshot["torch"]["device_name"] = torch.cuda.get_device_name(0)
    return snapshot


def load_experiment(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config_path = resolve_repo_path(config_path)
    config = load_json(config_path)
    validate_experiment_config(config, config_path)
    dataset_config = load_json(resolve_repo_path(config["dataset_config"]))
    run_dir = resolve_repo_path(config["run_dir"])
    return config, dataset_config, run_dir


def validate_experiment_config(config: dict[str, Any], config_path: str | Path) -> None:
    segmentation = config.get("segmentation")
    training = config.get("training")
    if not isinstance(segmentation, dict) or not isinstance(training, dict):
        return

    loss_name = str(training.get("loss_function", "")).lower()
    activation = segmentation.get("activation")
    probability_losses = {
        "diceloss",
        "threephasediceloss",
    }
    if loss_name in probability_losses and activation is None:
        raise ValueError(
            f"{config_path} configures {training.get('loss_function')} with activation=null. "
            "CourtAlign-2S Dice training expects probability outputs; use activation='softmax2d' "
            "for the current one-hot multi-class masks."
        )
    if loss_name == "threephasediceloss":
        loss_schedule = training.get("loss_schedule")
        if not isinstance(loss_schedule, list):
            raise ValueError(f"{config_path} must define training.loss_schedule as a list.")


def split_counts(manifest_path: str | Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for row in load_manifest(manifest_path):
        counts[row.split] = counts.get(row.split, 0) + 1
    return counts


def freeze_run_metadata(config_path: str | Path, config: dict[str, Any], dataset_config: dict[str, Any], run_dir: Path) -> None:
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolve_repo_path(config_path), metadata_dir / "experiment_config.json")
    shutil.copy2(resolve_repo_path(config["dataset_config"]), metadata_dir / "dataset_config.json")
    save_json(metadata_dir / "environment.json", environment_snapshot())
    save_json(
        metadata_dir / "data_summary.json",
        {
            "manifest": dataset_config["manifest"],
            "split_counts": split_counts(dataset_config["manifest"]),
            "class_names": dataset_config["class_names"],
            "class_values": dataset_config["class_values"],
        },
    )


def build_dataset(config: dict[str, Any], dataset_config: dict[str, Any], experiment_path: str | Path) -> CourtAlign2SManifestDataset:
    training = config["training"]
    dataset = CourtAlign2SManifestDataset(
        experiment_id=config["experiment_id"],
        dataset_name=dataset_config["dataset_id"],
        manifest_path=dataset_config["manifest"],
        experiment_path=experiment_path,
        input_height=int(dataset_config["input_height"]),
        input_width=int(dataset_config["input_width"]),
        class_names=list(dataset_config["class_names"]),
        augmentation_colour_format=training["augmentation_colour_format"],
        augmentation_spatial=bool(training["augmentation_spatial"]),
        augmentation_colour=bool(training["augmentation_colour"]),
    )
    dataset.prepare()
    return dataset


def build_model(
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    dataset: CourtAlign2SManifestDataset,
    run_dir: str | Path,
) -> DeepLabV3PlusModel:
    training = config["training"]
    segmentation = config["segmentation"]
    model_class = (
        ThreePhaseDiceDeepLabV3PlusModel
        if str(training["loss_function"]).lower() == "threephasediceloss"
        else DeepLabV3PlusModel
    )
    model_kwargs: dict[str, Any] = {}
    if model_class is ThreePhaseDiceDeepLabV3PlusModel:
        model_kwargs["loss_schedule"] = training["loss_schedule"]
        model_kwargs["checkpoint_epochs"] = training.get("checkpoint_epochs")
        model_kwargs["stop_after_epoch"] = training.get("stop_after_epoch")
    return model_class(
        dataset=dataset,
        experiment_id=config["experiment_id"],
        experiment_name=config["experiment_id"],
        model_name=segmentation["model_name"],
        input_height=int(dataset_config["input_height"]),
        input_width=int(dataset_config["input_width"]),
        encoder=segmentation["encoder"],
        encoder_weights=segmentation["encoder_weights"],
        freeze_encoder=bool(segmentation["freeze_encoder"]),
        activation=segmentation["activation"],
        optimizer=training["optimizer"],
        learning_rate=float(training["learning_rate"]),
        loss_function=training["loss_function"],
        batch_size=int(training["batch_size"]),
        n_epochs=int(training["n_epochs"]),
        experiment_path=str(run_dir),
        overlay_opacity=float(training["overlay_opacity"]),
        fine_tune=bool(segmentation["fine_tune"]),
        auto_batch_size=bool(training["auto_batch_size"]),
        freeze_batch_norm=bool(training.get("freeze_batch_norm", False)),
        **model_kwargs,
    )


def default_checkpoint_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoints" / "best_model.pth"


def resolve_checkpoint(path: str | Path | None, run_dir: str | Path) -> Path:
    if path is None:
        return default_checkpoint_path(run_dir)
    return resolve_repo_path(path)
