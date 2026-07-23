import copy
import json
from pathlib import Path

import pytest
import torch

from courtalign_2s.experiment import build_model
from courtalign_2s.segmentation.three_phase_dice import (
    ThreePhaseDiceDeepLabV3PlusModel,
    ThreePhaseDiceSchedule,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = REPOSITORY_ROOT / "configs/courtalign_2s/badminton.json"
TENNIS_EXPERIMENT_CONFIG = REPOSITORY_ROOT / "configs/courtalign_2s/tennis.json"
DATASET_CONFIG = REPOSITORY_ROOT / "configs/datasets/badminton_zones.json"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_final_badminton_configuration_is_resnet34_three_phase():
    config = load_json(EXPERIMENT_CONFIG)

    assert config["segmentation"]["model_name"] == "DeepLabV3Plus"
    assert config["segmentation"]["encoder"] == "resnet34"
    assert config["segmentation"]["encoder_weights"] == "imagenet"
    assert config["training"]["loss_function"] == "ThreePhaseDiceLoss"
    assert config["training"]["n_epochs"] == 80
    assert config["training"]["batch_size"] == 2


def test_final_tennis_configuration_is_resnet34_dice():
    config = load_json(TENNIS_EXPERIMENT_CONFIG)

    assert config["segmentation"]["model_name"] == "DeepLabV3Plus"
    assert config["segmentation"]["encoder"] == "resnet34"
    assert config["segmentation"]["encoder_weights"] == "imagenet"
    assert config["training"]["loss_function"] == "DiceLoss"
    assert config["training"]["n_epochs"] == 50
    assert config["training"]["batch_size"] == 4


def test_three_phase_schedule_matches_released_epochs_and_classes():
    config = load_json(EXPERIMENT_CONFIG)
    dataset_config = load_json(DATASET_CONFIG)
    schedule = ThreePhaseDiceSchedule.from_config(
        config["training"]["loss_schedule"],
        dataset_config["class_names"],
        config["training"]["n_epochs"],
    )

    assert schedule.phase_for_epoch(1).uses_all_classes
    assert schedule.phase_for_epoch(14).uses_all_classes
    assert schedule.phase_for_epoch(15).active_class_names == ("bl_singles", "br_singles")
    assert schedule.phase_for_epoch(15).active_class_ids == (10, 12)
    assert schedule.phase_for_epoch(16).uses_all_classes
    assert schedule.phase_for_epoch(80).uses_all_classes


def test_three_phase_schedule_rejects_incomplete_epoch_coverage():
    config = load_json(EXPERIMENT_CONFIG)
    dataset_config = load_json(DATASET_CONFIG)
    phases = copy.deepcopy(config["training"]["loss_schedule"])
    phases[2]["start_epoch"] = 17

    with pytest.raises(ValueError, match="cover every epoch exactly once"):
        ThreePhaseDiceSchedule.from_config(
            phases,
            dataset_config["class_names"],
            config["training"]["n_epochs"],
        )


class MinimalDataset:
    class_names = load_json(DATASET_CONFIG)["class_names"]
    n_classes = len(class_names)


def test_build_model_selects_final_three_phase_variant(tmp_path: Path):
    config = load_json(EXPERIMENT_CONFIG)
    dataset_config = load_json(DATASET_CONFIG)

    model = build_model(config, dataset_config, MinimalDataset(), tmp_path)

    assert isinstance(model, ThreePhaseDiceDeepLabV3PlusModel)
    assert model.encoder == "resnet34"
    assert model.dice_schedule.phase_for_epoch(15).active_class_ids == (10, 12)


def test_focused_loss_uses_only_back_singles_classes(tmp_path: Path):
    config = load_json(EXPERIMENT_CONFIG)
    dataset_config = load_json(DATASET_CONFIG)
    model = build_model(config, dataset_config, MinimalDataset(), tmp_path)
    model.model = torch.nn.Conv2d(1, 1, kernel_size=1)

    model.setup_train()

    assert model.loss_all_classes.ignore_channels is None
    assert model.loss_focus_classes.ignore_channels == [
        class_id for class_id in range(14) if class_id not in (10, 12)
    ]
