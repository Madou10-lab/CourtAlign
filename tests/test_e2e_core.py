from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from courtalign_e2e.geometry.homography_torch import project_h, weighted_dlt
from courtalign_e2e.models.checkpoint import checkpoint_state_dict, load_checkpoint_state


ROOT = Path(__file__).resolve().parents[1]


class _CheckpointFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Module()
        self.backbone.vit = torch.nn.Linear(2, 2)
        self.backbone.neck = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 1)


def test_released_e2e_configs_use_the_final_sam3_recipe():
    tennis = json.loads((ROOT / "configs/courtalign_e2e/tennis.json").read_text())
    badminton = json.loads((ROOT / "configs/courtalign_e2e/badminton.json").read_text())

    for config, sport, epochs in ((tennis, "tennis", 60), (badminton, "badminton", 80)):
        assert config["model"] == "courtalign_e2e"
        assert config["backbone"] == "sam3_seg_fpn"
        assert config["backbone_trainable"] is True
        assert config["backbone_lr"] == 3e-5
        assert config["input_hw"] == [1008, 1008]
        assert config["sport"] == sport
        assert config["n_epochs"] == epochs
        assert config["visibility_head"] is True
        assert config["use_negatives"] is True


def test_weighted_dlt_recovers_a_well_constrained_homography():
    source = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.3, 0.8], [0.8, 0.2]]],
        dtype=torch.float64,
    )
    target = source * torch.tensor([10.0, 10.0], dtype=torch.float64) + torch.tensor(
        [10.0, 20.0], dtype=torch.float64
    )
    homography = weighted_dlt(source, target, torch.ones(1, 6, dtype=torch.float64))

    assert torch.allclose(project_h(homography, source), target, atol=1e-6)


def test_e2e_checkpoint_excludes_the_frozen_sam3_trunk():
    source = _CheckpointFixture()
    state = checkpoint_state_dict(source)

    assert state
    assert not any(name.startswith("backbone.vit.") for name in state)
    assert any(name.startswith("backbone.neck.") for name in state)
    assert any(name.startswith("head.") for name in state)

    target = _CheckpointFixture()
    frozen_before = {name: value.clone() for name, value in target.backbone.vit.state_dict().items()}
    load_checkpoint_state(target, state)

    for name, value in target.backbone.vit.state_dict().items():
        assert torch.equal(value, frozen_before[name])
    for name, value in source.backbone.neck.state_dict().items():
        assert torch.equal(value, target.backbone.neck.state_dict()[name])


def test_e2e_checkpoint_rejects_missing_trained_parameters():
    model = _CheckpointFixture()
    state = checkpoint_state_dict(model)
    state.pop("head.weight")

    with pytest.raises(RuntimeError, match="Missing required keys"):
        load_checkpoint_state(model, state)


def test_e2e_checkpoint_loader_accepts_complete_state():
    source = _CheckpointFixture()
    target = _CheckpointFixture()

    load_checkpoint_state(target, source.state_dict())

    for name, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[name])
