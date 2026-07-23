"""Checkpoint helpers for full and compact CourtAlign-E2E checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch


FROZEN_TRUNK_PREFIX = "backbone.vit."


def checkpoint_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Return model state without the frozen SAM 3 vision trunk."""
    return OrderedDict(
        (name, tensor)
        for name, tensor in model.state_dict().items()
        if not name.startswith(FROZEN_TRUNK_PREFIX)
    )


def load_checkpoint_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
):
    """Load a full or compact checkpoint and require every trained parameter."""
    result = model.load_state_dict(state, strict=False)
    missing_required = [
        name for name in result.missing_keys if not name.startswith(FROZEN_TRUNK_PREFIX)
    ]
    if missing_required or result.unexpected_keys:
        raise RuntimeError(
            "Invalid CourtAlign-E2E checkpoint. "
            f"Missing required keys: {missing_required}. "
            f"Unexpected keys: {result.unexpected_keys}."
        )
    return result
