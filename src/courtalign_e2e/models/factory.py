"""Model and correspondence-lattice factory for CourtAlign-E2E."""

from __future__ import annotations

import numpy as np


def resolve_lattice(cfg: dict, pack: dict) -> np.ndarray:
    if cfg.get("lattice_mode", "axis_landmarks") == "zone_centroids":
        zids = sorted(pack["zone_polys"])
        return np.array([np.mean(np.asarray(pack["zone_polys"][z], np.float64), axis=0)
                         for z in zids])
    return np.asarray(pack["lattice"], np.float64)


def build_model(cfg: dict, pack: dict, lattice: np.ndarray):
    if cfg.get("model") != "courtalign_e2e":
        raise ValueError("CourtAlign-E2E configurations must set model='courtalign_e2e'")
    if cfg.get("backbone") != "sam3_seg_fpn":
        raise ValueError("The released CourtAlign-E2E model requires backbone='sam3_seg_fpn'")

    from .e2e import CourtAlignE2E

    mode = "zone_centroids" if cfg.get("lattice_mode") == "zone_centroids" else "landmarks"
    return CourtAlignE2E(
        mode,
        pack["n_classes"],
        lattice,
        heat_temp=cfg.get("heat_temp", 0.35),
        neck_trainable=cfg.get("backbone_trainable", True),
        visibility_head=cfg.get("visibility_head", False),
    )
