#!/usr/bin/env python3
"""Create qualitative overlays for a CourtAlign-E2E checkpoint.

Panels: input | aux segmentation | landmarks(conf-colored) | projected court
(green = official supervision, yellow = prediction).
Writes runs/<exp>/evaluation/<split>/overlays/.

Use a CourtAlign-E2E run directory containing checkpoints/best_model.pt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from courtalign_e2e.data.dataset import CourtAlignE2EDataset, NATIVE_W, NATIVE_H, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from courtalign_e2e.geometry import templates as T  # noqa: E402
from courtalign_e2e.models.checkpoint import load_checkpoint_state  # noqa: E402
from courtalign_e2e.models.factory import build_model, resolve_lattice  # noqa: E402
from courtalign_e2e.utils.viz import overlay_panel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    ck = torch.load(run_dir / "checkpoints" / "best_model.pt", map_location="cpu")
    cfg = ck["config"]
    pack = T.sport_pack(cfg["sport"])
    ih, iw = cfg.get("input_hw", [448, 784])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lattice = resolve_lattice(cfg, pack)
    model = build_model(cfg, pack, lattice).to(device)
    load_checkpoint_state(model, ck["model"])
    model.eval()

    lattice_override = lattice if cfg.get("lattice_mode") == "zone_centroids" else None
    ds = CourtAlignE2EDataset(cfg["sport"], args.split, (ih, iw), augment=False,
                   lattice_metric=lattice_override)
    out_dir = run_dir / "evaluation" / args.split / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(ds) // args.n)
    for i in range(0, len(ds), step):
        item = ds[i]
        with torch.no_grad():
            out = model(item["image"].unsqueeze(0).to(device))
        img = (item["image"].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        panel = overlay_panel(img, out["seg_logits"].argmax(1)[0].cpu().numpy(),
                              out["coords"][0].cpu().numpy(), out["conf"][0].cpu().numpy(),
                              out["H_pred"][0].cpu().numpy(), item["H_sup"].numpy(),
                              pack["lines"], pack["n_classes"])
        cv2.imwrite(str(out_dir / item["image_id"]), panel)
    print("wrote overlays ->", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
