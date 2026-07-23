#!/usr/bin/env python3
"""Train CourtAlign-E2E without overwriting an existing run.

Selection is REGISTRATION-AWARE: best epoch = lowest validation lattice
reprojection error in native pixels between the predicted and supervision homographies.
Loss, seg mIoU and landmark error are logged alongside.

Use the public entry point in scripts/train.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from courtalign_e2e.data.dataset import CourtAlignE2EDataset, NATIVE_W, NATIVE_H  # noqa: E402
from courtalign_e2e.geometry import templates as T  # noqa: E402
from courtalign_e2e.models.checkpoint import checkpoint_state_dict, load_checkpoint_state  # noqa: E402
from courtalign_e2e.models.factory import build_model, resolve_lattice  # noqa: E402
from courtalign_e2e.models.losses import CourtAlignE2ELoss  # noqa: E402
from courtalign_e2e.utils.run_dir import init_run_dir, set_seed  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", action="store_true",
                    help="continue from checkpoints/last_model.pt if it exists "
                         "(same config; optimizer/scheduler/best state restored)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    set_seed(args.seed)

    sport = cfg["sport"]
    pack = T.sport_pack(sport)
    ih, iw = cfg.get("input_hw", [448, 784])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir_path = ROOT / "runs" / cfg["experiment_id"]
    last_ck_path = run_dir_path / "checkpoints" / "last_model.pt"
    resuming = args.resume and last_ck_path.exists()
    if resuming:
        run_dir = run_dir_path
        print(f"[resume] continuing existing run at {run_dir}")
    else:
        run_dir = init_run_dir(run_dir_path, cfg, args.config, sys.argv, args.seed)

    lat = resolve_lattice(cfg, pack)
    lat_override = lat if cfg.get("lattice_mode") == "zone_centroids" else None
    use_neg = bool(cfg.get("use_negatives", False))
    tr = CourtAlignE2EDataset(sport, "train", (ih, iw), augment=cfg.get("augment", True),
                   aug_cfg=cfg.get("aug"), seed=args.seed, lattice_metric=lat_override,
                   include_negatives=use_neg)
    va = CourtAlignE2EDataset(sport, "val", (ih, iw), augment=False, lattice_metric=lat_override,
                   include_negatives=use_neg)
    if use_neg:
        n_neg = sum(r.get("court_visible", 1) == 0 for r in tr.records)
        print(f"[negatives] rejection training ON: {n_neg} negative frames in train, "
              f"{sum(r.get('court_visible', 1) == 0 for r in va.records)} in val")
    print(f"[data] train={len(tr)} val={len(va)} sport={sport} input={ih}x{iw}")
    dl_tr = DataLoader(tr, batch_size=cfg.get("batch_size", 4), shuffle=True,
                       num_workers=cfg.get("num_workers", 4), drop_last=len(tr) % cfg.get("batch_size", 4) == 1)
    dl_va = DataLoader(va, batch_size=cfg.get("batch_size", 4), shuffle=False,
                       num_workers=cfg.get("num_workers", 4))

    bb_trainable = bool(cfg.get("backbone_trainable", False))
    model = build_model(cfg, pack, lat).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())

    loss_fn = CourtAlignE2ELoss(cfg, lat, pack["zone_polys"], pack["zone_class_of_poly"],
                     (ih, iw), device)
    head_lr = cfg.get("lr", 3e-4)
    wd = cfg.get("weight_decay", 1e-4)
    if bb_trainable:
        bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
        head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
        bb_lr = cfg.get("backbone_lr", head_lr / 10.0)
        opt = torch.optim.AdamW([{"params": head, "lr": head_lr},
                                 {"params": bb, "lr": bb_lr}], weight_decay=wd)
        print(f"[model] backbone={cfg['backbone']} TRAINABLE | backbone {sum(p.numel() for p in bb)/1e6:.2f}M "
              f"@lr={bb_lr:.1e}, heads {sum(p.numel() for p in head)/1e6:.2f}M @lr={head_lr:.1e} "
              f"| total trainable {n_train/1e6:.2f}M")
    else:
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=head_lr, weight_decay=wd)
        print(f"[model] backbone={cfg['backbone']} FROZEN | trainable_params={n_train/1e6:.2f}M (heads only)")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["n_epochs"])

    metrics_csv = run_dir / "metrics.csv"
    if not resuming:
        with open(metrics_csv, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch", "lr", "train_loss", "val_loss", "val_reproj_px",
                                     "val_landmark_px", "val_seg_miou",
                                     "seg", "heat", "coord", "reproj", "tmpl"])

    src_metric = torch.tensor(pack["lattice"], dtype=torch.float32, device=device)
    sx, sy = NATIVE_W / iw, NATIVE_H / ih          # input px -> native px
    px_scale = float((sx + sy) / 2)

    # Badminton uses the same all-class, focused-class, all-class Dice schedule
    # as CourtAlign-2S. Tennis uses the all-class Dice objective throughout.
    seg_sched = cfg.get("seg_loss_schedule")
    if seg_sched:
        covered = []
        for ph in seg_sched:
            covered.extend(range(int(ph["start_epoch"]), int(ph["end_epoch"]) + 1))
        if covered != list(range(1, cfg["n_epochs"] + 1)):
            raise SystemExit(f"seg_loss_schedule must cover epochs 1..{cfg['n_epochs']} exactly once")
        print("[seg schedule] " + " | ".join(
            f"{p['name']} ep{p['start_epoch']}-{p['end_epoch']} "
            f"({'all' if p['active_class_ids'] == 'all' else p.get('active_class_names', p['active_class_ids'])})"
            for p in seg_sched))

    def seg_active_for(epoch: int):
        if not seg_sched:
            return None, "all"
        for ph in seg_sched:
            if ph["start_epoch"] <= epoch <= ph["end_epoch"]:
                ids = ph["active_class_ids"]
                return (None, "all") if ids == "all" else (list(ids), ph["name"])
        raise RuntimeError(f"no seg phase for epoch {epoch}")

    warmup = int(cfg.get("geo_warmup_epochs", 3))
    best = {"val_reproj_px": float("inf"), "epoch": 0}
    start_epoch = 1
    if resuming:
        st = torch.load(last_ck_path, map_location="cpu")
        load_checkpoint_state(model, st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best = st["best"]
        start_epoch = int(st["epoch"]) + 1
        print(f"[resume] restored epoch {st['epoch']} (best so far: "
              f"{best['val_reproj_px']:.2f}px @ep{best['epoch']}); continuing at {start_epoch}")
        if start_epoch > cfg["n_epochs"]:
            print("[resume] training already complete — nothing to do")
            return 0
    for epoch in range(start_epoch, cfg["n_epochs"] + 1):
        geo_scale = min(1.0, max(0.0, (epoch - 1) / max(warmup, 1)))
        seg_active, seg_phase = seg_active_for(epoch)
        model.train()
        tl, comp = 0.0, {}
        for batch in dl_tr:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["image"])
            loss, logs = loss_fn(out, batch, pack["n_classes"], geo_scale=geo_scale,
                                 seg_active_classes=seg_active)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
            opt.step()
            tl += float(loss)
            for k, v in logs.items():
                comp[k] = comp.get(k, 0.0) + v
        nb = max(1, len(dl_tr))
        tl /= nb
        comp = {k: v / nb for k, v in comp.items()}
        sched.step()

        model.eval()
        vl, vr, vlm, inter_u = 0.0, [], [], None
        from courtalign_e2e.eval.metrics import NATIVE_W as _  # noqa
        import torch.nn.functional as F
        conf_mat = np.zeros((pack["n_classes"], pack["n_classes"]), np.int64)
        with torch.no_grad():
            for batch in dl_va:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(batch["image"])
                loss, _logs = loss_fn(out, batch, pack["n_classes"])
                vl += float(loss)
                from courtalign_e2e.geometry.homography_torch import project_h
                B = out["coords"].shape[0]
                cvb = batch.get("court_visible", torch.ones(B, device=out["coords"].device))
                p = project_h(out["H_pred"], src_metric.unsqueeze(0).expand(B, -1, -1))
                q = project_h(batch["H_sup"], src_metric.unsqueeze(0).expand(B, -1, -1))
                vr_all = ((p - q).norm(dim=-1).mean(dim=1) * px_scale)
                lm = ((out["coords"] - batch["lattice_uv"]).norm(dim=-1) * batch["vis"]).sum(1) \
                    / batch["vis"].sum(1).clamp_min(1)
                lm_all = lm * px_scale
                # Geometric validation metrics apply only to court-visible frames.
                # Non-registrable frames carry identity supervision homographies.
                keep = (cvb > 0).cpu().numpy().astype(bool)
                vr += vr_all.cpu().numpy()[keep].tolist()
                vlm += lm_all.cpu().numpy()[keep].tolist()
                pred = out["seg_logits"].argmax(1).cpu().numpy()
                gt = batch["seg"].cpu().numpy()
                for b in range(pred.shape[0]):
                    if not keep[b]:
                        continue
                    idx = pack["n_classes"] * gt[b].ravel() + pred[b].ravel()
                    conf_mat += np.bincount(idx, minlength=pack["n_classes"] ** 2)\
                        .reshape(pack["n_classes"], pack["n_classes"])
        vl /= max(1, len(dl_va))
        val_reproj = float(np.mean(vr))
        val_lm = float(np.mean(vlm))
        tp = np.diag(conf_mat)
        denom = conf_mat.sum(0) + conf_mat.sum(1) - tp
        miou = float(np.nanmean(np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)))

        with open(metrics_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, f"{sched.get_last_lr()[0]:.2e}", f"{tl:.4f}",
                                     f"{vl:.4f}", f"{val_reproj:.2f}", f"{val_lm:.2f}",
                                     f"{miou:.4f}"] + [f"{comp.get(k, 0):.4f}" for k in
                                                       ("seg", "heat", "coord", "reproj", "tmpl")])
        print(f"ep {epoch:3d} | loss {tl:.4f}/{vl:.4f} | val reproj {val_reproj:7.2f}px "
              f"| lm {val_lm:6.2f}px | mIoU {miou:.4f}"
              + (f" | segphase={seg_phase}" if seg_phase != "all" else ""))

        if val_reproj <= best["val_reproj_px"]:
            best = {"val_reproj_px": val_reproj, "epoch": epoch,
                    "val_landmark_px": val_lm, "val_seg_miou": miou}
            torch.save({"model": checkpoint_state_dict(model), "config": cfg, "epoch": epoch},
                       run_dir / "checkpoints" / "best_model.pt")
        # resume-safe last checkpoint (model + optimizer + scheduler + best state)
        torch.save({"model": checkpoint_state_dict(model), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best,
                    "config": cfg},
                   run_dir / "checkpoints" / "last_model.pt")

    (run_dir / "summary.json").write_text(json.dumps(
        {"config": cfg, "selection": "min_val_lattice_reproj_px", "best": best,
         "seed": args.seed, "trainable_params": n_train}, indent=2))
    print("best:", json.dumps(best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
