#!/usr/bin/env python3
"""Evaluate a CourtAlign-E2E checkpoint and export the official prediction schema.

Official behavior under the frozen protocol:
  * runs on EVERY frame of the official split manifest (not only GT-valid ones);
  * applies the declared CourtAlign-E2E decision rule per frame and exports an explicit
    status for each: valid (homography provided) / skipped (no court detected) /
    failed (court detected but geometry unusable);
  * writes predicted_homographies.json covering the COMPLETE manifest — the
    canonical evaluator computes all official metrics + handling rates from it.

Decision rule (thresholds set on train/val, declared in config, never tuned on test):
  1. court_area_frac = fraction of non-background pixels in the aux segmentation
     < cfg.no_court_area_frac (default 0.02)        -> skipped: no_court_detected
  2. landmark-geometry degenerate (spread below gate) or mean landmark
     confidence < cfg.min_conf_mean (default 0.02)   -> failed: degenerate_geometry
  3. otherwise                                        -> valid (H exported)

Auxiliary metrics against valid ground-truth supervision are diagnostic. The
unsynchronized forward-pass timing is not part of the official evaluation.

Use the public entry point in scripts/evaluate.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from courtalign_e2e.data.dataset import IMAGENET_MEAN, IMAGENET_STD, NATIVE_W, NATIVE_H, load_supervision  # noqa: E402
from courtalign_e2e.geometry import templates as T  # noqa: E402
from courtalign_e2e.geometry.homography_np import project  # noqa: E402
from courtalign_e2e.models.checkpoint import load_checkpoint_state  # noqa: E402
from courtalign_e2e.models.factory import build_model, resolve_lattice  # noqa: E402
from courtalign_e2e.eval import metrics as M  # noqa: E402
from courtalign_e2e import paths  # noqa: E402


def manifest_rows(sport: str, split: str) -> list[dict]:
    dataset = paths.SOURCES[sport]["dataset"]
    rows = [r for r in csv.DictReader(open(paths.split_manifest(dataset))) if r["split"] == split]
    return [dict(image_id=r["image_id"], image_path=paths.data_path(r["image_path"]),
                 mask_path=paths.data_path(r["mask_path"])) for r in rows]


def prep_input(image_bgr: np.ndarray, ih: int, iw: int) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (iw, ih), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="override the export directory (default <run>/evaluation/<split>); "
                         "the directory must not contain an existing export")
    args = ap.parse_args()
    run_dir = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run

    ck = torch.load(args.checkpoint or run_dir / "checkpoints" / "best_model.pt",
                    map_location="cpu")
    cfg = ck["config"]
    sport = cfg["sport"]
    pack = T.sport_pack(sport)
    ih, iw = cfg.get("input_hw", [448, 784])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, pack, resolve_lattice(cfg, pack)).to(device)
    load_checkpoint_state(model, ck["model"])
    model.eval()

    no_court_frac = float(cfg["no_court_area_frac"])
    min_conf = float(cfg["min_conf_mean"])
    min_spread_frac = float(cfg["min_spread_frac"])
    diag = float((ih ** 2 + iw ** 2) ** 0.5)
    S_to_native = np.diag([NATIVE_W / iw, NATIVE_H / ih, 1.0])

    sup = {r["image_id"]: r for r in load_supervision(sport)}
    frames = manifest_rows(sport, args.split)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "evaluation" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    export, per, fwd_times = [], [], []
    conf_mat = np.zeros((pack["n_classes"],) * 2, np.int64)
    lines_dir = paths.line_mask_directory("tennis", args.split) if sport == "tennis" else None

    for fr in frames:
        img = cv2.imread(str(fr["image_path"]))
        rec = {"image_id": fr["image_id"], "split": args.split, "sport": sport}
        if img is None:
            rec.update(status="failed", reason="image_unreadable", metric_to_image=None)
            export.append(rec)
            continue
        nh_img, nw_img = img.shape[:2]
        rec["native_wh"] = [nw_img, nh_img]
        t0 = time.time()
        with torch.no_grad():
            out = model(prep_input(img, ih, iw).to(device))
        fwd_times.append(time.time() - t0)

        seg = out["seg_logits"].argmax(1)[0].cpu().numpy()
        court_frac = float((seg > 0).mean())
        coords = out["coords"][0].cpu().numpy()
        conf = out["conf"][0].cpu().numpy()
        spread = float(coords.std(axis=0).mean()) / diag

        # ---- declared decision rule ----
        if court_frac < no_court_frac:
            rec.update(status="skipped", reason="no_court_detected",
                       court_area_frac=round(court_frac, 4), metric_to_image=None)
            export.append(rec)
            continue
        if spread < min_spread_frac or float(conf.mean()) < min_conf:
            rec.update(status="failed", reason="degenerate_geometry",
                       court_area_frac=round(court_frac, 4), metric_to_image=None)
            export.append(rec)
            continue

        H_pred = S_to_native @ out["H_pred"][0].cpu().numpy().astype(np.float64)
        rec.update(status="valid", court_area_frac=round(court_frac, 4),
                   metric_to_image=[[float(x) for x in row] for row in H_pred])
        export.append(rec)

        # ---- diagnostic metrics (frames with GT supervision only) ----
        s = sup.get(fr["image_id"])
        if s is None:
            continue
        H_sup = np.array(s["H_sup"], np.float64)
        row = {"image_id": fr["image_id"]}
        row.update({f"corner_{k}": v for k, v in
                    M.corner_metrics(H_pred, H_sup, pack["corners"]).items()})
        row["reproj_px"] = M.reproj_px(H_pred, H_sup, np.asarray(pack["lattice"]))
        row["projection_m"] = M.projection_error_m(H_pred, H_sup, np.asarray(pack["lattice"]))
        gt_mask = cv2.imread(str(fr["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None and (gt_mask > 0).sum() > 0:
            row["region_iou"] = M.region_iou(H_pred, pack["corners"], gt_mask)
            if sport == "badminton":
                zm, _ = M.zone_miou(H_pred, pack["zone_polys"], gt_mask)
                row["zone_miou"] = zm
            gt_idx = (gt_mask > 0).astype(np.uint8) if pack["n_classes"] == 2 else gt_mask
            gt_rs = cv2.resize(gt_idx, (iw, ih), interpolation=cv2.INTER_NEAREST)
            idx = pack["n_classes"] * gt_rs.ravel().astype(int) + seg.ravel().astype(int)
            conf_mat += np.bincount(idx, minlength=pack["n_classes"] ** 2).reshape(conf_mat.shape)
        if lines_dir is not None:
            stroke = cv2.imread(str(lines_dir / fr["image_id"]), cv2.IMREAD_GRAYSCALE)
            if stroke is not None:
                row.update({f"line_iou_t{k}": v for k, v in
                            M.line_iou(H_pred, pack["lines"], stroke).items()})
        per.append(row)

    # ---- OFFICIAL export: complete manifest coverage, explicit statuses ----
    statuses = {s: sum(r["status"] == s for r in export) for s in ("valid", "failed", "skipped")}
    assert len(export) == len(frames), "export must cover every manifest frame"
    (out_dir / "predicted_homographies.json").write_text(json.dumps(
        {"records": export,
         "summary": {"n_frames": len(export), **statuses,
                     "decision_rule": {"no_court_area_frac": no_court_frac,
                                       "min_conf_mean": min_conf,
                                       "min_spread_frac": min_spread_frac},
                     "threshold_provenance": "declared in experiment config before counted training"},
         "frame_note": ("official line-axis frames (ITF 23.77x10.97 / BWF 13.4x6.1); "
                        "statuses: valid=H provided, skipped=no court detected, "
                        "failed=court visible but geometry unusable")}, indent=1))

    def agg(key):
        v = np.array([r[key] for r in per if key in r and np.isfinite(r[key])])
        return ({"mean": float(v.mean()), "median": float(np.median(v)),
                 "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
                if len(v) else {"n": 0})

    diag_metrics = {"role": "DIAGNOSTIC (official numbers come from the canonical evaluator)",
                    "run": run_dir.name, "sport": sport, "split": args.split,
                    "n_manifest_frames": len(frames), "statuses": statuses,
                    "corner_rmse_px": agg("corner_rmse_px"), "reproj_px": agg("reproj_px"),
                    "projection_m": agg("projection_m"), "region_iou": agg("region_iou"),
                    "zone_miou": agg("zone_miou"),
                    "line_iou": {t: agg(f"line_iou_t{t}") for t in ("0", "3", "5")},
                    "runtime_diagnostic_unsynchronized": {
                        "forward_sec_mean_excl_first": float(np.mean(fwd_times[1:]))
                        if len(fwd_times) > 1 else None}}
    (out_dir / "registration_metrics.json").write_text(json.dumps(diag_metrics, indent=2))
    with open(out_dir / "per_image.csv", "w", newline="") as fh:
        keys = sorted({k for r in per for k in r})
        if per:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(per)
    tp = np.diag(conf_mat)
    denom = conf_mat.sum(0) + conf_mat.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    (out_dir / "segmentation_metrics.json").write_text(json.dumps(
        {"role": "DIAGNOSTIC", "macro_mIoU": float(np.nanmean(iou)),
         "per_class_iou": [None if np.isnan(v) else float(v) for v in iou]}, indent=2))

    print(json.dumps({"export": statuses, "n_frames": len(export),
                      "diag_reproj_px": diag_metrics["reproj_px"]}, indent=2))
    print("wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
