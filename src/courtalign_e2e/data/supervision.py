"""Build per-image CourtAlign-E2E supervision from the frozen official ground truth.

For each court-visible image:
  * H_sup  : metric->image homography at NATIVE 1920x1080.
      - tennis  : official metric-to-image homography on the ITF court model.
      - badminton: official metric-to-image homography on the BWF court model.
  * lattice_uv : K lattice points projected through H_sup (native px).
  * vis        : per-point in-frame visibility (margin 8 px).
  * weight     : tier-based label confidence (gold 1.0, silver 0.7) scaled by
                 exp(-residual/4px) for badminton when fit residuals are available.
  * split      : frozen split label. Tennis training and validation use the
                 rally-group assignment in data/courtalign_e2e/splits/tennis_groups.csv.

Output: data/courtalign_e2e/supervision/<sport>.json, the single training-time label
artifact. The file is plain JSON and does not require PyTorch to inspect.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from ..geometry import templates as T
from ..geometry.homography_np import project
from .. import paths

NATIVE_W, NATIVE_H = 1920, 1080
VIS_MARGIN = 8.0
csv.field_size_limit(10 ** 8)


def _official(dataset: str) -> dict:
    """Records of the OFFICIAL benchmark GT artifact (line-axis convention)."""
    f = paths.official_gt_dir() / dataset / "all" / "homographies.json"
    return {r["image_id"]: r for r in json.load(open(f))["records"]}


def _build_from_official(sport: str, dataset: str, frame_label: str) -> list[dict]:
    """Supervision = official GT homographies + official axis-landmark lattice.

    Weights: gold 1.0, silver 0.8 (tennis silver clicks carry a measured <=1.5 px
    offset; badminton silver passes thresholds at lower inlier ratio), scaled by
    exp(-residual/4 px) where the artifact stores a fit residual.
    Visibility is computed per image at its NATIVE resolution.
    """
    pack = T.sport_pack(sport)
    lattice = pack["lattice"]
    out = []
    for iid, rec in sorted(_official(dataset).items()):
        if rec.get("status") != "valid":
            # Negative record for learned rejection: a manifest frame without a
            # court annotation = non-canonical / non-registrable view. Supervises the
            # landmark-PRESENCE channel to 0 ("not a registrable view"); the seg loss
            # is masked on these frames (court pixels may still exist, e.g. top-down
            # views), so they do not alter the semantic segmentation targets.
            row = paths.manifest_index(dataset).get(iid)
            if row is None or not paths.data_path(row["image_path"]).exists():
                continue
            K = len(lattice)
            out.append(dict(image_id=iid, sport=sport, split=rec["split"],
                            court_visible=0,
                            H_sup=np.eye(3).tolist(),
                            lattice_uv=np.zeros((K, 2)).tolist(),
                            vis=[0] * K, weight=1.0, tier="negative",
                            frame=frame_label, native_wh=rec.get("native_wh"),
                            gt_source="OFFICIAL_BENCHMARK_GT(non-registrable)",
                            geometry_source=pack.get("geometry_source", "builtin")))
            continue
        H = np.array(rec["metric_to_image"], np.float64)
        nw, nh = rec.get("native_wh") or [NATIVE_W, NATIVE_H]
        uv = project(H, lattice)
        vis = ((uv[:, 0] >= -VIS_MARGIN) & (uv[:, 0] < nw + VIS_MARGIN)
               & (uv[:, 1] >= -VIS_MARGIN) & (uv[:, 1] < nh + VIS_MARGIN))
        # normalize stored uv to the 1920x1080 reference frame used downstream
        uv_ref = uv * np.array([NATIVE_W / nw, NATIVE_H / nh])
        H_ref = np.diag([NATIVE_W / nw, NATIVE_H / nh, 1.0]) @ H
        tier = rec.get("tier", "silver")
        resid = (rec.get("validation") or {}).get("residual_px720_inlier_rmse")
        weight = (1.0 if tier == "gold" else 0.8)
        if resid is not None and np.isfinite(resid):
            weight *= math.exp(-float(resid) / 4.0)
        out.append(dict(image_id=iid, sport=sport, split=rec["split"],
                        court_visible=1,
                        H_sup=H_ref.tolist(), lattice_uv=uv_ref.round(3).tolist(),
                        vis=vis.astype(int).tolist(), weight=round(weight, 4), tier=tier,
                        frame=frame_label, native_wh=[int(nw), int(nh)],
                        gt_source="OFFICIAL_BENCHMARK_GT",
                        geometry_source=pack.get("geometry_source", "builtin")))
    return out


def build_tennis() -> list[dict]:
    return _build_from_official("tennis", "tennis_fullcourt", "official_itf_10.97x23.77_line_axis")


def build_badminton() -> list[dict]:
    return _build_from_official("badminton", "badminton_zones", "bwf_13.4x6.1_line_axis")


def build_all(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for sport, builder in [("tennis", build_tennis), ("badminton", build_badminton)]:
        recs = builder()
        (out_dir / f"{sport}.json").write_text(json.dumps(
            {"records": recs, "n": len(recs),
             "lattice_K": len(T.sport_pack(sport)["lattice"])}, indent=1))
        vis_frac = float(np.mean([np.mean(r["vis"]) for r in recs])) if recs else 0.0
        stats[sport] = {"n": len(recs),
                        "by_split": {s: sum(r["split"] == s for r in recs)
                                     for s in ("train", "val", "test", "unassigned")},
                        "mean_visible_fraction": round(vis_frac, 4),
                        "mean_weight": round(float(np.mean([r["weight"] for r in recs])), 4)}
    return stats
