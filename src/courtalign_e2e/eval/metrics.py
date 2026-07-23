"""CourtAlign-E2E diagnostic metrics computed at native 1920-by-1080 resolution.

Quantitative families (per the corrected protocol):
  * landmark and corner error against official supervision in pixels;
  * lattice reprojection error px (Ĥ vs H_sup);
  * metric projection error in metres (image lattice -> metric via Ĥ⁻¹);
  * projected court-region IoU vs GT mask (mask resolution);
  * projected per-zone mIoU (badminton) vs GT zone mask;
  * line-IoU vs manual OnlyLines strokes (tennis test), tolerances {0,3,5} px;
"""

from __future__ import annotations

import numpy as np
import cv2

from ..geometry.homography_np import project

NATIVE_W, NATIVE_H = 1920, 1080


def corner_metrics(H_pred, H_sup, corners_metric) -> dict:
    p = project(H_pred, corners_metric)
    q = project(H_sup, corners_metric)
    d = np.linalg.norm(p - q, axis=1)
    return {"rmse_px": float(np.sqrt((d ** 2).mean())), "mean_px": float(d.mean()),
            "max_px": float(d.max()),
            "pck5": float((d < 5).mean()), "pck10": float((d < 10).mean())}


def reproj_px(H_pred, H_sup, lattice_metric) -> float:
    p = project(H_pred, lattice_metric)
    q = project(H_sup, lattice_metric)
    return float(np.linalg.norm(p - q, axis=1).mean())


def projection_error_m(H_pred, H_sup, lattice_metric) -> float:
    """Project the GT image positions of the lattice back to metric via Ĥ⁻¹."""
    uv = project(H_sup, lattice_metric)
    inv = np.linalg.inv(H_pred)
    back = project(inv, uv)
    return float(np.linalg.norm(back - lattice_metric, axis=1).mean())


def region_iou(H_pred, corners_metric, gt_mask) -> float:
    h, w = gt_mask.shape[:2]
    S = np.diag([w / NATIVE_W, h / NATIVE_H, 1.0])
    c = project(S @ H_pred, corners_metric)
    tl, tr, bl, br = c
    canvas = np.zeros((h, w), np.uint8)
    cv2.fillPoly(canvas, [np.round(np.array([tl, tr, br, bl])).astype(np.int32)], 1)
    gt = (gt_mask > 0).astype(np.uint8)
    union = np.logical_or(canvas, gt).sum()
    return float(np.logical_and(canvas, gt).sum() / union) if union else float("nan")


def zone_miou(H_pred, zone_polys, gt_mask) -> tuple[float, dict]:
    h, w = gt_mask.shape[:2]
    S = np.diag([w / NATIVE_W, h / NATIVE_H, 1.0])
    proj = np.zeros((h, w), np.uint8)
    for cid, poly in zone_polys.items():
        tl, tr, bl, br = [np.asarray(p) for p in poly]
        q = project(S @ H_pred, np.array([tl, tr, br, bl], np.float64))
        cv2.fillPoly(proj, [np.round(q).astype(np.int32)], int(cid))
    per = {}
    for cid in zone_polys:
        g = gt_mask == cid
        if g.sum():
            inter = np.logical_and(proj == cid, g).sum()
            union = np.logical_or(proj == cid, g).sum()
            per[cid] = float(inter / union)
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def line_iou(H_pred, lines_metric, stroke_mask, tolerances=(0, 3, 5)) -> dict:
    h, w = stroke_mask.shape[:2]
    S = np.diag([w / NATIVE_W, h / NATIVE_H, 1.0])
    canvas = np.zeros((h, w), np.uint8)
    for (a, b) in lines_metric:
        p = project(S @ H_pred, np.array([a, b], np.float64))
        cv2.line(canvas, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), 1, 1)
    gt = (stroke_mask > 0).astype(np.uint8)
    out = {}
    for t in tolerances:
        if t > 0:
            k = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
            a, b = cv2.dilate(canvas, k), cv2.dilate(gt, k)
        else:
            a, b = canvas, gt
        union = np.logical_or(a, b).sum()
        out[str(t)] = float(np.logical_and(a, b).sum() / union) if union else float("nan")
    return out
