"""Homography utilities in numpy (supervision building, evaluation, sanity).

Conventions: H maps metric (x, y) -> image (u, v); points are (N, 2) arrays.
"""

from __future__ import annotations

import numpy as np


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ H.T
    return p[:, :2] / p[:, 2:3]


def normalize_points(pts: np.ndarray):
    """Hartley normalization: similarity T s.t. centroid 0, mean dist sqrt(2)."""
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return project(T, pts), T


def weighted_dlt(src: np.ndarray, dst: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted DLT with Hartley normalization. src: metric, dst: image, w >= 0."""
    if w is None:
        w = np.ones(len(src))
    sn, Ts = normalize_points(src)
    dn, Td = normalize_points(dst)
    sw = np.sqrt(np.maximum(np.asarray(w, np.float64), 1e-12))
    A = []
    for (x, y), (u, v), k in zip(sn, dn, sw):
        A.append(k * np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u]))
        A.append(k * np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v]))
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def reproj_rmse(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    e = project(H, src) - dst
    return float(np.sqrt((e ** 2).sum(axis=1).mean()))
