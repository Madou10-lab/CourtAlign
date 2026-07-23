"""Metric court templates, supervision lattices, and court lines (pure numpy).

Frames:
  * tennis  : official ITF doubles court, W=10.97 m (x), L=23.77 m (y).
  * badminton: BWF doubles court, W=6.1 m, L=13.4 m,
               singles 5.18 m, long-service strip 0.72 m, short service 1.98 m.

The supervision lattice contains only visually identifiable line intersections.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------- tennis
TEN_W, TEN_L = 10.97, 23.77

def tennis_corners() -> np.ndarray:
    return np.asarray([(0, 0), (TEN_W, 0), (0, TEN_L), (TEN_W, TEN_L)], np.float64)


# -------------------------------------------------------------------- badminton
BAD_W, BAD_L = 6.1, 13.4


def badminton_corners() -> np.ndarray:
    return np.asarray([(0, 0), (BAD_W, 0), (0, BAD_L), (BAD_W, BAD_L)], np.float64)


# --------------------------------------------------------- frozen court specification
def official_spec() -> dict:
    """Load the versioned court geometry used by training and evaluation."""
    import json
    from .. import paths

    p = paths.official_gt_dir() / "court_spec_tables.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing court specification: {p}")
    return json.loads(p.read_text())


# ------------------------------------------------------------------------ sport
def sport_pack(sport: str) -> dict:
    """Return the frozen landmarks, line segments, and semantic court regions."""
    spec = official_spec()
    if sport == "tennis":
        zones = {1: [(0, 0), (TEN_W, 0), (0, TEN_L), (TEN_W, TEN_L)]}
        lattice = np.asarray(spec["tennis"]["axis_landmarks_m"], np.float64)
        lines = [tuple(map(tuple, s)) for s in spec["tennis"]["axis_line_segments_m"]]
        return dict(sport=sport, W=TEN_W, L=TEN_L, lattice=lattice,
                    lines=lines, corners=tennis_corners(),
                    zone_polys=zones, n_classes=2, zone_class_of_poly={1: 1},
                    geometry_source="frozen_court_spec")
    if sport == "badminton":
        lattice = np.asarray(spec["badminton"]["axis_landmarks_m"], np.float64)
        lines = [tuple(map(tuple, s)) for s in spec["badminton"]["axis_line_segments_m"]]
        zones = {
            int(k): [tuple(p) for p in v]
            for k, v in spec["badminton"]["zones_inner_edges_m"].items()
        }
        return dict(sport=sport, W=BAD_W, L=BAD_L, lattice=lattice,
                    lines=lines, corners=badminton_corners(),
                    zone_polys=zones, n_classes=14,
                    zone_class_of_poly={k: k for k in range(1, 14)},
                    geometry_source="frozen_court_spec")
    raise ValueError(f"unknown sport {sport!r}")


def zone_interior_points(poly_tl_tr_bl_br, n_side: int = 5) -> np.ndarray:
    """Uniform interior sample grid of an axis-aligned metric rectangle (for the
    template-sample loss). Margins avoid boundary ambiguity."""
    tl, tr, bl, _ = [np.asarray(p, np.float64) for p in poly_tl_tr_bl_br]
    u = np.linspace(0.15, 0.85, n_side)
    pts = [tl + a * (tr - tl) + b * (bl - tl) for a in u for b in u]
    return np.asarray(pts, np.float64)
