"""CourtAlign-E2E losses with per-image confidence weights and visibility masks.

  L = λ_seg L_seg + λ_heat L_heat + λ_coord L_coord + λ_reproj L_reproj + λ_tmpl L_tmpl

The reprojection term applies a Huber penalty directly to pixel-space errors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..geometry.homography_torch import project_h


def dice_loss(logits: torch.Tensor, target_idx: torch.Tensor, n_classes: int,
              eps: float = 1.0, active_classes: list | None = None) -> torch.Tensor:
    """Multi-class soft Dice. `active_classes` restricts the channel average to a
    subset (the phase-2 'focused classes' of the three-phase class-focused Dice
    schedule shared with the validated CourtAlign-2S badminton recipe); None = all."""
    prob = logits.softmax(dim=1)
    oh = F.one_hot(target_idx.clamp(0, n_classes - 1), n_classes).permute(0, 3, 1, 2).float()
    inter = (prob * oh).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (denom + eps)            # (B, C)
    if active_classes is not None:
        dice = dice[:, active_classes]
    return (1 - dice).mean(dim=1)                       # (B,)


def heatmap_targets(uv: torch.Tensor, vis: torch.Tensor, hh: int, hw: int,
                    input_hw: tuple[int, int], sigma_px: float = 9.0) -> torch.Tensor:
    """Normalized gaussians at supervision landmarks, on the heatmap grid."""
    H, W = input_hw
    B, K, _ = uv.shape
    ys = torch.linspace(0.5, hh - 0.5, hh, device=uv.device) * (H / hh)
    xs = torch.linspace(0.5, hw - 0.5, hw, device=uv.device) * (W / hw)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    d2 = ((gx[None, None] - uv[..., 0, None, None]) ** 2
          + (gy[None, None] - uv[..., 1, None, None]) ** 2)
    t = torch.exp(-d2 / (2 * sigma_px ** 2)) * vis[..., None, None]
    s = t.flatten(2).sum(-1, keepdim=True).clamp_min(1e-8)
    return (t.flatten(2) / s).view(B, K, hh, hw)


def heat_ce(heat: torch.Tensor, target: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between target and predicted spatial distributions."""
    ce = -(target * heat.clamp_min(1e-9).log()).flatten(2).sum(-1)   # (B,K)
    return (ce * vis).sum(dim=1) / vis.sum(dim=1).clamp_min(1.0)


def heat_ce_with_absent(heat_mass: torch.Tensor, presence: torch.Tensor,
                        target: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """Full-categorical CE over spatial bins and the ABSENT bin.

    heat_mass: (B,K,hh,hw) softmax mass over spatial bins (sums to presence_k).
    Visible landmark: target = spatial Gaussian (absent target 0)
                      -> CE = -sum(gauss * log mass).
    Invisible landmark / negative frame: all target mass on the absent bin
                      -> CE = -log(1 - presence).
    Averaged over ALL K landmarks (this is what teaches abstention)."""
    ce_vis = -(target * heat_mass.clamp_min(1e-9).log()).flatten(2).sum(-1)   # (B,K)
    ce_abs = -(1.0 - presence).clamp_min(1e-9).log()                          # (B,K)
    return (vis * ce_vis + (1.0 - vis) * ce_abs).mean(dim=1)


def presence_bce(presence: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """Explicit per-landmark presence calibration (target = visibility flag)."""
    p = presence.clamp(1e-6, 1 - 1e-6)
    return -(vis * p.log() + (1 - vis) * (1 - p).log()).mean(dim=1)


def coord_huber(coords: torch.Tensor, uv: torch.Tensor, vis: torch.Tensor,
                diag: float, delta: float = 0.01) -> torch.Tensor:
    e = (coords - uv).norm(dim=-1) / diag
    h = torch.where(e < delta, 0.5 * e ** 2 / delta, e - 0.5 * delta)
    return (h * vis).sum(dim=1) / vis.sum(dim=1).clamp_min(1.0)


def reproj_loss(H_pred: torch.Tensor, H_sup: torch.Tensor, src_metric: torch.Tensor,
                delta_px: float = 1.0, e_max_px: float = 50.0) -> torch.Tensor:
    """Huber loss on lattice reprojection error measured in input pixels.

    The clamp bounds the contribution of degenerate homographies during the
    first optimization steps while preserving a physically meaningful Huber
    transition in pixels.
    """
    p = project_h(H_pred, src_metric)
    q = project_h(H_sup, src_metric)
    e = (p - q).norm(dim=-1).clamp(max=e_max_px)
    h = torch.where(e < delta_px, 0.5 * e ** 2 / delta_px, e - 0.5 * delta_px)
    return h.mean(dim=1)


def geometry_validity(coords: torch.Tensor, diag: float, min_spread: float = 0.05) -> torch.Tensor:
    """1 if predicted landmarks are spread enough for a meaningful DLT (else the
    geometric losses are gated off for that sample). Guards SVD-degenerate grads."""
    spread = coords.std(dim=1).norm(dim=-1) / diag      # (B,)
    return (spread > min_spread).float()


def template_sample_loss(H_pred: torch.Tensor, seg_target: torch.Tensor,
                         zone_pts_metric: torch.Tensor, zone_cls: torch.Tensor,
                         input_hw: tuple[int, int]) -> torch.Tensor:
    """Differentiable Ĥ↔mask coupling without rasterization: project fixed
    interior points of each metric zone via Ĥ and demand the GT mask agrees.
    zone_pts_metric: (P,2); zone_cls: (P,) class id of each point."""
    Himg, Wimg = input_hw
    B = H_pred.shape[0]
    pts = zone_pts_metric.unsqueeze(0).expand(B, -1, -1)
    p = project_h(H_pred, pts)                                 # (B,P,2)
    gx = (p[..., 0] / (Wimg - 1)) * 2 - 1
    gy = (p[..., 1] / (Himg - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)          # (B,P,1,2)
    onehot = F.one_hot(seg_target.clamp_min(0), int(zone_cls.max()) + 1).permute(0, 3, 1, 2).float()
    sampled = F.grid_sample(onehot, grid, mode="bilinear",
                            padding_mode="zeros", align_corners=True)  # (B,C,P,1)
    idx = zone_cls.view(1, 1, -1, 1).expand(B, 1, -1, 1)
    agree = sampled.gather(1, idx).squeeze(1).squeeze(-1)      # (B,P)
    inb = ((p[..., 0] >= 0) & (p[..., 0] < Wimg) & (p[..., 1] >= 0) & (p[..., 1] < Himg)).float()
    return 1.0 - (agree * inb).sum(1) / inb.sum(1).clamp_min(1.0)


class CourtAlignE2ELoss:
    def __init__(self, cfg: dict, lattice_metric: np.ndarray, zone_polys: dict,
                 zone_class_of_poly: dict, input_hw: tuple[int, int], device):
        from ..geometry.templates import zone_interior_points
        lw = cfg.get("loss_weights", {})
        self.w = {"seg": 1.0, "heat": 1.0, "coord": 1.0, "reproj": 1.0, "tmpl": 0.5,
                  "pres": 0.5, **lw}
        self.input_hw = input_hw
        self.diag = float((input_hw[0] ** 2 + input_hw[1] ** 2) ** 0.5)
        self.src = torch.tensor(lattice_metric, dtype=torch.float32, device=device).unsqueeze(0)
        pts, cls = [], []
        for zid, poly in zone_polys.items():
            q = zone_interior_points(poly, n_side=4)
            pts.append(q)
            cls += [zone_class_of_poly[zid]] * len(q)
        self.zone_pts = torch.tensor(np.concatenate(pts), dtype=torch.float32, device=device)
        self.zone_cls = torch.tensor(cls, dtype=torch.long, device=device)
        self.sigma = float(cfg.get("heat_sigma_px", 9.0))
        formulation = cfg.get("reproj_formulation", "pixel_huber_v1")
        if formulation != "pixel_huber_v1":
            raise ValueError(f"Unsupported reprojection-loss formulation: {formulation!r}")
        self.reproj_delta_px = float(cfg.get("reproj_delta_px", 1.0))
        self.reproj_e_max_px = float(cfg.get("reproj_e_max_px", 50.0))

    def __call__(self, out: dict, batch: dict, n_classes: int,
                 geo_scale: float = 1.0, seg_active_classes: list | None = None) -> tuple:
        """geo_scale: warmup multiplier for the H-dependent terms (reproj, tmpl);
        ramp it 0->1 over the first epochs so landmark/seg losses establish a
        non-degenerate geometry before homography gradients dominate.
        seg_active_classes: phase-2 focused-class subset of the three-phase Dice
        schedule for badminton small-zone recovery, consistent with CourtAlign-2S;
        None = all-class Dice. Applies to the seg term only."""
        B = out["coords"].shape[0]
        w_img = batch["weight"]                                   # (B,)
        dev = out["coords"].device
        cv = batch.get("court_visible", torch.ones(B, device=dev))  # (B,) 0 = negative frame
        if "heat_spatial" in out:             # spatial distribution plus absent bin
            hh, hw = out["heat_spatial"].shape[-2:]
            tgt = heatmap_targets(batch["lattice_uv"], batch["vis"], hh, hw,
                                  self.input_hw, self.sigma)
            heat_term = heat_ce_with_absent(out["heat_spatial"], out["presence"],
                                            tgt, batch["vis"])
        elif "heatmaps" in out:               # optional centroid formulation
            hh, hw = out["heatmaps"].shape[-2:]
            tgt = heatmap_targets(batch["lattice_uv"], batch["vis"], hh, hw,
                                  self.input_hw, self.sigma)
            heat_term = heat_ce(out["heatmaps"], tgt, batch["vis"])
        else:
            heat_term = torch.zeros(B, device=dev)
        pres_term = (presence_bce(out["presence"], batch["vis"])
                     if "presence" in out else torch.zeros(B, device=dev))
        valid = geometry_validity(out["coords"], self.diag)       # (B,) 0/1
        terms = {
            # seg masked on negative frames (their masks may falsely say "background"
            # for visible-but-non-canonical courts, e.g. top-down views)
            "seg": dice_loss(out["seg_logits"], batch["seg"], n_classes,
                             active_classes=seg_active_classes) * cv,
            "heat": heat_term,
            "pres": pres_term,
            "coord": coord_huber(out["coords"], batch["lattice_uv"], batch["vis"], self.diag),
            "reproj": reproj_loss(out["H_pred"], batch["H_sup"],
                                  self.src.expand(B, -1, -1),
                                  delta_px=self.reproj_delta_px,
                                  e_max_px=self.reproj_e_max_px)
                      * valid * geo_scale * cv,
            "tmpl": template_sample_loss(out["H_pred"], batch["seg"],
                                         self.zone_pts, self.zone_cls,
                                         self.input_hw) * valid * geo_scale * cv,
        }
        total = sum(self.w[k] * (v * w_img).mean() for k, v in terms.items())
        logs = {k: float((v * w_img).mean()) for k, v in terms.items()}
        logs["geo_valid_frac"] = float(valid.mean())
        return total, logs
