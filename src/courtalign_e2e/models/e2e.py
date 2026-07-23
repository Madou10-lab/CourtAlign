"""CourtAlign-E2E: registration-aware, parameter-efficient adaptation of SAM 3.

Architecture:
  frozen SAM 3 ViT trunk
  + fine-tuned SAM 3 feature-pyramid neck
  + trained segmentation and geometry heads
  + sport-specific geometry:
      "landmarks"      : landmark heatmaps on FPN -> soft-argmax
      "zone_centroids" : optional differentiable per-zone centroids of the
                         zone probability maps
  -> confidence-weighted differentiable DLT -> H_pred

The registration losses back-propagate through the DLT into the heads AND into
SAM3's neck: the segmentation representation is optimized for the final
registration objective, not for masks alone.

Output dict is CourtAlignE2ELoss/trainer/eval compatible:
  seg_logits (B,C,H,W) | coords (B,K,2 input px) | conf (B,K) | H_pred (B,3,3)
  heatmaps only in "landmarks" mode.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry.homography_torch import weighted_dlt


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class Sam3SegEncoder(nn.Module):
    """Frozen SAM 3 ViT with its trainable pretrained feature-pyramid neck."""

    out_channels = 256
    required_hw = (1008, 1008)

    def __init__(self, trainable: bool = True, checkpoint: str = "facebook/sam3"):
        super().__init__()
        from transformers import Sam3Model
        full = Sam3Model.from_pretrained(checkpoint, dtype=torch.float32)
        enc = full.vision_encoder
        del full
        self.vit = enc.backbone
        self.neck = enc.neck
        self.trainable = trainable
        for p in self.vit.parameters():
            p.requires_grad_(False)
        for p in self.neck.parameters():
            p.requires_grad_(trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        B, _, H, W = x.shape
        assert (H, W) == self.required_hw, f"SAM3 needs 1008x1008, got {H}x{W}"
        with torch.no_grad():
            tokens = self.vit(x).last_hidden_state    # (B, 72*72, 1024)
        h = w = H // 14
        spatial = tokens.view(B, h, w, -1).permute(0, 3, 1, 2)
        if self.trainable:
            fpn, _ = self.neck(spatial)
        else:
            with torch.no_grad():
                fpn, _ = self.neck(spatial)
        return list(fpn)


class CourtAlignE2E(nn.Module):
    def __init__(self, sport_mode: str, n_classes: int, lattice_metric: np.ndarray,
                 heat_temp: float = 0.35, neck_trainable: bool = True,
                 visibility_head: bool = False):
        """visibility_head adds a per-landmark ABSENT bin competing in
        the heatmap softmax. Trained on negative (non-registrable) frames, it
        gives the model an abstention channel: `conf` becomes the absolute
        presence probability instead of relative softmax peakiness.

        In zone_centroids mode, visibility_head attaches a per-zone presence
        head (GAP -> Linear -> sigmoid) so the learned-rejection recipe
        (presence supervision + negative frames + presence gate) carries over
        to the centroid geometric formulation."""
        super().__init__()
        assert sport_mode in ("landmarks", "zone_centroids")
        self.mode = sport_mode
        self.heat_temp = heat_temp
        self.visibility = visibility_head
        self.backbone = Sam3SegEncoder(trainable=neck_trainable)
        self.register_buffer("lattice_metric",
                             torch.tensor(lattice_metric, dtype=torch.float32),
                             persistent=False)
        self.K = len(lattice_metric)
        C = Sam3SegEncoder.out_channels
        self.seg_head = nn.Sequential(conv_block(C, 128), conv_block(128, 64),
                                      nn.Conv2d(64, n_classes, 1))
        if self.mode == "landmarks":
            self.heat_head = nn.Sequential(conv_block(C, 128), conv_block(128, 128),
                                           nn.Conv2d(128, self.K, 1))
            if visibility_head:
                self.absent_head = nn.Linear(C, self.K)   # per-landmark absent logit
        elif visibility_head:
            self.absent_head = nn.Linear(C, self.K)       # per-zone presence logit

    def forward(self, x: torch.Tensor) -> dict:
        B, _, H, W = x.shape
        fpn = self.backbone(x)
        fine, mid = fpn[0], fpn[1]                    # 288^2, 144^2 @1008

        seg_lr = self.seg_head(fine)                  # (B,C,288,288)
        seg = F.interpolate(seg_lr, size=(H, W), mode="bilinear", align_corners=False)
        out = {"seg_logits": seg}

        if self.mode == "landmarks":
            logits = self.heat_head(mid)              # (B,K,144,144)
            Bk, K, hh, hw = logits.shape
            flat = (logits / self.heat_temp).flatten(2)               # (B,K,S)
            if self.visibility:
                a = self.absent_head(mid.mean(dim=(2, 3))) / self.heat_temp  # (B,K)
                full = torch.cat([flat, a.unsqueeze(-1)], dim=-1).softmax(dim=-1)
                prob = full[..., :-1]                                 # spatial mass
                presence = prob.sum(-1)                               # (B,K) in (0,1)
                spatial = prob / presence.clamp_min(1e-6).unsqueeze(-1)
                conf = presence                                       # ABSOLUTE confidence
                out["presence"] = presence
                out["heat_spatial"] = prob.view(Bk, K, hh, hw)
            else:
                prob = flat.softmax(dim=-1)
                spatial = prob
                conf = prob.max(dim=-1).values
            heat = spatial.view(Bk, K, hh, hw)        # normalized spatial distribution
            ys = torch.linspace(0.5, hh - 0.5, hh, device=x.device)
            xs = torch.linspace(0.5, hw - 0.5, hw, device=x.device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            ex = (heat * gx).flatten(2).sum(-1) * (W / hw)
            ey = (heat * gy).flatten(2).sum(-1) * (H / hh)
            coords = torch.stack([ex, ey], dim=-1)
            out["heatmaps"] = heat
        else:
            # differentiable zone centroids: geometry FROM the segmentation.
            # zone class z (1..K) <-> lattice row z-1 (template zone centroids).
            probs = seg_lr.softmax(dim=1)[:, 1:self.K + 1]        # (B,K,288,288)
            _, _, hh, hw = probs.shape
            ys = torch.linspace(0.5, hh - 0.5, hh, device=x.device)
            xs = torch.linspace(0.5, hw - 0.5, hw, device=x.device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            mass = probs.flatten(2).sum(-1)                       # (B,K)
            cx = (probs * gx).flatten(2).sum(-1) / mass.clamp_min(1e-6) * (W / hw)
            cy = (probs * gy).flatten(2).sum(-1) / mass.clamp_min(1e-6) * (H / hh)
            coords = torch.stack([cx, cy], dim=-1)
            if self.visibility:
                # final learned-rejection recipe carried to the centroid
                # formulation: per-zone presence probability = conf = gate signal
                presence = torch.sigmoid(self.absent_head(fine.mean(dim=(2, 3))))
                conf = presence
                out["presence"] = presence
            else:
                frac = mass / (hh * hw)                           # zone area fraction
                conf = frac / frac.max(dim=1, keepdim=True).values.clamp_min(1e-6)

        src = self.lattice_metric.unsqueeze(0).expand(B, -1, -1)
        out.update(coords=coords, conf=conf, H_pred=weighted_dlt(src, coords, conf))
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
