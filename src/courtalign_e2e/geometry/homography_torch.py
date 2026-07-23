"""Differentiable confidence-weighted DLT (PyTorch).

This is the geometric core of CourtAlign-E2E. Gradients flow from registration losses
through the homography solve into landmark coordinates and confidences, and
from there into the backbone-adapted representation.

Stability choices:
  * Hartley normalization of both point sets (per batch item).
  * weights floored at `w_floor` and normalized to mean 1.
  * H obtained from the right-singular vector of the smallest singular value;
    torch.linalg.svd has a well-defined backward for distinct singular values
    (the normalized system is well-conditioned for >= 4 spread points).
"""

from __future__ import annotations

import torch


def project_h(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """H: (B,3,3), pts: (B,N,2) -> (B,N,2)."""
    ones = torch.ones_like(pts[..., :1])
    p = torch.cat([pts, ones], dim=-1) @ H.transpose(-1, -2)
    return p[..., :2] / p[..., 2:3].clamp_min(1e-8)


def _normalize(pts: torch.Tensor):
    c = pts.mean(dim=1, keepdim=True)                          # (B,1,2)
    # clamp bounds the Hartley scale for near-coincident point sets (degenerate
    # frames); real landmark sets have d >> 1e-3 px, so behavior is unchanged.
    d = (pts - c).norm(dim=-1).mean(dim=1).clamp_min(1e-3)     # (B,)
    s = (2.0 ** 0.5) / d
    B = pts.shape[0]
    T = torch.zeros(B, 3, 3, dtype=pts.dtype, device=pts.device)
    T[:, 0, 0] = s
    T[:, 1, 1] = s
    T[:, 2, 2] = 1.0
    T[:, 0, 2] = -s * c[:, 0, 0]
    T[:, 1, 2] = -s * c[:, 0, 1]
    return project_h(T, pts), T


def weighted_dlt(src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor,
                 w_floor: float = 1e-3, min_spread_px: float = 1.0) -> torch.Tensor:
    """src, dst: (B,N,2); w: (B,N) >= 0. Returns H: (B,3,3), H[2,2] = 1.

    Degeneracy guard: frames whose predicted points are numerically
    coincident (learned-rejection negatives collapse all landmarks to ~one
    point) are substituted with the well-conditioned identity problem
    (dst := src, w := 1). Their H is meaningless BY DESIGN — every loss that
    consumes it is masked/gated for such frames — this guard only keeps the
    batched SVD finite. No gradient flows through the substitution."""
    spread = (dst - dst.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1)  # (B,)
    bad = spread < min_spread_px
    if bad.any():
        dst = torch.where(bad.view(-1, 1, 1), src.detach(), dst)
        w = torch.where(bad.view(-1, 1), torch.ones_like(w), w)
    sn, Ts = _normalize(src)
    dn, Td = _normalize(dst)
    w = w.clamp_min(w_floor)
    w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
    sw = w.sqrt().unsqueeze(-1)                                # (B,N,1)

    x, y = sn[..., 0:1], sn[..., 1:2]
    u, v = dn[..., 0:1], dn[..., 1:2]
    z = torch.zeros_like(x)
    o = torch.ones_like(x)
    r1 = torch.cat([-x, -y, -o, z, z, z, u * x, u * y, u], dim=-1) * sw   # (B,N,9)
    r2 = torch.cat([z, z, z, -x, -y, -o, v * x, v * y, v], dim=-1) * sw
    A = torch.cat([r1, r2], dim=1)                              # (B,2N,9)

    try:
        _, _, Vh = torch.linalg.svd(A, full_matrices=False)
    except torch.linalg.LinAlgError:
        # cuSOLVER can fail to converge on borderline batches where LAPACK
        # succeeds; CPU fallback keeps autograd intact and is hit ~never.
        Vh = torch.linalg.svd(A.cpu(), full_matrices=False)[2].to(A.device)
    h = Vh[:, -1, :]                                            # (B,9)
    Hn = h.view(-1, 3, 3)
    H = torch.linalg.inv(Td) @ Hn @ Ts
    denom = H[:, 2:3, 2:3]
    denom = denom + (denom.abs() < 1e-12).to(denom.dtype) * 1e-12
    return H / denom
