"""Warp synthetic gathers by sampling at monotone tracewise ``psi(t)``."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import MonotoneWarpINR
from .reconstruct import MonotoneReconstruction, reconstruct_psi


@dataclass
class MonotoneWarpOutput:
    """Return record from applying a monotone warp."""

    warped: torch.Tensor
    a_raw: torch.Tensor
    reconstruction: MonotoneReconstruction


def _receiver_identity_grid_y(
    n_rec: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    r = torch.arange(n_rec, device=device, dtype=dtype)
    return (2.0 * r + 1.0) / n_rec - 1.0


def warp_with_psi(d: torch.Tensor, psi: torch.Tensor, dt: float) -> torch.Tensor:
    """Sample ``d(r, psi_r(t))`` with differentiable linear interpolation."""
    if d.ndim != 3:
        raise ValueError(f"d must have shape (S, R, nt), got {tuple(d.shape)}")
    if psi.shape != d.shape:
        raise ValueError(f"psi shape {tuple(psi.shape)} must equal d shape {tuple(d.shape)}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    s, r, nt = d.shape
    device, dtype = d.device, d.dtype
    n = torch.arange(nt, device=device, dtype=dtype)
    base_x = (2.0 * n + 1.0) / nt - 1.0
    t = n * float(dt)
    tau = psi - t
    grid_x = base_x.view(1, 1, nt).expand(s, r, nt) + (2.0 * tau) / (nt * float(dt))
    base_y = _receiver_identity_grid_y(r, device=device, dtype=dtype)
    grid_y = base_y.view(1, r, 1).expand(s, r, nt)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    warped = F.grid_sample(
        d.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped.squeeze(1)


def apply_monotone_warp(
    d: torch.Tensor,
    inr: MonotoneWarpINR,
    grid_inc: torch.Tensor,
    dt: float,
    increment_eps: float,
    detach_psi: bool = False,
) -> MonotoneWarpOutput:
    """Evaluate ``inr``, reconstruct ``psi``, and warp one or more gathers."""
    squeeze = d.ndim == 2
    d_b = d.unsqueeze(0) if squeeze else d
    if d_b.ndim != 3:
        raise ValueError(f"d must have shape (R, nt) or (S, R, nt), got {tuple(d.shape)}")
    a_raw = inr(grid_inc)
    rec = reconstruct_psi(a_raw, dt=dt, eps=increment_eps)
    psi = rec.psi.detach() if detach_psi else rec.psi
    psi_b = psi.unsqueeze(0).expand(d_b.shape[0], *psi.shape)
    warped = warp_with_psi(d_b, psi_b, dt=dt)
    if bool(torch.equal(a_raw.detach(), torch.zeros_like(a_raw.detach()))):
        # Preserve exact identity at initialization while keeping the
        # interpolation graph as the backward path for the first update.
        warped = d_b + (warped - warped.detach())
    return MonotoneWarpOutput(
        warped=warped.squeeze(0) if squeeze else warped,
        a_raw=a_raw.detach() if detach_psi else a_raw,
        reconstruction=rec,
    )


__all__ = ["MonotoneWarpOutput", "apply_monotone_warp", "warp_with_psi"]
