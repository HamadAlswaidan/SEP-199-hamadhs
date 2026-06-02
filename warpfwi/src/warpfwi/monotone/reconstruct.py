"""Monotone reconstruction ``a_raw -> increments -> psi``."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MonotoneReconstruction:
    """Fields produced by the structural monotone reconstruction."""

    psi: torch.Tensor
    tau: torch.Tensor
    increments: torch.Tensor
    slopes: torch.Tensor


def time_grid(nt: int, dt: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return physical sample times ``0, dt, ..., (nt - 1)dt``."""
    if nt < 2:
        raise ValueError(f"nt must be at least 2, got {nt}")
    return torch.arange(nt, device=device, dtype=dtype) * float(dt)


def reconstruct_psi(
    a_raw: torch.Tensor,
    dt: float,
    eps: float = 0.0,
) -> MonotoneReconstruction:
    """Convert raw logits to tracewise monotone ``psi`` with exact endpoints.

    ``a_raw`` has shape ``(..., R, nt - 1)``. Positive normalized weights are
    produced with a stable softmax, then converted to increments summing to
    ``T = (nt - 1)dt``. With ``a_raw == 0`` the reconstructed ``psi`` is
    exactly the sample-time grid because the learned cumulative distribution
    is compared against an identically constructed uniform cumulative
    distribution.
    """
    if a_raw.ndim < 2:
        raise ValueError(f"a_raw must have at least (R, nt-1), got {tuple(a_raw.shape)}")
    n_inc = int(a_raw.shape[-1])
    if n_inc < 1:
        raise ValueError("a_raw last axis must contain at least one increment")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    total_time = float(n_inc) * float(dt)
    weights = F.softmax(a_raw, dim=-1)
    if eps > 0.0:
        weights = (weights + float(eps)) / (1.0 + float(n_inc) * float(eps))
    increments = weights * total_time
    zeros = torch.zeros(*increments.shape[:-1], 1, device=a_raw.device, dtype=a_raw.dtype)
    cdf = torch.cumsum(weights, dim=-1)
    uniform = torch.full_like(weights, 1.0 / float(n_inc))
    uniform_cdf = torch.cumsum(uniform, dim=-1)
    t = time_grid(n_inc + 1, dt, device=a_raw.device, dtype=a_raw.dtype)
    tau_tail = (cdf - uniform_cdf) * total_time
    tau = torch.cat([zeros, tau_tail], dim=-1)
    psi = t + tau
    # Preserve the endpoint exactly; the normalized cumsum may otherwise miss
    # by a few ulps on long traces.
    psi = psi.clone()
    psi[..., -1] = t[-1]
    tau = psi - t
    slopes = increments / float(dt)
    return MonotoneReconstruction(psi=psi, tau=tau, increments=increments, slopes=slopes)


__all__ = ["MonotoneReconstruction", "reconstruct_psi", "time_grid"]
