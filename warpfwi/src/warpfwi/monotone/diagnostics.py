"""Diagnostics for monotone tracewise warp runs."""
from __future__ import annotations

import torch

from .reconstruct import MonotoneReconstruction


@torch.no_grad()
def monotone_stats(
    a_raw: torch.Tensor,
    rec: MonotoneReconstruction,
    raw_residual: torch.Tensor | None = None,
    warped_residual: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return scalar summaries for logs and notebooks."""
    stats = {
        "a_raw_mean": float(a_raw.mean().item()),
        "a_raw_std": float(a_raw.std(unbiased=False).item()),
        "a_raw_max_abs": float(a_raw.abs().max().item()),
        "tau_mean": float(rec.tau.mean().item()),
        "tau_max_abs": float(rec.tau.abs().max().item()),
        "increment_min": float(rec.increments.min().item()),
        "increment_max": float(rec.increments.max().item()),
        "slope_min": float(rec.slopes.min().item()),
        "slope_max": float(rec.slopes.max().item()),
        "monotone_violation_fraction": float(
            (rec.increments <= 0.0).to(torch.float32).mean().item()
        ),
    }
    if raw_residual is not None:
        stats["raw_residual_norm"] = float(raw_residual.norm().item())
    if warped_residual is not None:
        stats["warped_residual_norm"] = float(warped_residual.norm().item())
    return stats


def selected_psi_curves(
    rec: MonotoneReconstruction,
    receiver_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Extract CPU ``psi`` curves for selected receiver indices."""
    n_rec = rec.psi.shape[-2]
    out: dict[int, torch.Tensor] = {}
    for idx in receiver_indices:
        if idx < 0 or idx >= n_rec:
            raise IndexError(f"receiver index {idx} outside [0, {n_rec})")
        out[int(idx)] = rec.psi[..., idx, :].detach().cpu()
    return out


__all__ = ["monotone_stats", "selected_psi_curves"]
