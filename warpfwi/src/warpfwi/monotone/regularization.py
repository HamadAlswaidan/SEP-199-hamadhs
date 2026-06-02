"""Regularization terms for monotone raw increment logits."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from warpfwi.config import MonotoneRegularizationConfig
from warpfwi.schedules import lambda_at


@dataclass
class MonotoneRegTerms:
    identity: torch.Tensor
    time_smooth: torch.Tensor
    receiver_smooth: torch.Tensor

    @property
    def total_unweighted(self) -> torch.Tensor:
        return self.identity + self.time_smooth + self.receiver_smooth


def regularization_terms(
    a_raw: torch.Tensor,
    cfg: MonotoneRegularizationConfig,
) -> tuple[torch.Tensor, MonotoneRegTerms]:
    """Return weighted regularizer and detached-friendly component terms."""
    if a_raw.ndim < 2:
        raise ValueError(f"a_raw must have at least (R, nt-1), got {tuple(a_raw.shape)}")
    identity = torch.mean(a_raw ** 2)
    if a_raw.shape[-1] > 1:
        time_smooth = torch.mean((a_raw[..., 1:] - a_raw[..., :-1]) ** 2)
    else:
        time_smooth = torch.zeros((), device=a_raw.device, dtype=a_raw.dtype)
    if a_raw.shape[-2] > 1:
        receiver_smooth = torch.mean((a_raw[..., 1:, :] - a_raw[..., :-1, :]) ** 2)
    else:
        receiver_smooth = torch.zeros((), device=a_raw.device, dtype=a_raw.dtype)
    total = (
        float(cfg.identity) * identity
        + float(cfg.time_smooth) * time_smooth
        + float(cfg.receiver_smooth) * receiver_smooth
    )
    return total, MonotoneRegTerms(
        identity=identity,
        time_smooth=time_smooth,
        receiver_smooth=receiver_smooth,
    )


def monotone_lambda_at(k: int, total_iters: int, cfg: MonotoneRegularizationConfig) -> float:
    """Regularization schedule for monotone Stage 1/Stage 2."""
    return lambda_at(k, total_iters, cfg.lambda_0, cfg.lambda_final, cfg.schedule)


__all__ = ["MonotoneRegTerms", "regularization_terms", "monotone_lambda_at"]
