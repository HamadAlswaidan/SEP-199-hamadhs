"""Progressive spectral unmasking schedule for the INR Fourier encoder.

Given an iteration counter and a total-iterations target, return a ``(B,)``
tensor of per-band weights in ``[0, 1]``. The schedule sweeps a "frontier"
from the lowest band up to the highest over the course of ``total_iters``
iterations; each band's weight is either a hard step at the frontier (``hard``
mode) or a linear ramp of width :attr:`SpectralScheduleConfig.ramp_width`
bands (``soft`` mode).

The caller is responsible for moving the returned tensor to the training
device and calling :meth:`~warpfwi.inr.TwoChannelINR.set_band_weights`. When
:attr:`SpectralScheduleConfig.enabled` is ``False`` the caller should skip
this function entirely and leave the INR's internal buffer unset; that path
is bit-exact with the pre-feature code.
"""
from __future__ import annotations

import torch

from .config import SpectralScheduleConfig


def compute_band_weights(
    iteration: int,
    total_iters: int,
    n_bands: int,
    cfg: SpectralScheduleConfig,
) -> torch.Tensor:
    """Return per-band weights for the current iteration.

    Parameters
    ----------
    iteration:
        Current iteration index ``it``, expected in ``[0, total_iters]``.
        Values outside this range are clamped implicitly via the frontier
        clip to ``[0, n_bands]``.
    total_iters:
        Iteration at which the frontier reaches the top band (``n_bands``).
        Must be strictly positive.
    n_bands:
        Number of Fourier bands ``B``.
    cfg:
        :class:`~warpfwi.config.SpectralScheduleConfig`.

    Returns
    -------
    torch.Tensor
        Shape ``(n_bands,)``, dtype ``float32``, on CPU. All values in
        ``[0, 1]``. If ``cfg.enabled`` is ``False`` the tensor is all-ones.
    """
    if n_bands < 0:
        raise ValueError(f"n_bands must be >= 0, got {n_bands}")
    if total_iters <= 0:
        raise ValueError(f"total_iters must be > 0, got {total_iters}")
    if cfg.ramp_width <= 0:
        raise ValueError(f"ramp_width must be > 0, got {cfg.ramp_width}")
    if cfg.include_low_in_init < 0:
        raise ValueError(
            f"include_low_in_init must be >= 0, got {cfg.include_low_in_init}"
        )

    if not cfg.enabled or n_bands == 0:
        return torch.ones(n_bands, dtype=torch.float32)

    low = min(int(cfg.include_low_in_init), n_bands)
    progress = float(iteration) / float(total_iters)
    frontier = low + (n_bands - low) * progress
    if frontier < 0.0:
        frontier = 0.0
    if frontier > float(n_bands):
        frontier = float(n_bands)

    b = torch.arange(n_bands, dtype=torch.float32)
    if cfg.mode == "hard":
        weights = (frontier > b).to(torch.float32)
    elif cfg.mode == "soft":
        weights = ((frontier - b) / float(cfg.ramp_width)).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown spectral-schedule mode {cfg.mode!r}")
    return weights


def stage1_total_iters(cfg: SpectralScheduleConfig, stage1_iter: int) -> int:
    """Resolve the effective stage-1 frontier-traversal horizon.

    ``cfg.schedule_iters_stage1 is None`` means "use ``stage1_iter``".
    """
    if cfg.schedule_iters_stage1 is None:
        return int(stage1_iter)
    return int(cfg.schedule_iters_stage1)


def stage2_total_iters(
    cfg: SpectralScheduleConfig, stage1_iter: int
) -> int:
    """Resolve the effective stage-2 frontier-traversal horizon.

    Stage 2 carries forward from the end of stage 1's frontier and either
    stays pinned at the top (``schedule_iters_stage2 == 0``, the default) or
    continues ramping for an additional ``schedule_iters_stage2`` iterations.
    The returned value is the ``total_iters`` to pass to
    :func:`compute_band_weights` *when the caller offsets ``iteration`` by
    ``stage1_iter``* so that the frontier reaches ``n_bands`` at
    ``iteration == stage1_iter + schedule_iters_stage2``.
    """
    extra = cfg.schedule_iters_stage2
    if extra is None:
        extra = 0
    extra = int(extra)
    if extra < 0:
        raise ValueError(
            f"schedule_iters_stage2 must be >= 0, got {extra}"
        )
    return int(stage1_iter) + extra


__all__ = [
    "compute_band_weights",
    "stage1_total_iters",
    "stage2_total_iters",
]
