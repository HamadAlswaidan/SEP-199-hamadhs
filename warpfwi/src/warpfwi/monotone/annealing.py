"""Differentiable low-pass annealing for monotone NTW losses."""
from __future__ import annotations

import torch

from warpfwi.config import MonotoneAnnealConfig


def cutoff_at(k: int, total_iters: int, cfg: MonotoneAnnealConfig) -> float:
    """Return the internally scheduled cutoff frequency for iteration ``k``."""
    if not cfg.enabled:
        return float(cfg.f_max)
    if cfg.schedule != "geometric":
        raise ValueError(f"unknown monotone anneal schedule {cfg.schedule!r}")
    if cfg.f_min <= 0.0 or cfg.f_max <= 0.0:
        raise ValueError("anneal frequencies must be positive")
    if cfg.f_min > cfg.f_max:
        raise ValueError("anneal f_min must be <= f_max")
    if total_iters <= 1:
        return float(cfg.f_max)
    frac = float(k) / float(total_iters - 1)
    return float(cfg.f_min) * (float(cfg.f_max) / float(cfg.f_min)) ** frac


def smooth_lowpass(x: torch.Tensor, dt: float, cutoff_hz: float, order: float = 8.0) -> torch.Tensor:
    """Apply a differentiable Butterworth-style low-pass along time."""
    if cutoff_hz <= 0.0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    if order <= 0.0:
        raise ValueError(f"order must be positive, got {order}")
    freqs = torch.fft.rfftfreq(x.shape[-1], d=float(dt), device=x.device).to(x.dtype)
    ratio = freqs / float(cutoff_hz)
    taper = torch.rsqrt(1.0 + ratio.pow(2.0 * float(order)))
    shape = (1,) * (x.ndim - 1) + (taper.numel(),)
    spec = torch.fft.rfft(x, dim=-1)
    return torch.fft.irfft(spec * taper.view(shape), n=x.shape[-1], dim=-1)


def annealed_l2(
    pred: torch.Tensor,
    obs: torch.Tensor,
    dt: float,
    cutoff_hz: float,
    cfg: MonotoneAnnealConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return low-pass L2 plus filtered prediction/observation tensors."""
    if pred.shape != obs.shape:
        raise ValueError(f"pred/obs shape mismatch: {tuple(pred.shape)} vs {tuple(obs.shape)}")
    if cfg.enabled:
        pred_lp = smooth_lowpass(pred, dt=dt, cutoff_hz=cutoff_hz, order=cfg.order)
        obs_lp = smooth_lowpass(obs, dt=dt, cutoff_hz=cutoff_hz, order=cfg.order)
    else:
        pred_lp = pred
        obs_lp = obs
    return torch.mean((pred_lp - obs_lp) ** 2), pred_lp, obs_lp


__all__ = ["annealed_l2", "cutoff_at", "smooth_lowpass"]
