"""Shared Deepwave scalar modeling API.

Both classical FWI and Warp-FWI call the helpers here. Autograd is left intact
for :func:`simulate_batch`; Deepwave supplies the differentiable wave-equation
backward pass, so project code only calls ``loss.backward()`` and lets gradients
flow through ``deepwave.scalar`` back to velocity logits.
"""
from __future__ import annotations

from typing import Iterable

import torch
from deepwave import scalar

from .acquisition import Acquisition
from .config import ModelingConfig


def _resolve_modeling_cfg(acq: Acquisition, cfg: ModelingConfig | None) -> ModelingConfig:
    if cfg is None:
        return ModelingConfig(
            dx=acq.dx,
            dz=acq.dz,
            dt=acq.dt,
            pml_freq=acq.f_peak,
        )
    return cfg


def _grid_spacing(acq: Acquisition, cfg: ModelingConfig) -> float | tuple[float, float]:
    dx = float(acq.dx if cfg.dx is None else cfg.dx)
    dz = float(acq.dz if cfg.dz is None else cfg.dz)
    return dx if abs(dx - dz) < 1e-12 else (dz, dx)


def _check_shapes(acq: Acquisition, shot_idx: torch.Tensor) -> None:
    if shot_idx.ndim != 1:
        raise ValueError(f"shot_idx must be 1D, got shape {tuple(shot_idx.shape)}")
    if shot_idx.numel() == 0:
        raise ValueError("shot_idx must contain at least one shot")
    s = int(acq.source_locations.shape[0])
    if int(shot_idx.min()) < 0 or int(shot_idx.max()) >= s:
        raise IndexError(
            f"shot_idx range [{int(shot_idx.min())}, {int(shot_idx.max())}] "
            f"outside [0, {s})"
        )


def simulate_batch(
    v: torch.Tensor,
    acq: Acquisition,
    shot_indices: torch.Tensor,
    modeling_cfg: ModelingConfig | None = None,
    *,
    device: torch.device | None = None,
    detach: bool = False,
) -> torch.Tensor:
    """Forward-model a shot subset with autograd preserved by default.

    Parameters
    ----------
    v:
        Velocity model ``(nz, nx)`` in m/s.
    acq:
        Acquisition tensors. Locations are in Deepwave ``[iz, ix]`` order.
    shot_indices:
        1-D integer tensor selecting shots.
    modeling_cfg:
        Optional :class:`~warpfwi.config.ModelingConfig`.
    device:
        Compatibility argument. When supplied, tensors must already live there.
    detach:
        If ``True``, detach the returned synthetic data explicitly.
    """
    dev = device if device is not None else v.device
    if v.device != dev:
        raise RuntimeError(f"v is on {v.device}, expected {dev}; tensors are not moved silently")
    if v.ndim != 2:
        raise ValueError(f"v must have shape (nz, nx), got {tuple(v.shape)}")
    idx = shot_indices.to(device=dev, dtype=torch.long)
    _check_shapes(acq, idx)
    cfg = _resolve_modeling_cfg(acq, modeling_cfg)
    sl = acq.source_locations.index_select(0, idx)
    rl = acq.receiver_locations.index_select(0, idx)
    sa = acq.source_amplitudes.index_select(0, idx)
    out = scalar(
        v,
        _grid_spacing(acq, cfg),
        float(acq.dt if cfg.dt is None else cfg.dt),
        source_amplitudes=sa,
        source_locations=sl,
        receiver_locations=rl,
        pml_width=int(cfg.pml_width),
        pml_freq=float(acq.f_peak if cfg.pml_freq is None else cfg.pml_freq),
        accuracy=int(cfg.accuracy),
    )
    recv = out[-1]
    return recv.detach() if detach else recv


def simulate_one_shot(
    v: torch.Tensor,
    acq: Acquisition,
    shot_index: int,
    modeling_cfg: ModelingConfig | None = None,
    *,
    device: torch.device | None = None,
    detach: bool = False,
) -> torch.Tensor:
    """Forward-model one shot and return a gather ``(receivers, nt)``."""
    dev = device if device is not None else v.device
    idx = torch.tensor([int(shot_index)], device=dev, dtype=torch.long)
    return simulate_batch(v, acq, idx, modeling_cfg, device=dev, detach=detach)[0]


@torch.no_grad()
def simulate_dataset(
    v: torch.Tensor,
    acq: Acquisition,
    modeling_cfg: ModelingConfig | None = None,
    *,
    device: torch.device | None = None,
    batch_size: int = 4,
    detach: bool = True,
) -> torch.Tensor:
    """Forward-model every shot under ``torch.no_grad``."""
    dev = device if device is not None else v.device
    s = acq.n_shots
    r = acq.n_receivers
    nt = acq.nt
    out = torch.empty((s, r, nt), device=dev, dtype=torch.float32)
    for start in range(0, s, batch_size):
        idx = torch.arange(start, min(start + batch_size, s), device=dev, dtype=torch.long)
        out[idx] = simulate_batch(v, acq, idx, modeling_cfg, device=dev, detach=detach)
    return out


def iter_minibatches(
    n_shots: int,
    batch_size: int,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Iterable[torch.Tensor]:
    """Yield random minibatches of shot indices covering all shots once."""
    perm = torch.randperm(n_shots, generator=generator, device="cpu").to(
        device=device if device is not None else "cpu"
    )
    for start in range(0, n_shots, batch_size):
        yield perm[start : start + batch_size].contiguous().to(torch.long)


__all__ = ["iter_minibatches", "simulate_batch", "simulate_dataset", "simulate_one_shot"]
