"""Marmousi loading, cropping, and smoothed initial model.

The public entry point is :func:`load_velocity`, which returns a pair
``(v_true, v_init)`` of float32 torch tensors on the requested device. If the
configured Marmousi file is missing, a synthetic layered model is returned
instead (with a warning), so tests never depend on external data.
"""
from __future__ import annotations

import os
import warnings
from typing import Callable

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .config import DataConfig


def _load_marmousi_array(path: str) -> np.ndarray:
    """Load a Marmousi array from ``.npy`` or raw ``.bin`` on disk."""
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        # Raw float32, shape inferred from file size assuming Marmousi II
        # native shape (2801 × 13601). If that does not match, fail loudly.
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size != 2801 * 13601:
            raise ValueError(
                f"Raw Marmousi file {path!r} has size {raw.size}, expected "
                f"{2801 * 13601} for a Marmousi II float32 grid."
            )
        arr = raw.reshape(2801, 13601)
    return np.asarray(arr, dtype=np.float32)


def _synthetic_velocity(shape: tuple[int, int]) -> np.ndarray:
    """Build a deterministic layered velocity model for tests and fallbacks.

    Parameters
    ----------
    shape:
        ``(nz, nx)`` output shape.

    Returns
    -------
    numpy.ndarray
        Float32 velocity [m/s] with smooth vertical gradient plus two lateral
        lens-like perturbations. Kept within ``[1500, 4500]`` m/s so it fits
        inside the default ``(v_min, v_max)``.
    """
    nz, nx = shape
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :]
    base = 1800.0 + 2000.0 * z  # linear gradient 1800 → 3800
    lens1 = 300.0 * np.exp(-(((x - 0.3) / 0.08) ** 2 + ((z - 0.55) / 0.08) ** 2))
    lens2 = -200.0 * np.exp(-(((x - 0.7) / 0.1) ** 2 + ((z - 0.35) / 0.08) ** 2))
    v = base + lens1 + lens2
    return np.clip(v, 1500.0, 4500.0).astype(np.float32)


def _crop_and_resample(v: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``v[z0:z1, x0:x1]`` with bounds-safe clamping."""
    z0, z1, x0, x1 = crop
    nz, nx = v.shape
    z0 = max(0, min(z0, nz - 1))
    z1 = max(z0 + 1, min(z1, nz))
    x0 = max(0, min(x0, nx - 1))
    x1 = max(x0 + 1, min(x1, nx))
    return np.ascontiguousarray(v[z0:z1, x0:x1], dtype=np.float32)


def resolved_crop(cfg: DataConfig) -> tuple[int, int, int, int]:
    """Return the explicit ``(z0, z1, x0, x1)`` crop for ``cfg``.

    The scalar notebook controls ``crop_origin_*`` and ``crop_n*`` override the
    tuple crop only when both dimensions are provided. No orientation change is
    performed here: arrays are always interpreted as ``(nz, nx)``.
    """
    if cfg.crop_nx is None and cfg.crop_nz is None:
        return cfg.crop
    if cfg.crop_nx is None or cfg.crop_nz is None:
        raise ValueError("crop_nx and crop_nz must be provided together")
    if cfg.crop_nx <= 0 or cfg.crop_nz <= 0:
        raise ValueError(
            f"crop_nx and crop_nz must be positive, got {cfg.crop_nx}, {cfg.crop_nz}"
        )
    z0 = int(cfg.crop_origin_z)
    x0 = int(cfg.crop_origin_x)
    return (z0, z0 + int(cfg.crop_nz), x0, x0 + int(cfg.crop_nx))


def make_smoothed_initial(v_true: np.ndarray, sigma: float) -> np.ndarray:
    """Return a Gaussian-smoothed copy of ``v_true`` for use as ``v_0``.

    Parameters
    ----------
    v_true:
        Ground-truth velocity ``(nz, nx)`` in m/s.
    sigma:
        Standard deviation in grid samples. Applied isotropically.

    Returns
    -------
    numpy.ndarray
        Float32 ``v_0`` of the same shape.
    """
    if sigma <= 0.0:
        return v_true.astype(np.float32, copy=True)
    smoothed = gaussian_filter(v_true.astype(np.float32), sigma=sigma, mode="nearest")
    return smoothed.astype(np.float32)


def validate_velocity_pair(
    v_true: torch.Tensor,
    v_init: torch.Tensor,
    *,
    v_min: float,
    v_max: float,
) -> None:
    """Validate true/initial velocity tensors before inversion.

    Both tensors must be ``(nz, nx)``. Any transpose must happen explicitly at
    the caller with a comment explaining the source file orientation; this
    helper never guesses or silently reorients data.
    """
    if v_true.ndim != 2 or v_init.ndim != 2:
        raise ValueError(
            f"v_true and v_init must be 2-D (nz, nx), got {tuple(v_true.shape)} "
            f"and {tuple(v_init.shape)}"
        )
    if tuple(v_true.shape) != tuple(v_init.shape):
        raise ValueError(
            f"v_true and v_init must have the same shape, got {tuple(v_true.shape)} "
            f"and {tuple(v_init.shape)}"
        )
    if not torch.isfinite(v_true).all() or not torch.isfinite(v_init).all():
        raise ValueError("v_true and v_init must contain only finite values")
    if float(v_min) >= float(v_max):
        raise ValueError(f"v_min ({v_min}) must be < v_max ({v_max})")


def load_velocity(
    cfg: DataConfig,
    device: torch.device,
    log: Callable[[str], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``(v_true, v_init)`` as float32 tensors on ``device``.

    If ``cfg.marmousi_path`` is ``None`` or points to a missing file, a
    deterministic synthetic velocity of shape ``cfg.synthetic_shape`` is used
    and a warning is emitted via ``log`` (or :mod:`warnings`).

    The initial model ``v_init`` is always obtained by Gaussian-smoothing
    ``v_true`` with standard deviation ``cfg.smooth_sigma`` in grid samples.

    Parameters
    ----------
    cfg:
        :class:`DataConfig`.
    device:
        Target :class:`torch.device`.
    log:
        Optional logging callable. ``None`` uses :mod:`warnings`.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``(v_true, v_init)``, both shape ``(nz, nx)``, float32, on ``device``.
    """
    use_synthetic = (
        cfg.marmousi_path is None
        or not os.path.isfile(cfg.marmousi_path)
    )
    if use_synthetic:
        msg = (
            f"Marmousi file not found at {cfg.marmousi_path!r}; falling back to "
            f"synthetic layered model with shape {cfg.synthetic_shape}."
        )
        if log is not None:
            log(msg)
        else:
            warnings.warn(msg, stacklevel=2)
        v_np = _synthetic_velocity(cfg.synthetic_shape)
    else:
        full = _load_marmousi_array(cfg.marmousi_path)
        v_np = _crop_and_resample(full, resolved_crop(cfg))
    v_np = np.clip(v_np, cfg.v_min + 1.0, cfg.v_max - 1.0)
    v0_np = make_smoothed_initial(v_np, cfg.smooth_sigma)
    v0_np = np.clip(v0_np, cfg.v_min + 1.0, cfg.v_max - 1.0)
    v_true = torch.from_numpy(v_np).to(device=device, dtype=torch.float32)
    v_init = torch.from_numpy(v0_np).to(device=device, dtype=torch.float32)
    validate_velocity_pair(v_true, v_init, v_min=cfg.v_min, v_max=cfg.v_max)
    return v_true, v_init


__all__ = ["load_velocity", "make_smoothed_initial", "resolved_crop", "validate_velocity_pair"]
