"""Matplotlib plotting helpers for velocity models and shot gathers.

Each function returns a :class:`matplotlib.figure.Figure`; none call
``plt.show()``. The notebook is responsible for displaying.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def _to_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _velocity_extent(arr: np.ndarray, dx: float) -> tuple[float, float, float, float]:
    nz, nx = arr.shape
    return (0.0, nx * dx / 1000.0, nz * dx / 1000.0, 0.0)


def _gather_extent(arr: np.ndarray, dt: float) -> tuple[float, float, float, float]:
    n_receivers, nt = arr.shape
    return (0.0, float(n_receivers), nt * dt, 0.0)


def _ensure_velocity_shape(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"velocity must have shape (nz, nx), got {tuple(arr.shape)}")
    return arr


def _ensure_gather_shape(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"shot gather must have shape (R, nt), got {tuple(arr.shape)}")
    return arr


def _compute_symmetric_clim(
    arrays: Sequence[torch.Tensor | np.ndarray],
    robust_pct: float | None = None,
) -> tuple[float, float]:
    """Return a symmetric ``(vmin, vmax)`` around zero for gather plots."""
    if robust_pct is not None and not (0.0 < robust_pct <= 100.0):
        raise ValueError(f"robust_pct must lie in (0, 100], got {robust_pct}")

    abs_values = []
    for arr_like in arrays:
        arr = _ensure_gather_shape(_to_np(arr_like)).astype(np.float64, copy=False)
        abs_values.append(np.abs(arr).ravel())
    if not abs_values:
        raise ValueError("at least one array is required to compute a color scale")
    concat = np.concatenate(abs_values)
    lim = (
        float(np.percentile(concat, robust_pct)) if robust_pct is not None else float(concat.max())
    )
    if not np.isfinite(lim) or lim <= 0.0:
        lim = 1e-6
    return (-lim, lim)


def _compute_velocity_clim(
    arrays: Sequence[np.ndarray],
    robust_pct: float | None = None,
) -> tuple[float, float]:
    """Return a common ``(vmin, vmax)`` for velocity-model plots."""
    if robust_pct is not None and not (0.0 < robust_pct <= 100.0):
        raise ValueError(f"robust_pct must lie in (0, 100], got {robust_pct}")
    flat = np.concatenate(
        [_ensure_velocity_shape(arr).astype(np.float64, copy=False).ravel() for arr in arrays]
    )
    if robust_pct is None:
        vmin = float(flat.min())
        vmax = float(flat.max())
    else:
        lo = (100.0 - robust_pct) / 2.0
        hi = 100.0 - lo
        vmin, vmax = np.percentile(flat, [lo, hi]).astype(float)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("computed non-finite velocity color scale")
    if vmin == vmax:
        vmax = vmin + 1e-6
    return (vmin, vmax)


def _panel_clims(
    arrays: Sequence[torch.Tensor | np.ndarray],
    *,
    shared_clim: bool,
    robust_pct: float | None,
) -> list[tuple[float, float]]:
    if shared_clim:
        clim = _compute_symmetric_clim(arrays, robust_pct=robust_pct)
        return [clim] * len(arrays)
    return [_compute_symmetric_clim([arr], robust_pct=robust_pct) for arr in arrays]


def plot_velocity(
    v: torch.Tensor | np.ndarray,
    dx: float = 1.0,
    title: str = "velocity",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Plot a single ``(nz, nx)`` velocity field with a non-overlapping colorbar."""
    return plot_velocity_panels(
        [v],
        [title],
        dx=dx,
        vmin=vmin,
        vmax=vmax,
        shared_colorbar=False,
    )


def plot_velocity_panels(
    velocities: Sequence[torch.Tensor | np.ndarray],
    titles: Sequence[str],
    dx: float = 1.0,
    dz: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    *,
    shared_colorbar: bool = True,
    robust_pct: float | None = None,
    cmap: str = "turbo",
    colorbar_orientation: str = "vertical",  # "vertical" or "horizontal"
) -> Figure:
    if len(velocities) != len(titles):
        raise ValueError(f"velocities/titles length mismatch: {len(velocities)} vs {len(titles)}")
    if len(velocities) == 0:
        raise ValueError("at least one velocity panel is required")

    if dz is None:
        dz = dx

    arrs = [_ensure_velocity_shape(_to_np(v)) for v in velocities]

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _compute_velocity_clim(arrs, robust_pct=robust_pct)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    n = len(arrs)

    # Make figure wide (critical for your 3:1 models)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5.5 * n, 2.6),
        squeeze=False,
        constrained_layout=True,
    )
    axes_row = list(axes[0])

    images = []

    for ax, arr, title in zip(axes_row, arrs, titles):
        extent = [
            0,
            arr.shape[1] * dx / 1000,  # x in km
            arr.shape[0] * dz / 1000,  # z in km
            0,
        ]

        im = ax.imshow(
            arr,
            extent=extent,
            aspect="equal",  # ✅ FIX: preserve physical geometry
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        images.append(im)

        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")
        ax.set_title(title)

    # ---- Colorbar control (clean like your single plot) ----
    if shared_colorbar:
        cbar = fig.colorbar(
            images[-1],
            ax=axes_row,
            orientation=colorbar_orientation,
            pad=0.02,
        )
        cbar.set_label("v [m/s]")
    else:
        for ax, im in zip(axes_row, images):
            cbar = fig.colorbar(
                im,
                ax=ax,
                orientation=colorbar_orientation,
                pad=0.02,
            )
            cbar.set_label("v [m/s]")

    return fig


def plot_velocity_panel(
    velocities: Sequence[torch.Tensor | np.ndarray],
    titles: Sequence[str],
    dx: float = 1.0,
    dz: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    return plot_velocity_panels(
        velocities,
        titles,
        dx=dx,
        dz=dz,
        vmin=vmin,
        vmax=vmax,
        shared_colorbar=True,
    )


def plot_shot_gather(
    d: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    title: str = "shot gather",
    clip: float | None = None,
) -> Figure:
    """Plot one ``(R, nt)`` shot gather with time increasing downward."""
    arr = _ensure_gather_shape(_to_np(d))
    if clip is None:
        vmin, vmax = _compute_symmetric_clim([arr], robust_pct=99.0)
    else:
        vmin, vmax = (-float(clip), float(clip))
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    im = ax.imshow(
        arr.T,
        extent=_gather_extent(arr, dt),
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("receiver")
    ax.set_ylabel("t [s]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="amplitude", shrink=0.95)
    return fig


def plot_shot_triplet(
    observed: torch.Tensor | np.ndarray,
    synthetic: torch.Tensor | np.ndarray,
    *,
    dt: float,
    shot_idx: int | None = None,
    shared_clim: bool = False,
    robust_pct: float | None = 99.0,
    cmap: str = "RdBu_r",
) -> Figure:
    """Plot observed / synthetic / residual gathers for one selected shot."""
    obs = _ensure_gather_shape(_to_np(observed))
    syn = _ensure_gather_shape(_to_np(synthetic))
    if obs.shape != syn.shape:
        raise ValueError(f"observed/synthetic shape mismatch: {obs.shape} vs {syn.shape}")
    residual = syn - obs
    arrays = [obs, syn, residual]
    titles = [
        f"Observed shot {shot_idx}" if shot_idx is not None else "Observed shot",
        f"Synthetic shot {shot_idx}" if shot_idx is not None else "Synthetic shot",
        (
            f"Residual shot {shot_idx} (synthetic - observed)"
            if shot_idx is not None
            else "Residual (synthetic - observed)"
        ),
    ]
    clims = _panel_clims(arrays, shared_clim=shared_clim, robust_pct=robust_pct)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
    images = []
    for ax, arr, title, (vmin, vmax) in zip(axes, arrays, titles, clims):
        im = ax.imshow(
            arr.T,
            extent=_gather_extent(arr, dt),
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        images.append(im)
        ax.set_xlabel("receiver")
        ax.set_ylabel("t [s]")
        ax.set_title(title)

    if shared_clim:
        fig.colorbar(images[-1], ax=list(axes), label="amplitude", shrink=0.95)
    else:
        for ax, im in zip(axes, images):
            fig.colorbar(im, ax=ax, label="amplitude", shrink=0.95)
    return fig


def plot_stage1_shot_comparison(
    observed: torch.Tensor | np.ndarray,
    original_synthetic: torch.Tensor | np.ndarray,
    warped_synthetic: torch.Tensor | np.ndarray,
    *,
    dt: float,
    shot_idx: int | None = None,
    shared_clim: bool = False,
    robust_pct: float | None = 99.0,
    cmap: str = "RdBu_r",
) -> Figure:
    """Plot a 2x3 stage-1 comparison with repeated observed data."""
    obs = _ensure_gather_shape(_to_np(observed))
    syn = _ensure_gather_shape(_to_np(original_synthetic))
    warped = _ensure_gather_shape(_to_np(warped_synthetic))
    if obs.shape != syn.shape or obs.shape != warped.shape:
        raise ValueError(
            f"stage1 comparison expects matching shapes, got "
            f"obs={obs.shape}, syn={syn.shape}, warped={warped.shape}"
        )
    orig_res = syn - obs
    warped_res = warped - obs
    arrays = [obs, syn, orig_res, obs, warped, warped_res]
    shot_label = f" shot {shot_idx}" if shot_idx is not None else ""
    titles = [
        f"Observed{shot_label}",
        f"Original synthetic{shot_label}",
        f"Original residual{shot_label} (synthetic - observed)",
        f"Observed{shot_label}",
        f"Warped synthetic{shot_label}",
        f"Warped residual{shot_label} (warped - observed)",
    ]
    clims = _panel_clims(arrays, shared_clim=shared_clim, robust_pct=robust_pct)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), constrained_layout=True)
    axes_flat = list(axes.ravel())
    images = []
    for ax, arr, title, (vmin, vmax) in zip(axes_flat, arrays, titles, clims):
        im = ax.imshow(
            arr.T,
            extent=_gather_extent(arr, dt),
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        images.append(im)
        ax.set_xlabel("receiver")
        ax.set_ylabel("t [s]")
        ax.set_title(title)

    if shared_clim:
        fig.colorbar(images[-1], ax=axes_flat, label="amplitude", shrink=0.95)
    else:
        for ax, im in zip(axes_flat, images):
            fig.colorbar(im, ax=ax, label="amplitude", shrink=0.92)
    return fig


def plot_warp_field(
    tau: torch.Tensor | np.ndarray,
    dt: float = 1.0,
    title: str = r"$\tau(r, t)$",
) -> Figure:
    """Plot the time-shift field ``(R, nt)`` in seconds, diverging colormap."""
    arr = _ensure_gather_shape(_to_np(tau))
    _, lim = _compute_symmetric_clim([arr])
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    im = ax.imshow(
        arr.T,
        extent=_gather_extent(arr, dt),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-lim,
        vmax=lim,
    )
    ax.set_xlabel("receiver")
    ax.set_ylabel("t [s]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=r"$\tau$ [s]", shrink=0.95)
    return fig


def plot_loss_curves(
    curves: dict[str, Iterable[float]],
    title: str = "loss decomposition",
    logy: bool = True,
) -> Figure:
    """Plot one or more loss curves on a single axis."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for name, series in curves.items():
        y = np.asarray(list(series), dtype=np.float64)
        ax.plot(y, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return fig


def plot_gradient_image(
    grad: torch.Tensor | np.ndarray,
    dx: float = 1.0,
    title: str = "gradient",
    *,
    absolute: bool = False,
    robust_pct: float | None = 99.0,
    cmap: str | None = None,
) -> Figure:
    """Plot a 2-D optimization gradient or update field."""
    arr = _ensure_velocity_shape(_to_np(grad)).astype(np.float64, copy=False)
    if absolute:
        arr = np.abs(arr)
        vmin, vmax = (0.0, float(np.percentile(arr, robust_pct or 100.0)))
        cmap = cmap or "magma"
    else:
        lim = float(np.percentile(np.abs(arr), robust_pct or 100.0))
        if not np.isfinite(lim) or lim <= 0.0:
            lim = 1e-12
        vmin, vmax = (-lim, lim)
        cmap = cmap or "RdBu_r"
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    im = ax.imshow(
        arr,
        extent=_velocity_extent(arr, dx),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("x [km]")
    ax.set_ylabel("z [km]")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="value", shrink=0.95)
    return fig


def plot_gradient_comparison(
    raw_grad: torch.Tensor | np.ndarray,
    processed_grad: torch.Tensor | np.ndarray,
    dx: float = 1.0,
    *,
    titles: tuple[str, str] = ("raw gradient", "processed gradient"),
    robust_pct: float | None = 99.0,
) -> Figure:
    """Plot raw and processed gradients with one symmetric color scale."""
    raw = _ensure_velocity_shape(_to_np(raw_grad)).astype(np.float64, copy=False)
    proc = _ensure_velocity_shape(_to_np(processed_grad)).astype(np.float64, copy=False)
    if raw.shape != proc.shape:
        raise ValueError(f"gradient shape mismatch: {raw.shape} vs {proc.shape}")
    lim = float(
        np.percentile(np.abs(np.concatenate([raw.ravel(), proc.ravel()])), robust_pct or 100.0)
    )
    if not np.isfinite(lim) or lim <= 0.0:
        lim = 1e-12
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
    images = []
    for ax, arr, title in zip(axes, [raw, proc], titles):
        im = ax.imshow(
            arr,
            extent=_velocity_extent(arr, dx),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
        )
        images.append(im)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")
        ax.set_title(title)
    fig.colorbar(images[-1], ax=list(axes), label="gradient", shrink=0.95)
    return fig


def plot_gradient_histogram(
    grad: torch.Tensor | np.ndarray,
    title: str = "gradient histogram",
    *,
    bins: int = 80,
    logy: bool = True,
) -> Figure:
    """Plot a histogram of flattened gradient values."""
    arr = _ensure_velocity_shape(_to_np(grad)).astype(np.float64, copy=False)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.hist(arr.ravel(), bins=bins)
    ax.set_xlabel("gradient value")
    ax.set_ylabel("count")
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return fig


def plot_depth_profile(
    profile: torch.Tensor | np.ndarray,
    dx: float = 1.0,
    title: str = "depth profile",
) -> Figure:
    """Plot average magnitude versus depth."""
    arr = np.asarray(_to_np(profile), dtype=np.float64).reshape(-1)
    z_km = np.arange(arr.size, dtype=np.float64) * float(dx) / 1000.0
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ax.plot(arr, z_km)
    ax.invert_yaxis()
    ax.set_xlabel("mean |gradient|")
    ax.set_ylabel("z [km]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


def plot_taper_mask(
    mask: torch.Tensor | np.ndarray,
    title: str = "boundary update mask",
) -> Figure:
    """Plot a 2-D taper/freeze mask."""
    arr = _ensure_velocity_shape(_to_np(mask))
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    im = ax.imshow(arr, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xlabel("x cell")
    ax.set_ylabel("z cell")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="gradient multiplier", shrink=0.95)
    return fig


def plot_gather_spectra(
    gathers: dict[str, torch.Tensor | np.ndarray],
    *,
    dt: float,
    trace_idx: int,
    title: str = "trace spectra",
) -> Figure:
    """Plot amplitude spectra for one trace from one or more shot gathers."""
    if trace_idx < 0:
        raise ValueError(f"trace_idx must be non-negative, got {trace_idx}")
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name, gather in gathers.items():
        arr = _ensure_gather_shape(_to_np(gather))
        if trace_idx >= arr.shape[0]:
            raise ValueError(f"trace_idx={trace_idx} outside gather receiver count {arr.shape[0]}")
        trace = arr[trace_idx].astype(np.float64, copy=False)
        freqs = np.fft.rfftfreq(trace.size, d=float(dt))
        amp = np.abs(np.fft.rfft(trace))
        ax.semilogy(freqs, amp + 1e-12, label=name)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return fig


def plot_experiment_velocity_grid(
    results: dict[str, dict[str, object]],
    *,
    dx: float = 1.0,
    max_cols: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Plot final velocity from multiple experiment results."""
    if not results:
        raise ValueError("at least one result is required")
    names = list(results.keys())
    arrs = [_ensure_velocity_shape(_to_np(results[name]["v"])) for name in names]
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = _compute_velocity_clim(arrs)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax
    n = len(arrs)
    ncols = min(max_cols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    images = []
    for ax, name, arr in zip(axes.ravel(), names, arrs):
        im = ax.imshow(
            arr,
            extent=_velocity_extent(arr, dx),
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(im)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")
        ax.set_title(name)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(images[-1], ax=list(axes.ravel()[:n]), label="v [m/s]", shrink=0.95)
    return fig


__all__ = [
    "_compute_symmetric_clim",
    "plot_velocity",
    "plot_velocity_panel",
    "plot_velocity_panels",
    "plot_shot_gather",
    "plot_shot_triplet",
    "plot_stage1_shot_comparison",
    "plot_warp_field",
    "plot_loss_curves",
    "plot_depth_profile",
    "plot_experiment_velocity_grid",
    "plot_gather_spectra",
    "plot_gradient_comparison",
    "plot_gradient_histogram",
    "plot_gradient_image",
    "plot_taper_mask",
]
