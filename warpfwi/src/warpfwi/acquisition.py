"""Source/receiver geometry and source wavelets for Deepwave.

Deepwave receives models as ``(nz, nx)`` tensors and locations as integer
indices in ``[iz, ix]`` order. Physical coordinates are stored separately in
meters as ``[x_m, z_m]``. Keeping those conventions explicit prevents the most
expensive class of modeling bugs: transposed models with apparently plausible
but physically wrong source/receiver positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import AcquisitionConfig, DataConfig


@dataclass
class Acquisition:
    """Container for Deepwave inputs and geometry metadata."""

    source_locations: torch.Tensor
    receiver_locations: torch.Tensor
    source_amplitudes: torch.Tensor
    dx: float
    dt: float
    nt: int
    f_peak: float
    n_shots: int
    n_receivers: int
    dz: float | None = None
    source_locations_m: torch.Tensor | None = None
    receiver_locations_m: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def ricker(f_peak: float, nt: int, dt: float, device: torch.device) -> torch.Tensor:
    """Return a 1-D Ricker wavelet ``(nt,)`` as float32 on ``device``."""
    t = torch.arange(nt, device=device, dtype=torch.float32) * dt
    t0 = 1.2 / f_peak
    arg = (np.pi * f_peak * (t - t0)) ** 2
    return ((1.0 - 2.0 * arg) * torch.exp(-arg)).to(torch.float32)


def _round_index(values_m: torch.Tensor, spacing_m: float, mode: str) -> torch.Tensor:
    x = values_m / float(spacing_m)
    if mode == "round":
        return torch.round(x).to(torch.long)
    if mode == "floor":
        return torch.floor(x).to(torch.long)
    raise ValueError(f"x_rounding_mode must be 'round' or 'floor', got {mode!r}")


def _depth_index(
    cells: int | None,
    depth_m: float | None,
    legacy_depth_m: float,
    dz: float,
    rounding_mode: str,
) -> int:
    if cells is not None:
        return int(cells)
    meters = float(legacy_depth_m if depth_m is None else depth_m)
    return int(_round_index(torch.tensor([meters], dtype=torch.float32), dz, rounding_mode)[0])


def _legacy_line_indices(n: int, pad: int, nx: int) -> torch.Tensor:
    if n == 1:
        return torch.tensor([nx // 2], dtype=torch.long)
    if pad >= nx - pad:
        raise ValueError(f"pad={pad} too large for nx={nx}")
    return torch.from_numpy(np.round(np.linspace(pad, nx - 1 - pad, n)).astype(np.int64))


def _line_from_config(
    *,
    n: int,
    explicit_m: tuple[float, ...] | None,
    start_m: float | None,
    spacing_m: float | None,
    pad: int,
    nx: int,
    dx: float,
    rounding_mode: str,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n < 1:
        raise ValueError(f"{name} count must be >= 1, got {n}")
    if explicit_m is not None:
        if len(explicit_m) != n:
            raise ValueError(f"{name}_x_m_list length {len(explicit_m)} does not match {n}")
        x_m = torch.tensor(explicit_m, dtype=torch.float32)
        ix = _round_index(x_m, dx, rounding_mode)
        return x_m, ix
    if start_m is not None or spacing_m is not None:
        if start_m is None or spacing_m is None:
            raise ValueError(f"{name}_start_m and {name}_spacing_m must be provided together")
        if spacing_m <= 0.0:
            raise ValueError(f"{name}_spacing_m must be positive, got {spacing_m}")
        x_m = float(start_m) + torch.arange(n, dtype=torch.float32) * float(spacing_m)
        ix = _round_index(x_m, dx, rounding_mode)
        return x_m, ix
    ix = _legacy_line_indices(n, int(pad), nx)
    x_m = ix.to(torch.float32) * float(dx)
    return x_m, ix


def check_locations_in_bounds(
    loc_iz_ix: torch.Tensor,
    nz: int,
    nx: int,
    *,
    name: str,
) -> None:
    """Validate Deepwave ``[iz, ix]`` locations against a ``(nz, nx)`` model."""
    loc = loc_iz_ix.detach().cpu()
    iz = loc[..., 0]
    ix = loc[..., 1]
    bad = (iz < 0) | (iz >= nz) | (ix < 0) | (ix >= nx)
    if bad.any():
        first = torch.nonzero(bad, as_tuple=False)[:10]
        flat_bad = bad.reshape(-1)
        flat_loc = loc.reshape(-1, 2)
        examples = flat_loc[torch.nonzero(flat_bad, as_tuple=False).flatten()[:10]].tolist()
        raise ValueError(
            f"{name} out of bounds for model shape (nz={nz}, nx={nx}); "
            f"first tensor indices={first.tolist()}, locations [iz, ix]={examples}"
        )


def _fit_spread_to_bounds(
    rec_x_m: torch.Tensor,
    *,
    max_x_m: float,
    allow_partial_spread: bool,
    mode: str,
) -> torch.Tensor:
    if bool(((rec_x_m < 0.0) | (rec_x_m > max_x_m)).any()):
        if not allow_partial_spread:
            raise ValueError(
                f"{mode} receiver spread extends outside [0, {max_x_m:g}] m; "
                "set allow_partial_spread=True or adjust receiver positions."
            )
        span = float(rec_x_m.max().item() - rec_x_m.min().item())
        if span <= max_x_m:
            shift = 0.0
            if float(rec_x_m.min().item()) < 0.0:
                shift = -float(rec_x_m.min().item())
            if float(rec_x_m.max().item() + shift) > max_x_m:
                shift -= float(rec_x_m.max().item() + shift - max_x_m)
            rec_x_m = rec_x_m + shift
        else:
            rec_x_m = rec_x_m.clamp(0.0, max_x_m)
    return rec_x_m


def build_acquisition(
    acq_cfg: AcquisitionConfig,
    data_cfg: DataConfig,
    v_shape: tuple[int, int],
    device: torch.device | None = None,
) -> Acquisition:
    """Construct reusable Deepwave acquisition tensors.

    The returned ``source_locations`` and ``receiver_locations`` are integer
    tensors in Deepwave order ``[iz, ix]``. The corresponding meter-coordinate
    tensors are float tensors in physical order ``[x_m, z_m]``.
    """
    nz, nx = map(int, v_shape)
    dx = float(data_cfg.dx)
    dz = float(data_cfg.dz if data_cfg.dz is not None else data_cfg.dx)
    dev = device if device is not None else torch.device(acq_cfg.device or "cpu")
    n_shots = int(acq_cfg.n_shots)
    n_rec = int(acq_cfg.n_receivers)
    if n_rec < 1:
        raise ValueError("n_receivers must be >= 1")
    if n_shots < 1:
        raise ValueError("n_shots must be >= 1")

    rounding = str(acq_cfg.x_rounding_mode)
    src_iz = _depth_index(
        acq_cfg.source_depth_cells,
        acq_cfg.source_depth_m,
        acq_cfg.source_depth,
        dz,
        rounding,
    )
    rec_iz = _depth_index(
        acq_cfg.receiver_depth_cells,
        acq_cfg.receiver_depth_m,
        acq_cfg.receiver_depth,
        dz,
        rounding,
    )

    src_x_m, src_ix = _line_from_config(
        n=n_shots,
        explicit_m=acq_cfg.source_x_m_list,
        start_m=acq_cfg.source_start_m,
        spacing_m=acq_cfg.source_spacing_m,
        pad=acq_cfg.source_pad,
        nx=nx,
        dx=dx,
        rounding_mode=rounding,
        name="source",
    )
    base_rec_x_m, _base_rec_ix = _line_from_config(
        n=n_rec,
        explicit_m=acq_cfg.receiver_x_m_list,
        start_m=(
            0.0
            if acq_cfg.geometry_mode == "split_spread"
            and acq_cfg.receiver_start_m is None
            and acq_cfg.receiver_spacing_m is not None
            else acq_cfg.receiver_start_m
        ),
        spacing_m=acq_cfg.receiver_spacing_m,
        pad=acq_cfg.receiver_pad,
        nx=nx,
        dx=dx,
        rounding_mode=rounding,
        name="receiver",
    )

    max_x_m = float((nx - 1) * dx)
    mode = str(acq_cfg.geometry_mode)
    rec_x_rows: list[torch.Tensor] = []
    if mode == "fixed_spread":
        rec_x = _fit_spread_to_bounds(
            base_rec_x_m,
            max_x_m=max_x_m,
            allow_partial_spread=acq_cfg.allow_partial_spread,
            mode=mode,
        )
        rec_x_rows = [rec_x] * n_shots
    elif mode == "moving_spread":
        origin = float(base_rec_x_m[0].item())
        offsets = base_rec_x_m - origin
        for sx in src_x_m:
            rec_x_rows.append(
                _fit_spread_to_bounds(
                    sx + offsets,
                    max_x_m=max_x_m,
                    allow_partial_spread=acq_cfg.allow_partial_spread,
                    mode=mode,
                )
            )
    elif mode == "split_spread":
        if acq_cfg.receiver_spacing_m is None and acq_cfg.receiver_x_m_list is None:
            spacing = float(dx)
        elif acq_cfg.receiver_x_m_list is not None and n_rec > 1:
            spacing = float(torch.diff(base_rec_x_m).abs().median().item())
        else:
            spacing = float(acq_cfg.receiver_spacing_m or dx)
        offsets = (torch.arange(n_rec, dtype=torch.float32) - (n_rec - 1) / 2.0) * spacing
        for sx in src_x_m:
            rec_x_rows.append(
                _fit_spread_to_bounds(
                    sx + offsets,
                    max_x_m=max_x_m,
                    allow_partial_spread=acq_cfg.allow_partial_spread,
                    mode=mode,
                )
            )
    else:
        raise ValueError(
            "geometry_mode must be one of 'fixed_spread', 'moving_spread', "
            f"or 'split_spread', got {acq_cfg.geometry_mode!r}"
        )

    rec_x_m = torch.stack(rec_x_rows, dim=0)
    rec_ix = _round_index(rec_x_m, dx, rounding)

    src_z = torch.full((n_shots, 1), src_iz, dtype=torch.long)
    src_x = src_ix.view(n_shots, 1)
    source_locations = torch.stack([src_z, src_x], dim=-1).contiguous()

    rec_z = torch.full((n_shots, n_rec), rec_iz, dtype=torch.long)
    receiver_locations = torch.stack([rec_z, rec_ix], dim=-1).contiguous()
    check_locations_in_bounds(source_locations, nz, nx, name="source_locations")
    check_locations_in_bounds(receiver_locations, nz, nx, name="receiver_locations")

    source_locations_m = torch.stack(
        [
            src_x_m.view(n_shots, 1),
            torch.full((n_shots, 1), float(src_iz) * dz),
        ],
        dim=-1,
    ).contiguous()
    receiver_locations_m = torch.stack(
        [rec_x_m, torch.full((n_shots, n_rec), float(rec_iz) * dz)],
        dim=-1,
    ).contiguous()

    wav = ricker(acq_cfg.f_peak, int(acq_cfg.nt), float(acq_cfg.dt), device=dev)
    source_amplitudes = wav.view(1, 1, int(acq_cfg.nt)).expand(n_shots, 1, int(acq_cfg.nt)).contiguous()

    return Acquisition(
        source_locations=source_locations.to(device=dev),
        receiver_locations=receiver_locations.to(device=dev),
        source_amplitudes=source_amplitudes,
        dx=dx,
        dz=dz,
        dt=float(acq_cfg.dt),
        nt=int(acq_cfg.nt),
        f_peak=float(acq_cfg.f_peak),
        n_shots=n_shots,
        n_receivers=n_rec,
        source_locations_m=source_locations_m.to(device=dev),
        receiver_locations_m=receiver_locations_m.to(device=dev),
        metadata={
            "geometry_mode": mode,
            "index_order": "[iz, ix]",
            "physical_order": "[x_m, z_m]",
            "model_shape": (nz, nx),
            "dx": dx,
            "dz": dz,
            "x_rounding_mode": rounding,
        },
    )


__all__ = ["Acquisition", "build_acquisition", "check_locations_in_bounds", "ricker"]
