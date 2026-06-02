"""Reusable preprocessing helpers for FWI data and velocity updates."""
from __future__ import annotations

from dataclasses import replace
from typing import Literal, Protocol

import torch

TaperType = Literal["cosine", "hann"]
BoundaryMode = Literal["all", "top"]


class GradientTaperOptions(Protocol):
    gradient_taper_enabled: bool
    gradient_taper_width_cells: int
    gradient_taper_type: str


class BandpassOptions(Protocol):
    bandpass_enabled: bool
    bandpass_fmin_hz: float
    bandpass_fmax_hz: float
    bandpass_order: int
    filter_source_wavelet_for_fwi_band: bool


def _validate_taper_width(shape: tuple[int, int], width_cells: int) -> None:
    if width_cells < 0:
        raise ValueError(f"width_cells must be non-negative, got {width_cells}")
    if width_cells == 0:
        return
    nz, nx = shape
    max_width = min((nz - 1) // 2, (nx - 1) // 2)
    if width_cells > max_width:
        raise ValueError(
            f"width_cells={width_cells} is too large for model shape {shape}; "
            f"maximum safe width is {max_width}"
        )


def _make_1d_taper(
    n: int,
    width_cells: int,
    taper_type: TaperType,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    w = torch.ones(n, device=device, dtype=dtype)
    if width_cells == 0:
        return w
    x = torch.arange(width_cells + 1, device=device, dtype=dtype) / float(width_cells)
    if taper_type == "cosine":
        ramp = torch.sin(0.5 * torch.pi * x)
    elif taper_type == "hann":
        ramp = 0.5 * (1.0 - torch.cos(torch.pi * x))
    else:
        raise ValueError(
            f"taper_type must be one of 'cosine' or 'hann', got {taper_type!r}"
        )
    w[: width_cells + 1] = ramp
    w[-(width_cells + 1) :] = torch.flip(ramp, dims=(0,))
    return w


def make_boundary_taper(
    shape: tuple[int, int],
    width_cells: int,
    taper_type: TaperType = "cosine",
    mode: BoundaryMode = "all",
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a 2-D boundary taper mask for velocity-parameter gradients.

    For ``mode="all"``, the mask is zero on the outermost cells, ramps to one
    over ``width_cells`` cells on each side, and is exactly one in the
    interior. Row and column tapers are multiplied so corners are damped by
    both adjacent boundaries. For ``mode="top"``, only the shallow/top
    boundary is tapered.
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be 2-D (nz, nx), got {shape}")
    _validate_taper_width(shape, int(width_cells))
    dev = device if device is not None else torch.device("cpu")
    kind = str(taper_type).lower()
    if kind not in ("cosine", "hann"):
        raise ValueError(
            f"taper_type must be one of 'cosine' or 'hann', got {taper_type!r}"
        )
    if mode not in ("all", "top"):
        raise ValueError(f"mode must be one of 'all' or 'top', got {mode!r}")
    nz, nx = shape
    z = _make_1d_taper(nz, int(width_cells), kind, device=dev, dtype=dtype)
    if mode == "top":
        mask = torch.ones(shape, device=dev, dtype=dtype)
        mask[: width_cells + 1, :] = z[: width_cells + 1, None]
        return mask
    x = _make_1d_taper(nx, int(width_cells), kind, device=dev, dtype=dtype)
    return z[:, None] * x[None, :]


def make_boundary_freeze_mask(
    shape: tuple[int, int],
    width_cells: int,
    mode: BoundaryMode = "all",
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a binary mask that fully freezes boundary cells when multiplied.

    Ones denote trainable/updateable cells; zeros denote frozen cells. The mask
    is intended for velocity-parameter gradients, not direct velocity edits.
    """
    if len(shape) != 2:
        raise ValueError(f"shape must be 2-D (nz, nx), got {shape}")
    _validate_taper_width(shape, int(width_cells))
    if mode not in ("all", "top"):
        raise ValueError(f"mode must be one of 'all' or 'top', got {mode!r}")
    dev = device if device is not None else torch.device("cpu")
    mask = torch.ones(shape, device=dev, dtype=dtype)
    if width_cells == 0:
        return mask
    if mode == "top":
        mask[:width_cells, :] = 0.0
    else:
        mask[:width_cells, :] = 0.0
        mask[-width_cells:, :] = 0.0
        mask[:, :width_cells] = 0.0
        mask[:, -width_cells:] = 0.0
    return mask


@torch.no_grad()
def apply_gradient_boundary_taper_(
    grad: torch.Tensor | None,
    taper: torch.Tensor | None,
) -> torch.Tensor | None:
    """Apply ``taper`` in place to a velocity-related parameter gradient."""
    if grad is None or taper is None:
        return grad
    if grad.shape != taper.shape:
        raise ValueError(
            f"gradient shape {tuple(grad.shape)} does not match taper shape "
            f"{tuple(taper.shape)}"
        )
    if grad.device != taper.device:
        raise RuntimeError(f"gradient is on {grad.device}, taper is on {taper.device}")
    grad.mul_(taper.to(dtype=grad.dtype))
    return grad


def gradient_taper_from_options(
    opts: GradientTaperOptions,
    shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build a boundary taper from config-like options, or ``None`` if disabled."""
    if not bool(opts.gradient_taper_enabled):
        return None
    return make_boundary_taper(
        shape,
        int(opts.gradient_taper_width_cells),
        str(opts.gradient_taper_type).lower(),  # type: ignore[arg-type]
        "all",
        device=device,
        dtype=dtype,
    )


def time_mute(
    data: torch.Tensor,
    dt: float,
    mute_until_s: float,
    taper_s: float = 0.0,
) -> torch.Tensor:
    """Mute early samples along the last axis, with an optional linear ramp."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if mute_until_s < 0.0:
        raise ValueError(f"mute_until_s must be non-negative, got {mute_until_s}")
    if taper_s < 0.0:
        raise ValueError(f"taper_s must be non-negative, got {taper_s}")
    nt = int(data.shape[-1])
    t = torch.arange(nt, device=data.device, dtype=data.dtype) * float(dt)
    weight = torch.ones(nt, device=data.device, dtype=data.dtype)
    if mute_until_s > 0.0:
        weight = torch.where(t < float(mute_until_s), torch.zeros_like(weight), weight)
    if taper_s > 0.0:
        ramp_end = float(mute_until_s) + float(taper_s)
        ramp = ((t - float(mute_until_s)) / float(taper_s)).clamp(0.0, 1.0)
        weight = torch.where((t >= float(mute_until_s)) & (t < ramp_end), ramp, weight)
    return data * weight.reshape((1,) * (data.ndim - 1) + (nt,))


def offset_mask(
    offsets: torch.Tensor,
    *,
    mode: Literal["all", "near", "far"] = "all",
    max_offset: float | None = None,
    min_offset: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a receiver/shot mask from absolute source-receiver offsets."""
    if mode not in ("all", "near", "far"):
        raise ValueError(f"mode must be one of 'all', 'near', or 'far', got {mode!r}")
    off = offsets.abs()
    mask = torch.ones_like(off, dtype=dtype)
    if mode == "near":
        if max_offset is None:
            raise ValueError("max_offset is required for near-offset selection")
        mask = (off <= float(max_offset)).to(dtype)
    elif mode == "far":
        if min_offset is None:
            raise ValueError("min_offset is required for far-offset selection")
        mask = (off >= float(min_offset)).to(dtype)
    return mask


def validate_bandpass(
    dt: float,
    fmin_hz: float,
    fmax_hz: float,
    order: int,
    nt: int,
) -> None:
    """Validate bandpass settings against the sampling interval and trace length."""
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if nt < 2:
        raise ValueError(f"bandpass requires at least 2 time samples, got {nt}")
    if order < 1:
        raise ValueError(f"bandpass order must be >= 1, got {order}")
    nyquist = 0.5 / dt
    if not (0.0 < fmin_hz < fmax_hz < nyquist):
        raise ValueError(
            "bandpass frequencies must satisfy "
            f"0 < fmin_hz < fmax_hz < Nyquist ({nyquist:.6g} Hz); got "
            f"fmin_hz={fmin_hz}, fmax_hz={fmax_hz}"
        )


def bandpass_time_axis(
    data: torch.Tensor,
    dt: float,
    fmin_hz: float,
    fmax_hz: float,
    order: int = 4,
    *,
    time_dim: int = -1,
) -> torch.Tensor:
    """Apply a differentiable zero-phase Butterworth bandpass along time.

    ``data`` may have any leading dimensions. The implementation multiplies
    the real FFT by the squared
    magnitude response of high-pass and low-pass Butterworth prototypes, which
    is the zero-phase response produced by a forward/backward filter and
    therefore introduces no time shift.
    """
    if data.ndim == 0:
        raise ValueError("bandpass data must have at least one dimension")
    dim = int(time_dim)
    if dim < 0:
        dim += data.ndim
    if dim < 0 or dim >= data.ndim:
        raise ValueError(f"time_dim={time_dim} out of range for shape {tuple(data.shape)}")
    nt = int(data.shape[dim])
    validate_bandpass(dt, float(fmin_hz), float(fmax_hz), int(order), nt)
    if not data.is_floating_point():
        raise TypeError(f"bandpass data must be floating point, got {data.dtype}")

    calc_dtype = data.dtype if data.dtype in (torch.float32, torch.float64) else torch.float32
    x = data if data.dtype == calc_dtype else data.to(calc_dtype)
    freqs = torch.fft.rfftfreq(nt, d=float(dt), device=data.device).to(torch.float64)
    eps = torch.finfo(torch.float64).tiny
    n = float(order)
    lowpass = 1.0 / torch.sqrt(1.0 + (freqs / float(fmax_hz)).pow(2.0 * n))
    highpass = 1.0 / torch.sqrt(1.0 + (float(fmin_hz) / torch.clamp(freqs, min=eps)).pow(2.0 * n))
    highpass = torch.where(freqs > 0.0, highpass, torch.zeros_like(highpass))
    response = (lowpass * highpass).pow(2).to(dtype=x.dtype)
    response_shape = [1] * data.ndim
    response_shape[dim] = int(response.numel())
    response = response.reshape(response_shape)

    spectrum = torch.fft.rfft(x, dim=dim)
    filtered = torch.fft.irfft(spectrum * response, n=nt, dim=dim)
    return filtered.to(dtype=data.dtype) if filtered.dtype != data.dtype else filtered


def bandpass_shot_gather(
    data: torch.Tensor,
    dt: float,
    fmin_hz: float,
    fmax_hz: float,
    order: int = 4,
) -> torch.Tensor:
    """Bandpass a shot gather or dataset; the last tensor axis is time."""
    return bandpass_time_axis(data, dt, fmin_hz, fmax_hz, order=order, time_dim=-1)


def bandpass_dataset(
    data: torch.Tensor,
    dt: float,
    fmin_hz: float,
    fmax_hz: float,
    order: int = 4,
) -> torch.Tensor:
    """Bandpass a full shot dataset; the last tensor axis is time."""
    return bandpass_shot_gather(data, dt, fmin_hz, fmax_hz, order=order)


def maybe_bandpass_dataset(
    data: torch.Tensor,
    dt: float,
    opts: BandpassOptions,
) -> torch.Tensor:
    """Return ``data`` unchanged when disabled, otherwise bandpass it."""
    if not bool(opts.bandpass_enabled):
        return data
    return bandpass_dataset(
        data,
        dt,
        float(opts.bandpass_fmin_hz),
        float(opts.bandpass_fmax_hz),
        int(opts.bandpass_order),
    )


def source_wavelet_filter_enabled(opts: BandpassOptions) -> bool:
    """Return whether FWI source wavelets should be bandpassed before modeling."""
    return bool(opts.bandpass_enabled) and bool(opts.filter_source_wavelet_for_fwi_band)


def synthetic_receiver_filter_enabled(opts: BandpassOptions) -> bool:
    """Return whether modeled receiver data should be bandpassed after propagation."""
    return bool(opts.bandpass_enabled) and not source_wavelet_filter_enabled(opts)


def fwi_filtering_mode(opts: BandpassOptions) -> dict[str, bool]:
    """Return the receiver/source filtering switches implied by preprocessing config."""
    source_filtered = source_wavelet_filter_enabled(opts)
    synthetic_filtered = synthetic_receiver_filter_enabled(opts)
    if source_filtered and synthetic_filtered:
        raise RuntimeError(
            "invalid FWI bandpass mode: source wavelet filtering and synthetic "
            "receiver filtering are both active, which would double-filter synthetics"
        )
    return {
        "observed_receiver_data_filtered": bool(opts.bandpass_enabled),
        "source_wavelet_filtered_before_propagation": source_filtered,
        "synthetic_receiver_data_filtered_after_propagation": synthetic_filtered,
    }


def format_fwi_filtering_mode(prefix: str, opts: BandpassOptions) -> str:
    """Format a concise log line describing FWI bandpass placement."""
    mode = fwi_filtering_mode(opts)
    return (
        f"[{prefix}:preprocess] filtering mode: "
        f"observed receiver data filtered={mode['observed_receiver_data_filtered']}  "
        f"source wavelet filtered before propagation="
        f"{mode['source_wavelet_filtered_before_propagation']}  "
        f"synthetic receiver data filtered after propagation="
        f"{mode['synthetic_receiver_data_filtered_after_propagation']}"
    )


def maybe_bandpass_source_wavelet(
    source_amplitudes: torch.Tensor,
    dt: float,
    opts: BandpassOptions,
) -> torch.Tensor:
    """Return source amplitudes unchanged unless source-wavelet bandpass is active."""
    if not source_wavelet_filter_enabled(opts):
        return source_amplitudes
    return bandpass_time_axis(
        source_amplitudes,
        dt,
        float(opts.bandpass_fmin_hz),
        float(opts.bandpass_fmax_hz),
        int(opts.bandpass_order),
        time_dim=-1,
    )


def maybe_bandpass_acquisition_source(acq, opts: BandpassOptions):
    """Return an acquisition with filtered source amplitudes when requested."""
    filtered = maybe_bandpass_source_wavelet(acq.source_amplitudes, acq.dt, opts)
    if filtered is acq.source_amplitudes:
        return acq
    return replace(acq, source_amplitudes=filtered)


def maybe_bandpass_synthetic_dataset(
    data: torch.Tensor,
    dt: float,
    opts: BandpassOptions,
) -> torch.Tensor:
    """Bandpass generated receiver data unless source filtering already did it."""
    if not synthetic_receiver_filter_enabled(opts):
        return data
    return maybe_bandpass_dataset(data, dt, opts)


__all__ = [
    "apply_gradient_boundary_taper_",
    "bandpass_dataset",
    "bandpass_shot_gather",
    "bandpass_time_axis",
    "format_fwi_filtering_mode",
    "fwi_filtering_mode",
    "gradient_taper_from_options",
    "make_boundary_freeze_mask",
    "make_boundary_taper",
    "maybe_bandpass_acquisition_source",
    "maybe_bandpass_dataset",
    "maybe_bandpass_source_wavelet",
    "maybe_bandpass_synthetic_dataset",
    "offset_mask",
    "source_wavelet_filter_enabled",
    "synthetic_receiver_filter_enabled",
    "time_mute",
    "validate_bandpass",
]
