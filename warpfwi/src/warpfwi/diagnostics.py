"""SNR, SSIM, and warp diagnostics.

All helpers accept CPU or GPU tensors. They return Python floats (for logs)
or small tensors as appropriate.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from .config import GainParam


def snr_db(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Return signal-to-noise ratio in dB of ``estimate`` relative to ``reference``.

    ``SNR = 10 · log10( ‖ref‖² / ‖est − ref‖² )``.
    """
    if estimate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch {tuple(estimate.shape)} vs {tuple(reference.shape)}"
        )
    ref = reference.to(torch.float32)
    est = estimate.to(torch.float32)
    num = torch.sum(ref ** 2)
    den = torch.sum((est - ref) ** 2)
    if float(den) == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(num / den).item())


def ssim(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    window_size: int = 11,
    data_range: float | None = None,
) -> float:
    """Structural similarity, simplified 2-D implementation.

    Uses a uniform box window (cheap, dependency-free). For v1 this is
    precise enough for a relative quality metric; if we later need the exact
    Wang et al. formulation we can depend on ``skimage``.

    Parameters
    ----------
    estimate, reference:
        ``(nz, nx)`` float tensors on the same device.
    window_size:
        Odd integer, default 11.
    data_range:
        Dynamic range of the reference. ``None`` uses ``max − min`` of
        ``reference``.
    """
    if estimate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch {tuple(estimate.shape)} vs {tuple(reference.shape)}"
        )
    if estimate.ndim != 2:
        raise ValueError("ssim expects 2D tensors")
    if window_size % 2 != 1:
        raise ValueError("window_size must be odd")
    est = estimate.to(torch.float32).unsqueeze(0).unsqueeze(0)
    ref = reference.to(torch.float32).unsqueeze(0).unsqueeze(0)
    device = est.device
    w = torch.ones((1, 1, window_size, window_size), device=device, dtype=torch.float32)
    w = w / float(window_size * window_size)
    pad = window_size // 2

    mu_x = F.conv2d(est, w, padding=pad)
    mu_y = F.conv2d(ref, w, padding=pad)
    mu_x2, mu_y2 = mu_x ** 2, mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(est * est, w, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(ref * ref, w, padding=pad) - mu_y2
    sigma_xy = F.conv2d(est * ref, w, padding=pad) - mu_xy

    if data_range is None:
        data_range = float((reference.max() - reference.min()).item())
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return float((num / den).mean().item())


def warp_stats(
    tau: torch.Tensor,
    gain_raw: torch.Tensor | None,
    gain_param: GainParam,
    T_max: float,
) -> dict[str, float]:
    """Return summary statistics of the warp and gain fields.

    Parameters
    ----------
    tau:
        Bounded time-shift field.
    gain_raw:
        Raw INR gain channel, or ``None`` for
        :attr:`~warpfwi.config.GainParam.NONE`.
    gain_param:
        The :class:`~warpfwi.config.GainParam` value selecting which gain
        statistics to report. See DESIGN.md §11.
    T_max:
        Structural bound on ``|τ|`` used to report saturation.

    Returns
    -------
    dict[str, float]
        Always includes ``max_abs_tau``, ``mean_abs_tau``, ``saturation``.
        For :attr:`~warpfwi.config.GainParam.ADDITIVE` also
        ``mean_abs_gain_raw``, ``max_abs_gain_raw``, and the collapse
        indicator ``min_gain_factor = min(1 + gain_raw)``. For
        :attr:`~warpfwi.config.GainParam.MULTIPLICATIVE` also
        ``mean_abs_gain_raw``, ``max_abs_gain_raw``, ``min_gain_factor``,
        and ``max_gain_factor`` (the latter two being ``exp(gain_raw)``
        extrema). :attr:`~warpfwi.config.GainParam.NONE` omits all
        gain-specific entries.
    """
    with torch.no_grad():
        stats: dict[str, float] = {
            "max_abs_tau": float(tau.abs().max().item()),
            "mean_abs_tau": float(tau.abs().mean().item()),
            "saturation": float(tau.abs().max().item()) / T_max if T_max > 0 else 0.0,
        }
        if gain_param is GainParam.NONE:
            if gain_raw is not None:
                raise ValueError(
                    "gain_raw must be None when gain_param is GainParam.NONE"
                )
            return stats
        if gain_raw is None:
            raise ValueError(
                f"gain_raw must be a tensor when gain_param is {gain_param!r}"
            )
        stats["mean_abs_gain_raw"] = float(gain_raw.abs().mean().item())
        stats["max_abs_gain_raw"] = float(gain_raw.abs().max().item())
        if gain_param is GainParam.ADDITIVE:
            stats["min_gain_factor"] = float((1.0 + gain_raw).min().item())
        elif gain_param is GainParam.MULTIPLICATIVE:
            # exp(g_raw) is the actual multiplier; report its extrema.
            g_factor = torch.exp(gain_raw)
            stats["min_gain_factor"] = float(g_factor.min().item())
            stats["max_gain_factor"] = float(g_factor.max().item())
        else:
            raise ValueError(f"Unknown gain_param {gain_param!r}")
        return stats


def cumulative_warp_stats(
    tau: torch.Tensor,
    delta_raw: torch.Tensor,
    delta_bounded: torch.Tensor,
    T_max: float,
) -> dict[str, float]:
    """Per-iteration diagnostics for the cumulative warp parameterization.

    See DESIGN.md §16.6. All four metrics are evaluated on the *bounded* or
    *reconstructed* quantities (not the raw network outputs), except
    ``delta_saturation`` which uses ``tanh(δ_raw)`` by definition.

    Parameters
    ----------
    tau:
        Reconstructed ``τ(r, t)`` of shape ``(..., R, nt)``.
    delta_raw:
        Raw offset-derivative field, same leading shape as ``tau``. Used only
        for the saturation fraction on ``|tanh(δ_raw)|``.
    delta_bounded:
        Bounded ``δ_max · tanh(δ_raw)``, same shape as ``tau``.
    T_max:
        Structural bound on ``|τ|`` [s]. Used to normalize the outer
        saturation fraction.

    Returns
    -------
    dict[str, float]
        * ``delta_abs_mean`` — ``mean(|δ_bounded|)``.
        * ``delta_saturation`` — fraction of ``|tanh(δ_raw)| > 0.95``.
        * ``tau_outer_saturation`` — fraction of ``|τ| / T_max > 0.95``.
        * ``tau_monotonicity`` — averaged over ``t``, the fraction of
          ``r`` where ``sign(τ[r+1, t] − τ[r, t])`` matches the modal sign
          across ``r`` at that ``t``. Scalar in ``[0.5, 1]``; near 1 means
          the velocity error has a consistent sign across offsets.
    """
    if tau.shape != delta_bounded.shape:
        raise ValueError(
            f"tau/delta_bounded shape mismatch: {tuple(tau.shape)} vs "
            f"{tuple(delta_bounded.shape)}"
        )
    if tau.shape != delta_raw.shape:
        raise ValueError(
            f"tau/delta_raw shape mismatch: {tuple(tau.shape)} vs "
            f"{tuple(delta_raw.shape)}"
        )
    if tau.ndim < 2:
        raise ValueError(f"tau must have at least (R, nt); got {tuple(tau.shape)}")
    with torch.no_grad():
        t = tau.detach().to(torch.float32)
        db = delta_bounded.detach().to(torch.float32)
        dr_raw = delta_raw.detach().to(torch.float32)
        stats: dict[str, float] = {
            "delta_abs_mean": float(db.abs().mean().item()),
            "delta_saturation": float(
                (torch.tanh(dr_raw).abs() > 0.95).to(torch.float32).mean().item()
            ),
        }
        if T_max > 0:
            stats["tau_outer_saturation"] = float(
                (t.abs() / T_max > 0.95).to(torch.float32).mean().item()
            )
        else:
            stats["tau_outer_saturation"] = 0.0

        # tau_monotonicity: sign of ∂_r τ vs modal sign at each t.
        # Flatten leading batch dims (if any) into receiver axis-major slices
        # so the sign comparison is along the receiver axis.
        if t.ndim > 2:
            t2 = t.reshape(-1, t.shape[-2], t.shape[-1])
        else:
            t2 = t.unsqueeze(0)
        # t2 shape: (B, R, nt). Diff along R.
        r_diff = t2[..., 1:, :] - t2[..., :-1, :]  # (B, R-1, nt)
        if r_diff.shape[-2] == 0:
            stats["tau_monotonicity"] = 1.0
        else:
            sign = torch.sign(r_diff)  # values in {-1, 0, 1}
            # Modal sign per t is the sign of the sum (majority vote with
            # ties breaking toward 0). Count agreements (ignore zeros).
            sum_per_t = sign.sum(dim=-2, keepdim=True)  # (B, 1, nt)
            modal = torch.sign(sum_per_t)  # (B, 1, nt)
            # Fraction of sign entries matching modal across r, per (B, t),
            # then mean over all (B, t).
            match = (sign == modal).to(torch.float32)
            frac_per_bt = match.mean(dim=-2)  # (B, nt)
            stats["tau_monotonicity"] = float(frac_per_bt.mean().item())
        return stats


def residual_ratio(
    d_syn: torch.Tensor, d_warped: torch.Tensor, d_obs: torch.Tensor
) -> dict[str, float]:
    """Return ``‖raw‖²``, ``‖warped‖²`` residual energies and their ratio."""
    with torch.no_grad():
        raw = float(((d_syn - d_obs) ** 2).mean().item())
        warped = float(((d_warped - d_obs) ** 2).mean().item())
        return {
            "raw_residual_mse": raw,
            "warped_residual_mse": warped,
            "ratio_warped_over_raw": warped / raw if raw > 0 else 0.0,
        }


def tau_spectral_energy_per_band(
    tau: torch.Tensor,
    n_bands: int,
    omega_max: float,
    dt: float,
) -> np.ndarray:
    """Cumulative energy of ``τ(r, t)`` in each Fourier-feature band.

    Takes a converged time-shift field and an FFT along the time axis,
    then bins the energy spectrum into the same ``B`` logarithmic bands
    that the encoder uses (``ω_b = omega_max^{η_b}``, ``η_b`` linearly
    spaced in ``[0, 1]``). Useful to verify the spectral schedule actually
    band-limits the INR output.

    Parameters
    ----------
    tau:
        Time-shift field ``(R, nt)`` or ``(nt,)``. Values are in whatever
        units the training loop uses (typically seconds); only the energy
        distribution across frequencies matters here.
    n_bands:
        Number of bands ``B`` to report. Matches
        :attr:`~warpfwi.config.INRConfig.n_fourier_bands`.
    omega_max:
        Maximum Fourier-feature frequency. Matches
        :attr:`~warpfwi.config.INRConfig.omega_max`. The band centers are
        normalized frequencies in ``[1, omega_max]``.
    dt:
        Temporal sampling [s]. The FFT frequency axis is scaled by
        ``1 / (nt · dt)`` but because the encoder's bands are in the same
        *normalized* frequency units used for ``τ_n ∈ [−1, 1]``, we only
        need ``nt · dt`` to convert back. In practice the absolute mapping
        is not critical; what matters is that the ordering is preserved.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_bands,)``, dtype ``float64``. Entry ``b`` is the total
        energy of the FFT bins whose absolute frequency falls in band
        ``b``'s half-open interval (edges are the band-center midpoints in
        log-space, with band 0 starting at 0 Hz and the last band ending
        at Nyquist).
    """
    if n_bands <= 0:
        raise ValueError(f"n_bands must be > 0, got {n_bands}")
    if omega_max <= 1.0:
        raise ValueError(f"omega_max must be > 1, got {omega_max}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be 1D or 2D, got shape {tuple(tau.shape)}")
    with torch.no_grad():
        t = tau.detach().to(torch.float32).cpu()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        # Real FFT along the time axis, energy summed over receivers.
        nt = t.shape[-1]
        spec = torch.fft.rfft(t, dim=-1)
        power = (spec.real ** 2 + spec.imag ** 2).sum(dim=0).numpy()  # (nt//2 + 1,)
    freqs = np.fft.rfftfreq(nt, d=dt)  # (nt//2 + 1,)
    nyq = freqs[-1] if freqs[-1] > 0 else 1.0
    # Band centers in normalized units [0, 1] matching the encoder's η_b.
    eta = np.linspace(0.0, 1.0, n_bands)
    centers = np.power(omega_max, eta)  # (B,), in [1, omega_max]
    # Map centers onto the FFT frequency axis: scale so that band 0 is near
    # DC and band B−1 is at Nyquist. Use a log-spaced band-edge scheme
    # anchored at the centers.
    center_freq = centers * (nyq / omega_max)  # (B,)
    # Edges: midpoints in log-space between consecutive centers, extended
    # with 0 at the low end and Nyquist at the high end.
    if n_bands == 1:
        edges = np.array([0.0, nyq], dtype=np.float64)
    else:
        log_centers = np.log(np.maximum(center_freq, 1e-12))
        mid = 0.5 * (log_centers[:-1] + log_centers[1:])
        edges = np.concatenate([[0.0], np.exp(mid), [nyq]])
    out = np.zeros(n_bands, dtype=np.float64)
    for b in range(n_bands):
        lo = edges[b]
        hi = edges[b + 1]
        if b == n_bands - 1:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)
        out[b] = float(power[mask].sum())
    return out


def source_wavelet_bandpass_diagnostics(
    acq,
    preprocess,
    *,
    shot_index: int = 0,
) -> dict[str, object]:
    """Plot the raw source wavelet, configured bandpassed wavelet, and spectra."""
    if not bool(preprocess.bandpass_enabled):
        raise ValueError("source wavelet bandpass diagnostics require bandpass_enabled=True")
    if shot_index < 0 or shot_index >= int(acq.n_shots):
        raise IndexError(f"shot_index={shot_index} outside [0, {acq.n_shots})")

    import matplotlib.pyplot as plt

    from .preprocessing import bandpass_time_axis

    with torch.no_grad():
        raw = acq.source_amplitudes[int(shot_index)].detach()
        filtered = bandpass_time_axis(
            raw,
            acq.dt,
            float(preprocess.bandpass_fmin_hz),
            float(preprocess.bandpass_fmax_hz),
            int(preprocess.bandpass_order),
            time_dim=-1,
        )
        raw_1d = raw.reshape(-1, raw.shape[-1])[0].to(torch.float32).cpu()
        filtered_1d = filtered.reshape(-1, filtered.shape[-1])[0].to(torch.float32).cpu()
        freqs = torch.fft.rfftfreq(raw_1d.numel(), d=float(acq.dt)).cpu()
        raw_amp = torch.fft.rfft(raw_1d).abs()
        filtered_amp = torch.fft.rfft(filtered_1d).abs()
        time_s = torch.arange(raw_1d.numel(), dtype=torch.float32) * float(acq.dt)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].plot(time_s.numpy(), raw_1d.numpy(), label="raw")
    axes[0].plot(time_s.numpy(), filtered_1d.numpy(), label="bandpassed")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude")
    axes[0].legend()
    axes[1].semilogy(freqs.numpy(), raw_amp.numpy() + 1e-12, label="raw")
    axes[1].semilogy(freqs.numpy(), filtered_amp.numpy() + 1e-12, label="bandpassed")
    axes[1].axvline(float(preprocess.bandpass_fmin_hz), color="k", linestyle="--", linewidth=1)
    axes[1].axvline(float(preprocess.bandpass_fmax_hz), color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("amplitude spectrum")
    axes[1].legend()
    return {
        "figure": fig,
        "time_s": time_s,
        "frequency_hz": freqs,
        "raw_source_wavelet": raw_1d,
        "filtered_source_wavelet": filtered_1d,
        "raw_amplitude_spectrum": raw_amp,
        "filtered_amplitude_spectrum": filtered_amp,
    }


def compare_receiver_vs_source_bandpass(
    v: torch.Tensor,
    acq,
    preprocess,
    modeling_cfg=None,
    *,
    shot_index: int = 0,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Compare ``B F(v, q)`` with ``F(v, Bq)`` for one shot and plot the result."""
    if not bool(preprocess.bandpass_enabled):
        raise ValueError("bandpass comparison requires bandpass_enabled=True")
    if shot_index < 0 or shot_index >= int(acq.n_shots):
        raise IndexError(f"shot_index={shot_index} outside [0, {acq.n_shots})")

    import matplotlib.pyplot as plt

    from .modeling import simulate_batch
    from .preprocessing import (
        maybe_bandpass_acquisition_source,
        maybe_bandpass_dataset,
    )

    dev = device if device is not None else v.device
    idx = torch.tensor([int(shot_index)], device=dev, dtype=torch.long)
    source_opts = replace(preprocess, filter_source_wavelet_for_fwi_band=True)
    acq_source = maybe_bandpass_acquisition_source(acq, source_opts)
    with torch.no_grad():
        receiver_filtered = maybe_bandpass_dataset(
            simulate_batch(v, acq, idx, modeling_cfg, device=dev),
            acq.dt,
            preprocess,
        )
        source_filtered = simulate_batch(v, acq_source, idx, modeling_cfg, device=dev)
        diff = receiver_filtered - source_filtered
        denom = receiver_filtered.norm().clamp_min(torch.finfo(receiver_filtered.dtype).eps)
        relative_error = float((diff.norm() / denom).item())

    gathers = [
        (receiver_filtered[0].detach().cpu(), "B F(v, q)"),
        (source_filtered[0].detach().cpu(), "F(v, Bq)"),
        (diff[0].detach().cpu(), "difference"),
    ]
    vmax = max(float(g.abs().max().item()) for g, _title in gathers[:2])
    diff_vmax = float(gathers[2][0].abs().max().item())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, (gather, title) in zip(axes, gathers):
        limit = diff_vmax if title == "difference" else vmax
        im = ax.imshow(
            gather.numpy(),
            aspect="auto",
            cmap="seismic",
            vmin=-limit if limit > 0 else None,
            vmax=limit if limit > 0 else None,
        )
        ax.set_title(title)
        ax.set_xlabel("time sample")
        ax.set_ylabel("receiver")
        fig.colorbar(im, ax=ax, shrink=0.85)
    return {
        "figure": fig,
        "relative_error": relative_error,
        "receiver_filtered_gather": receiver_filtered,
        "source_filtered_gather": source_filtered,
        "difference_gather": diff,
    }


__all__ = [
    "snr_db",
    "ssim",
    "warp_stats",
    "cumulative_warp_stats",
    "residual_ratio",
    "tau_spectral_energy_per_band",
    "source_wavelet_bandpass_diagnostics",
    "compare_receiver_vs_source_bandpass",
]
