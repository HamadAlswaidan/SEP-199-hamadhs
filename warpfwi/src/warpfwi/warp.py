"""Warp-and-gain operator ``W_θ[d](r, t) = g(r, t) · d(r, t − τ(r, t))``.

Implemented with :func:`torch.nn.functional.grid_sample` in ``mode='bilinear'``,
``padding_mode='border'``, ``align_corners=False``.

The multiplicative gain factor ``g`` is chosen from the raw INR channel via
:func:`compute_gain`, dispatched on :class:`~warpfwi.config.GainParam`:

* :attr:`~warpfwi.config.GainParam.NONE`: ``g ≡ 1.0`` (Python float). There is
  no gain channel.
* :attr:`~warpfwi.config.GainParam.ADDITIVE`: ``g = 1 + gain_raw``. At
  ``gain_raw = 0`` this yields ``g = 1`` exactly.
* :attr:`~warpfwi.config.GainParam.MULTIPLICATIVE`: ``g = exp(gain_raw)``. At
  ``gain_raw = 0`` this yields ``g = 1`` exactly (``exp(0) == 1`` in IEEE-754).

Identity invariant
------------------

At ``τ ≡ 0`` and ``gain_raw ≡ 0`` (or ``gain_raw`` absent for
:attr:`~warpfwi.config.GainParam.NONE`) the output equals the input bit-exactly
in float32. Achieving this requires building the sampling grid so that the
normalized coordinate corresponding to time sample ``n`` is exactly the
coordinate that ``grid_sample`` maps back to the cell center at ``n``. See
:func:`_time_identity_grid_x` for the derivation. The identity holds for all
three :class:`~warpfwi.config.GainParam` values because each gain map satisfies
``g(0) = 1`` exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import GainParam, WarpParam
from .inr import INROutput


@dataclass
class WarpGainOutput:
    """Return record from :class:`WarpGain.forward`.

    Supports 3-field tuple unpacking ``(warped, tau, gain_raw)`` for
    backward compatibility with the v1 signature. The additional
    ``delta_bounded`` attribute is ``None`` under
    :attr:`~warpfwi.config.WarpParam.DIRECT` and is the bounded
    offset-derivative field ``δ_max · tanh(δ_raw)`` under
    :attr:`~warpfwi.config.WarpParam.CUMULATIVE` — used by the
    cumulative-parameterization regularizer branch and the cumulative
    diagnostics.
    """

    warped: torch.Tensor
    tau: torch.Tensor
    gain_raw: torch.Tensor | None
    delta_bounded: torch.Tensor | None = None

    def __iter__(self):
        yield self.warped
        yield self.tau
        yield self.gain_raw


# Numerical guard for the ``exp`` branch. ``exp(50) ~ 5.18e21`` — already well
# outside anything that could arise from healthy training. Beyond this we'd
# rather fail loudly than silently produce inf.
_MULTIPLICATIVE_GAIN_RAW_MAX_ABS = 50.0


def compute_gain(
    gain_raw: torch.Tensor | None,
    gain_param: GainParam,
) -> torch.Tensor | float:
    """Map the raw INR gain channel to the multiplicative gain factor ``g``.

    Parameters
    ----------
    gain_raw:
        Raw gain tensor from the INR, or ``None`` for
        :attr:`~warpfwi.config.GainParam.NONE`.
    gain_param:
        The gain parameterization.

    Returns
    -------
    torch.Tensor | float
        ``1.0`` (Python float) for :attr:`~warpfwi.config.GainParam.NONE`,
        ``1 + gain_raw`` for :attr:`~warpfwi.config.GainParam.ADDITIVE`, or
        ``exp(gain_raw)`` for :attr:`~warpfwi.config.GainParam.MULTIPLICATIVE`.

    Notes
    -----
    For :attr:`~warpfwi.config.GainParam.MULTIPLICATIVE` an ``assert`` in
    ``__debug__`` mode checks that ``|gain_raw| < 50`` before ``exp`` to catch
    runaway training; the check is elided under ``python -O``.
    """
    if gain_param is GainParam.NONE:
        if gain_raw is not None:
            raise ValueError(
                "gain_raw must be None when gain_param is GainParam.NONE"
            )
        return 1.0
    if gain_raw is None:
        raise ValueError(
            f"gain_raw must be a tensor when gain_param is {gain_param!r}"
        )
    if gain_param is GainParam.ADDITIVE:
        return 1.0 + gain_raw
    if gain_param is GainParam.MULTIPLICATIVE:
        if __debug__:
            max_abs = float(gain_raw.detach().abs().max().item())
            assert max_abs < _MULTIPLICATIVE_GAIN_RAW_MAX_ABS, (
                f"gain_raw max |·| = {max_abs:.3e} exceeds safety bound "
                f"{_MULTIPLICATIVE_GAIN_RAW_MAX_ABS}; "
                "multiplicative gain would overflow. "
                "Increase β_gain or check the optimizer."
            )
        return torch.exp(gain_raw)
    raise ValueError(f"Unknown gain_param {gain_param!r}")


def reconstruct_tau(
    delta: torch.Tensor,
    tau_0: torch.Tensor,
    Dr_norm: torch.Tensor | float,
    T_max: float,
    delta_max: float,
    outer_tanh: bool,
) -> torch.Tensor:
    """Reconstruct ``τ(r, t)`` from the cumulative-offset parameterization.

    See DESIGN.md §16. The steps:

        δ_bounded = δ_max · tanh(δ)
        τ_0_bounded = T_max · tanh(τ_0)
        τ_inc[r, t] = Σ_{r' < r} δ_bounded[r', t] · Δr_norm     (exclusive cumsum)
        τ[r, t]     = τ_0_bounded[t] + τ_inc[r, t]
        τ          ← T_max · tanh(τ / T_max)                     (if outer_tanh)

    Parameters
    ----------
    delta:
        Raw offset-derivative field of shape ``(R, nt)``.
    tau_0:
        Raw baseline field of shape ``(nt,)``. Taking the ``r = 0`` slice of
        the INR's ``tau_0_raw`` channel is the caller's responsibility.
    Dr_norm:
        Normalized receiver spacing. Scalar for uniform arrays; ``(R,)`` for
        non-uniform geometries (each entry is the spacing used at that step).
    T_max:
        Structural bound on ``|τ|`` [s].
    delta_max:
        Per-receiver-step bound on ``|δ|`` [s].
    outer_tanh:
        If ``True`` re-bound the reconstructed ``τ`` via ``T_max · tanh(τ /
        T_max)`` so ``|τ| ≤ T_max`` is preserved; otherwise ``τ`` is free to
        grow with offset as far as the integrated ``δ`` carries it.

    Returns
    -------
    torch.Tensor
        ``τ(r, t)`` of shape ``(R, nt)``.

    Notes
    -----
    Zero-init invariant: with ``delta ≡ 0`` and ``tau_0 ≡ 0`` the returned
    tensor is exactly zero (``tanh(0) = 0``, cumsum of zeros is zero).
    """
    if delta.ndim != 2:
        raise ValueError(f"delta must be 2D (R, nt); got {tuple(delta.shape)}")
    if tau_0.ndim != 1 or tau_0.shape[0] != delta.shape[1]:
        raise ValueError(
            f"tau_0 must have shape (nt={delta.shape[1]},); got {tuple(tau_0.shape)}"
        )
    delta_bounded = delta_max * torch.tanh(delta)
    tau_0_bounded = T_max * torch.tanh(tau_0)
    if isinstance(Dr_norm, torch.Tensor) and Dr_norm.ndim == 1:
        if Dr_norm.shape[0] != delta.shape[0]:
            raise ValueError(
                f"Dr_norm vector must have shape (R={delta.shape[0]},); "
                f"got {tuple(Dr_norm.shape)}"
            )
        increments = delta_bounded * Dr_norm.unsqueeze(-1)
    else:
        increments = delta_bounded * Dr_norm
    # Exclusive cumsum along the receiver axis: row 0 = 0, row r = Σ_{r'<r}.
    cum = torch.cumsum(increments, dim=0)
    tau_inc = cum - increments
    tau = tau_0_bounded.unsqueeze(0) + tau_inc
    if outer_tanh:
        if T_max <= 0:
            raise ValueError("outer_tanh requires T_max > 0")
        tau = T_max * torch.tanh(tau / T_max)
    return tau


def _time_identity_grid_x(nt: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the 1-D normalized time coordinates that act as identity for ``grid_sample``.

    With ``align_corners=False``, ``grid_sample`` samples the center of pixel
    ``n`` at normalized coordinate ``x = (2n + 1) / nt − 1``. Using this grid
    yields bit-exact identity.
    """
    n = torch.arange(nt, device=device, dtype=dtype)
    return (2.0 * n + 1.0) / nt - 1.0


def _time_identity_grid_y(
    n_rec: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return the 1-D normalized receiver coordinates for the identity mapping."""
    r = torch.arange(n_rec, device=device, dtype=dtype)
    return (2.0 * r + 1.0) / n_rec - 1.0


def warp_and_gain(
    d: torch.Tensor,
    tau: torch.Tensor,
    gain: torch.Tensor | float,
    dt: float,
) -> torch.Tensor:
    """Apply the warp-and-gain operator ``g · d(·, t − τ)`` along time.

    Parameters
    ----------
    d:
        Shot gathers of shape ``(S, R, nt)``, float32. The "time" axis is the
        last one.
    tau:
        Time-shift field of shape ``(S, R, nt)``, float32, in **seconds**.
        The caller is responsible for any tanh bounding: this function uses
        ``tau`` directly.
    gain:
        Multiplicative gain factor ``g``. Either a Python ``float`` (typically
        ``1.0`` for :attr:`~warpfwi.config.GainParam.NONE`) broadcast against
        the warped tensor, or a tensor of the same shape as ``d``.
    dt:
        Temporal sampling [s]. Needed to convert ``tau`` seconds into the
        normalized grid coordinate ``2 · tau / (nt · dt)`` consumed by
        :func:`grid_sample`.

    Returns
    -------
    torch.Tensor
        Warped gathers ``g · d(·, t − τ)``, same shape and dtype as ``d``.

    Notes
    -----
    The implementation treats the ``(R, nt)`` gather as a 2-D "image" and
    calls :func:`grid_sample` with a grid that leaves the receiver axis
    untouched (identity) and shifts only along the time axis. This avoids a
    manual per-receiver loop and is differentiable w.r.t. ``d``, ``tau``, and
    ``gain`` (when ``gain`` is a tensor).

    The identity invariant ``W[d; τ=0, g=1] == d`` in float32 relies on the
    precise form of :func:`_time_identity_grid_x`.
    """
    if d.ndim != 3:
        raise ValueError(f"d must have shape (S, R, nt), got {tuple(d.shape)}")
    if tau.shape != d.shape:
        raise ValueError(
            f"tau shape must match d; got d={tuple(d.shape)}, "
            f"tau={tuple(tau.shape)}"
        )
    if isinstance(gain, torch.Tensor) and gain.shape != d.shape:
        raise ValueError(
            f"gain tensor shape must match d; got d={tuple(d.shape)}, "
            f"gain={tuple(gain.shape)}"
        )

    s, r, nt = d.shape
    device, dtype = d.device, d.dtype

    # Input image for grid_sample: (N, C, H, W) = (S, 1, R, nt)
    img = d.unsqueeze(1)

    # Base grid along time (W axis). The normalized time coordinate that maps
    # sample n -> sample n exactly is (2n + 1)/nt - 1 with align_corners=False.
    base_x = _time_identity_grid_x(nt, device=device, dtype=dtype)  # (nt,)
    base_y = _time_identity_grid_y(r, device=device, dtype=dtype)  # (R,)

    # Broadcast to (S, R, nt) so we can subtract tau (in normalized units).
    # Subtracting shifts the sampling point backward in time, which gives
    # d(·, t − τ) as required.
    # Normalized shift per sample: 2 * tau / (nt * dt).
    shift_norm = (2.0 * tau) / (nt * dt)
    grid_x = base_x.view(1, 1, nt).expand(s, r, nt) - shift_norm  # (S, R, nt)
    grid_y = base_y.view(1, r, 1).expand(s, r, nt)  # (S, R, nt)

    # grid_sample expects grid shape (N, H_out, W_out, 2) with (x, y) ordering.
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (S, R, nt, 2)

    warped = F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )  # (S, 1, R, nt)
    warped = warped.squeeze(1)
    return gain * warped


class WarpGain(nn.Module):
    """Compose a per-shot INR with the warp-and-gain operator.

    This module is stateless except for the wrapped ``inr``. Given a shot
    gather ``d`` and a normalized ``(ξ_r, τ_n)`` mesh, it:

    1. Calls the INR, obtaining an :class:`~warpfwi.inr.INROutput` record.
    2. Applies ``τ = T_max · tanh(τ_raw)`` (if ``T_max > 0``).
    3. Maps ``gain_raw`` to the gain factor ``g`` via :func:`compute_gain`
       using the ``gain_param`` carried by the INR output.
    4. Returns ``g · d(·, t − τ)`` along with the ``(τ, gain_raw)`` fields for
       downstream regularization.

    The ``gain_param`` is resolved at forward time from the INR output record
    (not cached at construction) — this makes the module robust to a swapped-in
    INR and keeps the single source of truth on the INR itself.
    """

    def __init__(
        self,
        inr: nn.Module,
        T_max: float,
        delta_max: float = 0.0,
        outer_tanh: bool = True,
        Dr_norm: torch.Tensor | float = 1.0,
    ) -> None:
        """Create a :class:`WarpGain`.

        Parameters
        ----------
        inr:
            A :class:`torch.nn.Module` whose ``forward(grid_rt)`` returns an
            :class:`~warpfwi.inr.INROutput`. See
            :class:`warpfwi.inr.TwoChannelINR`.
        T_max:
            Structural bound on the time shift [s]. Pass ``0`` to disable the
            tanh bounding (only for debugging identity).
        delta_max:
            Per-receiver-step bound on ``|δ|`` [s], consumed only when the
            wrapped INR is configured with
            :attr:`~warpfwi.config.WarpParam.CUMULATIVE`. Ignored for DIRECT.
        outer_tanh:
            Whether to re-bound the reconstructed ``τ`` by ``T_max ·
            tanh(τ / T_max)`` under CUMULATIVE. Ignored for DIRECT.
        Dr_norm:
            Normalized receiver spacing for the cumulative reconstruction.
            Scalar for uniform arrays; ``(R,)`` tensor for non-uniform. The
            default ``1.0`` matches the convention ``δ_max`` is expressed in
            seconds-per-receiver-step. Ignored for DIRECT.
        """
        super().__init__()
        if T_max < 0:
            raise ValueError(f"T_max must be >= 0, got {T_max}")
        self.inr = inr
        self.T_max = float(T_max)
        self.delta_max = float(delta_max)
        self.outer_tanh = bool(outer_tanh)
        self.Dr_norm = Dr_norm

    def forward(
        self,
        d: torch.Tensor,
        grid_rt: torch.Tensor,
        dt: float,
        detach_warp_fields: bool = False,
    ) -> WarpGainOutput:
        """Apply the warp-and-gain operator to ``d``.

        Parameters
        ----------
        d:
            ``(R, nt)`` single-shot gather, or ``(S, R, nt)`` batch.
        grid_rt:
            ``(R, nt, 2)`` mesh in ``[−1, 1]²`` consumed by the INR.
        dt:
            Temporal sampling [s].
        detach_warp_fields:
            If ``True``, detach the reconstructed ``τ`` and raw gain field
            before applying the warp. This preserves gradients through the
            input gather ``d`` while preventing this branch from updating the
            INR parameters.

        Returns
        -------
        WarpGainOutput
            Record exposing ``warped`` (same shape as ``d``), ``tau`` (the
            bounded ``(R, nt)`` time-shift field), ``gain_raw`` (the raw
            gain channel or ``None`` for
            :attr:`~warpfwi.config.GainParam.NONE`), and ``delta_bounded``
            (``None`` for :attr:`~warpfwi.config.WarpParam.DIRECT`; the
            bounded offset-derivative ``δ_max · tanh(δ_raw)`` for
            :attr:`~warpfwi.config.WarpParam.CUMULATIVE`). The record unpacks
            as the 3-tuple ``(warped, tau, gain_raw)`` for backward
            compatibility.
        """
        out = self.inr(grid_rt)
        if not isinstance(out, INROutput):
            raise TypeError(
                f"INR must return INROutput; got {type(out).__name__}"
            )
        gain_raw = out.gain_raw
        gain_param = out.gain_param
        warp_param = out.warp_param

        delta_bounded: torch.Tensor | None = None
        if warp_param is WarpParam.DIRECT:
            tau_raw = out.tau_raw
            if tau_raw is None:
                raise ValueError(
                    "INROutput.tau_raw must be populated under WarpParam.DIRECT"
                )
            if tau_raw.ndim != 2:
                raise ValueError(
                    f"tau_raw must have shape (R, nt); got {tuple(tau_raw.shape)}"
                )
            if gain_raw is not None and gain_raw.shape != tau_raw.shape:
                raise ValueError(
                    f"gain_raw shape {tuple(gain_raw.shape)} must equal "
                    f"tau_raw shape {tuple(tau_raw.shape)}"
                )
            if self.T_max > 0.0:
                tau = self.T_max * torch.tanh(tau_raw)
            else:
                tau = tau_raw
        elif warp_param is WarpParam.CUMULATIVE:
            delta_raw = out.delta_raw
            tau_0_raw_full = out.tau_0_raw
            if delta_raw is None or tau_0_raw_full is None:
                raise ValueError(
                    "INROutput.delta_raw and tau_0_raw must be populated "
                    "under WarpParam.CUMULATIVE"
                )
            if delta_raw.ndim != 2:
                raise ValueError(
                    f"delta_raw must have shape (R, nt); got {tuple(delta_raw.shape)}"
                )
            if gain_raw is not None and gain_raw.shape != delta_raw.shape:
                raise ValueError(
                    f"gain_raw shape {tuple(gain_raw.shape)} must equal "
                    f"delta_raw shape {tuple(delta_raw.shape)}"
                )
            # Take τ_0 as the r=0 slice; under δ ≡ 0 this gives the intuitive
            # semantics τ(0, t) = T_max · tanh(τ_0_raw[0, t]).
            tau_0 = tau_0_raw_full[0, :]
            Dr_norm = self.Dr_norm
            if isinstance(Dr_norm, torch.Tensor):
                Dr_norm = Dr_norm.to(device=delta_raw.device, dtype=delta_raw.dtype)
            tau = reconstruct_tau(
                delta=delta_raw,
                tau_0=tau_0,
                Dr_norm=Dr_norm,
                T_max=self.T_max,
                delta_max=self.delta_max,
                outer_tanh=self.outer_tanh,
            )
            delta_bounded = self.delta_max * torch.tanh(delta_raw)
        else:
            raise ValueError(f"Unknown warp_param {warp_param!r}")

        if detach_warp_fields:
            tau = tau.detach()
            if gain_raw is not None:
                gain_raw = gain_raw.detach()
            if delta_bounded is not None:
                delta_bounded = delta_bounded.detach()

        squeeze = d.ndim == 2
        if squeeze:
            d_b = d.unsqueeze(0)
            tau_b = tau.unsqueeze(0)
        else:
            if d.shape[-2:] != tau.shape[-2:]:
                raise ValueError(
                    f"d last-two dims {tuple(d.shape[-2:])} must equal tau "
                    f"last-two dims {tuple(tau.shape[-2:])}"
                )
            d_b = d
            tau_b = tau.unsqueeze(0).expand(d.shape[0], *tau.shape)

        gain_factor = compute_gain(gain_raw, gain_param)
        if isinstance(gain_factor, torch.Tensor):
            if squeeze:
                gain_b: torch.Tensor | float = gain_factor.unsqueeze(0)
            else:
                gain_b = gain_factor.unsqueeze(0).expand(
                    d.shape[0], *gain_factor.shape
                )
        else:
            gain_b = gain_factor  # Python float 1.0

        warped = warp_and_gain(d_b, tau_b, gain_b, dt=dt)
        if squeeze:
            warped_out = warped.squeeze(0)
        else:
            warped_out = warped
        return WarpGainOutput(
            warped=warped_out,
            tau=tau,
            gain_raw=gain_raw,
            delta_bounded=delta_bounded,
        )


def normalized_rt_grid(
    n_rec: int, nt: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Return the canonical normalized ``(ξ_r, τ_n)`` mesh on ``[−1, 1]²``.

    Shape ``(R, nt, 2)`` with last dim ``(ξ_r, τ_n)``. Uses cell-centers
    matching :func:`_time_identity_grid_x` so that at ``τ ≡ 0`` the sampler
    is the identity.
    """
    xi_r = (2.0 * torch.arange(n_rec, device=device, dtype=dtype) + 1.0) / n_rec - 1.0
    tau_n = (2.0 * torch.arange(nt, device=device, dtype=dtype) + 1.0) / nt - 1.0
    mesh_r, mesh_t = torch.meshgrid(xi_r, tau_n, indexing="ij")
    return torch.stack([mesh_r, mesh_t], dim=-1).contiguous()


__all__ = [
    "warp_and_gain",
    "WarpGain",
    "WarpGainOutput",
    "normalized_rt_grid",
    "compute_gain",
    "reconstruct_tau",
]
