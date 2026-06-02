"""Fourier-feature MLP for the warp auxiliary.

``f_θ : [−1, 1]² → ℝ^C`` producing ``τ_raw`` and (optionally) ``gain_raw`` at
each ``(ξ_r, τ_n)`` coordinate. The output layer is zero-initialized so at init
both channels are exactly zero; combined with the identity grid used by
:func:`warpfwi.warp.warp_and_gain` this makes ``W_θ ≡ I`` bit-exactly at init.

The number of output channels depends on the gain parameterization
(see :class:`warpfwi.config.GainParam`):

* :attr:`~warpfwi.config.GainParam.NONE` — 1 output (``τ_raw``).
* :attr:`~warpfwi.config.GainParam.ADDITIVE` or
  :attr:`~warpfwi.config.GainParam.MULTIPLICATIVE` — 2 outputs
  (``τ_raw``, ``gain_raw``).

The forward method always returns an :class:`INROutput` record whose
``gain_raw`` attribute is ``None`` exactly when the INR has no gain channel.
Callers must branch on :attr:`INROutput.gain_raw is None` rather than assuming
a particular output arity: that silent-bug surface is what the record avoids.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from .config import Activation, GainParam, INRConfig, WarpParam


@dataclass
class INROutput:
    """Structured INR output.

    Which of ``tau_raw`` or (``delta_raw``, ``tau_0_raw``) is populated is
    determined by :attr:`warp_param`. The caller dispatches on
    ``warp_param`` (or equivalently on which field is ``None``) to reconstruct
    the bounded time-shift field ``τ(r, t)``.

    Attributes
    ----------
    tau_raw:
        Raw time-shift channel, shape ``(R, nt)``. Populated for
        :attr:`~warpfwi.config.WarpParam.DIRECT`; ``None`` otherwise.
    delta_raw:
        Raw offset-derivative channel, shape ``(R, nt)``. Populated for
        :attr:`~warpfwi.config.WarpParam.CUMULATIVE`; ``None`` otherwise.
    tau_0_raw:
        Raw baseline channel, shape ``(R, nt)`` (the ``r = 0`` slice is taken
        downstream). Populated for :attr:`~warpfwi.config.WarpParam.CUMULATIVE`;
        ``None`` otherwise.
    gain_raw:
        Raw gain channel of shape ``(R, nt)`` when present. ``None`` for
        :attr:`~warpfwi.config.GainParam.NONE`.
    gain_param:
        The :class:`~warpfwi.config.GainParam` value the producing INR was
        configured with.
    warp_param:
        The :class:`~warpfwi.config.WarpParam` value selecting which time-shift
        parameterization the producing INR was configured with.
    """

    tau_raw: torch.Tensor | None
    gain_raw: torch.Tensor | None
    gain_param: GainParam
    delta_raw: torch.Tensor | None = None
    tau_0_raw: torch.Tensor | None = None
    warp_param: WarpParam = WarpParam.DIRECT


class _Sine(nn.Module):
    """Sine activation (SIREN-style, with fixed ω_0 = 1 here)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def _make_activation(kind: Activation) -> nn.Module:
    if kind == "gelu":
        return nn.GELU()
    if kind == "silu":
        return nn.SiLU()
    if kind == "tanh":
        return nn.Tanh()
    if kind == "sine":
        return _Sine()
    raise ValueError(f"Unknown activation {kind!r}")


def _n_output_channels(gain_param: GainParam, warp_param: WarpParam) -> int:
    """Return the number of output channels the INR head should produce.

    Joint dispatch on (``warp_param``, ``gain_param``):

    * ``DIRECT + NONE``           -> 1 (``τ_raw``)
    * ``DIRECT + ADD/MULT``       -> 2 (``τ_raw``, ``gain_raw``)
    * ``CUMULATIVE + NONE``       -> 2 (``δ_raw``, ``τ_0_raw``)
    * ``CUMULATIVE + ADD/MULT``   -> 3 (``δ_raw``, ``τ_0_raw``, ``gain_raw``)
    """
    if gain_param not in (GainParam.NONE, GainParam.ADDITIVE, GainParam.MULTIPLICATIVE):
        raise ValueError(f"Unknown gain_param {gain_param!r}")
    if warp_param is WarpParam.DIRECT:
        warp_channels = 1
    elif warp_param is WarpParam.CUMULATIVE:
        warp_channels = 2
    else:
        raise ValueError(f"Unknown warp_param {warp_param!r}")
    gain_channels = 0 if gain_param is GainParam.NONE else 1
    return warp_channels + gain_channels


class FourierFeatureEncoder(nn.Module):
    """Fixed Fourier feature encoder.

    Given a coordinate ``(ξ, τ) ∈ [−1, 1]²`` it produces

        ``[ξ, τ, sin(2π ω_b ξ), cos(2π ω_b ξ), sin(2π ω_b τ), cos(2π ω_b τ)]_{b}``

    where ``ω_b = omega_max^{η_b}`` with ``η_b`` linearly spaced in ``[0, 1]``.

    Output feature dimension: ``2 + 4 · B``.
    """

    omega: torch.Tensor

    def __init__(self, n_bands: int, omega_max: float) -> None:
        super().__init__()
        if n_bands < 0:
            raise ValueError(f"n_bands must be >= 0, got {n_bands}")
        if n_bands == 0:
            omega = torch.empty(0)
        else:
            eta = torch.linspace(0.0, 1.0, n_bands)
            omega = torch.tensor(float(omega_max), dtype=torch.float32).pow(eta)
        self.register_buffer("omega", omega.to(torch.float32), persistent=False)
        self.n_bands = int(n_bands)
        self.out_dim = 2 + 4 * int(n_bands)

    def forward(
        self,
        coords: torch.Tensor,
        band_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of 2-D coordinates.

        Parameters
        ----------
        coords:
            Shape ``(..., 2)`` with last dim ``(ξ, τ)``.
        band_weights:
            Optional ``(B,)`` tensor of per-band weights applied to the
            ``(sin, cos)`` channels for each of the ``B`` Fourier bands. The
            raw coordinate channels ``(ξ, τ)`` are never masked. When
            ``None`` (the default), the encoder is bit-exact with its
            pre-feature behavior — no multiplication is inserted.
        """
        if coords.shape[-1] != 2:
            raise ValueError(f"coords last dim must be 2, got {coords.shape[-1]}")
        xi = coords[..., 0:1]
        ta = coords[..., 1:2]
        parts = [xi, ta]
        if self.n_bands > 0:
            two_pi_omega = 2.0 * math.pi * self.omega  # (B,)
            xi_w = xi * two_pi_omega  # (..., B)
            ta_w = ta * two_pi_omega
            sin_xi = torch.sin(xi_w)
            cos_xi = torch.cos(xi_w)
            sin_ta = torch.sin(ta_w)
            cos_ta = torch.cos(ta_w)
            if band_weights is not None:
                if band_weights.shape != (self.n_bands,):
                    raise ValueError(
                        f"band_weights must have shape ({self.n_bands},), "
                        f"got {tuple(band_weights.shape)}"
                    )
                sin_xi = sin_xi * band_weights
                cos_xi = cos_xi * band_weights
                sin_ta = sin_ta * band_weights
                cos_ta = cos_ta * band_weights
            parts.extend([sin_xi, cos_xi, sin_ta, cos_ta])
        return torch.cat(parts, dim=-1)


class TwoChannelINR(nn.Module):
    """Fourier-feature MLP producing ``τ_raw`` and optionally ``gain_raw``.

    The class name is historical: the head may have either 1 or 2 output
    channels depending on ``cfg.gain_param``. Forward returns an
    :class:`INROutput` record in all cases.

    Parameters
    ----------
    cfg:
        :class:`INRConfig`. ``cfg.gain_param`` determines the head width:
        1 channel for :attr:`~warpfwi.config.GainParam.NONE`, 2 channels
        otherwise.
    """

    band_weights: torch.Tensor | None

    def __init__(self, cfg: INRConfig) -> None:
        super().__init__()
        self.gain_param: GainParam = cfg.gain_param
        self.warp_param: WarpParam = cfg.warp_param
        self.encoder = FourierFeatureEncoder(cfg.n_fourier_bands, cfg.omega_max)
        in_dim = self.encoder.out_dim
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(cfg.depth):
            lin = nn.Linear(last, cfg.width)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(_make_activation(cfg.activation))
            last = cfg.width
        self.trunk = nn.Sequential(*layers)
        n_out = _n_output_channels(self.gain_param, self.warp_param)
        out = nn.Linear(last, n_out)
        # Zero-initialize the output head so every channel is exactly zero at
        # init. This guarantees bit-exact identity of the warp operator for
        # every gain parameterization (1 + 0 = 1, exp(0) = 1, 1 ≡ 1).
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.head = out
        # Non-parameter buffer holding per-band weights for progressive
        # spectral unmasking. None means "feature disabled, bit-exact with
        # pre-feature behavior" — the encoder skips the multiplication.
        self.register_buffer("band_weights", None, persistent=False)

    def set_band_weights(self, weights: torch.Tensor) -> None:
        """Install per-band weights for the Fourier encoder.

        Parameters
        ----------
        weights:
            ``(B,)`` tensor in ``[0, 1]`` where ``B == encoder.n_bands``.
            Stored as a non-parameter buffer; callers are expected to move
            it to the module's device first.
        """
        if weights.ndim != 1 or weights.shape[0] != self.encoder.n_bands:
            raise ValueError(
                f"band_weights must have shape ({self.encoder.n_bands},), "
                f"got {tuple(weights.shape)}"
            )
        self.band_weights = weights

    def clear_band_weights(self) -> None:
        """Remove installed band weights, restoring bit-exact default behavior."""
        self.band_weights = None

    def forward(self, coords: torch.Tensor) -> INROutput:
        """Evaluate the INR on a coordinate mesh.

        Parameters
        ----------
        coords:
            ``(R, nt, 2)`` or generally ``(..., 2)``.

        Returns
        -------
        INROutput
            Record with ``tau_raw`` and (if applicable) ``gain_raw``. Both
            tensors share the leading shape of ``coords``.
        """
        feats = self.encoder(coords, band_weights=self.band_weights)
        h = self.trunk(feats)
        raw = self.head(h)  # (..., C)
        tau_raw: torch.Tensor | None
        delta_raw: torch.Tensor | None
        tau_0_raw: torch.Tensor | None
        if self.warp_param is WarpParam.DIRECT:
            tau_raw = raw[..., 0]
            delta_raw = None
            tau_0_raw = None
            gain_idx = 1
        elif self.warp_param is WarpParam.CUMULATIVE:
            tau_raw = None
            delta_raw = raw[..., 0]
            tau_0_raw = raw[..., 1]
            gain_idx = 2
        else:
            raise ValueError(f"Unknown warp_param {self.warp_param!r}")
        gain_raw: torch.Tensor | None
        if self.gain_param is GainParam.NONE:
            gain_raw = None
        else:
            gain_raw = raw[..., gain_idx]
        return INROutput(
            tau_raw=tau_raw,
            gain_raw=gain_raw,
            gain_param=self.gain_param,
            delta_raw=delta_raw,
            tau_0_raw=tau_0_raw,
            warp_param=self.warp_param,
        )


def build_per_shot_inrs(
    n_shots: int,
    cfg: INRConfig,
    device: torch.device,
) -> dict[int, TwoChannelINR]:
    """Instantiate one INR per shot and move each to ``device``.

    Parameters
    ----------
    n_shots:
        Number of shots ``S``.
    cfg:
        INR architecture (including ``gain_param``).
    device:
        Target :class:`torch.device`.

    Returns
    -------
    dict[int, TwoChannelINR]
        Mapping from shot index ``s ∈ [0, n_shots)`` to its INR.
    """
    return {s: TwoChannelINR(cfg).to(device) for s in range(n_shots)}


__all__ = [
    "FourierFeatureEncoder",
    "TwoChannelINR",
    "INROutput",
    "build_per_shot_inrs",
    "Activation",
    "Literal",
    "np",
]
