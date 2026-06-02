"""Data misfit and warp regularizer.

    R(τ, gain_raw) = ‖τ‖² + α · ‖∂_t τ‖² + β · ‖gain_raw‖²  (+ optional ‖∂_r τ‖²)

All norms are mean-squared over the ``(R, nt)`` grid per shot, then averaged
over shots in a minibatch. The smoothness penalty ``‖∂_t τ‖²`` is a first-order
finite difference along the time axis.

Regularizer dispatch on :class:`~warpfwi.config.GainParam`
---------------------------------------------------------

The ``β`` term is always computed on the **raw** INR gain channel:

* :attr:`~warpfwi.config.GainParam.NONE` — no gain channel exists; the β term
  is dropped. ``gain_raw`` must be ``None``.
* :attr:`~warpfwi.config.GainParam.ADDITIVE` — ``β · mean(gain_raw²)``. This
  leaves the "delete the synthetic" state ``gain_raw = −1`` at finite cost,
  which is the amplitude-collapse failure mode.
* :attr:`~warpfwi.config.GainParam.MULTIPLICATIVE` — ``β · mean(gain_raw²)``
  on the raw output. Collapse requires ``gain_raw → −∞``, which pays infinite
  regularizer cost and is thus structurally ruled out. Penalizing the gain
  factor ``‖exp(gain_raw) − 1‖²`` instead would *not* diverge at collapse.
"""
from __future__ import annotations

import torch
from torch import nn

from .config import GainParam


class Misfit(nn.Module):
    """Base class for differentiable data misfit terms."""

    def forward(self, pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _check_pred_obs_shape(pred: torch.Tensor, obs: torch.Tensor) -> None:
    if pred.shape != obs.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} must equal obs shape {tuple(obs.shape)}"
        )


class L2Misfit(Misfit):
    """Mean-squared residual data misfit."""

    def forward(self, pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        _check_pred_obs_shape(pred, obs)
        return torch.mean((pred - obs) ** 2)


def hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """Differentiable Hilbert transform along the last dimension.

    The last dimension is interpreted as time. Leading dimensions may contain
    shots, receivers, or any other batch axes.
    """
    if x.ndim < 1:
        raise ValueError("hilbert_transform expects at least one dimension")
    if not torch.is_floating_point(x):
        raise TypeError(f"hilbert_transform expects a real floating tensor, got {x.dtype}")
    nt = int(x.shape[-1])
    if nt < 1:
        raise ValueError("time axis must contain at least one sample")

    spectrum = torch.fft.fft(x, dim=-1)
    h = torch.zeros(nt, dtype=x.dtype, device=x.device)
    if nt % 2 == 0:
        h[0] = 1.0
        h[nt // 2] = 1.0
        h[1 : nt // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (nt + 1) // 2] = 2.0
    h_shape = (1,) * (x.ndim - 1) + (nt,)
    analytic = torch.fft.ifft(spectrum * h.view(h_shape), dim=-1)
    return analytic.imag


def analytic_envelope(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return ``sqrt(x^2 + H[x]^2 + eps)`` along the last/time axis."""
    if eps < 0.0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    hx = hilbert_transform(x)
    return torch.sqrt(x ** 2 + hx ** 2 + float(eps))


class EnvelopeL2Misfit(Misfit):
    """Mean-squared residual between analytic-signal envelopes."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        _check_pred_obs_shape(pred, obs)
        pred_env = analytic_envelope(pred, eps=self.eps)
        obs_env = analytic_envelope(obs, eps=self.eps)
        return torch.mean((pred_env - obs_env) ** 2)


def build_misfit(name: str, **kwargs: object) -> Misfit:
    """Construct a data misfit by name.

    Accepted names are ``"l2"`` and ``"envelope_l2"``.
    """
    normalized = str(name).strip().lower()
    eps = kwargs.pop("eps", None)
    envelope_eps = kwargs.pop("envelope_eps", None)
    if eps is not None and envelope_eps is not None:
        raise ValueError("pass only one of eps or envelope_eps")
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise ValueError(f"unknown misfit option(s): {unknown}")

    if normalized == "l2":
        return L2Misfit()
    if normalized == "envelope_l2":
        value = 1e-8 if eps is None and envelope_eps is None else eps
        if value is None:
            value = envelope_eps
        return EnvelopeL2Misfit(eps=float(value))
    raise ValueError(
        f"unknown misfit {name!r}; expected one of 'l2', 'envelope_l2'"
    )


def data_misfit(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error data misfit.

    Parameters
    ----------
    pred:
        Predicted gathers ``(B, R, nt)`` (warped synthetics).
    obs:
        Observed gathers ``(B, R, nt)``.

    Returns
    -------
    torch.Tensor
        Scalar mean-squared misfit, averaged over all ``(b, r, n)``.
    """
    return L2Misfit()(pred, obs)


def warp_regularizer(
    tau: torch.Tensor,
    gain_raw: torch.Tensor | None,
    gain_param: GainParam,
    alpha: float,
    beta: float,
    offset_smooth: bool = False,
    delta_bounded: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``R(τ, gain_raw) = mean(τ²) + α · mean((∂_t τ)²) + β · mean(gain_raw²)``.

    The ``β`` term is dispatched on ``gain_param``: for
    :attr:`~warpfwi.config.GainParam.NONE` it is dropped entirely; for the
    other two it is computed on the **raw** gain channel.

    Parameters
    ----------
    tau:
        Time-shift field ``(B, R, nt)`` or ``(R, nt)``.
    gain_raw:
        Raw gain channel, same shape as ``tau``, or ``None`` when
        ``gain_param == GainParam.NONE``.
    gain_param:
        :class:`~warpfwi.config.GainParam` selecting which gain
        parameterization the caller is using. Determines whether the β term
        is included and how ``gain_raw`` is interpreted (semantically — the
        penalty is always on the raw channel).
    alpha:
        Weight on ``‖∂_t τ‖²``.
    beta:
        Weight on ``‖gain_raw‖²``. Ignored for
        :attr:`~warpfwi.config.GainParam.NONE`.
    offset_smooth:
        If ``True`` also include the offset-smoothness term with weight
        ``alpha``. Under the :attr:`~warpfwi.config.WarpParam.DIRECT`
        parameterization this is ``‖∂_r τ‖²``. Under
        :attr:`~warpfwi.config.WarpParam.CUMULATIVE` (signalled by passing a
        non-``None`` ``delta_bounded``) the term is redirected to
        ``‖δ_bounded‖²`` — they are equal up to discretization and
        penalizing both would double-count the tanh nonlinearity.
    delta_bounded:
        Bounded offset-derivative field ``δ_max · tanh(δ_raw)``, same shape
        as ``tau``. When provided and ``offset_smooth`` is ``True``, replaces
        the finite-difference ``∂_r τ`` in the offset-smoothness term.

    Returns
    -------
    torch.Tensor
        Scalar regularizer value (averaged across all leading dims).
    """
    if tau.ndim not in (2, 3):
        raise ValueError(f"tau must be 2D or 3D, got shape {tuple(tau.shape)}")

    if gain_param is GainParam.NONE:
        if gain_raw is not None:
            raise ValueError(
                "gain_raw must be None when gain_param is GainParam.NONE"
            )
    else:
        if gain_raw is None:
            raise ValueError(
                f"gain_raw must be a tensor when gain_param is {gain_param!r}"
            )
        if gain_raw.shape != tau.shape:
            raise ValueError(
                f"tau/gain_raw shape mismatch: {tuple(tau.shape)} vs "
                f"{tuple(gain_raw.shape)}"
            )

    if delta_bounded is not None and delta_bounded.shape != tau.shape:
        raise ValueError(
            f"tau/delta_bounded shape mismatch: {tuple(tau.shape)} vs "
            f"{tuple(delta_bounded.shape)}"
        )

    term_tau = torch.mean(tau ** 2)

    # ∂_t τ via first-order forward difference along the last axis.
    dtau_dt = tau[..., 1:] - tau[..., :-1]
    term_smooth_t = torch.mean(dtau_dt ** 2)

    total = term_tau + alpha * term_smooth_t
    if gain_param is not GainParam.NONE:
        assert gain_raw is not None  # narrowed by the branch above
        total = total + beta * torch.mean(gain_raw ** 2)
    if offset_smooth:
        if delta_bounded is not None:
            # CUMULATIVE path: ‖∂_r τ‖² ≡ ‖δ_bounded‖² up to discretization.
            total = total + alpha * torch.mean(delta_bounded ** 2)
        else:
            dtau_dr = tau[..., 1:, :] - tau[..., :-1, :]
            total = total + alpha * torch.mean(dtau_dr ** 2)
    return total


def combined_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    tau: torch.Tensor,
    gain_raw: torch.Tensor | None,
    gain_param: GainParam,
    alpha: float,
    beta: float,
    lam: float,
    offset_smooth: bool = False,
    delta_bounded: torch.Tensor | None = None,
    misfit: Misfit | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return ``data + λ · R`` along with a dict decomposition for logging.

    Parameters
    ----------
    pred, obs, tau, gain_raw, gain_param, alpha, beta, offset_smooth:
        See :func:`data_misfit` and :func:`warp_regularizer`.
    lam:
        λ weight applied to the regularizer.
    misfit:
        Optional data-misfit module. ``None`` preserves the original L2
        residual path exactly.

    Returns
    -------
    (total, terms):
        ``total`` is a scalar loss ready for ``.backward()``. ``terms`` maps
        ``"data"``, ``"reg"``, ``"tau_sq"``, ``"gain_raw_sq"``,
        ``"dtau_dt_sq"`` to **detached** scalar tensors useful for
        diagnostics. ``"gain_raw_sq"`` is ``0.0`` when
        ``gain_param == GainParam.NONE``.
    """
    data = data_misfit(pred, obs) if misfit is None else misfit(pred, obs)
    reg = warp_regularizer(
        tau,
        gain_raw,
        gain_param=gain_param,
        alpha=alpha,
        beta=beta,
        offset_smooth=offset_smooth,
        delta_bounded=delta_bounded,
    )
    total = data + lam * reg
    with torch.no_grad():
        dtau_dt = tau[..., 1:] - tau[..., :-1]
        if gain_raw is None:
            gain_raw_sq = torch.zeros((), dtype=tau.dtype, device=tau.device)
        else:
            gain_raw_sq = torch.mean(gain_raw ** 2).detach()
        terms = {
            "data": data.detach(),
            "reg": reg.detach(),
            "tau_sq": torch.mean(tau ** 2).detach(),
            "gain_raw_sq": gain_raw_sq,
            "dtau_dt_sq": torch.mean(dtau_dt ** 2).detach(),
        }
    return total, terms


__all__ = [
    "Misfit",
    "L2Misfit",
    "EnvelopeL2Misfit",
    "analytic_envelope",
    "hilbert_transform",
    "build_misfit",
    "data_misfit",
    "warp_regularizer",
    "combined_loss",
]
