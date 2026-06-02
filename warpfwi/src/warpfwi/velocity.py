"""Bounded velocity parameterization via logits and sigmoid.

``v(φ) = v_min + (v_max − v_min) · sigmoid(φ)``

The module holds unconstrained logits ``φ`` as its single trainable parameter
and exposes :meth:`BoundedVelocity.v` which returns ``v`` in m/s. Factory
:func:`BoundedVelocity.from_velocity` inverts the sigmoid to seed ``φ`` from
an initial velocity ``v_0``.
"""
from __future__ import annotations

import torch
from torch import nn


def _inverse_sigmoid(p: torch.Tensor) -> torch.Tensor:
    """Stable logit: ``log(p / (1 - p))``."""
    return torch.log(p) - torch.log1p(-p)


@torch.no_grad()
def tensor_bound_diagnostics(
    v: torch.Tensor,
    v_min: float,
    v_max: float,
    near_bound_fraction: float = 0.01,
) -> dict[str, float]:
    """Return bound-occupancy diagnostics for a raw velocity tensor."""
    if near_bound_fraction <= 0.0:
        raise ValueError(
            f"near_bound_fraction must be positive, got {near_bound_fraction}"
        )
    span = float(v_max - v_min)
    margin = span * near_bound_fraction
    v_detached = v.detach().to(torch.float32)
    return {
        "v_min": float(v_detached.min().item()),
        "v_max": float(v_detached.max().item()),
        "frac_near_vmin": float(
            (v_detached <= v_min + margin).to(torch.float32).mean().item()
        ),
        "frac_near_vmax": float(
            (v_detached >= v_max - margin).to(torch.float32).mean().item()
        ),
        "frac_exact_vmin": float((v_detached == v_min).to(torch.float32).mean().item()),
        "frac_exact_vmax": float((v_detached == v_max).to(torch.float32).mean().item()),
    }


class BoundedVelocity(nn.Module):
    """Bounded velocity parameterization.

    Attributes
    ----------
    phi:
        Unconstrained logits, trainable :class:`torch.nn.Parameter` of shape
        ``(nz, nx)``.
    v_min, v_max:
        Bounds on the resulting velocity [m/s].
    """

    phi: nn.Parameter

    def __init__(self, phi_init: torch.Tensor, v_min: float, v_max: float) -> None:
        """Initialize from a logit tensor.

        Parameters
        ----------
        phi_init:
            ``(nz, nx)`` float32 tensor of initial logits.
        v_min, v_max:
            Bounds on ``v`` in m/s. Must satisfy ``v_min < v_max``.
        """
        super().__init__()
        if v_min >= v_max:
            raise ValueError(f"v_min ({v_min}) must be < v_max ({v_max})")
        if phi_init.ndim != 2:
            raise ValueError(f"phi_init must be 2D, got shape {tuple(phi_init.shape)}")
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.phi = nn.Parameter(phi_init.detach().to(torch.float32).clone())

    @classmethod
    def from_velocity(
        cls,
        v_init: torch.Tensor,
        v_min: float,
        v_max: float,
        eps: float = 1e-3,
    ) -> "BoundedVelocity":
        """Construct from an initial velocity ``v_0`` by inverting the sigmoid.

        Parameters
        ----------
        v_init:
            ``(nz, nx)`` float tensor in m/s on any device.
        v_min, v_max:
            Same bounds that ``v()`` will respect.
        eps:
            Relative clamp applied to ``(v_init - v_min) / (v_max - v_min)``
            to keep it strictly inside ``(0, 1)`` so the logit is finite.

        Returns
        -------
        BoundedVelocity
            Module with ``phi`` on the device of ``v_init``.
        """
        p = (v_init - v_min) / (v_max - v_min)
        p = p.clamp(min=eps, max=1.0 - eps)
        phi = _inverse_sigmoid(p)
        out = cls(phi, v_min, v_max)
        out.register_buffer(
            "v_init_reference",
            v_init.detach().to(torch.float32).clone(),
            persistent=False,
        )
        out.register_buffer(
            "phi_init_reference",
            phi.detach().to(torch.float32).clone(),
            persistent=False,
        )
        return out

    def v(self) -> torch.Tensor:
        """Return the bounded velocity ``v(φ)`` in m/s."""
        return self.v_min + (self.v_max - self.v_min) * torch.sigmoid(self.phi)

    def forward(self) -> torch.Tensor:  # pragma: no cover - alias
        """Alias for :meth:`v` to allow ``model()``-style calls."""
        return self.v()

    @torch.no_grad()
    def clamp_(self, v_low: float | None = None, v_high: float | None = None) -> None:
        """Clamp ``φ`` so that ``v(φ) ∈ [v_low, v_high]``.

        Useful after an Adam step that may have pushed ``φ`` into a region
        where the sigmoid is numerically saturated and gradients vanish.
        Defaults to ``(v_min + 1, v_max - 1)`` to stay strictly inside.
        """
        lo = float(v_low if v_low is not None else self.v_min + 1.0)
        hi = float(v_high if v_high is not None else self.v_max - 1.0)
        span = self.v_max - self.v_min
        p_lo = (lo - self.v_min) / span
        p_hi = (hi - self.v_min) / span
        phi_lo = float(torch.log(torch.tensor(p_lo)) - torch.log1p(torch.tensor(-p_lo)))
        phi_hi = float(torch.log(torch.tensor(p_hi)) - torch.log1p(torch.tensor(-p_hi)))
        self.phi.data.clamp_(min=phi_lo, max=phi_hi)


@torch.no_grad()
def velocity_diagnostics(
    v_param: BoundedVelocity,
    near_bound_fraction: float = 0.01,
) -> dict[str, float]:
    """Return scalar diagnostics for the bounded velocity parameterization.

    Parameters
    ----------
    v_param:
        Velocity parameterization to summarize.
    near_bound_fraction:
        Relative fraction of the configured velocity span used to define
        "near the lower/upper bound" occupancy. ``0.01`` means within 1% of
        ``(v_max - v_min)`` from either bound.
    """
    if near_bound_fraction <= 0.0:
        raise ValueError(
            f"near_bound_fraction must be positive, got {near_bound_fraction}"
        )
    v = v_param.v().detach()
    phi = v_param.phi.detach()
    span = v_param.v_max - v_param.v_min
    margin = span * near_bound_fraction
    return {
        "phi_min": float(phi.min().item()),
        "phi_max": float(phi.max().item()),
        "v_min": float(v.min().item()),
        "v_max": float(v.max().item()),
        "frac_near_vmin": float((v <= v_param.v_min + margin).to(torch.float32).mean().item()),
        "frac_near_vmax": float((v >= v_param.v_max - margin).to(torch.float32).mean().item()),
    }


__all__ = ["BoundedVelocity", "tensor_bound_diagnostics", "velocity_diagnostics"]
