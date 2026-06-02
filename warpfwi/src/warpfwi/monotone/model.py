"""Per-shot INR for monotone tracewise time-warp increments."""
from __future__ import annotations

import math

import torch
from torch import nn

from warpfwi.config import Activation, MonotoneModelConfig


class _Sine(nn.Module):
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


class FourierFeatureEncoder(nn.Module):
    """Fixed Fourier features for normalized ``(receiver, time)`` coordinates."""

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

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != 2:
            raise ValueError(f"coords last dim must be 2, got {coords.shape[-1]}")
        xi = coords[..., 0:1]
        ta = coords[..., 1:2]
        parts = [xi, ta]
        if self.n_bands > 0:
            w = 2.0 * math.pi * self.omega
            xi_w = xi * w
            ta_w = ta * w
            parts.extend([torch.sin(xi_w), torch.cos(xi_w), torch.sin(ta_w), torch.cos(ta_w)])
        return torch.cat(parts, dim=-1)


class MonotoneWarpINR(nn.Module):
    """INR that predicts raw positive-increment logits ``a_raw(r, n)``."""

    def __init__(self, cfg: MonotoneModelConfig) -> None:
        super().__init__()
        self.encoder = FourierFeatureEncoder(cfg.n_fourier_bands, cfg.omega_max)
        layers: list[nn.Module] = []
        last = self.encoder.out_dim
        for _ in range(cfg.depth):
            lin = nn.Linear(last, cfg.width)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(_make_activation(cfg.activation))
            last = cfg.width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(last, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Return ``a_raw`` on an ``(R, nt - 1, 2)`` coordinate mesh."""
        h = self.trunk(self.encoder(coords))
        return self.head(h).squeeze(-1)


def normalized_increment_grid(
    n_rec: int,
    nt: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return normalized coordinates for the ``nt - 1`` time increments."""
    if nt < 2:
        raise ValueError(f"nt must be at least 2, got {nt}")
    xi_r = (2.0 * torch.arange(n_rec, device=device, dtype=dtype) + 1.0) / n_rec - 1.0
    n_inc = nt - 1
    tau_n = (2.0 * torch.arange(n_inc, device=device, dtype=dtype) + 1.0) / n_inc - 1.0
    mesh_r, mesh_t = torch.meshgrid(xi_r, tau_n, indexing="ij")
    return torch.stack([mesh_r, mesh_t], dim=-1).contiguous()


def build_per_shot_monotone_inrs(
    n_shots: int,
    cfg: MonotoneModelConfig,
    device: torch.device,
) -> dict[int, MonotoneWarpINR]:
    """Instantiate one monotone warp INR per shot."""
    return {s: MonotoneWarpINR(cfg).to(device) for s in range(n_shots)}


__all__ = [
    "FourierFeatureEncoder",
    "MonotoneWarpINR",
    "build_per_shot_monotone_inrs",
    "normalized_increment_grid",
]
