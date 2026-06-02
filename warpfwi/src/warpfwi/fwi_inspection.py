"""Opt-in raw/classical FWI inspection experiments.

This module keeps the production classical FWI path untouched while exposing a
more diagnostic runner for notebooks that need raw gradients, processed
gradients, update snapshots, muting, simple preconditioning, and alternative
velocity parameterizations.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Callable, Literal, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from .acquisition import Acquisition
from .config import FWIPreprocessingConfig, OptimConfig
from .diagnostics import snr_db, ssim
from .losses import data_misfit
from .modeling import iter_minibatches, simulate_batch
from .preprocessing import (
    bandpass_dataset,
    make_boundary_freeze_mask,
    make_boundary_taper,
    offset_mask,
    time_mute,
    validate_bandpass,
)
from .velocity import BoundedVelocity

ParameterizationName = Literal["bounded", "log_velocity"]
SmoothingKind = Literal["none", "gaussian", "box"]
ClipKind = Literal["none", "value", "norm", "percentile_soft"]
PreconditionerKind = Literal["none", "depth", "rms", "diagonal_energy"]
BoundaryMode = Literal["all", "top"]


class VelocityParameterization(Protocol):
    """Small interface shared by inspection velocity parameterizations."""

    parameter: nn.Parameter

    def v(self) -> torch.Tensor:
        """Return physical velocity in m/s."""


class BoundedInspectionVelocity(BoundedVelocity):
    """Bounded velocity with a common ``parameter`` alias for inspection code."""

    @property
    def parameter(self) -> nn.Parameter:
        """Optimization variable, equal to bounded-velocity ``phi``."""
        return self.phi


class LogVelocity(nn.Module):
    """Positive velocity parameterization ``v = exp(log_v)`` with optional clamp."""

    parameter: nn.Parameter

    def __init__(
        self,
        log_v_init: torch.Tensor,
        *,
        v_min: float | None = None,
        v_max: float | None = None,
    ) -> None:
        super().__init__()
        if log_v_init.ndim != 2:
            raise ValueError(f"log_v_init must be 2D, got {tuple(log_v_init.shape)}")
        if v_min is not None and v_max is not None and v_min >= v_max:
            raise ValueError(f"v_min ({v_min}) must be < v_max ({v_max})")
        self.parameter = nn.Parameter(log_v_init.detach().to(torch.float32).clone())
        self.v_min = None if v_min is None else float(v_min)
        self.v_max = None if v_max is None else float(v_max)

    @classmethod
    def from_velocity(
        cls,
        v_init: torch.Tensor,
        *,
        v_min: float | None = None,
        v_max: float | None = None,
    ) -> "LogVelocity":
        """Construct from physical velocity in m/s."""
        if torch.any(v_init <= 0.0):
            raise ValueError("log-velocity parameterization requires positive v_init")
        return cls(torch.log(v_init), v_min=v_min, v_max=v_max)

    def v(self) -> torch.Tensor:
        """Return physical velocity in m/s."""
        v = torch.exp(self.parameter)
        if self.v_min is not None or self.v_max is not None:
            v = torch.clamp(v, min=self.v_min, max=self.v_max)
        return v

    def forward(self) -> torch.Tensor:  # pragma: no cover - alias
        return self.v()


def make_velocity_parameterization(
    v_init: torch.Tensor,
    kind: ParameterizationName,
    *,
    v_min: float,
    v_max: float,
) -> VelocityParameterization:
    """Build a velocity parameterization for raw FWI inspection."""
    if kind == "bounded":
        return BoundedInspectionVelocity.from_velocity(v_init, v_min, v_max)
    if kind == "log_velocity":
        return LogVelocity.from_velocity(v_init, v_min=v_min, v_max=v_max)
    raise ValueError(f"unknown parameterization {kind!r}")


@dataclass
class BoundaryUpdateConfig:
    """Boundary/PML update suppression applied to the optimization gradient."""

    enabled: bool = False
    taper_type: Literal["cosine", "hann"] = "cosine"
    width_cells: int = 0
    mode: BoundaryMode = "all"
    freeze: bool = False


@dataclass
class GradientProcessingConfig:
    """Composable opt-in processing for raw optimization gradients."""

    smoothing: SmoothingKind = "none"
    sigma: float = 0.0
    sigma_z: float | None = None
    sigma_x: float | None = None
    box_radius: int = 0
    clip: ClipKind = "none"
    clip_value: float | None = None
    clip_norm: float | None = None
    percentile: float = 99.0
    normalize: bool = False
    scale: float = 1.0
    laplacian_smoothness: float = 0.0


@dataclass
class PreconditionerConfig:
    """Simple diagonal-like raw-gradient preconditioners."""

    kind: PreconditionerKind = "none"
    depth_power: float = 1.0
    depth_floor: float = 0.25
    rms_eps: float = 1e-8
    diagonal_decay: float = 0.95


@dataclass
class MuteConfig:
    """Data-domain mute and offset-selection controls."""

    early_time_enabled: bool = False
    mute_until_s: float = 0.0
    taper_s: float = 0.0
    offset_mode: Literal["all", "near", "far"] = "all"
    max_offset_m: float | None = None
    min_offset_m: float | None = None


@dataclass
class BandpassStage:
    """One optional frequency band for a fixed number of iterations."""

    fmin_hz: float
    fmax_hz: float
    n_iter: int


@dataclass
class ClassicalInspectionConfig:
    """Top-level config for the opt-in raw FWI inspection runner."""

    n_iter: int = 3
    optim: OptimConfig = field(default_factory=OptimConfig)
    parameterization: ParameterizationName = "bounded"
    boundary: BoundaryUpdateConfig = field(default_factory=BoundaryUpdateConfig)
    gradient: GradientProcessingConfig = field(default_factory=GradientProcessingConfig)
    preconditioner: PreconditionerConfig = field(default_factory=PreconditionerConfig)
    mute: MuteConfig = field(default_factory=MuteConfig)
    preprocess: FWIPreprocessingConfig = field(default_factory=FWIPreprocessingConfig)
    bandpass_schedule: list[BandpassStage] = field(default_factory=list)
    cache_gradients: bool = True
    cache_snapshots: bool = True
    snapshot_every: int = 1
    inspect_first_batch_only: bool = False
    seed: int = 0


@dataclass
class ExperimentSpec:
    """Named raw-FWI experiment with its own inspection config."""

    name: str
    cfg: ClassicalInspectionConfig


def _gaussian_kernel1d(sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive for Gaussian smoothing, got {sigma}")
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / float(sigma)).square())
    return kernel / kernel.sum()


def _smooth_gradient(
    grad: torch.Tensor,
    cfg: GradientProcessingConfig,
) -> torch.Tensor:
    if cfg.smoothing == "none":
        return grad
    x = grad.unsqueeze(0).unsqueeze(0)
    if cfg.smoothing == "box":
        radius = int(cfg.box_radius)
        if radius < 1:
            raise ValueError("box_radius must be >= 1 when box smoothing is enabled")
        k = 2 * radius + 1
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius).squeeze(0).squeeze(0)
    if cfg.smoothing == "gaussian":
        sigma_z = float(cfg.sigma_z if cfg.sigma_z is not None else cfg.sigma)
        sigma_x = float(cfg.sigma_x if cfg.sigma_x is not None else cfg.sigma)
        kz = _gaussian_kernel1d(sigma_z, device=grad.device, dtype=grad.dtype)
        kx = _gaussian_kernel1d(sigma_x, device=grad.device, dtype=grad.dtype)
        pad_z = kz.numel() // 2
        pad_x = kx.numel() // 2
        z_kernel = kz.view(1, 1, -1, 1)
        x_kernel = kx.view(1, 1, 1, -1)
        y = F.conv2d(F.pad(x, (0, 0, pad_z, pad_z), mode="replicate"), z_kernel)
        y = F.conv2d(F.pad(y, (pad_x, pad_x, 0, 0), mode="replicate"), x_kernel)
        return y.squeeze(0).squeeze(0)
    raise ValueError(f"unknown smoothing kind {cfg.smoothing!r}")


def _laplacian(v: torch.Tensor) -> torch.Tensor:
    x = v.unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=v.device,
        dtype=v.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel).squeeze(0).squeeze(0)


def _apply_gradient_processing(
    grad: torch.Tensor,
    cfg: GradientProcessingConfig,
    *,
    current_velocity: torch.Tensor,
) -> torch.Tensor:
    out = grad
    if cfg.laplacian_smoothness != 0.0:
        out = out + float(cfg.laplacian_smoothness) * _laplacian(current_velocity)
    out = _smooth_gradient(out, cfg)
    if cfg.clip == "value":
        if cfg.clip_value is None or cfg.clip_value <= 0.0:
            raise ValueError("clip_value must be positive for value clipping")
        out = out.clamp(-float(cfg.clip_value), float(cfg.clip_value))
    elif cfg.clip == "norm":
        if cfg.clip_norm is None or cfg.clip_norm <= 0.0:
            raise ValueError("clip_norm must be positive for norm clipping")
        norm = out.norm()
        if float(norm.item()) > float(cfg.clip_norm):
            out = out * (float(cfg.clip_norm) / (norm + 1e-12))
    elif cfg.clip == "percentile_soft":
        if not (0.0 < cfg.percentile <= 100.0):
            raise ValueError(f"percentile must be in (0, 100], got {cfg.percentile}")
        limit = torch.quantile(out.detach().abs().flatten(), float(cfg.percentile) / 100.0)
        if float(limit.item()) > 0.0:
            out = limit * torch.tanh(out / limit)
    elif cfg.clip != "none":
        raise ValueError(f"unknown clip kind {cfg.clip!r}")
    if cfg.normalize:
        rms = torch.sqrt(out.square().mean()).clamp_min(1e-12)
        out = out / rms
    return out * float(cfg.scale)


def make_boundary_update_mask(
    shape: tuple[int, int],
    cfg: BoundaryUpdateConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build the configured taper/freezing mask, or ``None`` when disabled."""
    if not cfg.enabled:
        return None
    if cfg.width_cells < 0:
        raise ValueError(f"boundary width_cells must be non-negative, got {cfg.width_cells}")
    if cfg.freeze:
        return make_boundary_freeze_mask(
            shape, cfg.width_cells, cfg.mode, device=device, dtype=dtype
        )
    return make_boundary_taper(
        shape, cfg.width_cells, cfg.taper_type, cfg.mode, device=device, dtype=dtype
    )


def _depth_weights(
    shape: tuple[int, int],
    cfg: PreconditionerConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    nz, nx = shape
    z = torch.linspace(0.0, 1.0, nz, device=device, dtype=dtype)
    weights = float(cfg.depth_floor) + (1.0 - float(cfg.depth_floor)) * z.pow(
        float(cfg.depth_power)
    )
    return weights[:, None].expand(nz, nx)


def _apply_preconditioner(
    grad: torch.Tensor,
    cfg: PreconditionerConfig,
    *,
    energy: torch.Tensor | None,
) -> torch.Tensor:
    if cfg.kind == "none":
        return grad
    if cfg.kind == "depth":
        return grad * _depth_weights(
            tuple(grad.shape), cfg, device=grad.device, dtype=grad.dtype
        )
    if cfg.kind == "rms":
        return grad / torch.sqrt(grad.square().mean()).clamp_min(float(cfg.rms_eps))
    if cfg.kind == "diagonal_energy":
        if energy is None:
            raise ValueError("diagonal_energy preconditioner requires an energy tensor")
        return grad / torch.sqrt(energy + float(cfg.rms_eps))
    raise ValueError(f"unknown preconditioner kind {cfg.kind!r}")


def _offsets_for_batch(acq: Acquisition, idx: torch.Tensor) -> torch.Tensor:
    src_x = acq.source_locations.index_select(0, idx)[..., 1].to(torch.float32)
    rec_x = acq.receiver_locations.index_select(0, idx)[..., 1].to(torch.float32)
    return (rec_x - src_x) * float(acq.dx)


def apply_data_domain_controls(
    data: torch.Tensor,
    acq: Acquisition,
    idx: torch.Tensor,
    cfg: MuteConfig,
) -> torch.Tensor:
    """Apply configured early-time and offset masks to a batch of shot gathers."""
    out = data
    if cfg.early_time_enabled:
        out = time_mute(out, acq.dt, cfg.mute_until_s, cfg.taper_s)
    if cfg.offset_mode != "all":
        offsets = _offsets_for_batch(acq, idx)
        mask = offset_mask(
            offsets,
            mode=cfg.offset_mode,
            max_offset=cfg.max_offset_m,
            min_offset=cfg.min_offset_m,
            dtype=out.dtype,
        ).to(out.device)
        out = out * mask[..., None]
    return out


def _band_for_iteration(
    cfg: ClassicalInspectionConfig,
    iteration: int,
) -> tuple[float, float, int] | None:
    if cfg.bandpass_schedule:
        acc = 0
        for stage in cfg.bandpass_schedule:
            if stage.n_iter < 1:
                raise ValueError("each BandpassStage.n_iter must be >= 1")
            acc += int(stage.n_iter)
            if iteration < acc:
                return (float(stage.fmin_hz), float(stage.fmax_hz), int(cfg.preprocess.bandpass_order))
        last = cfg.bandpass_schedule[-1]
        return (float(last.fmin_hz), float(last.fmax_hz), int(cfg.preprocess.bandpass_order))
    if cfg.preprocess.bandpass_enabled:
        return (
            float(cfg.preprocess.bandpass_fmin_hz),
            float(cfg.preprocess.bandpass_fmax_hz),
            int(cfg.preprocess.bandpass_order),
        )
    return None


def _maybe_bandpass(
    data: torch.Tensor,
    acq: Acquisition,
    cfg: ClassicalInspectionConfig,
    iteration: int,
) -> torch.Tensor:
    band = _band_for_iteration(cfg, iteration)
    if band is None:
        return data
    fmin, fmax, order = band
    validate_bandpass(acq.dt, fmin, fmax, order, data.shape[-1])
    return bandpass_dataset(data, acq.dt, fmin, fmax, order=order)


def gradient_depth_profile(grad: torch.Tensor) -> dict[str, torch.Tensor | float]:
    """Return shallow/deep average absolute gradient magnitudes."""
    if grad.ndim != 2:
        raise ValueError(f"grad must be 2-D, got {tuple(grad.shape)}")
    abs_grad = grad.detach().abs().to(torch.float32)
    nz = abs_grad.shape[0]
    split = max(1, nz // 3)
    shallow = abs_grad[:split].mean()
    deep = abs_grad[-split:].mean()
    return {
        "profile": abs_grad.mean(dim=1).cpu(),
        "shallow_mean_abs": float(shallow.item()),
        "deep_mean_abs": float(deep.item()),
        "deep_over_shallow": float((deep / shallow.clamp_min(1e-12)).item()),
    }


def summarize_run(
    name: str,
    result: dict[str, Any],
    *,
    v_true: torch.Tensor | None = None,
) -> dict[str, float | str]:
    """Create one compact metrics row for an inspection result."""
    history = result["history"]
    losses = history.get("loss", [])
    grad_norms = history.get("gradient_norm", [])
    update_norms = history.get("update_norm", [])
    v_final = result["v"]
    row: dict[str, float | str] = {
        "name": name,
        "initial_loss": float(losses[0]) if losses else float("nan"),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "loss_reduction_ratio": (
            float(losses[-1]) / float(losses[0]) if losses and losses[0] != 0.0 else float("nan")
        ),
        "v_min": float(v_final.min().item()),
        "v_max": float(v_final.max().item()),
        "grad_norm_last": float(grad_norms[-1]) if grad_norms else float("nan"),
        "update_norm_last": float(update_norms[-1]) if update_norms else float("nan"),
        "runtime_s": float(result.get("runtime_s", float("nan"))),
    }
    if v_true is not None:
        row["snr_db"] = snr_db(v_final.detach(), v_true.detach())
        row["ssim"] = ssim(v_final.detach(), v_true.detach())
    return row


def summary_rows_to_markdown(rows: list[dict[str, float | str]]) -> str:
    """Format experiment summary rows as a compact Markdown table."""
    if not rows:
        return "_No experiment rows._"
    columns = [
        "name",
        "initial_loss",
        "final_loss",
        "loss_reduction_ratio",
        "v_min",
        "v_max",
        "grad_norm_last",
        "update_norm_last",
        "snr_db",
        "ssim",
        "runtime_s",
    ]
    present = [col for col in columns if any(col in row for row in rows)]

    def fmt(value: float | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if abs(value) >= 1e4 or (abs(value) < 1e-3 and value != 0.0):
            return f"{value:.3e}"
        return f"{value:.4g}"

    lines = [
        "| " + " | ".join(present) + " |",
        "| " + " | ".join(["---"] * len(present)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in present) + " |")
    return "\n".join(lines)


def run_classical_fwi_inspection(
    v_init: torch.Tensor,
    acq: Acquisition,
    d_obs: torch.Tensor,
    cfg: ClassicalInspectionConfig,
    *,
    device: torch.device,
    v_min: float,
    v_max: float,
    v_true: torch.Tensor | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run opt-in diagnostic raw FWI and return rich inspection outputs."""
    if v_init.device != device:
        raise RuntimeError(f"v_init on {v_init.device}, expected {device}")
    if d_obs.device != device:
        raise RuntimeError(f"d_obs on {d_obs.device}, expected {device}")
    if cfg.n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {cfg.n_iter}")
    torch.manual_seed(cfg.seed)
    param = make_velocity_parameterization(
        v_init,
        cfg.parameterization,
        v_min=v_min,
        v_max=v_max,
    )
    if isinstance(param, nn.Module):
        param = param.to(device)  # type: ignore[assignment]
    opt = torch.optim.Adam([param.parameter], lr=float(cfg.optim.lr_v))
    boundary_mask = make_boundary_update_mask(
        tuple(param.parameter.shape),
        cfg.boundary,
        device=device,
        dtype=param.parameter.dtype,
    )
    energy: torch.Tensor | None = None
    if cfg.preconditioner.kind == "diagonal_energy":
        energy = torch.zeros_like(param.parameter.detach())

    history: dict[str, list[float]] = {
        "loss": [],
        "gradient_norm": [],
        "raw_gradient_norm": [],
        "update_norm": [],
        "v_min": [],
        "v_max": [],
    }
    diagnostics: dict[str, Any] = {
        "raw_gradients": [],
        "processed_gradients": [],
        "updates": [],
        "synthetic": [],
        "residual": [],
        "loss_decomposition": [],
        "gradient_depth_profiles": [],
    }
    snapshots: list[torch.Tensor] = [param.v().detach().cpu().clone()] if cfg.cache_snapshots else []
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    start_time = perf_counter()

    for k in range(int(cfg.n_iter)):
        epoch_loss = 0.0
        epoch_n = 0
        last_update_norm = 0.0
        last_grad_norm = 0.0
        last_raw_grad_norm = 0.0
        for batch_i, idx in enumerate(
            iter_minibatches(acq.n_shots, int(cfg.optim.batch_size), generator=gen, device=device)
        ):
            opt.zero_grad(set_to_none=True)
            before = param.parameter.detach().clone()
            pred = simulate_batch(param.v(), acq, idx, device=device)
            obs_batch = d_obs.index_select(0, idx)
            pred = _maybe_bandpass(pred, acq, cfg, k)
            obs_batch = _maybe_bandpass(obs_batch, acq, cfg, k)
            pred = apply_data_domain_controls(pred, acq, idx, cfg.mute)
            obs_batch = apply_data_domain_controls(obs_batch, acq, idx, cfg.mute)
            residual = pred - obs_batch
            loss = data_misfit(pred, obs_batch)
            loss.backward()
            raw_grad = param.parameter.grad.detach().clone()
            last_raw_grad_norm = float(raw_grad.norm().item())
            processed = _apply_gradient_processing(raw_grad, cfg.gradient, current_velocity=param.v().detach())
            if cfg.preconditioner.kind == "diagonal_energy":
                assert energy is not None
                energy.mul_(float(cfg.preconditioner.diagonal_decay)).add_(
                    processed.detach().square(),
                    alpha=1.0 - float(cfg.preconditioner.diagonal_decay),
                )
            processed = _apply_preconditioner(processed, cfg.preconditioner, energy=energy)
            if cfg.optim.grad_clip is not None:
                norm = processed.norm()
                if float(norm.item()) > float(cfg.optim.grad_clip):
                    processed = processed * (float(cfg.optim.grad_clip) / (norm + 1e-12))
            if boundary_mask is not None:
                processed = processed * boundary_mask
            param.parameter.grad.copy_(processed)
            last_grad_norm = float(processed.norm().item())
            opt.step()
            update = param.parameter.detach() - before
            last_update_norm = float(update.norm().item())

            if cfg.cache_gradients and (k == 0 or not cfg.inspect_first_batch_only):
                diagnostics["raw_gradients"].append(raw_grad.detach().cpu().clone())
                diagnostics["processed_gradients"].append(processed.detach().cpu().clone())
                diagnostics["updates"].append(update.detach().cpu().clone())
                diagnostics["gradient_depth_profiles"].append(gradient_depth_profile(raw_grad))
            if k == 0 and batch_i == 0:
                diagnostics["synthetic"].append(pred.detach().cpu().clone())
                diagnostics["residual"].append(residual.detach().cpu().clone())
                diagnostics["loss_decomposition"].append({"data_misfit": float(loss.detach().item())})

            epoch_loss += float(loss.detach().item()) * int(idx.numel())
            epoch_n += int(idx.numel())
            if cfg.inspect_first_batch_only:
                break

        v_now = param.v().detach()
        mean_loss = epoch_loss / max(1, epoch_n)
        history["loss"].append(mean_loss)
        history["gradient_norm"].append(last_grad_norm)
        history["raw_gradient_norm"].append(last_raw_grad_norm)
        history["update_norm"].append(last_update_norm)
        history["v_min"].append(float(v_now.min().item()))
        history["v_max"].append(float(v_now.max().item()))
        if cfg.cache_snapshots and ((k + 1) % max(1, int(cfg.snapshot_every)) == 0 or k == cfg.n_iter - 1):
            snapshots.append(v_now.cpu().clone())
        if log is not None:
            log(
                f"[inspection] iter {k + 1}/{cfg.n_iter} loss={mean_loss:.4e} "
                f"|raw_grad|={last_raw_grad_norm:.3e} |grad|={last_grad_norm:.3e} "
                f"|update|={last_update_norm:.3e} v=[{history['v_min'][-1]:.1f}, {history['v_max'][-1]:.1f}]"
            )

    result: dict[str, Any] = {
        "v": param.v().detach(),
        "parameter": param.parameter.detach().clone(),
        "parameterization": cfg.parameterization,
        "model": param,
        "history": history,
        "diagnostics": diagnostics,
        "snapshots": snapshots,
        "boundary_mask": None if boundary_mask is None else boundary_mask.detach().cpu().clone(),
        "cfg": cfg,
        "runtime_s": perf_counter() - start_time,
    }
    result["summary"] = summarize_run("run", result, v_true=v_true)
    return result


def run_experiments(
    specs: list[ExperimentSpec],
    *,
    v_init: torch.Tensor,
    acq: Acquisition,
    d_obs: torch.Tensor,
    device: torch.device,
    v_min: float,
    v_max: float,
    v_true: torch.Tensor | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, float | str]]]:
    """Run named inspection experiments and return results plus summary rows."""
    if not specs:
        raise ValueError("at least one ExperimentSpec is required")
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, float | str]] = []
    for spec in specs:
        if spec.name in results:
            raise ValueError(f"duplicate experiment name {spec.name!r}")
        if log is not None:
            log(f"[experiment] {spec.name}")
        result = run_classical_fwi_inspection(
            v_init,
            acq,
            d_obs,
            spec.cfg,
            device=device,
            v_min=v_min,
            v_max=v_max,
            v_true=v_true,
            log=log,
        )
        results[spec.name] = result
        rows.append(summarize_run(spec.name, result, v_true=v_true))
    return results, rows


def default_experiment_presets(base: ClassicalInspectionConfig) -> list[ExperimentSpec]:
    """Return practical notebook presets for raw/classical FWI inspection."""
    smooth = replace(
        base.gradient,
        smoothing="gaussian",
        sigma=1.5,
        clip="percentile_soft",
        percentile=99.0,
    )
    taper = replace(base.boundary, enabled=True, taper_type="cosine", width_cells=8, mode="all")
    return [
        ExperimentSpec("baseline_raw_fwi", base),
        ExperimentSpec("boundary_taper_only", replace(base, boundary=taper)),
        ExperimentSpec("gradient_smoothing_only", replace(base, gradient=smooth)),
        ExperimentSpec("taper_plus_smoothing", replace(base, boundary=taper, gradient=smooth)),
        ExperimentSpec(
            "depth_preconditioned",
            replace(base, preconditioner=PreconditionerConfig(kind="depth", depth_power=1.0)),
        ),
        ExperimentSpec(
            "bandlimited_multiscale",
            replace(
                base,
                bandpass_schedule=[
                    BandpassStage(2.0, 4.0, 1),
                    BandpassStage(2.0, 6.0, 1),
                    BandpassStage(2.0, 8.0, max(1, base.n_iter - 2)),
                ],
            ),
        ),
        ExperimentSpec(
            "early_time_muted",
            replace(
                base,
                mute=replace(
                    base.mute,
                    early_time_enabled=True,
                    mute_until_s=0.25,
                    taper_s=0.08,
                ),
            ),
        ),
        ExperimentSpec("alternative_parameterization", replace(base, parameterization="log_velocity")),
        ExperimentSpec(
            "combined_best_guess",
            replace(
                base,
                boundary=taper,
                gradient=smooth,
                preconditioner=PreconditionerConfig(kind="depth", depth_power=1.0),
                mute=replace(base.mute, early_time_enabled=True, mute_until_s=0.20, taper_s=0.08),
            ),
        ),
    ]


__all__ = [
    "BandpassStage",
    "BoundaryUpdateConfig",
    "ClassicalInspectionConfig",
    "ExperimentSpec",
    "GradientProcessingConfig",
    "LogVelocity",
    "MuteConfig",
    "PreconditionerConfig",
    "apply_data_domain_controls",
    "default_experiment_presets",
    "gradient_depth_profile",
    "make_boundary_update_mask",
    "make_velocity_parameterization",
    "run_classical_fwi_inspection",
    "run_experiments",
    "summarize_run",
    "summary_rows_to_markdown",
]
