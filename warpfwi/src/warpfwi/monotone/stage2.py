"""Stage 2 joint velocity/monotone-warp optimization with routed gradients."""
from __future__ import annotations

from itertools import chain
from typing import Any, Callable

import numpy as np
import torch

from warpfwi.acquisition import Acquisition
from warpfwi.config import MonotoneNTWConfig
from warpfwi.modeling import iter_minibatches, simulate_batch
from warpfwi.velocity import BoundedVelocity, velocity_diagnostics

from .annealing import annealed_l2, cutoff_at
from .diagnostics import monotone_stats
from .model import MonotoneWarpINR, normalized_increment_grid
from .regularization import monotone_lambda_at, regularization_terms
from .warp import apply_monotone_warp


def _theta_params(inrs: dict[int, MonotoneWarpINR]) -> list[torch.nn.Parameter]:
    return list(chain.from_iterable(m.parameters() for m in inrs.values()))


def _warp_batch(
    syn: torch.Tensor,
    idx: torch.Tensor,
    inrs: dict[int, MonotoneWarpINR],
    grid_inc: torch.Tensor,
    cfg: MonotoneNTWConfig,
    acq: Acquisition,
    detach_input: bool = False,
    detach_psi: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Any]]:
    preds: list[torch.Tensor] = []
    raws: list[torch.Tensor] = []
    taus: list[torch.Tensor] = []
    recs: list[Any] = []
    for b, s in enumerate(idx.tolist()):
        syn_s = syn[b].detach() if detach_input else syn[b]
        out = apply_monotone_warp(
            syn_s,
            inrs[int(s)],
            grid_inc,
            dt=acq.dt,
            increment_eps=cfg.model.increment_eps,
            detach_psi=detach_psi,
        )
        preds.append(out.warped)
        raws.append(out.a_raw)
        taus.append(out.reconstruction.tau)
        recs.append(out.reconstruction)
    return torch.stack(preds, dim=0), torch.stack(raws, dim=0), torch.stack(taus, dim=0), recs


def run_monotone_warp_fwi(
    v_param: BoundedVelocity,
    inrs: dict[int, MonotoneWarpINR],
    acq: Acquisition,
    d_obs: torch.Tensor,
    cfg: MonotoneNTWConfig,
    device: torch.device,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Jointly update velocity and monotone warp INRs with routed branches."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if d_obs.device != device or v_param.phi.device != device:
        raise RuntimeError("Tensors must live on `device` before entry")

    grid_inc = normalized_increment_grid(
        acq.n_receivers, acq.nt, device=device, dtype=d_obs.dtype
    )
    theta_params = _theta_params(inrs)
    opt_v = torch.optim.Adam([v_param.phi], lr=cfg.optim.lr_v)
    opt_theta = torch.optim.Adam(theta_params, lr=cfg.optim.lr_theta)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    k_total = int(cfg.stage2_iter)

    hist: dict[str, list[float]] = {
        "loss": [],
        "theta_data": [],
        "v_data": [],
        "reg": [],
        "lambda": [],
        "cutoff": [],
        "a_raw_max_abs": [],
        "tau_max_abs": [],
        "increment_min": [],
        "increment_max": [],
        "slope_min": [],
        "slope_max": [],
        "raw_residual_norm": [],
        "warped_residual_norm": [],
        "phi_grad_norm": [],
        "theta_grad_norm": [],
        "v_min": [],
        "v_max": [],
        "frac_near_vmin": [],
        "frac_near_vmax": [],
    }
    snapshots = [v_param.v().detach().cpu().clone()]
    tau_snapshots: list[torch.Tensor] = []
    psi_snapshots: list[torch.Tensor] = []
    snap_every = max(1, k_total // 10)

    if log is not None:
        log("[monotone stage2] routed theta branch and detached-warp velocity branch")

    for k in range(k_total):
        fc = cutoff_at(k, k_total, cfg.anneal)
        lam = monotone_lambda_at(k, k_total, cfg.reg)
        per_batch_keys = (
            "loss",
            "theta_data",
            "v_data",
            "reg",
            "lambda",
            "cutoff",
            "a_raw_max_abs",
            "tau_max_abs",
            "increment_min",
            "increment_max",
            "slope_min",
            "slope_max",
            "raw_residual_norm",
            "warped_residual_norm",
        )
        agg = {key: 0.0 for key in per_batch_keys}
        n_samples = 0
        last_phi_g = 0.0
        last_theta_g = 0.0

        for idx in iter_minibatches(acq.n_shots, cfg.optim.batch_size, generator=gen, device=device):
            opt_v.zero_grad(set_to_none=True)
            opt_theta.zero_grad(set_to_none=True)
            v = v_param.v()
            syn = simulate_batch(v, acq, idx, device=device)
            obs = d_obs.index_select(0, idx)

            pred_theta, a_raw_b, _tau_b, recs = _warp_batch(
                syn,
                idx,
                inrs,
                grid_inc,
                cfg,
                acq,
                detach_input=True,
            )
            theta_data, _pred_lp, _obs_lp = annealed_l2(
                pred_theta, obs, dt=acq.dt, cutoff_hz=fc, cfg=cfg.anneal
            )
            reg, _terms = regularization_terms(a_raw_b, cfg.reg)

            pred_v, _a_raw_v, _tau_v, _recs_v = _warp_batch(
                syn,
                idx,
                inrs,
                grid_inc,
                cfg,
                acq,
                detach_psi=True,
            )
            v_data, _pred_v_lp, _obs_v_lp = annealed_l2(
                pred_v, obs, dt=acq.dt, cutoff_hz=fc, cfg=cfg.anneal
            )
            total = theta_data + v_data + float(lam) * reg
            total.backward()
            if cfg.optim.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([v_param.phi], cfg.optim.grad_clip)
                torch.nn.utils.clip_grad_norm_(theta_params, cfg.optim.grad_clip)
            last_phi_g = (
                0.0 if v_param.phi.grad is None else float(v_param.phi.grad.detach().norm().item())
            )
            sq = 0.0
            for p in theta_params:
                if p.grad is not None:
                    sq += float(p.grad.detach().pow(2).sum().item())
            last_theta_g = float(sq ** 0.5)
            opt_v.step()
            opt_theta.step()

            b = int(idx.numel())
            n_samples += b
            diag = monotone_stats(
                a_raw_b.detach(),
                recs[0],
                raw_residual=syn.detach() - obs,
                warped_residual=pred_theta.detach() - obs,
            )
            agg["loss"] += float(total.detach().item()) * b
            agg["theta_data"] += float(theta_data.detach().item()) * b
            agg["v_data"] += float(v_data.detach().item()) * b
            agg["reg"] += float(reg.detach().item()) * b
            agg["lambda"] += float(lam) * b
            agg["cutoff"] += float(fc) * b
            for key in (
                "a_raw_max_abs",
                "tau_max_abs",
                "increment_min",
                "increment_max",
                "slope_min",
                "slope_max",
                "raw_residual_norm",
                "warped_residual_norm",
            ):
                agg[key] += float(diag[key]) * b

        denom = max(1, n_samples)
        for key, val in agg.items():
            hist[key].append(val / denom)
        hist["phi_grad_norm"].append(last_phi_g)
        hist["theta_grad_norm"].append(last_theta_g)
        v_stats = velocity_diagnostics(v_param)
        for key in ("v_min", "v_max", "frac_near_vmin", "frac_near_vmax"):
            hist[key].append(v_stats[key])

        if (k + 1) % snap_every == 0 or k == k_total - 1:
            snapshots.append(v_param.v().detach().cpu().clone())
            with torch.no_grad():
                out0 = apply_monotone_warp(
                    simulate_batch(
                        v_param.v(),
                        acq,
                        torch.tensor([0], device=device),
                        device=device,
                    )[0],
                    inrs[0],
                    grid_inc,
                    dt=acq.dt,
                    increment_eps=cfg.model.increment_eps,
                )
                tau_snapshots.append(out0.reconstruction.tau.detach().cpu().clone())
                psi_snapshots.append(out0.reconstruction.psi.detach().cpu().clone())

        if log is not None:
            log(
                f"[monotone stage2] iter {k + 1}/{k_total}  fc={fc:.2f}Hz  "
                f"loss={hist['loss'][-1]:.3e}  v_data={hist['v_data'][-1]:.3e}  "
                f"reg={hist['reg'][-1]:.3e}  tau|max={hist['tau_max_abs'][-1]:.3e}"
            )

    return {
        **hist,
        "v_snapshots": snapshots,
        "tau_snapshots": tau_snapshots,
        "psi_snapshots": psi_snapshots,
        "cfg": cfg,
    }


__all__ = ["run_monotone_warp_fwi"]
