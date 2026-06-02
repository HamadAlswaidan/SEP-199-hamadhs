"""Stage 1 for monotone NTW: fit tracewise warps on frozen velocity."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from warpfwi.acquisition import Acquisition
from warpfwi.config import MonotoneNTWConfig
from warpfwi.modeling import simulate_dataset
from warpfwi.velocity import BoundedVelocity

from .annealing import annealed_l2, cutoff_at
from .diagnostics import monotone_stats
from .model import MonotoneWarpINR, normalized_increment_grid
from .regularization import monotone_lambda_at, regularization_terms
from .warp import apply_monotone_warp


def pretrain_monotone_warp(
    v_param: BoundedVelocity,
    inrs: dict[int, MonotoneWarpINR],
    acq: Acquisition,
    d_obs: torch.Tensor,
    cfg: MonotoneNTWConfig,
    device: torch.device,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Freeze velocity and optimize one monotone warp INR per shot."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if d_obs.device != device:
        raise RuntimeError(f"d_obs on {d_obs.device}, expected {device}")
    v0 = v_param.v().detach()
    d_syn = simulate_dataset(v0, acq, device=device, batch_size=cfg.optim.batch_size)
    grid_inc = normalized_increment_grid(
        acq.n_receivers, acq.nt, device=device, dtype=d_obs.dtype
    )

    result: dict[str, Any] = {
        "loss_per_shot": [],
        "data_per_shot": [],
        "reg_per_shot": [],
        "cutoff_per_shot": [],
        "lambda_per_shot": [],
        "diagnostics_per_shot": [],
        "cfg": cfg,
    }
    log_every = max(1, acq.n_shots // 10)
    if log is not None:
        log(
            f"[monotone stage1] pretraining {acq.n_shots} shots; "
            f"anneal={cfg.anneal.enabled}"
        )

    for s in range(acq.n_shots):
        inr = inrs[s]
        opt = torch.optim.Adam(inr.parameters(), lr=cfg.optim.lr_theta)
        syn_s = d_syn[s]
        obs_s = d_obs[s]
        losses: list[float] = []
        data_terms: list[float] = []
        reg_terms: list[float] = []
        cutoffs: list[float] = []
        lambdas: list[float] = []
        diag_hist: list[dict[str, float]] = []

        for k in range(cfg.stage1_iter):
            fc = cutoff_at(k, cfg.stage1_iter, cfg.anneal)
            lam = monotone_lambda_at(k, cfg.stage1_iter, cfg.reg)
            opt.zero_grad(set_to_none=True)
            out = apply_monotone_warp(
                syn_s,
                inr,
                grid_inc,
                dt=acq.dt,
                increment_eps=cfg.model.increment_eps,
            )
            data, _pred_lp, _obs_lp = annealed_l2(
                out.warped.unsqueeze(0),
                obs_s.unsqueeze(0),
                dt=acq.dt,
                cutoff_hz=fc,
                cfg=cfg.anneal,
            )
            reg, terms = regularization_terms(out.a_raw, cfg.reg)
            total = data + float(lam) * reg
            total.backward()
            if cfg.optim.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(inr.parameters(), cfg.optim.grad_clip)
            opt.step()

            losses.append(float(total.detach().item()))
            data_terms.append(float(data.detach().item()))
            reg_terms.append(float(reg.detach().item()))
            cutoffs.append(float(fc))
            lambdas.append(float(lam))
            raw_res = syn_s.detach() - obs_s
            warped_res = out.warped.detach() - obs_s
            diag = monotone_stats(out.a_raw.detach(), out.reconstruction, raw_res, warped_res)
            diag["reg_identity"] = float(terms.identity.detach().item())
            diag["reg_time_smooth"] = float(terms.time_smooth.detach().item())
            diag["reg_receiver_smooth"] = float(terms.receiver_smooth.detach().item())
            diag_hist.append(diag)

        result["loss_per_shot"].append(losses)
        result["data_per_shot"].append(data_terms)
        result["reg_per_shot"].append(reg_terms)
        result["cutoff_per_shot"].append(cutoffs)
        result["lambda_per_shot"].append(lambdas)
        result["diagnostics_per_shot"].append(diag_hist)
        if log is not None and ((s + 1) % log_every == 0 or s == acq.n_shots - 1):
            if losses:
                log(
                    f"[monotone stage1] shot {s + 1}/{acq.n_shots}  "
                    f"fc={cutoffs[-1]:.2f}Hz  loss={losses[-1]:.3e}  "
                    f"warped/raw={diag_hist[-1]['warped_residual_norm']:.3e}/"
                    f"{diag_hist[-1]['raw_residual_norm']:.3e}"
                )

    return result


__all__ = ["pretrain_monotone_warp"]
