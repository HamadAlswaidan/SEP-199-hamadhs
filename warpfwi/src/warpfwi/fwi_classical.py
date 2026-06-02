"""Classical L2-FWI baseline on the bounded-velocity logits.

One public function :func:`run_classical_fwi`. Adam preserves the historical
minibatched behavior. L-BFGS uses a deterministic closure, normally full-batch,
over the same Deepwave modeling and preprocessing backend.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from .acquisition import Acquisition
from .config import ClassicalFWIConfig
from .losses import data_misfit
from .modeling import iter_minibatches, simulate_batch
from .optim import build_optimizer, optimizer_from_legacy
from .preprocessing import (
    apply_gradient_boundary_taper_,
    format_fwi_filtering_mode,
    gradient_taper_from_options,
    maybe_bandpass_acquisition_source,
    maybe_bandpass_dataset,
    maybe_bandpass_synthetic_dataset,
)
from .velocity import BoundedVelocity, tensor_bound_diagnostics, velocity_diagnostics


def run_classical_fwi(
    v_param: BoundedVelocity,
    acq: Acquisition,
    d_obs: torch.Tensor,
    cfg: ClassicalFWIConfig,
    device: torch.device,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the classical FWI baseline in-place on ``v_param``.

    Parameters
    ----------
    v_param:
        :class:`~warpfwi.velocity.BoundedVelocity`, updated in place. Its
        parameter ``phi`` is the optimization variable.
    acq:
        Acquisition; tensors must reside on ``device``.
    d_obs:
        Observed data of shape ``(S, R, nt)``, float32, on ``device``.
    cfg:
        :class:`~warpfwi.config.ClassicalFWIConfig`.
    device:
        Target :class:`torch.device`. All tensors in ``v_param`` and ``acq``
        must already be on this device.
    log:
        Optional logging callable invoked as ``log(message)`` once per
        outer iteration. If ``None``, nothing is printed.

    Returns
    -------
    dict
        Log dict with keys ``"loss"`` (list[float] per outer iter),
        ``"phi_grad_norm"`` (list[float]), ``"v_snapshots"``
        (list[torch.Tensor] sampled every few iters), ``"cfg"``.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if v_param.phi.device != device:
        raise RuntimeError(
            f"v_param.phi is on {v_param.phi.device}, expected {device}. "
            f"Move the module before calling run_classical_fwi."
        )
    if d_obs.device != device:
        raise RuntimeError(f"d_obs on {d_obs.device}, expected {device}")
    d_obs = maybe_bandpass_dataset(d_obs, acq.dt, cfg.preprocess)
    acq_modeling = maybe_bandpass_acquisition_source(acq, cfg.preprocess)

    opt_cfg = cfg.optimizer if cfg.optimizer is not None else optimizer_from_legacy(cfg.optim)
    opt_name = str(opt_cfg.name).lower()
    grad_taper = gradient_taper_from_options(
        cfg.preprocess,
        tuple(v_param.phi.shape),  # type: ignore[arg-type]
        device=device,
        dtype=v_param.phi.dtype,
    )
    batch_size = int(cfg.optim.batch_size)
    n_iter = int(cfg.n_iter)
    n_shots = acq.n_shots

    loss_hist: list[float] = []
    residual_norm_hist: list[float] = []
    grad_hist: list[float] = []
    phi_min_hist: list[float] = []
    phi_max_hist: list[float] = []
    v_min_hist: list[float] = []
    v_max_hist: list[float] = []
    frac_low_hist: list[float] = []
    frac_high_hist: list[float] = []
    snapshots: list[torch.Tensor] = [v_param.v().detach().cpu().clone()]
    snap_every = max(1, n_iter // 10)

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    init_v = (
        v_param.v_init_reference
        if hasattr(v_param, "v_init_reference")
        else v_param.v().detach()
    )
    init_phi = (
        v_param.phi_init_reference
        if hasattr(v_param, "phi_init_reference")
        else v_param.phi.detach()
    )
    init_bounds = tensor_bound_diagnostics(init_v, v_param.v_min, v_param.v_max)
    init_recon = v_param.v().detach()
    init_recon_max_err = float((init_recon - init_v).abs().max().item())
    init_recon_mean_err = float((init_recon - init_v).abs().mean().item())
    init_phi_abs = init_phi.abs()
    init_over_20 = float((init_phi_abs > 20.0).to(torch.float32).mean().item())
    init_over_40 = float((init_phi_abs > 40.0).to(torch.float32).mean().item())
    init_over_60 = float((init_phi_abs > 60.0).to(torch.float32).mean().item())
    if log is not None:
        if grad_taper is not None:
            log(
                "[classical:preprocess] "
                f"gradient_taper enabled width={cfg.preprocess.gradient_taper_width_cells} "
                f"type={cfg.preprocess.gradient_taper_type}"
            )
        if cfg.preprocess.bandpass_enabled:
            log(
                "[classical:preprocess] "
                f"bandpass enabled f=[{cfg.preprocess.bandpass_fmin_hz:g}, "
                f"{cfg.preprocess.bandpass_fmax_hz:g}] Hz "
                f"order={cfg.preprocess.bandpass_order}; time_axis=-1"
            )
        log(format_fwi_filtering_mode("classical", cfg.preprocess))
        log(
            f"[classical:init] optimizer={opt_name} lr={opt_cfg.lr:.3g}  "
            f"v_init=[{init_bounds['v_min']:.1f}, {init_bounds['v_max']:.1f}]  "
            f"near_bounds=({init_bounds['frac_near_vmin']:.3f}, {init_bounds['frac_near_vmax']:.3f})  "
            f"exact_bounds=({init_bounds['frac_exact_vmin']:.3f}, {init_bounds['frac_exact_vmax']:.3f})  "
            f"phi0=[{float(init_phi.min().item()):.2f}, {float(init_phi.max().item()):.2f}]  "
            f"|phi0|>(20,40,60)=({init_over_20:.3f}, {init_over_40:.3f}, {init_over_60:.3f})  "
            f"recon_err(max,mean)=({init_recon_max_err:.3e}, {init_recon_mean_err:.3e})"
        )

    if opt_name == "lbfgs":
        opt = build_optimizer([v_param.phi], opt_cfg)
        shot_idx_full = torch.arange(n_shots, device=device, dtype=torch.long)
        if not opt_cfg.use_full_batch_for_lbfgs and batch_size < n_shots:
            shot_sets = list(iter_minibatches(n_shots, batch_size, generator=gen, device=device))
        else:
            shot_sets = [shot_idx_full]

        closure_loss_hist: list[float] = []
        closure_grad_hist: list[float] = []
        closure_eval_hist: list[int] = []
        closure_counter = 0

        for k in range(n_iter):
            outer_loss = 0.0
            outer_n = 0
            last_grad = 0.0
            outer_closure_start = closure_counter
            for idx in shot_sets:
                obs_batch = d_obs.index_select(0, idx)

                def closure() -> torch.Tensor:
                    nonlocal closure_counter, last_grad
                    opt.zero_grad(set_to_none=True)
                    v = v_param.v()
                    pred = simulate_batch(v, acq_modeling, idx, cfg.modeling, device=device)
                    pred = maybe_bandpass_synthetic_dataset(pred, acq.dt, cfg.preprocess)
                    loss = data_misfit(pred, obs_batch)
                    loss.backward()
                    if opt_cfg.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_([v_param.phi], opt_cfg.gradient_clip)
                    apply_gradient_boundary_taper_(v_param.phi.grad, grad_taper)
                    last_grad = (
                        0.0
                        if v_param.phi.grad is None
                        else float(v_param.phi.grad.detach().norm().item())
                    )
                    closure_counter += 1
                    closure_loss_hist.append(float(loss.detach().item()))
                    closure_grad_hist.append(last_grad)
                    if log is not None and opt_cfg.log_closure_evals:
                        log(
                            f"[classical:lbfgs:closure] outer={k + 1} "
                            f"eval={closure_counter} loss={float(loss.detach().item()):.4e} "
                            f"|grad|={last_grad:.3e}"
                        )
                    return loss

                loss_tensor = opt.step(closure)
                outer_loss += float(loss_tensor.detach().item()) * int(idx.numel())
                outer_n += int(idx.numel())

            mean_loss = outer_loss / max(1, outer_n)
            residual_norm = float(mean_loss ** 0.5)
            stats = velocity_diagnostics(v_param)
            loss_hist.append(mean_loss)
            residual_norm_hist.append(residual_norm)
            grad_hist.append(last_grad)
            phi_min_hist.append(stats["phi_min"])
            phi_max_hist.append(stats["phi_max"])
            v_min_hist.append(stats["v_min"])
            v_max_hist.append(stats["v_max"])
            frac_low_hist.append(stats["frac_near_vmin"])
            frac_high_hist.append(stats["frac_near_vmax"])
            closure_eval_hist.append(closure_counter - outer_closure_start)
            if (k + 1) % snap_every == 0 or k == n_iter - 1:
                snapshots.append(v_param.v().detach().cpu().clone())
            if log is not None:
                log(
                    f"[classical:lbfgs] iter {k + 1}/{n_iter}  loss={mean_loss:.4e}  "
                    f"resid={residual_norm:.3e}  closures={closure_eval_hist[-1]}  "
                    f"|grad|={last_grad:.3e}  "
                    f"phi=[{stats['phi_min']:.2f}, {stats['phi_max']:.2f}]  "
                    f"v=[{stats['v_min']:.1f}, {stats['v_max']:.1f}]"
                )

        return {
            "loss": loss_hist,
            "residual_norm": residual_norm_hist,
            "phi_grad_norm": grad_hist,
            "closure_loss": closure_loss_hist,
            "closure_phi_grad_norm": closure_grad_hist,
            "closure_evals_per_iter": closure_eval_hist,
            "optimizer_name": opt_name,
            "phi_min": phi_min_hist,
            "phi_max": phi_max_hist,
            "v_min": v_min_hist,
            "v_max": v_max_hist,
            "frac_near_vmin": frac_low_hist,
            "frac_near_vmax": frac_high_hist,
            "v_snapshots": snapshots,
            "cfg": cfg,
        }
    if opt_name != "adam":
        raise ValueError(f"classical optimizer must be 'adam' or 'lbfgs', got {opt_cfg.name!r}")

    opt = build_optimizer([v_param.phi], opt_cfg)
    for k in range(n_iter):
        epoch_loss = 0.0
        epoch_n = 0
        last_grad = 0.0
        for idx in iter_minibatches(n_shots, batch_size, generator=gen, device=device):
            opt.zero_grad(set_to_none=True)
            v = v_param.v()
            pred = simulate_batch(v, acq_modeling, idx, cfg.modeling, device=device)
            pred = maybe_bandpass_synthetic_dataset(pred, acq.dt, cfg.preprocess)
            obs_batch = d_obs.index_select(0, idx)
            loss = data_misfit(pred, obs_batch)
            loss.backward()
            if opt_cfg.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_([v_param.phi], opt_cfg.gradient_clip)
            apply_gradient_boundary_taper_(v_param.phi.grad, grad_taper)
            last_grad = (
                0.0
                if v_param.phi.grad is None
                else float(v_param.phi.grad.detach().norm().item())
            )
            if log is not None and k == 0 and epoch_n == 0:
                pre_stats = velocity_diagnostics(v_param)
                log(
                    f"[classical:first-batch:pre-step]  "
                    f"phi=[{pre_stats['phi_min']:.2f}, {pre_stats['phi_max']:.2f}]  "
                    f"v=[{pre_stats['v_min']:.1f}, {pre_stats['v_max']:.1f}]  "
                    f"near_bounds=({pre_stats['frac_near_vmin']:.3f}, {pre_stats['frac_near_vmax']:.3f})  "
                    f"|grad|={last_grad:.3e}"
                )
            opt.step()
            if log is not None and k == 0 and epoch_n == 0:
                post_stats = velocity_diagnostics(v_param)
                log(
                    f"[classical:first-batch:post-step]  "
                    f"phi=[{post_stats['phi_min']:.2f}, {post_stats['phi_max']:.2f}]  "
                    f"v=[{post_stats['v_min']:.1f}, {post_stats['v_max']:.1f}]  "
                    f"near_bounds=({post_stats['frac_near_vmin']:.3f}, {post_stats['frac_near_vmax']:.3f})"
                )
            epoch_loss += float(loss.detach().item()) * idx.numel()
            epoch_n += int(idx.numel())
        mean_loss = epoch_loss / max(1, epoch_n)
        residual_norm = float(mean_loss ** 0.5)
        stats = velocity_diagnostics(v_param)
        loss_hist.append(mean_loss)
        residual_norm_hist.append(residual_norm)
        grad_hist.append(last_grad)
        phi_min_hist.append(stats["phi_min"])
        phi_max_hist.append(stats["phi_max"])
        v_min_hist.append(stats["v_min"])
        v_max_hist.append(stats["v_max"])
        frac_low_hist.append(stats["frac_near_vmin"])
        frac_high_hist.append(stats["frac_near_vmax"])
        if (k + 1) % snap_every == 0 or k == n_iter - 1:
            snapshots.append(v_param.v().detach().cpu().clone())
        if log is not None:
            log(
                f"[classical] iter {k + 1}/{n_iter}  loss={mean_loss:.4e}  "
                f"resid={residual_norm:.3e}  |grad|={last_grad:.3e}  "
                f"phi=[{stats['phi_min']:.2f}, {stats['phi_max']:.2f}]  "
                f"v=[{stats['v_min']:.1f}, {stats['v_max']:.1f}]  "
                f"near_bounds=({stats['frac_near_vmin']:.3f}, {stats['frac_near_vmax']:.3f})"
            )

    return {
        "loss": loss_hist,
        "residual_norm": residual_norm_hist,
        "phi_grad_norm": grad_hist,
        "phi_min": phi_min_hist,
        "phi_max": phi_max_hist,
        "v_min": v_min_hist,
        "v_max": v_max_hist,
        "frac_near_vmin": frac_low_hist,
        "frac_near_vmax": frac_high_hist,
        "v_snapshots": snapshots,
        "optimizer_name": opt_name,
        "cfg": cfg,
    }


__all__ = ["run_classical_fwi"]
