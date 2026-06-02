"""Stage-1 warp pretraining and stage-2 joint Warp-FWI.

Two public functions:

* :func:`pretrain_warp` — stage 1. Freezes ``v = v_0`` and fits each per-shot
  INR so that ``W_θ[d_syn_s(v_0)] ≈ d_obs_s`` under a small regularizer.
* :func:`run_warp_fwi` — stage 2. Joint Adam on ``φ`` and all ``θ_s`` with
  the λ schedule from :mod:`warpfwi.schedules`.
"""
from __future__ import annotations

from itertools import chain
from typing import Any, Callable

import numpy as np
import torch

from .acquisition import Acquisition
from .config import (
    CumulativeWarpConfig,
    GainParam,
    SpectralScheduleConfig,
    Stage2Routing,
    WarpFWIConfig,
    WarpParam,
)
from .diagnostics import cumulative_warp_stats
from .inr import TwoChannelINR
from .losses import Misfit, build_misfit, combined_loss
from .modeling import iter_minibatches, simulate_batch, simulate_dataset
from .preprocessing import (
    apply_gradient_boundary_taper_,
    format_fwi_filtering_mode,
    gradient_taper_from_options,
    maybe_bandpass_acquisition_source,
    maybe_bandpass_dataset,
    maybe_bandpass_synthetic_dataset,
)
from .schedules import lambda_at
from .spectral_schedule import (
    compute_band_weights,
    stage1_total_iters,
    stage2_total_iters,
)
from .velocity import BoundedVelocity, velocity_diagnostics
from .warp import WarpGain, normalized_rt_grid


def _theta_params(inrs: dict[int, TwoChannelINR]) -> list[torch.nn.Parameter]:
    return list(chain.from_iterable(m.parameters() for m in inrs.values()))


def select_stage2_routing_mode(
    stage2_theta_misfit: str,
    stage2_v_misfit: str,
    stage2_routing: Stage2Routing | str,
) -> str:
    """Return ``"single"`` for the legacy L2 path or ``"dual"`` for routing."""
    routing = str(stage2_routing).strip().lower()
    if routing not in ("auto", "single", "dual"):
        raise ValueError(
            f"stage2_routing must be one of 'auto', 'single', 'dual'; got "
            f"{stage2_routing!r}"
        )
    if routing in ("single", "dual"):
        return routing
    theta_name = str(stage2_theta_misfit).strip().lower()
    v_name = str(stage2_v_misfit).strip().lower()
    return "single" if theta_name == "l2" and v_name == "l2" else "dual"


def _configured_misfit(name: str, envelope_eps: float) -> Misfit:
    return build_misfit(name, eps=envelope_eps)


def _forward_warp_batch(
    syn: torch.Tensor,
    idx: torch.Tensor,
    inrs: dict[int, TwoChannelINR],
    grid_rt: torch.Tensor,
    dt: float,
    warp_cfg_T_max: float,
    delta_max: float,
    outer_tanh: bool,
    detach_input: bool = False,
    detach_warp_fields: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    GainParam,
]:
    """Apply per-shot warps for a minibatch and stack diagnostics fields."""
    batch_gain_param = inrs[int(idx[0].item())].gain_param
    preds: list[torch.Tensor] = []
    taus: list[torch.Tensor] = []
    gain_raws: list[torch.Tensor] = []
    delta_boundeds: list[torch.Tensor] = []
    any_gain_raw = False
    any_delta = False
    for b, s in enumerate(idx.tolist()):
        warp_mod = WarpGain(
            inrs[s],
            T_max=warp_cfg_T_max,
            delta_max=delta_max,
            outer_tanh=outer_tanh,
            Dr_norm=1.0,
        )
        syn_s = syn[b].detach() if detach_input else syn[b]
        wg_out = warp_mod(
            syn_s,
            grid_rt,
            dt=dt,
            detach_warp_fields=detach_warp_fields,
        )
        preds.append(wg_out.warped)
        taus.append(wg_out.tau)
        if wg_out.gain_raw is not None:
            gain_raws.append(wg_out.gain_raw)
            any_gain_raw = True
        if wg_out.delta_bounded is not None:
            delta_boundeds.append(wg_out.delta_bounded)
            any_delta = True
    pred_b = torch.stack(preds, dim=0)
    tau_b = torch.stack(taus, dim=0)
    gain_raw_b = torch.stack(gain_raws, dim=0) if any_gain_raw else None
    delta_bounded_b = torch.stack(delta_boundeds, dim=0) if any_delta else None
    return pred_b, tau_b, gain_raw_b, delta_bounded_b, batch_gain_param


def pretrain_warp(
    v_param: BoundedVelocity,
    inrs: dict[int, TwoChannelINR],
    acq: Acquisition,
    d_obs: torch.Tensor,
    cfg: WarpFWIConfig,
    warp_cfg_T_max: float,
    warp_cfg_alpha: float,
    warp_cfg_beta: float,
    warp_cfg_offset_smooth: bool,
    device: torch.device,
    log: Callable[[str], None] | None = None,
    spectral_cfg: SpectralScheduleConfig | None = None,
    warp_param: WarpParam = WarpParam.DIRECT,
    cumulative_cfg: CumulativeWarpConfig | None = None,
) -> dict[str, Any]:
    """Stage 1 — fit per-shot warps on frozen ``v = v_0``.

    The synthetic dataset ``d_syn = F(v_0)`` is computed once under
    :func:`torch.no_grad`. Each INR is then fit shot-by-shot with Adam on its
    own parameters to minimize
    ``mean( (W_θ[d_syn_s] − d_obs_s)² ) + λ_pre · R(τ, a)``.

    Parameters
    ----------
    v_param:
        Velocity parameterization. Not updated here but used to produce
        ``d_syn``.
    inrs:
        Mapping ``shot_index -> TwoChannelINR``, updated in place.
    acq:
        Acquisition.
    d_obs:
        Observed gathers ``(S, R, nt)``.
    cfg:
        :class:`WarpFWIConfig`.
    warp_cfg_T_max, warp_cfg_alpha, warp_cfg_beta, warp_cfg_offset_smooth:
        Warp operator and regularizer hyperparameters (pulled from the
        calling notebook's :class:`WarpConfig` — kept as explicit args to
        avoid a circular import of the whole config dataclass).
    device:
        Target :class:`torch.device`.
    log:
        Optional logging callable.

    Returns
    -------
    dict
        ``{"loss_per_shot": list[list[float]], "cfg": cfg}``.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if d_obs.device != device:
        raise RuntimeError(f"d_obs on {d_obs.device}, expected {device}")
    d_obs = maybe_bandpass_dataset(d_obs, acq.dt, cfg.preprocess)
    acq_modeling = maybe_bandpass_acquisition_source(acq, cfg.preprocess)

    v0 = v_param.v().detach()
    d_syn = simulate_dataset(v0, acq_modeling, cfg.modeling, device=device, batch_size=cfg.optim.batch_size)
    d_syn = maybe_bandpass_synthetic_dataset(d_syn, acq.dt, cfg.preprocess)

    grid_rt = normalized_rt_grid(acq.n_receivers, acq.nt, device=device)
    stage1_misfit = _configured_misfit(cfg.stage1_misfit, cfg.envelope_eps)
    log_every = max(1, acq.n_shots // 10)
    if log is not None:
        if cfg.preprocess.bandpass_enabled:
            log(
                "[stage1:preprocess] "
                f"bandpass enabled f=[{cfg.preprocess.bandpass_fmin_hz:g}, "
                f"{cfg.preprocess.bandpass_fmax_hz:g}] Hz "
                f"order={cfg.preprocess.bandpass_order}; time_axis=-1"
            )
        log(format_fwi_filtering_mode("stage1", cfg.preprocess))
        log(
            f"[stage1] pretraining all {acq.n_shots} shots; "
            f"misfit={cfg.stage1_misfit}; logging every {log_every} shot(s)"
        )

    spectral_enabled = spectral_cfg is not None and spectral_cfg.enabled
    if spectral_enabled:
        assert spectral_cfg is not None  # narrowed by the check above
        s1_total = stage1_total_iters(spectral_cfg, cfg.stage1_iter)
    spectral_frontier_history: list[list[float]] = []
    spectral_active_bands_history: list[list[float]] = []

    cumulative_enabled = warp_param is WarpParam.CUMULATIVE
    if cumulative_enabled:
        if cumulative_cfg is None:
            raise ValueError(
                "cumulative_cfg must be provided when warp_param is CUMULATIVE"
            )
        delta_max = cumulative_cfg.delta_max_periods / float(acq.f_peak)
        outer_tanh = cumulative_cfg.outer_tanh
    else:
        delta_max = 0.0
        outer_tanh = True

    loss_per_shot: list[list[float]] = []
    data_per_shot: list[list[float]] = []
    for s in range(acq.n_shots):
        inr = inrs[s]
        warp_mod = WarpGain(
            inr,
            T_max=warp_cfg_T_max,
            delta_max=delta_max,
            outer_tanh=outer_tanh,
            Dr_norm=1.0,
        ).to(device)
        opt = torch.optim.Adam(inr.parameters(), lr=cfg.optim.lr_theta)
        syn_s = d_syn[s]
        obs_s = d_obs[s]
        gain_param = inr.gain_param
        n_bands = inr.encoder.n_bands
        losses: list[float] = []
        data_terms: list[float] = []
        frontier_hist: list[float] = []
        active_hist: list[float] = []
        for it in range(cfg.stage1_iter):
            if spectral_enabled:
                # Progressive spectral unmasking: install per-band weights on
                # the INR before the forward pass. Disabled path skips this
                # entirely so the encoder is bit-exact with the legacy code.
                weights = compute_band_weights(
                    it, s1_total, n_bands, spectral_cfg
                ).to(device)
                inr.set_band_weights(weights)
                low = min(int(spectral_cfg.include_low_in_init), n_bands)
                progress = float(it) / float(s1_total)
                frontier_val = min(
                    float(n_bands),
                    max(0.0, low + (n_bands - low) * progress),
                )
                frontier_hist.append(frontier_val)
                active_hist.append(float(weights.sum().item()))
            opt.zero_grad(set_to_none=True)
            wg_out = warp_mod(syn_s, grid_rt, dt=acq.dt)
            warped, tau, gain_raw = wg_out.warped, wg_out.tau, wg_out.gain_raw
            # `tau` and `gain_raw` are 2-D (R, nt) on the single-shot path;
            # only `pred`/`obs` need the leading batch dim for `data_misfit`.
            # `warp_regularizer` accepts 2-D tau/gain_raw directly and
            # requires they share shape, so leave them unsqueezed here.
            total, _terms = combined_loss(
                warped.unsqueeze(0), obs_s.unsqueeze(0), tau, gain_raw,
                gain_param=gain_param,
                alpha=warp_cfg_alpha, beta=warp_cfg_beta,
                lam=cfg.lambda_pre, offset_smooth=warp_cfg_offset_smooth,
                delta_bounded=wg_out.delta_bounded,
                misfit=stage1_misfit,
            )
            total.backward()
            if cfg.optim.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(inr.parameters(), cfg.optim.grad_clip)
            opt.step()
            losses.append(float(total.detach().item()))
            data_terms.append(float(_terms["data"].item()))
        loss_per_shot.append(losses)
        data_per_shot.append(data_terms)
        if spectral_enabled:
            spectral_frontier_history.append(frontier_hist)
            spectral_active_bands_history.append(active_hist)
        if log is not None and ((s + 1) % log_every == 0 or s == acq.n_shots - 1):
            if losses:
                log(
                    f"[stage1] shot {s + 1}/{acq.n_shots}  "
                    f"loss[0]={losses[0]:.3e}  loss[-1]={losses[-1]:.3e}"
                )
            else:
                log(f"[stage1] shot {s + 1}/{acq.n_shots}  no stage1 iterations")

    if len(loss_per_shot) != acq.n_shots:
        raise AssertionError(
            f"stage1 expected {acq.n_shots} shot histories, got {len(loss_per_shot)}"
        )
    expected_len = int(cfg.stage1_iter)
    if any(len(hist) != expected_len for hist in loss_per_shot):
        bad = [i for i, hist in enumerate(loss_per_shot) if len(hist) != expected_len]
        raise AssertionError(
            f"stage1 expected {expected_len} iterations per shot; mismatches at shots {bad}"
        )
    if log is not None:
        completed = sum(1 for hist in loss_per_shot if len(hist) == expected_len)
        log(f"[stage1] pretrained {completed}/{acq.n_shots} shots")
    result: dict[str, Any] = {
        "loss_per_shot": loss_per_shot,
        "stage1_data_term": data_per_shot,
        "stage1_misfit_name": str(cfg.stage1_misfit),
        "n_shots_pretrained": len(loss_per_shot),
        "iterations_per_shot": expected_len,
        "cfg": cfg,
    }
    if spectral_enabled:
        result["spectral_frontier"] = spectral_frontier_history
        result["spectral_active_bands"] = spectral_active_bands_history
    return result


def run_warp_fwi(
    v_param: BoundedVelocity,
    inrs: dict[int, TwoChannelINR],
    acq: Acquisition,
    d_obs: torch.Tensor,
    cfg: WarpFWIConfig,
    warp_cfg_T_max: float,
    warp_cfg_alpha: float,
    warp_cfg_beta: float,
    warp_cfg_offset_smooth: bool,
    warp_cfg_lambda_0: float,
    warp_cfg_lambda_final: float,
    warp_cfg_schedule: str,
    device: torch.device,
    log: Callable[[str], None] | None = None,
    spectral_cfg: SpectralScheduleConfig | None = None,
    warp_param: WarpParam = WarpParam.DIRECT,
    cumulative_cfg: CumulativeWarpConfig | None = None,
) -> dict[str, Any]:
    """Stage 2 — joint Adam on ``(φ, {θ_s})`` with a λ schedule.

    Per outer iteration ``k``:

    * ``λ_k`` is computed from :func:`warpfwi.schedules.lambda_at`.
    * For each minibatch ``B`` of shots, ``pred_s = W_{θ_s}[F_s(v(φ))]``,
      the combined loss is backpropagated, and both ``opt_v`` (Adam on
      ``φ``) and ``opt_theta`` (Adam on all ``θ_s``) step once.

    Returns
    -------
    dict
        Log dict with per-iteration lists and ``"v_snapshots"`` for plotting.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.velocity_optimizer == "lbfgs_velocity_only":
        raise NotImplementedError(
            "WarpFWI velocity-only L-BFGS is configured but not implemented in this "
            "version. Classical FWI L-BFGS is available. The intended WarpFWI "
            "variant must freeze theta during each velocity L-BFGS closure and "
            "only update theta in a separate deterministic Adam step."
        )
    if cfg.velocity_optimizer != "adam":
        raise ValueError(
            "warpfwi.velocity_optimizer must be 'adam' or "
            f"'lbfgs_velocity_only', got {cfg.velocity_optimizer!r}"
        )
    if cfg.theta_optimizer != "adam":
        raise NotImplementedError(
            "WarpFWI currently supports theta_optimizer='adam' only; joint or "
            "theta L-BFGS would make the velocity line-search objective "
            "non-deterministic."
        )

    if d_obs.device != device or v_param.phi.device != device:
        raise RuntimeError("Tensors must live on `device` before entry")
    d_obs = maybe_bandpass_dataset(d_obs, acq.dt, cfg.preprocess)
    acq_modeling = maybe_bandpass_acquisition_source(acq, cfg.preprocess)

    grid_rt = normalized_rt_grid(acq.n_receivers, acq.nt, device=device)

    opt_v = torch.optim.Adam([v_param.phi], lr=cfg.optim.lr_v)
    theta_params = _theta_params(inrs)
    opt_theta = torch.optim.Adam(theta_params, lr=cfg.optim.lr_theta)
    grad_taper = gradient_taper_from_options(
        cfg.preprocess,
        tuple(v_param.phi.shape),  # type: ignore[arg-type]
        device=device,
        dtype=v_param.phi.dtype,
    )

    batch_size = int(cfg.optim.batch_size)
    K = int(cfg.stage2_iter)
    routing_mode = select_stage2_routing_mode(
        cfg.stage2_theta_misfit,
        cfg.stage2_v_misfit,
        cfg.stage2_routing,
    )
    stage2_theta_misfit = _configured_misfit(
        cfg.stage2_theta_misfit, cfg.envelope_eps
    )
    stage2_v_misfit = _configured_misfit(cfg.stage2_v_misfit, cfg.envelope_eps)

    hist: dict[str, list[float]] = {
        "lambda": [], "loss": [], "data": [], "reg": [],
        "tau_sq": [], "gain_raw_sq": [], "dtau_dt_sq": [],
        "phi_grad_norm": [], "theta_grad_norm": [],
        "max_abs_tau_over_Tmax": [],
        "phi_min": [], "phi_max": [],
        "v_min": [], "v_max": [],
        "frac_near_vmin": [], "frac_near_vmax": [],
    }
    if routing_mode == "dual":
        hist["stage2_theta_data_term"] = []
        hist["stage2_v_data_term"] = []
        hist["stage2_total_loss"] = []
    spectral_enabled = spectral_cfg is not None and spectral_cfg.enabled
    if spectral_enabled:
        assert spectral_cfg is not None  # narrowed
        s2_total = stage2_total_iters(spectral_cfg, cfg.stage1_iter)
        hist["spectral_frontier"] = []
        hist["spectral_active_bands"] = []

    cumulative_enabled = warp_param is WarpParam.CUMULATIVE
    if cumulative_enabled:
        if cumulative_cfg is None:
            raise ValueError(
                "cumulative_cfg must be provided when warp_param is CUMULATIVE"
            )
        delta_max = cumulative_cfg.delta_max_periods / float(acq.f_peak)
        outer_tanh = cumulative_cfg.outer_tanh
        hist["delta_abs_mean"] = []
        hist["delta_saturation"] = []
        hist["tau_outer_saturation"] = []
        hist["tau_monotonicity"] = []
    else:
        delta_max = 0.0
        outer_tanh = True
    snapshots: list[torch.Tensor] = [v_param.v().detach().cpu().clone()]
    tau_snapshots: list[torch.Tensor] = []
    snap_every = max(1, K // 10)

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    if log is not None:
        if grad_taper is not None:
            log(
                "[stage2:preprocess] "
                f"gradient_taper enabled width={cfg.preprocess.gradient_taper_width_cells} "
                f"type={cfg.preprocess.gradient_taper_type}"
            )
        if cfg.preprocess.bandpass_enabled:
            log(
                "[stage2:preprocess] "
                f"bandpass enabled f=[{cfg.preprocess.bandpass_fmin_hz:g}, "
                f"{cfg.preprocess.bandpass_fmax_hz:g}] Hz "
                f"order={cfg.preprocess.bandpass_order}; time_axis=-1"
            )
        log(format_fwi_filtering_mode("stage2", cfg.preprocess))
        log(
            f"[stage2] routing={routing_mode}  "
            f"theta_misfit={cfg.stage2_theta_misfit}  "
            f"v_misfit={cfg.stage2_v_misfit}"
        )

    for k in range(K):
        lam = lambda_at(
            k, K, warp_cfg_lambda_0, warp_cfg_lambda_final, warp_cfg_schedule
        )
        if spectral_enabled:
            assert spectral_cfg is not None  # narrowed by the stage-2 precheck
            # Stage-2 iteration counter is offset by cfg.stage1_iter so the
            # frontier picks up where stage 1 left off. Applied to every INR
            # before the forward passes in this outer iteration.
            first_s = 0
            n_bands = inrs[first_s].encoder.n_bands
            weights = compute_band_weights(
                cfg.stage1_iter + k, s2_total, n_bands, spectral_cfg
            ).to(device)
            for _inr in inrs.values():
                _inr.set_band_weights(weights)
            low = min(int(spectral_cfg.include_low_in_init), n_bands)
            progress = float(cfg.stage1_iter + k) / float(s2_total)
            frontier_val = min(
                float(n_bands),
                max(0.0, low + (n_bands - low) * progress),
            )
            active_val = float(weights.sum().item())
        agg_keys = ("loss", "data", "reg", "tau_sq", "gain_raw_sq", "dtau_dt_sq")
        if routing_mode == "dual":
            agg_keys = (
                *agg_keys,
                "stage2_theta_data_term",
                "stage2_v_data_term",
                "stage2_total_loss",
            )
        agg = {kk: 0.0 for kk in agg_keys}
        n_samples = 0
        last_phi_g = 0.0
        last_theta_g = 0.0
        max_abs_tau = 0.0

        for idx in iter_minibatches(acq.n_shots, batch_size, generator=gen, device=device):
            opt_v.zero_grad(set_to_none=True)
            opt_theta.zero_grad(set_to_none=True)
            v = v_param.v()
            syn = simulate_batch(v, acq_modeling, idx, cfg.modeling, device=device)  # (B, R, nt)
            syn = maybe_bandpass_synthetic_dataset(syn, acq.dt, cfg.preprocess)
            obs_batch = d_obs.index_select(0, idx)

            if routing_mode == "single":
                pred_b, tau_b, gain_raw_b, delta_bounded_b, batch_gain_param = (
                    _forward_warp_batch(
                        syn,
                        idx,
                        inrs,
                        grid_rt,
                        dt=acq.dt,
                        warp_cfg_T_max=warp_cfg_T_max,
                        delta_max=delta_max,
                        outer_tanh=outer_tanh,
                    )
                )
                total, terms = combined_loss(
                    pred_b, obs_batch, tau_b, gain_raw_b,
                    gain_param=batch_gain_param,
                    alpha=warp_cfg_alpha, beta=warp_cfg_beta,
                    lam=lam, offset_smooth=warp_cfg_offset_smooth,
                    delta_bounded=delta_bounded_b,
                )
                data_for_log = terms["data"]
            else:
                pred_theta, tau_b, gain_raw_b, delta_bounded_b, batch_gain_param = (
                    _forward_warp_batch(
                        syn,
                        idx,
                        inrs,
                        grid_rt,
                        dt=acq.dt,
                        warp_cfg_T_max=warp_cfg_T_max,
                        delta_max=delta_max,
                        outer_tanh=outer_tanh,
                        detach_input=True,
                    )
                )
                theta_loss, terms = combined_loss(
                    pred_theta, obs_batch, tau_b, gain_raw_b,
                    gain_param=batch_gain_param,
                    alpha=warp_cfg_alpha, beta=warp_cfg_beta,
                    lam=lam, offset_smooth=warp_cfg_offset_smooth,
                    delta_bounded=delta_bounded_b,
                    misfit=stage2_theta_misfit,
                )
                pred_v, _tau_v, _gain_raw_v, _delta_bounded_v, _ = (
                    _forward_warp_batch(
                        syn,
                        idx,
                        inrs,
                        grid_rt,
                        dt=acq.dt,
                        warp_cfg_T_max=warp_cfg_T_max,
                        delta_max=delta_max,
                        outer_tanh=outer_tanh,
                        detach_warp_fields=True,
                    )
                )
                v_data_term = stage2_v_misfit(pred_v, obs_batch)
                total = theta_loss + v_data_term
                data_for_log = terms["data"] + v_data_term.detach()
            total.backward()
            if cfg.optim.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([v_param.phi], cfg.optim.grad_clip)
                torch.nn.utils.clip_grad_norm_(theta_params, cfg.optim.grad_clip)
            apply_gradient_boundary_taper_(v_param.phi.grad, grad_taper)
            if v_param.phi.grad is None:
                last_phi_g = 0.0
            else:
                last_phi_g = float(v_param.phi.grad.detach().norm().item())
            # aggregate theta grad norm
            sq = 0.0
            for p in theta_params:
                if p.grad is not None:
                    sq += float(p.grad.detach().pow(2).sum().item())
            last_theta_g = float(sq ** 0.5)

            opt_v.step()
            opt_theta.step()

            b = int(idx.numel())
            n_samples += b
            agg["loss"] += float(total.detach().item()) * b
            agg["data"] += float(data_for_log.item()) * b
            agg["reg"] += float(terms["reg"].item()) * b
            agg["tau_sq"] += float(terms["tau_sq"].item()) * b
            agg["gain_raw_sq"] += float(terms["gain_raw_sq"].item()) * b
            agg["dtau_dt_sq"] += float(terms["dtau_dt_sq"].item()) * b
            if routing_mode == "dual":
                agg["stage2_theta_data_term"] += float(terms["data"].item()) * b
                agg["stage2_v_data_term"] += float(v_data_term.detach().item()) * b
                agg["stage2_total_loss"] += float(total.detach().item()) * b
            mat = float(tau_b.detach().abs().max().item())
            if mat > max_abs_tau:
                max_abs_tau = mat
            # Stash the last batch's cumulative fields for per-iter stats.
            last_tau_b = tau_b.detach()
            last_delta_bounded_b = (
                delta_bounded_b.detach() if delta_bounded_b is not None else None
            )
            last_idx_tolist = idx.tolist()

        denom = max(1, n_samples)
        for kk, val in agg.items():
            hist[kk].append(val / denom)
        hist["lambda"].append(lam)
        hist["phi_grad_norm"].append(last_phi_g)
        hist["theta_grad_norm"].append(last_theta_g)
        hist["max_abs_tau_over_Tmax"].append(
            max_abs_tau / warp_cfg_T_max if warp_cfg_T_max > 0 else 0.0
        )
        v_stats = velocity_diagnostics(v_param)
        hist["phi_min"].append(v_stats["phi_min"])
        hist["phi_max"].append(v_stats["phi_max"])
        hist["v_min"].append(v_stats["v_min"])
        hist["v_max"].append(v_stats["v_max"])
        hist["frac_near_vmin"].append(v_stats["frac_near_vmin"])
        hist["frac_near_vmax"].append(v_stats["frac_near_vmax"])
        if spectral_enabled:
            hist["spectral_frontier"].append(frontier_val)
            hist["spectral_active_bands"].append(active_val)

        if cumulative_enabled and last_delta_bounded_b is not None:
            # Recompute delta_raw for the last batch's INRs to feed the
            # saturation metric. Using the last-batch snapshot keeps the
            # logging cost O(batch_size) per outer iteration.
            with torch.no_grad():
                delta_raws_list: list[torch.Tensor] = []
                for s_ in last_idx_tolist:
                    out_ = inrs[int(s_)](grid_rt)
                    if out_.delta_raw is None:
                        raise AssertionError(
                            "CUMULATIVE run produced INR output without delta_raw"
                        )
                    delta_raws_list.append(out_.delta_raw)
                delta_raw_b = torch.stack(delta_raws_list, dim=0)
            cum_stats = cumulative_warp_stats(
                tau=last_tau_b,
                delta_raw=delta_raw_b,
                delta_bounded=last_delta_bounded_b,
                T_max=warp_cfg_T_max,
            )
            hist["delta_abs_mean"].append(cum_stats["delta_abs_mean"])
            hist["delta_saturation"].append(cum_stats["delta_saturation"])
            hist["tau_outer_saturation"].append(cum_stats["tau_outer_saturation"])
            hist["tau_monotonicity"].append(cum_stats["tau_monotonicity"])

        if (k + 1) % snap_every == 0 or k == K - 1:
            snapshots.append(v_param.v().detach().cpu().clone())
            with torch.no_grad():
                s0 = 0
                warp_mod = WarpGain(
                    inrs[s0],
                    T_max=warp_cfg_T_max,
                    delta_max=delta_max,
                    outer_tanh=outer_tanh,
                    Dr_norm=1.0,
                )
                v = v_param.v()
                syn0 = simulate_batch(v, acq_modeling, torch.tensor([s0], device=device), cfg.modeling, device=device)
                syn0 = maybe_bandpass_synthetic_dataset(syn0, acq.dt, cfg.preprocess)
                wg_snap = warp_mod(syn0[0], grid_rt, dt=acq.dt)
                tau0 = wg_snap.tau
                tau_snapshots.append(tau0.detach().cpu().clone())

        if log is not None:
            log(
                f"[stage2] iter {k + 1}/{K}  lam={lam:.3f}  "
                f"loss={hist['loss'][-1]:.3e}  data={hist['data'][-1]:.3e}  "
                f"reg={hist['reg'][-1]:.3e}  |tau|/Tmax={hist['max_abs_tau_over_Tmax'][-1]:.3f}  "
                f"phi=[{v_stats['phi_min']:.2f}, {v_stats['phi_max']:.2f}]  "
                f"v=[{v_stats['v_min']:.1f}, {v_stats['v_max']:.1f}]  "
                f"near_bounds=({v_stats['frac_near_vmin']:.3f}, {v_stats['frac_near_vmax']:.3f})"
            )

    return {
        **hist,
        "stage2_routing_mode": routing_mode,
        "stage2_theta_misfit_name": str(cfg.stage2_theta_misfit),
        "stage2_v_misfit_name": str(cfg.stage2_v_misfit),
        "v_snapshots": snapshots,
        "tau_snapshots": tau_snapshots,
        "cfg": cfg,
    }


__all__ = ["pretrain_warp", "run_warp_fwi", "select_stage2_routing_mode"]
