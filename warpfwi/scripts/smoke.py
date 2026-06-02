"""End-to-end smoke test: tiny classical + Warp-FWI run on synthetic data.

Not a replacement for the notebook; used to validate the pipeline end-to-end
from the command line. Run with:

    .venv/bin/python scripts/smoke.py
"""
from __future__ import annotations

import sys
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import torch

from warpfwi.acquisition import build_acquisition
from warpfwi.config import (
    AcquisitionConfig,
    ClassicalFWIConfig,
    Config,
    DataConfig,
    OptimConfig,
    WarpFWIConfig,
)
from warpfwi.data import load_velocity
from warpfwi.device import select_device
from warpfwi.diagnostics import snr_db, ssim
from warpfwi.fwi_classical import run_classical_fwi
from warpfwi.fwi_warp import pretrain_warp, run_warp_fwi
from warpfwi.inr import build_per_shot_inrs
from warpfwi.modeling import simulate_dataset
from warpfwi.velocity import BoundedVelocity


def main() -> None:
    device = select_device("cpu")
    print(f"device: {device}")
    cfg = Config()
    # Shrink everything for a ~1-minute smoke run.
    cfg.data = DataConfig(marmousi_path=None, synthetic_shape=(48, 96),
                          smooth_sigma=10.0, dx=20.0)
    cfg.acquisition = AcquisitionConfig(
        n_shots=4, f_peak=5.0, dt=0.004, nt=300,
        source_depth=40.0, receiver_depth=40.0,
        n_receivers=40, source_pad=4, receiver_pad=4,
    )
    cfg.classical = ClassicalFWIConfig(
        n_iter=10, optim=OptimConfig(lr_v=0.1, batch_size=2, grad_clip=1.0),
    )
    cfg.warpfwi = WarpFWIConfig(
        stage1_iter=30, stage2_iter=10, lambda_pre=0.1,
        optim=OptimConfig(lr_v=0.1, lr_theta=1e-3, batch_size=2, grad_clip=1.0),
    )

    v_true, v_init = load_velocity(cfg.data, device=device, log=print)
    acq = build_acquisition(cfg.acquisition, cfg.data, v_true.shape, device=device)
    d_obs = simulate_dataset(v_true, acq, device=device, batch_size=cfg.classical.optim.batch_size)
    print(f"obs shape: {tuple(d_obs.shape)}")

    # --- classical baseline
    vp = BoundedVelocity.from_velocity(v_init, cfg.data.v_min, cfg.data.v_max).to(device)
    log_c = run_classical_fwi(vp, acq, d_obs, cfg.classical, device=device, log=print)
    v_c = vp.v().detach()
    print(f"classical  SNR={snr_db(v_c, v_true):.2f} dB  SSIM={ssim(v_c, v_true):.3f}")

    # --- warp-FWI
    vp2 = BoundedVelocity.from_velocity(v_init, cfg.data.v_min, cfg.data.v_max).to(device)
    inrs = build_per_shot_inrs(acq.n_shots, cfg.inr, device=device)
    T_max = cfg.warp.T_max_periods / cfg.acquisition.f_peak
    pretrain_warp(
        vp2, inrs, acq, d_obs, cfg.warpfwi,
        warp_cfg_T_max=T_max,
        warp_cfg_alpha=cfg.warp.alpha_smooth,
        warp_cfg_beta=cfg.warp.beta_gain,
        warp_cfg_offset_smooth=cfg.warp.offset_smooth,
        device=device, log=print,
    )
    run_warp_fwi(
        vp2, inrs, acq, d_obs, cfg.warpfwi,
        warp_cfg_T_max=T_max,
        warp_cfg_alpha=cfg.warp.alpha_smooth,
        warp_cfg_beta=cfg.warp.beta_gain,
        warp_cfg_offset_smooth=cfg.warp.offset_smooth,
        warp_cfg_lambda_0=cfg.warp.lambda_0,
        warp_cfg_lambda_final=cfg.warp.lambda_final,
        warp_cfg_schedule=cfg.warp.schedule,
        device=device, log=print,
    )
    v_w = vp2.v().detach()
    print(f"warp-FWI  SNR={snr_db(v_w, v_true):.2f} dB  SSIM={ssim(v_w, v_true):.3f}")
    print("smoke OK")


if __name__ == "__main__":
    main()
