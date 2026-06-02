# Warp-FWI

Research implementation of **Warp-FWI**, a two-stage full-waveform inversion
method that uses a local time-warp auxiliary, parameterized by a coordinate-based
neural network, to reduce cycle-skipping during early iterations.

See [`DESIGN.md`](DESIGN.md) for method details and [`SKILL.md`](SKILL.md) for
project conventions.

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Warp-FWI
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install the package in editable mode

```bash
pip install -e .
```

## Verify installation

```bash
python -c "import torch, deepwave, numpy; print('OK')"
```

Tests do not require Marmousi data. A synthetic layered model is used as fallback.

## Notes

- Python 3.9 or newer is recommended.
- GPU is optional but strongly recommended for FWI experiments.
- If using CUDA, make sure your PyTorch version matches your CUDA runtime.
- The `.venv/` directory is intentionally not tracked by Git. Each user should create their own local environment.

## Run the notebook

```bash
.venv/bin/jupyter lab
```

Open `notebooks/01_warpfwi_marmousi.ipynb`. Top cell selects the device via
`warpfwi.device.select_device('auto')` and the second cell constructs every
configuration dataclass the experiment uses. Edit those — not the library —
to change iteration counts, learning rates, the INR architecture, etc.

Sections in the notebook:

1. Imports and device.
2. Config construction (all dataclasses).
3. Data loading (Marmousi or synthetic fallback).
4. Observed data generation.
5. Classical FWI baseline + plots.
6. Warp-FWI stage 1 (warp pretraining) + plots.
7. Warp-FWI stage 2 (joint `(v, θ)` training) + plots.
8. Comparison panel.

## Where to configure

| Knob | Dataclass | Location |
| ---- | --------- | -------- |
| Marmousi path, crop, smoothing | `DataConfig` | `src/warpfwi/config.py` |
| Source/receiver counts, spacing, depths, geometry mode | `AcquisitionConfig` | same |
| Deepwave `dx`, `dt`, PML, accuracy | `ModelingConfig` | same |
| INR depth / width / activation / Fourier bands | `INRConfig` | same |
| `T_max`, α, β, λ schedule | `WarpConfig` | same |
| Stage-specific data misfits / routing | `WarpFWIConfig.stage1_misfit`, `.stage2_theta_misfit`, `.stage2_v_misfit`, `.stage2_routing` | same |
| Adam/L-BFGS selection, lrs, batch size, grad clip | `OptimConfig` or `OptimizerConfig` | same |
| Classical FWI iterations | `ClassicalFWIConfig.n_iter` | same |
| Warp stage-1 / stage-2 iterations | `WarpFWIConfig.stage1_iter`, `.stage2_iter` | same |

## Acquisition and modeling

`build_acquisition(...)` returns both Deepwave index tensors and physical
meter-coordinate tensors:

- `source_locations`, `receiver_locations`: integer `[iz, ix]` locations for
  models shaped `(nz, nx)`.
- `source_locations_m`, `receiver_locations_m`: physical `[x_m, z_m]`
  coordinates for plotting and diagnostics.

`AcquisitionConfig.geometry_mode` supports `fixed_spread`, `moving_spread`, and
`split_spread`. The modeling wrappers in `warpfwi.modeling` are shared by
classical FWI and Warp-FWI, so both methods use the same acquisition/backend.

## Optimizers

Classical FWI supports Adam and PyTorch L-BFGS:

```python
cfg.classical.optimizer = OptimizerConfig(
    name="lbfgs",
    lr=1.0,
    max_iter=1,
    history_size=10,
    line_search_fn="strong_wolfe",
)
```

Adam remains the default through `OptimConfig(name="adam", lr_v=...)`.
Observed and synthetic data use the same optional bandpass preprocessing.
Velocity-gradient tapering is applied to `phi.grad`, because velocity is
optimized through bounded logits.

Warp-FWI remains Adam-by-default. `WarpFWIConfig.velocity_optimizer =
"lbfgs_velocity_only"` is reserved for the experimental velocity-only L-BFGS
path and currently raises a clear `NotImplementedError`; joint L-BFGS over
velocity and INR weights is intentionally not exposed because it would mutate
theta inside a velocity line-search closure.

Gradient note: project code relies on PyTorch autograd. The loops call
`loss.backward()`, Deepwave handles the wave-equation backward pass internally,
and gradients flow through the bounded velocity logits. Warp-FWI additionally
backpropagates through the differentiable warp operator and INR parameters when
those fields are not detached.

## Smoke test

A small end-to-end CLI run (no notebook, no Marmousi):

```bash
.venv/bin/python scripts/smoke.py
```

Prints iteration logs for classical FWI, stage-1 pretraining, and stage-2
joint training on a tiny synthetic model (~1 minute on CPU).

## Repository layout

```
warpfwi/
├── CLAUDE.md              operational rules for Claude Code
├── DESIGN.md              method specification
├── SKILL.md               project conventions and invariants
├── README.md              this file
├── pyproject.toml
├── src/warpfwi/
│   ├── __init__.py
│   ├── device.py          device selection, no silent fallback
│   ├── config.py          dataclasses for every hyperparameter
│   ├── data.py            Marmousi loading + synthetic fallback
│   ├── acquisition.py     geometry + Ricker wavelet
│   ├── modeling.py        deepwave.scalar wrappers
│   ├── velocity.py        bounded sigmoid-reparameterized velocity
│   ├── warp.py            W_θ[d] = (1+a)·d(t−τ) via grid_sample
│   ├── inr.py             two-channel Fourier-feature MLP
│   ├── losses.py          configurable data misfits + R(τ, a) regularizer
│   ├── schedules.py       LambdaSchedule enum
│   ├── fwi_classical.py   baseline L2-FWI
│   ├── fwi_warp.py        stage-1 + stage-2 loops
│   ├── diagnostics.py     SNR, SSIM, warp stats
│   └── plotting.py        matplotlib helpers (no plt.show)
├── notebooks/
│   └── 01_warpfwi.ipynb
└── scripts/
    ├── smoke.py           end-to-end CLI smoke run
    └── download_marmousi.py
```

## Device policy

One `torch.device` per run, selected once via
`warpfwi.device.select_device()` at the top of the notebook and passed
explicitly into every function that allocates tensors. MPS is not supported
(Deepwave's scalar solver is CPU/CUDA only); requesting it raises.
