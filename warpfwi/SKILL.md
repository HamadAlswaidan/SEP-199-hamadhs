# SKILL.md — Warp-FWI project conventions

This file is loaded by Claude Code whenever it works in this repository. It is an
operational reference, not a methods paper. For method derivations and invariants,
see `DESIGN.md`.

## When this skill applies

Any work inside the `warpfwi/` repository: adding features, fixing bugs, running
experiments, editing the notebook, writing tests.

## Environment

This repository ships with a pre-existing Python virtual environment at `.venv/` at
the repo root. All Python and pip commands must use this venv. See `CLAUDE.md` for
the exact activation and invocation patterns.

Installing new packages into `.venv` is allowed when a task requires it. After any
install, pin the version in `pyproject.toml`. See `CLAUDE.md` for the full protocol.

## Architecture at a glance

```
warpfwi/src/warpfwi/
  config.py        dataclasses for every knob the notebook exposes
  device.py        device selection; NEVER silently falls back
  data.py          Marmousi load/crop/smooth; synthetic fallback
  acquisition.py   Ricker wavelet; source+receiver geometry
  modeling.py      deepwave.scalar wrappers (simulate_batch, simulate_dataset)
  velocity.py      bounded velocity via logits + sigmoid
  inr.py           Fourier-feature MLP, 2-channel (τ_raw, a_raw) output
  warp.py          W_θ[d] = (1+a)·d(t−τ)  via grid_sample
  losses.py        data misfit + R(τ, a) regularizer
  schedules.py     LambdaSchedule enum + implementations
  fwi_classical.py baseline L2-FWI loop
  fwi_warp.py      stage-1 pretrain + stage-2 joint (v, θ) loop
  diagnostics.py   SNR, SSIM, warp stats, loss decomposition
  plotting.py      velocity, gather, warp-field, loss-curve plots
```

The notebook at `notebooks/01_warpfwi_marmousi.ipynb` is the driver. It constructs
configs, calls functions from `warpfwi.*`, and plots. It never defines classes or
runs training loops inline.

## Math crib sheet

Warp-and-gain operator:

    W_θ[d](r, t) = (1 + a_θ(r, t)) · d(r, t − τ_θ(r, t))
    τ_θ = T_max · tanh(τ_raw)

INR initialization: Xavier uniform weights, zero bias on the output layer, so
`τ_raw ≈ 0` and `a_raw ≈ 0` at init, hence `W_θ ≈ I` at init.

Joint loss:

    L(v, θ) = data_term(v, θ) + λ_k · R(τ, a)
    R(τ, a) = ‖τ‖² + α·‖∂_t τ‖² + β·‖a‖²

λ schedule (default): geometric from `λ_0 = 1` to `λ_final = 20` over `K` iters.

`T_max` default: `0.4 / f_peak` (seconds).

## Non-negotiable invariants

These are tested in `tests/test_warp.py`. Do not break them.

1. **Identity at init.** With `τ_raw ≡ 0` and `a_raw ≡ 0`, `W_θ[d] == d` bit-exactly.
2. **Constant-shift correctness.** For constant `τ = c` inside the bounded range,
   `W_θ[d](r, t) = d(r, t − c)` at interior samples up to linear-interp error `< 1e-5`.
3. **Linearity in `(1 + a)`.** `W[d; τ, a] = (1+a) · W[d; τ, 0]` exactly.
4. **Autograd.** `loss.backward()` populates gradients for both `v`-logits and `θ`
   when the warp is active.
5. **λ → ∞ limit.** With `θ` frozen and a very large `λ`, the velocity gradient
   matches the classical FWI gradient up to numerical tolerance.

If a change risks any of these, add a regression test before the change.

## Device policy

- One `torch.device` per run. Selected by `warpfwi.device.select_device()` at the
  top of the notebook. Passed explicitly into every function that allocates.
- Never call `.cuda()`. Use `.to(device)` always.
- Never infer device from an input tensor mid-function. If you need the device,
  accept it as a parameter.
- Deepwave's scalar solver supports CPU and CUDA. MPS is not supported — requesting
  MPS for simulation must raise a clear error.
- On MacOS M4 the default is `cpu`. On A100 the default is `cuda`. Nothing else
  should change between environments.

## Notebook conventions

- Every hyperparameter comes from a dataclass in `config.py`. Do not hardcode values
  in the notebook.
- Classical FWI iteration count lives in `ClassicalFWIConfig.n_iter`. Warp-FWI
  stage-1 and stage-2 iteration counts live in `WarpFWIConfig.stage1_iter` and
  `WarpFWIConfig.stage2_iter`. The notebook surfaces these as the top of each
  section so the user can edit them in one place.
- Architecture knobs live in `INRConfig` (depth, width, activation, Fourier bands).
- Training knobs live in `OptimConfig` (lrs, batch size, gradient clip).
- Warp-specific knobs live in `WarpConfig` (`T_max`, α, β, schedule params).
- Plots are figures returned from `plotting.py`, displayed with `plt.show()` (or
  IPython display) in the notebook only.

## Configuration shape (sketch)

```python
@dataclass
class DeviceConfig:
    prefer: Literal["auto", "cuda", "cpu"] = "auto"

@dataclass
class DataConfig:
    marmousi_path: str | None = None
    crop: tuple[int, int, int, int] = ...      # (z0, z1, x0, x1)
    smooth_sigma: float = 40.0                 # for initial model
    v_min: float = 1500.0
    v_max: float = 5000.0

@dataclass
class AcquisitionConfig:
    n_shots: int = 30
    f_peak: float = 5.0
    dt: float = 0.004
    nt: int = 750
    receiver_geometry: Literal["surface"] = "surface"

@dataclass
class INRConfig:
    depth: int = 4
    width: int = 128
    activation: Literal["gelu", "silu", "sine", "tanh"] = "gelu"
    n_fourier_bands: int = 8
    omega_max: float = 16.0

@dataclass
class WarpConfig:
    T_max_periods: float = 0.4                 # fraction of 1/f_peak
    alpha_smooth: float = 1.0
    beta_gain: float = 1.0
    lambda_0: float = 1.0
    lambda_final: float = 20.0
    schedule: Literal["geometric", "linear", "constant"] = "geometric"

@dataclass
class OptimConfig:
    lr_v: float = 0.1
    lr_theta: float = 1e-3
    batch_size: int = 4
    grad_clip: float | None = 1.0

@dataclass
class ClassicalFWIConfig:
    n_iter: int = 200
    optim: OptimConfig = field(default_factory=OptimConfig)

@dataclass
class WarpFWIConfig:
    stage1_iter: int = 200
    stage2_iter: int = 200
    lambda_pre: float = 0.1
    optim: OptimConfig = field(default_factory=OptimConfig)
    warm_start: Literal["none", "near_offset_fwi", "lowfreq_fwi"] = "none"
    n_iter_warmup: int = 30
    n_offsets_warmup: int = 32
    f_lp_warmup: float = 4.0
```

Use `dataclasses.field(default_factory=...)` for nested dataclasses.

## Testing conventions

- `pytest` at the repo root runs all tests.
- Tests must not depend on the Marmousi file being present. Use the synthetic
  fallback in `data.py`.
- Tests for the wave simulation use small grids (e.g. 64×64) to keep CI fast.
- Invariant tests for the warp operator (see list above) must run in under a second.

## Style

- Type hints on all public functions; `from __future__ import annotations` at the
  top of every module.
- Docstrings in NumPy style.
- `ruff` for linting. `black` for formatting (line length 100).
- No `*` imports.
- No `print()` in library code. Use a `log` argument (callable) or `logging`.
- Plotting functions return `matplotlib.figure.Figure`; they do not call
  `plt.show()`.

## Known pitfalls

- **Axis conventions.** Shot gathers are `(R, nt)`. `grid_sample` expects shape
  `(N, C, H, W)`. Reshape carefully; add an explicit shape assertion at the top
  of `warp_and_gain`.
- **Gain baseline.** The network outputs `a_raw` directly; the multiplier is
  `1 + a_raw`. Do not apply an activation to the gain output.
- **`tanh` bound on τ.** Always. Never let `τ` be unbounded. `T_max` in seconds is
  `T_max_periods / f_peak`.
- **Stage-1 early stopping.** Do not iterate stage 1 to convergence with small λ.
  The warp will perfectly fit the residual and stage 2 will start with zero
  data-term gradient. Cap `stage1_iter` at a modest value and keep `λ_pre` at 0.1
  or larger.
- **Deepwave `torch.no_grad()`.** Observed-data generation in `simulate_dataset`
  must run under `torch.no_grad()`. Inside training loops, `simulate_batch` must
  not.
- **Per-shot INR state.** One INR per shot, stored in a `dict[int, TwoChannelINR]`.
  Their parameters must all be passed into the `opt_theta` optimizer. If you add
  shots mid-run, extend the optimizer's parameter groups explicitly.

## Scope guard

If a request falls outside v1 (shared-latent INRs, 3D, elastic, non-acoustic,
time-lapse, different wave physics), leave a TODO and propose it as a v2 ticket
before implementing. Do not silently expand scope.
