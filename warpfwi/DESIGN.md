# Warp-FWI: Local Time-Shift and Gain Auxiliary for Cycle-Skip Mitigation in FWI

Design document — v1.

## 1. Purpose

Classical full-waveform inversion (FWI) with an L2 data misfit is ill-posed when the
initial velocity model is far from the true model: the data residual is dominated by
cycle-skipped phase errors and the gradient with respect to velocity is not a descent
direction for the true model. Warp-FWI introduces an auxiliary variable — a local
time-shift and gain applied to the synthetic data — parameterized by a coordinate-based
neural network (INR). The auxiliary absorbs phase misalignment in early iterations and
is penalized toward identity, so that as optimization proceeds the velocity must absorb
the alignment that the auxiliary is giving up. The construction follows the
model-extension principle (Van Leeuwen and Herrmann 2013, Warner and Guasch 2016) with
a specific, physically meaningful parameterization of the extension.

The method is most closely related to adaptive waveform inversion (AWI) with a
non-stationary, parametric matching filter, and to differentiable dynamic time warping.

## 2. Notation

- `v ∈ ℝ^(nz × nx)`: gridded velocity model.
- `Δx`, `Δt`: spatial grid spacing and temporal sampling.
- Shots indexed by `s ∈ {1, …, S}`.
- Each shot gather has shape `(R, nt)`: receivers on axis 0, time on axis 1.
- `F_s`: differentiable acoustic forward map for shot `s`, implemented by
  `deepwave.scalar`.
- `d_obs_s ∈ ℝ^(R × nt)`: observed gather.
- `d_syn_s(v) = F_s(v)`: synthetic gather.
- `θ_s`: parameters of the auxiliary network for shot `s`.
- `τ_θ ∈ ℝ^(R × nt)`: time-shift field produced by the network.
- `gain_θ ∈ ℝ^(R × nt)`: gain field produced by the network (definition depends on
  the parameterization, see §3.1).
- `W_θ`: warp-and-gain operator acting on a shot gather.

## 3. The warp-and-gain operator

Definition:

    W_θ[d](r, t) = gain_θ(r, t) · d(r, t − τ_θ(r, t))

with

- `τ_θ = T_max · tanh(τ_raw)`: structurally bounded to |τ| ≤ T_max.
- `gain_θ` defined by one of three parameterizations selected at INR construction
  time (see §3.1).
- `d(r, t − τ)` evaluated by linear interpolation in time via
  `torch.nn.functional.grid_sample(mode='bilinear', padding_mode='border',
  align_corners=False)`.

Choice of `T_max`: a fraction of the dominant period at the source peak frequency,
default `0.4 / f_peak`. This structurally prevents the warp from itself cycle-skipping.

### 3.1 Gain parameterizations

The gain factor admits three parameterizations, selected by `INRConfig.gain_param`
(`GainParam` enum in `config.py`). The choice affects three places consistently:
the INR output head width (§4), the gain-factor computation in `warp.py`, and the
regularizer in §5.

`NONE` — pure time warp, no gain channel:

    gain_θ(r, t) ≡ 1

The INR outputs only `τ_raw` (one channel). The β term in the regularizer is
ignored. This is the cleanest baseline for isolating the contribution of the gain
channel.

`ADDITIVE` — additive perturbation around unity:

    gain_θ(r, t) = 1 + a_raw(r, t)

The INR outputs `(τ_raw, a_raw)`. At the zero-init output layer, `a_raw ≈ 0` so
`gain_θ ≈ 1`. The regularizer penalizes `β · ‖a_raw‖²`. This was the original v1
form. It admits a structural failure mode — the optimizer can drive `a_raw → −1`
to zero out the synthetic at problematic offsets, which is finite penalty cost
(`β · 1`) but can yield a large data-term reduction. No finite β fully prevents
this because increasing β enough to outlaw `a_raw → −1` also kills legitimate
amplitude corrections.

`MULTIPLICATIVE` — log-space gain (preferred):

    gain_θ(r, t) = exp(g_raw(r, t))

The INR outputs `(τ_raw, g_raw)`. At the zero-init output layer, `g_raw ≈ 0` so
`gain_θ ≈ 1` exactly. The regularizer penalizes `β · ‖g_raw‖²`. The "delete the
synthetic" state requires `g_raw → −∞`, which is infinite penalty cost — amplitude
collapse is structurally unreachable. This mirrors the design pattern of bounding
`τ` via `tanh`: pathological states are made unreachable rather than merely
expensive.

The penalty is always computed on the *raw* network output, not on the gain factor
itself. For `MULTIPLICATIVE`, penalizing `‖g_raw‖²` is what diverges as
`g_raw → −∞`; penalizing `‖exp(g_raw) − 1‖²` would be a finite cost and would not
have the same structural effect.

The default in v1 is `ADDITIVE` for reproducibility of earlier experiments.
`MULTIPLICATIVE` is the recommended form for new experiments and is expected to
become the default in v2.

### 3.2 Differentiability contract

- Differentiable w.r.t. `d` (hence w.r.t. `v` through `F_s`).
- Differentiable w.r.t. `τ_raw` and the gain raw output (hence w.r.t. `θ`).
- At `τ_raw ≡ 0` and gain-raw-output `≡ 0` (or absent for `NONE`), `W_θ[d] == d`
  bit-exactly, for all three parameterizations. This is a unit-tested invariant.
- Under `torch.no_grad()` the operator returns the correct forward value with no
  graph.

## 4. The INR

Network `f_θ : [−1, 1]² → ℝ^C` with `C = 1` for `GainParam.NONE` and `C = 2`
otherwise. Inputs are evaluated on a normalized receiver-time mesh
`(ξ_r, τ_n) ∈ [−1, 1]²`.

- Fourier-feature encoding:
  `γ(ξ, τ) = [ξ, τ, sin(2π ω_b ξ), cos(2π ω_b ξ), sin(2π ω_b τ), cos(2π ω_b τ)]_{b=1..B}`
  with `ω_b = ω_max^{η_b}`, `η_b` linearly spaced in `[0, 1]`.
- MLP: `L` hidden layers, width `W`, activation `σ ∈ {gelu, silu, sine, tanh}`.
- Output layer: linear, `C` outputs, Xavier uniform weights, zero bias — so at init
  all raw outputs are near zero, and `W_θ ≈ I`.
- Forward returns `INROutput(tau_raw, gain_raw)` where `gain_raw is None` for
  `GainParam.NONE`. Returning a named structure (rather than a raw tuple)
  prevents downstream modules from misinterpreting which channel is which.

Per-shot independence: one INR per shot, stored in a dictionary keyed by shot index.
This mirrors the existing INR-FWI convention and is the simplest choice for v1.
Shared-latent variants are out of scope for v1.

## 5. Regularization

The regularizer dispatches on `GainParam`:

`NONE`:
    R(τ) = ‖τ‖² + α · ‖∂_t τ‖²

`ADDITIVE`:
    R(τ, a_raw) = ‖τ‖² + α · ‖∂_t τ‖² + β · ‖a_raw‖²

`MULTIPLICATIVE`:
    R(τ, g_raw) = ‖τ‖² + α · ‖∂_t τ‖² + β · ‖g_raw‖²

Norms are mean-squared over `(R, nt)` per shot, then averaged over shots in a
minibatch. The smoothness penalty `‖∂_t τ‖²` is a first-order finite difference along
the time axis. An offset-smoothness term `‖∂_r τ‖²` (and the analogous term on the
gain raw output for `ADDITIVE`/`MULTIPLICATIVE`) is exposed as a config flag
`offset_smooth`; when enabled, it is added to `R` with the same `α` weight.

`α` defaults to 1.0, `β` defaults to 1.0. These are relative weights; the overall
scale is controlled by `λ_τ`.

## 6. Data misfits and joint loss

Data misfit terms are configurable through a small `Misfit` interface in
`src/warpfwi/losses.py`. The default `L2Misfit` is the original mean squared
residual. `EnvelopeL2Misfit` compares analytic-signal envelopes, with the Hilbert
transform implemented by `torch.fft` along the time axis.

Stage 1 uses `WarpFWIConfig.stage1_misfit` for the alignment pretraining data
term. Stage 2 has separate `stage2_theta_misfit` and `stage2_v_misfit` fields plus
`stage2_routing`:

`single`:
    Original Warp-FWI objective. This is selected automatically when both stage-2
    misfits are `l2`, and is the default behavior.

`dual`:
    Routed two-branch objective for using different alignment and inversion
    losses. This is selected automatically when either stage-2 misfit is non-L2.

For the original single-objective path and a minibatch `B` of shots:

    L(v, θ) = (1 / |B|) · Σ_{s ∈ B} mean_{r,n} (W_{θ_s}[F_s(v)](r,n) − d_obs_s(r,n))²
              + λ_τ · (1 / |B|) · Σ_{s ∈ B} R(τ_{θ_s}, gain_raw_{θ_s})

Both `v` (via logits `φ` and bounded sigmoid, see §9) and `{θ_s : s ∈ B}` are updated
with Adam per outer iteration. There is no detached alternation.

In dual routing, one synthetic gather `syn = F(v)` is generated per minibatch and
used by two branches:

    pred_θ = W_θ[syn.detach()]
    θ_loss = misfit_θ(pred_θ, d_obs) + λ_τ · R(θ)

    pred_v = W_detached_θ[syn]
    v_loss = misfit_v(pred_v, d_obs)

    loss = θ_loss + v_loss

The theta branch updates only the INR/alignment parameters. The velocity branch
uses detached reconstructed warp fields, so it differentiates through the wave
solver and interpolation with respect to the synthetic data and velocity, but not
through θ. A motivating configuration is `stage1_misfit = "envelope_l2"`,
`stage2_theta_misfit = "envelope_l2"`, `stage2_v_misfit = "l2"`, and
`stage2_routing = "auto"`.

## 7. Algorithm

### Stage 1 — warp pretraining (optional but recommended)

Freeze `v = v_0`, precompute `d_syn_s(v_0)` once under `torch.no_grad()`, and apply
the optional bandpass to both `d_syn` and `d_obs` before fitting the auxiliary.

    for s in shots:
        for it in range(pretrain_iters):
            loss = misfit_stage1(W_{θ_s}[d_syn_s], d_obs_s)
                   + λ_τ_pre · R(τ_{θ_s}, gain_raw_{θ_s})
            adam_step(θ_s)

`λ_τ_pre` is small (default 0.1) so the warp actually aligns events. Do not drive
this to convergence; the point is to initialize θ with an informative warp, not to
zero the residual. Driving stage 1 too long invites the optimizer to discover
data-term-reducing solutions that exploit the auxiliary's freedom (e.g. amplitude
collapse under `ADDITIVE`). Default `pretrain_iters = 200` per shot.

### Stage 2 — joint warp-FWI

    for k in range(K):
        λ_k = schedule(k)
        for batch B of shots:
            syn = {s: F_s(v(φ)) for s in B}
            syn = optional_bandpass(syn)
            if routing_mode == "single":
                pred = {s: W_{θ_s}[syn[s]] for s in B}
                data_term = L2(pred, d_obs)
                reg_term = mean_s R(τ_{θ_s}, gain_raw_{θ_s})
                loss = data_term + λ_k · reg_term
            else:
                pred_θ = {s: W_{θ_s}[syn[s].detach()] for s in B}
                pred_v = {s: W_detached_θ_s[syn[s]] for s in B}
                reg_term = mean_s R(τ_{θ_s}, gain_raw_{θ_s})
                loss = misfit_θ(pred_θ, d_obs) + λ_k · reg_term
                       + misfit_v(pred_v, d_obs)
            loss.backward()
            optional_boundary_taper_(φ.grad)
            adam_step(φ); adam_step({θ_s : s ∈ B})
        log_diagnostics(k)

Gradients flow through `F_s` (deepwave handles this) and through `W_θ` (grid_sample
handles this) back to both `φ` and `θ`.

## 8. λ schedule

Geometric growth from `λ_0` to `λ_final` over `K` outer iterations:

    λ_k = λ_0 · (λ_final / λ_0)^(k / (K − 1))

Defaults: `λ_0 = 1`, `λ_final = 20`, `K = 200`. Constant and linear schedules are also
exposed via a `LambdaSchedule` enum in `config.py`.

The schedule is a hyperparameter. At the end of the schedule, `τ` and the gain raw
output should be close to zero everywhere that the velocity is well-determined.
Regions where they remain large at the final iteration are under-illuminated or
ambiguous.

## 9. Bounded velocity parameterization

Identical to current INR-FWI. Unconstrained logits `φ ∈ ℝ^(nz × nx)` mapped via

    v(φ) = v_min + (v_max − v_min) · sigmoid(φ)

with `v_min, v_max` from config. `φ_0 = sigmoid⁻¹((v_0 − v_min) / (v_max − v_min))`
after clipping `v_0` away from the interval endpoints.

### 9.0 Acquisition, modeling, and optimizer controls

Acquisition geometry is configured by `AcquisitionConfig` and built by
`build_acquisition`. Deepwave receives models shaped `(nz, nx)` and locations as
integer `[iz, ix]` indices. Physical coordinates are stored separately as
`[x_m, z_m]` in `source_locations_m` and `receiver_locations_m`. The builder
validates all source/receiver indices and supports:

- `fixed_spread`: one receiver line reused for every shot.
- `moving_spread`: receiver line translated by shot position.
- `split_spread`: receivers centered around each shot when possible.

`ModelingConfig` controls the shared Deepwave scalar call: grid spacing, `dt`,
`pml_width`, `pml_freq`, and `accuracy`. Classical FWI and Warp-FWI both call
`simulate_batch` / `simulate_dataset`, so they use the same acquisition and
modeling backend.

Classical FWI now supports `OptimizerConfig(name="adam")` and
`OptimizerConfig(name="lbfgs")`. Adam preserves the legacy minibatch behavior.
L-BFGS defaults to a deterministic full-batch closure:

1. zero gradients,
2. map `φ` to bounded velocity `v(φ)`,
3. simulate selected shots with Deepwave,
4. apply the configured synthetic-data preprocessing,
5. compute the data misfit,
6. call `loss.backward()`,
7. apply optional gradient clipping and boundary taper to `φ.grad`,
8. log closure diagnostics,
9. return the loss.

The project uses PyTorch automatic differentiation. We do not manually code the
adjoint-state equations in `fwi_classical.py` or `fwi_warp.py`; Deepwave provides
differentiable propagation and its backward pass. Velocity gradients flow through
Deepwave back to `φ`. Warp-FWI gradients additionally flow through the
differentiable warp operator and INR parameters when those branches are not
detached.

Warp-FWI remains Adam-by-default. The experimental config value
`velocity_optimizer="lbfgs_velocity_only"` is reserved for a velocity-only L-BFGS
step in which theta is frozen during each closure; joint velocity/theta L-BFGS is
out of scope because updating theta inside a line-search closure makes the
velocity objective non-deterministic.

### 9.1 Reusable FWI preprocessing

Classical FWI and Warp-FWI share optional preprocessing in
`src/warpfwi/preprocessing.py`, configured independently through
`ClassicalFWIConfig.preprocess` and `WarpFWIConfig.preprocess`. All options are
disabled by default, so the original code path is unchanged unless a user opts in.

Velocity-gradient boundary taper:

- `gradient_taper_enabled: bool = False`
- `gradient_taper_width_cells: int = 0`
- `gradient_taper_type: Literal["cosine", "hann"] = "cosine"`

The velocity is optimized as the bounded logit parameter `φ`, not as a raw velocity
tensor. Therefore the taper is applied to `φ.grad` after `loss.backward()` and
gradient clipping, but before `Adam.step()`. This damps the actual update used by the
optimizer while preserving the bounded velocity map `v(φ)`.

The helper `make_boundary_taper((nz, nx), width_cells, taper_type)` creates a 2-D mask
with value exactly 0 on the top, bottom, left, and right boundary cells. It ramps to 1
over `width_cells` cells on each side and is exactly 1 in the interior. Row and column
tapers are multiplied, so corners receive both boundary tapers. `width_cells = 0`
returns an all-ones mask. Invalid taper types and widths too large to leave an
interior region raise explicit errors.

Data bandpass filtering:

- `bandpass_enabled: bool = False`
- `bandpass_fmin_hz: float = 2.0`
- `bandpass_fmax_hz: float = 8.0`
- `bandpass_order: int = 4`

The data tensors have shape `(S, R, nt)` for full datasets and `(B, R, nt)` for
minibatches; the last axis is always time. When enabled, observed data are filtered
once at FWI entry and synthetic gathers are filtered immediately after each
`simulate_batch` or cached `simulate_dataset` call, before the misfit is computed.
In Warp-FWI, this means the warp auxiliary acts on band-limited synthetic data and is
compared to the same band-limited observed data in both stage 1 and stage 2.

Observed and synthetic data must be filtered consistently: filtering only one side
would change the objective into a comparison between different bandwidths, and
filtering after the warp in one branch but before the warp in another would make the
auxiliary solve a different alignment problem than the velocity branch.

Implementation caveat: the bandpass is a differentiable zero-phase Butterworth-style
filter implemented in torch via `rfft`/`irfft`. It multiplies the time-axis spectrum by
the squared magnitude response of high-pass and low-pass Butterworth prototypes, the
zero-phase response corresponding to a forward/backward filter. This introduces no
phase delay and keeps gradients flowing through synthetic data. Frequencies are
validated against Nyquist with `0 < fmin < fmax < 0.5 / dt`; invalid settings raise a
clear `ValueError`.

Example:

```python
gradient_taper_enabled = True
gradient_taper_width_cells = 4
gradient_taper_type = "cosine"

bandpass_enabled = True
bandpass_fmin_hz = 2.0
bandpass_fmax_hz = 6.0
bandpass_order = 4

fwi_preprocess = FWIPreprocessingConfig(
    gradient_taper_enabled=gradient_taper_enabled,
    gradient_taper_width_cells=gradient_taper_width_cells,
    gradient_taper_type=gradient_taper_type,
    bandpass_enabled=bandpass_enabled,
    bandpass_fmin_hz=bandpass_fmin_hz,
    bandpass_fmax_hz=bandpass_fmax_hz,
    bandpass_order=bandpass_order,
)

cfg.classical.preprocess = replace(fwi_preprocess)
cfg.warpfwi.preprocess = replace(fwi_preprocess)
```

The notebook exposes these exact scalar variables in its main config cell, prints the
resolved settings before inversion, plots the taper mask when enabled, and plots a raw
versus bandpassed observed-trace spectrum after `d_obs` generation when the bandpass is
enabled. The inversion functions still receive raw `d_obs`; they apply the configured
filter internally to observed and synthetic data so both sides of the misfit remain
consistent.

## 10. Initial model and optional warm-start

Three modes, set by `warm_start`:

- `none` (default): use user-provided `v_0` as-is.
- `near_offset_fwi`: before stage 1, run `n_iter_warmup` iterations of classical FWI on
  `v_0` using only the nearest `n_offsets_warmup` receivers per shot. Populates `d_syn`
  with reflectivity without introducing cycle-skipped mid/far-offset contributions into
  `v`.
- `lowfreq_fwi`: before stage 1, run `n_iter_warmup` iterations of classical FWI with
  observed and synthetic data low-pass filtered below `f_lp`. The filter cutoff is a
  config parameter.

Recommendation: try `none` first. Only enable a warm-start if the gain channel on
its own (under `ADDITIVE` or `MULTIPLICATIVE`) cannot amplify the weak reflectivity
in a smoothed initial model enough for the warp to align events.

## 11. Diagnostics and invariants

Logged per outer iteration (stored in a dict; plotted from the notebook):

- Loss decomposition: `data_term`, `λ_k · R`, and each of `‖τ‖²`, `‖∂_t τ‖²`,
  `‖gain_raw‖²` (the last omitted for `NONE`).
- Warp statistics: `max(|τ|)/T_max` (saturation indicator), `mean(|τ|)`, plus
  parameterization-specific gain statistics:
    - `ADDITIVE`: `mean(|a_raw|)`, `min(1 + a_raw)` (collapse indicator).
    - `MULTIPLICATIVE`: `mean(|g_raw|)`, `min(exp(g_raw))`, `max(exp(g_raw))`.
- Residual norms: raw physical residual and warped residual, and their ratio.
- Velocity diagnostics: current SNR and SSIM relative to ground truth (if provided),
  velocity range, logit gradient norm.
- INR diagnostics: parameter gradient norms per shot.

Invariants (unit tests, see §13):

- For each `GainParam` value: `W_θ[d] == d` exactly when all raw network outputs are
  zero (tolerance 0 in float32).
- For constant `τ ≡ c` within the bounded range, `W_θ[d](r, t) == d(r, t − c)` at
  interior samples up to interpolation error.
- For `MULTIPLICATIVE`: as `g_raw → −∞` the gain factor approaches 0 and the
  regularizer term diverges. This is the structural property that prevents
  amplitude collapse.
- The identity `∂L/∂v | (θ fixed, λ → ∞)` equals the classical FWI gradient up to
  numerical tolerance.
- `L` decreases monotonically on a fixed random seed with a small enough step size.
- Boundary taper masks are zero on all four model boundaries and one in the interior
  after the taper zone.
- Bandpass preprocessing preserves tensor shape and differentiability and rejects
  invalid frequency/order settings.

## 12. Repository layout

```
warpfwi/
├── README.md
├── DESIGN.md                (this document)
├── SKILL.md                 (in-repo context for Claude Code)
├── CLAUDE.md                (operational rules for Claude Code sessions)
├── pyproject.toml
├── src/warpfwi/
│   ├── __init__.py
│   ├── config.py            dataclasses for every config block; GainParam enum
│   ├── device.py            device selection + sanity checks
│   ├── data.py              Marmousi loading, cropping, smoothed v_0
│   ├── acquisition.py       source + receiver geometry, Ricker wavelet
│   ├── modeling.py          simulate_batch / simulate_dataset (deepwave.scalar)
│   ├── preprocessing.py     boundary gradient taper + differentiable bandpass
│   ├── velocity.py          bounded velocity parameterization
│   ├── inr.py               Fourier-feature MLP, 1- or 2-channel output
│   ├── warp.py              W_θ operator via grid_sample; gain dispatch
│   ├── losses.py            data misfit + regularizer (dispatches on GainParam)
│   ├── schedules.py         LambdaSchedule enum + implementations
│   ├── fwi_classical.py     baseline L2-FWI loop
│   ├── fwi_warp.py          stage-1 pretraining + stage-2 joint loop
│   ├── diagnostics.py       SNR, SSIM, warp stats, loss decomposition
│   └── plotting.py          velocity / gather / warp-field / loss plots
├── tests/
│   ├── test_warp.py
│   ├── test_inr.py
│   ├── test_losses.py
│   ├── test_preprocessing.py
│   └── test_modeling.py
├── notebooks/
│   └── 01_warpfwi_marmousi.ipynb
└── scripts/
    └── download_marmousi.py
```

## 13. Minimum viable experiment

Dataset: Marmousi patch (same crop as current INR-FWI repo), 5 Hz Ricker, ~30 shots,
surface receiver array.

Baselines:
- Classical L2-FWI with the bounded velocity parameterization, Adam, K_baseline outer
  iterations.

Method:
- Warp-FWI, stage 1 with 200 pretrain iters per shot, stage 2 with K = 200 joint
  iterations, geometric λ schedule `1 → 20`.
- Run all three `GainParam` settings (`NONE`, `ADDITIVE`, `MULTIPLICATIVE`) on the
  same shot for side-by-side comparison.

Deliverables:
- Inverted velocity models, SNR and SSIM against ground truth.
- Shot-gather residual plots before and after inversion, for each `GainParam`.
- Warp field `τ(r, t)` visualized at `k = 0`, `k = K/2`, `k = K − 1`.
- Gain field statistics (collapse indicator for `ADDITIVE`, log-gain extrema for
  `MULTIPLICATIVE`).
- Loss decomposition curves over training.

## 14. Diagnosed failure mode in v1 (motivation for §15 and §16)

Empirically, the v1 construction above fails on the Marmousi patch with a
smoothed initial model at 5 Hz: classical L2-FWI converges cleanly to a useful
model while warp-FWI stalls at a lower SNR regardless of `GainParam`. The
failure has a consistent structure across runs.

Observed pattern, stage 1:

- The warp aligns events well at near offsets.
- At far offsets, the warped synthetic is driven toward zero amplitude rather
  than being time-shifted into alignment. Under `ADDITIVE` this is `a_raw → -1`
  (amplitude collapse). Under `MULTIPLICATIVE` it appears as large negative
  `g_raw` at problematic offsets, mitigated but not eliminated by finite `β`.
- The warp field `τ(r, t)` is smooth at most offsets but develops sharp
  receiver-direction transitions at the boundary between "alignable by
  warping" and "not alignable by warping."

Observed pattern, stage 2:

- The warp becomes *more* complex, not less, over the first several tens of
  iterations. Magnitudes grow; structure sharpens.
- The regularizer term `λ_k · R` stays about an order of magnitude below the
  data term throughout the schedule, so the regularizer is never a binding
  constraint on the optimizer.
- The velocity does not improve and in fact degrades relative to classical
  L2-FWI.

Diagnosis. The inner loss driving the auxiliary is the same L2 misfit
`‖W_θ[d_syn] − d_obs‖²` that cycle-skips the outer problem. Finding `τ(r, t)`
by gradient descent on this objective reproduces the outer cycle-skipping
pathology one level down: the auxiliary rolls into the nearest basin per
trace, which is the correct basin at near offsets (sub-half-cycle shift) and
the wrong basin at far offsets (more than half a cycle misalignment). Where
the optimizer cannot bridge the cycle barrier in `τ`, it discovers a cheaper
descent direction via the gain channel, suppressing the troublesome trace
instead of aligning it.

In stage 2 this is pathological because the suppressed traces no longer carry
a useful velocity gradient. `∂(W[F(v)])/∂v` at a suppressed trace is a
derivative of a near-zero synthetic, not of a misaligned one. The velocity
update cannot recover alignment at those offsets, and as it attempts to fit
the unsuppressed (near-offset) residuals, it moves in directions that
sacrifice the far-offset fit. The sections below describe two structural
modifications to the INR parameterization that attack this mechanism from
different angles. Each is additive, toggled off by default, and composable
with the existing `GainParam` dispatch.

## 15. Spectral schedule on the INR Fourier encoding (Direction 1)

### 15.1 Motivation

The Fourier-feature encoder in §4 admits bands at `ω_b = ω_max^{η_b}` for
`η_b` linearly spaced in `[0, 1]`. With default `B = 12` and `ω_max = 24`,
the highest band has period `1/24` in normalized coordinates — roughly a
~5-receiver / ~5-sample scale. At iteration 0 the INR has full representational
capacity to express `τ(r, t)` varying on this scale. This is what enables the
sharp receiver-direction transitions in the stage-1 warp field (§14).

A coarse-to-fine schedule on the encoding bands removes this freedom early in
training. During the first iterations only the lowest band (plus the raw
coordinate channels) is active; the INR can only represent slowly-varying
`τ(r, t)`. It is forced to find a single coherent large-scale shift across
the gather — the best low-frequency approximation to the true alignment,
which at the scale of a whole gather averages out the per-trace cycle
ambiguity. Higher bands are progressively unmasked on a schedule. Each new
band refines the warp locally as a *correction* on top of an already-converged
coarse warp, not as a replacement.

This is a schedule on the *auxiliary parameterization*, not on the data.
Data, wavelet, and forward modeling are unchanged. Connects to BACON-style
band-limited INRs (Lindell et al. 2022) and to coarse-to-fine NeRF training
(Park et al., Lin et al.).

### 15.2 Mechanism

Modify the encoder to accept per-band weights `w_b ∈ [0, 1]`:

    γ(ξ, τ) = [ξ, τ,
               w_1 · sin(2π ω_1 ξ), w_1 · cos(2π ω_1 ξ),
               w_1 · sin(2π ω_1 τ), w_1 · cos(2π ω_1 τ),
               …,
               w_B · sin(2π ω_B ξ), w_B · cos(2π ω_B ξ),
               w_B · sin(2π ω_B τ), w_B · cos(2π ω_B τ)]

The raw coordinate channels `ξ, τ` are never masked. Weights are stored as a
non-parameter buffer on the INR module.

Schedule (soft mode, default):

    frontier(k) = include_low_in_init + (B − include_low_in_init) · (k / K_sched)
    w_b(k) = clip((frontier(k) − b) / ramp_width, 0, 1)

with `k` the current iteration, `K_sched` the total schedule length, and
`ramp_width ∈ (0, B]` controlling how abruptly each band turns on. Hard mode
replaces the soft ramp with a step function.

Default schedule duration: the full `stage1_iter` from `WarpFWIConfig`, with
all bands on by the start of stage 2. Continuing the schedule into stage 2 is
exposed but not recommended for v1 experiments; stage 2 is where the warp is
supposed to give up alignment back to velocity, and artificially restricting
its capacity works against that.

Bit-exactness contract: with `SpectralScheduleConfig.enabled = False`, the
encoder skips the weighting multiplication entirely and produces outputs
bit-exact identical to the v1 encoder on a fixed seed. Unit-tested.

### 15.3 Config

New dataclass `SpectralScheduleConfig`:

- `enabled: bool = False`
- `mode: Literal['hard', 'soft'] = 'soft'`
- `ramp_width: float = 1.0` (in units of bands)
- `schedule_iters_stage1: int | None = None` (None → use `stage1_iter`)
- `schedule_iters_stage2: int | None = 0` (0 → fully unmasked by stage 2)
- `include_low_in_init: int = 1` (lowest N bands start fully on)

Attached to `INRConfig.spectral_schedule`.

### 15.4 Diagnostics

When enabled, logged per iteration:

- `spectral_frontier`: current frontier value in [0, B].
- `spectral_active_bands`: sum of current band weights (continuous measure).

End-of-stage: FFT of `τ(r, t)` along the time axis, cumulative spectral
energy per band — verifies the schedule actually band-limits the output.

### 15.5 Empirical result

Run on the same Marmousi patch as §13 with default schedule parameters
produced results qualitatively indistinguishable from the v1 baseline. The
stage-1 warp is slightly smoother in the receiver direction during the first
tens of iterations but the failure mode re-emerges as high bands unmask,
and the final inverted velocity is not meaningfully different. The schedule
delays the failure rather than preventing it.

Interpretation. The spectral schedule restricts *how* the warp can vary
across receivers early in training, but it does not constrain the gain
channel at all, and does not remove the optimizer's ability to find
amplitude-collapse solutions once high bands are available. The coarse warp
found during the low-band phase is informative, but stage 1 runs long enough
past full unmasking for the same pathologies to reappear. §16 addresses the
warp-side freedom more aggressively and is composable with §15.

## 16. Cumulative offset parameterization of the warp (Direction 3)

### 16.1 Motivation

The diagnosed failure (§14) has a specific geometric signature: the warp at
receiver `r+1` is free to be completely different from the warp at receiver
`r`, and the optimizer uses this freedom to produce sharp transitions
between an "aligned" near-offset region and a "suppressed" far-offset region.
Physically, this is unmotivated. Velocity errors translate into traveltime
errors that vary smoothly with offset along neighboring raypaths, so
`∂_r τ(r, t)` should be bounded in magnitude.

The v1 regularizer optionally includes a soft `α · ‖∂_r τ‖²` term. This
penalizes but does not prohibit sharp transitions, and empirically the
optimizer pays the penalty when the data-term reduction is worth it.

Direction 3 replaces the soft penalty with a *structural* bound. Parameterize
the warp via its derivative:

    τ(r, t) = τ_0(t) + ∫_{r' = 0}^{r} δ(r', t) dr'

with `δ` bounded in magnitude by a tanh. Then `|τ(r+1, t) − τ(r, t)| ≤
δ_max · Δr` as a hard invariant, for any representable network output. Sharp
receiver-direction jumps are removed from the parameter space, not merely
made expensive. This matches the design philosophy of §3 (tanh-bounded `τ`,
`MULTIPLICATIVE` gain): pathological states are unreachable, not just
penalized.

### 16.2 Mechanism

Discrete form, with receivers indexed `r = 0, …, R − 1`:

    δ[r, t]      = δ_max · tanh(δ_raw[r, t])
    τ_0[t]       = T_max · tanh(τ_0_raw[t])
    τ_inc[r, t]  = Σ_{r' = 0}^{r − 1} δ[r', t] · Δr_norm
    τ[r, t]      = τ_0[t] + τ_inc[r, t]
    τ[r, t]      = T_max · tanh(τ[r, t] / T_max)        # optional outer bound

The cumulative sum is `torch.cumsum` along the receiver axis — exact autograd
support, `O(R)` forward and backward. `τ_0_raw` is taken as the INR's
corresponding output at `r = 0` (simpler than defining a separate 1D network;
preserves the zero-init invariant).

Outer tanh: when `outer_tanh = True` (default), `|τ| ≤ T_max` is preserved as
in §3. When `False`, `τ` can grow with offset as far as the integrated `δ`
carries it; use only if diagnostics show the outer bound is binding.

Regularizer: with `WarpParam.CUMULATIVE` and `offset_smooth = True`, the
`α · ‖∂_r τ‖²` term is redirected to `α · ‖δ‖²` (they are equal up to
discretization, and penalizing both would double-count the tanh
nonlinearity). Time-axis smoothness and the gain-channel penalty are
unchanged.

### 16.3 INR output channels

The INR output width depends on the joint `(WarpParam, GainParam)` setting:

    DIRECT + NONE:           C = 1   (τ_raw)
    DIRECT + ADD/MULT:       C = 2   (τ_raw, gain_raw)
    CUMULATIVE + NONE:       C = 2   (δ_raw, τ_0_raw)
    CUMULATIVE + ADD/MULT:   C = 3   (δ_raw, τ_0_raw, gain_raw)

`INROutput` carries optional `tau_raw` and `delta_raw` fields; the consumer
(`warp.py`) dispatches on which is populated.

### 16.4 Config

New enum `WarpParam` with members `DIRECT` (v1 behavior) and `CUMULATIVE`.
New dataclass `CumulativeWarpConfig`:

- `delta_max_periods: float = 0.05`
  Per-receiver-step bound on `|δ|` in units of `1 / f_peak`. The physical
  reasoning: at 5 Hz with 60 receivers, a bound of 0.05 dominant periods per
  step gives `δ_max ≈ 10 ms` per receiver, enough to represent reasonable
  offset-dependent traveltime errors.
- `outer_tanh: bool = True`

Attached to `WarpConfig.warp_param` and `WarpConfig.cumulative`. Existing
runs are bit-exact unchanged at the new defaults (`WarpParam.DIRECT`).

### 16.5 Differentiability contract

- `WarpParam.DIRECT` path is bit-exact identical to v1 on a fixed seed.
- `WarpParam.CUMULATIVE` at zero-init (all raw outputs zero): reconstructed
  `τ ≡ 0` exactly, `W_θ[d] == d` bit-exactly.
- `WarpParam.CUMULATIVE` with constant `δ ≡ c`, `τ_0 ≡ 0`, `outer_tanh = False`:
  `τ[r, t] = c · Δr_norm · r` exactly (linear in `r`).
- `outer_tanh = True` stress test: reconstructed `τ` respects `|τ| ≤ T_max`
  for arbitrarily large `δ_raw`, `τ_0_raw`.

### 16.6 Diagnostics

When `WarpParam.CUMULATIVE` is active, logged per iteration:

- `delta_abs_mean`: `mean(|δ|)` (bounded values, not raw).
- `delta_saturation`: fraction of `|tanh(δ_raw)| > 0.95`. The primary
  indicator of whether `δ_max` is too tight; target < 10%.
- `tau_outer_saturation`: fraction of reconstructed `|τ|/T_max > 0.95`.
  If large, the outer tanh is binding and the warp would naturally grow
  beyond `T_max`.
- `tau_monotonicity`: for each `t`, fraction of `r` where `sign(τ[r+1, t] −
  τ[r, t])` agrees with the modal sign at that `t`. Scalar in `[0.5, 1]`.
  Near 1 indicates the velocity error has a consistent sign across offsets
  (potential case for a sign-restricted `δ` variant); near 0.5 indicates
  sign-changing `τ` and justifies free-signed `δ`.

### 16.7 Composability with §15

The spectral schedule and the cumulative parameterization are independent
and can be enabled together. Under `WarpParam.CUMULATIVE`, the spectral
schedule acts on the encoding bands that feed the MLP producing `δ_raw` and
`τ_0_raw`; the cumulative reconstruction happens downstream of the MLP. The
two features do not interact at the parameter level.

### 16.8 Relation to the failure mode

The pathological stage-1 configuration (§14) — aligned near offsets,
suppressed far offsets, sharp transition in between — requires `τ(r, t)` to
change rapidly across the transition receiver. The cumulative
parameterization caps this rate at `δ_max` per receiver step. If `δ_max` is
calibrated so that legitimate offset-dependent traveltime errors are within
reach but the pathological jump is not, the optimizer cannot find the
suppression solution even with a fully permissive gain channel. The expected
stage-1 warped-residual pattern shifts from "good near, suppressed far" to
"mediocre everywhere" — mediocre but informative, which is what stage 2
needs to proceed.
