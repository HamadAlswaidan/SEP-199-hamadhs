"""Configuration dataclasses for Warp-FWI.

Every hyperparameter lives on a dataclass here. The notebook constructs these
and passes them to library functions. Nothing in the library should rely on a
hardcoded default outside of this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

DevicePref = Literal["auto", "cuda", "cpu", "mps"]
ReceiverGeometry = Literal["surface"]
GeometryMode = Literal["fixed_spread", "moving_spread", "split_spread"]
XRoundingMode = Literal["round", "floor"]
Activation = Literal["gelu", "silu", "sine", "tanh"]
ScheduleKind = Literal["geometric", "linear", "constant"]
WarmStart = Literal["none", "near_offset_fwi", "lowfreq_fwi"]
MisfitName = Literal["l2", "envelope_l2"]
Stage2Routing = Literal["auto", "single", "dual"]
MethodName = Literal["local_shift", "monotone_ntw"]
AnnealSchedule = Literal["geometric"]
TaperKind = Literal["cosine", "hann"]
OptimizerName = Literal["adam", "lbfgs"]
WarpVelocityOptimizer = Literal["adam", "lbfgs_velocity_only"]


class WarpParam(str, Enum):
    """Parameterization of the time-shift field ``τ(r, t)``.

    * :attr:`DIRECT` (default) — The INR directly produces ``τ_raw`` at each
      ``(r, t)``. This is the v1 behavior and is bit-exact preserved.
    * :attr:`CUMULATIVE` — The INR produces an offset-derivative ``δ_raw`` and
      a baseline ``τ_0_raw`` at ``r = 0``. The warp is reconstructed by
      cumulative sum over the receiver axis,
      ``τ(r, t) = τ_0(t) + Σ_{r' < r} δ(r', t) · Δr_norm``. The per-step
      magnitude of ``δ`` is tanh-bounded by ``δ_max``, turning the soft
      ``α · ‖∂_r τ‖²`` penalty into a structural invariant. See DESIGN.md §16.
    """

    DIRECT = "direct"
    CUMULATIVE = "cumulative"


class GainParam(str, Enum):
    """Gain parameterization for the warp auxiliary.

    The warp operator has the general form ``W_θ[d](r, t) = g(r, t) · d(r, t − τ(r, t))``
    where ``g`` is a non-negative multiplicative gain. The INR always produces a
    ``τ_raw`` channel; depending on this enum it may also produce a second
    ``gain_raw`` channel that becomes the gain factor via a chosen map:

    * :attr:`NONE` — gain ≡ 1 (no gain channel, network is warp-only). Use when
      you want to study pure time-alignment without amplitude compensation.
    * :attr:`ADDITIVE` — gain = ``1 + gain_raw``. This is the original
      formulation from ``DESIGN.md`` v1. Known failure mode: the optimizer can
      drive ``gain_raw → −1`` so the warped synthetic is suppressed and the L²
      loss drops without producing a useful alignment. No finite ``β`` prevents
      this: the "delete the synthetic" state sits at finite regularizer cost.
    * :attr:`MULTIPLICATIVE` — gain = ``exp(gain_raw)``. The "do nothing" state
      is ``gain_raw = 0`` (matching the zero-init output head, so ``g = 1``
      exactly at init). The "delete the synthetic" state requires
      ``gain_raw → −∞``, which incurs infinite regularizer cost — structurally
      preventing the amplitude-collapse failure mode.

    The regularizer penalty is always computed on the **raw** output
    (``β · mean(gain_raw²)`` for both :attr:`ADDITIVE` and
    :attr:`MULTIPLICATIVE`). For :attr:`MULTIPLICATIVE` this is what makes the
    collapse mode diverge; penalizing ``‖exp(g) − 1‖²`` would not.
    """

    NONE = "none"
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


@dataclass
class DeviceConfig:
    """Device preference. See :func:`warpfwi.device.select_device`."""

    prefer: DevicePref = "auto"


@dataclass
class DataConfig:
    """Marmousi loading, cropping, and smoothing for the initial model.

    Parameters
    ----------
    marmousi_path:
        Path to a ``.npy`` or ``.bin`` Marmousi velocity file. If ``None`` or
        missing, :func:`warpfwi.data.load_velocity` falls back to a synthetic
        layered model so tests do not require the real data.
    crop:
        ``(z0, z1, x0, x1)`` in grid indices, applied to the full Marmousi
        array before resampling.
    dx:
        Grid spacing in meters along both axes.
    smooth_sigma:
        Standard deviation (in grid samples) of the Gaussian applied to the
        true model to produce the initial model ``v_0``. A large value gives a
        deliberately poor initial model that cycle-skips.
    v_min, v_max:
        Bounds on the velocity [m/s] used by the bounded-velocity sigmoid
        parameterization (see :mod:`warpfwi.velocity`).
    synthetic_shape:
        Output ``(nz, nx)`` for the synthetic-fallback model.
    """

    marmousi_path: str | None = None
    crop: tuple[int, int, int, int] = (0, 120, 0, 240)
    crop_origin_x: int = 0
    crop_origin_z: int = 0
    crop_nx: int | None = None
    crop_nz: int | None = None
    dx: float = 20.0
    dz: float | None = None
    smooth_sigma: float = 40.0
    v_min: float = 1500.0
    v_max: float = 5000.0
    synthetic_shape: tuple[int, int] = (80, 160)
    model_source: Literal["true", "smoothed", "user"] = "true"
    pad_or_crop: Literal["crop", "pad", "none"] = "crop"


@dataclass
class AcquisitionConfig:
    """Surface acquisition geometry and Ricker source wavelet.

    Parameters
    ----------
    n_shots:
        Number of shots evenly spaced along the surface.
    f_peak:
        Ricker peak frequency [Hz].
    dt:
        Temporal sampling interval [s].
    nt:
        Number of time samples per shot gather.
    receiver_geometry:
        Only ``"surface"`` is supported in v1.
    source_depth, receiver_depth:
        Depth in meters for source and receiver lines.
    n_receivers:
        Number of receivers per shot (independent of shot location; each shot
        sees the same receiver array).
    source_pad, receiver_pad:
        Distance in grid cells from the left/right edges before placing the
        first source / first receiver.
    """

    n_shots: int = 30
    f_peak: float = 5.0
    dt: float = 0.004
    nt: int = 750
    receiver_geometry: ReceiverGeometry = "surface"
    source_depth: float = 20.0
    receiver_depth: float = 20.0
    n_receivers: int = 120
    source_pad: int = 4
    receiver_pad: int = 4
    source_spacing_m: float | None = None
    receiver_spacing_m: float | None = None
    source_start_m: float | None = None
    receiver_start_m: float | None = None
    source_depth_m: float | None = None
    receiver_depth_m: float | None = None
    source_depth_cells: int | None = None
    receiver_depth_cells: int | None = None
    geometry_mode: GeometryMode = "fixed_spread"
    source_x_m_list: tuple[float, ...] | None = None
    receiver_x_m_list: tuple[float, ...] | None = None
    x_rounding_mode: XRoundingMode = "round"
    allow_partial_spread: bool = False
    device: str | None = None


@dataclass
class ModelingConfig:
    """Deepwave scalar modeling options shared by classical FWI and Warp-FWI."""

    dx: float | None = None
    dz: float | None = None
    dt: float | None = None
    pml_width: int = 20
    pml_freq: float | None = None
    accuracy: int = 4
    source_type: Literal["ricker"] = "ricker"
    device: str | None = None


@dataclass
class SpectralScheduleConfig:
    """Progressive spectral unmasking of the INR's Fourier-feature encoding.

    Coarse-to-fine schedule on the *auxiliary's representational capacity*.
    The encoder's per-band ``(sin, cos)`` channels are multiplied by a
    frontier-driven weight in ``[0, 1]``; the raw ``(ξ, τ)`` channels are
    never masked. With :attr:`enabled` = ``False`` (the default) the entire
    code path is a no-op and every existing result is bit-exact preserved.

    Parameters
    ----------
    enabled:
        Master switch. When ``False``, all other fields are ignored and the
        INR forward path is identical to the pre-feature behavior.
    mode:
        ``"hard"`` gates each band with a step at the frontier crossing;
        ``"soft"`` ramps each band linearly over :attr:`ramp_width` bands.
    ramp_width:
        In units of "bands". With soft mode and ``ramp_width = 1.0``, each
        band ramps from 0 to 1 over the iteration interval during which the
        frontier crosses it. Larger values give smoother transitions.
    schedule_iters_stage1:
        Iterations over which the frontier reaches the highest band in
        stage 1. ``None`` means use the full ``stage1_iter`` from
        :class:`WarpFWIConfig`.
    schedule_iters_stage2:
        Extra iterations (into stage 2) over which the frontier continues to
        climb. ``0`` (default) means all bands are already on by stage 2 start.
        Positive values are rare and not recommended for v1.
    include_low_in_init:
        Number of lowest bands that start fully on at iteration 0, in
        addition to the always-on raw coordinate channels.
    """

    enabled: bool = False
    mode: Literal["hard", "soft"] = "soft"
    ramp_width: float = 1.0
    schedule_iters_stage1: int | None = None
    schedule_iters_stage2: int | None = 0
    include_low_in_init: int = 1


@dataclass
class INRConfig:
    """Fourier-feature MLP architecture for the warp auxiliary.

    See :mod:`warpfwi.inr`. The output layer is zero-initialized so
    ``τ_raw = 0`` (and ``gain_raw = 0`` if present) exactly at init, and the
    warp operator reduces to the identity bit-exactly.

    Parameters
    ----------
    depth, width, activation, n_fourier_bands, omega_max:
        Standard MLP knobs plus Fourier-feature encoding frequencies.
    gain_param:
        :class:`GainParam` selecting the gain parameterization. The output
        head is sized as 1 channel for :attr:`GainParam.NONE` (warp only) and
        2 channels for :attr:`GainParam.ADDITIVE` / :attr:`GainParam.MULTIPLICATIVE`.
        Default is :attr:`GainParam.ADDITIVE` to preserve the original v1
        behavior; switch to :attr:`GainParam.MULTIPLICATIVE` to prevent the
        amplitude-collapse failure mode.
    spectral_schedule:
        :class:`SpectralScheduleConfig` controlling progressive spectral
        unmasking of the Fourier-feature bands. Disabled by default.
    """

    depth: int = 4
    width: int = 128
    activation: Activation = "gelu"
    n_fourier_bands: int = 8
    omega_max: float = 16.0
    gain_param: GainParam = GainParam.ADDITIVE
    warp_param: WarpParam = WarpParam.DIRECT
    spectral_schedule: SpectralScheduleConfig = field(
        default_factory=SpectralScheduleConfig
    )


@dataclass
class CumulativeWarpConfig:
    """Cumulative-offset parameterization of ``τ(r, t)`` (DESIGN.md §16).

    Only consulted when :attr:`WarpConfig.warp_param` is
    :attr:`WarpParam.CUMULATIVE`.

    Parameters
    ----------
    delta_max_periods:
        Per-receiver-step bound on ``|δ|`` in units of ``1 / f_peak``. With
        60 receivers and ``delta_max_periods = 0.05`` at 5 Hz, the total
        reachable ``τ`` across the gather is ~``60 · 0.05 · 0.2 s = 0.6 s``,
        which is well above ``T_max`` and will be re-bounded if
        :attr:`outer_tanh` is ``True``.
    outer_tanh:
        If ``True`` (default), re-bound the reconstructed ``τ`` via
        ``T_max · tanh(τ / T_max)``. Preserves ``|τ| ≤ T_max``. Set to
        ``False`` only after diagnosing that the outer bound is binding.
    """

    delta_max_periods: float = 0.05
    outer_tanh: bool = True


@dataclass
class WarpConfig:
    """Warp-and-gain operator and regularizer weights.

    Parameters
    ----------
    T_max_periods:
        ``T_max = T_max_periods / f_peak`` (seconds). Structural bound on
        ``|τ|`` applied via ``T_max · tanh(τ_raw)``.
    alpha_smooth:
        Coefficient on ``‖∂_t τ‖²`` in the regularizer ``R``.
    beta_gain:
        Coefficient on ``‖a‖²`` in the regularizer ``R``.
    offset_smooth:
        If ``True`` also add ``‖∂_r τ‖²`` to ``R``. Disabled by default in v1.
    lambda_0, lambda_final:
        Endpoints of the λ schedule used in stage 2.
    schedule:
        ``"geometric"`` (default), ``"linear"``, or ``"constant"``.
    warp_param:
        :class:`WarpParam` selecting the time-shift parameterization.
        Default :attr:`WarpParam.DIRECT` reproduces v1 behavior bit-exactly.
    cumulative:
        :class:`CumulativeWarpConfig` consumed when ``warp_param`` is
        :attr:`WarpParam.CUMULATIVE`.
    """

    T_max_periods: float = 0.4
    alpha_smooth: float = 1.0
    beta_gain: float = 1.0
    offset_smooth: bool = False
    lambda_0: float = 1.0
    lambda_final: float = 20.0
    schedule: ScheduleKind = "geometric"
    warp_param: WarpParam = WarpParam.DIRECT
    cumulative: CumulativeWarpConfig = field(default_factory=CumulativeWarpConfig)


@dataclass
class OptimConfig:
    """Optimization knobs shared across FWI variants."""

    name: OptimizerName = "adam"
    lr_v: float = 0.1
    lr_theta: float = 1e-3
    batch_size: int = 4
    grad_clip: float | None = 1.0
    lr: float | None = None
    max_iter: int = 1
    history_size: int = 10
    line_search_fn: str | None = "strong_wolfe"
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    max_eval: int | None = None
    gradient_clip: float | None = None
    use_full_batch_for_lbfgs: bool = True
    log_closure_evals: bool = True


@dataclass
class OptimizerConfig:
    """Generic optimizer config for explicit Adam/L-BFGS selection."""

    name: OptimizerName = "adam"
    lr: float = 0.1
    max_iter: int = 1
    history_size: int = 10
    line_search_fn: str | None = "strong_wolfe"
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    max_eval: int | None = None
    gradient_clip: float | None = 1.0
    use_full_batch_for_lbfgs: bool = True
    log_closure_evals: bool = True


@dataclass
class FWIPreprocessingConfig:
    """Reusable preprocessing options for velocity updates and data misfit.

    Disabled defaults preserve the original FWI behavior.
    """

    gradient_taper_enabled: bool = False
    gradient_taper_width_cells: int = 0
    gradient_taper_type: TaperKind = "cosine"
    bandpass_enabled: bool = False
    bandpass_fmin_hz: float = 2.0
    bandpass_fmax_hz: float = 8.0
    bandpass_order: int = 4
    filter_source_wavelet_for_fwi_band: bool = False


@dataclass
class ClassicalFWIConfig:
    """Classical L2-FWI baseline config."""

    n_iter: int = 200
    optim: OptimConfig = field(default_factory=OptimConfig)
    optimizer: OptimizerConfig | None = None
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    preprocess: FWIPreprocessingConfig = field(default_factory=FWIPreprocessingConfig)
    seed: int = 0


@dataclass
class WarpFWIConfig:
    """Two-stage Warp-FWI config.

    Parameters
    ----------
    stage1_iter:
        Number of pretraining iterations per shot during stage 1.
    stage2_iter:
        Number of outer iterations in stage 2.
    lambda_pre:
        λ used during stage-1 pretraining (kept small but nonzero to prevent
        the warp from perfectly fitting the residual).
    stage1_misfit:
        Data misfit used to pretrain the per-shot warp/alignment INR.
    stage2_theta_misfit:
        Data misfit for the stage-2 theta/alignment branch when dual routing
        is active.
    stage2_v_misfit:
        Data misfit for the stage-2 velocity/inversion branch when dual
        routing is active.
    stage2_routing:
        ``"auto"`` keeps the original single-objective L2 path when both
        stage-2 misfits are L2 and switches to dual gradient routing otherwise.
        ``"single"`` forces the original L2 objective; ``"dual"`` forces the
        routed two-branch objective.
    envelope_eps:
        Small positive constant used by envelope misfits.
    optim:
        Shared optimization knobs.
    warm_start:
        One of ``"none"``, ``"near_offset_fwi"``, ``"lowfreq_fwi"`` (see
        :doc:`DESIGN` §10).
    n_iter_warmup, n_offsets_warmup, f_lp_warmup:
        Parameters for the warm-start variants.
    seed:
        PRNG seed consumed by :mod:`warpfwi.fwi_warp` at function entry.
    """

    stage1_iter: int = 200
    stage2_iter: int = 200
    lambda_pre: float = 0.1
    stage1_misfit: MisfitName = "l2"
    stage2_theta_misfit: MisfitName = "l2"
    stage2_v_misfit: MisfitName = "l2"
    stage2_routing: Stage2Routing = "auto"
    envelope_eps: float = 1e-8
    optim: OptimConfig = field(default_factory=OptimConfig)
    velocity_optimizer: WarpVelocityOptimizer = "adam"
    theta_optimizer: OptimizerName = "adam"
    theta_step_after_velocity_lbfgs: bool = False
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    preprocess: FWIPreprocessingConfig = field(default_factory=FWIPreprocessingConfig)
    warm_start: WarmStart = "none"
    n_iter_warmup: int = 30
    n_offsets_warmup: int = 32
    f_lp_warmup: float = 4.0
    seed: int = 0


@dataclass
class MonotoneAnnealConfig:
    """Internal data-frequency annealing for monotone NTW losses."""

    enabled: bool = True
    schedule: AnnealSchedule = "geometric"
    f_min: float = 2.0
    f_max: float = 8.0
    order: float = 8.0


@dataclass
class MonotoneModelConfig:
    """Per-shot INR architecture for tracewise monotone time-warp increments."""

    depth: int = 4
    width: int = 128
    activation: Activation = "gelu"
    n_fourier_bands: int = 8
    omega_max: float = 16.0
    increment_eps: float = 0.0


@dataclass
class MonotoneRegularizationConfig:
    """Regularization weights applied directly to ``a_raw``."""

    identity: float = 1.0
    time_smooth: float = 1.0
    receiver_smooth: float = 1.0
    lambda_0: float = 0.1
    lambda_final: float = 1.0
    schedule: ScheduleKind = "geometric"


@dataclass
class MonotoneNTWConfig:
    """Stage-1/stage-2 config for tracewise monotone NTW-style warp-FWI."""

    stage1_iter: int = 200
    stage2_iter: int = 200
    run_stage2: bool = False
    anneal: MonotoneAnnealConfig = field(default_factory=MonotoneAnnealConfig)
    model: MonotoneModelConfig = field(default_factory=MonotoneModelConfig)
    reg: MonotoneRegularizationConfig = field(
        default_factory=MonotoneRegularizationConfig
    )
    optim: OptimConfig = field(default_factory=OptimConfig)
    seed: int = 0


@dataclass
class Config:
    """Top-level container grouping every sub-config for notebook convenience."""

    method: MethodName = "local_shift"
    device: DeviceConfig = field(default_factory=DeviceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    inr: INRConfig = field(default_factory=INRConfig)
    warp: WarpConfig = field(default_factory=WarpConfig)
    classical: ClassicalFWIConfig = field(default_factory=ClassicalFWIConfig)
    warpfwi: WarpFWIConfig = field(default_factory=WarpFWIConfig)
    monotone_ntw: MonotoneNTWConfig = field(default_factory=MonotoneNTWConfig)


__all__ = [
    "DevicePref",
    "ReceiverGeometry",
    "GeometryMode",
    "XRoundingMode",
    "Activation",
    "ScheduleKind",
    "WarmStart",
    "MisfitName",
    "Stage2Routing",
    "MethodName",
    "AnnealSchedule",
    "OptimizerName",
    "WarpVelocityOptimizer",
    "GainParam",
    "WarpParam",
    "DeviceConfig",
    "DataConfig",
    "AcquisitionConfig",
    "ModelingConfig",
    "SpectralScheduleConfig",
    "INRConfig",
    "CumulativeWarpConfig",
    "WarpConfig",
    "OptimConfig",
    "OptimizerConfig",
    "FWIPreprocessingConfig",
    "ClassicalFWIConfig",
    "WarpFWIConfig",
    "MonotoneAnnealConfig",
    "MonotoneModelConfig",
    "MonotoneRegularizationConfig",
    "MonotoneNTWConfig",
    "Config",
]
