"""λ schedules for stage-2 Warp-FWI.

Three kinds supported:

* ``"geometric"``: ``λ_k = λ_0 · (λ_final / λ_0) ** (k / (K − 1))``
* ``"linear"``: ``λ_k = λ_0 + (λ_final − λ_0) · k / (K − 1)``
* ``"constant"``: ``λ_k = λ_0`` for all ``k``
"""
from __future__ import annotations

from enum import Enum


class LambdaSchedule(str, Enum):
    """Enumeration of available schedule kinds, matching ``ScheduleKind`` strings."""

    GEOMETRIC = "geometric"
    LINEAR = "linear"
    CONSTANT = "constant"


def lambda_at(
    k: int,
    K: int,
    lambda_0: float,
    lambda_final: float,
    schedule: str | LambdaSchedule = LambdaSchedule.GEOMETRIC,
) -> float:
    """Evaluate the λ schedule at iteration ``k``.

    Parameters
    ----------
    k:
        Current iteration, ``0 <= k < K``.
    K:
        Total number of iterations. Must be ``>= 1``.
    lambda_0:
        Starting λ.
    lambda_final:
        Ending λ (ignored if schedule is ``"constant"``).
    schedule:
        ``"geometric"`` (default), ``"linear"``, or ``"constant"``.

    Returns
    -------
    float
        The scheduled λ value.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if not (0 <= k < K):
        raise ValueError(f"k={k} outside [0, {K})")

    kind = LambdaSchedule(schedule) if not isinstance(schedule, LambdaSchedule) else schedule

    if kind is LambdaSchedule.CONSTANT:
        return float(lambda_0)
    if K == 1:
        return float(lambda_final)
    t = k / (K - 1)
    if kind is LambdaSchedule.LINEAR:
        return float(lambda_0 + (lambda_final - lambda_0) * t)
    if kind is LambdaSchedule.GEOMETRIC:
        if lambda_0 <= 0.0 or lambda_final <= 0.0:
            raise ValueError(
                f"Geometric schedule requires positive endpoints, got "
                f"lambda_0={lambda_0}, lambda_final={lambda_final}"
            )
        return float(lambda_0 * (lambda_final / lambda_0) ** t)
    raise ValueError(f"Unknown schedule {schedule!r}")


__all__ = ["LambdaSchedule", "lambda_at"]
