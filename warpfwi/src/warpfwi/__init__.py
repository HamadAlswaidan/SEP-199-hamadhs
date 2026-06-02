"""Warp-FWI: local time-shift and gain auxiliary for cycle-skip mitigation in FWI.

See ``DESIGN.md`` at the repository root for the method specification, and
``SKILL.md`` for project conventions.
"""
from __future__ import annotations

from .config import GainParam

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "GainParam",
]
