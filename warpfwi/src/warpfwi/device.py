"""Device selection for Warp-FWI.

A single :class:`torch.device` is selected once per run via :func:`select_device`
and passed explicitly into every function that allocates tensors. The selection
never silently falls back: requesting an unsupported device raises an error.
"""
from __future__ import annotations

from typing import Literal

import torch

DevicePreference = Literal["auto", "cuda", "cpu", "mps"]


def select_device(prefer: DevicePreference = "auto") -> torch.device:
    """Select a :class:`torch.device` for the current run.

    Parameters
    ----------
    prefer:
        One of ``"auto"``, ``"cuda"``, ``"cpu"``, ``"mps"``.

        * ``"auto"`` — return ``cuda`` if available, else ``cpu``.
        * ``"cuda"`` — return ``cuda``; raise :class:`RuntimeError` if unavailable.
        * ``"cpu"`` — return ``cpu``.
        * ``"mps"`` — always raise :class:`RuntimeError`. Deepwave's scalar
          solver does not support Apple Metal.

    Returns
    -------
    torch.device
        The resolved device. Callers must pass this device explicitly into any
        function that allocates tensors; functions must never infer the device
        from an input tensor.

    Raises
    ------
    RuntimeError
        If ``prefer="cuda"`` and CUDA is unavailable, or if ``prefer="mps"``.
    ValueError
        If ``prefer`` is not one of the accepted literals.
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested via prefer='cuda' but is not available in this "
                "PyTorch build. Install a CUDA-enabled PyTorch or use prefer='cpu'."
            )
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps":
        raise RuntimeError(
            "MPS is not supported for Warp-FWI. Deepwave's scalar wave solver "
            "runs on CPU or CUDA only; Apple Metal has no compatible kernel. "
            "Use prefer='cpu' on Apple silicon, or prefer='cuda' on an NVIDIA host."
        )
    raise ValueError(f"Unknown device preference {prefer!r}")


__all__ = ["DevicePreference", "select_device"]
