"""Small optimizer helpers for FWI loops."""
from __future__ import annotations

from collections.abc import Iterable

import torch

from .config import OptimConfig, OptimizerConfig


def optimizer_from_legacy(cfg: OptimConfig) -> OptimizerConfig:
    """Convert the historical ``OptimConfig`` into explicit optimizer config."""
    lr = cfg.lr if cfg.lr is not None else cfg.lr_v
    grad_clip = cfg.gradient_clip if cfg.gradient_clip is not None else cfg.grad_clip
    return OptimizerConfig(
        name=cfg.name,
        lr=float(lr),
        max_iter=int(cfg.max_iter),
        history_size=int(cfg.history_size),
        line_search_fn=cfg.line_search_fn,
        tolerance_grad=float(cfg.tolerance_grad),
        tolerance_change=float(cfg.tolerance_change),
        max_eval=cfg.max_eval,
        gradient_clip=grad_clip,
        use_full_batch_for_lbfgs=bool(cfg.use_full_batch_for_lbfgs),
        log_closure_evals=bool(cfg.log_closure_evals),
    )


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    cfg: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Build Adam or PyTorch L-BFGS from a transparent config."""
    params = list(parameters)
    name = str(cfg.name).lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=float(cfg.lr))
    if name == "lbfgs":
        return torch.optim.LBFGS(
            params,
            lr=float(cfg.lr),
            max_iter=int(cfg.max_iter),
            max_eval=cfg.max_eval,
            history_size=int(cfg.history_size),
            line_search_fn=cfg.line_search_fn,
            tolerance_grad=float(cfg.tolerance_grad),
            tolerance_change=float(cfg.tolerance_change),
        )
    raise ValueError(f"optimizer name must be 'adam' or 'lbfgs', got {cfg.name!r}")


__all__ = ["build_optimizer", "optimizer_from_legacy"]
