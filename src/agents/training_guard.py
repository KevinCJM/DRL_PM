from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn


LOGGER = logging.getLogger(__name__)


def assert_finite_tensor(value: torch.Tensor, label: str, error_code: str) -> None:
    if torch.isfinite(value).all().item():
        return
    LOGGER.error("%s: %s contains NaN or Inf", error_code, label)
    raise ValueError(f"{error_code}: {label} contains NaN or Inf")


def assert_finite_losses(losses: Mapping[str, torch.Tensor], context: str) -> None:
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            assert_finite_tensor(value, f"{context}.{key}", "ERR_TRAINING_NON_FINITE_LOSS")


def clip_grad_norm_checked(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
    context: str,
) -> torch.Tensor:
    params = list(parameters)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    assert_finite_tensor(grad_norm, f"{context}.grad_norm", "ERR_TRAINING_NON_FINITE_GRAD")
    return grad_norm


__all__ = ["assert_finite_losses", "assert_finite_tensor", "clip_grad_norm_checked"]
