from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOCritic(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Sequence[int] | None = None):
        super().__init__()
        self.latent_dim = int(latent_dim)
        dims = [self.latent_dim, *(hidden_dims or (256, 128))]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.extend([nn.Linear(int(in_dim), int(out_dim)), nn.GELU()])
        layers.append(nn.Linear(int(dims[-1]), 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("ERR_CRITIC_SHAPE_MISMATCH: latent must be [batch,latent_dim]")
        if not torch.isfinite(latent).all():
            raise ValueError("ERR_CRITIC_INPUT_NON_FINITE: latent contains NaN or Inf")
        return self.fc(latent)

    @staticmethod
    def clipped_value_loss(
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        clip_range: float = 0.20,
    ) -> torch.Tensor:
        _assert_value_shape(values, "values")
        if old_values.shape != values.shape or returns.shape != values.shape:
            raise ValueError("ERR_CRITIC_SHAPE_MISMATCH: values, old_values and returns must share shape")
        if not torch.isfinite(values).all() or not torch.isfinite(old_values).all() or not torch.isfinite(returns).all():
            raise ValueError("ERR_CRITIC_INPUT_NON_FINITE: value loss inputs contain NaN or Inf")
        if clip_range < 0.0:
            raise ValueError("ERR_CRITIC_INVALID_CLIP_RANGE: clip_range must be >= 0")

        clipped_values = old_values + (values - old_values).clamp(-float(clip_range), float(clip_range))
        unclipped_loss = F.mse_loss(values, returns, reduction="none")
        clipped_loss = F.mse_loss(clipped_values, returns, reduction="none")
        return 0.5 * torch.max(unclipped_loss, clipped_loss).mean()


def _assert_value_shape(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"ERR_CRITIC_SHAPE_MISMATCH: {name} must be [batch,1]")


__all__ = ["PPOCritic"]
