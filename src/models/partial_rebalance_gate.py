from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


BETA_MIN_CONCENTRATION = 1.0e-3
RHO_EPS = 1.0e-6
DEFAULT_DISCRETE_RHO_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


class BetaIntensityActor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        min_concentration: float = BETA_MIN_CONCENTRATION,
        hidden_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.beta_min_concentration = float(min_concentration)
        if not math.isfinite(self.beta_min_concentration) or self.beta_min_concentration <= 0.0:
            raise ValueError("ERR_BETA_MIN_CONCENTRATION_INVALID: min_concentration must be > 0")

        dims = [self.latent_dim, *(hidden_dims or (128,))]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.extend([nn.Linear(int(in_dim), int(out_dim)), nn.GELU()])
        layers.append(nn.Linear(int(dims[-1]), 2))
        self.param_net = nn.Sequential(*layers)

    def concentration(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("ERR_POLICY_SHAPE_MISMATCH: latent must be [batch,latent_dim]")
        raw_params = self.param_net(latent)
        params = F.softplus(raw_params) + self.beta_min_concentration
        if not torch.isfinite(params).all():
            raise ValueError("ERR_BETA_CONCENTRATION_NON_FINITE: Beta concentration contains NaN or Inf")
        return params[:, 0:1], params[:, 1:2]

    def get_distribution(self, latent: torch.Tensor) -> Beta:
        concentration1, concentration0 = self.concentration(latent)
        return Beta(concentration1, concentration0)

    def sample_with_log_prob(
        self,
        latent: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(latent)
        if deterministic:
            rho = dist.mean
            log_prob = torch.zeros_like(rho)
            return rho.clamp(0.0, 1.0), log_prob

        raw_rho = dist.rsample()
        log_prob = dist.log_prob(raw_rho)
        rho = raw_rho.clamp(RHO_EPS, 1.0 - RHO_EPS)
        return rho, log_prob

    def log_prob(self, latent: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("ERR_POLICY_SHAPE_MISMATCH: latent must be [batch,latent_dim]")
        if value.ndim != 2 or value.shape != (latent.shape[0], 1):
            raise ValueError("ERR_POLICY_SHAPE_MISMATCH: value must be [batch,1]")
        if not torch.isfinite(value).all():
            raise ValueError("ERR_ACTION_NON_FINITE: rebalance intensity contains NaN or Inf")
        if ((value < 0.0) | (value > 1.0)).any():
            raise ValueError("ERR_ACTION_INVALID_INTENSITY: rebalance intensity must be in [0,1]")
        safe_value = value.clamp(RHO_EPS, 1.0 - RHO_EPS)
        log_prob = self.get_distribution(latent).log_prob(safe_value)
        if not torch.isfinite(log_prob).all():
            raise ValueError("ERR_BETA_LOG_PROB_NON_FINITE: Beta log_prob contains NaN or Inf")
        return log_prob

    def forward(self, latent: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        rho, _ = self.sample_with_log_prob(latent, deterministic=deterministic)
        return rho

class DiscretePartialGate(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_assets: int,
        rho_values: Sequence[float] | None = None,
    ):
        super().__init__()
        resolved_rho_values = _validate_rho_values(
            DEFAULT_DISCRETE_RHO_VALUES if rho_values is None else rho_values
        )
        self.register_buffer("rho_values", torch.tensor(resolved_rho_values, dtype=torch.float32))
        self.n_rho = len(resolved_rho_values)

        from .dqn_gate import DQNGate

        self.gate = DQNGate(latent_dim, n_assets, dueling=True, output_dim=self.n_rho)

    def forward(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_values = self.gate(latent, candidate_weights, current_weights, estimated_turnover, estimated_cost)
        action_idx = torch.argmax(q_values, dim=1)
        return self.rho_values[action_idx].view(-1, 1), q_values


def _validate_rho_values(rho_values: Sequence[float]) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in rho_values)
    if not resolved:
        raise ValueError("ERR_PARTIAL_RHO_VALUES_INVALID: discrete_rho_values must be non-empty")
    if any((not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in resolved):
        raise ValueError("ERR_PARTIAL_RHO_VALUES_INVALID: discrete_rho_values must be within [0,1]")
    return resolved


__all__ = ["BetaIntensityActor", "DiscretePartialGate", "DEFAULT_DISCRETE_RHO_VALUES"]
