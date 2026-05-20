from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from src.data.leakage_checks import assert_decision_visibility_contract

from .dqn_gate import DQNGate
from .ppo_actor import PPOActor


UNCERTAINTY_GATE_INPUT_FIELDS = (
    "weight_uncertainty",
    "q_uncertainty",
    "uncertainty_features",
)


class MultiHeadUncertaintyHeads(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_assets: int,
        n_heads: int = 5,
        actor_hidden_dims: Sequence[int] | None = None,
        gate_hidden_dims: Sequence[int] | None = None,
        gate_output_dim: int = 2,
        dueling: bool = True,
        dropout: float = 0.10,
        min_alpha: float = 1.0e-3,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_assets = int(n_assets)
        self.n_heads = int(n_heads)
        self.gate_output_dim = int(gate_output_dim)
        if self.latent_dim <= 0 or self.n_assets <= 0 or self.n_heads <= 0 or self.gate_output_dim <= 0:
            raise ValueError("ERR_UNCERTAINTY_CONFIG_INVALID: dimensions and n_heads must be > 0")

        self.actor_heads = nn.ModuleList(
            [
                PPOActor(
                    self.latent_dim,
                    self.n_assets,
                    min_alpha=float(min_alpha),
                    hidden_dims=actor_hidden_dims,
                )
                for _ in range(self.n_heads)
            ]
        )
        self.gate_heads = nn.ModuleList(
            [
                DQNGate(
                    self.latent_dim,
                    self.n_assets,
                    dueling=dueling,
                    output_dim=self.gate_output_dim,
                    hidden_dims=gate_hidden_dims,
                    dropout=float(dropout),
                )
                for _ in range(self.n_heads)
            ]
        )

    def forward(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
        deterministic: bool = False,
        candidate_weights_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        candidate_samples = []
        q_samples = []
        for actor, gate in zip(self.actor_heads, self.gate_heads, strict=True):
            if candidate_weights_override is None:
                candidate_weights = actor(latent, mask, deterministic=deterministic)
            else:
                candidate_weights = candidate_weights_override.to(device=latent.device, dtype=latent.dtype)
            gate_q = gate(latent, candidate_weights, current_weights, estimated_turnover, estimated_cost)
            candidate_samples.append(candidate_weights)
            q_samples.append(gate_q)
        return summarize_uncertainty(candidate_samples, q_samples)

    def log_prob(self, latent: torch.Tensor, mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        log_probs = [actor.log_prob(latent, mask, value) for actor in self.actor_heads]
        return torch.stack(log_probs, dim=0).mean(dim=0)


def summarize_uncertainty(
    candidate_samples: Sequence[torch.Tensor] | torch.Tensor,
    q_samples: Sequence[torch.Tensor] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    candidate_stack = _stack_samples(candidate_samples, "candidate_samples")
    q_stack = _stack_samples(q_samples, "q_samples")
    if candidate_stack.shape[0] != q_stack.shape[0] or candidate_stack.shape[1] != q_stack.shape[1]:
        raise ValueError("ERR_UNCERTAINTY_SHAPE_MISMATCH: sample and batch dimensions must match")
    if candidate_stack.ndim != 3 or q_stack.ndim != 3:
        raise ValueError("ERR_UNCERTAINTY_SHAPE_MISMATCH: samples must be [n_samples,batch,dim]")

    mean_candidate_weights = candidate_stack.mean(dim=0)
    candidate_weight_variance = candidate_stack.var(dim=0, unbiased=False)
    mean_gate_q = q_stack.mean(dim=0)
    q_uncertainty = q_stack.var(dim=0, unbiased=False)
    uncertainty_features = build_uncertainty_features(candidate_weight_variance, q_uncertainty)
    return {
        "mean_candidate_weights": mean_candidate_weights,
        "candidate_weight_variance": candidate_weight_variance,
        "weight_uncertainty": candidate_weight_variance,
        "mean_gate_q": mean_gate_q,
        "q_uncertainty": q_uncertainty,
        "uncertainty_features": uncertainty_features,
    }


def build_uncertainty_features(weight_uncertainty: torch.Tensor, q_uncertainty: torch.Tensor) -> torch.Tensor:
    if weight_uncertainty.ndim != 2 or q_uncertainty.ndim != 2:
        raise ValueError("ERR_UNCERTAINTY_SHAPE_MISMATCH: uncertainty tensors must be [batch,dim]")
    if weight_uncertainty.shape[0] != q_uncertainty.shape[0]:
        raise ValueError("ERR_UNCERTAINTY_SHAPE_MISMATCH: batch dimensions must match")
    if not torch.isfinite(weight_uncertainty).all() or not torch.isfinite(q_uncertainty).all():
        raise ValueError("ERR_UNCERTAINTY_NON_FINITE: uncertainty tensors contain NaN or Inf")
    return torch.cat([weight_uncertainty, q_uncertainty], dim=1)


def uncertainty_gate_input_fields() -> tuple[str, ...]:
    assert_decision_visibility_contract(gate_input=UNCERTAINTY_GATE_INPUT_FIELDS)
    return UNCERTAINTY_GATE_INPUT_FIELDS


def _stack_samples(samples: Sequence[torch.Tensor] | torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(samples, torch.Tensor):
        stacked = samples
    else:
        if not samples:
            raise ValueError(f"ERR_UNCERTAINTY_SHAPE_MISMATCH: {name} is empty")
        stacked = torch.stack(tuple(samples), dim=0)
    if not torch.isfinite(stacked).all():
        raise ValueError(f"ERR_UNCERTAINTY_NON_FINITE: {name} contains NaN or Inf")
    return stacked


__all__ = [
    "MultiHeadUncertaintyHeads",
    "summarize_uncertainty",
    "build_uncertainty_features",
    "uncertainty_gate_input_fields",
]
