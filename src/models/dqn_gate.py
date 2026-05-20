from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class DQNGate(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_assets: int,
        dueling: bool = True,
        output_dim: int = 2,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_assets = int(n_assets)
        self.dueling = bool(dueling)
        self.output_dim = int(output_dim)
        if self.latent_dim <= 0 or self.n_assets <= 0 or self.output_dim <= 0:
            raise ValueError("ERR_DQN_GATE_INVALID_CONFIG: latent_dim,n_assets,output_dim must be > 0")

        hidden = tuple(int(dim) for dim in (hidden_dims or (256, 128)))
        if not hidden or any(dim <= 0 for dim in hidden):
            raise ValueError("ERR_DQN_GATE_INVALID_CONFIG: hidden_dims must be positive")

        self.input_dim = self.latent_dim + 2 * self.n_assets + 2
        feature_layers: list[nn.Module] = []
        in_dim = self.input_dim
        for out_dim in hidden:
            feature_layers.extend([nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(float(dropout))])
            in_dim = out_dim
        self.feature_net = nn.Sequential(*feature_layers)
        feature_dim = hidden[-1]

        if self.dueling:
            self.value_net = nn.Linear(feature_dim, 1)
            self.advantage_net = nn.Linear(feature_dim, self.output_dim)
        else:
            self.q_net = nn.Linear(feature_dim, self.output_dim)

    def forward(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(latent, candidate_weights, current_weights, estimated_turnover, estimated_cost)
        x = torch.cat([latent, candidate_weights, current_weights, estimated_turnover, estimated_cost], dim=1)
        features = self.feature_net(x)
        if self.dueling:
            value = self.value_net(features)
            advantage = self.advantage_net(features)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.q_net(features)

    def select_action(
        self,
        q_values: torch.Tensor,
        q_gap_threshold: float = 0.0,
        previous_action: torch.Tensor | None = None,
        cooldown_mask: torch.Tensor | None = None,
        hysteresis_margin: float = 0.0,
    ) -> torch.Tensor:
        self._validate_q_values(q_values)
        raw_action = torch.argmax(q_values, dim=1)
        if q_values.shape[1] != 2:
            return raw_action

        q_gap = q_values[:, 1] - q_values[:, 0]
        action = torch.where(q_gap >= float(q_gap_threshold), torch.ones_like(raw_action), torch.zeros_like(raw_action))
        if previous_action is not None and hysteresis_margin > 0.0:
            if previous_action.shape != action.shape:
                raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: previous_action must be [batch]")
            previous_action = previous_action.to(device=q_values.device, dtype=torch.long)
            keep_rebalance = (previous_action == 1) & (q_gap >= -float(hysteresis_margin))
            keep_hold = (previous_action == 0) & (q_gap <= float(q_gap_threshold) + float(hysteresis_margin))
            action = torch.where(keep_rebalance, torch.ones_like(action), action)
            action = torch.where(keep_hold, torch.zeros_like(action), action)
        if cooldown_mask is not None:
            if cooldown_mask.shape != action.shape:
                raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: cooldown_mask must be [batch]")
            action = torch.where(cooldown_mask.to(device=q_values.device, dtype=torch.bool), torch.zeros_like(action), action)
        return action

    @staticmethod
    def compute_double_dqn_target(
        rewards: torch.Tensor,
        next_q_online: torch.Tensor,
        next_q_target: torch.Tensor,
        done: torch.Tensor,
        gamma: float = 0.99,
        n_steps: int = 1,
    ) -> torch.Tensor:
        _assert_column(rewards, "rewards")
        _assert_column(done, "done")
        if next_q_online.ndim != 2 or next_q_target.shape != next_q_online.shape:
            raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: next_q_online/next_q_target must share [batch,actions]")
        if rewards.shape[0] != next_q_online.shape[0] or done.shape[0] != next_q_online.shape[0]:
            raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: batch sizes must match")
        if not _all_finite(rewards, next_q_online, next_q_target, done):
            raise ValueError("ERR_DQN_GATE_NON_FINITE: target inputs contain NaN or Inf")
        if n_steps < 1:
            raise ValueError("ERR_DQN_GATE_INVALID_N_STEPS: n_steps must be >= 1")

        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
        next_values = next_q_target.gather(1, next_actions)
        bootstrap_mask = 1.0 - done.to(dtype=rewards.dtype)
        return rewards + bootstrap_mask * (float(gamma) ** int(n_steps)) * next_values

    @staticmethod
    def compute_n_step_returns(
        rewards: torch.Tensor,
        done: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        if rewards.ndim != 2 or done.shape != rewards.shape:
            raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: rewards/done must be [batch,n_steps]")
        _assert_column(next_value, "next_value")
        if next_value.shape[0] != rewards.shape[0]:
            raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: next_value batch must match rewards")
        if not _all_finite(rewards, done, next_value):
            raise ValueError("ERR_DQN_GATE_NON_FINITE: n-step inputs contain NaN or Inf")

        returns = torch.zeros(rewards.shape[0], 1, device=rewards.device, dtype=rewards.dtype)
        discount = torch.ones_like(returns)
        active = torch.ones_like(returns)
        for step in range(rewards.shape[1]):
            step_reward = rewards[:, step : step + 1]
            step_done = done[:, step : step + 1].to(dtype=rewards.dtype)
            returns = returns + active * discount * step_reward
            active = active * (1.0 - step_done)
            discount = discount * float(gamma)
        return returns + active * discount * next_value

    def _validate_inputs(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
    ) -> None:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: latent must be [batch,latent_dim]")
        batch_size = latent.shape[0]
        if candidate_weights.shape != (batch_size, self.n_assets):
            raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: candidate_weights must be [batch,n_assets]")
        if current_weights.shape != (batch_size, self.n_assets):
            raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: current_weights must be [batch,n_assets]")
        if estimated_turnover.shape != (batch_size, 1) or estimated_cost.shape != (batch_size, 1):
            raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: estimated_turnover/estimated_cost must be [batch,1]")
        if not _all_finite(latent, candidate_weights, current_weights, estimated_turnover, estimated_cost):
            raise ValueError("ERR_DQN_GATE_NON_FINITE: gate inputs contain NaN or Inf")

    def _validate_q_values(self, q_values: torch.Tensor) -> None:
        if q_values.ndim != 2 or q_values.shape[1] != self.output_dim:
            raise ValueError("ERR_DQN_GATE_SHAPE_MISMATCH: q_values must be [batch,output_dim]")
        if not torch.isfinite(q_values).all():
            raise ValueError("ERR_DQN_GATE_NON_FINITE: q_values contain NaN or Inf")


def _assert_column(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"ERR_DQN_GATE_SHAPE_MISMATCH: {name} must be [batch,1]")


def _all_finite(*values: torch.Tensor) -> bool:
    return all(torch.isfinite(value).all().item() for value in values)


__all__ = ["DQNGate"]
