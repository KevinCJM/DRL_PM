from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .dqn_gated_multitask_cnn_ppo import FullGatedModel
from .ppo_actor import PPOActor


class OTarCQRCritic(nn.Module):
    """Action-aware Quantile Regression critic for CQR gate.

    Input: latent || pre_trade_drifted_weights || action_portfolio_weights
    Output: [batch, n_quantiles] return-to-go quantiles.
    """

    def __init__(self, latent_dim: int, n_assets: int, n_quantiles: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_assets = int(n_assets)
        self.n_quantiles = int(n_quantiles)
        input_dim = self.latent_dim + self.n_assets + self.n_assets
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.n_quantiles),
        )

    def forward(
        self,
        latent: torch.Tensor,
        pre_trade_drifted_weights: torch.Tensor,
        action_portfolio_weights: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([latent, pre_trade_drifted_weights, action_portfolio_weights], dim=-1)
        return self.fc(x)

    @staticmethod
    def expected_value(quantiles: torch.Tensor) -> torch.Tensor:
        return quantiles.mean(dim=1, keepdim=True)

    @staticmethod
    def lower_tail_loss(quantiles: torch.Tensor, tail_alpha: float) -> torch.Tensor:
        n_quantiles = quantiles.shape[1]
        n_below = max(1, math.ceil(tail_alpha * n_quantiles))
        sorted_q, _ = torch.sort(quantiles, dim=1)
        lower_tail = sorted_q[:, :n_below]
        return torch.clamp(-lower_tail.mean(dim=1, keepdim=True), min=0.0)


class OTarCQRGate(FullGatedModel):
    """OTAR CQR Gate model — action-aware quantile gate on top of FullGatedModel."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)

        cqr_config = _mapping(config.get("cqr"))
        model_config = _mapping(config.get("model"))
        ppo_config = _mapping(config.get("ppo") or model_config.get("ppo"))
        reward_config = _mapping(config.get("reward"))

        n_quantiles = int(cqr_config.get("n_quantiles", 51))

        self.cqr_critic = OTarCQRCritic(
            latent_dim=self.effective_latent_dim,
            n_assets=self.n_assets,
            n_quantiles=n_quantiles,
        )
        self.target_cqr_critic = OTarCQRCritic(
            latent_dim=self.effective_latent_dim,
            n_assets=self.n_assets,
            n_quantiles=n_quantiles,
        )
        self.target_cqr_critic.load_state_dict(self.cqr_critic.state_dict())
        for p in self.target_cqr_critic.parameters():
            p.requires_grad = False

        ppo_hidden_dims = ppo_config.get("hidden_dims")
        self.target_actor = PPOActor(
            self.effective_latent_dim,
            self.n_assets,
            min_alpha=self.actor.min_alpha,
            max_alpha=self.actor.max_alpha,
            hidden_dims=ppo_config.get("actor_hidden_dims", ppo_hidden_dims),
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        for p in self.target_actor.parameters():
            p.requires_grad = False

        self.gate_margin = float(cqr_config.get("gate_margin", 0.0))
        self.lambda_tail = float(reward_config.get("lambda_tail", 0.10))
        self.target_update_interval = int(cqr_config.get("target_update_interval", 100))
        self.gamma = float(cqr_config.get("gate_gamma", 0.99))
        self.quantile_huber_kappa = float(cqr_config.get("quantile_huber_kappa", 1.0))
        self.quantile_tail_enabled = bool(cqr_config.get("quantile_tail_enabled", True))
        self.step_counter = 0
        self.n_quantiles = n_quantiles
        self.confidence_q = float(reward_config.get("confidence_q", 0.95))
        self.tail_alpha = max(1.0 - self.confidence_q, 1.0e-6)

    def encode_latent_from_observation(
        self,
        observation: Mapping[str, Any],
        risk_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        market_image = observation["market_image"]
        if not isinstance(market_image, torch.Tensor):
            market_image = torch.as_tensor(np.asarray(market_image), dtype=torch.float32)
        if market_image.ndim == 3:
            market_image = market_image.unsqueeze(0)
        market_image = market_image.to(device=self._device, dtype=torch.float32)

        latent = self.encoder(market_image)

        if self.use_risk_state:
            if risk_state is None:
                rs = observation.get("risk_state")
                if rs is not None:
                    risk_state = torch.as_tensor(np.asarray(rs), dtype=torch.float32)
            if risk_state is None:
                raise ValueError("ERR_OBSERVATION_RISK_STATE_MISSING: use_risk_state=True but risk_state not provided")
            if risk_state.ndim == 1:
                risk_state = risk_state.unsqueeze(0)
            risk_state = risk_state.to(device=self._device, dtype=torch.float32)
            latent = torch.cat([latent, risk_state], dim=-1)

        return latent

    def propose_candidate(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        dist = self.actor.get_distribution(latent, mask)
        if deterministic:
            candidate_weights = dist.mean
        else:
            candidate_weights = dist.sample()
        log_prob = dist.log_prob(candidate_weights)
        return {
            "candidate_weights": candidate_weights,
            "log_prob": log_prob,
            "distribution": dist,
        }

    def gate_decision(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor,
        pre_trade_drifted_weights: torch.Tensor,
        candidate_weights: torch.Tensor,
        estimated_cost_candidate: torch.Tensor,
        estimated_cost_hold: torch.Tensor,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        candidate_quantiles = self.cqr_critic(latent, pre_trade_drifted_weights, candidate_weights)
        hold_quantiles = self.cqr_critic(latent, pre_trade_drifted_weights, pre_trade_drifted_weights)

        pred_candidate_mean_return = OTarCQRCritic.expected_value(candidate_quantiles)
        pred_hold_mean_return = OTarCQRCritic.expected_value(hold_quantiles)

        if self.quantile_tail_enabled:
            pred_candidate_lower_tail_loss = OTarCQRCritic.lower_tail_loss(candidate_quantiles, self.tail_alpha)
            pred_hold_lower_tail_loss = OTarCQRCritic.lower_tail_loss(hold_quantiles, self.tail_alpha)
        else:
            pred_candidate_lower_tail_loss = torch.zeros_like(pred_candidate_mean_return)
            pred_hold_lower_tail_loss = torch.zeros_like(pred_hold_mean_return)

        pred_candidate_utility = (
            pred_candidate_mean_return
            - self.lambda_tail * pred_candidate_lower_tail_loss
            - estimated_cost_candidate
        )
        pred_hold_utility = (
            pred_hold_mean_return
            - self.lambda_tail * pred_hold_lower_tail_loss
            - estimated_cost_hold
        )
        pred_delta_utility = pred_candidate_utility - pred_hold_utility

        gate_action = (pred_delta_utility > self.gate_margin).long()

        candidate_quantiles_sorted, _ = torch.sort(candidate_quantiles, dim=1)
        hold_quantiles_sorted, _ = torch.sort(hold_quantiles, dim=1)
        q_idx = max(0, math.ceil(0.05 * self.n_quantiles) - 1)

        return {
            "gate_action": gate_action,
            "pred_delta_utility": pred_delta_utility,
            "pred_candidate_mean_return": pred_candidate_mean_return,
            "pred_hold_mean_return": pred_hold_mean_return,
            "pred_candidate_lower_tail_loss": pred_candidate_lower_tail_loss,
            "pred_hold_lower_tail_loss": pred_hold_lower_tail_loss,
            "pred_candidate_utility": pred_candidate_utility,
            "pred_hold_utility": pred_hold_utility,
            "quantile_spread_candidate": candidate_quantiles.std(dim=1, keepdim=True),
            "quantile_spread_hold": hold_quantiles.std(dim=1, keepdim=True),
            "predicted_5pct_quantile_candidate": candidate_quantiles_sorted[:, q_idx : q_idx + 1],
            "predicted_5pct_quantile_hold": hold_quantiles_sorted[:, q_idx : q_idx + 1],
            "candidate_quantiles": candidate_quantiles,
            "hold_quantiles": hold_quantiles,
        }

    def update_targets(self) -> None:
        self.target_cqr_critic.load_state_dict(self.cqr_critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

    @property
    def _device(self) -> torch.device:
        return next(self.encoder.parameters()).device


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["OTarCQRGate", "OTarCQRCritic"]
