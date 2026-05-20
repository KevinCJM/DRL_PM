from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from .deep_training import collect_training_batch, deep_baseline_training_config, iter_minibatches, training_summary
from .ppo_baseline import PPOBaselineStrategy, _blend_with_fitted_prior
from ..envs.state import DecisionMarketState, PortfolioState, PortfolioAction
from ..models.cost_estimator import CostEstimator
from ..models.dqn_gate import DQNGate

GATE_INPUT_FIELDS = (
    "market_image",
    "available_mask_at_decision",
    "current_weights",
    "candidate_weights",
    "adv20_at_decision",
    "volatility_20d_at_decision",
    "amount_at_decision",
    "turnover_rate_at_decision",
    "portfolio_value",
)

class BernoulliGatedPPOStrategy(PPOBaselineStrategy):
    strategy_name = "bernoulli_gated_ppo"
    default_encoder_type = "cnn"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        dqn_config = config.get("dqn", {}) if isinstance(config.get("dqn"), Mapping) else {}
        self.gate = DQNGate(
            latent_dim=self.model.latent_dim,
            n_assets=self.model.n_assets,
            output_dim=2,
            dueling=bool(dqn_config.get("dueling", True)),
        )
        self.gate.to(self.device)
        self.gate.eval()
        training_config = deep_baseline_training_config(self.config)
        self.gate_training_result = training_summary(
            "not_started",
            current_weight_mode=training_config.current_weight_mode,
        )

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> BernoulliGatedPPOStrategy:
        super().fit(train_data, validation_data)
        self.gate_training_result = self._train_gate(train_data)
        return self

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        self.validate_portfolio_state(portfolio_state)
        
        market_image = torch.as_tensor(state.market_image, dtype=torch.float32, device=self.device).unsqueeze(0)
        available_mask = torch.as_tensor(
            state.available_mask_at_decision,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        current_weights = torch.as_tensor(
            portfolio_state.current_weights,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(market_image, available_mask, current_weights, deterministic=True)
            candidate_weights = outputs["candidate_weights"]
            fitted_candidate = _blend_with_fitted_prior(
                candidate_weights.squeeze(0).detach().cpu().numpy(),
                self.fitted_prior_weights,
                state.available_mask_at_decision,
                self.prior_blend_weight,
            )
            candidate_weights = torch.as_tensor(fitted_candidate, dtype=torch.float32, device=self.device).unsqueeze(0)
            estimated_turnover, estimated_cost = CostEstimator.estimate_from_decision_state(
                candidate_weights,
                current_weights,
                state,
                portfolio_state,
                self.config,
            )
            gate_q = self.gate(
                outputs["latent"],
                candidate_weights,
                current_weights,
                estimated_turnover,
                estimated_cost,
            )
            gate_logit = gate_q[:, 1] - gate_q[:, 0]
            gate_action = (gate_logit >= 0.0).long()
            gate_log_prob = Bernoulli(logits=gate_logit).log_prob(gate_action.to(dtype=gate_logit.dtype))
            
        target_weights = candidate_weights.squeeze(0).detach().cpu().numpy()
        rebalance_action = int(gate_action.item())
        q_values = gate_q.squeeze(0).detach().cpu().numpy()
        policy_log_prob = float(outputs["log_prob"].squeeze(0).detach().cpu().item())
        gate_log_prob_value = float(gate_log_prob.squeeze(0).detach().cpu().item())
        estimated_turnover_value = float(estimated_turnover.squeeze(0).detach().cpu().item())
        estimated_cost_value = float(estimated_cost.squeeze(0).detach().cpu().item())
        
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=target_weights,
                rebalance_action=rebalance_action,
                rebalance_intensity=1.0,
                action_info={
                    "strategy": self.strategy_name,
                    "gate_action": rebalance_action,
                    "gate_log_prob": gate_log_prob_value,
                    "decision_log_prob": gate_log_prob_value,
                    "candidate_log_prob": policy_log_prob,
                    "q_hold": float(q_values[0]),
                    "q_rebalance": float(q_values[1]),
                    "q_gap": float(q_values[1] - q_values[0]),
                    "estimated_turnover": estimated_turnover_value,
                    "estimated_cost": estimated_cost_value,
                    "prior_blend_weight": self.prior_blend_weight,
                    "gate_input_fields": GATE_INPUT_FIELDS,
                }
            )
        )

    def _train_gate(self, train_data: Any | None) -> dict[str, Any]:
        training_config = deep_baseline_training_config(self.config)
        if not training_config.enabled or training_config.epochs <= 0:
            return training_summary("disabled", current_weight_mode=training_config.current_weight_mode)
        batch = collect_training_batch(
            train_data,
            n_features=self.model.n_features,
            window_size=self.model.window_size,
            n_assets=self.model.n_assets,
            device=self.device,
            max_samples=training_config.max_samples,
        )
        if batch is None:
            return training_summary("skipped_no_samples", current_weight_mode=training_config.current_weight_mode)
        self.model.eval()
        self.gate.train()
        optimizer = torch.optim.Adam(self.gate.parameters(), lr=training_config.learning_rate)
        last_loss: torch.Tensor | None = None
        for _ in range(training_config.epochs):
            for indices in iter_minibatches(batch, training_config.batch_size):
                with torch.no_grad():
                    outputs = self.model(
                        batch.market_image[indices],
                        batch.availability_mask[indices],
                        batch.current_weights[indices],
                        deterministic=True,
                    )
                    candidate_weights = outputs["candidate_weights"]
                    latent = outputs["latent"].detach()
                    hold_reward = (batch.current_weights[indices] * batch.future_returns[indices]).sum(dim=1)
                    rebalance_reward = (candidate_weights * batch.future_returns[indices]).sum(dim=1)
                    target_q = torch.stack([hold_reward, rebalance_reward], dim=1)
                turnover = 0.5 * torch.sum(torch.abs(candidate_weights - batch.current_weights[indices]), dim=1, keepdim=True)
                estimated_cost = torch.zeros_like(turnover)
                q_values = self.gate(
                    latent,
                    candidate_weights.detach(),
                    batch.current_weights[indices],
                    turnover,
                    estimated_cost,
                )
                loss = F.mse_loss(q_values, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gate.parameters(), max_norm=1.0)
                optimizer.step()
                last_loss = loss.detach()
        self.gate.eval()
        return training_summary(
            "completed",
            samples=batch.size,
            loss=None if last_loss is None else float(last_loss.cpu()),
            current_weight_mode=training_config.current_weight_mode,
        )
