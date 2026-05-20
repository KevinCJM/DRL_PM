from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .base_strategy import BaseStrategy
from .deep_training import collect_training_batch, deep_baseline_training_config, iter_minibatches, training_summary
from .ppo_baseline import _enforce_masked_simplex, _fit_return_prior_weights
from ..envs.state import DecisionMarketState, PortfolioState, PortfolioAction
from ..models.cost_estimator import CostEstimator
from ..models.dqn_gate import DQNGate
from ..models.encoders import EncoderFactory

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
ALLOWED_TEMPLATES = {"hold", "equal_weight"}

class DQNOnlyStrategy(BaseStrategy):
    strategy_name = "dqn_only"
    fit_required = True

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        model_config = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
        self.latent_dim = int(config.get("latent_dim", model_config.get("latent_dim", 256)))
        self.n_assets = config["n_assets"]
        
        self.templates = [str(template) for template in config.get("dqn_only", {}).get("templates", ["hold", "equal_weight"])]
        if not self.templates:
            raise ValueError("ERR_DQN_ONLY_CONFIG_INVALID: templates must not be empty")
        unknown_templates = sorted(set(self.templates) - ALLOWED_TEMPLATES)
        if unknown_templates:
            raise ValueError(f"ERR_DQN_ONLY_TEMPLATE_NOT_IMPLEMENTED: {unknown_templates}")
        self.n_templates = len(self.templates)
        self.device = torch.device("cpu")
        dqn_config = config.get("dqn", {}) if isinstance(config.get("dqn"), Mapping) else {}
        self.encoder = EncoderFactory.create(config)
        self.gate = DQNGate(
            latent_dim=self.latent_dim,
            n_assets=self.n_assets,
            output_dim=self.n_templates,
            dueling=bool(dqn_config.get("dueling", True)),
        )
        self.encoder.to(self.device)
        self.gate.to(self.device)
        self.encoder.eval()
        self.gate.eval()
        self.fitted_prior_weights: np.ndarray | None = None
        training_config = deep_baseline_training_config(self.config)
        self.training_result: dict[str, Any] = training_summary(
            "not_started",
            current_weight_mode=training_config.current_weight_mode,
        )

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> DQNOnlyStrategy:
        super().fit(train_data, validation_data)
        self.fitted_prior_weights = _fit_return_prior_weights(train_data, self.n_assets)
        self.training_result = self._train_dqn_templates(train_data)
        return self

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        self.validate_portfolio_state(portfolio_state)
        
        market_image = torch.as_tensor(state.market_image, dtype=torch.float32, device=self.device).unsqueeze(0)
        current_weights = torch.as_tensor(
            portfolio_state.current_weights,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        candidate_template = "equal_weight" if "equal_weight" in self.templates else self.templates[0]
        candidate_weights_np = self._template_weights(candidate_template, state, portfolio_state)
        candidate_weights = torch.as_tensor(candidate_weights_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            latent = self.encoder(market_image)
            estimated_turnover, estimated_cost = CostEstimator.estimate_from_decision_state(
                candidate_weights,
                current_weights,
                state,
                portfolio_state,
                self.config,
            )
            q_values = self.gate(latent, candidate_weights, current_weights, estimated_turnover, estimated_cost)
            action_idx = torch.argmax(q_values, dim=1).item()
            
        template = self.templates[action_idx]
        target_weights, rebalance_action = self._target_from_template(template, state, portfolio_state)
        q_row = q_values.squeeze(0).detach().cpu().numpy()
        estimated_turnover_value = float(estimated_turnover.squeeze(0).detach().cpu().item())
        estimated_cost_value = float(estimated_cost.squeeze(0).detach().cpu().item())
        
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=target_weights,
                rebalance_action=rebalance_action,
                rebalance_intensity=1.0,
                action_info={
                    "strategy": "dqn_only",
                    "template_chosen": template,
                    "template_index": int(action_idx),
                    "target_source": "template",
                    "gate_candidate_template": candidate_template,
                    "q_values": q_row,
                    "estimated_turnover": estimated_turnover_value,
                    "estimated_cost": estimated_cost_value,
                    "gate_input_fields": GATE_INPUT_FIELDS,
                    **_binary_q_info(q_row),
                }
            )
        )

    def _target_from_template(
        self,
        template: str,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> tuple[np.ndarray, int]:
        if template == "hold":
            return np.asarray(portfolio_state.current_weights, dtype=float).copy(), 0
        if template == "equal_weight":
            return self._template_weights(template, decision_market_state, portfolio_state), 1
        raise ValueError(f"ERR_DQN_ONLY_TEMPLATE_NOT_IMPLEMENTED: {template}")

    def _template_weights(
        self,
        template: str,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> np.ndarray:
        if template == "hold":
            return np.asarray(portfolio_state.current_weights, dtype=float).copy()
        if template == "equal_weight":
            if self.fitted_prior_weights is not None:
                return _enforce_masked_simplex(self.fitted_prior_weights, decision_market_state.available_mask_at_decision)
            mask = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
            target_weights = np.zeros(self.n_assets, dtype=float)
            if mask.any():
                target_weights[mask] = 1.0 / float(mask.sum())
            return target_weights
        raise ValueError(f"ERR_DQN_ONLY_TEMPLATE_NOT_IMPLEMENTED: {template}")

    def _fitted_action_index(self) -> int:
        if "equal_weight" in self.templates:
            return int(self.templates.index("equal_weight"))
        return 0

    def _train_dqn_templates(self, train_data: Any | None) -> dict[str, Any]:
        training_config = deep_baseline_training_config(self.config)
        if not training_config.enabled or training_config.epochs <= 0:
            return training_summary("disabled", current_weight_mode=training_config.current_weight_mode)
        batch = collect_training_batch(
            train_data,
            n_features=int(self.config.get("n_features", 1)),
            window_size=int(self.config.get("window_size", 1)),
            n_assets=self.n_assets,
            device=self.device,
            max_samples=training_config.max_samples,
        )
        if batch is None:
            return training_summary("skipped_no_samples", current_weight_mode=training_config.current_weight_mode)

        self.encoder.train()
        self.gate.train()
        optimizer = torch.optim.Adam(
            [*self.encoder.parameters(), *self.gate.parameters()],
            lr=training_config.learning_rate,
        )
        last_loss: torch.Tensor | None = None
        for _ in range(training_config.epochs):
            for indices in iter_minibatches(batch, training_config.batch_size):
                market_image = batch.market_image[indices]
                current_weights = batch.current_weights[indices]
                equal_weights = batch.equal_weights[indices]
                future_returns = batch.future_returns[indices]
                candidate_weights = equal_weights
                turnover = 0.5 * torch.sum(torch.abs(candidate_weights - current_weights), dim=1, keepdim=True)
                estimated_cost = torch.zeros_like(turnover)
                latent = self.encoder(market_image)
                q_values = self.gate(latent, candidate_weights, current_weights, turnover, estimated_cost)
                target_q = self._template_target_values(current_weights, equal_weights, future_returns)
                loss = F.mse_loss(q_values, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*self.encoder.parameters(), *self.gate.parameters()], max_norm=1.0)
                optimizer.step()
                last_loss = loss.detach()
        self.encoder.eval()
        self.gate.eval()
        return training_summary(
            "completed",
            samples=batch.size,
            loss=None if last_loss is None else float(last_loss.cpu()),
            current_weight_mode=training_config.current_weight_mode,
        )

    def _template_target_values(
        self,
        current_weights: torch.Tensor,
        equal_weights: torch.Tensor,
        future_returns: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        for template in self.templates:
            if template == "equal_weight":
                weights = equal_weights
            elif template == "hold":
                weights = current_weights
            else:
                raise ValueError(f"ERR_DQN_ONLY_TEMPLATE_NOT_IMPLEMENTED: {template}")
            values.append((weights * future_returns).sum(dim=1))
        return torch.stack(values, dim=1)


def _binary_q_info(q_values: np.ndarray) -> dict[str, float]:
    if q_values.shape[0] != 2:
        return {}
    return {
        "q_hold": float(q_values[0]),
        "q_rebalance": float(q_values[1]),
        "q_gap": float(q_values[1] - q_values[0]),
    }
