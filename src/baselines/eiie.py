from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .base_strategy import BaseStrategy
from .deep_training import collect_training_batch, deep_baseline_training_config, iter_minibatches, training_summary
from .ppo_baseline import _blend_with_fitted_prior, _fit_return_prior_weights
from ..envs.state import DecisionMarketState, PortfolioState, PortfolioAction

MASKED_SCORE_VALUE = -torch.inf

class EIIEStrategy(BaseStrategy):
    strategy_name = "eiie"
    fit_required = True

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.n_assets = config["n_assets"]
        self.n_features = config["n_features"]
        self.window_size = config["window_size"]
        
        self.evaluator = nn.Sequential(
            nn.Conv1d(self.n_features + 1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )
        self.device = torch.device("cpu")
        self.evaluator.to(self.device)
        self.evaluator.eval()
        self.fitted_prior_weights: np.ndarray | None = None
        training_config = deep_baseline_training_config(self.config)
        self.prior_blend_weight = training_config.prior_blend_weight
        self.training_result: dict[str, Any] = training_summary(
            "not_started",
            current_weight_mode=training_config.current_weight_mode,
        )

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> EIIEStrategy:
        super().fit(train_data, validation_data)
        self.fitted_prior_weights = _fit_return_prior_weights(train_data, self.n_assets)
        self.training_result = self._train_evaluator(train_data)
        return self

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        self.validate_portfolio_state(portfolio_state)
        
        x = _eiie_asset_tensor(
            state.market_image,
            _previous_weights(portfolio_state),
            self.device,
            self.n_features,
            self.window_size,
            self.n_assets,
        )

        with torch.no_grad():
            raw_scores = self.evaluator(x).squeeze(-1)

        mask = torch.as_tensor(state.available_mask_at_decision, dtype=torch.bool, device=self.device)
        if not bool(mask.any().item()):
            raise ValueError("ERR_CONSTRAINT_NO_AVAILABLE_ASSET: EIIE available asset mask is empty")
        scores = raw_scores.masked_fill(~mask, MASKED_SCORE_VALUE)

        weights = torch.softmax(scores, dim=0).cpu().numpy()
        weights = _blend_with_fitted_prior(
            weights,
            self.fitted_prior_weights,
            state.available_mask_at_decision,
            self.prior_blend_weight,
        )

        rebalance_decision = _continuous_weight_rebalance_decision(
            self.config,
            "eiie",
            portfolio_state,
            weights,
            getattr(self, "decision_context", {}),
        )

        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=weights,
                rebalance_action=rebalance_decision["rebalance_action"],
                rebalance_intensity=rebalance_decision["rebalance_intensity"],
                action_info={
                    "strategy": "eiie",
                    "scores": scores.cpu().numpy(),
                    "raw_scores": raw_scores.cpu().numpy(),
                    "previous_weights": _previous_weights(portfolio_state),
                    "prior_blend_weight": self.prior_blend_weight,
                    "score_input_fields": (
                        "market_image",
                        "available_mask_at_decision",
                        "previous_weights",
                    ),
                    **rebalance_decision["action_info"],
                },
            )
        )

    def _train_evaluator(self, train_data: Any | None) -> dict[str, Any]:
        training_config = deep_baseline_training_config(self.config)
        if not training_config.enabled or training_config.epochs <= 0:
            return training_summary("disabled", current_weight_mode=training_config.current_weight_mode)
        batch = collect_training_batch(
            train_data,
            n_features=self.n_features,
            window_size=self.window_size,
            n_assets=self.n_assets,
            device=self.device,
            max_samples=training_config.max_samples,
        )
        if batch is None:
            return training_summary("skipped_no_samples", current_weight_mode=training_config.current_weight_mode)
        self.evaluator.train()
        optimizer = torch.optim.Adam(self.evaluator.parameters(), lr=training_config.learning_rate)
        last_loss: torch.Tensor | None = None
        for _ in range(training_config.epochs):
            for indices in iter_minibatches(batch, training_config.batch_size):
                market_image = batch.market_image[indices]
                previous = batch.current_weights[indices].view(-1, 1, 1, self.n_assets).expand(
                    -1,
                    1,
                    self.window_size,
                    self.n_assets,
                )
                x = torch.cat([market_image, previous], dim=1).permute(0, 3, 1, 2)
                scores = self.evaluator(x.reshape(-1, self.n_features + 1, self.window_size)).view(-1, self.n_assets)
                scores = scores.masked_fill(~batch.availability_mask[indices], MASKED_SCORE_VALUE)
                weights = torch.softmax(scores, dim=1)
                gross_return = (weights * batch.future_returns[indices]).sum(dim=1, keepdim=True)
                turnover = 0.5 * torch.sum(torch.abs(weights - batch.current_weights[indices]), dim=1, keepdim=True)
                loss = -(gross_return - float(training_config.turnover_penalty) * turnover).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.evaluator.parameters(), max_norm=1.0)
                optimizer.step()
                last_loss = loss.detach()
        self.evaluator.eval()
        return training_summary(
            "completed",
            samples=batch.size,
            loss=None if last_loss is None else float(last_loss.cpu()),
            current_weight_mode=training_config.current_weight_mode,
        )


def _eiie_asset_tensor(
    market_image: np.ndarray,
    previous_weights: np.ndarray,
    device: torch.device,
    n_features: int,
    window_size: int,
    n_assets: int,
) -> torch.Tensor:
    x = torch.as_tensor(market_image, dtype=torch.float32, device=device)
    if x.ndim != 3:
        raise ValueError("ERR_MODEL_INPUT_SHAPE: EIIE market_image must be [n_features,window_size,n_assets]")
    if tuple(x.shape) != (int(n_features), int(window_size), int(n_assets)):
        raise ValueError("ERR_MODEL_INPUT_SHAPE: EIIE market_image")
    previous = torch.as_tensor(previous_weights, dtype=torch.float32, device=device)
    if previous.shape != (n_assets,):
        raise ValueError("ERR_MODEL_INPUT_SHAPE: EIIE previous_weights")
    previous_channel = previous.view(1, 1, n_assets).expand(1, x.shape[1], n_assets)
    return torch.cat([x, previous_channel], dim=0).permute(2, 0, 1)


def _previous_weights(portfolio_state: PortfolioState) -> np.ndarray:
    return np.asarray(portfolio_state.current_weights, dtype=float).copy()


def _continuous_weight_rebalance_decision(
    config: Mapping[str, Any],
    model_key: str,
    portfolio_state: PortfolioState,
    target_weights: np.ndarray,
    decision_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = np.asarray(portfolio_state.current_weights, dtype=float).copy()
    target = np.asarray(target_weights, dtype=float).copy()
    estimated_turnover = float(0.5 * np.sum(np.abs(target - current)))
    threshold = _rebalance_turnover_threshold(config, model_key)
    context = _mapping(decision_context)
    first_trade = bool(context.get("first_trade", bool(portfolio_state.step_index == 0 and current.sum() <= 0.0)))
    scheduler_allowed = bool(context.get("scheduler_allowed_rebalance", True))
    raw_requested = bool(first_trade or estimated_turnover > threshold + 1.0e-12)
    requested = bool(first_trade or (scheduler_allowed and raw_requested))
    forced_hold_reason = None
    if not requested:
        if raw_requested and not scheduler_allowed:
            forced_hold_reason = "scheduler_blocked"
        else:
            forced_hold_reason = "below_rebalance_turnover_threshold"
    return {
        "rebalance_action": int(requested),
        "rebalance_intensity": 1.0 if requested else 0.0,
        "action_info": {
            "continuous_weight_rebalance_gate": True,
            "estimated_turnover": estimated_turnover,
            "candidate_turnover": estimated_turnover,
            "candidate_turnover_estimate": estimated_turnover,
            "rebalance_turnover_threshold": threshold,
            "raw_model_requested_rebalance": raw_requested,
            "raw_action": int(raw_requested),
            "raw_rho": 1.0 if raw_requested else 0.0,
            "raw_rebalance_intensity": 1.0 if raw_requested else 0.0,
            "rebalance_intensity": 1.0 if requested else 0.0,
            "scheduler_allowed_rebalance": scheduler_allowed,
            "first_trade": first_trade,
            "forced_hold_reason": forced_hold_reason,
        },
    }


def _binary_gate_rebalance_decision(
    config: Mapping[str, Any],
    model_key: str,
    portfolio_state: PortfolioState,
    target_weights: np.ndarray,
    raw_gate_action: int | bool,
    decision_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = np.asarray(portfolio_state.current_weights, dtype=float).copy()
    target = np.asarray(target_weights, dtype=float).copy()
    estimated_turnover = float(0.5 * np.sum(np.abs(target - current)))
    threshold = _rebalance_turnover_threshold(config, model_key)
    context = _mapping(decision_context)
    first_trade = bool(context.get("first_trade", bool(portfolio_state.step_index == 0 and current.sum() <= 0.0)))
    scheduler_allowed = bool(context.get("scheduler_allowed_rebalance", True))
    raw_gate_requested = bool(int(raw_gate_action))
    raw_requested = bool(raw_gate_requested and estimated_turnover > threshold + 1.0e-12)
    requested = bool(first_trade or (raw_requested and scheduler_allowed))
    forced_hold_reason = None
    if not requested:
        if not raw_gate_requested:
            forced_hold_reason = "model_chosen_hold"
        elif not raw_requested:
            forced_hold_reason = "below_rebalance_turnover_threshold"
        elif not scheduler_allowed:
            forced_hold_reason = "scheduler_blocked"
    return {
        "rebalance_action": int(requested),
        "rebalance_intensity": 1.0 if requested else 0.0,
        "action_info": {
            "continuous_weight_rebalance_gate": True,
            "estimated_turnover": estimated_turnover,
            "candidate_turnover": estimated_turnover,
            "candidate_turnover_estimate": estimated_turnover,
            "rebalance_turnover_threshold": threshold,
            "raw_gate_requested_rebalance": raw_gate_requested,
            "raw_model_requested_rebalance": raw_requested,
            "raw_action": int(raw_requested),
            "raw_rho": 1.0 if raw_requested else 0.0,
            "raw_rebalance_intensity": 1.0 if raw_requested else 0.0,
            "rebalance_intensity": 1.0 if requested else 0.0,
            "scheduler_allowed_rebalance": scheduler_allowed,
            "first_trade": first_trade,
            "forced_hold_reason": forced_hold_reason,
        },
    }


def _rebalance_turnover_threshold(config: Mapping[str, Any], model_key: str) -> float:
    model_config = _mapping(config.get(model_key))
    for key in ("rebalance_turnover_threshold", "turnover_gate_threshold", "min_rebalance_turnover"):
        if model_config.get(key) is not None:
            return _non_negative_float(model_config[key])
    activity = _mapping(config.get("execution_activity"))
    for key in ("model_rebalance_turnover_threshold", "rebalance_turnover_threshold", "turnover_gate_threshold"):
        if activity.get(key) is not None:
            return _non_negative_float(activity[key])
    rebalance = _mapping(config.get("rebalance"))
    if str(rebalance.get("mode", "")) == "threshold_turnover":
        return _non_negative_float(rebalance.get("threshold_turnover", 0.0))
    return 0.0


def _non_negative_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("ERR_EIIE_REBALANCE_THRESHOLD_INVALID")
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
