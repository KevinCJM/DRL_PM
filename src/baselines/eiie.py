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

        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=weights,
                rebalance_action=1,
                rebalance_intensity=1.0,
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
    weights = portfolio_state.previous_executed_weights
    if weights is None:
        weights = portfolio_state.current_weights
    return np.asarray(weights, dtype=float).copy()
