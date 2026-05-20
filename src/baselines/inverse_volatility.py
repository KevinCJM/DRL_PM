from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.baselines.base_strategy import TraditionalStrategyBase
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


DEFAULT_VOLATILITY_FLOOR = 1.0e-6


class InverseVolatilityStrategy(TraditionalStrategyBase):
    strategy_name = "inverse_volatility"

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self._inverse_volatility_report: dict[str, Any] = {}
        action = super().compute_target_weights(decision_market_state, portfolio_state)
        report = dict(self._inverse_volatility_report)
        report["projected_weights"] = action.target_weights.astype(float).tolist()
        action.action_info["inverse_volatility"] = report
        return action

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        config = _inverse_volatility_config(self.config)
        available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
        volatility = np.asarray(decision_market_state.volatility_20d_at_decision, dtype=float)
        volatility_floor = _volatility_floor(config)
        raw_weights = inverse_volatility_weights(available, volatility, volatility_floor)
        self._inverse_volatility_report = {
            "volatility_floor": volatility_floor,
            "raw_weights": raw_weights.astype(float).tolist(),
        }
        return raw_weights


def inverse_volatility_weights(
    available_mask: np.ndarray,
    volatility: np.ndarray,
    volatility_floor: float = DEFAULT_VOLATILITY_FLOOR,
) -> np.ndarray:
    available = np.asarray(available_mask, dtype=bool)
    values = np.asarray(volatility, dtype=float)
    weights = np.zeros(available.shape, dtype=float)
    if not available.any():
        return weights
    safe_volatility = np.where(np.isfinite(values) & (values > 0.0), values, volatility_floor)
    safe_volatility = np.maximum(safe_volatility, volatility_floor)
    inverse = np.zeros(available.shape, dtype=float)
    inverse[available] = 1.0 / safe_volatility[available]
    inverse_sum = float(inverse[available].sum())
    if inverse_sum <= 0.0:
        weights[available] = 1.0 / int(available.sum())
    else:
        weights[available] = inverse[available] / inverse_sum
    return weights


def _inverse_volatility_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("inverse_volatility"), Mapping):
        return dict(config["inverse_volatility"])
    return dict(config)


def _volatility_floor(config: Mapping[str, Any]) -> float:
    return max(float(config.get("volatility_floor", DEFAULT_VOLATILITY_FLOOR)), DEFAULT_VOLATILITY_FLOOR)


__all__ = ["InverseVolatilityStrategy", "inverse_volatility_weights"]
