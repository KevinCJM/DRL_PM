from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.baselines.base_strategy import TraditionalStrategyBase
from src.baselines.inverse_volatility import DEFAULT_VOLATILITY_FLOOR, inverse_volatility_weights
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.utils.optimization import OPT_EPS


MIN_HISTORY_OBSERVATIONS = 2
DEFAULT_MAXITER = 300


class MinimumDrawdownStrategy(TraditionalStrategyBase):
    strategy_name = "minimum_drawdown"

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self._minimum_drawdown_report: dict[str, Any] = {}
        action = super().compute_target_weights(decision_market_state, portfolio_state)
        report = dict(self._minimum_drawdown_report)
        report["projected_weights"] = action.target_weights.astype(float).tolist()
        action.action_info["minimum_drawdown"] = report
        if report.get("fallback_reason"):
            action.action_info["fallback_reason"] = report["fallback_reason"]
        return action

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        config = _minimum_drawdown_config(self.config)
        available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
        raw_weights = np.zeros(available.shape, dtype=float)
        lookback_window = _lookback_window(config, decision_market_state.log_return_window.shape[0])
        self._minimum_drawdown_report = {
            "lookback_window": lookback_window,
            "objective_terms": ["max_drawdown", "volatility", "mean_return"],
            "optimizer_success": False,
        }

        if not available.any():
            self._minimum_drawdown_report["fallback_reason"] = "no_available_asset"
            return raw_weights
        if int(available.sum()) < 2:
            return self._fallback_weights(decision_market_state, available, raw_weights, "insufficient_assets", config)

        returns = np.asarray(decision_market_state.log_return_window[-lookback_window:, :], dtype=float)
        active_returns = returns[:, available]
        finite_rows = np.isfinite(active_returns).all(axis=1)
        clean_returns = active_returns[finite_rows]
        if clean_returns.shape[0] < _min_observations(config):
            return self._fallback_weights(decision_market_state, available, raw_weights, "insufficient_history", config)

        result = _optimize_minimum_drawdown(clean_returns, config)
        if not result.success:
            return self._fallback_weights(
                decision_market_state,
                available,
                raw_weights,
                result.fallback_reason or "optimizer_failed",
                config,
            )

        raw_weights[available] = result.weights
        metrics = _portfolio_path_metrics(clean_returns, result.weights)
        self._minimum_drawdown_report.update(
            {
                "optimizer_success": True,
                "max_drawdown": metrics["max_drawdown"],
                "volatility": metrics["volatility"],
                "mean_return": metrics["mean_return"],
                "objective_value": _minimum_drawdown_objective(result.weights, clean_returns, config),
                "raw_weights": raw_weights.astype(float).tolist(),
            }
        )
        return raw_weights

    def _fallback_weights(
        self,
        decision_market_state: DecisionMarketState,
        available: np.ndarray,
        raw_weights: np.ndarray,
        fallback_reason: str,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        mode = str(config.get("fallback", "equal_weight"))
        if mode == "inverse_volatility":
            raw_weights[:] = inverse_volatility_weights(
                available,
                np.asarray(decision_market_state.volatility_20d_at_decision, dtype=float),
                _volatility_floor(config),
            )
        else:
            mode = "equal_weight"
            raw_weights[available] = 1.0 / int(available.sum())
        self._minimum_drawdown_report.update(
            {
                "fallback_reason": fallback_reason,
                "fallback": mode,
                "raw_weights": raw_weights.astype(float).tolist(),
            }
        )
        return raw_weights


class _MinimumDrawdownResult:
    def __init__(self, weights: np.ndarray, success: bool, fallback_reason: str | None = None) -> None:
        self.weights = weights
        self.success = success
        self.fallback_reason = fallback_reason


def _optimize_minimum_drawdown(returns: np.ndarray, config: Mapping[str, Any]) -> _MinimumDrawdownResult:
    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        return _MinimumDrawdownResult(np.zeros(0, dtype=float), False, "invalid_history")
    n_assets = matrix.shape[1]
    x0 = _initial_weights(matrix, config)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}
    try:
        result = minimize(
            lambda weights: _minimum_drawdown_objective(weights, matrix, config),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": _optimizer_maxiter(config), "ftol": 1.0e-12, "disp": False},
        )
    except Exception:
        return _MinimumDrawdownResult(np.zeros(n_assets, dtype=float), False, "optimizer_exception")
    if not result.success:
        return _MinimumDrawdownResult(np.zeros(n_assets, dtype=float), False, "optimizer_failed")
    weights = np.asarray(result.x, dtype=float)
    if not np.isfinite(weights).all():
        return _MinimumDrawdownResult(np.zeros(n_assets, dtype=float), False, "non_finite_weights")
    weights = np.clip(weights, 0.0, 1.0)
    weight_sum = float(weights.sum())
    if weight_sum <= OPT_EPS:
        return _MinimumDrawdownResult(np.zeros(n_assets, dtype=float), False, "zero_weight_sum")
    return _MinimumDrawdownResult(weights / weight_sum, True)


def _initial_weights(returns: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    n_assets = returns.shape[1]
    candidates = [np.full(n_assets, 1.0 / n_assets, dtype=float)]
    volatility = np.std(returns, axis=0, ddof=1)
    candidates.append(inverse_volatility_weights(np.ones(n_assets, dtype=bool), volatility, _volatility_floor(config)))
    for index in range(n_assets):
        unit = np.zeros(n_assets, dtype=float)
        unit[index] = 1.0
        candidates.append(unit)
    best = min(candidates, key=lambda weights: _minimum_drawdown_objective(weights, returns, config))
    return np.asarray(best, dtype=float)


def _minimum_drawdown_objective(weights: np.ndarray, returns: np.ndarray, config: Mapping[str, Any]) -> float:
    metrics = _portfolio_path_metrics(returns, weights)
    return (
        _max_drawdown_weight(config) * metrics["max_drawdown"]
        + _volatility_weight(config) * metrics["volatility"]
        - _mean_return_weight(config) * metrics["mean_return"]
    )


def _portfolio_path_metrics(returns: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    portfolio_returns = np.asarray(returns, dtype=float) @ np.asarray(weights, dtype=float)
    cumulative_nav = np.exp(np.cumsum(portfolio_returns))
    running_max = np.maximum.accumulate(cumulative_nav)
    drawdown = np.maximum(0.0, 1.0 - cumulative_nav / np.maximum(running_max, OPT_EPS))
    volatility = float(np.std(portfolio_returns, ddof=1)) if portfolio_returns.shape[0] > 1 else 0.0
    return {
        "max_drawdown": float(np.max(drawdown)) if drawdown.size else 0.0,
        "volatility": volatility,
        "mean_return": float(np.mean(portfolio_returns)) if portfolio_returns.size else 0.0,
    }


def _minimum_drawdown_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("minimum_drawdown"), Mapping):
        return dict(config["minimum_drawdown"])
    return dict(config)


def _lookback_window(config: Mapping[str, Any], available_rows: int) -> int:
    configured = config.get("lookback_window", available_rows)
    lookback_window = int(configured)
    if lookback_window <= 0:
        lookback_window = available_rows
    return max(1, min(lookback_window, available_rows))


def _min_observations(config: Mapping[str, Any]) -> int:
    return max(MIN_HISTORY_OBSERVATIONS, int(config.get("min_observations", MIN_HISTORY_OBSERVATIONS)))


def _optimizer_maxiter(config: Mapping[str, Any]) -> int:
    return max(1, int(config.get("optimizer_maxiter", DEFAULT_MAXITER)))


def _volatility_floor(config: Mapping[str, Any]) -> float:
    return max(float(config.get("volatility_floor", DEFAULT_VOLATILITY_FLOOR)), DEFAULT_VOLATILITY_FLOOR)


def _max_drawdown_weight(config: Mapping[str, Any]) -> float:
    return float(config.get("max_drawdown_weight", 1.0))


def _volatility_weight(config: Mapping[str, Any]) -> float:
    return float(config.get("volatility_weight", 0.1))


def _mean_return_weight(config: Mapping[str, Any]) -> float:
    return float(config.get("mean_return_weight", 0.1))


__all__ = ["MinimumDrawdownStrategy"]
