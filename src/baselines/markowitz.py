from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.baselines.base_strategy import TraditionalStrategyBase
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.utils.optimization import optimize_long_only_portfolio, shrink_covariance


MIN_HISTORY_OBSERVATIONS = 2
DEFAULT_SHRINKAGE = 0.1
DEFAULT_MAXITER = 200

OBJECTIVE_BY_STRATEGY = {
    "markowitz": "mean_variance",
    "traditional_markowitz_mean_variance": "mean_variance",
    "markowitz_min_variance": "min_variance",
    "markowitz_max_sharpe": "max_sharpe",
}


class MarkowitzStrategy(TraditionalStrategyBase):
    strategy_name = "markowitz"

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self._markowitz_report: dict[str, Any] = {}
        action = super().compute_target_weights(decision_market_state, portfolio_state)
        report = dict(self._markowitz_report)
        report["projected_weights"] = action.target_weights.astype(float).tolist()
        action.action_info["markowitz"] = report
        if report.get("fallback_reason"):
            action.action_info["fallback_reason"] = report["fallback_reason"]
        return action

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        config = _markowitz_config(self.config, self.strategy_name)
        available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
        raw_weights = np.zeros(available.shape, dtype=float)
        variant = _strategy_variant(self.strategy_name)
        objective = OBJECTIVE_BY_STRATEGY[variant]
        lookback_window = _lookback_window(config, decision_market_state.log_return_window.shape[0])
        self._markowitz_report = {
            "variant": variant,
            "objective": objective,
            "lookback_window": lookback_window,
            "covariance_method": "diagonal_shrinkage",
            "optimizer_success": False,
        }

        if not available.any():
            self._markowitz_report["fallback_reason"] = "no_available_asset"
            return raw_weights
        if int(available.sum()) == 1:
            raw_weights[available] = 1.0
            self._markowitz_report["optimizer_success"] = True
            self._markowitz_report["raw_weights"] = raw_weights.astype(float).tolist()
            return raw_weights

        returns = np.asarray(decision_market_state.log_return_window[-lookback_window:, :], dtype=float)
        active_returns = returns[:, available]
        finite_rows = np.isfinite(active_returns).all(axis=1)
        clean_returns = active_returns[finite_rows]
        min_observations = _min_observations(config)
        if clean_returns.shape[0] < min_observations:
            return self._fallback_weights(decision_market_state, available, raw_weights, "insufficient_history", config)

        expected_returns = clean_returns.mean(axis=0)
        covariance = shrink_covariance(clean_returns, _covariance_shrinkage(config))
        result = optimize_long_only_portfolio(
            expected_returns,
            covariance,
            objective,
            lambda_risk=_lambda_risk(config),
            risk_free_rate=_risk_free_rate(self.config, config),
            maxiter=_optimizer_maxiter(config),
        )
        if not result.success:
            return self._fallback_weights(
                decision_market_state,
                available,
                raw_weights,
                result.fallback_reason or "optimizer_failed",
                config,
            )

        raw_weights[available] = result.weights
        self._markowitz_report["optimizer_success"] = True
        self._markowitz_report["raw_weights"] = raw_weights.astype(float).tolist()
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
            volatility = np.asarray(decision_market_state.volatility_20d_at_decision, dtype=float)
            inverse = np.zeros(available.shape, dtype=float)
            valid = available & np.isfinite(volatility) & (volatility > 0.0)
            inverse[valid] = 1.0 / volatility[valid]
            inverse_sum = float(inverse[available].sum())
            if inverse_sum > 0.0:
                raw_weights[available] = inverse[available] / inverse_sum
            else:
                mode = "equal_weight"
        if mode != "inverse_volatility":
            raw_weights[available] = 1.0 / int(available.sum())
        self._markowitz_report.update(
            {
                "fallback_reason": fallback_reason,
                "fallback": mode,
                "raw_weights": raw_weights.astype(float).tolist(),
            }
        )
        return raw_weights


class MarkowitzMeanVarianceStrategy(MarkowitzStrategy):
    strategy_name = "traditional_markowitz_mean_variance"


class MarkowitzMinVarianceStrategy(MarkowitzStrategy):
    strategy_name = "markowitz_min_variance"


class MarkowitzMaxSharpeStrategy(MarkowitzStrategy):
    strategy_name = "markowitz_max_sharpe"


def _markowitz_config(config: Mapping[str, Any], strategy_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(config.get("markowitz"), Mapping):
        result.update(dict(config["markowitz"]))
    else:
        result.update(dict(config))
    if isinstance(config.get(strategy_name), Mapping):
        result.update(dict(config[strategy_name]))
    return result


def _strategy_variant(strategy_name: str) -> str:
    if strategy_name in OBJECTIVE_BY_STRATEGY:
        return strategy_name
    return "traditional_markowitz_mean_variance"


def _lookback_window(config: Mapping[str, Any], available_rows: int) -> int:
    configured = config.get("lookback_window", available_rows)
    lookback_window = int(configured)
    if lookback_window <= 0:
        lookback_window = available_rows
    return max(1, min(lookback_window, available_rows))


def _min_observations(config: Mapping[str, Any]) -> int:
    return max(MIN_HISTORY_OBSERVATIONS, int(config.get("min_observations", MIN_HISTORY_OBSERVATIONS)))


def _covariance_shrinkage(config: Mapping[str, Any]) -> float:
    return float(config.get("covariance_shrinkage", config.get("shrinkage", DEFAULT_SHRINKAGE)))


def _lambda_risk(config: Mapping[str, Any]) -> float:
    return float(config.get("lambda_risk", 1.0))


def _risk_free_rate(root_config: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    if "risk_free_rate" in config:
        return float(config["risk_free_rate"])
    evaluation = root_config.get("evaluation")
    if isinstance(evaluation, Mapping) and "risk_free_rate_annual" in evaluation:
        return float(evaluation["risk_free_rate_annual"]) / float(evaluation.get("annualization_factor", 252))
    return 0.0


def _optimizer_maxiter(config: Mapping[str, Any]) -> int:
    return max(1, int(config.get("optimizer_maxiter", DEFAULT_MAXITER)))


__all__ = [
    "MarkowitzMaxSharpeStrategy",
    "MarkowitzMeanVarianceStrategy",
    "MarkowitzMinVarianceStrategy",
    "MarkowitzStrategy",
]
