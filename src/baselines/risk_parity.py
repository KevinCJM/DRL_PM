from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import minimize

from src.baselines.base_strategy import TraditionalStrategyBase
from src.baselines.inverse_volatility import DEFAULT_VOLATILITY_FLOOR, inverse_volatility_weights
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.utils.optimization import OPT_EPS, shrink_covariance


MIN_HISTORY_OBSERVATIONS = 2
DEFAULT_SHRINKAGE = 0.1
DEFAULT_MAXITER = 300


class RiskParityStrategy(TraditionalStrategyBase):
    strategy_name = "risk_parity"

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self._risk_parity_report: dict[str, Any] = {}
        action = super().compute_target_weights(decision_market_state, portfolio_state)
        report = dict(self._risk_parity_report)
        report["projected_weights"] = action.target_weights.astype(float).tolist()
        action.action_info["risk_parity"] = report
        if report.get("fallback_reason"):
            action.action_info["fallback_reason"] = report["fallback_reason"]
        return action

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        config = _risk_parity_config(self.config)
        available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
        raw_weights = np.zeros(available.shape, dtype=float)
        lookback_window = _lookback_window(config, decision_market_state.log_return_window.shape[0])
        volatility_floor = _volatility_floor(config)
        self._risk_parity_report = {
            "lookback_window": lookback_window,
            "covariance_method": "diagonal_shrinkage",
            "volatility_floor": volatility_floor,
            "optimizer_success": False,
        }

        if not available.any():
            self._risk_parity_report["fallback_reason"] = "no_available_asset"
            return raw_weights
        if int(available.sum()) == 1:
            raw_weights[available] = 1.0
            self._risk_parity_report["optimizer_success"] = True
            self._risk_parity_report["raw_weights"] = raw_weights.astype(float).tolist()
            return raw_weights

        returns = np.asarray(decision_market_state.log_return_window[-lookback_window:, :], dtype=float)
        active_returns = returns[:, available]
        finite_rows = np.isfinite(active_returns).all(axis=1)
        clean_returns = active_returns[finite_rows]
        if clean_returns.shape[0] < _min_observations(config):
            return self._fallback_weights(
                decision_market_state,
                available,
                raw_weights,
                "insufficient_history",
                volatility_floor,
            )

        covariance = _apply_volatility_floor(shrink_covariance(clean_returns, _covariance_shrinkage(config)), volatility_floor)
        result = _optimize_risk_parity(covariance, _optimizer_maxiter(config))
        if not result.success:
            return self._fallback_weights(
                decision_market_state,
                available,
                raw_weights,
                result.fallback_reason or "optimizer_failed",
                volatility_floor,
            )

        raw_weights[available] = result.weights
        self._risk_parity_report["optimizer_success"] = True
        self._risk_parity_report["risk_contributions"] = _risk_contribution_fraction(
            result.weights,
            covariance,
        ).astype(float).tolist()
        self._risk_parity_report["raw_weights"] = raw_weights.astype(float).tolist()
        return raw_weights

    def _fallback_weights(
        self,
        decision_market_state: DecisionMarketState,
        available: np.ndarray,
        raw_weights: np.ndarray,
        fallback_reason: str,
        volatility_floor: float,
    ) -> np.ndarray:
        raw_weights[:] = inverse_volatility_weights(
            available,
            np.asarray(decision_market_state.volatility_20d_at_decision, dtype=float),
            volatility_floor,
        )
        self._risk_parity_report.update(
            {
                "fallback_reason": fallback_reason,
                "fallback": "inverse_volatility",
                "raw_weights": raw_weights.astype(float).tolist(),
            }
        )
        return raw_weights


class _RiskParityResult:
    def __init__(self, weights: np.ndarray, success: bool, fallback_reason: str | None = None) -> None:
        self.weights = weights
        self.success = success
        self.fallback_reason = fallback_reason


def _optimize_risk_parity(covariance: np.ndarray, maxiter: int) -> _RiskParityResult:
    sigma = np.asarray(covariance, dtype=float)
    n_assets = sigma.shape[0]
    if sigma.shape != (n_assets, n_assets) or n_assets == 0 or not np.isfinite(sigma).all():
        return _RiskParityResult(np.zeros(n_assets, dtype=float), False, "invalid_covariance")
    if n_assets == 1:
        return _RiskParityResult(np.array([1.0], dtype=float), True)

    x0 = np.full(n_assets, 1.0 / n_assets, dtype=float)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}
    try:
        result = minimize(
            lambda weights: _risk_parity_objective(weights, sigma),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": int(maxiter), "ftol": 1.0e-12, "disp": False},
        )
    except Exception:
        return _RiskParityResult(np.zeros(n_assets, dtype=float), False, "optimizer_exception")
    if not result.success:
        return _RiskParityResult(np.zeros(n_assets, dtype=float), False, "optimizer_failed")
    weights = np.asarray(result.x, dtype=float)
    if not np.isfinite(weights).all():
        return _RiskParityResult(np.zeros(n_assets, dtype=float), False, "non_finite_weights")
    weights = np.clip(weights, 0.0, 1.0)
    weight_sum = float(weights.sum())
    if weight_sum <= OPT_EPS:
        return _RiskParityResult(np.zeros(n_assets, dtype=float), False, "zero_weight_sum")
    return _RiskParityResult(weights / weight_sum, True)


def _risk_parity_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    contributions = _risk_contribution_fraction(weights, covariance)
    target = np.full(contributions.shape, 1.0 / contributions.shape[0], dtype=float)
    return float(np.sum((contributions - target) ** 2))


def _risk_contribution_fraction(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    portfolio_variance = float(weights @ covariance @ weights)
    if portfolio_variance <= OPT_EPS:
        return np.zeros_like(weights, dtype=float)
    contributions = weights * (covariance @ weights) / portfolio_variance
    return np.where(np.isfinite(contributions), contributions, 0.0)


def _apply_volatility_floor(covariance: np.ndarray, volatility_floor: float) -> np.ndarray:
    sigma = np.asarray(covariance, dtype=float)
    volatility = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    safe_volatility = np.maximum(volatility, volatility_floor)
    denominator = np.outer(np.maximum(volatility, OPT_EPS), np.maximum(volatility, OPT_EPS))
    correlation = np.divide(sigma, denominator, out=np.eye(sigma.shape[0], dtype=float), where=denominator > OPT_EPS)
    correlation = np.clip(np.where(np.isfinite(correlation), correlation, 0.0), -1.0, 1.0)
    floored = correlation * np.outer(safe_volatility, safe_volatility)
    return 0.5 * (floored + floored.T) + np.eye(sigma.shape[0], dtype=float) * OPT_EPS


def _risk_parity_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("risk_parity"), Mapping):
        return dict(config["risk_parity"])
    return dict(config)


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


def _volatility_floor(config: Mapping[str, Any]) -> float:
    return max(float(config.get("volatility_floor", DEFAULT_VOLATILITY_FLOOR)), DEFAULT_VOLATILITY_FLOOR)


def _optimizer_maxiter(config: Mapping[str, Any]) -> int:
    return max(1, int(config.get("optimizer_maxiter", DEFAULT_MAXITER)))


__all__ = ["RiskParityStrategy"]
