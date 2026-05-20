from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from src.baselines.base_strategy import TraditionalStrategyBase
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.utils.optimization import OPT_EPS, shrink_covariance


MIN_HISTORY_OBSERVATIONS = 2
DEFAULT_SHRINKAGE = 0.1


class HRPStrategy(TraditionalStrategyBase):
    strategy_name = "hrp"

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self._hrp_report: dict[str, Any] = {}
        action = super().compute_target_weights(decision_market_state, portfolio_state)
        report = dict(self._hrp_report)
        report["projected_weights"] = action.target_weights.astype(float).tolist()
        action.action_info["hrp"] = report
        if report.get("fallback_reason"):
            action.action_info["fallback_reason"] = report["fallback_reason"]
        return action

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        config = _hrp_config(self.config)
        available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
        raw_weights = np.zeros(available.shape, dtype=float)
        lookback_window = _lookback_window(config, decision_market_state.log_return_window.shape[0])
        self._hrp_report = {
            "lookback_window": lookback_window,
            "covariance_method": "diagonal_shrinkage",
            "distance_method": "correlation_distance",
            "linkage_method": _linkage_method(config),
            "quasi_diagonalization": False,
            "recursive_bisection": False,
            "cluster_success": False,
        }

        if not available.any():
            self._hrp_report["fallback_reason"] = "no_available_asset"
            return raw_weights
        if int(available.sum()) == 1:
            raw_weights[available] = 1.0
            self._hrp_report.update(
                {
                    "cluster_success": True,
                    "quasi_diagonalization": True,
                    "recursive_bisection": True,
                    "cluster_order": [0],
                    "raw_weights": raw_weights.astype(float).tolist(),
                }
            )
            return raw_weights

        returns = np.asarray(decision_market_state.log_return_window[-lookback_window:, :], dtype=float)
        active_returns = returns[:, available]
        finite_rows = np.isfinite(active_returns).all(axis=1)
        clean_returns = active_returns[finite_rows]
        if clean_returns.shape[0] < _min_observations(config):
            return self._fallback_weights(available, raw_weights, "insufficient_history")

        try:
            covariance = shrink_covariance(clean_returns, _covariance_shrinkage(config))
            distance = _correlation_distance(covariance)
            clusters = linkage(squareform(distance, checks=False), method=_linkage_method(config))
            ordered_indices = _quasi_diagonalize(clusters)
            active_weights = _recursive_bisection(covariance, ordered_indices)
        except Exception:
            return self._fallback_weights(available, raw_weights, "hrp_failed")

        if not np.isfinite(active_weights).all() or float(active_weights.sum()) <= OPT_EPS:
            return self._fallback_weights(available, raw_weights, "invalid_hrp_weights")

        active_weights = np.clip(active_weights, 0.0, 1.0)
        active_weights = active_weights / float(active_weights.sum())
        raw_weights[available] = active_weights
        self._hrp_report.update(
            {
                "cluster_success": True,
                "quasi_diagonalization": True,
                "recursive_bisection": True,
                "cluster_order": [int(index) for index in ordered_indices],
                "raw_weights": raw_weights.astype(float).tolist(),
            }
        )
        return raw_weights

    def _fallback_weights(self, available: np.ndarray, raw_weights: np.ndarray, fallback_reason: str) -> np.ndarray:
        raw_weights[available] = 1.0 / int(available.sum())
        self._hrp_report.update(
            {
                "fallback_reason": fallback_reason,
                "fallback": "equal_weight",
                "raw_weights": raw_weights.astype(float).tolist(),
            }
        )
        return raw_weights


def _correlation_distance(covariance: np.ndarray) -> np.ndarray:
    sigma = np.asarray(covariance, dtype=float)
    volatility = np.sqrt(np.maximum(np.diag(sigma), 0.0))
    denominator = np.outer(np.maximum(volatility, OPT_EPS), np.maximum(volatility, OPT_EPS))
    correlation = np.divide(sigma, denominator, out=np.zeros_like(sigma, dtype=float), where=denominator > OPT_EPS)
    correlation = np.clip(np.where(np.isfinite(correlation), correlation, 0.0), -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    distance = np.sqrt(np.maximum(0.0, (1.0 - correlation) / 2.0))
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)
    if not np.isfinite(distance).all():
        raise ValueError("invalid correlation distance")
    return distance


def _quasi_diagonalize(linkage_matrix: np.ndarray) -> list[int]:
    n_assets = linkage_matrix.shape[0] + 1
    ordered = [int(linkage_matrix[-1, 0]), int(linkage_matrix[-1, 1])]
    while any(index >= n_assets for index in ordered):
        expanded: list[int] = []
        for index in ordered:
            if index < n_assets:
                expanded.append(int(index))
            else:
                cluster_index = int(index - n_assets)
                expanded.extend([int(linkage_matrix[cluster_index, 0]), int(linkage_matrix[cluster_index, 1])])
        ordered = expanded
    return [int(index) for index in ordered]


def _recursive_bisection(covariance: np.ndarray, ordered_indices: list[int]) -> np.ndarray:
    weights = np.ones(covariance.shape[0], dtype=float)
    clusters = [list(ordered_indices)]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]
        left_variance = _cluster_variance(covariance, left)
        right_variance = _cluster_variance(covariance, right)
        denominator = left_variance + right_variance
        if denominator <= OPT_EPS:
            alpha = 0.5
        else:
            alpha = 1.0 - left_variance / denominator
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        clusters.extend([left, right])
    weight_sum = float(weights.sum())
    if weight_sum <= OPT_EPS:
        return np.zeros_like(weights, dtype=float)
    return weights / weight_sum


def _cluster_variance(covariance: np.ndarray, indices: list[int]) -> float:
    cluster_covariance = np.asarray(covariance[np.ix_(indices, indices)], dtype=float)
    inverse_diagonal = 1.0 / np.maximum(np.diag(cluster_covariance), OPT_EPS)
    weights = inverse_diagonal / float(inverse_diagonal.sum())
    variance = float(weights @ cluster_covariance @ weights)
    return max(variance, OPT_EPS)


def _hrp_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("hrp"), Mapping):
        return dict(config["hrp"])
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


def _linkage_method(config: Mapping[str, Any]) -> str:
    return str(config.get("linkage_method", "single"))


__all__ = ["HRPStrategy"]
