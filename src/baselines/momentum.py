from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.baselines.base_strategy import TraditionalStrategyBase
from src.baselines.inverse_volatility import DEFAULT_VOLATILITY_FLOOR, inverse_volatility_weights
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.utils.optimization import OPT_EPS


MIN_HISTORY_OBSERVATIONS = 2
DEFAULT_SOFTMAX_TEMPERATURE = 1.0


class MomentumStrategy(TraditionalStrategyBase):
    strategy_name = "momentum"

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self._momentum_report: dict[str, Any] = {}
        action = super().compute_target_weights(decision_market_state, portfolio_state)
        report = dict(self._momentum_report)
        report["projected_weights"] = action.target_weights.astype(float).tolist()
        action.action_info["momentum"] = report
        if report.get("fallback_reason"):
            action.action_info["fallback_reason"] = report["fallback_reason"]
        return action

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        config = _momentum_config(self.config)
        available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
        raw_weights = np.zeros(available.shape, dtype=float)
        lookback_window = _lookback_window(config, decision_market_state.log_return_window.shape[0])
        score_mode = _score_mode(config)
        weight_mode = _weight_mode(config)
        threshold = _threshold(config)
        top_k = _top_k(config, int(available.sum()))
        self._momentum_report = {
            "lookback_window": lookback_window,
            "score_mode": score_mode,
            "top_k": top_k,
            "threshold": threshold,
            "weight_mode": weight_mode,
        }

        if not available.any():
            self._momentum_report["fallback_reason"] = "no_available_asset"
            return raw_weights

        returns = np.asarray(decision_market_state.log_return_window[-lookback_window:, :], dtype=float)
        scores, history_count = _momentum_scores(returns, score_mode, _volatility_floor(config))
        candidate_mask = available & np.isfinite(scores) & (history_count >= _min_observations(config))
        if threshold is not None:
            candidate_mask &= scores >= threshold
        selected_indices = _select_top_indices(scores, candidate_mask, top_k)
        if not selected_indices:
            return self._fallback_weights(available, raw_weights, "no_candidate_asset")

        if weight_mode == "inverse_volatility":
            selected_mask = np.zeros(available.shape, dtype=bool)
            selected_mask[selected_indices] = True
            raw_weights[:] = inverse_volatility_weights(
                selected_mask,
                np.asarray(decision_market_state.volatility_20d_at_decision, dtype=float),
                _volatility_floor(config),
            )
        elif weight_mode == "softmax":
            active_scores = scores[selected_indices]
            raw_weights[selected_indices] = _softmax_weights(active_scores, _softmax_temperature(config))
        else:
            raw_weights[selected_indices] = 1.0 / len(selected_indices)

        if not np.isfinite(raw_weights).all() or float(raw_weights.sum()) <= OPT_EPS:
            return self._fallback_weights(available, raw_weights, "invalid_momentum_weights")

        self._momentum_report.update(
            {
                "selected_asset_indices": [int(index) for index in selected_indices],
                "score": np.where(np.isfinite(scores), scores, np.nan).astype(float).tolist(),
                "history_count": history_count.astype(int).tolist(),
                "raw_weights": raw_weights.astype(float).tolist(),
            }
        )
        return raw_weights

    def _fallback_weights(self, available: np.ndarray, raw_weights: np.ndarray, fallback_reason: str) -> np.ndarray:
        raw_weights[:] = 0.0
        raw_weights[available] = 1.0 / int(available.sum())
        self._momentum_report.update(
            {
                "fallback_reason": fallback_reason,
                "fallback": "equal_weight",
                "raw_weights": raw_weights.astype(float).tolist(),
            }
        )
        return raw_weights


def _momentum_scores(returns: np.ndarray, score_mode: str, volatility_floor: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(returns, dtype=float)
    n_assets = matrix.shape[1]
    scores = np.full(n_assets, np.nan, dtype=float)
    history_count = np.zeros(n_assets, dtype=int)
    for index in range(n_assets):
        series = matrix[:, index]
        finite = series[np.isfinite(series)]
        history_count[index] = int(finite.size)
        if finite.size < MIN_HISTORY_OBSERVATIONS:
            continue
        momentum = float(np.sum(finite))
        if score_mode == "risk_adjusted_momentum":
            volatility = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            scores[index] = momentum / max(volatility, volatility_floor)
        else:
            scores[index] = momentum
    return scores, history_count


def _select_top_indices(scores: np.ndarray, candidate_mask: np.ndarray, top_k: int) -> list[int]:
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return []
    candidate_scores = np.asarray(scores[candidate_indices], dtype=float)
    order = np.argsort(-candidate_scores, kind="mergesort")
    selected = candidate_indices[order[:top_k]]
    return [int(index) for index in selected]


def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(scores, dtype=float) / max(float(temperature), OPT_EPS)
    values = values - float(np.max(values))
    exp_values = np.exp(values)
    total = float(exp_values.sum())
    if total <= OPT_EPS or not np.isfinite(exp_values).all():
        return np.full(values.shape, 1.0 / values.shape[0], dtype=float)
    return exp_values / total


def _momentum_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("momentum"), Mapping):
        return dict(config["momentum"])
    return dict(config)


def _lookback_window(config: Mapping[str, Any], available_rows: int) -> int:
    configured = config.get("lookback_window", available_rows)
    lookback_window = int(configured)
    if lookback_window <= 0:
        lookback_window = available_rows
    return max(1, min(lookback_window, available_rows))


def _min_observations(config: Mapping[str, Any]) -> int:
    return max(MIN_HISTORY_OBSERVATIONS, int(config.get("min_observations", MIN_HISTORY_OBSERVATIONS)))


def _score_mode(config: Mapping[str, Any]) -> str:
    mode = str(config.get("score_mode", config.get("momentum_mode", config.get("mode", "momentum"))))
    if mode in {"risk_adjusted", "risk_adjusted_momentum"}:
        return "risk_adjusted_momentum"
    return "momentum"


def _top_k(config: Mapping[str, Any], available_count: int) -> int:
    configured = config.get("top_k", available_count)
    value = int(configured)
    if value <= 0:
        value = available_count
    return max(1, min(value, max(available_count, 1)))


def _threshold(config: Mapping[str, Any]) -> float | None:
    if "threshold" not in config or config["threshold"] is None:
        return None
    return float(config["threshold"])


def _weight_mode(config: Mapping[str, Any]) -> str:
    mode = str(config.get("weight_mode", config.get("allocation_mode", "equal")))
    if mode in {"equal", "equal_weight"}:
        return "equal"
    if mode in {"inverse_volatility", "inverse_vol"}:
        return "inverse_volatility"
    if mode in {"softmax", "softmax_score"}:
        return "softmax"
    return "equal"


def _volatility_floor(config: Mapping[str, Any]) -> float:
    return max(float(config.get("volatility_floor", DEFAULT_VOLATILITY_FLOOR)), DEFAULT_VOLATILITY_FLOOR)


def _softmax_temperature(config: Mapping[str, Any]) -> float:
    return max(float(config.get("softmax_temperature", DEFAULT_SOFTMAX_TEMPERATURE)), OPT_EPS)


__all__ = ["MomentumStrategy"]
