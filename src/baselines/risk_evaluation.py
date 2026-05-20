from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.baselines.base_strategy import TraditionalStrategyBase
from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


DEFAULT_ADV_EPS = float(DEFAULT_CONFIG["cost_model"]["adv_eps"])
MIN_HISTORY_OBSERVATIONS = 2
DEFAULT_CVAR_ALPHA = 0.05
SCORE_EPS = 1.0e-12


class RiskEvaluationStrategy(TraditionalStrategyBase):
    strategy_name = "risk_evaluation"
    fit_required = True

    def fit(
        self,
        train_data: Any | None = None,
        validation_data: Any | None = None,
    ) -> RiskEvaluationStrategy:
        super().fit(train_data, validation_data)
        config = _risk_evaluation_config(self.config)
        self._train_liquidity_median = _train_liquidity_median(train_data, self.config, config)
        self._train_turnover_mean, self._train_turnover_std = _train_turnover_stats(train_data)
        return self

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self._risk_evaluation_report: dict[str, Any] = {}
        action = super().compute_target_weights(decision_market_state, portfolio_state)
        report = dict(self._risk_evaluation_report)
        report["projected_weights"] = action.target_weights.astype(float).tolist()
        action.action_info["risk_evaluation"] = report
        if report.get("fallback_reason"):
            action.action_info["fallback_reason"] = report["fallback_reason"]
        return action

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        config = _risk_evaluation_config(self.config)
        available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
        raw_weights = np.zeros(available.shape, dtype=float)
        lookback_window = _lookback_window(config, decision_market_state.log_return_window.shape[0])
        self._risk_evaluation_report = {
            "lookback_window": lookback_window,
            "score_terms": ["return", "volatility", "max_drawdown", "downside", "cvar", "liquidity"],
        }

        if not available.any():
            self._risk_evaluation_report["fallback_reason"] = "no_available_asset"
            return raw_weights
        if int(available.sum()) == 1:
            raw_weights[available] = 1.0
            self._risk_evaluation_report["raw_weights"] = raw_weights.astype(float).tolist()
            return raw_weights

        returns = np.asarray(decision_market_state.log_return_window[-lookback_window:, :], dtype=float)
        active_returns = returns[:, available]
        if active_returns.shape[0] < _min_observations(config):
            raw_weights[available] = 1.0 / int(available.sum())
            self._risk_evaluation_report.update(
                {
                    "fallback_reason": "insufficient_history",
                    "fallback": "equal_weight",
                    "raw_weights": raw_weights.astype(float).tolist(),
                }
            )
            return raw_weights

        metrics = _asset_risk_metrics(active_returns, _cvar_alpha(config))
        liquidity_values, liquidity_source, liquidity_fallback = _liquidity_values(
            decision_market_state,
            available,
            self.config,
            config,
            getattr(self, "_train_liquidity_median", None),
            getattr(self, "_train_turnover_mean", None),
            getattr(self, "_train_turnover_std", None),
        )
        component_scores = {
            "return": _cross_section_score(metrics["return"], higher_is_better=True),
            "volatility": _cross_section_score(metrics["volatility"], higher_is_better=False),
            "max_drawdown": _cross_section_score(metrics["max_drawdown"], higher_is_better=False),
            "downside": _cross_section_score(metrics["downside"], higher_is_better=False),
            "cvar": _cross_section_score(metrics["cvar"], higher_is_better=False),
            "liquidity": _cross_section_score(liquidity_values, higher_is_better=True)
            if liquidity_fallback is None
            else np.zeros(int(available.sum()), dtype=float),
        }
        score = (
            _return_weight(config) * component_scores["return"]
            + _liquidity_weight(config) * component_scores["liquidity"]
            + _volatility_weight(config) * component_scores["volatility"]
            + _max_drawdown_weight(config) * component_scores["max_drawdown"]
            + _downside_weight(config) * component_scores["downside"]
            + _cvar_weight(config) * component_scores["cvar"]
        )
        active_weights = _weights_from_score(score)
        raw_weights[available] = active_weights

        self._risk_evaluation_report.update(
            {
                "liquidity_source": liquidity_source,
                "adv_eps": _adv_eps(self.config, config),
                "metrics": {key: value.astype(float).tolist() for key, value in metrics.items()},
                "liquidity": liquidity_values.astype(float).tolist(),
                "component_scores": {key: value.astype(float).tolist() for key, value in component_scores.items()},
                "score": score.astype(float).tolist(),
                "raw_weights": raw_weights.astype(float).tolist(),
            }
        )
        if liquidity_fallback is not None:
            self._risk_evaluation_report["fallback_reason"] = liquidity_fallback
        return raw_weights


def _asset_risk_metrics(returns: np.ndarray, cvar_alpha: float) -> dict[str, np.ndarray]:
    matrix = np.asarray(returns, dtype=float)
    n_assets = matrix.shape[1]
    mean_return = np.zeros(n_assets, dtype=float)
    volatility = np.zeros(n_assets, dtype=float)
    max_drawdown = np.zeros(n_assets, dtype=float)
    downside = np.zeros(n_assets, dtype=float)
    cvar = np.zeros(n_assets, dtype=float)
    for index in range(n_assets):
        series = matrix[:, index]
        finite = series[np.isfinite(series)]
        if finite.size == 0:
            continue
        mean_return[index] = float(np.mean(finite))
        volatility[index] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        negative = finite[finite < 0.0]
        downside[index] = float(np.std(negative, ddof=1)) if negative.size > 1 else 0.0
        cvar[index] = _cvar_loss(finite, cvar_alpha)
        cumulative_nav = np.exp(np.cumsum(finite))
        running_max = np.maximum.accumulate(cumulative_nav)
        drawdown = np.maximum(0.0, 1.0 - cumulative_nav / np.maximum(running_max, SCORE_EPS))
        max_drawdown[index] = float(np.max(drawdown)) if drawdown.size else 0.0
    return {
        "return": mean_return,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "downside": downside,
        "cvar": cvar,
    }


def _liquidity_values(
    decision_market_state: DecisionMarketState,
    available: np.ndarray,
    root_config: Mapping[str, Any],
    config: Mapping[str, Any],
    train_liquidity_median: np.ndarray | None,
    train_turnover_mean: np.ndarray | None,
    train_turnover_std: np.ndarray | None,
) -> tuple[np.ndarray, str, str | None]:
    turnover = np.asarray(decision_market_state.turnover_rate_at_decision, dtype=float)[available]
    if np.isfinite(turnover).any():
        active_mean = _active_train_values(train_turnover_mean, available)
        active_std = _active_train_values(train_turnover_std, available)
        if active_mean is not None and active_std is not None and np.isfinite(active_mean).any():
            values = np.zeros(turnover.shape, dtype=float)
            valid = np.isfinite(turnover) & np.isfinite(active_mean) & np.isfinite(active_std) & (active_std > SCORE_EPS)
            values[valid] = (turnover[valid] - active_mean[valid]) / np.maximum(active_std[valid], SCORE_EPS)
            return values, "turnover_rate_train_zscore", None
        values = np.where(np.isfinite(turnover), turnover, np.nan)
        return _fill_missing_with_median(values), "turnover_rate_unfitted", None
    if _turnover_rate_required(root_config, config):
        raise DataContractError(
            "ERR_DATA_TURNOVER_RATE_REQUIRED",
            "ERR_DATA_TURNOVER_RATE_REQUIRED: turnover_rate_at_decision",
        )

    amount = np.asarray(decision_market_state.amount_at_decision, dtype=float)[available]
    adv20 = np.asarray(decision_market_state.adv20_at_decision, dtype=float)[available]
    adv_eps = _adv_eps(root_config, config)
    valid = np.isfinite(amount) & np.isfinite(adv20) & (adv20 > 0.0)
    ratio = np.full(amount.shape, np.nan, dtype=float)
    if valid.any():
        ratio[valid] = amount[valid] / np.maximum(adv20[valid], adv_eps)
    active_train_liquidity = _active_train_values(train_liquidity_median, available)
    missing = ~np.isfinite(ratio)
    if missing.any() and active_train_liquidity is not None:
        train_valid = missing & np.isfinite(active_train_liquidity)
        ratio[train_valid] = active_train_liquidity[train_valid]
    if np.isfinite(ratio).all():
        source = "amount_adv20" if valid.all() else "amount_adv20_train_median"
        return ratio, source, None
    return np.zeros(int(available.sum()), dtype=float), "neutral", "liquidity_unavailable"


def _train_liquidity_median(
    train_data: Any | None,
    root_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray | None:
    amount = _train_frame(train_data, "amount")
    if amount is None:
        return None
    adv_eps = _adv_eps(root_config, config)
    amount_values = amount.to_numpy(dtype=float, copy=True)
    adv20_values = amount.rolling(20, min_periods=1).mean().to_numpy(dtype=float, copy=True)
    valid = np.isfinite(amount_values) & np.isfinite(adv20_values) & (adv20_values > 0.0)
    ratio = np.full(amount_values.shape, np.nan, dtype=float)
    ratio[valid] = amount_values[valid] / np.maximum(adv20_values[valid], adv_eps)
    return _nanmedian_by_column(ratio)


def _train_turnover_stats(train_data: Any | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    turnover = _train_frame(train_data, "turnover_rate")
    if turnover is None:
        return None, None
    values = turnover.to_numpy(dtype=float, copy=True)
    return _nanmean_by_column(values), _nanstd_by_column(values)


def _train_frame(train_data: Any | None, field: str) -> Any | None:
    if not isinstance(train_data, Mapping):
        return None
    dataset = train_data.get("dataset")
    wide = getattr(dataset, "wide", None)
    if not isinstance(wide, Mapping) or field not in wide:
        return None
    frame = wide[field]
    asset_ids = _train_asset_order(dataset)
    dates = train_data.get("dates")
    if asset_ids:
        frame = frame.reindex(columns=asset_ids)
    if dates is not None:
        frame = frame.reindex(index=dates)
    return frame


def _train_asset_order(dataset: Any | None) -> list[str]:
    if dataset is None:
        return []
    manifest = getattr(dataset, "data_manifest", {})
    if isinstance(manifest, Mapping):
        asset_order = manifest.get("canonical_asset_order")
        if isinstance(asset_order, list) and asset_order:
            return [str(value) for value in asset_order]
    availability = getattr(dataset, "availability_mask", None)
    columns = getattr(availability, "columns", None)
    if columns is not None:
        return [str(value) for value in columns]
    return []


def _active_train_values(values: np.ndarray | None, available: np.ndarray) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != available.shape:
        return None
    return array[available]


def _nanmedian_by_column(values: np.ndarray) -> np.ndarray | None:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        return None
    result = np.full(array.shape[1], np.nan, dtype=float)
    for index in range(array.shape[1]):
        finite = array[:, index][np.isfinite(array[:, index])]
        if finite.size:
            result[index] = float(np.median(finite))
    return result


def _nanmean_by_column(values: np.ndarray) -> np.ndarray | None:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        return None
    result = np.full(array.shape[1], np.nan, dtype=float)
    for index in range(array.shape[1]):
        finite = array[:, index][np.isfinite(array[:, index])]
        if finite.size:
            result[index] = float(np.mean(finite))
    return result


def _nanstd_by_column(values: np.ndarray) -> np.ndarray | None:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        return None
    result = np.full(array.shape[1], np.nan, dtype=float)
    for index in range(array.shape[1]):
        finite = array[:, index][np.isfinite(array[:, index])]
        if finite.size:
            result[index] = float(np.std(finite, ddof=0))
    return result


def _fill_missing_with_median(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=float)
    result = array.copy()
    result[~np.isfinite(result)] = float(np.median(finite))
    return result


def _cross_section_score(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if array.size == 0 or finite.size == 0:
        return np.zeros(array.shape, dtype=float)
    result = array.copy()
    median = float(np.median(finite))
    result[~np.isfinite(result)] = median
    min_value = float(np.min(result))
    max_value = float(np.max(result))
    if max_value - min_value <= SCORE_EPS:
        return np.zeros(array.shape, dtype=float)
    scaled = (result - min_value) / (max_value - min_value)
    if higher_is_better:
        return scaled
    return 1.0 - scaled


def _weights_from_score(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    shifted = values - float(np.min(values)) + SCORE_EPS
    total = float(shifted.sum())
    if total <= SCORE_EPS:
        return np.full(values.shape, 1.0 / values.shape[0], dtype=float)
    return shifted / total


def _cvar_loss(returns: np.ndarray, cvar_alpha: float) -> float:
    sorted_returns = np.sort(np.asarray(returns, dtype=float))
    tail_count = max(1, int(np.ceil(float(cvar_alpha) * sorted_returns.size)))
    return max(0.0, -float(np.mean(sorted_returns[:tail_count])))


def _risk_evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("risk_evaluation"), Mapping):
        return dict(config["risk_evaluation"])
    return dict(config)


def _lookback_window(config: Mapping[str, Any], available_rows: int) -> int:
    configured = config.get("lookback_window", available_rows)
    lookback_window = int(configured)
    if lookback_window <= 0:
        lookback_window = available_rows
    return max(1, min(lookback_window, available_rows))


def _min_observations(config: Mapping[str, Any]) -> int:
    return max(MIN_HISTORY_OBSERVATIONS, int(config.get("min_observations", MIN_HISTORY_OBSERVATIONS)))


def _cvar_alpha(config: Mapping[str, Any]) -> float:
    return float(np.clip(float(config.get("cvar_alpha", DEFAULT_CVAR_ALPHA)), SCORE_EPS, 1.0))


def _turnover_rate_required(root_config: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if "turnover_rate_required" in config:
        return bool(config["turnover_rate_required"])
    data_governance = root_config.get("data_governance")
    if isinstance(data_governance, Mapping):
        return bool(data_governance.get("turnover_rate_required", False))
    return False


def _adv_eps(root_config: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    if "adv_eps" in config:
        return max(float(config["adv_eps"]), SCORE_EPS)
    cost_model = root_config.get("cost_model")
    if isinstance(cost_model, Mapping):
        return max(float(cost_model.get("adv_eps", DEFAULT_ADV_EPS)), SCORE_EPS)
    return DEFAULT_ADV_EPS


def _return_weight(config: Mapping[str, Any]) -> float:
    return float(config.get("return_weight", 1.0))


def _liquidity_weight(config: Mapping[str, Any]) -> float:
    return float(config.get("liquidity_weight", 0.2))


def _volatility_weight(config: Mapping[str, Any]) -> float:
    return float(config.get("volatility_weight", 0.5))


def _max_drawdown_weight(config: Mapping[str, Any]) -> float:
    return float(config.get("max_drawdown_weight", 0.5))


def _downside_weight(config: Mapping[str, Any]) -> float:
    return float(config.get("downside_weight", 0.3))


def _cvar_weight(config: Mapping[str, Any]) -> float:
    return float(config.get("cvar_weight", 0.3))


__all__ = ["RiskEvaluationStrategy"]
