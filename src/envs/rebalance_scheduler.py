from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError
from src.envs.state import DecisionMarketState, PortfolioState


VALID_REBALANCE_MODES = {
    "never",
    "once",
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "yearly",
    "every_n_days",
    "calendar_dates",
    "threshold_weight_drift",
    "threshold_turnover",
    "volatility_event",
    "drawdown_event",
    "risk_budget_breach",
}
FIRST_CALENDAR_RULES = {"first", "first_trading_day", "period_start"}
LAST_CALENDAR_RULES = {"last", "last_trading_day", "period_end"}
TRADING_DAYS_PER_YEAR = 252.0
WEIGHT_EPS = 1.0e-12


@dataclass(frozen=True)
class SchedulerDecisionEvaluation:
    scheduler_pre_allowed: bool
    scheduler_post_allowed: bool
    scheduler_final_allowed: bool


class RebalanceScheduler:
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        date_index: Sequence[Any] | pd.Index | None = None,
    ) -> None:
        self.raw_config = config or DEFAULT_CONFIG
        self.rebalance_config = _rebalance_config(config)
        self.date_index = _date_index(date_index)
        self._last_allowed_date: pd.Timestamp | None = None
        self._has_rebalanced = False

    def pre_check(
        self,
        date: Any,
        portfolio_state: PortfolioState,
        decision_market_state: DecisionMarketState,
        strategy_state: Any | None = None,
    ) -> bool:
        current_date = _timestamp(date)
        mode = self._mode()
        if mode == "never":
            return False
        if mode == "once":
            return not self._has_rebalanced
        if mode == "daily":
            return True
        if mode in {"weekly", "monthly", "quarterly", "yearly"}:
            return self._calendar_period_allowed(current_date, mode, strategy_state)
        if mode == "every_n_days":
            return self._every_n_days_allowed(current_date, strategy_state)
        if mode == "calendar_dates":
            return current_date.normalize() in self._calendar_dates()
        if mode == "volatility_event":
            return self._volatility_event(decision_market_state)
        if mode == "drawdown_event":
            return self._drawdown_event(portfolio_state)
        if mode == "risk_budget_breach":
            return self._risk_budget_breach(strategy_state)
        if mode in {"threshold_weight_drift", "threshold_turnover"}:
            return True
        return False

    def post_check(
        self,
        date: Any,
        portfolio_state: PortfolioState,
        decision_market_state: DecisionMarketState,
        strategy_state: Any | None = None,
        candidate_weights: np.ndarray | None = None,
    ) -> bool:
        mode = self._mode()
        if mode not in {"threshold_weight_drift", "threshold_turnover"}:
            return True
        if candidate_weights is None:
            return False

        candidate = _weights("candidate_weights", candidate_weights)
        current = _weights("portfolio_state.current_weights", portfolio_state.current_weights, candidate.shape)
        if mode == "threshold_weight_drift":
            threshold = _non_negative_config_float(
                "rebalance.threshold_weight_drift",
                self.rebalance_config.get("threshold_weight_drift", 0.0),
            )
            return bool(np.max(np.abs(candidate - current)) > threshold + WEIGHT_EPS)

        threshold = _non_negative_config_float(
            "rebalance.threshold_turnover",
            self.rebalance_config.get("threshold_turnover", 0.0),
        )
        turnover = 0.5 * float(np.sum(np.abs(candidate - current)))
        return bool(turnover > threshold + WEIGHT_EPS)

    def should_rebalance(
        self,
        date: Any,
        portfolio_state: PortfolioState,
        decision_market_state: DecisionMarketState,
        strategy_state: Any | None = None,
        candidate_weights: np.ndarray | None = None,
        *,
        gate_action: int | None = None,
    ) -> bool:
        current_date = _timestamp(date)
        scheduler_allowed = self.pre_check(current_date, portfolio_state, decision_market_state, strategy_state)
        if scheduler_allowed:
            scheduler_allowed = self.post_check(
                current_date,
                portfolio_state,
                decision_market_state,
                strategy_state,
                candidate_weights,
            )
        scheduler_allowed_before_gate = bool(scheduler_allowed)
        if gate_action is not None:
            scheduler_allowed = bool(self.final_rebalance_action(scheduler_allowed, gate_action))
        if scheduler_allowed_before_gate:
            self._last_allowed_date = current_date
            self._has_rebalanced = True
        return bool(scheduler_allowed)

    def evaluate_pre_post_no_mutation(
        self,
        date: Any,
        portfolio_state: PortfolioState,
        decision_market_state: DecisionMarketState,
        strategy_state: Any | None = None,
        candidate_weights: np.ndarray | None = None,
    ) -> SchedulerDecisionEvaluation:
        current_date = _timestamp(date)
        scheduler_pre_allowed = self.pre_check(
            current_date,
            portfolio_state,
            decision_market_state,
            strategy_state,
        )
        scheduler_post_allowed = False
        if scheduler_pre_allowed:
            scheduler_post_allowed = self.post_check(
                current_date,
                portfolio_state,
                decision_market_state,
                strategy_state,
                candidate_weights,
            )
        return SchedulerDecisionEvaluation(
            scheduler_pre_allowed=bool(scheduler_pre_allowed),
            scheduler_post_allowed=bool(scheduler_post_allowed),
            scheduler_final_allowed=bool(scheduler_pre_allowed and scheduler_post_allowed),
        )

    def commit_scheduler_decision(
        self,
        decision_ts: Any,
        *,
        scheduler_pre_allowed: bool,
        scheduler_post_allowed: bool,
        scheduler_final_allowed: bool,
        raw_model_requested_rebalance: bool,
        final_action: bool | int,
        execution_accepted: bool,
    ) -> None:
        _ = bool(raw_model_requested_rebalance), bool(final_action), bool(execution_accepted)
        if bool(scheduler_pre_allowed) and bool(scheduler_post_allowed) and bool(scheduler_final_allowed):
            self._last_allowed_date = _timestamp(decision_ts)
            self._has_rebalanced = True

    @staticmethod
    def final_rebalance_action(scheduler_allowed: bool, gate_action: int | bool) -> int:
        try:
            gate = int(gate_action)
        except (TypeError, ValueError) as exc:
            raise DataContractError(
                "ERR_STATE_SCHEMA_MISMATCH",
                "ERR_STATE_SCHEMA_MISMATCH: gate_action",
            ) from exc
        if gate not in {0, 1}:
            raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: gate_action")
        return int(bool(scheduler_allowed) and gate == 1)

    def reset(self) -> None:
        self._last_allowed_date = None
        self._has_rebalanced = False

    def _mode(self) -> str:
        mode = str(self.rebalance_config.get("mode", "monthly"))
        if mode not in VALID_REBALANCE_MODES:
            raise DataContractError(
                "ERR_CONFIG_INVALID_REBALANCE_MODE",
                "ERR_CONFIG_INVALID_REBALANCE_MODE: rebalance.mode",
            )
        return mode

    def _calendar_period_allowed(self, date: pd.Timestamp, mode: str, strategy_state: Any | None) -> bool:
        date_index = self._resolved_date_index(strategy_state)
        if date_index.empty:
            return _calendar_fallback(date, mode, self._calendar_rule())

        if date.normalize() not in set(date_index.normalize()):
            return False
        dates = pd.Index(date_index[date_index.map(lambda item: _same_period(item, date, mode))])
        if dates.empty:
            return False
        if self._calendar_rule() in FIRST_CALENDAR_RULES:
            return date.normalize() == dates.min().normalize()
        return date.normalize() == dates.max().normalize()

    def _calendar_rule(self) -> str:
        default_position = DEFAULT_CONFIG["rebalance"]["calendar_position"]
        configured_position = self.rebalance_config.get("calendar_position", default_position)
        if configured_position != default_position:
            value = configured_position
        else:
            value = self.rebalance_config.get("calendar_rule", default_position)
        rule = str(value)
        if rule in FIRST_CALENDAR_RULES | LAST_CALENDAR_RULES:
            return rule
        raise DataContractError(
            "ERR_CONFIG_INVALID_REBALANCE_MODE",
            "ERR_CONFIG_INVALID_REBALANCE_MODE: rebalance.calendar_position",
        )

    def _every_n_days_allowed(self, date: pd.Timestamp, strategy_state: Any | None) -> bool:
        try:
            n_days = int(self.rebalance_config.get("every_n_days", 1))
        except (TypeError, ValueError) as exc:
            raise DataContractError(
                "ERR_CONFIG_INVALID_REBALANCE_MODE",
                "ERR_CONFIG_INVALID_REBALANCE_MODE: rebalance.every_n_days",
            ) from exc
        if n_days <= 0:
            raise DataContractError(
                "ERR_CONFIG_INVALID_REBALANCE_MODE",
                "ERR_CONFIG_INVALID_REBALANCE_MODE: rebalance.every_n_days",
            )
        if self._last_allowed_date is None:
            return True

        date_index = self._resolved_date_index(strategy_state)
        if not date_index.empty:
            positions = {value.normalize(): index for index, value in enumerate(date_index)}
            current_pos = positions.get(date.normalize())
            last_pos = positions.get(self._last_allowed_date.normalize())
            if current_pos is None or last_pos is None:
                return False
            return current_pos - last_pos >= n_days
        return (date.normalize() - self._last_allowed_date.normalize()).days >= n_days

    def _calendar_dates(self) -> set[pd.Timestamp]:
        return {_timestamp(value).normalize() for value in self.rebalance_config.get("calendar_dates", [])}

    def _volatility_event(self, decision_market_state: DecisionMarketState) -> bool:
        threshold = _non_negative_config_float(
            "rebalance.volatility_threshold_annual",
            self.rebalance_config.get("volatility_threshold_annual", 0.25),
        )
        volatility = np.asarray(decision_market_state.volatility_20d_at_decision, dtype=float)
        if volatility.size == 0 or not np.isfinite(volatility).any():
            return False
        annualized = float(np.nanmax(volatility) * np.sqrt(TRADING_DAYS_PER_YEAR))
        return bool(annualized > threshold + WEIGHT_EPS)

    def _drawdown_event(self, portfolio_state: PortfolioState) -> bool:
        threshold = _non_negative_config_float(
            "rebalance.drawdown_threshold",
            self.rebalance_config.get("drawdown_threshold", 0.10),
        )
        return bool(float(portfolio_state.current_drawdown_abs) > threshold + WEIGHT_EPS)

    def _risk_budget_breach(self, strategy_state: Any | None) -> bool:
        if _state_get(strategy_state, "risk_budget_breach", False):
            return True
        threshold = _non_negative_config_float(
            "rebalance.risk_budget_tolerance",
            self.rebalance_config.get("risk_budget_tolerance", 0.05),
        )
        deviation = _state_get(strategy_state, "risk_budget_deviation")
        if deviation is not None:
            return _max_abs(deviation) > threshold + WEIGHT_EPS
        current = _state_get(strategy_state, "current_risk_budget")
        target = _state_get(strategy_state, "target_risk_budget")
        if current is None or target is None:
            return False
        return _max_abs(np.asarray(current, dtype=float) - np.asarray(target, dtype=float)) > threshold + WEIGHT_EPS

    def _resolved_date_index(self, strategy_state: Any | None) -> pd.DatetimeIndex:
        state_date_index = _state_get(strategy_state, "date_index")
        if state_date_index is not None:
            return _date_index(state_date_index)
        return self.date_index


def _rebalance_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_CONFIG["rebalance"]
    if config is None:
        return dict(source)
    if "rebalance" in config:
        return {**source, **dict(config["rebalance"])}
    return {**source, **dict(config)}


def _timestamp(value: Any) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: date") from exc
    if pd.isna(result):
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: date")
    return result


def _date_index(values: Sequence[Any] | pd.Index | None) -> pd.DatetimeIndex:
    if values is None:
        return pd.DatetimeIndex([])
    index = pd.DatetimeIndex(values)
    if index.empty:
        return index
    return pd.DatetimeIndex(sorted(index.normalize().unique()))


def _weights(name: str, values: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}")
    if shape is not None and array.shape != shape:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}")
    if not np.isfinite(array).all():
        raise DataContractError("ERR_ACTION_NON_FINITE", f"ERR_ACTION_NON_FINITE: {name}")
    return array


def _non_negative_config_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_CONFIG_INVALID_REBALANCE_MODE", f"ERR_CONFIG_INVALID_REBALANCE_MODE: {name}") from exc
    if result < 0.0 or not np.isfinite(result):
        raise DataContractError("ERR_CONFIG_INVALID_REBALANCE_MODE", f"ERR_CONFIG_INVALID_REBALANCE_MODE: {name}")
    return result


def _state_get(state: Any | None, key: str, default: Any = None) -> Any:
    if state is None:
        return default
    if isinstance(state, Mapping):
        return state.get(key, default)
    return getattr(state, key, default)


def _same_period(left: pd.Timestamp, right: pd.Timestamp, mode: str) -> bool:
    if mode == "weekly":
        return left.isocalendar().year == right.isocalendar().year and left.isocalendar().week == right.isocalendar().week
    if mode == "monthly":
        return left.year == right.year and left.month == right.month
    if mode == "quarterly":
        return left.year == right.year and left.quarter == right.quarter
    if mode == "yearly":
        return left.year == right.year
    return False


def _calendar_fallback(date: pd.Timestamp, mode: str, rule: str) -> bool:
    if mode == "weekly":
        return date.weekday() == (0 if rule in FIRST_CALENDAR_RULES else 4)
    if mode == "monthly":
        return bool(date.is_month_start if rule in FIRST_CALENDAR_RULES else date.is_month_end)
    if mode == "quarterly":
        return bool(date.is_quarter_start if rule in FIRST_CALENDAR_RULES else date.is_quarter_end)
    if mode == "yearly":
        return bool(date.is_year_start if rule in FIRST_CALENDAR_RULES else date.is_year_end)
    return False


def _max_abs(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).any():
        return 0.0
    return float(np.nanmax(np.abs(array)))
