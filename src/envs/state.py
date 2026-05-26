from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.leakage_checks import assert_no_execution_field_in_observation
from src.data.loader import DataContractError


VALID_REBALANCE_ACTIONS = {0, 1}


@dataclass
class PortfolioState:
    date: pd.Timestamp
    nav: float
    portfolio_value: float
    current_weights: np.ndarray
    drifted_weights: np.ndarray | None = None
    previous_executed_weights: np.ndarray | None = None
    running_max_nav: float | None = None
    current_drawdown_abs: float = 0.0
    max_drawdown_abs: float = 0.0
    rolling_returns: list[float] = field(default_factory=list)
    step_index: int = 0
    last_buy_date_per_asset: np.ndarray | None = None
    last_valuation_price: np.ndarray | None = None
    sellable_mask: np.ndarray | None = None
    frozen_weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.date = _required_timestamp("date", self.date)
        self.nav = _positive_float("nav", self.nav)
        self.portfolio_value = _positive_float("portfolio_value", self.portfolio_value)
        self.current_weights = _array_1d("current_weights", self.current_weights)
        shape = self.current_weights.shape
        self.drifted_weights = _optional_array_1d(
            "drifted_weights",
            self.drifted_weights,
            shape,
            default=self.current_weights.copy(),
        )
        self.previous_executed_weights = _optional_array_1d(
            "previous_executed_weights",
            self.previous_executed_weights,
            shape,
            default=self.current_weights.copy(),
        )
        self.running_max_nav = _positive_float(
            "running_max_nav",
            self.nav if self.running_max_nav is None else self.running_max_nav,
        )
        self.current_drawdown_abs = _non_negative_float("current_drawdown_abs", self.current_drawdown_abs)
        self.max_drawdown_abs = max(
            _non_negative_float("max_drawdown_abs", self.max_drawdown_abs),
            self.current_drawdown_abs,
        )
        self.rolling_returns = [_finite_float("rolling_returns", value) for value in self.rolling_returns]
        if int(self.step_index) < 0:
            _raise_state_schema("step_index must be non-negative")
        self.step_index = int(self.step_index)
        self.last_buy_date_per_asset = _optional_object_array_1d(
            "last_buy_date_per_asset",
            self.last_buy_date_per_asset,
            shape,
        )
        self.last_valuation_price = _optional_array_1d(
            "last_valuation_price",
            self.last_valuation_price,
            shape,
            finite=False,
        )
        self.sellable_mask = _optional_bool_array_1d(
            "sellable_mask",
            self.sellable_mask,
            shape,
            default=np.ones(shape, dtype=bool),
        )
        self.frozen_weight = _optional_array_1d(
            "frozen_weight",
            self.frozen_weight,
            shape,
            default=np.zeros(shape, dtype=float),
        )


@dataclass
class PortfolioAction:
    target_weights: np.ndarray
    rebalance_action: int = 1
    rebalance_intensity: float | None = None
    action_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_weights = _action_weights(self.target_weights)
        self.rebalance_action = _rebalance_action(self.rebalance_action)
        if self.rebalance_intensity is None:
            self.rebalance_intensity = float(self.rebalance_action)
        self.rebalance_intensity = _bounded_float("rebalance_intensity", self.rebalance_intensity, 0.0, 1.0)
        self.action_info = dict(self.action_info)

    @property
    def weights(self) -> np.ndarray:
        return self.target_weights

    @property
    def rebalance(self) -> int:
        return self.rebalance_action


@dataclass
class PendingAction:
    decision_date: pd.Timestamp
    execution_date: pd.Timestamp
    next_valuation_date: pd.Timestamp
    target_weights: np.ndarray
    candidate_weights: np.ndarray
    rebalance_action: int
    rebalance_intensity: float
    execution_price: str
    execution_price_type: str
    q_hold: float | None = None
    q_rebalance: float | None = None
    q_gap: float | None = None
    decision_value: float | None = None
    action_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.decision_date = _required_timestamp("decision_date", self.decision_date)
        self.execution_date = _required_timestamp("execution_date", self.execution_date)
        self.next_valuation_date = _required_timestamp("next_valuation_date", self.next_valuation_date)
        if self.execution_date < self.decision_date or self.next_valuation_date < self.execution_date:
            _raise_state_schema("pending action date order")
        self.target_weights = _array_1d("target_weights", self.target_weights)
        self.candidate_weights = _array_1d("candidate_weights", self.candidate_weights, self.target_weights.shape)
        self.rebalance_action = _rebalance_action(self.rebalance_action)
        self.rebalance_intensity = _bounded_float("rebalance_intensity", self.rebalance_intensity, 0.0, 1.0)
        self.execution_price = _required_string("execution_price", self.execution_price)
        self.execution_price_type = _required_string("execution_price_type", self.execution_price_type)
        self.q_hold = _optional_finite_float("q_hold", self.q_hold)
        self.q_rebalance = _optional_finite_float("q_rebalance", self.q_rebalance)
        self.q_gap = _optional_finite_float("q_gap", self.q_gap)
        self.decision_value = _optional_finite_float("decision_value", self.decision_value)
        self.action_info = dict(self.action_info)


@dataclass
class DecisionMarketState:
    decision_date: pd.Timestamp
    available_mask_at_decision: np.ndarray
    availability_reason_at_decision: np.ndarray | None
    close_at_decision: np.ndarray
    log_return_at_decision: np.ndarray
    log_return_window: np.ndarray
    amount_at_decision: np.ndarray
    volume_at_decision: np.ndarray
    adv20_at_decision: np.ndarray
    volatility_20d_at_decision: np.ndarray
    turnover_rate_at_decision: np.ndarray
    feature_window: np.ndarray
    market_image: np.ndarray

    def __post_init__(self) -> None:
        assert_no_execution_field_in_observation(tuple(self.__dataclass_fields__))
        self.decision_date = _required_timestamp("decision_date", self.decision_date)
        self.available_mask_at_decision = _bool_array_1d("available_mask_at_decision", self.available_mask_at_decision)
        shape = self.available_mask_at_decision.shape
        self.availability_reason_at_decision = _optional_object_array_1d(
            "availability_reason_at_decision",
            self.availability_reason_at_decision,
            shape,
        )
        self.close_at_decision = _array_1d("close_at_decision", self.close_at_decision, shape, finite=False)
        self.log_return_at_decision = _array_1d("log_return_at_decision", self.log_return_at_decision, shape, finite=False)
        self.log_return_window = _array_with_asset_axis("log_return_window", self.log_return_window, shape[0], finite=False)
        self.amount_at_decision = _array_1d("amount_at_decision", self.amount_at_decision, shape, finite=False)
        self.volume_at_decision = _array_1d("volume_at_decision", self.volume_at_decision, shape, finite=False)
        self.adv20_at_decision = _array_1d("adv20_at_decision", self.adv20_at_decision, shape, finite=False)
        self.volatility_20d_at_decision = _array_1d(
            "volatility_20d_at_decision",
            self.volatility_20d_at_decision,
            shape,
            finite=False,
        )
        self.turnover_rate_at_decision = _array_1d(
            "turnover_rate_at_decision",
            self.turnover_rate_at_decision,
            shape,
            finite=False,
        )
        self.feature_window = _array_with_asset_axis("feature_window", self.feature_window, shape[0], finite=False)
        self.market_image = _array_with_asset_axis("market_image", self.market_image, shape[0], finite=False)


@dataclass
class ExecutionMarketState:
    decision_date: pd.Timestamp
    execution_date: pd.Timestamp
    next_valuation_date: pd.Timestamp
    execution_price_type: str
    execution_price: np.ndarray
    tradeable_mask_at_execution: np.ndarray
    availability_reason_at_execution: np.ndarray | None
    return_from_decision_to_execution: np.ndarray
    holding_simple_return: np.ndarray
    amount_at_execution: np.ndarray
    volume_at_execution: np.ndarray
    adv20_at_execution: np.ndarray
    volatility_20d_at_execution: np.ndarray
    valuation_price_at_decision: np.ndarray | None = None
    valuation_price_at_execution: np.ndarray | None = None
    valuation_price_at_next: np.ndarray | None = None
    turnover_rate_at_execution: np.ndarray | None = None
    cost_observation_date: pd.Timestamp | None = None
    cost_observation_timing: str | None = None
    amount_at_cost_observation: np.ndarray | None = None
    volume_at_cost_observation: np.ndarray | None = None
    adv20_at_cost_observation: np.ndarray | None = None
    volatility_20d_at_cost_observation: np.ndarray | None = None
    turnover_rate_at_cost_observation: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.decision_date = _required_timestamp("decision_date", self.decision_date)
        self.execution_date = _required_timestamp("execution_date", self.execution_date)
        self.next_valuation_date = _required_timestamp("next_valuation_date", self.next_valuation_date)
        if self.execution_date < self.decision_date or self.next_valuation_date < self.execution_date:
            _raise_state_schema("execution market state date order")
        self.execution_price_type = _required_string("execution_price_type", self.execution_price_type)
        self.execution_price = _array_1d("execution_price", self.execution_price, finite=False)
        shape = self.execution_price.shape
        self.tradeable_mask_at_execution = _bool_array_1d("tradeable_mask_at_execution", self.tradeable_mask_at_execution, shape)
        self.availability_reason_at_execution = _optional_object_array_1d(
            "availability_reason_at_execution",
            self.availability_reason_at_execution,
            shape,
        )
        self.return_from_decision_to_execution = _array_1d(
            "return_from_decision_to_execution",
            self.return_from_decision_to_execution,
            shape,
            finite=False,
        )
        self.holding_simple_return = _array_1d("holding_simple_return", self.holding_simple_return, shape, finite=False)
        self.valuation_price_at_decision = _array_1d(
            "valuation_price_at_decision",
            self.execution_price if self.valuation_price_at_decision is None else self.valuation_price_at_decision,
            shape,
            finite=False,
        )
        self.valuation_price_at_execution = _array_1d(
            "valuation_price_at_execution",
            self.execution_price if self.valuation_price_at_execution is None else self.valuation_price_at_execution,
            shape,
            finite=False,
        )
        self.valuation_price_at_next = _array_1d(
            "valuation_price_at_next",
            self.valuation_price_at_execution if self.valuation_price_at_next is None else self.valuation_price_at_next,
            shape,
            finite=False,
        )
        self.amount_at_execution = _array_1d("amount_at_execution", self.amount_at_execution, shape, finite=False)
        self.volume_at_execution = _array_1d("volume_at_execution", self.volume_at_execution, shape, finite=False)
        self.adv20_at_execution = _array_1d("adv20_at_execution", self.adv20_at_execution, shape, finite=False)
        self.volatility_20d_at_execution = _array_1d(
            "volatility_20d_at_execution",
            self.volatility_20d_at_execution,
            shape,
            finite=False,
        )
        if self.turnover_rate_at_execution is not None:
            self.turnover_rate_at_execution = _array_1d(
                "turnover_rate_at_execution",
                self.turnover_rate_at_execution,
                shape,
                finite=False,
            )
        self.cost_observation_date = _required_timestamp(
            "cost_observation_date",
            self.execution_date if self.cost_observation_date is None else self.cost_observation_date,
        )
        if self.cost_observation_timing is None:
            self.cost_observation_timing = (
                "decision_observable" if self.cost_observation_date <= self.decision_date else "execution_observed"
            )
        self.cost_observation_timing = _required_string("cost_observation_timing", self.cost_observation_timing)
        self.amount_at_cost_observation = _array_1d(
            "amount_at_cost_observation",
            self.amount_at_execution if self.amount_at_cost_observation is None else self.amount_at_cost_observation,
            shape,
            finite=False,
        )
        self.volume_at_cost_observation = _array_1d(
            "volume_at_cost_observation",
            self.volume_at_execution if self.volume_at_cost_observation is None else self.volume_at_cost_observation,
            shape,
            finite=False,
        )
        self.adv20_at_cost_observation = _array_1d(
            "adv20_at_cost_observation",
            self.adv20_at_execution if self.adv20_at_cost_observation is None else self.adv20_at_cost_observation,
            shape,
            finite=False,
        )
        self.volatility_20d_at_cost_observation = _array_1d(
            "volatility_20d_at_cost_observation",
            self.volatility_20d_at_execution
            if self.volatility_20d_at_cost_observation is None
            else self.volatility_20d_at_cost_observation,
            shape,
            finite=False,
        )
        if self.turnover_rate_at_cost_observation is None:
            self.turnover_rate_at_cost_observation = self.turnover_rate_at_execution
        if self.turnover_rate_at_cost_observation is not None:
            self.turnover_rate_at_cost_observation = _array_1d(
                "turnover_rate_at_cost_observation",
                self.turnover_rate_at_cost_observation,
                shape,
                finite=False,
            )


@dataclass
class ExecutionResult:
    executed_weights: np.ndarray
    pre_execution_drifted_weights: np.ndarray
    turnover: float
    transaction_cost: float
    transaction_cost_on_initial_nav: float
    proportional_cost: float
    fixed_cost: float
    slippage_cost: float
    market_impact_cost: float
    total_transaction_cost: float
    estimated_turnover: float | None
    realized_turnover: float
    estimated_cost: float | None
    realized_cost: float
    gross_return: float
    net_return: float
    pre_execution_return: float
    post_execution_return: float
    portfolio_log_return: float
    nav_execution: float
    nav_after_cost: float
    nav_next: float
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.executed_weights = _array_1d("executed_weights", self.executed_weights)
        self.pre_execution_drifted_weights = _array_1d(
            "pre_execution_drifted_weights",
            self.pre_execution_drifted_weights,
            self.executed_weights.shape,
        )
        for field_name in (
            "turnover",
            "transaction_cost",
            "transaction_cost_on_initial_nav",
            "proportional_cost",
            "fixed_cost",
            "slippage_cost",
            "market_impact_cost",
            "total_transaction_cost",
            "realized_turnover",
            "realized_cost",
            "gross_return",
            "net_return",
            "pre_execution_return",
            "post_execution_return",
            "portfolio_log_return",
        ):
            setattr(self, field_name, _finite_float(field_name, getattr(self, field_name)))
        self.estimated_turnover = _optional_finite_float("estimated_turnover", self.estimated_turnover)
        self.estimated_cost = _optional_finite_float("estimated_cost", self.estimated_cost)
        self.nav_execution = _positive_float("nav_execution", self.nav_execution)
        self.nav_after_cost = _positive_float("nav_after_cost", self.nav_after_cost)
        self.nav_next = _positive_float("nav_next", self.nav_next)
        self.info = dict(self.info)


def _required_timestamp(name: str, value: Any) -> pd.Timestamp:
    if value is None:
        _raise_state_schema(f"{name} is required")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}") from exc
    if pd.isna(timestamp):
        _raise_state_schema(f"{name} is required")
    return timestamp


def _required_string(name: str, value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        _raise_state_schema(f"{name} is required")
    return text


def _action_weights(values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: target_weights") from exc
    if array.ndim != 1:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", f"ERR_ACTION_SHAPE_MISMATCH: target_weights")
    if not np.isfinite(array).all():
        raise DataContractError("ERR_ACTION_NON_FINITE", "ERR_ACTION_NON_FINITE: target_weights")
    return array


def _array_1d(
    name: str,
    values: Any,
    shape: tuple[int, ...] | None = None,
    *,
    finite: bool = True,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}") from exc
    if array.ndim != 1:
        _raise_state_schema(f"{name} ndim")
    if shape is not None and array.shape != shape:
        _raise_state_schema(f"{name} shape")
    if finite and not np.isfinite(array).all():
        _raise_state_schema(f"{name} finite")
    return array


def _optional_array_1d(
    name: str,
    values: Any,
    shape: tuple[int, ...],
    *,
    default: np.ndarray | None = None,
    finite: bool = True,
) -> np.ndarray | None:
    if values is None:
        if default is None:
            return None
        return np.asarray(default, dtype=float)
    return _array_1d(name, values, shape, finite=finite)


def _bool_array_1d(name: str, values: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.ndim != 1:
        _raise_state_schema(f"{name} ndim")
    if shape is not None and array.shape != shape:
        _raise_state_schema(f"{name} shape")
    return array


def _optional_bool_array_1d(
    name: str,
    values: Any,
    shape: tuple[int, ...],
    *,
    default: np.ndarray,
) -> np.ndarray:
    if values is None:
        return np.asarray(default, dtype=bool)
    return _bool_array_1d(name, values, shape)


def _optional_object_array_1d(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or array.shape != shape:
        _raise_state_schema(f"{name} shape")
    return array


def _array_with_asset_axis(name: str, values: Any, n_assets: int, *, finite: bool = True) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}") from exc
    if array.ndim < 1:
        _raise_state_schema(f"{name} ndim")
    if array.shape[-1] != n_assets:
        _raise_state_schema(f"{name} asset axis")
    if finite and not np.isfinite(array).all():
        _raise_state_schema(f"{name} finite")
    return array


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}") from exc
    if not np.isfinite(result):
        _raise_state_schema(f"{name} finite")
    return result


def _optional_finite_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float(name, value)


def _positive_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        _raise_state_schema(f"{name} positive")
    return result


def _non_negative_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        _raise_state_schema(f"{name} non-negative")
    return result


def _bounded_float(name: str, value: Any, lower: float, upper: float) -> float:
    result = _finite_float(name, value)
    if result < lower or result > upper:
        _raise_state_schema(f"{name} bounds")
    return result


def _rebalance_action(value: Any) -> int:
    try:
        action = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: rebalance_action") from exc
    if action != numeric:
        _raise_state_schema("rebalance_action")
    if action not in VALID_REBALANCE_ACTIONS:
        _raise_state_schema("rebalance_action")
    return action


def _raise_state_schema(detail: str) -> None:
    raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {detail}")


__all__ = [
    "DecisionMarketState",
    "ExecutionMarketState",
    "ExecutionResult",
    "PendingAction",
    "PortfolioAction",
    "PortfolioState",
]
