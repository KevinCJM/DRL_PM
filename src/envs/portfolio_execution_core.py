from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError, MarketDatasetBundle
from src.envs.constraint_manager import ConstraintManager
from src.envs.cost_model import CostModel
from src.envs.state import ExecutionMarketState, ExecutionResult, PendingAction, PortfolioState


VALID_EXECUTION_PRICES = {"next_open", "next_close"}
ROLLING_WINDOW = 20
WEIGHT_SUM_EPS = 1.0e-12
RETURN_IMPUTATION_REASONS = {"suspended", "missing_quote", "missing_return"}


class PortfolioExecutionCore:
    def __init__(self, config: Mapping[str, Any] | None = None, cost_model: CostModel | None = None) -> None:
        self.raw_config = config or DEFAULT_CONFIG
        self.execution_config = _execution_model_config(config)
        self.data_governance_config = _data_governance_config(config)
        self.constraint_config = _constraint_config(config)
        self.cost_model = cost_model or CostModel(config)
        self.constraint_manager = ConstraintManager(config)
        self.execution_manifest_flags: dict[str, Any] = {}

    def build_execution_market_state(
        self,
        dataset: MarketDatasetBundle,
        decision_date: Any | None = None,
        *,
        pending_action: PendingAction | None = None,
    ) -> ExecutionMarketState:
        execution_price = _pending_or_config_execution_price(pending_action, self.execution_config)
        same_close_enabled = _same_close_enabled(self.execution_config, self.data_governance_config)
        if execution_price not in VALID_EXECUTION_PRICES:
            raise DataContractError(
                "ERR_CONFIG_INVALID_EXECUTION_MODEL",
                "ERR_CONFIG_INVALID_EXECUTION_MODEL: execution_model.execution_price",
            )
        if execution_price == "next_close" and self.execution_config.get("delayed_action_execution") is not True:
            raise DataContractError(
                "ERR_CONFIG_INVALID_EXECUTION_MODEL",
                "ERR_CONFIG_INVALID_EXECUTION_MODEL: execution_model.delayed_action_execution",
            )

        decision_ts = _decision_date(decision_date, pending_action)
        date_index = _date_index(dataset)
        decision_pos = _date_position(date_index, decision_ts)
        close = _wide_table(dataset, "close")
        valuation = _valuation_table(dataset, self.data_governance_config)
        valuation_source = "adj_nav" if valuation is not close else "close"
        valuation_price_at_decision = _row(valuation, decision_ts, valuation_source)

        if same_close_enabled:
            execution_ts = decision_ts
            valuation_ts = _date_at(date_index, decision_pos + 1)
            execution_price_type = "close"
            execution_price_array = _row(close, execution_ts, "close")
            valuation_price_at_execution = _row(valuation, execution_ts, valuation_source)
            valuation_price_at_next = _row(valuation, valuation_ts, valuation_source)
            return_from_decision_to_execution = np.zeros_like(execution_price_array, dtype=float)
            holding_simple_return = _safe_return(
                valuation_price_at_next,
                valuation_price_at_execution,
            )
        elif execution_price == "next_open":
            execution_ts = _date_at(date_index, decision_pos + 1)
            valuation_ts = execution_ts
            execution_price_type = "open"
            open_table = _wide_table(dataset, "open")
            execution_price_array = _row(open_table, execution_ts, "open")
            if valuation_source == "adj_nav":
                valuation_price_at_execution = _row(valuation, execution_ts, valuation_source)
                valuation_price_at_next = valuation_price_at_execution.copy()
                return_from_decision_to_execution = _safe_return(
                    valuation_price_at_execution,
                    valuation_price_at_decision,
                )
                holding_simple_return = np.zeros_like(execution_price_array, dtype=float)
            else:
                valuation_price_at_execution = execution_price_array.copy()
                valuation_price_at_next = _row(close, valuation_ts, "close")
                return_from_decision_to_execution = _safe_return(
                    execution_price_array,
                    _row(close, decision_ts, "close"),
                )
                holding_simple_return = _safe_return(
                    valuation_price_at_next,
                    execution_price_array,
                )
        else:
            if pending_action is None:
                execution_ts = _date_at(date_index, decision_pos + 1)
                valuation_ts = _date_at(date_index, decision_pos + 2)
            else:
                execution_ts = pending_action.execution_date
                valuation_ts = pending_action.next_valuation_date
                _date_position(date_index, execution_ts)
                _date_position(date_index, valuation_ts)
            execution_price_type = "close"
            execution_price_array = _row(close, execution_ts, "close")
            valuation_price_at_execution = _row(valuation, execution_ts, valuation_source)
            valuation_price_at_next = _row(valuation, valuation_ts, valuation_source)
            return_from_decision_to_execution = _safe_return(
                valuation_price_at_execution,
                valuation_price_at_decision,
            )
            holding_simple_return = _safe_return(
                valuation_price_at_next,
                valuation_price_at_execution,
            )

        cost_observation_ts = decision_ts if execution_price == "next_open" and not same_close_enabled else execution_ts
        cost_observation_timing = "decision_observable" if cost_observation_ts <= decision_ts else "execution_observed"
        amount_at_cost_observation = _row(_wide_table(dataset, "amount"), cost_observation_ts, "amount")
        volume_at_cost_observation = _row(_wide_table(dataset, "vol"), cost_observation_ts, "vol")
        adv20_at_cost_observation = _rolling_mean_row(_wide_table(dataset, "amount"), cost_observation_ts, "amount")
        volatility_20d_at_cost_observation = _rolling_volatility_row(dataset, cost_observation_ts)
        turnover_rate_at_cost_observation = _optional_row(dataset, "turnover_rate", cost_observation_ts)
        self.execution_manifest_flags = {
            "execution_price": execution_price,
            "execution_price_type": execution_price_type,
            "delayed_action_execution": bool(self.execution_config.get("delayed_action_execution", False)),
            "same_close_idealized_execution_enabled": bool(same_close_enabled),
            "idealized_execution": bool(same_close_enabled),
            "cost_observation_date": str(cost_observation_ts.date()),
            "cost_observation_timing": cost_observation_timing,
            "valuation_source": valuation_source,
            "return_source": valuation_source,
            "valuation_execution_split": bool(valuation_source == "adj_nav"),
            "reward_valuation_split": bool(valuation_source == "adj_nav"),
        }
        return ExecutionMarketState(
            decision_date=decision_ts,
            execution_date=execution_ts,
            next_valuation_date=valuation_ts,
            execution_price_type=execution_price_type,
            execution_price=execution_price_array,
            tradeable_mask_at_execution=_tradeable_mask(dataset, execution_ts),
            availability_reason_at_execution=_availability_reason(dataset, execution_ts),
            return_from_decision_to_execution=return_from_decision_to_execution,
            holding_simple_return=holding_simple_return,
            valuation_price_at_decision=valuation_price_at_decision,
            valuation_price_at_execution=valuation_price_at_execution,
            valuation_price_at_next=valuation_price_at_next,
            amount_at_execution=amount_at_cost_observation,
            volume_at_execution=volume_at_cost_observation,
            adv20_at_execution=adv20_at_cost_observation,
            volatility_20d_at_execution=volatility_20d_at_cost_observation,
            turnover_rate_at_execution=turnover_rate_at_cost_observation,
            cost_observation_date=cost_observation_ts,
            cost_observation_timing=cost_observation_timing,
            amount_at_cost_observation=amount_at_cost_observation,
            volume_at_cost_observation=volume_at_cost_observation,
            adv20_at_cost_observation=adv20_at_cost_observation,
            volatility_20d_at_cost_observation=volatility_20d_at_cost_observation,
            turnover_rate_at_cost_observation=turnover_rate_at_cost_observation,
        )

    def execute_step(
        self,
        decision_weights: np.ndarray,
        target_weights: np.ndarray,
        execution_market_state: ExecutionMarketState,
        portfolio_state: PortfolioState,
        *,
        rebalance_action: int = 1,
        rebalance_intensity: float = 1.0,
        asset_ids: np.ndarray | list[Any] | None = None,
        estimated_turnover: Any | None = None,
        estimated_cost: Any | None = None,
    ) -> ExecutionResult:
        decision = _array_1d("decision_weights", decision_weights)
        target = _array_1d("target_weights", target_weights, decision.shape)
        action = _rebalance_action(rebalance_action)
        intensity = _bounded_float("rebalance_intensity", rebalance_intensity, 0.0, 1.0)
        nav_t = _positive_float("portfolio_state.nav", portfolio_state.nav)
        constraint_violations: list[dict[str, Any]] = []
        info: dict[str, Any] = {"return_imputation": [], "constraint_violations": constraint_violations}
        tradeable = _bool_array_1d(
            "tradeable_mask_at_execution",
            execution_market_state.tradeable_mask_at_execution,
            decision.shape,
        )
        sellable = _execution_sellable_mask(portfolio_state, execution_market_state.execution_date, self.execution_config)

        valuation_freeze_mask = _valuation_freeze_mask(
            execution_market_state.return_from_decision_to_execution,
            decision,
            execution_market_state.valuation_price_at_execution,
            execution_market_state.availability_reason_at_execution,
            asset_ids=asset_ids,
            info=info,
        )
        r_pre = _repair_pre_execution_returns(
            execution_market_state.return_from_decision_to_execution,
            decision,
            execution_market_state.valuation_price_at_execution,
            portfolio_state,
            valuation_freeze_mask,
            execution_market_state.availability_reason_at_execution,
            asset_ids=asset_ids,
            info=info,
        )
        r_pre = sanitize_execution_returns(
            r_pre,
            decision,
            tradeable,
            execution_market_state.availability_reason_at_execution,
            asset_ids=asset_ids,
            info=info,
        )
        tradeable_for_rebalance = tradeable & ~valuation_freeze_mask
        if valuation_freeze_mask.any():
            constraint_violations.append(
                {
                    "constraint": "valuation",
                    "reason": "missing_decision_valuation_position_frozen",
                    "asset_indices": np.flatnonzero(valuation_freeze_mask).astype(int).tolist(),
                }
            )

        pre_execution_return = float(np.dot(decision, r_pre))
        nav_execution = nav_t * (1.0 + pre_execution_return)
        _validate_positive_nav("nav_execution", nav_execution)
        pre_execution_drifted_weights = drift_weights(
            decision,
            r_pre,
            cash_enabled=bool(self.execution_config.get("cash_enabled", False)),
        )

        if action == 0:
            executed_weights = pre_execution_drifted_weights.copy()
            turnover = 0.0
            proportional_cost = 0.0
            fixed_cost = 0.0
            slippage_cost = 0.0
            market_impact_cost = 0.0
            total_transaction_cost = 0.0
            cost_info: dict[str, Any] = {}
        else:
            target = _apply_tradeable_mask(
                target,
                pre_execution_drifted_weights,
                tradeable_for_rebalance,
                constraint_violations,
            )
            projected_target = _project_target_weights(
                self.constraint_manager,
                target,
                tradeable_for_rebalance,
                pre_execution_drifted_weights,
                constraint_violations,
            )
            interpolated = pre_execution_drifted_weights + intensity * (projected_target - pre_execution_drifted_weights)
            executed_weights = _post_check_partial_rebalance(
                interpolated,
                projected_target,
                pre_execution_drifted_weights,
                tradeable_for_rebalance,
                self.constraint_config,
                self.constraint_manager,
                constraint_violations,
            )
            executed_weights = _apply_t_plus_one_freeze(
                executed_weights,
                pre_execution_drifted_weights,
                sellable,
                self.execution_config,
                constraint_violations,
            )
            cost = self.cost_model.estimate(
                pre_execution_drifted_weights,
                executed_weights,
                execution_market_state,
                portfolio_state,
            )
            turnover = cost.turnover
            proportional_cost = cost.proportional_cost
            fixed_cost = cost.fixed_cost
            slippage_cost = cost.slippage_cost
            market_impact_cost = cost.market_impact_cost
            total_transaction_cost = cost.total_transaction_cost
            cost_info = dict(cost.info)

        nav_after_cost = nav_execution * (1.0 - total_transaction_cost)
        _validate_positive_nav("nav_after_cost", nav_after_cost)
        r_hold = sanitize_execution_returns(
            execution_market_state.holding_simple_return,
            executed_weights,
            tradeable,
            execution_market_state.availability_reason_at_execution,
            asset_ids=asset_ids,
            info=info,
        )
        post_execution_return = float(np.dot(executed_weights, r_hold))
        nav_next = nav_after_cost * (1.0 + post_execution_return)
        _validate_positive_nav("nav_next", nav_next)
        next_valuation_price = _next_valuation_price(
            execution_market_state.valuation_price_at_next,
            np.zeros_like(r_hold, dtype=float),
            portfolio_state.last_valuation_price,
        )

        gross_return = (1.0 + pre_execution_return) * (1.0 + post_execution_return) - 1.0
        transaction_cost_on_initial_nav = nav_execution * total_transaction_cost / nav_t
        net_return = nav_next / nav_t - 1.0
        portfolio_log_return = float(np.log(nav_next / nav_t))
        next_weights = drift_weights(
            executed_weights,
            r_hold,
            cash_enabled=bool(self.execution_config.get("cash_enabled", False)),
        )

        portfolio_state.nav = nav_next
        portfolio_state.portfolio_value = _positive_float("portfolio_state.portfolio_value", portfolio_state.portfolio_value) * (
            nav_next / nav_t
        )
        portfolio_state.current_weights = next_weights
        portfolio_state.drifted_weights = pre_execution_drifted_weights
        portfolio_state.previous_executed_weights = executed_weights.copy()
        portfolio_state.last_valuation_price = next_valuation_price
        portfolio_state.running_max_nav = max(
            _positive_float("portfolio_state.running_max_nav", portfolio_state.running_max_nav or nav_t),
            nav_next,
        )
        portfolio_state.prev_drawdown_abs = portfolio_state.current_drawdown_abs
        portfolio_state.current_drawdown_abs = max(0.0, 1.0 - nav_next / portfolio_state.running_max_nav)
        portfolio_state.max_drawdown_abs = max(portfolio_state.max_drawdown_abs, portfolio_state.current_drawdown_abs)
        portfolio_state.rolling_returns.append(net_return)
        portfolio_state.step_index += 1
        portfolio_state.date = execution_market_state.next_valuation_date
        _update_t_plus_one_state(
            portfolio_state,
            pre_execution_drifted_weights,
            executed_weights,
            sellable,
            execution_market_state.execution_date,
            self.execution_config,
        )
        info.update(cost_info)
        info["pre_execution_asset_simple_return"] = r_pre.tolist()
        info["post_execution_asset_simple_return"] = r_hold.tolist()

        return ExecutionResult(
            executed_weights=executed_weights,
            pre_execution_drifted_weights=pre_execution_drifted_weights,
            turnover=turnover,
            transaction_cost=total_transaction_cost,
            transaction_cost_on_initial_nav=transaction_cost_on_initial_nav,
            proportional_cost=proportional_cost,
            fixed_cost=fixed_cost,
            slippage_cost=slippage_cost,
            market_impact_cost=market_impact_cost,
            total_transaction_cost=total_transaction_cost,
            estimated_turnover=_optional_result_float("estimated_turnover", estimated_turnover),
            realized_turnover=turnover,
            estimated_cost=_optional_result_float("estimated_cost", estimated_cost),
            realized_cost=total_transaction_cost,
            gross_return=gross_return,
            net_return=net_return,
            pre_execution_return=pre_execution_return,
            post_execution_return=post_execution_return,
            portfolio_log_return=portfolio_log_return,
            nav_execution=nav_execution,
            nav_after_cost=nav_after_cost,
            nav_next=nav_next,
            info=info,
        )


def build_execution_market_state(
    dataset: MarketDatasetBundle,
    config: Mapping[str, Any],
    decision_date: Any | None = None,
    *,
    pending_action: PendingAction | None = None,
) -> ExecutionMarketState:
    return PortfolioExecutionCore(config).build_execution_market_state(
        dataset,
        decision_date,
        pending_action=pending_action,
    )


def _valuation_freeze_mask(
    raw_returns: np.ndarray,
    weights: np.ndarray,
    execution_price: np.ndarray | None = None,
    availability_reason: np.ndarray | None = None,
    *,
    asset_ids: np.ndarray | list[Any] | None = None,
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    returns = _array_1d("raw_returns", raw_returns, finite=False)
    weights_array = _array_1d("weights", weights, returns.shape)
    prices = _optional_array_1d("execution_price", execution_price, returns.shape, finite=False)
    reasons = _optional_object_array_1d("availability_reason", availability_reason, returns.shape)
    ids = _optional_object_array_1d("asset_ids", asset_ids, returns.shape)
    invalid_execution_price = np.zeros_like(returns, dtype=bool)
    if prices is not None:
        invalid_execution_price = (~np.isfinite(prices)) | (prices <= 0.0)
    missing_valuation = (~np.isfinite(returns)) | ((returns <= -1.0 + 1.0e-12) & invalid_execution_price)
    mask = missing_valuation & (np.abs(weights_array) > WEIGHT_SUM_EPS)
    if mask.any() and info is not None:
        info.setdefault("valuation_freeze", []).extend(
            {
                "asset_id": _asset_id(ids, int(index)),
                "reason": _reason_value(reasons, int(index)),
            }
            for index in np.flatnonzero(mask)
        )
    return mask


def _repair_pre_execution_returns(
    raw_returns: np.ndarray,
    weights: np.ndarray,
    execution_price: np.ndarray,
    portfolio_state: PortfolioState,
    valuation_freeze_mask: np.ndarray,
    availability_reason: np.ndarray | None = None,
    *,
    asset_ids: np.ndarray | list[Any] | None = None,
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    returns = _array_1d("raw_returns", raw_returns, finite=False).astype(float, copy=True)
    _array_1d("weights", weights, returns.shape)
    prices = _array_1d("execution_price", execution_price, returns.shape, finite=False)
    freeze = _bool_array_1d("valuation_freeze_mask", valuation_freeze_mask, returns.shape)
    previous_prices = _optional_array_1d(
        "portfolio_state.last_valuation_price",
        portfolio_state.last_valuation_price,
        returns.shape,
        finite=False,
    )
    reasons = _optional_object_array_1d("availability_reason", availability_reason, returns.shape)
    ids = _optional_object_array_1d("asset_ids", asset_ids, returns.shape)
    if not freeze.any():
        return returns

    for index in np.flatnonzero(freeze):
        repaired = np.nan
        if previous_prices is not None and np.isfinite(previous_prices[index]) and previous_prices[index] > 0.0:
            repaired = prices[index] / previous_prices[index] - 1.0
        if not np.isfinite(repaired) or repaired <= -1.0 + 1.0e-12:
            repaired = 0.0
        returns[index] = float(repaired)
        if info is not None:
            info.setdefault("return_imputation", []).append(
                {
                    "asset_id": _asset_id(ids, int(index)),
                    "reason": "missing_decision_valuation",
                    "availability_reason": _reason_value(reasons, int(index)),
                    "value": float(repaired),
                }
            )
    return returns


def sanitize_execution_returns(
    raw_returns: np.ndarray,
    weights: np.ndarray,
    tradeable_mask: np.ndarray,
    availability_reason: np.ndarray | None = None,
    *,
    asset_ids: np.ndarray | list[Any] | None = None,
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    returns = _array_1d("raw_returns", raw_returns, finite=False)
    weights_array = _array_1d("weights", weights, returns.shape)
    tradeable = _bool_array_1d("tradeable_mask", tradeable_mask, returns.shape)
    reasons = _optional_object_array_1d("availability_reason", availability_reason, returns.shape)
    ids = _optional_object_array_1d("asset_ids", asset_ids, returns.shape)
    sanitized = returns.astype(float, copy=True)
    imputations: list[dict[str, Any]] = []

    for index in np.flatnonzero(~np.isfinite(sanitized)):
        if abs(weights_array[index]) <= WEIGHT_SUM_EPS:
            sanitized[index] = 0.0
            continue
        reason = _reason_value(reasons, int(index))
        if not bool(tradeable[index]) and reason in RETURN_IMPUTATION_REASONS:
            sanitized[index] = 0.0
            imputations.append(
                {
                    "asset_id": _asset_id(ids, int(index)),
                    "reason": reason,
                    "value": 0.0,
                }
            )
            continue
        raise DataContractError(
            "ERR_EXECUTION_RETURN_MISSING",
            f"ERR_EXECUTION_RETURN_MISSING: index={int(index)}, reason={reason}",
        )

    if imputations and info is not None:
        info.setdefault("return_imputation", []).extend(imputations)
    return sanitized


def _next_valuation_price(
    execution_price: np.ndarray,
    holding_simple_return: np.ndarray,
    previous_valuation_price: np.ndarray | None,
) -> np.ndarray | None:
    prices = _array_1d("execution_price", execution_price, finite=False)
    returns = _array_1d("holding_simple_return", holding_simple_return, prices.shape, finite=False)
    previous = _optional_array_1d(
        "previous_valuation_price",
        previous_valuation_price,
        prices.shape,
        finite=False,
    )
    next_prices = prices * (1.0 + returns)
    if previous is not None:
        next_prices = np.where(np.isfinite(next_prices) & (next_prices > 0.0), next_prices, previous)
    return next_prices


def drift_weights(
    weights: np.ndarray,
    simple_returns: np.ndarray,
    *,
    cash_enabled: bool = False,
    cash_index: int = -1,
    weight_sum_eps: float = WEIGHT_SUM_EPS,
) -> np.ndarray:
    weights_array = _array_1d("weights", weights)
    returns = _array_1d("simple_returns", simple_returns, weights_array.shape).astype(float, copy=True)
    if cash_enabled:
        returns[_cash_index(cash_index, returns.shape[0])] = 0.0
    gross = weights_array * np.maximum(1.0 + returns, 0.0)
    denom = float(np.sum(gross))
    if not np.isfinite(denom) or denom <= float(weight_sum_eps):
        if float(np.sum(np.abs(weights_array))) <= float(weight_sum_eps):
            return np.zeros_like(weights_array, dtype=float)
        raise DataContractError("ERR_EXECUTION_INVALID_NAV", "ERR_EXECUTION_INVALID_NAV: drift_weights")
    return gross / denom


def _execution_model_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_CONFIG["execution_model"]
    if config is None or "execution_model" not in config:
        return dict(source)
    return {**source, **dict(config["execution_model"])}


def _data_governance_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_CONFIG["data_governance"]
    if config is None or "data_governance" not in config:
        return dict(source)
    return {**source, **dict(config["data_governance"])}


def _constraint_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_CONFIG["constraints"]
    if config is None or "constraints" not in config:
        return dict(source)
    return {**source, **dict(config["constraints"])}


def _pending_or_config_execution_price(
    pending_action: PendingAction | None,
    execution_config: Mapping[str, Any],
) -> str:
    if pending_action is not None:
        return str(pending_action.execution_price)
    return str(execution_config.get("execution_price", "next_open"))


def _same_close_enabled(
    execution_config: Mapping[str, Any],
    data_governance_config: Mapping[str, Any],
) -> bool:
    return bool(
        execution_config.get("same_close_idealized_execution_enabled", False)
        or data_governance_config.get("same_close_idealized_execution_enabled", False)
    )


def _decision_date(decision_date: Any | None, pending_action: PendingAction | None) -> pd.Timestamp:
    value = pending_action.decision_date if pending_action is not None else decision_date
    if value is None:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: decision_date")
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: decision_date") from exc
    if pd.isna(result):
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: decision_date")
    return result


def _date_index(dataset: MarketDatasetBundle) -> pd.DatetimeIndex:
    close = _wide_table(dataset, "close")
    index = pd.DatetimeIndex(close.index)
    if index.empty or not index.is_monotonic_increasing:
        raise DataContractError("ERR_DATA_SCHEMA_MISMATCH", "ERR_DATA_SCHEMA_MISMATCH: wide_close date index")
    return index


def _date_position(date_index: pd.DatetimeIndex, date: pd.Timestamp) -> int:
    matches = np.flatnonzero(date_index == date)
    if matches.size == 0:
        raise DataContractError("ERR_EXECUTION_DATE_OUT_OF_RANGE", "ERR_EXECUTION_DATE_OUT_OF_RANGE: decision_date")
    return int(matches[0])


def _date_at(date_index: pd.DatetimeIndex, position: int) -> pd.Timestamp:
    if position < 0 or position >= len(date_index):
        raise DataContractError("ERR_EXECUTION_DATE_OUT_OF_RANGE", "ERR_EXECUTION_DATE_OUT_OF_RANGE: execution_date")
    return pd.Timestamp(date_index[position])


def _wide_table(dataset: MarketDatasetBundle, field: str) -> pd.DataFrame:
    if field not in dataset.wide:
        raise DataContractError("ERR_DATA_MISSING_FILE", f"ERR_DATA_MISSING_FILE: wide_{field}")
    return dataset.wide[field]


def _valuation_table(dataset: MarketDatasetBundle, data_governance_config: Mapping[str, Any]) -> pd.DataFrame:
    source = str(
        data_governance_config.get("valuation_source")
        or data_governance_config.get("return_source")
        or ""
    ).lower()
    split_required = bool(data_governance_config.get("valuation_execution_split", False) or source == "adj_nav")
    if split_required:
        if "adj_nav" not in dataset.wide:
            raise DataContractError("ERR_DATA_MISSING_FILE", "ERR_DATA_MISSING_FILE: wide_adj_nav")
        return dataset.wide["adj_nav"]
    return _wide_table(dataset, "close")


def _row(table: pd.DataFrame, date: pd.Timestamp, field: str) -> np.ndarray:
    if date not in table.index:
        raise DataContractError("ERR_EXECUTION_DATE_OUT_OF_RANGE", f"ERR_EXECUTION_DATE_OUT_OF_RANGE: wide_{field}")
    return table.loc[date].to_numpy(dtype=float, copy=True)


def _optional_row(dataset: MarketDatasetBundle, field: str, date: pd.Timestamp) -> np.ndarray | None:
    if field not in dataset.wide:
        return None
    return _row(dataset.wide[field], date, field)


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
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}")
    if shape is not None and array.shape != shape:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}")
    if finite and not np.isfinite(array).all():
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}")
    return array


def _bool_array_1d(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.ndim != 1 or array.shape != shape:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}")
    return array


def _optional_array_1d(
    name: str,
    values: Any,
    shape: tuple[int, ...],
    *,
    finite: bool = True,
) -> np.ndarray | None:
    if values is None:
        return None
    return _array_1d(name, values, shape, finite=finite)


def _optional_result_float(name: str, value: Any | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise DataContractError("ERR_ACTION_NON_FINITE", f"ERR_ACTION_NON_FINITE: {name}")
    return result


def _optional_object_array_1d(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or array.shape != shape:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}")
    return array


def _rebalance_action(value: Any) -> int:
    try:
        action = int(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: rebalance_action") from exc
    if action not in {0, 1}:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: rebalance_action")
    return action


def _bounded_float(name: str, value: Any, lower: float, upper: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}") from exc
    if not np.isfinite(result) or result < lower or result > upper:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}")
    return result


def _positive_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", f"ERR_STATE_SCHEMA_MISMATCH: {name}")
    return result


def _validate_positive_nav(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise DataContractError("ERR_EXECUTION_INVALID_NAV", f"ERR_EXECUTION_INVALID_NAV: {name}")


def _apply_tradeable_mask(
    target: np.ndarray,
    reference: np.ndarray,
    tradeable: np.ndarray,
    violations: list[dict[str, Any]],
) -> np.ndarray:
    adjusted = target.astype(float, copy=True)
    blocked = (~tradeable) & (adjusted > reference + WEIGHT_SUM_EPS)
    frozen = (~tradeable) & (reference > WEIGHT_SUM_EPS)
    if blocked.any():
        violations.append(
            {
                "constraint": "availability",
                "reason": "untradeable_buy_blocked",
                "asset_indices": np.flatnonzero(blocked).astype(int).tolist(),
            }
        )
    if frozen.any():
        violations.append(
            {
                "constraint": "availability",
                "reason": "untradeable_position_frozen",
                "asset_indices": np.flatnonzero(frozen).astype(int).tolist(),
            }
        )
    adjusted[~tradeable] = reference[~tradeable]
    return _normalize_with_fixed(adjusted, tradeable, adjusted[~tradeable].sum())


def _project_target_weights(
    constraint_manager: ConstraintManager,
    target: np.ndarray,
    tradeable: np.ndarray,
    reference: np.ndarray,
    violations: list[dict[str, Any]],
) -> np.ndarray:
    if np.any((~tradeable) & (reference > WEIGHT_SUM_EPS)):
        return target
    result = constraint_manager.project(target, tradeable, reference_weights=reference)
    violations.extend(result.constraint_violations)
    return result.projected_weights


def _post_check_partial_rebalance(
    interpolated: np.ndarray,
    projected_target: np.ndarray,
    reference: np.ndarray,
    tradeable: np.ndarray,
    constraint_config: Mapping[str, Any],
    constraint_manager: ConstraintManager,
    violations: list[dict[str, Any]],
) -> np.ndarray:
    adjusted = _apply_tradeable_mask(interpolated, reference, tradeable, violations)
    policy = str(constraint_config.get("partial_rebalance_post_check_policy", "report_only"))
    if policy not in {"report_only", "project_executed", "force_full_rebalance"}:
        raise DataContractError(
            "ERR_CONFIG_INVALID_CONSTRAINT",
            "ERR_CONFIG_INVALID_CONSTRAINT: constraints.partial_rebalance_post_check_policy",
        )
    post_violations = _partial_post_check_violations(adjusted, reference, tradeable, constraint_config)
    if not post_violations:
        return adjusted
    for record in post_violations:
        record["policy"] = policy
    violations.extend(post_violations)
    if policy == "report_only":
        return adjusted
    if policy == "force_full_rebalance":
        return projected_target
    if np.any((~tradeable) & (reference > WEIGHT_SUM_EPS)):
        return adjusted
    result = constraint_manager.project(adjusted, tradeable, reference_weights=reference)
    violations.extend(result.constraint_violations)
    return result.projected_weights


def _partial_post_check_violations(
    weights: np.ndarray,
    reference: np.ndarray,
    tradeable: np.ndarray,
    constraint_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    unavailable_weight = np.abs(weights[~tradeable] - reference[~tradeable]) > WEIGHT_SUM_EPS
    if unavailable_weight.any():
        records.append(
            {
                "constraint": "availability",
                "reason": "partial_rebalance_untradeable_weight_changed",
                "asset_indices": np.flatnonzero(~tradeable)[unavailable_weight].astype(int).tolist(),
            }
        )
    if np.any(weights[tradeable] < -WEIGHT_SUM_EPS):
        records.append({"constraint": "long_only", "reason": "partial_rebalance_negative_weight"})
    if abs(float(np.sum(weights)) - 1.0) > WEIGHT_SUM_EPS:
        records.append({"constraint": "simplex", "reason": "partial_rebalance_sum_not_one"})
    max_weight = constraint_config.get("max_weight")
    if max_weight is not None and tradeable.any():
        limit = float(max_weight)
        if np.any(weights[tradeable] > limit + WEIGHT_SUM_EPS):
            records.append(
                {
                    "constraint": "max_weight",
                    "reason": "partial_rebalance_post_check_violation",
                    "limit": limit,
                    "max_weight_after": float(np.max(weights[tradeable])),
                }
            )
    turnover_limit = constraint_config.get("turnover_limit")
    if turnover_limit is not None:
        turnover = 0.5 * float(np.sum(np.abs(weights - reference)))
        limit = float(turnover_limit)
        if turnover > limit + WEIGHT_SUM_EPS:
            records.append(
                {
                    "constraint": "turnover",
                    "reason": "partial_rebalance_post_check_violation",
                    "limit": limit,
                    "turnover_after": turnover,
                }
            )
    hhi_limit = constraint_config.get("hhi_limit")
    if hhi_limit is not None:
        hhi = float(np.sum(np.square(weights)))
        limit = float(hhi_limit)
        if hhi > limit + WEIGHT_SUM_EPS:
            records.append(
                {
                    "constraint": "hhi",
                    "reason": "partial_rebalance_post_check_violation",
                    "limit": limit,
                    "hhi_after": hhi,
                }
            )
    return records


def _execution_sellable_mask(
    portfolio_state: PortfolioState,
    execution_date: pd.Timestamp,
    execution_config: Mapping[str, Any],
) -> np.ndarray:
    if not bool(execution_config.get("t_plus_one", False)):
        return np.ones_like(portfolio_state.current_weights, dtype=bool)
    last_buy = _optional_object_array_1d(
        "last_buy_date_per_asset",
        portfolio_state.last_buy_date_per_asset,
        portfolio_state.current_weights.shape,
    )
    if last_buy is None:
        return np.ones_like(portfolio_state.current_weights, dtype=bool)
    sellable = np.ones_like(portfolio_state.current_weights, dtype=bool)
    for index, value in enumerate(last_buy):
        if value is None or pd.isna(value):
            continue
        sellable[index] = pd.Timestamp(execution_date) > pd.Timestamp(value)
    return sellable


def _apply_t_plus_one_freeze(
    weights: np.ndarray,
    reference: np.ndarray,
    sellable: np.ndarray,
    execution_config: Mapping[str, Any],
    violations: list[dict[str, Any]],
) -> np.ndarray:
    if not bool(execution_config.get("t_plus_one", False)):
        return weights
    blocked_sell = (~sellable) & (weights < reference - WEIGHT_SUM_EPS)
    if not blocked_sell.any():
        return weights
    adjusted = weights.astype(float, copy=True)
    adjusted[blocked_sell] = reference[blocked_sell]
    violations.append(
        {
            "constraint": "t_plus_one",
            "reason": "sell_blocked_frozen_weight",
            "asset_indices": np.flatnonzero(blocked_sell).astype(int).tolist(),
        }
    )
    return _normalize_with_fixed(adjusted, ~blocked_sell, float(np.sum(adjusted[blocked_sell])))


def _update_t_plus_one_state(
    portfolio_state: PortfolioState,
    reference: np.ndarray,
    executed: np.ndarray,
    sellable: np.ndarray,
    execution_date: pd.Timestamp,
    execution_config: Mapping[str, Any],
) -> None:
    if not bool(execution_config.get("t_plus_one", False)):
        return
    last_buy = _optional_object_array_1d(
        "last_buy_date_per_asset",
        portfolio_state.last_buy_date_per_asset,
        executed.shape,
    )
    if last_buy is None:
        last_buy = np.array([None] * executed.shape[0], dtype=object)
    else:
        last_buy = last_buy.copy()
    bought = executed > reference + WEIGHT_SUM_EPS
    last_buy[bought] = pd.Timestamp(execution_date)
    portfolio_state.last_buy_date_per_asset = last_buy
    next_sellable = _execution_sellable_mask(portfolio_state, execution_date, execution_config)
    portfolio_state.sellable_mask = next_sellable
    frozen = np.zeros_like(executed, dtype=float)
    frozen[~next_sellable] = executed[~next_sellable]
    portfolio_state.frozen_weight = frozen


def _normalize_with_fixed(weights: np.ndarray, adjustable_mask: np.ndarray, fixed_sum: float) -> np.ndarray:
    adjusted = weights.astype(float, copy=True)
    residual = 1.0 - float(fixed_sum)
    if residual < -WEIGHT_SUM_EPS:
        raise DataContractError("ERR_EXECUTION_INVALID_NAV", "ERR_EXECUTION_INVALID_NAV: frozen_weight")
    if not adjustable_mask.any():
        if abs(residual) > WEIGHT_SUM_EPS:
            raise DataContractError("ERR_EXECUTION_NO_AVAILABLE_ASSET", "ERR_EXECUTION_NO_AVAILABLE_ASSET")
        return adjusted
    if residual <= WEIGHT_SUM_EPS:
        adjusted[adjustable_mask] = 0.0
        return adjusted
    values = np.maximum(adjusted[adjustable_mask], 0.0)
    total = float(np.sum(values))
    if total <= WEIGHT_SUM_EPS:
        adjusted[adjustable_mask] = residual / int(adjustable_mask.sum())
    else:
        adjusted[adjustable_mask] = values / total * residual
    return adjusted


def _safe_return(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator - 1.0


def _tradeable_mask(dataset: MarketDatasetBundle, execution_date: pd.Timestamp) -> np.ndarray:
    if execution_date not in dataset.availability_mask.index:
        raise DataContractError(
            "ERR_EXECUTION_DATE_OUT_OF_RANGE",
            "ERR_EXECUTION_DATE_OUT_OF_RANGE: availability_mask",
        )
    return dataset.availability_mask.loc[execution_date].to_numpy(dtype=bool, copy=True)


def _availability_reason(dataset: MarketDatasetBundle, execution_date: pd.Timestamp) -> np.ndarray | None:
    if dataset.availability_reason is None:
        return None
    if execution_date not in dataset.availability_reason.index:
        raise DataContractError(
            "ERR_EXECUTION_DATE_OUT_OF_RANGE",
            "ERR_EXECUTION_DATE_OUT_OF_RANGE: availability_reason",
        )
    return dataset.availability_reason.loc[execution_date].to_numpy(dtype=object, copy=True)


def _rolling_mean_row(table: pd.DataFrame, date: pd.Timestamp, field: str) -> np.ndarray:
    rolling = table.rolling(ROLLING_WINDOW, min_periods=1).mean()
    return _row(rolling, date, field)


def _rolling_volatility_row(dataset: MarketDatasetBundle, execution_date: pd.Timestamp) -> np.ndarray:
    log_return = _wide_table(dataset, "log_return")
    rolling = log_return.rolling(ROLLING_WINDOW, min_periods=1).std(ddof=0)
    return _row(rolling.fillna(0.0), execution_date, "log_return")


def _reason_value(reasons: np.ndarray | None, index: int) -> str:
    if reasons is None:
        return "unknown"
    reason = reasons[index]
    if reason is None or pd.isna(reason):
        return "unknown"
    return str(reason)


def _asset_id(asset_ids: np.ndarray | None, index: int) -> Any:
    if asset_ids is None:
        return int(index)
    return asset_ids[index]


def _cash_index(cash_index: int, size: int) -> int:
    index = int(cash_index)
    if index < 0:
        index += size
    if index < 0 or index >= size:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: cash_index")
    return index


__all__ = [
    "PortfolioExecutionCore",
    "build_execution_market_state",
    "drift_weights",
    "sanitize_execution_returns",
]
