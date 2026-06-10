from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.data.leakage_checks import assert_decision_visibility_contract
from src.data.loader import DataContractError, MarketDatasetBundle
from src.envs.portfolio_execution_core import PortfolioExecutionCore
from src.envs.rebalance_scheduler import RebalanceScheduler
from src.envs.reward_calculator import RewardCalculator, resolve_training_warmup_v_init
from src.envs.risk_state_manager import RiskStateManager
from src.envs.state import DecisionMarketState, PendingAction, PortfolioAction, PortfolioState


DAILY_RETURNS_COLUMNS = [
    "date",
    "decision_date",
    "execution_date",
    "execution_price_type",
    "next_valuation_date",
    "split",
    "seed",
    "fold_id",
    "model_name",
    "pre_execution_return",
    "post_execution_return",
    "gross_return",
    "transaction_cost",
    "transaction_cost_on_initial_nav",
    "net_return",
    "portfolio_log_return",
    "nav",
    "reward",
]
DAILY_WEIGHTS_COLUMNS = [
    "date",
    "split",
    "seed",
    "fold_id",
    "model_name",
    "asset_id",
    "weight",
]
DAILY_TURNOVER_COLUMNS = [
    "date",
    "decision_date",
    "execution_date",
    "execution_price_type",
    "next_valuation_date",
    "split",
    "seed",
    "fold_id",
    "model_name",
    "turnover",
    "rebalance_action",
    "rebalance_intensity",
    "average_holding_period",
]
DAILY_REBALANCE_COLUMNS = [
    "date",
    "decision_date",
    "execution_date",
    "execution_price_type",
    "next_valuation_date",
    "split",
    "seed",
    "fold_id",
    "model_name",
    "rebalance_action",
    "rebalance_intensity",
    "estimated_turnover",
    "realized_turnover",
    "turnover",
    "estimated_cost",
    "realized_cost",
    "q_hold",
    "q_rebalance",
    "q_gap",
    "fallback_reason",
]
DAILY_COSTS_COLUMNS = [
    "date",
    "decision_date",
    "execution_date",
    "execution_price_type",
    "next_valuation_date",
    "split",
    "seed",
    "fold_id",
    "model_name",
    "proportional_cost",
    "fixed_cost",
    "slippage_cost",
    "market_impact_cost",
    "total_transaction_cost",
    "estimated_cost",
    "realized_cost",
    "turnover",
]
CONTRACT_ERROR_CODES = {
    "ERR_ACTION_NON_FINITE",
    "ERR_ACTION_SHAPE_MISMATCH",
    "ERR_STATE_SCHEMA_MISMATCH",
    "ERR_STRATEGY_ACTION_CONTRACT",
    "ERR_STRATEGY_STATE_CONTRACT",
}
PENDING_TRUNCATION_REASON = "ERR_EXECUTION_DATE_OUT_OF_RANGE"
BASELINE_DAILY_DIAGNOSTICS_ALIGNMENT_COLUMNS = (
    "date",
    "decision_date",
    "execution_date",
    "model_name",
    "paper_model_id",
    "seed",
    "fold_id",
)


@dataclass(frozen=True)
class BacktestResult:
    daily_returns: pd.DataFrame
    daily_weights: pd.DataFrame
    daily_turnover: pd.DataFrame
    daily_rebalance: pd.DataFrame
    daily_costs: pd.DataFrame
    metrics: dict[str, Any]
    run_manifest: dict[str, Any]
    portfolio_state: PortfolioState
    baseline_daily_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_asset_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    artifact_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class PendingActionQueue:
    pending_actions: list[PendingAction] = field(default_factory=list)

    def append(self, pending_action: PendingAction) -> None:
        self.pending_actions.append(pending_action)
        self.pending_actions.sort(key=lambda item: item.execution_date)

    def pop_ready(self, current_date: Any) -> list[PendingAction]:
        current = pd.Timestamp(current_date)
        ready = [item for item in self.pending_actions if item.execution_date == current]
        self.pending_actions = [item for item in self.pending_actions if item.execution_date != current]
        return ready

    def clear(self) -> None:
        self.pending_actions.clear()

    def __len__(self) -> int:
        return len(self.pending_actions)


class BacktestEngine:
    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        execution_core: PortfolioExecutionCore | None = None,
        scheduler: RebalanceScheduler | None = None,
        reward_calculator: RewardCalculator | None = None,
        market_image_dataset: Any | None = None,
    ) -> None:
        self.config = deepcopy(DEFAULT_CONFIG)
        if config is not None:
            _deep_update(self.config, config)
        self.execution_config = self.config["execution_model"]
        self.execution_activity_config = _execution_activity_config(self.config)
        self.portfolio_config = self.config["portfolio"]
        self.execution_core = execution_core or PortfolioExecutionCore(self.config)
        self.scheduler = scheduler
        self.reward_calculator = reward_calculator or RewardCalculator(self.config)
        self.market_image_dataset = market_image_dataset
        self._pending_truncation_count = 0

    def run(
        self,
        dataset: MarketDatasetBundle,
        split: Any,
        strategy: Any,
        *,
        segment: str = "test",
        output_dir: str | Path | None = None,
    ) -> BacktestResult:
        if not hasattr(strategy, "compute_target_weights"):
            raise DataContractError(
                "ERR_STRATEGY_STATE_CONTRACT",
                "ERR_STRATEGY_STATE_CONTRACT: strategy.compute_target_weights",
            )
        if bool(self.execution_config.get("delayed_action_execution", False)):
            return self._run_delayed(dataset, split, strategy, segment=segment, output_dir=output_dir)

        date_index = _date_index(dataset)
        asset_ids = _asset_order(dataset)
        decision_dates = _decision_dates(split, segment, date_index)
        scheduler = self.scheduler or RebalanceScheduler(self.config, date_index=date_index)
        if getattr(strategy, "fit_required", False):
            strategy.fit(
                _fit_payload(dataset, split, "train", market_image_dataset=self.market_image_dataset, config=self.config),
                _fit_payload(dataset, split, "validation", market_image_dataset=self.market_image_dataset, config=self.config),
            )
            _assert_strategy_training_completed(strategy)
        if hasattr(strategy, "reset"):
            strategy.reset()
        scheduler.reset()
        v_init_resolved_value = resolve_training_warmup_v_init(
            self.reward_calculator,
            dataset,
            split,
            self.config,
        )
        self.reward_calculator.reset_episode()
        record_metadata = _record_metadata(segment, split, strategy, self.config)

        portfolio_state = self._initial_portfolio_state(asset_ids, decision_dates[0])
        risk_state_manager = RiskStateManager(self.config) if _backtest_risk_state_required(self.config) else None
        if risk_state_manager is not None:
            risk_state_manager.reset()
            portfolio_state.risk_state_vector = risk_state_manager.get_observation_vector()
        daily_returns: list[dict[str, Any]] = []
        daily_weights: list[dict[str, Any]] = []
        daily_turnover: list[dict[str, Any]] = []
        daily_rebalance: list[dict[str, Any]] = []
        daily_costs: list[dict[str, Any]] = []
        daily_diagnostics: list[dict[str, Any]] = []
        daily_asset_returns: list[dict[str, Any]] = []
        first_trade = True
        boundary_timestep = 0
        run_id = record_metadata.get("run_id", record_metadata.get("model_name", "unknown"))
        episode_id = f"{run_id}:{segment}"

        for decision_date in decision_dates:
            decision_state = _build_decision_market_state(
                dataset,
                decision_date,
                self.config,
                market_image_dataset=self.market_image_dataset,
            )
            scheduler_pre_allowed = scheduler.pre_check(decision_date, portfolio_state, decision_state)
            action = _action_for_step(
                strategy,
                decision_state,
                portfolio_state,
                scheduler_pre_allowed,
                first_trade,
                scheduler_blocks_model_actions=bool(self.execution_activity_config["scheduler_blocks_model_actions"]),
            )
            action = _finalize_execution_action(
                scheduler,
                decision_date,
                portfolio_state,
                decision_state,
                action,
                first_trade,
                self.execution_activity_config,
            )
            final_action = action.rebalance_action

            pre_step_previous_executed_weights = portfolio_state.previous_executed_weights.copy() if portfolio_state.previous_executed_weights is not None else None
            execution_state = self.execution_core.build_execution_market_state(dataset, decision_date)
            execution_result = self._execute_step(
                portfolio_state,
                action,
                execution_state,
                final_action,
                first_trade,
                asset_ids,
            )
            reward_context: dict[str, Any] = {}
            if risk_state_manager is not None:
                risk_state_manager.update_pre_reward(execution_result, portfolio_state, final_action)
                reward_context["drawdown_increment_t"] = risk_state_manager.drawdown_increment
            reward, reward_info = self.reward_calculator.calculate(
                execution_result,
                portfolio_state,
                reward_context=reward_context,
            )
            if risk_state_manager is not None:
                risk_state_manager.update_reward_info(reward_info)
                portfolio_state.risk_state_vector = risk_state_manager.get_observation_vector()
            boundary = {
                "run_id": run_id,
                "episode_id": episode_id,
                "timestep": boundary_timestep,
                "done_t": False,
                "split": segment,
                "termination_reason": "",
            }
            record_context = {
                "action": action,
                "execution_result": execution_result,
                "execution_state": execution_state,
                "final_action": final_action,
                "reward": reward,
                "reward_info": reward_info,
                "metadata": record_metadata,
                "boundary": boundary,
                "pre_step_previous_executed_weights": pre_step_previous_executed_weights,
            }
            daily_returns.append(_daily_returns_record(record_context))
            daily_turnover.append(_daily_turnover_record(record_context, final_action))
            daily_rebalance.append(_daily_rebalance_record(record_context, final_action))
            daily_costs.append(_daily_costs_record(record_context))
            if _should_record_daily_diagnostics(action, self.execution_activity_config):
                daily_diagnostics.append(_baseline_diagnostics_record(record_context))
            daily_asset_returns.extend(
                _daily_asset_returns_records(record_context, asset_ids, boundary, record_metadata)
            )
            daily_weights.extend(
                _daily_weights_records(
                    execution_state.next_valuation_date,
                    asset_ids,
                    execution_result.executed_weights,
                    record_metadata,
                )
            )
            first_trade = False
            boundary_timestep += 1

        if daily_diagnostics:
            daily_diagnostics[-1]["done_t"] = True
            daily_diagnostics[-1]["termination_reason"] = "segment_end"

        result = BacktestResult(
            daily_returns=pd.DataFrame(daily_returns, columns=DAILY_RETURNS_COLUMNS),
            daily_weights=pd.DataFrame(daily_weights, columns=DAILY_WEIGHTS_COLUMNS),
            daily_turnover=pd.DataFrame(daily_turnover, columns=DAILY_TURNOVER_COLUMNS),
            daily_rebalance=pd.DataFrame(daily_rebalance, columns=DAILY_REBALANCE_COLUMNS),
            daily_costs=pd.DataFrame(daily_costs, columns=DAILY_COSTS_COLUMNS),
            baseline_daily_diagnostics=pd.DataFrame(daily_diagnostics),
            daily_asset_returns=pd.DataFrame(daily_asset_returns),
            metrics=_metrics(daily_returns, daily_turnover, daily_costs, daily_diagnostics),
            run_manifest=self._run_manifest(dataset),
            portfolio_state=portfolio_state,
        )
        result.run_manifest["v_init_resolved_value"] = float(v_init_resolved_value)
        if output_dir is not None:
            result = _write_outputs(result, output_dir)
        return result

    def _run_delayed(
        self,
        dataset: MarketDatasetBundle,
        split: Any,
        strategy: Any,
        *,
        segment: str,
        output_dir: str | Path | None,
    ) -> BacktestResult:
        date_index = _date_index(dataset)
        asset_ids = _asset_order(dataset)
        segment_dates = _segment_dates(split, segment)
        segment_dates = pd.DatetimeIndex(segment_dates[segment_dates.isin(date_index)])
        if segment_dates.empty:
            raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: {segment}")
        decision_date_set = set(_decision_dates(split, segment, date_index))
        scheduler = self.scheduler or RebalanceScheduler(self.config, date_index=date_index)
        if getattr(strategy, "fit_required", False):
            strategy.fit(
                _fit_payload(dataset, split, "train", market_image_dataset=self.market_image_dataset, config=self.config),
                _fit_payload(dataset, split, "validation", market_image_dataset=self.market_image_dataset, config=self.config),
            )
            _assert_strategy_training_completed(strategy)
        if hasattr(strategy, "reset"):
            strategy.reset()
        scheduler.reset()
        v_init_resolved_value = resolve_training_warmup_v_init(
            self.reward_calculator,
            dataset,
            split,
            self.config,
        )
        self.reward_calculator.reset_episode()
        self._pending_truncation_count = 0
        record_metadata = _record_metadata(segment, split, strategy, self.config)

        portfolio_state = self._initial_portfolio_state(asset_ids, segment_dates[0])
        risk_state_manager = RiskStateManager(self.config) if _backtest_risk_state_required(self.config) else None
        if risk_state_manager is not None:
            risk_state_manager.reset()
            portfolio_state.risk_state_vector = risk_state_manager.get_observation_vector()
        pending_queue = PendingActionQueue()
        daily_returns: list[dict[str, Any]] = []
        daily_weights: list[dict[str, Any]] = []
        daily_turnover: list[dict[str, Any]] = []
        daily_rebalance: list[dict[str, Any]] = []
        daily_costs: list[dict[str, Any]] = []
        daily_diagnostics: list[dict[str, Any]] = []
        daily_asset_returns: list[dict[str, Any]] = []
        first_trade = True
        boundary_timestep = 0
        run_id = record_metadata.get("run_id", record_metadata.get("model_name", "unknown"))
        episode_id = f"{run_id}:{segment}"

        for current_date in segment_dates:
            for pending_action in pending_queue.pop_ready(current_date):
                execution_state = self.execution_core.build_execution_market_state(
                    dataset,
                    pending_action=pending_action,
                )
                action = _action_from_pending(pending_action)
                pre_step_previous_executed_weights = portfolio_state.previous_executed_weights.copy() if portfolio_state.previous_executed_weights is not None else None
                execution_result = self._execute_step(
                    portfolio_state,
                    action,
                    execution_state,
                    pending_action.rebalance_action,
                    first_trade,
                    asset_ids,
                )
                reward_context: dict[str, Any] = {}
                if risk_state_manager is not None:
                    risk_state_manager.update_pre_reward(execution_result, portfolio_state, pending_action.rebalance_action)
                    reward_context["drawdown_increment_t"] = risk_state_manager.drawdown_increment
                reward, reward_info = self.reward_calculator.calculate(
                    execution_result,
                    portfolio_state,
                    reward_context=reward_context,
                )
                if risk_state_manager is not None:
                    risk_state_manager.update_reward_info(reward_info)
                    portfolio_state.risk_state_vector = risk_state_manager.get_observation_vector()
                boundary = {
                    "run_id": run_id,
                    "episode_id": episode_id,
                    "timestep": boundary_timestep,
                    "done_t": False,
                    "split": segment,
                    "termination_reason": "",
                }
                record_context = {
                    "action": action,
                    "execution_result": execution_result,
                    "execution_state": execution_state,
                    "final_action": pending_action.rebalance_action,
                    "reward": reward,
                    "reward_info": reward_info,
                    "metadata": record_metadata,
                    "boundary": boundary,
                    "pre_step_previous_executed_weights": pre_step_previous_executed_weights,
                }
                daily_returns.append(_daily_returns_record(record_context))
                daily_turnover.append(_daily_turnover_record(record_context, pending_action.rebalance_action))
                daily_rebalance.append(_daily_rebalance_record(record_context, pending_action.rebalance_action))
                daily_costs.append(_daily_costs_record(record_context))
                if _should_record_daily_diagnostics(action, self.execution_activity_config):
                    daily_diagnostics.append(_baseline_diagnostics_record(record_context))
                daily_asset_returns.extend(
                    _daily_asset_returns_records(record_context, asset_ids, boundary, record_metadata)
                )
                daily_weights.extend(
                    _daily_weights_records(
                        execution_state.next_valuation_date,
                        asset_ids,
                        execution_result.executed_weights,
                        record_metadata,
                    )
                )
                first_trade = False
                boundary_timestep += 1

            if current_date < portfolio_state.date or current_date not in decision_date_set or len(pending_queue) > 0:
                continue

            decision_state = _build_decision_market_state(
                dataset,
                current_date,
                self.config,
                market_image_dataset=self.market_image_dataset,
            )
            scheduler_pre_allowed = scheduler.pre_check(current_date, portfolio_state, decision_state)
            action = _action_for_step(
                strategy,
                decision_state,
                portfolio_state,
                scheduler_pre_allowed,
                first_trade,
                scheduler_blocks_model_actions=bool(self.execution_activity_config["scheduler_blocks_model_actions"]),
            )
            action = _finalize_execution_action(
                scheduler,
                current_date,
                portfolio_state,
                decision_state,
                action,
                first_trade,
                self.execution_activity_config,
            )
            final_action = action.rebalance_action

            pending_action = _pending_action_for(
                action,
                current_date,
                final_action,
                segment_dates,
                self.execution_config,
            )
            if pending_action is None:
                self._pending_truncation_count += 1
                break
            pending_queue.append(pending_action)

        if len(pending_queue) > 0:
            self._pending_truncation_count += len(pending_queue)
            pending_queue.clear()

        if daily_diagnostics:
            daily_diagnostics[-1]["done_t"] = True
            if self._pending_truncation_count > 0:
                daily_diagnostics[-1]["termination_reason"] = "pending_truncation"
            else:
                daily_diagnostics[-1]["termination_reason"] = "segment_end"

        result = BacktestResult(
            daily_returns=pd.DataFrame(daily_returns, columns=DAILY_RETURNS_COLUMNS),
            daily_weights=pd.DataFrame(daily_weights, columns=DAILY_WEIGHTS_COLUMNS),
            daily_turnover=pd.DataFrame(daily_turnover, columns=DAILY_TURNOVER_COLUMNS),
            daily_rebalance=pd.DataFrame(daily_rebalance, columns=DAILY_REBALANCE_COLUMNS),
            daily_costs=pd.DataFrame(daily_costs, columns=DAILY_COSTS_COLUMNS),
            baseline_daily_diagnostics=pd.DataFrame(daily_diagnostics),
            daily_asset_returns=pd.DataFrame(daily_asset_returns),
            metrics=_metrics(daily_returns, daily_turnover, daily_costs, daily_diagnostics),
            run_manifest=self._run_manifest(dataset),
            portfolio_state=portfolio_state,
        )
        result.run_manifest["v_init_resolved_value"] = float(v_init_resolved_value)
        if output_dir is not None:
            result = _write_outputs(result, output_dir)
        return result

    def _initial_portfolio_state(self, asset_ids: Sequence[str], date: pd.Timestamp) -> PortfolioState:
        n_assets = len(asset_ids)
        zeros = np.zeros(n_assets, dtype=float)
        initial_nav = float(self.portfolio_config.get("initial_nav", 1.0))
        initial_capital = float(self.portfolio_config.get("initial_capital_currency", 0.0))
        return PortfolioState(
            date=date,
            nav=initial_nav,
            portfolio_value=initial_capital,
            current_weights=zeros.copy(),
            drifted_weights=zeros.copy(),
            previous_executed_weights=zeros.copy(),
            running_max_nav=initial_nav,
            current_drawdown_abs=0.0,
            rolling_returns=[],
            step_index=0,
            sellable_mask=np.ones(n_assets, dtype=bool),
            frozen_weight=zeros.copy(),
        )

    def _execute_step(
        self,
        portfolio_state: PortfolioState,
        action: PortfolioAction,
        execution_state: Any,
        final_action: int,
        first_trade: bool,
        asset_ids: Sequence[str],
    ) -> Any:
        if first_trade and not bool(self.execution_config.get("initial_build_cost", True)):
            original_cost_model = self.execution_core.cost_model
            self.execution_core.cost_model = _zero_cost_model(self.config)
            try:
                result = self.execution_core.execute_step(
                    portfolio_state.current_weights,
                    action.target_weights,
                    execution_state,
                    portfolio_state,
                    rebalance_action=final_action,
                    rebalance_intensity=action.rebalance_intensity,
                    asset_ids=list(asset_ids),
                    estimated_turnover=action.action_info.get("estimated_turnover"),
                    estimated_cost=action.action_info.get("estimated_cost"),
                )
            finally:
                self.execution_core.cost_model = original_cost_model
            result.info["initial_build_cost"] = False
            return result

        result = self.execution_core.execute_step(
            portfolio_state.current_weights,
            action.target_weights,
            execution_state,
            portfolio_state,
            rebalance_action=final_action,
            rebalance_intensity=action.rebalance_intensity,
            asset_ids=list(asset_ids),
            estimated_turnover=action.action_info.get("estimated_turnover"),
            estimated_cost=action.action_info.get("estimated_cost"),
        )
        result.info["initial_build_cost"] = bool(self.execution_config.get("initial_build_cost", True))
        return result

    def _run_manifest(self, dataset: MarketDatasetBundle) -> dict[str, Any]:
        flags = dict(self.execution_core.execution_manifest_flags)
        manifest = {
            "execution_model": deepcopy(self.config["execution_model"]),
            "data_governance": deepcopy(self.config["data_governance"]),
            "portfolio_initial_nav": float(self.portfolio_config.get("initial_nav", 1.0)),
            "portfolio_initial_capital_currency": float(self.portfolio_config.get("initial_capital_currency", 0.0)),
            "portfolio_currency": str(self.portfolio_config.get("currency", "")),
            "execution_price": flags.get("execution_price", self.execution_config.get("execution_price")),
            "execution_price_type": flags.get("execution_price_type", _execution_price_type(self.execution_config)),
            "valuation_source": flags.get("valuation_source", self.config["data_governance"].get("valuation_source")),
            "return_source": flags.get("return_source", self.config["data_governance"].get("return_source")),
            "valuation_execution_split": bool(
                flags.get(
                    "valuation_execution_split",
                    self.config["data_governance"].get("valuation_execution_split", False),
                )
            ),
            "reward_valuation_split": bool(
                flags.get(
                    "reward_valuation_split",
                    self.config["data_governance"].get("reward_valuation_split", False),
                )
            ),
            "delayed_action_execution": bool(
                flags.get("delayed_action_execution", self.execution_config.get("delayed_action_execution", False))
            ),
            "same_close_idealized_execution_enabled": bool(flags.get("same_close_idealized_execution_enabled", False)),
            "idealized_execution": bool(flags.get("idealized_execution", False)),
            "strict_no_lookahead_execution": bool(self.execution_config.get("strict_no_lookahead_execution", False)),
            "t_plus_one": bool(self.execution_config.get("t_plus_one", False)),
            "amount_is_proxy": bool(dataset.data_manifest.get("amount_is_proxy", False)),
            "initial_build_cost": bool(self.execution_config.get("initial_build_cost", True)),
            "execution_activity": deepcopy(self.execution_activity_config),
            "v_init_resolved_value": float(self.reward_calculator.v_init_resolved_value),
        }
        if self._pending_truncation_count:
            manifest["pending_action_truncation_count"] = int(self._pending_truncation_count)
            manifest["pending_action_truncation_reason"] = PENDING_TRUNCATION_REASON
        return manifest


def _action_for_step(
    strategy: Any,
    decision_state: DecisionMarketState,
    portfolio_state: PortfolioState,
    scheduler_allowed: bool,
    first_trade: bool,
    *,
    scheduler_blocks_model_actions: bool = True,
) -> PortfolioAction:
    if (
        bool(scheduler_blocks_model_actions)
        and not scheduler_allowed
        and not first_trade
        and not _requires_daily_diagnostics(strategy)
    ):
        return PortfolioAction(portfolio_state.current_weights.copy(), 0, 0.0, {})
    try:
        _set_strategy_decision_context(
            strategy,
            decision_state,
            portfolio_state,
            scheduler_allowed,
            first_trade,
            scheduler_blocks_model_actions=scheduler_blocks_model_actions,
        )
        action = strategy.compute_target_weights(decision_state, portfolio_state)
        if not isinstance(action, PortfolioAction):
            raise DataContractError(
                "ERR_STRATEGY_ACTION_CONTRACT",
                "ERR_STRATEGY_ACTION_CONTRACT: compute_target_weights must return PortfolioAction",
            )
        return action
    except Exception as exc:
        if _is_contract_error(exc):
            raise
        return _fallback_action(decision_state, exc)


def _set_strategy_decision_context(
    strategy: Any,
    decision_state: DecisionMarketState,
    portfolio_state: PortfolioState,
    scheduler_allowed: bool,
    first_trade: bool,
    *,
    scheduler_blocks_model_actions: bool = True,
) -> None:
    setter = getattr(strategy, "set_decision_context", None)
    if not callable(setter):
        return
    scheduler_allowed_rebalance = bool((not scheduler_blocks_model_actions) or scheduler_allowed or first_trade)
    setter(
        scheduler_allowed_rebalance=scheduler_allowed_rebalance,
        scheduler_pre_allowed=bool(scheduler_allowed),
        first_trade=bool(first_trade),
        decision_date=pd.Timestamp(decision_state.decision_date),
        portfolio_step_index=int(portfolio_state.step_index),
        scheduler_blocks_model_actions=bool(scheduler_blocks_model_actions),
    )


def _fallback_action(decision_state: DecisionMarketState, exc: Exception) -> PortfolioAction:
    available = np.asarray(decision_state.available_mask_at_decision, dtype=bool)
    if not available.any():
        raise DataContractError("ERR_CONSTRAINT_NO_AVAILABLE_ASSET", "ERR_CONSTRAINT_NO_AVAILABLE_ASSET: fallback") from exc
    weights = np.zeros_like(available, dtype=float)
    weights[available] = 1.0 / int(available.sum())
    reason = getattr(exc, "code", exc.__class__.__name__)
    return PortfolioAction(weights, 1, 1.0, {"fallback_reason": str(reason)})


def _assert_strategy_training_completed(strategy: Any) -> None:
    result = getattr(strategy, "training_result", None)
    if result is None:
        return
    status = result.get("status") if isinstance(result, Mapping) else None
    if status != "completed" or getattr(strategy, "is_fitted", False) is not True:
        model_name = getattr(strategy, "strategy_name", strategy.__class__.__name__)
        raise DataContractError(
            "ERR_STRATEGY_TRAINING_FAILED",
            f"ERR_STRATEGY_TRAINING_FAILED: {model_name} status={status or 'missing'}",
        )


def _pending_action_for(
    action: PortfolioAction,
    decision_date: pd.Timestamp,
    final_action: int,
    segment_dates: pd.DatetimeIndex,
    execution_config: Mapping[str, Any],
) -> PendingAction | None:
    decision_pos = _date_position(segment_dates, pd.Timestamp(decision_date))
    execution_price = str(execution_config.get("execution_price", "next_open"))
    if execution_price == "next_close":
        execution_offset = 1
        valuation_offset = 2
    elif execution_price == "next_open":
        execution_offset = 1
        valuation_offset = 1
    else:
        raise DataContractError(
            "ERR_CONFIG_INVALID_EXECUTION_MODEL",
            "ERR_CONFIG_INVALID_EXECUTION_MODEL: execution_model.execution_price",
        )
    execution_price_type = _execution_price_type(execution_config)
    if decision_pos + valuation_offset >= len(segment_dates):
        return None

    action_info = dict(action.action_info)
    return PendingAction(
        decision_date=pd.Timestamp(decision_date),
        execution_date=pd.Timestamp(segment_dates[decision_pos + execution_offset]),
        next_valuation_date=pd.Timestamp(segment_dates[decision_pos + valuation_offset]),
        target_weights=action.target_weights.copy(),
        candidate_weights=action.target_weights.copy(),
        rebalance_action=final_action,
        rebalance_intensity=float(action.rebalance_intensity),
        execution_price=execution_price,
        execution_price_type=execution_price_type,
        q_hold=_optional_action_float(action_info.get("q_hold")),
        q_rebalance=_optional_action_float(action_info.get("q_rebalance")),
        q_gap=_optional_action_float(action_info.get("q_gap")),
        decision_value=_optional_action_float(action_info.get("decision_value")),
        action_info=action_info,
    )


def _execution_price_type(execution_config: Mapping[str, Any]) -> str | None:
    execution_price = str(execution_config.get("execution_price", "next_open"))
    if execution_price == "next_close":
        return "close"
    if execution_price == "next_open":
        return "open"
    return None


def _action_from_pending(pending_action: PendingAction) -> PortfolioAction:
    return PortfolioAction(
        pending_action.target_weights.copy(),
        pending_action.rebalance_action,
        pending_action.rebalance_intensity,
        dict(pending_action.action_info),
    )


def _optional_action_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _build_decision_market_state(
    dataset: MarketDatasetBundle,
    decision_date: pd.Timestamp,
    config: Mapping[str, Any],
    *,
    market_image_dataset: Any | None = None,
) -> DecisionMarketState:
    assert_decision_visibility_contract(
        market_image=dataset.feature_cols,
        feature_window=dataset.feature_cols,
    )
    date_index = _date_index(dataset)
    position = _date_position(date_index, decision_date)
    asset_ids = _asset_order(dataset)
    window_size = int(config.get("env", {}).get("window_size", config.get("feature_matrix", {}).get("window_size", 60)))
    log_return = _wide(dataset, "log_return", asset_ids)
    log_return_window = _window(log_return, position, window_size)
    market_image = _decision_market_image(dataset, asset_ids, position, window_size, market_image_dataset, decision_date, log_return_window)
    return DecisionMarketState(
        decision_date=decision_date,
        available_mask_at_decision=_row(dataset.availability_mask, decision_date, dtype=bool),
        availability_reason_at_decision=_availability_reason(dataset, decision_date),
        close_at_decision=_row(_wide(dataset, "close", asset_ids), decision_date),
        log_return_at_decision=_row(log_return, decision_date),
        log_return_window=log_return_window,
        amount_at_decision=_row(_wide(dataset, "amount", asset_ids), decision_date),
        volume_at_decision=_row(_wide(dataset, "vol", asset_ids), decision_date),
        adv20_at_decision=_row(_wide(dataset, "amount", asset_ids).rolling(20, min_periods=1).mean(), decision_date),
        volatility_20d_at_decision=_row(log_return.rolling(20, min_periods=1).std(ddof=0).fillna(0.0), decision_date),
        turnover_rate_at_decision=_optional_wide_row(dataset, "turnover_rate", asset_ids, decision_date),
        feature_window=market_image,
        market_image=market_image,
    )


def _decision_dates(split: Any, segment: str, date_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = _segment_dates(split, segment)
    if dates.empty:
        raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: {segment}")
    dates = dates[dates.isin(date_index)]
    last_decision = getattr(split, f"{segment}_last_decision_date", None)
    if last_decision is not None:
        dates = dates[dates <= pd.Timestamp(last_decision)]
    elif len(dates) > 1:
        dates = dates[:-1]
    else:
        dates = pd.DatetimeIndex([])
    if dates.empty:
        raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: {segment}_decision_dates")
    return pd.DatetimeIndex(dates)


def _segment_dates(split: Any, segment: str) -> pd.DatetimeIndex:
    if segment == "all":
        parts = []
        for name in ("train", "validation", "test"):
            values = getattr(split, f"{name}_dates", None)
            if values is not None:
                parts.extend(list(values))
        return pd.DatetimeIndex(pd.to_datetime(parts)).sort_values()
    values = getattr(split, f"{segment}_dates", None)
    if values is None:
        raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: split.{segment}_dates")
    return pd.DatetimeIndex(pd.to_datetime(list(values))).sort_values()


def _fit_payload(
    dataset: MarketDatasetBundle,
    split: Any,
    segment: str,
    *,
    market_image_dataset: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dates": _segment_dates(split, segment),
        "segment": segment,
        "market_image_dataset": market_image_dataset,
        "config": {} if config is None else dict(config),
    }


def _date_index(dataset: MarketDatasetBundle) -> pd.DatetimeIndex:
    close = _wide(dataset, "close", _asset_order(dataset))
    index = pd.DatetimeIndex(close.index)
    if index.empty or not index.is_monotonic_increasing:
        raise DataContractError("ERR_DATA_SCHEMA_MISMATCH", "ERR_DATA_SCHEMA_MISMATCH: wide_close date index")
    return index


def _date_position(date_index: pd.DatetimeIndex, date: pd.Timestamp) -> int:
    matches = np.flatnonzero(date_index == pd.Timestamp(date))
    if matches.size == 0:
        raise DataContractError("ERR_EXECUTION_DATE_OUT_OF_RANGE", "ERR_EXECUTION_DATE_OUT_OF_RANGE: decision_date")
    return int(matches[0])


def _asset_order(dataset: MarketDatasetBundle) -> list[str]:
    manifest_order = dataset.data_manifest.get("canonical_asset_order")
    if isinstance(manifest_order, list) and manifest_order:
        return [str(asset) for asset in manifest_order]
    return [str(column) for column in dataset.availability_mask.columns]


def _wide(dataset: MarketDatasetBundle, field: str, asset_ids: Sequence[str]) -> pd.DataFrame:
    if field not in dataset.wide:
        raise DataContractError("ERR_DATA_MISSING_FILE", f"ERR_DATA_MISSING_FILE: wide_{field}")
    frame = dataset.wide[field].copy()
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.reindex(columns=list(asset_ids)).sort_index()


def _row(frame: pd.DataFrame, date: pd.Timestamp, *, dtype: Any = float) -> np.ndarray:
    if date not in frame.index:
        raise DataContractError("ERR_EXECUTION_DATE_OUT_OF_RANGE", "ERR_EXECUTION_DATE_OUT_OF_RANGE: row")
    return frame.loc[date].to_numpy(dtype=dtype, copy=True)


def _availability_reason(dataset: MarketDatasetBundle, date: pd.Timestamp) -> np.ndarray | None:
    if dataset.availability_reason is None:
        return None
    return _row(dataset.availability_reason, date, dtype=object)


def _optional_wide_row(
    dataset: MarketDatasetBundle,
    field: str,
    asset_ids: Sequence[str],
    date: pd.Timestamp,
) -> np.ndarray:
    if field not in dataset.wide:
        return np.full(len(asset_ids), np.nan, dtype=float)
    return _row(_wide(dataset, field, asset_ids), date)


def _window(frame: pd.DataFrame, end_position: int, window_size: int) -> np.ndarray:
    start = max(0, end_position - max(1, window_size) + 1)
    return frame.iloc[start : end_position + 1].to_numpy(dtype=float, copy=True)


def _decision_market_image(
    dataset: MarketDatasetBundle,
    asset_ids: Sequence[str],
    position: int,
    window_size: int,
    market_image_dataset: Any | None,
    decision_date: pd.Timestamp,
    fallback_window: np.ndarray,
) -> np.ndarray:
    if market_image_dataset is not None:
        try:
            return np.asarray(market_image_dataset[decision_date], dtype=float)
        except Exception:
            pass
    feature_cols = [str(item) for item in getattr(dataset, "feature_cols", [])]
    feature_windows = []
    for feature in feature_cols:
        if feature in dataset.wide:
            feature_windows.append(_window(_wide(dataset, feature, asset_ids), position, window_size))
    if feature_windows:
        return np.stack(feature_windows, axis=0)
    return fallback_window[np.newaxis, :, :]


def _requires_daily_diagnostics(strategy: Any) -> bool:
    return bool(
        getattr(strategy, "requires_daily_diagnostics", False)
        or getattr(strategy, "requires_daily_output", False)
    )


def _backtest_risk_state_required(config: Mapping[str, Any]) -> bool:
    model_config = config.get("model")
    reward_config = config.get("reward")
    risk_config = config.get("risk_state")
    model_requires = bool(isinstance(model_config, Mapping) and model_config.get("use_risk_state", False))
    reward_requires = bool(
        isinstance(reward_config, Mapping)
        and str(reward_config.get("mode", "")) == "A13_otar_soft_ru_cvar_fixed"
    )
    risk_enabled = bool(isinstance(risk_config, Mapping) and risk_config.get("enabled", False))
    return model_requires or reward_requires or risk_enabled


def _mark_scheduler_rebalanced(scheduler: RebalanceScheduler, date: pd.Timestamp) -> None:
    scheduler._last_allowed_date = pd.Timestamp(date)
    scheduler._has_rebalanced = True


def _finalize_execution_action(
    scheduler: RebalanceScheduler,
    decision_date: pd.Timestamp,
    portfolio_state: PortfolioState,
    decision_state: DecisionMarketState,
    action: PortfolioAction,
    first_trade: bool,
    execution_activity_config: Mapping[str, Any],
) -> PortfolioAction:
    evaluation = scheduler.evaluate_pre_post_no_mutation(
        decision_date,
        portfolio_state,
        decision_state,
        candidate_weights=action.target_weights,
    )
    scheduler_blocks = bool(execution_activity_config["scheduler_blocks_model_actions"])
    protocol = str(execution_activity_config["protocol"])
    action_info = dict(action.action_info)
    raw_rho = _raw_rebalance_intensity(action)
    raw_model_requested = _raw_model_requested(action, raw_rho)
    scheduler_final_allowed = bool(evaluation.scheduler_final_allowed)
    execution_gate_allowed = True if not scheduler_blocks else scheduler_final_allowed
    if first_trade:
        final_action_bool = True
        final_rho = raw_rho if raw_rho > 0.0 else 1.0
        _mark_scheduler_rebalanced(scheduler, pd.Timestamp(decision_date))
    else:
        final_action_bool = bool(raw_model_requested and execution_gate_allowed)
        final_rho = raw_rho if final_action_bool else 0.0
        scheduler.commit_scheduler_decision(
            decision_date,
            scheduler_pre_allowed=evaluation.scheduler_pre_allowed,
            scheduler_post_allowed=evaluation.scheduler_post_allowed,
            scheduler_final_allowed=evaluation.scheduler_final_allowed,
            raw_model_requested_rebalance=raw_model_requested,
            final_action=final_action_bool,
            execution_accepted=final_action_bool,
        )
    scheduler_pre_blocked = bool(raw_model_requested and not evaluation.scheduler_pre_allowed)
    scheduler_post_blocked = bool(
        raw_model_requested
        and evaluation.scheduler_pre_allowed
        and not evaluation.scheduler_post_allowed
    )
    scheduler_final_blocked = bool(raw_model_requested and not scheduler_final_allowed)
    execution_scheduler_blocked = bool(
        raw_model_requested
        and not final_action_bool
        and scheduler_blocks
        and not scheduler_final_allowed
    )
    model_chosen_hold = bool(
        not first_trade
        and execution_gate_allowed
        and raw_rho == 0.0
        and not raw_model_requested
    )
    action_info.update(
        {
            "execution_activity_protocol": protocol,
            "activity_protocol": protocol,
            "scheduler_blocks_model_actions": scheduler_blocks,
            "activity_gate_enforced": bool(execution_activity_config.get("activity_gate_enforced", False)),
            "turnover_optimization_protocol_id": execution_activity_config.get("turnover_optimization_protocol_id"),
            "scheduler_pre_allowed": bool(evaluation.scheduler_pre_allowed),
            "scheduler_post_allowed": bool(evaluation.scheduler_post_allowed),
            "scheduler_final_allowed": scheduler_final_allowed,
            "scheduler_allowed_rebalance": scheduler_final_allowed,
            "first_trade": bool(first_trade),
            "execution_gate_allowed": bool(execution_gate_allowed),
            "raw_rho": float(raw_rho),
            "raw_rebalance_intensity": float(raw_rho),
            "raw_model_requested_rebalance": bool(raw_model_requested),
            "raw_action": int(raw_model_requested),
            "final_action": int(final_action_bool),
            "final_rho": float(final_rho),
            "final_rebalance_intensity": float(final_rho),
            "scheduler_pre_blocked": scheduler_pre_blocked,
            "scheduler_post_blocked": scheduler_post_blocked,
            "scheduler_final_blocked": scheduler_final_blocked,
            "scheduler_blocked_rebalance": scheduler_final_blocked,
            "execution_scheduler_blocked": execution_scheduler_blocked,
            "model_chosen_hold": model_chosen_hold,
            "trade_opportunity": bool(first_trade or execution_gate_allowed),
            "non_initial_trade_opportunity": bool((not first_trade) and execution_gate_allowed),
        }
    )
    if execution_scheduler_blocked:
        action_info["forced_hold_reason"] = "scheduler_blocked"
    elif model_chosen_hold and not action_info.get("forced_hold_reason"):
        action_info["forced_hold_reason"] = "model_chosen_hold"
    return PortfolioAction(
        target_weights=action.target_weights.copy(),
        rebalance_action=int(final_action_bool),
        rebalance_intensity=float(final_rho),
        action_info=action_info,
    )


def _raw_rebalance_intensity(action: PortfolioAction) -> float:
    info = dict(action.action_info)
    for key in ("raw_rho", "raw_rebalance_intensity", "rho", "rebalance_intensity"):
        if key in info and info[key] is not None:
            return _bounded_optional_float(info[key], key)
    if action.rebalance_action == 1:
        return float(action.rebalance_intensity)
    return 0.0


def _raw_model_requested(action: PortfolioAction, raw_rho: float) -> bool:
    info = dict(action.action_info)
    if "raw_model_requested_rebalance" in info:
        return bool(info["raw_model_requested_rebalance"])
    if "raw_action" in info:
        return bool(int(info["raw_action"]))
    return bool(raw_rho > 0.0 and action.rebalance_action == 1)


def _bounded_optional_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", f"ERR_ACTION_SHAPE_MISMATCH: {name}") from exc
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", f"ERR_ACTION_SHAPE_MISMATCH: {name}")
    return result


def _execution_activity_config(config: Mapping[str, Any]) -> dict[str, Any]:
    defaults = DEFAULT_CONFIG["execution_activity"]
    raw = config.get("execution_activity", {})
    activity = {**defaults, **(dict(raw) if isinstance(raw, Mapping) else {})}
    protocol = str(activity.get("protocol", "monthly_gate"))
    if protocol not in {"monthly_gate", "weekly_gate", "daily_gate_with_cost_constraint"}:
        raise DataContractError(
            "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY",
            "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY: execution_activity.protocol",
        )
    scheduler_blocks = bool(activity.get("scheduler_blocks_model_actions", True))
    if protocol in {"monthly_gate", "weekly_gate"} and scheduler_blocks is not True:
        raise DataContractError(
            "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY",
            "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY: scheduler_blocks_model_actions",
        )
    if protocol == "daily_gate_with_cost_constraint" and scheduler_blocks is not False:
        raise DataContractError(
            "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY",
            "ERR_CONFIG_INVALID_EXECUTION_ACTIVITY: daily_gate_with_cost_constraint requires scheduler_blocks_model_actions=false",
        )
    return activity


def _daily_returns_record(context: Mapping[str, Any]) -> dict[str, Any]:
    result = context["execution_result"]
    state = context["execution_state"]
    metadata = context["metadata"]
    return {
        "date": state.next_valuation_date,
        "decision_date": state.decision_date,
        "execution_date": state.execution_date,
        "execution_price_type": state.execution_price_type,
        "next_valuation_date": state.next_valuation_date,
        "split": metadata["split"],
        "seed": metadata["seed"],
        "fold_id": metadata["fold_id"],
        "model_name": metadata["model_name"],
        "pre_execution_return": result.pre_execution_return,
        "post_execution_return": result.post_execution_return,
        "gross_return": result.gross_return,
        "transaction_cost": result.transaction_cost,
        "transaction_cost_on_initial_nav": result.transaction_cost_on_initial_nav,
        "net_return": result.net_return,
        "portfolio_log_return": result.portfolio_log_return,
        "nav": result.nav_next,
        "reward": context["reward"],
    }


def _is_contract_error(exc: Exception) -> bool:
    return isinstance(exc, DataContractError) and exc.code in CONTRACT_ERROR_CODES


def _daily_turnover_record(context: Mapping[str, Any], final_action: int) -> dict[str, Any]:
    result = context["execution_result"]
    action = context["action"]
    state = context["execution_state"]
    metadata = context["metadata"]
    return {
        "date": state.next_valuation_date,
        "decision_date": state.decision_date,
        "execution_date": state.execution_date,
        "execution_price_type": state.execution_price_type,
        "next_valuation_date": state.next_valuation_date,
        "split": metadata["split"],
        "seed": metadata["seed"],
        "fold_id": metadata["fold_id"],
        "model_name": metadata["model_name"],
        "turnover": result.turnover,
        "rebalance_action": final_action,
        "rebalance_intensity": action.rebalance_intensity,
        "average_holding_period": np.nan if result.turnover == 0.0 else 1.0 / result.turnover,
    }


def _daily_rebalance_record(context: Mapping[str, Any], final_action: int) -> dict[str, Any]:
    result = context["execution_result"]
    action = context["action"]
    state = context["execution_state"]
    metadata = context["metadata"]
    action_info = dict(action.action_info)
    return {
        "date": state.next_valuation_date,
        "decision_date": state.decision_date,
        "execution_date": state.execution_date,
        "execution_price_type": state.execution_price_type,
        "next_valuation_date": state.next_valuation_date,
        "split": metadata["split"],
        "seed": metadata["seed"],
        "fold_id": metadata["fold_id"],
        "model_name": metadata["model_name"],
        "rebalance_action": final_action,
        "rebalance_intensity": action.rebalance_intensity,
        "estimated_turnover": result.estimated_turnover,
        "realized_turnover": result.realized_turnover,
        "turnover": result.turnover,
        "estimated_cost": result.estimated_cost,
        "realized_cost": result.realized_cost,
        "q_hold": action_info.get("q_hold"),
        "q_rebalance": action_info.get("q_rebalance"),
        "q_gap": action_info.get("q_gap"),
        "fallback_reason": action_info.get("fallback_reason"),
    }


def _baseline_diagnostics_record(context: Mapping[str, Any]) -> dict[str, Any]:
    action = context["action"]
    result = context["execution_result"]
    state = context["execution_state"]
    metadata = context["metadata"]
    action_info = dict(action.action_info)
    boundary = context.get("boundary", {})
    reward_info = context.get("reward_info", {})
    pre_step_prev_weights = context.get("pre_step_previous_executed_weights")

    passive_weight_drift_l1 = 0.0
    if pre_step_prev_weights is not None:
        passive_weight_drift_l1 = float(np.sum(np.abs(result.pre_execution_drifted_weights - pre_step_prev_weights)))
    turnover_from_active_rebalance = float(np.sum(np.abs(result.executed_weights - result.pre_execution_drifted_weights)))

    pre_execution_return = float(result.pre_execution_return) if hasattr(result, "pre_execution_return") else 0.0
    realized_gross_simple_return = float(result.gross_return) if hasattr(result, "gross_return") else 0.0
    realized_gate_action_cost = float(result.total_transaction_cost)
    soft_tail_proxy = float(reward_info.get("realized_gate_action_soft_tail_proxy", 0.0))

    executed_gate_action = int(action_info.get("executed_gate_action", context.get("final_action", action.rebalance_action)))
    if executed_gate_action == 0:
        estimated_cost = float(action_info.get("hold_estimated_cost", 0.0))
    else:
        estimated_cost = float(action_info.get("candidate_estimated_cost", action_info.get("estimated_cost", 0.0)))
    cost_estimation_error = abs(estimated_cost - realized_gate_action_cost)

    v_t_pre = float(reward_info.get("soft_cvar_v_t_pre_update", 0.0))
    v_t_post = float(reward_info.get("soft_cvar_v_t_post_update", 0.0))

    return {
        **action_info,
        "date": state.next_valuation_date,
        "decision_date": state.decision_date,
        "execution_date": state.execution_date,
        "model_name": metadata["model_name"],
        "paper_model_id": action_info.get("paper_model_id") or metadata["model_name"],
        "seed": metadata["seed"],
        "fold_id": metadata["fold_id"],
        "rebalance_action": int(context.get("final_action", action.rebalance_action)),
        "rebalance_intensity": float(action.rebalance_intensity),
        "target_weights_json": _weights_json(action.target_weights),
        "candidate_weights_json": action_info.get("candidate_weights_json", _weights_json(action.target_weights)),
        "executed_weights_json": _weights_json(result.executed_weights),
        "pre_execution_drifted_weights_json": _weights_json(result.pre_execution_drifted_weights),
        "active_weight_change_l1": float(np.sum(np.abs(result.executed_weights - result.pre_execution_drifted_weights))),
        "estimated_turnover": result.estimated_turnover,
        "realized_turnover": result.realized_turnover,
        "turnover": result.turnover,
        "estimated_cost": result.estimated_cost,
        "realized_cost": result.realized_cost,
        "total_transaction_cost": result.total_transaction_cost,
        "net_return": result.net_return,
        "portfolio_log_return": result.portfolio_log_return,
        "nav": result.nav_next,
        # Weight drift
        "pre_trade_drifted_weights_json": _weights_json(result.pre_execution_drifted_weights),
        "post_trade_target_weights_json": _weights_json(action.target_weights),
        "passive_weight_drift_l1_t": passive_weight_drift_l1,
        "turnover_from_active_rebalance_t": turnover_from_active_rebalance,
        # Post-step realized
        "actual_pre_execution_return_t": pre_execution_return,
        "realized_gross_simple_return_t": realized_gross_simple_return,
        "realized_gate_action_cost": realized_gate_action_cost,
        "realized_gate_action_soft_tail_proxy": soft_tail_proxy,
        "cost_estimation_error_t": cost_estimation_error,
        # v_t series
        "soft_cvar_v_t_pre_update": v_t_pre,
        "soft_cvar_v_t_post_update": v_t_post,
        # Boundary
        "run_id": boundary.get("run_id", ""),
        "episode_id": boundary.get("episode_id", ""),
        "timestep": boundary.get("timestep", 0),
        "done_t": boundary.get("done_t", False),
        "split": boundary.get("split", metadata.get("split", "")),
        "termination_reason": boundary.get("termination_reason", ""),
    }


def _weights_json(weights: Any) -> str:
    array = np.asarray(weights, dtype=float).reshape(-1)
    return json.dumps([float(value) for value in array], separators=(",", ":"))


def _daily_asset_returns_records(
    context: Mapping[str, Any],
    asset_ids: Sequence[str],
    boundary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = context["execution_result"]
    state = context["execution_state"]
    info = result.info
    pre_returns = info.get("pre_execution_asset_simple_return", [])
    post_returns = info.get("post_execution_asset_simple_return", [])
    if not pre_returns or not post_returns:
        return []
    records: list[dict[str, Any]] = []
    for idx, asset_id in enumerate(asset_ids):
        pre_ret = float(pre_returns[idx]) if idx < len(pre_returns) else 0.0
        post_ret = float(post_returns[idx]) if idx < len(post_returns) else 0.0
        decision_to_next = (1.0 + pre_ret) * (1.0 + post_ret) - 1.0
        records.append({
            "run_id": boundary.get("run_id", ""),
            "episode_id": boundary.get("episode_id", ""),
            "timestep": boundary.get("timestep", 0),
            "date": state.next_valuation_date,
            "decision_date": state.decision_date,
            "execution_date": state.execution_date,
            "next_valuation_date": state.next_valuation_date,
            "split": boundary.get("split", metadata.get("split", "")),
            "seed": metadata.get("seed", 0),
            "fold_id": metadata.get("fold_id", 0),
            "model_name": metadata.get("model_name", ""),
            "asset_index": idx,
            "asset_id": str(asset_id),
            "pre_execution_simple_return": pre_ret,
            "post_execution_simple_return": post_ret,
            "decision_to_next_simple_return": decision_to_next,
        })
    return records


def _has_paper_model_id(value: Any) -> bool:
    if value is None:
        return False
    return bool(str(value).strip())


def _should_record_daily_diagnostics(action: PortfolioAction, execution_activity_config: Mapping[str, Any]) -> bool:
    if _has_paper_model_id(action.action_info.get("paper_model_id")):
        return True
    return (
        str(execution_activity_config.get("protocol", "")) == "daily_gate_with_cost_constraint"
        or bool(execution_activity_config.get("activity_gate_enforced", False))
    )


def _daily_costs_record(context: Mapping[str, Any]) -> dict[str, Any]:
    result = context["execution_result"]
    state = context["execution_state"]
    metadata = context["metadata"]
    return {
        "date": state.next_valuation_date,
        "decision_date": state.decision_date,
        "execution_date": state.execution_date,
        "execution_price_type": state.execution_price_type,
        "next_valuation_date": state.next_valuation_date,
        "split": metadata["split"],
        "seed": metadata["seed"],
        "fold_id": metadata["fold_id"],
        "model_name": metadata["model_name"],
        "proportional_cost": result.proportional_cost,
        "fixed_cost": result.fixed_cost,
        "slippage_cost": result.slippage_cost,
        "market_impact_cost": result.market_impact_cost,
        "total_transaction_cost": result.total_transaction_cost,
        "estimated_cost": result.estimated_cost,
        "realized_cost": result.realized_cost,
        "turnover": result.turnover,
    }


def _daily_weights_records(
    date: pd.Timestamp,
    asset_ids: Sequence[str],
    weights: np.ndarray,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "date": date,
            "split": metadata["split"],
            "seed": metadata["seed"],
            "fold_id": metadata["fold_id"],
            "model_name": metadata["model_name"],
            "asset_id": str(asset_id),
            "weight": float(weight),
        }
        for asset_id, weight in zip(asset_ids, weights, strict=True)
    ]


def _record_metadata(
    segment: str,
    split: Any,
    strategy: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    output_config = config.get("output")
    run_id = None
    if isinstance(output_config, Mapping):
        run_id = output_config.get("run_name")
    return {
        "run_id": str(run_id or _model_name(strategy)),
        "split": str(segment),
        "seed": _seed(config),
        "fold_id": getattr(split, "fold_id", None),
        "model_name": _model_name(strategy),
    }


def _seed(config: Mapping[str, Any]) -> int | None:
    reproducibility = config.get("reproducibility", {})
    if isinstance(reproducibility, Mapping) and reproducibility.get("seed") is not None:
        return int(reproducibility["seed"])
    return None


def _model_name(strategy: Any) -> str:
    strategy_name = getattr(strategy, "strategy_name", None)
    if strategy_name is not None:
        return str(strategy_name)
    return strategy.__class__.__name__


def _metrics(
    daily_returns: Sequence[Mapping[str, Any]],
    daily_turnover: Sequence[Mapping[str, Any]],
    daily_costs: Sequence[Mapping[str, Any]],
    daily_diagnostics: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not daily_returns:
        return {"n_steps": 0.0, "final_nav": np.nan, "cumulative_return": np.nan}
    final_nav = float(daily_returns[-1]["nav"])
    total_cost = float(sum(float(row["total_transaction_cost"]) for row in daily_costs))
    avg_turnover = float(np.mean([float(row["turnover"]) for row in daily_turnover]))
    metrics = {
        "n_steps": float(len(daily_returns)),
        "final_nav": final_nav,
        "cumulative_return": final_nav - 1.0,
        "total_transaction_cost": total_cost,
        "average_turnover": avg_turnover,
    }
    metrics.update(_risk_metrics(daily_returns))
    metrics.update(_activity_metrics(daily_diagnostics or []))
    return metrics


def _risk_metrics(
    daily_returns: Sequence[Mapping[str, Any]],
    *,
    annualization_factor: int = 252,
    cvar_alpha: float = 0.05,
) -> dict[str, Any]:
    returns_arr = np.array([float(row["net_return"]) for row in daily_returns], dtype=np.float64)
    returns_arr = returns_arr[np.isfinite(returns_arr)]
    if len(returns_arr) < 2:
        return {
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "max_drawdown_abs": np.nan,
            "var95_loss": np.nan,
            "cvar95_loss": np.nan,
        }
    cumulative_return = float(np.prod(1.0 + returns_arr) - 1.0)
    n = len(returns_arr)
    annualized_return = float((1.0 + cumulative_return) ** (annualization_factor / n) - 1.0)
    std = float(returns_arr.std(ddof=0))
    annualized_volatility = float(std * np.sqrt(annualization_factor))
    sharpe = 0.0 if std == 0.0 else float(returns_arr.mean() / std * np.sqrt(annualization_factor))
    downside = returns_arr[returns_arr < 0.0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
    sortino = 0.0 if downside_std == 0.0 else float(returns_arr.mean() / downside_std * np.sqrt(annualization_factor))
    nav = np.cumprod(1.0 + returns_arr)
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    max_drawdown_abs = float(abs(drawdown.min()))
    calmar = 0.0 if max_drawdown_abs == 0.0 else float(annualized_return / max_drawdown_abs)
    threshold = float(np.quantile(returns_arr, cvar_alpha))
    tail = returns_arr[returns_arr <= threshold]
    cvar95_loss = float(-tail.mean()) if len(tail) > 0 else np.nan
    var95_loss = float(-threshold)
    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown_abs": max_drawdown_abs,
        "var95_loss": var95_loss,
        "cvar95_loss": cvar95_loss,
    }


def _activity_metrics(daily_diagnostics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not daily_diagnostics:
        return {}
    frame = pd.DataFrame(daily_diagnostics)
    if frame.empty:
        return {}
    raw_requested = _numeric_series(frame, "raw_model_requested_rebalance").fillna(0.0) > 0.0
    final_action = _numeric_series(frame, "final_action").fillna(_numeric_series(frame, "rebalance_action").fillna(0.0)) > 0.0
    trade_opportunity = _numeric_series(frame, "trade_opportunity").fillna(0.0) > 0.0
    non_initial_opportunity = _numeric_series(frame, "non_initial_trade_opportunity").fillna(0.0) > 0.0
    first_trade = _numeric_series(frame, "first_trade").fillna(0.0) > 0.0
    non_initial_final = final_action & ~first_trade
    turnover = _numeric_series(frame, "turnover").fillna(0.0)
    non_initial_turnover_sum = float(turnover[~first_trade].sum())
    non_initial_trade_opportunity_count = int(non_initial_opportunity.sum())
    model_rebalance_hit_rate = float(non_initial_final.sum()) / float(max(1, non_initial_trade_opportunity_count))
    metrics: dict[str, Any] = {
        "trade_opportunity_count": float(trade_opportunity.sum()),
        "non_initial_trade_opportunity_count": float(non_initial_trade_opportunity_count),
        "model_rebalance_hit_rate": model_rebalance_hit_rate,
        "raw_model_requested_rebalance_count": float(raw_requested.sum()),
        "non_initial_rebalance_count": float(non_initial_final.sum()),
        "non_initial_turnover_sum": non_initial_turnover_sum,
        "non_initial_turnover_per_opportunity": non_initial_turnover_sum / float(max(1, non_initial_trade_opportunity_count)),
        "raw_rho_mean": float(_numeric_series(frame, "raw_rho").mean()),
        "final_rho_mean": float(_numeric_series(frame, "final_rho").mean()),
        "active_weight_change_l1_mean": float(_numeric_series(frame, "active_weight_change_l1").mean()),
        "scheduler_pre_blocked_count": float((_numeric_series(frame, "scheduler_pre_blocked").fillna(0.0) > 0.0).sum()),
        "scheduler_post_blocked_count": float((_numeric_series(frame, "scheduler_post_blocked").fillna(0.0) > 0.0).sum()),
        "scheduler_final_blocked_count": float((_numeric_series(frame, "scheduler_final_blocked").fillna(0.0) > 0.0).sum()),
        "scheduler_blocked_rebalance_count": float((_numeric_series(frame, "scheduler_blocked_rebalance").fillna(0.0) > 0.0).sum()),
        "model_chosen_hold_count": float((_numeric_series(frame, "model_chosen_hold").fillna(0.0) > 0.0).sum()),
        "execution_scheduler_blocked_count": float((_numeric_series(frame, "execution_scheduler_blocked").fillna(0.0) > 0.0).sum()),
    }
    protocol = _first_text_value(frame, "activity_protocol", "execution_activity_protocol")
    if protocol:
        metrics["activity_protocol"] = protocol
        metrics["execution_activity_protocol"] = protocol
    turnover_protocol = _first_text_value(frame, "turnover_optimization_protocol_id")
    if turnover_protocol:
        metrics["turnover_optimization_protocol_id"] = turnover_protocol
    scheduler_blocks = _first_text_value(frame, "scheduler_blocks_model_actions")
    if scheduler_blocks:
        metrics["scheduler_blocks_model_actions"] = scheduler_blocks
    activity_gate = _first_text_value(frame, "activity_gate_enforced")
    if activity_gate:
        metrics["activity_gate_enforced"] = activity_gate

    # CQR gate aggregation fields
    executed_gate = _numeric_series(frame, "executed_gate_action")
    if executed_gate.isna().all():
        executed_gate = _numeric_series(frame, "gate_action")
    executed_gate = executed_gate.fillna(0.0)
    gate_action_count = float(executed_gate.sum())
    gate_hold_count = float((executed_gate < 0.5).sum())
    gate_total = gate_action_count + gate_hold_count
    metrics["gate_action_ratio"] = gate_action_count / float(max(1, gate_total))
    metrics["gate_action_hold_count"] = gate_hold_count
    metrics["gate_action_rebalance_count"] = gate_action_count
    if gate_total > 0:
        p_rebalance = gate_action_count / gate_total
        p_hold = gate_hold_count / gate_total
        entropy = 0.0
        if p_rebalance > 0:
            entropy -= p_rebalance * np.log(p_rebalance)
        if p_hold > 0:
            entropy -= p_hold * np.log(p_hold)
        metrics["gate_action_entropy"] = float(entropy)
    else:
        metrics["gate_action_entropy"] = 0.0
    raw_gate = _numeric_series(frame, "raw_gate_action")
    if raw_gate.isna().all():
        raw_gate = _numeric_series(frame, "raw_model_requested_rebalance")
    metrics["raw_gate_action_ratio"] = float(raw_gate.fillna(0.0).mean())
    pred_delta = _numeric_series(frame, "pred_delta_utility").dropna()
    metrics["q_value_gap_mean"] = float(pred_delta.mean()) if len(pred_delta) > 0 else 0.0
    metrics["q_value_gap_std"] = float(pred_delta.std()) if len(pred_delta) > 1 else 0.0
    actor_update = _numeric_series(frame, "ppo_actor_update_mask").fillna(1.0)
    metrics["actor_update_count"] = float(actor_update.sum())
    metrics["actor_skipped_by_gate_count"] = float((actor_update < 0.5).sum())
    total_actor = metrics["actor_update_count"] + metrics["actor_skipped_by_gate_count"]
    metrics["effective_actor_update_ratio"] = metrics["actor_update_count"] / float(max(1, total_actor))

    return metrics


def _first_text_value(frame: pd.DataFrame, *columns: str) -> str | None:
    for column in columns:
        if column not in frame.columns:
            continue
        for value in frame[column].dropna().tolist():
            text = str(value).strip()
            if text:
                return text
    return None


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _zero_cost_model(config: Mapping[str, Any]) -> Any:
    from src.envs.cost_model import CostModel

    zero_config = deepcopy(dict(config))
    zero_config["cost_model"]["proportional_cost"] = 0.0
    zero_config["cost_model"]["fixed_cost"] = 0.0
    zero_config["cost_model"]["slippage"] = 0.0
    zero_config["cost_model"]["market_impact_enabled"] = False
    return CostModel(zero_config)


def _write_outputs(result: BacktestResult, output_dir: str | Path) -> BacktestResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "daily_returns": output_path / "daily_returns.csv",
        "daily_weights": output_path / "daily_weights.csv",
        "daily_turnover": output_path / "daily_turnover.csv",
        "daily_rebalance": output_path / "daily_rebalance.csv",
        "daily_costs": output_path / "daily_costs.csv",
    }
    result.daily_returns.to_csv(artifact_paths["daily_returns"], index=False)
    result.daily_weights.to_csv(artifact_paths["daily_weights"], index=False)
    result.daily_turnover.to_csv(artifact_paths["daily_turnover"], index=False)
    result.daily_rebalance.to_csv(artifact_paths["daily_rebalance"], index=False)
    result.daily_costs.to_csv(artifact_paths["daily_costs"], index=False)
    if not result.baseline_daily_diagnostics.empty:
        artifact_paths["baseline_daily_diagnostics"] = output_path / "baseline_daily_diagnostics.csv"
        result.baseline_daily_diagnostics.to_csv(artifact_paths["baseline_daily_diagnostics"], index=False)
    if not result.daily_asset_returns.empty and _requires_daily_asset_returns_artifact(result):
        artifact_paths["daily_asset_returns"] = output_path / "daily_asset_returns.csv"
        result.daily_asset_returns.to_csv(artifact_paths["daily_asset_returns"], index=False)
    return BacktestResult(
        daily_returns=result.daily_returns,
        daily_weights=result.daily_weights,
        daily_turnover=result.daily_turnover,
        daily_rebalance=result.daily_rebalance,
        daily_costs=result.daily_costs,
        metrics=result.metrics,
        run_manifest=result.run_manifest,
        portfolio_state=result.portfolio_state,
        baseline_daily_diagnostics=result.baseline_daily_diagnostics,
        daily_asset_returns=result.daily_asset_returns,
        artifact_paths=artifact_paths,
    )


def _requires_daily_asset_returns_artifact(result: BacktestResult) -> bool:
    diagnostics = result.baseline_daily_diagnostics
    if diagnostics.empty:
        return False
    return any(
        column in diagnostics.columns
        for column in (
            "pred_candidate_utility",
            "pred_hold_utility",
            "pred_candidate_lower_tail_loss",
            "pred_hold_lower_tail_loss",
        )
    )


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)


__all__ = ["BacktestEngine", "BacktestResult", "PendingActionQueue"]
