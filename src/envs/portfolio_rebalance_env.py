from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.config import DEFAULT_CONFIG
from src.data.leakage_checks import assert_decision_visibility_contract
from src.data.loader import DataContractError, MarketDatasetBundle
from src.envs.backtest_engine import (
    PENDING_TRUNCATION_REASON,
    PendingActionQueue,
    _asset_order,
    _build_decision_market_state,
    _date_index,
    _deep_update,
    _decision_dates,
    _execution_activity_config,
    _finalize_execution_action,
    _pending_action_for,
    _segment_dates,
    _zero_cost_model,
)
from src.envs.portfolio_execution_core import PortfolioExecutionCore
from src.envs.rebalance_scheduler import RebalanceScheduler
from src.envs.reward_calculator import RewardCalculator
from src.envs.state import DecisionMarketState, ExecutionMarketState, ExecutionResult, PortfolioAction, PortfolioState


RELATED_WORK_ACTION_INFO_KEYS = (
    "paper_model_id",
    "model_extension_id",
    "post_hoc_development_disclosure",
    "test_used_for_model_selection",
    "hierarchy_action",
    "hierarchy_action_name",
    "gate_action",
    "gate_action_index",
    "rho",
    "rho_logits",
    "rho_probs",
    "rho_entropy",
    "rho_expected",
    "rho_action_index",
    "rho_policy_mode",
    "rho_temperature",
    "raw_rho",
    "raw_rebalance_intensity",
    "raw_model_requested_rebalance",
    "raw_gate_requested_rebalance",
    "raw_gate_action_index",
    "raw_action",
    "final_rho",
    "final_rebalance_intensity",
    "final_action",
    "execution_activity_protocol",
    "activity_protocol",
    "scheduler_blocks_model_actions",
    "activity_gate_enforced",
    "turnover_optimization_protocol_id",
    "execution_gate_allowed",
    "scheduler_pre_allowed",
    "scheduler_post_allowed",
    "scheduler_final_allowed",
    "scheduler_pre_blocked",
    "scheduler_post_blocked",
    "scheduler_final_blocked",
    "scheduler_blocked_rebalance",
    "execution_scheduler_blocked",
    "model_chosen_hold",
    "trade_opportunity",
    "non_initial_trade_opportunity",
    "rebalance_values",
    "scheduler_allowed_rebalance",
    "forced_hold_reason",
    "execution_weight_mode",
    "candidate_weights_json",
    "candidate_turnover",
    "candidate_turnover_estimate",
    "candidate_cost_estimate",
    "CVaR_loss_5",
    "drawdown",
    "ppo_actor_update_mask",
    "ppo_attribution_weight",
    "platform_adapted_surrogate",
    "child_model_name",
    "baseline_family",
    "optimizer_name",
    "include_count",
    "exclude_count",
    "neutral_count",
    "selected_asset_count",
    "optimizer_asset_count",
    "optimizer_status",
    "fallback_reason",
    "factorized_q",
    "portfolio_level_reward_shared",
    "counterfactual_asset_reward",
    "platform_adapted_approximation",
)


class PortfolioRebalanceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dataset: MarketDatasetBundle,
        split: Any,
        *,
        config: Mapping[str, Any] | None = None,
        segment: str = "train",
        execution_core: PortfolioExecutionCore | None = None,
        scheduler: RebalanceScheduler | None = None,
        reward_calculator: RewardCalculator | None = None,
        market_image_dataset: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.segment = segment
        self.config = deepcopy(DEFAULT_CONFIG)
        if config is not None:
            _deep_update(self.config, config)
        self.execution_config = self.config["execution_model"]
        self.execution_activity_config = _execution_activity_config(self.config)
        self.data_governance_config = self.config.get("data_governance", {})
        self.portfolio_config = self.config["portfolio"]
        self.observation_dtype = np.dtype(self.config.get("env", {}).get("observation_dtype", "float32"))
        self.window_size = int(
            self.config.get("env", {}).get("window_size", self.config.get("feature_matrix", {}).get("window_size", 60))
        )
        self.execution_core = execution_core or PortfolioExecutionCore(self.config)
        self.scheduler = scheduler
        self.reward_calculator = reward_calculator or RewardCalculator(self.config)
        self.market_image_dataset = market_image_dataset
        self.market_image_feature_cols = _feature_names(market_image_dataset, fallback=dataset.feature_cols)
        assert_decision_visibility_contract(
            market_image=self.market_image_feature_cols,
            feature_window=dataset.feature_cols,
        )

        self.date_index = _date_index(dataset)
        self.asset_ids = _asset_order(dataset)
        self.decision_dates = _decision_dates(split, segment, self.date_index)
        self.decision_dates = _align_to_market_image_dates(self.decision_dates, market_image_dataset)
        self.decision_dates = _filter_available_decision_dates(self.decision_dates, dataset.availability_mask)
        self.decision_dates = _filter_executable_decision_dates(
            self.decision_dates,
            dataset,
            self.date_index,
            self.execution_config,
            self.data_governance_config,
        )
        self.segment_dates = _segment_dates(split, segment)
        self.segment_dates = pd.DatetimeIndex(self.segment_dates[self.segment_dates.isin(self.date_index)])
        if self.segment_dates.empty:
            raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: {segment}")
        if self.decision_dates.empty:
            raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: {segment}_decision_dates")

        initial_state = _build_decision_market_state(dataset, self.decision_dates[0], self.config)
        market_image_shape = self._market_image(initial_state, self.decision_dates[0]).shape
        n_assets = len(self.asset_ids)
        self.observation_space = spaces.Dict(
            {
                "market_image": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=market_image_shape,
                    dtype=self.observation_dtype,
                ),
                "current_weights": spaces.Box(low=0.0, high=1.0, shape=(n_assets,), dtype=self.observation_dtype),
                "availability_mask": spaces.MultiBinary(n_assets),
                "adv20_at_decision": spaces.Box(low=0.0, high=np.inf, shape=(n_assets,), dtype=self.observation_dtype),
                "volatility_20d_at_decision": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(n_assets,),
                    dtype=self.observation_dtype,
                ),
                "amount_at_decision": spaces.Box(low=0.0, high=np.inf, shape=(n_assets,), dtype=self.observation_dtype),
                "turnover_rate_at_decision": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(n_assets,),
                    dtype=self.observation_dtype,
                ),
                "portfolio_value": spaces.Box(low=0.0, high=np.inf, shape=(), dtype=self.observation_dtype),
            }
        )
        self.action_space = spaces.Dict(
            {
                "weights": spaces.Box(low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32),
                "rebalance": spaces.Discrete(2),
                "rebalance_intensity": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            }
        )

        self.portfolio_state: PortfolioState | None = None
        self.pending_queue = PendingActionQueue()
        self._scheduler_runtime: RebalanceScheduler | None = None
        self._step_pos = 0
        self._done = False
        self._first_trade = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._step_pos = 0
        self._done = False
        self._first_trade = True
        self.pending_queue.clear()
        self._scheduler_runtime = self.scheduler or RebalanceScheduler(self.config, date_index=self.date_index)
        self._scheduler_runtime.reset()
        self.reward_calculator.reset_episode()
        self.portfolio_state = _initial_portfolio_state(
            self.asset_ids,
            self.decision_dates[0],
            self.portfolio_config,
        )
        obs = self._observation(self.decision_dates[0])
        info = self._reset_info(self.decision_dates[0], options)
        return obs, info

    def step(
        self,
        action: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.portfolio_state is None:
            raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: reset required")
        if self._done or self._step_pos >= len(self.decision_dates):
            self._done = True
            return self._observation(self.portfolio_state.date), 0.0, False, True, {"truncated_reason": "split_exhausted"}

        decision_date = pd.Timestamp(self.decision_dates[self._step_pos])
        decision_state = _build_decision_market_state(self.dataset, decision_date, self.config)
        if bool(self.execution_config.get("delayed_action_execution", False)):
            return self._step_delayed(decision_date, action, decision_state)

        portfolio_action = self._portfolio_action(action, decision_state)
        portfolio_action = self._finalized_portfolio_action(decision_date, decision_state, portfolio_action)
        return self._step_immediate(decision_date, portfolio_action, portfolio_action.rebalance_action)

    def _step_immediate(
        self,
        decision_date: pd.Timestamp,
        action: PortfolioAction,
        final_action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        execution_state = self.execution_core.build_execution_market_state(self.dataset, decision_date)
        execution_result = self._execute_step(action, execution_state, final_action)
        reward, reward_info = self.reward_calculator.calculate(
            execution_result,
            self._state,
            omega=self._preference_omega(),
            reward_context=self._reward_context(),
        )
        self._first_trade = False
        self._advance_position()
        truncated = self._step_pos >= len(self.decision_dates)
        self._done = truncated
        obs = self._observation(self._state.date)
        info = self._step_info(action, final_action, execution_state, execution_result, reward_info)
        return obs, float(reward), False, bool(truncated), info

    def _step_delayed(
        self,
        decision_date: pd.Timestamp,
        raw_action: Mapping[str, Any],
        decision_state: DecisionMarketState,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        ready_actions = self.pending_queue.pop_ready(decision_date)
        execution_payload: tuple[PortfolioAction, PendingAction, ExecutionMarketState, ExecutionResult, float, dict[str, Any]] | None = None
        if ready_actions:
            ready_action = ready_actions[0]
            delayed_action = PortfolioAction(
                ready_action.target_weights.copy(),
                ready_action.rebalance_action,
                ready_action.rebalance_intensity,
                dict(ready_action.action_info),
            )
            execution_state = self.execution_core.build_execution_market_state(
                self.dataset,
                pending_action=ready_action,
            )
            execution_result = self._execute_step(delayed_action, execution_state, ready_action.rebalance_action)
            reward, reward_info = self.reward_calculator.calculate(
                execution_result,
                self._state,
                omega=self._preference_omega(),
                reward_context=self._reward_context(),
            )
            self._first_trade = False
            execution_payload = (delayed_action, ready_action, execution_state, execution_result, float(reward), reward_info)

        action = self._portfolio_action(raw_action, decision_state)
        action = self._finalized_portfolio_action(decision_date, decision_state, action)
        final_action = action.rebalance_action
        pending_action = _pending_action_for(
            action,
            decision_date,
            final_action,
            self.segment_dates,
            self.execution_config,
        )
        if pending_action is None:
            if execution_payload is not None:
                self._advance_position()
                truncated = self._step_pos >= len(self.decision_dates)
                self._done = truncated
                delayed_action, ready_action, execution_state, execution_result, reward, reward_info = execution_payload
                info = self._step_info(
                    delayed_action,
                    ready_action.rebalance_action,
                    execution_state,
                    execution_result,
                    reward_info,
                )
                info["pending_action_truncation_count"] = 1
                info["pending_action_truncation_reason"] = PENDING_TRUNCATION_REASON
                return self._observation(self._state.date), reward, False, bool(truncated), info
            self._done = True
            info = {
                "decision_date": decision_date,
                "truncated_reason": PENDING_TRUNCATION_REASON,
                "pending_action_truncation_count": 1,
            }
            return self._observation(self._state.date), 0.0, False, True, info

        self.pending_queue.append(pending_action)
        self._advance_position()
        truncated = self._step_pos >= len(self.decision_dates)
        self._done = truncated
        obs = self._observation(self._state.date)
        if execution_payload is None:
            info = {
                "decision_date": decision_date,
                "delayed_action_queued": True,
                "delayed_action_execution": True,
                "execution_price_type": pending_action.execution_price_type,
                "pending_execution_date": pending_action.execution_date,
                "pending_next_valuation_date": pending_action.next_valuation_date,
                "pending_queue_size": len(self.pending_queue),
            }
            return obs, 0.0, False, bool(truncated), info

        delayed_action, ready_action, execution_state, execution_result, reward, reward_info = execution_payload
        info = self._step_info(delayed_action, ready_action.rebalance_action, execution_state, execution_result, reward_info)
        info["delayed_action_queued"] = True
        info["pending_execution_date"] = pending_action.execution_date
        info["pending_next_valuation_date"] = pending_action.next_valuation_date
        info["pending_queue_size"] = len(self.pending_queue)
        return obs, reward, False, bool(truncated), info

    def _portfolio_action(self, action: Mapping[str, Any], decision_state: DecisionMarketState) -> PortfolioAction:
        if not isinstance(action, Mapping):
            raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: action")
        weights = _action_weights(action.get("weights"), len(self.asset_ids))
        rebalance_action = _action_rebalance(action.get("rebalance", action.get("rebalance_action", 1)))
        if "rebalance_intensity" in action and action["rebalance_intensity"] is not None:
            rebalance_intensity = _action_intensity(action["rebalance_intensity"])
        else:
            rebalance_intensity = 1.0 if rebalance_action == 1 else 0.0
        target_weights = _mask_decision_weights(
            weights,
            decision_state.available_mask_at_decision,
            self._state.current_weights,
        )
        return PortfolioAction(target_weights, rebalance_action, rebalance_intensity, _action_info(action))

    def _finalized_portfolio_action(
        self,
        decision_date: pd.Timestamp,
        decision_state: DecisionMarketState,
        action: PortfolioAction,
    ) -> PortfolioAction:
        scheduler = self._scheduler
        return _finalize_execution_action(
            scheduler,
            decision_date,
            self._state,
            decision_state,
            action,
            self._first_trade,
            self.execution_activity_config,
        )

    def _execute_step(
        self,
        action: PortfolioAction,
        execution_state: ExecutionMarketState,
        final_action: int,
    ) -> ExecutionResult:
        if self._first_trade and not bool(self.execution_config.get("initial_build_cost", True)):
            original_cost_model = self.execution_core.cost_model
            self.execution_core.cost_model = _zero_cost_model(self.config)
            try:
                result = self.execution_core.execute_step(
                    self._state.current_weights,
                    action.target_weights,
                    execution_state,
                    self._state,
                    rebalance_action=final_action,
                    rebalance_intensity=action.rebalance_intensity,
                    asset_ids=list(self.asset_ids),
                    estimated_turnover=action.action_info.get("estimated_turnover"),
                    estimated_cost=action.action_info.get("estimated_cost"),
                )
            finally:
                self.execution_core.cost_model = original_cost_model
            result.info["initial_build_cost"] = False
            return result

        result = self.execution_core.execute_step(
            self._state.current_weights,
            action.target_weights,
            execution_state,
            self._state,
            rebalance_action=final_action,
            rebalance_intensity=action.rebalance_intensity,
            asset_ids=list(self.asset_ids),
            estimated_turnover=action.action_info.get("estimated_turnover"),
            estimated_cost=action.action_info.get("estimated_cost"),
        )
        result.info["initial_build_cost"] = bool(self.execution_config.get("initial_build_cost", True))
        return result

    def _advance_position(self) -> None:
        next_pos = self._step_pos + 1
        while next_pos < len(self.decision_dates) and pd.Timestamp(self.decision_dates[next_pos]) < self._state.date:
            next_pos += 1
        self._step_pos = next_pos

    def _observation(self, date: Any) -> dict[str, np.ndarray]:
        observation_date = _observation_date(date, self.date_index)
        decision_state = _build_decision_market_state(self.dataset, observation_date, self.config)
        observation = {
            "market_image": self._market_image(decision_state, observation_date),
            "current_weights": np.asarray(self._state.current_weights, dtype=self.observation_dtype),
            "availability_mask": np.asarray(decision_state.available_mask_at_decision, dtype=np.int8),
            "adv20_at_decision": _finite_observation_array(decision_state.adv20_at_decision, self.observation_dtype),
            "volatility_20d_at_decision": _finite_observation_array(
                decision_state.volatility_20d_at_decision,
                self.observation_dtype,
            ),
            "amount_at_decision": _finite_observation_array(decision_state.amount_at_decision, self.observation_dtype),
            "turnover_rate_at_decision": _finite_observation_array(
                decision_state.turnover_rate_at_decision,
                self.observation_dtype,
            ),
            "portfolio_value": np.asarray(self._state.portfolio_value, dtype=self.observation_dtype),
        }
        omega = self._preference_omega()
        if omega is not None:
            observation["preference_omega"] = np.asarray(omega, dtype=self.observation_dtype)
        assert_decision_visibility_contract(
            observation=observation.keys(),
            market_image=self.market_image_feature_cols,
            feature_window=self.dataset.feature_cols,
        )
        return observation

    def _market_image(self, decision_state: DecisionMarketState, date: pd.Timestamp) -> np.ndarray:
        if self.market_image_dataset is None:
            image = decision_state.market_image
        else:
            image = _market_image_from_dataset(self.market_image_dataset, date)
            if image is None:
                image = decision_state.market_image
        return _pad_window(np.asarray(image, dtype=self.observation_dtype), self.window_size)

    def _step_info(
        self,
        action: PortfolioAction,
        final_action: int,
        execution_state: ExecutionMarketState,
        execution_result: ExecutionResult,
        reward_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        action_info = dict(action.action_info)
        info = {
            "decision_date": execution_state.decision_date,
            "execution_date": execution_state.execution_date,
            "execution_price_type": execution_state.execution_price_type,
            "next_valuation_date": execution_state.next_valuation_date,
            "delayed_action_execution": bool(self.execution_config.get("delayed_action_execution", False)),
            "raw_return": float(reward_info.get("raw_return", execution_result.gross_return)),
            "net_return": execution_result.net_return,
            "portfolio_log_return": execution_result.portfolio_log_return,
            "proportional_cost": execution_result.proportional_cost,
            "fixed_cost": execution_result.fixed_cost,
            "slippage_cost": execution_result.slippage_cost,
            "market_impact_cost": execution_result.market_impact_cost,
            "total_transaction_cost": execution_result.total_transaction_cost,
            "transaction_cost": execution_result.transaction_cost,
            "estimated_turnover": execution_result.estimated_turnover,
            "realized_turnover": execution_result.realized_turnover,
            "estimated_cost": execution_result.estimated_cost,
            "realized_cost": execution_result.realized_cost,
            "turnover": execution_result.turnover,
            "pre_execution_return": execution_result.pre_execution_return,
            "post_execution_return": execution_result.post_execution_return,
            "rolling_volatility": float(reward_info.get("rolling_volatility", 0.0)),
            "downside_deviation": float(np.sqrt(max(float(reward_info.get("downside_penalty", 0.0)), 0.0))),
            "current_drawdown": self._state.current_drawdown_abs,
            "max_drawdown": self._state.max_drawdown_abs,
            "rolling_CVaR": float(reward_info.get("cvar_loss", 0.0)),
            "HHI": float(reward_info.get("hhi", np.sum(np.square(execution_result.executed_weights)))),
            "concentration": float(reward_info.get("concentration_penalty", 0.0)),
            "rebalance_action": int(final_action),
            "rebalance_intensity": float(action.rebalance_intensity),
            "active_weight_change_l1": float(
                np.sum(np.abs(execution_result.executed_weights - execution_result.pre_execution_drifted_weights))
            ),
            "candidate_weights": action.target_weights.copy(),
            "executed_weights": execution_result.executed_weights.copy(),
            "constraint_violations": list(execution_result.info.get("constraint_violations", [])),
            "q_hold": action_info.get("q_hold"),
            "q_rebalance": action_info.get("q_rebalance"),
            "q_gap": action_info.get("q_gap"),
            "reward_vector": reward_info.get("reward_vector"),
            "omega": reward_info.get("omega"),
            "preference_vector": reward_info.get("omega"),
        }
        for key in RELATED_WORK_ACTION_INFO_KEYS:
            if key in action_info and action_info[key] is not None:
                info[key] = action_info[key]
        return info

    def _reset_info(self, decision_date: pd.Timestamp, options: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "decision_date": pd.Timestamp(decision_date),
            "date": pd.Timestamp(decision_date),
            "nav": self._state.nav,
            "portfolio_value": self._state.portfolio_value,
            "options": {} if options is None else dict(options),
        }

    @property
    def _state(self) -> PortfolioState:
        if self.portfolio_state is None:
            raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: reset required")
        return self.portfolio_state

    def _reward_context(self) -> dict[str, Any]:
        reward_config = self.config.get("reward")
        preference_config = self.config.get("preference")
        reward_mode = str(reward_config.get("mode", "")) if isinstance(reward_config, Mapping) else ""
        preference_enabled = bool(isinstance(preference_config, Mapping) and preference_config.get("enabled") is True)
        return {"preference_conditioned": preference_enabled or reward_mode == "A12_multi_objective_preference_conditioned"}

    def _preference_omega(self) -> np.ndarray | None:
        preference_config = self.config.get("preference")
        if not isinstance(preference_config, Mapping):
            return None
        reward_config = self.config.get("reward")
        reward_mode = str(reward_config.get("mode", "")) if isinstance(reward_config, Mapping) else ""
        if preference_config.get("enabled") is not True and reward_mode != "A12_multi_objective_preference_conditioned":
            return None
        omega = preference_config.get("omega")
        if omega is None:
            return None
        return np.asarray(omega, dtype=float)

    @property
    def _scheduler(self) -> RebalanceScheduler:
        if self._scheduler_runtime is None:
            self._scheduler_runtime = self.scheduler or RebalanceScheduler(self.config, date_index=self.date_index)
        return self._scheduler_runtime


def _initial_portfolio_state(
    asset_ids: Sequence[str],
    date: pd.Timestamp,
    portfolio_config: Mapping[str, Any],
) -> PortfolioState:
    n_assets = len(asset_ids)
    zeros = np.zeros(n_assets, dtype=float)
    initial_nav = float(portfolio_config.get("initial_nav", 1.0))
    initial_capital = float(portfolio_config.get("initial_capital_currency", 0.0))
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


def _action_weights(values: Any, n_assets: int) -> np.ndarray:
    try:
        weights = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: weights") from exc
    if weights.ndim != 1 or weights.shape != (n_assets,):
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: weights")
    if not np.isfinite(weights).all():
        raise DataContractError("ERR_ACTION_NON_FINITE", "ERR_ACTION_NON_FINITE: weights")
    return weights


def _action_rebalance(value: Any) -> int:
    try:
        rebalance = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: rebalance") from exc
    if rebalance != numeric or rebalance not in {0, 1}:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: rebalance")
    return rebalance


def _action_intensity(value: Any) -> float:
    try:
        intensity = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: rebalance_intensity") from exc
    if not np.isfinite(intensity):
        raise DataContractError("ERR_ACTION_NON_FINITE", "ERR_ACTION_NON_FINITE: rebalance_intensity")
    if intensity < 0.0 or intensity > 1.0:
        raise DataContractError("ERR_ACTION_SHAPE_MISMATCH", "ERR_ACTION_SHAPE_MISMATCH: rebalance_intensity")
    return intensity


def _action_info(action: Mapping[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for key in (
        "gate_action",
        "q_hold",
        "q_rebalance",
        "q_gap",
        "estimated_turnover",
        "estimated_cost",
        "log_prob",
        "decision_value",
        *RELATED_WORK_ACTION_INFO_KEYS,
    ):
        if key in action and action[key] is not None:
            info[key] = action[key]
    return info


def _finite_observation_array(values: Any, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array.astype(dtype, copy=False)


def _mask_decision_weights(
    weights: np.ndarray,
    available_mask: np.ndarray,
    current_weights: np.ndarray,
) -> np.ndarray:
    available = np.asarray(available_mask, dtype=bool)
    masked = np.asarray(weights, dtype=float).copy()
    masked[~available] = 0.0
    positive_sum = float(np.sum(np.clip(masked, 0.0, None)))
    if positive_sum > 0.0:
        masked = np.clip(masked, 0.0, None) / positive_sum
        return masked
    if float(np.sum(current_weights)) > 0.0:
        return np.asarray(current_weights, dtype=float).copy()
    if available.any():
        result = np.zeros_like(masked, dtype=float)
        result[available] = 1.0 / int(available.sum())
        return result
    raise DataContractError("ERR_CONSTRAINT_NO_AVAILABLE_ASSET", "ERR_CONSTRAINT_NO_AVAILABLE_ASSET: action")


def _pad_window(image: np.ndarray, window_size: int) -> np.ndarray:
    if image.ndim < 2:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: market_image")
    current = image.shape[-2]
    if current == window_size:
        return image
    if current > window_size:
        slicer = [slice(None)] * image.ndim
        slicer[-2] = slice(current - window_size, current)
        return image[tuple(slicer)]
    pad_width = [(0, 0)] * image.ndim
    pad_width[-2] = (window_size - current, 0)
    return np.pad(image, pad_width, mode="constant", constant_values=0.0)


def _observation_date(date: Any, date_index: pd.DatetimeIndex) -> pd.Timestamp:
    timestamp = pd.Timestamp(date)
    if timestamp in date_index:
        return timestamp
    prior_dates = date_index[date_index <= timestamp]
    if len(prior_dates) == 0:
        return pd.Timestamp(date_index[0])
    return pd.Timestamp(prior_dates[-1])


def _align_to_market_image_dates(decision_dates: pd.DatetimeIndex, market_image_dataset: Any | None) -> pd.DatetimeIndex:
    image_dates = getattr(market_image_dataset, "date_index", None)
    if image_dates is None:
        return pd.DatetimeIndex(decision_dates)
    valid_dates = pd.DatetimeIndex(pd.to_datetime(list(image_dates)))
    return pd.DatetimeIndex(decision_dates[decision_dates.isin(valid_dates)])


def _filter_available_decision_dates(
    decision_dates: pd.DatetimeIndex,
    availability_mask: pd.DataFrame,
) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(decision_dates[decision_dates.isin(availability_mask.index)])
    if dates.empty:
        return dates
    available = availability_mask.loc[dates].any(axis=1)
    return pd.DatetimeIndex(dates[available.to_numpy(dtype=bool)])


def _filter_executable_decision_dates(
    decision_dates: pd.DatetimeIndex,
    dataset: MarketDatasetBundle,
    date_index: pd.DatetimeIndex,
    execution_config: Mapping[str, Any],
    data_governance_config: Mapping[str, Any],
) -> pd.DatetimeIndex:
    availability_mask = dataset.availability_mask
    same_close = bool(
        execution_config.get("same_close_idealized_execution_enabled", False)
        or data_governance_config.get("same_close_idealized_execution_enabled", False)
    )
    execution_price = str(execution_config.get("execution_price", "next_open"))
    keep = []
    positions = {pd.Timestamp(date): index for index, date in enumerate(date_index)}
    for decision_date in decision_dates:
        position = positions.get(pd.Timestamp(decision_date))
        if position is None:
            continue
        if same_close:
            execution_date = pd.Timestamp(decision_date)
            valuation_date = pd.Timestamp(date_index[position + 1]) if position + 1 < len(date_index) else None
            execution_field = "close"
        elif execution_price == "next_close":
            execution_date = pd.Timestamp(date_index[position + 1]) if position + 1 < len(date_index) else None
            valuation_date = pd.Timestamp(date_index[position + 2]) if position + 2 < len(date_index) else None
            execution_field = "close"
        else:
            execution_date = pd.Timestamp(date_index[position + 1]) if position + 1 < len(date_index) else None
            valuation_date = execution_date
            execution_field = "open"
        if execution_date is None:
            continue
        if valuation_date is None and bool(execution_config.get("delayed_action_execution", False)):
            valuation_date = execution_date
        if valuation_date is None:
            continue
        if _has_executable_asset(dataset, pd.Timestamp(decision_date), execution_date, valuation_date, execution_field):
            keep.append(pd.Timestamp(decision_date))
    return pd.DatetimeIndex(keep)


def _has_executable_asset(
    dataset: MarketDatasetBundle,
    decision_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    valuation_date: pd.Timestamp,
    execution_field: str,
) -> bool:
    availability_mask = dataset.availability_mask
    if decision_date not in availability_mask.index or execution_date not in availability_mask.index:
        return False
    decision_available = availability_mask.loc[decision_date].to_numpy(dtype=bool, copy=True)
    execution_available = availability_mask.loc[execution_date].to_numpy(dtype=bool, copy=True)
    decision_close = _positive_finite_wide_row(dataset, "close", decision_date)
    execution_price = _positive_finite_wide_row(dataset, execution_field, execution_date)
    valuation_close = _positive_finite_wide_row(dataset, "close", valuation_date)
    executable = decision_available & execution_available & decision_close & execution_price & valuation_close
    return bool(executable.any())


def _positive_finite_wide_row(dataset: MarketDatasetBundle, field: str, date: pd.Timestamp) -> np.ndarray:
    table = dataset.wide.get(field)
    if table is None or date not in table.index:
        return np.zeros(len(_asset_order(dataset)), dtype=bool)
    values = table.loc[date].to_numpy(dtype=float, copy=True)
    return np.isfinite(values) & (values > 0.0)


def _market_image_from_dataset(market_image_dataset: Any, date: pd.Timestamp) -> np.ndarray | None:
    try:
        return np.asarray(market_image_dataset[date])
    except Exception:
        return None


def _feature_names(source: Any, *, fallback: Sequence[str] | None = None) -> list[str]:
    if source is None:
        return [str(name) for name in (fallback or [])]
    for attr in ("feature_cols", "feature_names", "columns"):
        values = getattr(source, attr, None)
        if values is not None:
            return [str(name) for name in list(values)]
    return [str(name) for name in (fallback or [])]


__all__ = ["PortfolioRebalanceEnv"]
