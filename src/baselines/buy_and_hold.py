from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.base_strategy import TraditionalStrategyBase
from src.baselines.fixed_ratio import FixedRatioStrategy
from src.data.loader import DataContractError
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


class BuyAndHoldStrategy(TraditionalStrategyBase):
    strategy_name = "buy_and_hold"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.buy_and_hold_config = _buy_and_hold_config(self.config)
        self._has_built_position = False
        self._last_reset_date: pd.Timestamp | None = None
        self._step_count = 0

    def reset(self) -> None:
        self._has_built_position = False
        self._last_reset_date = None
        self._step_count = 0

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        portfolio = self.validate_portfolio_state(portfolio_state)
        self._step_count += 1
        should_rebalance, reason = self._should_rebalance(state, portfolio)
        if not should_rebalance:
            return self.validate_portfolio_action(
                PortfolioAction(
                    target_weights=portfolio.current_weights.copy(),
                    rebalance_action=0,
                    rebalance_intensity=0.0,
                    action_info={
                        "strategy": self.strategy_name,
                        "rebalance_intensity": 0.0,
                        "rebalance_reason": "hold",
                    },
                )
            )

        raw_weights = self._initial_weights(state, portfolio)
        constraint_result = self.constraint_manager.project(
            raw_weights,
            state.available_mask_at_decision,
            reference_weights=portfolio.current_weights,
        )
        self._has_built_position = True
        self._last_reset_date = pd.Timestamp(state.decision_date)
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=constraint_result.projected_weights,
                rebalance_action=1,
                rebalance_intensity=1.0,
                action_info={
                    "strategy": self.strategy_name,
                    "rebalance_intensity": 1.0,
                    "rebalance_reason": reason,
                    "initial_weight_mode": str(self.buy_and_hold_config.get("initial_weight_mode", "equal_weight")),
                    "constraint_violations": constraint_result.constraint_violations,
                },
            )
        )

    def _should_rebalance(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> tuple[bool, str]:
        if not self._has_built_position:
            return True, "initial_build"
        if _forced_rebalance_on_unavailable(self.buy_and_hold_config):
            unavailable_held = (
                np.asarray(portfolio_state.current_weights, dtype=float) > 0.0
            ) & ~np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
            if unavailable_held.any():
                return True, "asset_unavailable"
        if _reset_due(
            self.buy_and_hold_config,
            decision_market_state.decision_date,
            self._last_reset_date,
            self._step_count,
        ):
            return True, "reset_frequency"
        return False, "hold"

    def _initial_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> np.ndarray:
        mode = str(self.buy_and_hold_config.get("initial_weight_mode", "equal_weight"))
        if mode == "equal_weight":
            available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
            weights = np.zeros(available.shape, dtype=float)
            if available.any():
                weights[available] = 1.0 / int(available.sum())
            return weights
        if mode == "fixed_ratio":
            config = dict(self.config)
            fixed_ratio_config = self.buy_and_hold_config.get("fixed_ratio")
            if isinstance(fixed_ratio_config, Mapping):
                config["fixed_ratio"] = dict(fixed_ratio_config)
            action = FixedRatioStrategy(config).compute_target_weights(decision_market_state, portfolio_state)
            return action.target_weights
        raise DataContractError(
            "ERR_STRATEGY_CONFIG_INVALID",
            "ERR_STRATEGY_CONFIG_INVALID: buy_and_hold.initial_weight_mode",
        )


def _buy_and_hold_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("buy_and_hold"), Mapping):
        return dict(config["buy_and_hold"])
    return dict(config)


def _forced_rebalance_on_unavailable(config: Mapping[str, Any]) -> bool:
    forced = config.get("forced_rebalance", {})
    if isinstance(forced, Mapping):
        return bool(forced.get("on_asset_unavailable", True))
    return bool(config.get("forced_rebalance_on_asset_unavailable", False))


def _reset_due(
    config: Mapping[str, Any],
    decision_date: pd.Timestamp,
    last_reset_date: pd.Timestamp | None,
    step_count: int,
) -> bool:
    frequency = config.get("reset_frequency")
    forced = config.get("forced_rebalance", {})
    if frequency is None and isinstance(forced, Mapping):
        frequency = forced.get("reset_frequency")
    if frequency is None or last_reset_date is None:
        return False
    if isinstance(frequency, int):
        return frequency > 0 and (step_count - 1) % frequency == 0

    value = str(frequency)
    current = pd.Timestamp(decision_date)
    previous = pd.Timestamp(last_reset_date)
    if value == "daily":
        return True
    if value == "weekly":
        return current.isocalendar().week != previous.isocalendar().week or current.year != previous.year
    if value == "monthly":
        return current.to_period("M") != previous.to_period("M")
    if value == "quarterly":
        return current.to_period("Q") != previous.to_period("Q")
    if value == "yearly":
        return current.year != previous.year
    return False


__all__ = ["BuyAndHoldStrategy"]
