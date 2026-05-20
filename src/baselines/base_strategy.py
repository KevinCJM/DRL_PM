from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np

from src.data.loader import DataContractError
from src.envs.constraint_manager import ConstraintManager
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


class BaseStrategy(ABC):
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.is_fitted = False

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> BaseStrategy:
        self.is_fitted = True
        return self

    def reset(self) -> None:
        return None

    @abstractmethod
    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self.validate_decision_market_state(decision_market_state)
        self.validate_portfolio_state(portfolio_state)
        raise NotImplementedError

    @staticmethod
    def validate_decision_market_state(decision_market_state: DecisionMarketState) -> DecisionMarketState:
        if not isinstance(decision_market_state, DecisionMarketState):
            raise DataContractError(
                "ERR_STRATEGY_STATE_CONTRACT",
                "ERR_STRATEGY_STATE_CONTRACT: decision_market_state must be DecisionMarketState",
            )
        return decision_market_state

    @staticmethod
    def validate_portfolio_state(portfolio_state: PortfolioState) -> PortfolioState:
        if not isinstance(portfolio_state, PortfolioState):
            raise DataContractError(
                "ERR_STRATEGY_STATE_CONTRACT",
                "ERR_STRATEGY_STATE_CONTRACT: portfolio_state must be PortfolioState",
            )
        return portfolio_state

    @staticmethod
    def validate_portfolio_action(action: PortfolioAction) -> PortfolioAction:
        if not isinstance(action, PortfolioAction):
            raise DataContractError(
                "ERR_STRATEGY_ACTION_CONTRACT",
                "ERR_STRATEGY_ACTION_CONTRACT: compute_target_weights must return PortfolioAction",
            )
        return action


class TraditionalStrategyBase(BaseStrategy):
    strategy_name = "traditional"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.constraint_manager = ConstraintManager(config)

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        portfolio = self.validate_portfolio_state(portfolio_state)
        raw_weights = self._raw_weights(state)
        constraint_result = self.constraint_manager.project(
            raw_weights,
            state.available_mask_at_decision,
            reference_weights=portfolio.current_weights,
        )
        rebalance_intensity = float(self.config.get("rebalance_intensity", 1.0))
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=constraint_result.projected_weights,
                rebalance_action=1,
                rebalance_intensity=rebalance_intensity,
                action_info={
                    "strategy": self.strategy_name,
                    "rebalance_intensity": rebalance_intensity,
                    "constraint_violations": constraint_result.constraint_violations,
                },
            )
        )

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        return _equal_available_weights(decision_market_state.available_mask_at_decision)


def _equal_available_weights(available_mask: Any) -> np.ndarray:
    available = np.asarray(available_mask, dtype=bool)
    weights = np.zeros(available.shape, dtype=float)
    if available.any():
        weights[available] = 1.0 / int(available.sum())
    return weights


__all__ = ["BaseStrategy", "TraditionalStrategyBase"]
