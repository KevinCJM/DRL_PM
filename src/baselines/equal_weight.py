from __future__ import annotations

import numpy as np

from src.baselines.base_strategy import TraditionalStrategyBase
from src.envs.state import DecisionMarketState


class EqualWeightStrategy(TraditionalStrategyBase):
    strategy_name = "equal_weight"

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
        weights = np.zeros(available.shape, dtype=float)
        if available.any():
            weights[available] = 1.0 / int(available.sum())
        return weights


__all__ = ["EqualWeightStrategy"]
