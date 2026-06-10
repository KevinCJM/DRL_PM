"""Tests for M6-T2: Weight drift and cost diagnostics.

Verifies:
- Cost computed from pre_trade_drifted_weights to executed_weights
- passive_weight_drift_l1_t field exists
- turnover_from_active_rebalance_t field exists
"""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.portfolio_execution_core import PortfolioExecutionCore, drift_weights
from src.envs.state import (
    DecisionMarketState,
    ExecutionMarketState,
    PortfolioState,
)


# ---------------------------------------------------------------------------
# drift_weights
# ---------------------------------------------------------------------------

class TestDriftWeights:
    """drift_weights must correctly apply market returns to weights."""

    def test_drift_weights_basic(self) -> None:
        weights = np.array([0.6, 0.4])
        returns = np.array([0.01, -0.02])
        drifted = drift_weights(weights, returns)
        assert np.isfinite(drifted).all()
        assert abs(drifted.sum() - 1.0) < 1e-10

    def test_drift_weights_preserves_zero(self) -> None:
        weights = np.array([1.0, 0.0])
        returns = np.array([0.05, 0.03])
        drifted = drift_weights(weights, returns)
        assert abs(drifted[1]) < 1e-10

    def test_drift_weights_all_zero(self) -> None:
        weights = np.array([0.0, 0.0])
        returns = np.array([0.05, 0.03])
        drifted = drift_weights(weights, returns)
        np.testing.assert_array_equal(drifted, np.zeros(2))
