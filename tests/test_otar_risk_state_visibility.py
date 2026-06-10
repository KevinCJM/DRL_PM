"""Tests for M2: Risk state observation visibility.

Verifies:
- Observation contains all required risk state fields
- Risk state is computed after close t-1
- execution_result is not visible at decision time
"""

from __future__ import annotations

import numpy as np
import pytest

from src.envs.risk_state_manager import RiskStateManager
from src.envs.state import ExecutionResult, PortfolioState


def _make_execution_result(
    net_return: float = 0.01,
    turnover: float = 0.1,
    transaction_cost: float = 0.001,
    gross_return: float = 0.011,
    portfolio_log_return: float = 0.0109,
    nav_next: float = 1.01,
    nav_after_cost: float = 1.009,
    post_execution_return: float = 0.001,
) -> ExecutionResult:
    return ExecutionResult(
        executed_weights=np.array([0.5, 0.5]),
        pre_execution_drifted_weights=np.array([0.5, 0.5]),
        turnover=turnover,
        transaction_cost=transaction_cost,
        transaction_cost_on_initial_nav=transaction_cost,
        proportional_cost=transaction_cost * 0.5,
        fixed_cost=0.0,
        slippage_cost=0.0,
        market_impact_cost=0.0,
        total_transaction_cost=transaction_cost,
        estimated_turnover=turnover,
        realized_turnover=turnover,
        estimated_cost=transaction_cost,
        realized_cost=transaction_cost,
        gross_return=gross_return,
        net_return=net_return,
        pre_execution_return=0.0,
        post_execution_return=post_execution_return,
        portfolio_log_return=portfolio_log_return,
        nav_execution=1.0,
        nav_after_cost=nav_after_cost,
        nav_next=nav_next,
        info={},
    )


def _make_portfolio_state(drawdown: float = 0.0) -> PortfolioState:
    return PortfolioState(
        date="2024-01-01",
        nav=1.0,
        portfolio_value=100000.0,
        current_weights=np.array([0.5, 0.5]),
        current_drawdown_abs=drawdown,
    )


# ---------------------------------------------------------------------------
# RiskStateManager shape and content
# ---------------------------------------------------------------------------

class TestRiskStateManagerObservation:
    """RiskStateManager.get_observation_vector() must return shape (8,)."""

    def test_observation_vector_shape(self) -> None:
        manager = RiskStateManager({})
        manager.reset()
        vec = manager.get_observation_vector()
        assert vec.shape == (8,), f"Expected shape (8,), got {vec.shape}"

    def test_observation_vector_dtype(self) -> None:
        manager = RiskStateManager({})
        manager.reset()
        vec = manager.get_observation_vector()
        assert vec.dtype == np.float64, f"Expected float64, got {vec.dtype}"

    def test_observation_vector_fields_after_reset(self) -> None:
        manager = RiskStateManager({})
        manager.reset()
        vec = manager.get_observation_vector()
        # After reset, all fields should be 0.0
        np.testing.assert_array_equal(vec, np.zeros(8))

    def test_observation_vector_updates_after_step(self) -> None:
        manager = RiskStateManager({})
        manager.reset()
        exec_result = _make_execution_result(net_return=-0.02)
        portfolio_state = _make_portfolio_state(drawdown=0.05)
        manager.update_pre_reward(exec_result, portfolio_state, final_action=0)
        manager.update_reward_info({})
        vec = manager.get_observation_vector()
        # After a step with negative return, downside fields should be non-zero
        assert vec[0] > 0.0, "downside_vol_ewma should be > 0 after negative return"
        assert vec[1] < 0.0, "downside_return_ewma should be < 0 after negative return"

    def test_days_since_last_rebalance_resets_on_rebalance(self) -> None:
        manager = RiskStateManager({})
        manager.reset()
        exec_result = _make_execution_result()
        portfolio_state = _make_portfolio_state()
        # Step 1: hold
        manager.update_pre_reward(exec_result, portfolio_state, final_action=0)
        assert manager.get_observation_vector()[7] == 1.0
        # Step 2: rebalance
        manager.update_pre_reward(exec_result, portfolio_state, final_action=1)
        assert manager.get_observation_vector()[7] == 0.0


# ---------------------------------------------------------------------------
# Diagnostics dict
# ---------------------------------------------------------------------------

class TestRiskStateManagerDiagnostics:
    """RiskStateManager.get_diagnostics_dict() must return 8 fields."""

    def test_diagnostics_dict_keys(self) -> None:
        manager = RiskStateManager({})
        manager.reset()
        diag = manager.get_diagnostics_dict()
        expected_keys = {
            "downside_vol_ewma", "downside_return_ewma", "soft_cvar_loss_state",
            "drawdown_abs", "drawdown_increment", "turnover_prev", "cost_prev",
            "days_since_last_rebalance",
        }
        assert set(diag.keys()) == expected_keys

    def test_diagnostics_warmup_nan(self) -> None:
        manager = RiskStateManager({})
        manager.reset()
        diag = manager.get_diagnostics_dict()
        # Before any update, downside and cvar fields should be NaN
        assert np.isnan(diag["downside_vol_ewma"])
        assert np.isnan(diag["downside_return_ewma"])
        assert np.isnan(diag["soft_cvar_loss_state"])
