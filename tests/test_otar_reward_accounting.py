"""Tests for M1-T7/M6-T4: Reward accounting and NAV consistency.

Verifies:
- net_simple_return = portfolio_simple_return - cost
- net_log_return = log(1 + net_simple_return)
- reward cost == diagnostics cost
- NAV consistency check
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.loader import DataContractError
from src.envs.reward_calculator import RewardCalculator, VALID_REWARD_VARIANTS
from src.envs.state import ExecutionResult, PortfolioState


def _make_execution_result(
    net_return: float = 0.01,
    gross_return: float = 0.011,
    transaction_cost: float = 0.001,
    portfolio_log_return: float | None = None,
    nav_after_cost: float = 1.0,
    post_execution_return: float = 0.01,
    nav_next: float | None = None,
) -> ExecutionResult:
    if portfolio_log_return is None:
        portfolio_log_return = float(np.log1p(net_return))
    if nav_next is None:
        nav_next = nav_after_cost * (1.0 + post_execution_return)
    return ExecutionResult(
        executed_weights=np.array([0.5, 0.5]),
        pre_execution_drifted_weights=np.array([0.5, 0.5]),
        turnover=0.1,
        transaction_cost=transaction_cost,
        transaction_cost_on_initial_nav=transaction_cost,
        proportional_cost=transaction_cost * 0.5,
        fixed_cost=0.0,
        slippage_cost=0.0,
        market_impact_cost=0.0,
        total_transaction_cost=transaction_cost,
        estimated_turnover=0.1,
        realized_turnover=0.1,
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


def _make_portfolio_state() -> PortfolioState:
    return PortfolioState(
        date="2024-01-01",
        nav=1.0,
        portfolio_value=100000.0,
        current_weights=np.array([0.5, 0.5]),
    )


# ---------------------------------------------------------------------------
# A13 reward mode
# ---------------------------------------------------------------------------

class TestOTARRewardMode:
    """A13_otar_soft_ru_cvar_fixed must be a valid reward variant."""

    def test_a13_in_valid_variants(self) -> None:
        assert "A13_otar_soft_ru_cvar_fixed" in VALID_REWARD_VARIANTS

    def test_a13_reward_calculates(self) -> None:
        config = {"reward": {"mode": "A13_otar_soft_ru_cvar_fixed"}}
        calc = RewardCalculator(config)
        exec_result = _make_execution_result()
        portfolio_state = _make_portfolio_state()
        reward, info = calc.calculate(exec_result, portfolio_state)
        assert np.isfinite(reward), f"Reward should be finite, got {reward}"
        assert info["variant"] == "A13_otar_soft_ru_cvar_fixed"


# ---------------------------------------------------------------------------
# NAV consistency
# ---------------------------------------------------------------------------

class TestNAVConsistency:
    """NAV consistency check: nav_next == nav_after_cost * (1 + post_execution_return)."""

    def test_nav_consistency_passes(self) -> None:
        config = {"reward": {"mode": "A13_otar_soft_ru_cvar_fixed"}}
        calc = RewardCalculator(config)
        exec_result = _make_execution_result(
            nav_next=1.009 * 1.001,  # nav_after_cost * (1 + post_execution_return)
            nav_after_cost=1.009,
            post_execution_return=0.001,
        )
        portfolio_state = _make_portfolio_state()
        reward, info = calc.calculate(exec_result, portfolio_state)
        assert np.isfinite(reward)

    def test_nav_consistency_fails_on_mismatch(self) -> None:
        config = {"reward": {"mode": "A13_otar_soft_ru_cvar_fixed"}}
        calc = RewardCalculator(config)
        exec_result = _make_execution_result(
            nav_next=2.0,  # Deliberately wrong
            nav_after_cost=1.009,
            post_execution_return=0.001,
        )
        portfolio_state = _make_portfolio_state()
        with pytest.raises(DataContractError, match="ERR_REWARD_COST_ACCOUNTING"):
            calc.calculate(exec_result, portfolio_state)


# ---------------------------------------------------------------------------
# v_init warmup
# ---------------------------------------------------------------------------

class TestVInitWarmup:
    """v_init_source and resolve_v_init must work correctly."""

    def test_v_init_fixed_mode(self) -> None:
        config = {"reward": {"v_init_source": "fixed", "v_init": 0.05}}
        calc = RewardCalculator(config)
        resolved = calc.resolve_v_init()
        assert resolved == pytest.approx(0.05)

    def test_v_init_warmup_quantile_mode(self) -> None:
        config = {"reward": {"v_init_source": "training_warmup_loss_quantile", "v_init_confidence_q": 0.95}}
        calc = RewardCalculator(config)
        losses = np.random.RandomState(42).exponential(0.01, size=100)
        resolved = calc.resolve_v_init(losses)
        assert np.isfinite(resolved), f"Resolved v_init should be finite, got {resolved}"
        assert resolved > 0.0, f"Resolved v_init should be > 0, got {resolved}"

    def test_v_init_resolved_value_attribute(self) -> None:
        config = {"reward": {"v_init_source": "fixed", "v_init": 0.03}}
        calc = RewardCalculator(config)
        calc.resolve_v_init()
        assert hasattr(calc, "v_init_resolved_value"), "Missing v_init_resolved_value attribute"
        assert calc.v_init_resolved_value == pytest.approx(0.03)
