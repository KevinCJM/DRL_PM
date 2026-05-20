from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.envs.reward_calculator import RewardCalculator
from src.envs.state import ExecutionResult, PortfolioState


def test_default_reward_uses_net_log_return_without_double_cost():
    result = _execution_result(portfolio_log_return=0.012, transaction_cost=0.004, turnover=0.25)
    reward, info = RewardCalculator(DEFAULT_CONFIG).calculate(result, _portfolio_state([0.012]))

    assert reward == pytest.approx(0.012)
    assert info["variant"] == "A2_net_log_return_after_cost"
    assert info["transaction_cost"] == pytest.approx(0.004)


def test_turnover_downside_drawdown_reward_variant():
    config = deepcopy(DEFAULT_CONFIG)
    config["reward"]["mode"] = "A6_net_log_return_plus_turnover_downside_drawdown"
    result = _execution_result(portfolio_log_return=0.01, net_return=-0.02, turnover=0.5)
    state = _portfolio_state([-0.02], current_drawdown_abs=0.15)

    reward, info = RewardCalculator(config).calculate(result, state)

    expected = 0.01 - 0.001 * 0.5 - 0.10 * (0.02**2) - 0.20 * (0.15 - 0.10)
    assert reward == pytest.approx(expected)
    assert info["downside_penalty"] == pytest.approx(0.02**2)
    assert info["drawdown_penalty"] == pytest.approx(0.05)


def test_differential_sharpe_resets_episode_state():
    config = deepcopy(DEFAULT_CONFIG)
    config["reward"]["mode"] = "A7_differential_sharpe"
    calc = RewardCalculator(config)
    result = _execution_result(portfolio_log_return=0.02, net_return=0.02)
    state = _portfolio_state([0.02])

    reward, info = calc.calculate(result, state, reset_episode=True)
    assert reward == pytest.approx(0.02)
    assert info["differential_sharpe_step"] == 1
    first_a = info["differential_sharpe_A"]
    calc.calculate(_execution_result(portfolio_log_return=-0.01, net_return=-0.01), _portfolio_state([-0.01]))
    reward, info = calc.calculate(result, state, reset_episode=True)
    assert reward == pytest.approx(0.02)
    assert info["differential_sharpe_step"] == 1
    assert info["differential_sharpe_A"] == pytest.approx(first_a)


def test_cvar_sensitive_uses_configured_confidence_mapping():
    config = deepcopy(DEFAULT_CONFIG)
    config["reward"]["mode"] = "A8_cvar_sensitive"
    config["reward"]["lambda_turnover"] = 0.0
    config["reward"]["lambda_downside"] = 0.0
    config["reward"]["lambda_drawdown"] = 0.0
    config["reward"]["lambda_volatility"] = 0.0
    config["reward"]["lambda_cvar"] = 1.0
    config["reward"]["lambda_concentration"] = 0.0
    result = _execution_result(portfolio_log_return=0.01, net_return=0.01)
    state = _portfolio_state([-0.10, -0.03, 0.01, 0.02])

    reward, info = RewardCalculator(config).calculate(result, state)

    assert info["cvar_confidence"] == pytest.approx(0.95)
    assert info["cvar_alpha"] == pytest.approx(0.05)
    assert info["cvar_loss"] == pytest.approx(0.10)
    assert reward == pytest.approx(0.01 - 0.10)


def test_preference_conditioned_reward_vector_dot():
    config = deepcopy(DEFAULT_CONFIG)
    config["reward"]["mode"] = "A12_multi_objective_preference_conditioned"
    omega = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    reward, info = RewardCalculator(config).calculate(
        _execution_result(portfolio_log_return=0.02, turnover=0.5),
        _portfolio_state([0.02]),
        omega=omega,
    )

    assert reward == pytest.approx(0.02 - 0.05)
    assert info["omega"] == omega.tolist()


def _execution_result(
    *,
    portfolio_log_return: float,
    net_return: float | None = None,
    gross_return: float | None = None,
    transaction_cost: float = 0.0,
    turnover: float = 0.0,
) -> ExecutionResult:
    if net_return is None:
        net_return = float(np.expm1(portfolio_log_return))
    if gross_return is None:
        gross_return = net_return + transaction_cost
    weights = np.array([0.6, 0.4], dtype=float)
    return ExecutionResult(
        executed_weights=weights,
        pre_execution_drifted_weights=weights,
        turnover=turnover,
        transaction_cost=transaction_cost,
        transaction_cost_on_initial_nav=transaction_cost,
        proportional_cost=transaction_cost,
        fixed_cost=0.0,
        slippage_cost=0.0,
        market_impact_cost=0.0,
        total_transaction_cost=transaction_cost,
        estimated_turnover=None,
        realized_turnover=turnover,
        estimated_cost=None,
        realized_cost=transaction_cost,
        gross_return=gross_return,
        net_return=net_return,
        pre_execution_return=0.0,
        post_execution_return=net_return,
        portfolio_log_return=portfolio_log_return,
        nav_execution=1.0,
        nav_after_cost=1.0,
        nav_next=1.0 + net_return,
        info={},
    )


def _portfolio_state(rolling_returns, *, current_drawdown_abs: float = 0.0) -> PortfolioState:
    return PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=100000000.0,
        current_weights=np.array([0.6, 0.4]),
        current_drawdown_abs=current_drawdown_abs,
        rolling_returns=list(rolling_returns),
    )
