import torch
import pytest
import inspect
import numpy as np
import pandas as pd
from src.models.dqn_gate import DQNGate

def test_gate_input_contract():
    batch_size = 4
    latent_dim = 256
    n_assets = 10
    
    gate = DQNGate(latent_dim=latent_dim, n_assets=n_assets, dueling=True)
    
    latent = torch.randn(batch_size, latent_dim)
    candidate_weights = torch.randn(batch_size, n_assets)
    current_weights = torch.randn(batch_size, n_assets)
    estimated_turnover = torch.randn(batch_size, 1)
    estimated_cost = torch.randn(batch_size, 1)
    
    q_values = gate(
        latent, 
        candidate_weights, 
        current_weights, 
        estimated_turnover, 
        estimated_cost
    )
    
    assert q_values.shape == (batch_size, 2) # Binary gate: Hold (0) or Rebalance (1)
    assert gate.output_dim == 2

    forward_args = set(inspect.signature(gate.forward).parameters)
    assert forward_args == {
        "latent",
        "candidate_weights",
        "current_weights",
        "estimated_turnover",
        "estimated_cost",
    }

def test_dqn_gate_dueling():
    gate = DQNGate(latent_dim=128, n_assets=5, dueling=True)
    latent = torch.randn(1, 128)
    cw = torch.randn(1, 5)
    curw = torch.randn(1, 5)
    et = torch.randn(1, 1)
    ec = torch.randn(1, 1)
    
    q_values = gate(latent, cw, curw, et, ec)
    assert q_values.shape == (1, 2)
    
    # Check if internal components exist
    assert hasattr(gate, "advantage_net")
    assert hasattr(gate, "value_net")

def test_dqn_gate_partial_output_and_action_controls():
    gate = DQNGate(latent_dim=32, n_assets=3, dueling=False, output_dim=5)
    latent = torch.randn(2, 32)
    cw = torch.randn(2, 3)
    curw = torch.randn(2, 3)
    et = torch.randn(2, 1)
    ec = torch.randn(2, 1)

    q_values = gate(latent, cw, curw, et, ec)
    assert q_values.shape == (2, 5)

    binary_gate = DQNGate(latent_dim=32, n_assets=3)
    binary_q = torch.tensor([[0.0, 0.04], [0.0, 0.20], [0.10, 0.00]])
    action = binary_gate.select_action(binary_q, q_gap_threshold=0.05)
    assert torch.equal(action, torch.tensor([0, 1, 0]))

    previous = torch.tensor([1, 0, 1])
    hysteresis_action = binary_gate.select_action(
        binary_q,
        q_gap_threshold=0.05,
        previous_action=previous,
        hysteresis_margin=0.10,
    )
    assert torch.equal(hysteresis_action, torch.tensor([1, 1, 1]))

    cooldown_action = binary_gate.select_action(binary_q, cooldown_mask=torch.tensor([False, True, False]))
    assert torch.equal(cooldown_action, torch.tensor([1, 0, 0]))

def test_double_dqn_target_and_n_step_returns():
    rewards = torch.tensor([[1.0], [2.0]])
    next_q_online = torch.tensor([[0.1, 0.5], [0.9, 0.2]])
    next_q_target = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    done = torch.tensor([[0.0], [1.0]])

    target = DQNGate.compute_double_dqn_target(
        rewards,
        next_q_online,
        next_q_target,
        done,
        gamma=0.5,
        n_steps=2,
    )
    assert torch.allclose(target, torch.tensor([[6.0], [2.0]]))

    n_step_rewards = torch.tensor([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]])
    n_step_done = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    next_value = torch.tensor([[10.0], [10.0]])
    n_step_return = DQNGate.compute_n_step_returns(
        n_step_rewards,
        n_step_done,
        next_value,
        gamma=0.5,
    )
    assert torch.allclose(n_step_return, torch.tensor([[3.0], [3.5]]))

def test_estimated_cost_uses_decision_state_only():
    from src.models.cost_estimator import CostEstimator
    from src.envs.state import DecisionMarketState, PortfolioState
    
    batch_size = 2
    n_assets = 4
    
    candidate_weights = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    current_weights = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
    adv20 = torch.tensor([[1e7, 2e7, 3e7, 4e7], [1e7, 1e7, 1e7, 1e7]])
    sigma20 = torch.tensor([[0.02, 0.02, 0.02, 0.02], [0.01, 0.01, 0.01, 0.01]])
    portfolio_value = 1e8
    
    config = {
        "cost_model": {
            "proportional_cost": 0.0005,
            "slippage": 0.0002,
            "market_impact_enabled": True,
            "market_impact_coef": 0.1,
            "adv_eps": 1e6
        }
    }
    
    turnover, cost = CostEstimator.estimate(
        candidate_weights,
        current_weights,
        adv20,
        sigma20,
        portfolio_value,
        config
    )
    
    assert turnover.shape == (batch_size, 1)
    assert cost.shape == (batch_size, 1)
    
    # Manual check for first sample
    # trade_weights = [0.15, 0.05, 0.05, 0.15]
    # turnover = 0.5 * 0.4 = 0.2
    assert torch.allclose(turnover[0], torch.tensor([0.2]))

    decision_state = DecisionMarketState(
        decision_date=pd.Timestamp("2024-01-02"),
        available_mask_at_decision=np.ones(n_assets, dtype=bool),
        availability_reason_at_decision=None,
        close_at_decision=np.ones(n_assets),
        log_return_at_decision=np.zeros(n_assets),
        log_return_window=np.zeros((3, n_assets)),
        amount_at_decision=np.array([1e8, 2e8, 3e8, 4e8]),
        volume_at_decision=np.ones(n_assets),
        adv20_at_decision=adv20[0].numpy(),
        volatility_20d_at_decision=sigma20[0].numpy(),
        turnover_rate_at_decision=np.array([0.1, 0.2, 0.3, 0.4]),
        feature_window=np.zeros((1, 3, n_assets)),
        market_image=np.zeros((1, 3, n_assets)),
    )
    portfolio_state = PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=portfolio_value,
        current_weights=current_weights[0].numpy(),
    )
    state_turnover, state_cost = CostEstimator.estimate_from_decision_state(
        candidate_weights[0:1],
        current_weights[0:1],
        decision_state,
        portfolio_state,
        config,
    )
    assert torch.allclose(state_turnover, turnover[0:1])
    assert torch.allclose(state_cost, cost[0:1])

    calibrated_config = {
        "cost_model": {
            "mode": "calibrated",
            "proportional_cost": 0.0005,
            "fixed_cost": 10_000.0,
            "slippage": 0.99,
            "market_impact_enabled": True,
            "market_impact_coef": 99.0,
            "adv_eps": 1e6,
        },
        "execution_model": {"fixed_cost_unit": "currency"},
    }
    calibration_table = {"realized_bps_median": torch.full((1, n_assets), 5.0)}
    calibrated_turnover, calibrated_cost = CostEstimator.estimate_from_decision_state(
        candidate_weights[0:1],
        current_weights[0:1],
        decision_state,
        portfolio_state,
        calibrated_config,
        calibration_table=calibration_table,
    )
    expected_calibrated_cost = 0.0005 * calibrated_turnover + (10_000.0 / portfolio_value) + (
        5.0 / 10000.0
    ) * torch.sum(torch.abs(candidate_weights[0:1] - current_weights[0:1]), dim=1, keepdim=True)
    assert torch.allclose(calibrated_cost, expected_calibrated_cost)

    with pytest.raises(ValueError, match="ERR_COST_CALIBRATION_NOT_FITTED"):
        CostEstimator.estimate_from_decision_state(
            candidate_weights[0:1],
            current_weights[0:1],
            decision_state,
            portfolio_state,
            calibrated_config,
        )
