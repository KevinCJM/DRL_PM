from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError
from src.envs.backtest_engine import _activity_metrics, _execution_activity_config, _finalize_execution_action
from src.envs.rebalance_scheduler import RebalanceScheduler
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


def test_scheduler_blocked_and_model_chosen_hold_are_distinct():
    config = deepcopy(DEFAULT_CONFIG)
    config["rebalance"]["mode"] = "monthly"
    config["execution_activity"]["protocol"] = "monthly_gate"
    scheduler = RebalanceScheduler(config, date_index=pd.date_range("2024-01-01", periods=5, freq="B"))
    action = PortfolioAction(
        target_weights=np.array([0.2, 0.8]),
        rebalance_action=1,
        rebalance_intensity=0.5,
        action_info={"paper_model_id": "model", "raw_rho": 0.5, "raw_model_requested_rebalance": True},
    )

    final = _finalize_execution_action(
        scheduler,
        pd.Timestamp("2024-01-03"),
        _portfolio_state(),
        _decision_state(),
        action,
        False,
        _execution_activity_config(config),
    )

    assert final.action_info["raw_rho"] == 0.5
    assert final.action_info["final_rho"] == 0.0
    assert final.action_info["execution_scheduler_blocked"] is True
    assert final.action_info["model_chosen_hold"] is False


def test_daily_nonblocking_protocol_preserves_raw_execution_request():
    config = deepcopy(DEFAULT_CONFIG)
    config["rebalance"]["mode"] = "daily"
    config["execution_activity"].update(
        {
            "protocol": "daily_gate_with_cost_constraint",
            "scheduler_blocks_model_actions": False,
            "activity_gate_enforced": True,
        }
    )
    scheduler = RebalanceScheduler(config, date_index=pd.date_range("2024-01-01", periods=5, freq="B"))
    action = PortfolioAction(
        target_weights=np.array([0.2, 0.8]),
        rebalance_action=1,
        rebalance_intensity=0.25,
        action_info={"paper_model_id": "model", "raw_rho": 0.25, "raw_model_requested_rebalance": True},
    )

    final = _finalize_execution_action(
        scheduler,
        pd.Timestamp("2024-01-03"),
        _portfolio_state(),
        _decision_state(),
        action,
        False,
        _execution_activity_config(config),
    )

    assert final.rebalance_action == 1
    assert final.rebalance_intensity == 0.25
    assert final.action_info["execution_scheduler_blocked"] is False


def test_daily_nonblocking_protocol_does_not_resurrect_threshold_hold():
    config = deepcopy(DEFAULT_CONFIG)
    config["rebalance"]["mode"] = "daily"
    config["execution_activity"].update(
        {
            "protocol": "daily_gate_with_cost_constraint",
            "scheduler_blocks_model_actions": False,
            "activity_gate_enforced": True,
        }
    )
    scheduler = RebalanceScheduler(config, date_index=pd.date_range("2024-01-01", periods=5, freq="B"))
    action = PortfolioAction(
        target_weights=np.array([0.2, 0.8]),
        rebalance_action=0,
        rebalance_intensity=0.0,
        action_info={
            "paper_model_id": "model",
            "raw_gate_requested_rebalance": True,
            "raw_model_requested_rebalance": False,
            "raw_action": 0,
            "raw_rho": 0.0,
            "forced_hold_reason": "below_rebalance_turnover_threshold",
        },
    )

    final = _finalize_execution_action(
        scheduler,
        pd.Timestamp("2024-01-03"),
        _portfolio_state(),
        _decision_state(),
        action,
        False,
        _execution_activity_config(config),
    )

    assert final.rebalance_action == 0
    assert final.rebalance_intensity == 0.0
    assert final.action_info["raw_gate_requested_rebalance"] is True
    assert final.action_info["raw_model_requested_rebalance"] is False
    assert final.action_info["final_action"] == 0
    assert final.action_info["final_rho"] == 0.0
    assert final.action_info["forced_hold_reason"] == "below_rebalance_turnover_threshold"


def test_activity_metrics_include_protocol_labels():
    metrics = _activity_metrics(
        [
            {
                "activity_protocol": "daily_gate_with_cost_constraint",
                "turnover_optimization_protocol_id": "turnover_active_v1",
                "scheduler_blocks_model_actions": False,
                "activity_gate_enforced": True,
                "raw_model_requested_rebalance": True,
                "final_action": 1,
                "trade_opportunity": 1,
                "non_initial_trade_opportunity": 1,
                "first_trade": 0,
                "turnover": 0.01,
                "raw_rho": 0.5,
                "final_rho": 0.5,
            }
        ]
    )

    assert metrics["activity_protocol"] == "daily_gate_with_cost_constraint"
    assert metrics["execution_activity_protocol"] == "daily_gate_with_cost_constraint"
    assert metrics["turnover_optimization_protocol_id"] == "turnover_active_v1"


def test_evaluate_pre_post_no_mutation_preserves_scheduler_state():
    config = deepcopy(DEFAULT_CONFIG)
    config["rebalance"]["mode"] = "daily"
    scheduler = RebalanceScheduler(config, date_index=pd.date_range("2024-01-01", periods=5, freq="B"))

    evaluation = scheduler.evaluate_pre_post_no_mutation(
        pd.Timestamp("2024-01-03"),
        _portfolio_state(),
        _decision_state(),
        candidate_weights=np.array([0.2, 0.8]),
    )

    assert evaluation.scheduler_final_allowed is True
    assert scheduler._last_allowed_date is None
    scheduler.commit_scheduler_decision(
        pd.Timestamp("2024-01-03"),
        scheduler_pre_allowed=evaluation.scheduler_pre_allowed,
        scheduler_post_allowed=evaluation.scheduler_post_allowed,
        scheduler_final_allowed=evaluation.scheduler_final_allowed,
        raw_model_requested_rebalance=True,
        final_action=True,
        execution_accepted=True,
    )
    assert scheduler._last_allowed_date == pd.Timestamp("2024-01-03")


def test_weekly_gate_rejects_nonblocking_scheduler():
    config = deepcopy(DEFAULT_CONFIG)
    config["execution_activity"].update({"protocol": "weekly_gate", "scheduler_blocks_model_actions": False})

    with pytest.raises(DataContractError):
        _execution_activity_config(config)


def _decision_state():
    returns = np.array([[0.0, 0.0], [0.01, -0.01], [0.02, 0.0]], dtype=float)
    return DecisionMarketState(
        decision_date=pd.Timestamp("2024-01-03"),
        available_mask_at_decision=np.array([True, True]),
        availability_reason_at_decision=np.array(["listed", "listed"], dtype=object),
        close_at_decision=np.array([10.0, 20.0]),
        log_return_at_decision=returns[-1],
        log_return_window=returns,
        amount_at_decision=np.array([1000.0, 2000.0]),
        volume_at_decision=np.array([100.0, 200.0]),
        adv20_at_decision=np.array([1000.0, 2000.0]),
        volatility_20d_at_decision=np.array([0.1, 0.2]),
        turnover_rate_at_decision=np.array([0.01, 0.02]),
        feature_window=returns[np.newaxis, :, :],
        market_image=returns[np.newaxis, :, :],
    )


def _portfolio_state():
    return PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=100.0,
        current_weights=np.array([0.6, 0.4]),
        rolling_returns=[0.01, -0.02, 0.005],
    )
