from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError
from src.envs.rebalance_scheduler import RebalanceScheduler
from src.envs.state import DecisionMarketState, PortfolioState


def _portfolio_state(step_index=0, drawdown=0.0):
    return PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=100000000.0,
        current_weights=np.array([0.60, 0.40]),
        current_drawdown_abs=drawdown,
        step_index=step_index,
    )


def _decision_state(volatility=0.01):
    ones = np.ones(2)
    zeros = np.zeros(2)
    return DecisionMarketState(
        decision_date=pd.Timestamp("2024-01-02"),
        available_mask_at_decision=np.array([True, True]),
        availability_reason_at_decision=np.array(["listed", "listed"], dtype=object),
        close_at_decision=ones,
        log_return_at_decision=zeros,
        log_return_window=np.zeros((3, 2)),
        amount_at_decision=ones * 1000.0,
        volume_at_decision=ones * 10.0,
        adv20_at_decision=ones * 1000000.0,
        volatility_20d_at_decision=ones * volatility,
        turnover_rate_at_decision=ones * 0.01,
        feature_window=np.zeros((4, 3, 2)),
        market_image=np.zeros((4, 3, 2)),
    )


def _config(mode):
    config = deepcopy(DEFAULT_CONFIG)
    config["rebalance"]["mode"] = mode
    return config


def test_pre_check_never_once_and_daily_modes():
    portfolio_state = _portfolio_state()
    decision_state = _decision_state()

    assert RebalanceScheduler(_config("never")).pre_check("2024-01-02", portfolio_state, decision_state, {}) is False
    assert RebalanceScheduler(_config("daily")).pre_check("2024-01-02", portfolio_state, decision_state, {}) is True

    scheduler = RebalanceScheduler(_config("once"))
    assert scheduler.should_rebalance("2024-01-02", portfolio_state, decision_state, {}) is True
    assert scheduler.should_rebalance("2024-01-03", portfolio_state, decision_state, {}) is False


def test_calendar_modes_use_trading_date_index_first_and_last():
    date_index = pd.to_datetime(
        [
            "2024-01-29",
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
            "2024-02-02",
            "2024-02-05",
        ]
    )
    portfolio_state = _portfolio_state()
    decision_state = _decision_state()

    monthly_last = _config("monthly")
    monthly_last["rebalance"]["calendar_position"] = "last_trading_day"
    scheduler = RebalanceScheduler(monthly_last, date_index=date_index)
    assert scheduler.pre_check("2024-01-30", portfolio_state, decision_state, {}) is False
    assert scheduler.pre_check("2024-01-31", portfolio_state, decision_state, {}) is True

    weekly_first = _config("weekly")
    weekly_first["rebalance"]["calendar_position"] = "first_trading_day"
    scheduler = RebalanceScheduler(weekly_first, date_index=date_index)
    assert scheduler.pre_check("2024-02-02", portfolio_state, decision_state, {}) is False
    assert scheduler.pre_check("2024-02-05", portfolio_state, decision_state, {}) is True

    monthly_first = _config("monthly")
    monthly_first["rebalance"]["calendar_rule"] = "first_trading_day"
    scheduler = RebalanceScheduler(monthly_first, date_index=date_index)
    assert scheduler.pre_check("2024-02-01", portfolio_state, decision_state, {}) is True
    assert scheduler.pre_check("2024-02-02", portfolio_state, decision_state, {}) is False


def test_every_n_days_tracks_last_allowed_trading_day():
    config = _config("every_n_days")
    config["rebalance"]["every_n_days"] = 2
    scheduler = RebalanceScheduler(config, date_index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    portfolio_state = _portfolio_state()
    decision_state = _decision_state()

    assert scheduler.should_rebalance("2024-01-02", portfolio_state, decision_state, {}) is True
    assert scheduler.should_rebalance("2024-01-03", portfolio_state, decision_state, {}) is False
    assert scheduler.should_rebalance("2024-01-04", portfolio_state, decision_state, {}) is True


def test_calendar_dates_mode():
    config = _config("calendar_dates")
    config["rebalance"]["calendar_dates"] = ["2024-01-03"]
    scheduler = RebalanceScheduler(config)

    assert scheduler.pre_check("2024-01-02", _portfolio_state(), _decision_state(), {}) is False
    assert scheduler.pre_check("2024-01-03", _portfolio_state(), _decision_state(), {}) is True


def test_threshold_modes_require_candidate_weights():
    portfolio_state = _portfolio_state()
    decision_state = _decision_state()

    drift_config = _config("threshold_weight_drift")
    drift_config["rebalance"]["threshold_weight_drift"] = 0.10
    drift_scheduler = RebalanceScheduler(drift_config)
    assert drift_scheduler.should_rebalance("2024-01-02", portfolio_state, decision_state, {}, None) is False
    assert (
        drift_scheduler.should_rebalance(
            "2024-01-02",
            portfolio_state,
            decision_state,
            {},
            np.array([0.75, 0.25]),
        )
        is True
    )

    turnover_config = _config("threshold_turnover")
    turnover_config["rebalance"]["threshold_turnover"] = 0.20
    turnover_scheduler = RebalanceScheduler(turnover_config)
    assert (
        turnover_scheduler.should_rebalance(
            "2024-01-02",
            portfolio_state,
            decision_state,
            {},
            np.array([0.65, 0.35]),
        )
        is False
    )
    assert (
        turnover_scheduler.should_rebalance(
            "2024-01-02",
            portfolio_state,
            decision_state,
            {},
            np.array([0.90, 0.10]),
        )
        is True
    )


def test_event_modes_use_state_thresholds():
    volatility_config = _config("volatility_event")
    volatility_config["rebalance"]["volatility_threshold_annual"] = 0.20
    assert RebalanceScheduler(volatility_config).pre_check(
        "2024-01-02",
        _portfolio_state(),
        _decision_state(volatility=0.02),
        {},
    )

    drawdown_config = _config("drawdown_event")
    drawdown_config["rebalance"]["drawdown_threshold"] = 0.10
    assert RebalanceScheduler(drawdown_config).pre_check(
        "2024-01-02",
        _portfolio_state(drawdown=0.12),
        _decision_state(),
        {},
    )

    risk_config = _config("risk_budget_breach")
    risk_config["rebalance"]["risk_budget_tolerance"] = 0.05
    assert RebalanceScheduler(risk_config).pre_check(
        "2024-01-02",
        _portfolio_state(),
        _decision_state(),
        {"risk_budget_deviation": np.array([0.01, 0.07])},
    )


def test_final_rebalance_action_requires_scheduler_and_gate():
    assert RebalanceScheduler.final_rebalance_action(True, 1) == 1
    assert RebalanceScheduler.final_rebalance_action(True, 0) == 0
    assert RebalanceScheduler.final_rebalance_action(False, 1) == 0

    assert (
        RebalanceScheduler(_config("daily")).should_rebalance(
            "2024-01-02",
            _portfolio_state(),
            _decision_state(),
            {},
            gate_action=0,
        )
        is False
    )

    every_n_config = _config("every_n_days")
    every_n_config["rebalance"]["every_n_days"] = 2
    scheduler = RebalanceScheduler(
        every_n_config,
        date_index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    assert scheduler.should_rebalance("2024-01-02", _portfolio_state(), _decision_state(), {}, gate_action=0) is False
    assert scheduler.should_rebalance("2024-01-03", _portfolio_state(), _decision_state(), {}, gate_action=1) is False
    assert scheduler.should_rebalance("2024-01-04", _portfolio_state(), _decision_state(), {}, gate_action=1) is True

    with pytest.raises(DataContractError) as error:
        RebalanceScheduler.final_rebalance_action(True, 2)
    assert error.value.code == "ERR_STATE_SCHEMA_MISMATCH"
