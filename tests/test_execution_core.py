from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError, MarketDatasetBundle
from src.envs.cost_model import CostModel
from src.envs.portfolio_execution_core import PortfolioExecutionCore, drift_weights, sanitize_execution_returns
from src.envs.state import (
    DecisionMarketState,
    ExecutionMarketState,
    ExecutionResult,
    PendingAction,
    PortfolioAction,
    PortfolioState,
)


def test_state_dataclass_contracts():
    date = pd.Timestamp("2024-01-02")
    execution_date = pd.Timestamp("2024-01-03")
    weights = np.array([0.4, 0.6], dtype=float)
    zeros = np.zeros(2, dtype=float)
    ones = np.ones(2, dtype=float)

    portfolio_state = PortfolioState(
        date=date,
        nav=1.0,
        portfolio_value=100000000.0,
        current_weights=weights,
        rolling_returns=[0.01],
    )
    assert portfolio_state.date == date
    assert portfolio_state.drifted_weights.shape == (2,)
    assert portfolio_state.previous_executed_weights.shape == (2,)
    assert portfolio_state.sellable_mask.dtype == np.bool_
    assert portfolio_state.frozen_weight.tolist() == [0.0, 0.0]

    action = PortfolioAction(weights, 1, 0.5, {"source": "unit"})
    assert action.target_weights.shape == (2,)
    assert action.weights is action.target_weights
    assert action.rebalance == 1
    assert action.rebalance_intensity == 0.5

    pending_action = PendingAction(
        decision_date=date,
        execution_date=execution_date,
        next_valuation_date=execution_date,
        target_weights=weights,
        candidate_weights=weights,
        rebalance_action=1,
        rebalance_intensity=0.5,
        execution_price="next_open",
        execution_price_type="open",
        q_hold=None,
        q_rebalance=0.2,
        q_gap=0.1,
        decision_value=0.3,
        action_info={"gate_action": 1, "decision_log_prob": -0.7},
    )
    assert pending_action.execution_date == execution_date
    assert pending_action.target_weights.shape == (2,)
    assert pending_action.action_info["decision_log_prob"] == -0.7

    decision_state = DecisionMarketState(
        decision_date=date,
        available_mask_at_decision=np.array([True, False]),
        availability_reason_at_decision=np.array(["listed", "suspended"], dtype=object),
        close_at_decision=ones,
        log_return_at_decision=zeros,
        log_return_window=np.zeros((3, 2)),
        amount_at_decision=ones * 1000.0,
        volume_at_decision=ones * 10.0,
        adv20_at_decision=ones * 1000000.0,
        volatility_20d_at_decision=ones * 0.02,
        turnover_rate_at_decision=ones * 0.01,
        feature_window=np.zeros((4, 3, 2)),
        market_image=np.zeros((4, 3, 2)),
    )
    assert decision_state.market_image.shape == (4, 3, 2)
    assert "execution_date" not in decision_state.__dataclass_fields__

    execution_state = ExecutionMarketState(
        decision_date=date,
        execution_date=execution_date,
        next_valuation_date=execution_date,
        execution_price_type="open",
        execution_price=ones,
        tradeable_mask_at_execution=np.array([True, False]),
        availability_reason_at_execution=np.array(["listed", "suspended"], dtype=object),
        return_from_decision_to_execution=zeros,
        holding_simple_return=zeros,
        amount_at_execution=ones * 1000.0,
        volume_at_execution=ones * 10.0,
        adv20_at_execution=ones * 1000000.0,
        volatility_20d_at_execution=ones * 0.02,
        turnover_rate_at_execution=ones * 0.01,
    )
    assert execution_state.execution_price.shape == (2,)
    assert execution_state.tradeable_mask_at_execution.dtype == np.bool_
    np.testing.assert_allclose(execution_state.turnover_rate_at_execution, ones * 0.01)

    result = ExecutionResult(
        executed_weights=weights,
        pre_execution_drifted_weights=weights,
        turnover=0.0,
        transaction_cost=0.0,
        transaction_cost_on_initial_nav=0.0,
        proportional_cost=0.0,
        fixed_cost=0.0,
        slippage_cost=0.0,
        market_impact_cost=0.0,
        total_transaction_cost=0.0,
        estimated_turnover=None,
        realized_turnover=0.0,
        estimated_cost=None,
        realized_cost=0.0,
        gross_return=0.0,
        net_return=0.0,
        pre_execution_return=0.0,
        post_execution_return=0.0,
        portfolio_log_return=0.0,
        nav_execution=1.0,
        nav_after_cost=1.0,
        nav_next=1.0,
        info={"return_imputation": []},
    )
    assert result.nav_next == 1.0
    assert result.info["return_imputation"] == []


def test_state_dataclass_shape_and_finite_validation():
    with pytest.raises(DataContractError) as non_finite_error:
        PortfolioAction(np.array([0.5, np.nan]), 1)
    assert non_finite_error.value.code == "ERR_ACTION_NON_FINITE"

    with pytest.raises(DataContractError) as action_shape_error:
        PortfolioAction(np.array([[0.5, 0.5]]), 1)
    assert action_shape_error.value.code == "ERR_ACTION_SHAPE_MISMATCH"

    with pytest.raises(DataContractError) as state_shape_error:
        ExecutionMarketState(
            decision_date=pd.Timestamp("2024-01-02"),
            execution_date=pd.Timestamp("2024-01-03"),
            next_valuation_date=pd.Timestamp("2024-01-03"),
            execution_price_type="open",
            execution_price=np.ones(2),
            tradeable_mask_at_execution=np.ones(3, dtype=bool),
            availability_reason_at_execution=None,
            return_from_decision_to_execution=np.zeros(2),
            holding_simple_return=np.zeros(2),
            amount_at_execution=np.ones(2),
            volume_at_execution=np.ones(2),
            adv20_at_execution=np.ones(2),
            volatility_20d_at_execution=np.ones(2),
        )
    assert state_shape_error.value.code == "ERR_STATE_SCHEMA_MISMATCH"


def test_next_open_execution_returns():
    config = _config("next_open")
    state = PortfolioExecutionCore(config).build_execution_market_state(
        _market_dataset_bundle(),
        pd.Timestamp("2024-01-02"),
    )

    assert state.execution_date == pd.Timestamp("2024-01-03")
    assert state.next_valuation_date == pd.Timestamp("2024-01-03")
    assert state.execution_price_type == "open"
    np.testing.assert_allclose(state.execution_price, np.array([11.0, 18.0]))
    np.testing.assert_allclose(state.return_from_decision_to_execution, np.array([0.10, -0.10]))
    np.testing.assert_allclose(state.holding_simple_return, np.array([12.0 / 11.0 - 1.0, 21.0 / 18.0 - 1.0]))
    np.testing.assert_allclose(state.amount_at_execution, np.array([1000.0, 2000.0]))
    np.testing.assert_allclose(state.volume_at_execution, np.array([10.0, 20.0]))
    np.testing.assert_allclose(state.adv20_at_execution, np.array([1000.0, 2000.0]))
    np.testing.assert_allclose(state.volatility_20d_at_execution, np.zeros(2))
    assert state.cost_observation_date == pd.Timestamp("2024-01-02")
    assert state.cost_observation_timing == "decision_observable"
    np.testing.assert_allclose(state.amount_at_cost_observation, np.array([1000.0, 2000.0]))


def test_next_open_calibrated_cost_path_uses_observable_amount():
    state = PortfolioExecutionCore(_config("next_open")).build_execution_market_state(
        _market_dataset_bundle(),
        pd.Timestamp("2024-01-02"),
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["cost_model"]["mode"] = "calibrated"
    config["cost_model"]["proportional_cost"] = 0.0
    config["cost_model"]["fixed_cost"] = 0.0
    config["cost_model"]["slippage"] = 0.0
    config["cost_model"]["market_impact_enabled"] = False
    config["cost_model"]["calibration"]["min_bucket_samples"] = 1
    model = CostModel(config)
    model.is_calibrated_ = True
    model.calibration_bins_ = {
        "amount": (-np.inf, 1050.0, 1500.0, 2050.0, np.inf),
        "turnover_rate": None,
        "sigma20": None,
    }
    model.calibration_tables_ = {
        "exact": {
            ("q0", "all", "all"): {"sample_count": 1, "realized_bps_median": 10.0},
            ("q1", "all", "all"): {"sample_count": 1, "realized_bps_median": 90.0},
            ("q2", "all", "all"): {"sample_count": 1, "realized_bps_median": 20.0},
            ("q3", "all", "all"): {"sample_count": 1, "realized_bps_median": 90.0},
        },
        "amount_sigma": {},
        "amount": {},
    }
    prev_weights = np.array([0.0, 1.0])
    target_weights = np.array([0.5, 0.5])

    portfolio_state = PortfolioState(
        date=state.decision_date,
        nav=1.0,
        portfolio_value=100000000.0,
        current_weights=prev_weights,
    )
    result = model.estimate(prev_weights, target_weights, state, portfolio_state)

    assert result.market_impact_cost == pytest.approx((10.0 * 0.5 + 20.0 * 0.5) / 10000.0)
    assert result.info["turnover_rate_source"] == "amount_over_adv20_at_cost_observation"


def test_sanitize_execution_returns_imputes_only_frozen_missing_returns():
    info = {}
    result = sanitize_execution_returns(
        np.array([0.02, np.nan, np.nan]),
        np.array([0.5, 0.0, 0.5]),
        np.array([True, True, False]),
        np.array(["listed", "listed", "suspended"], dtype=object),
        asset_ids=["a", "b", "c"],
        info=info,
    )

    np.testing.assert_allclose(result, np.array([0.02, 0.0, 0.0]))
    assert info["return_imputation"] == [{"asset_id": "c", "reason": "suspended", "value": 0.0}]


def test_sanitize_execution_returns_raises_for_tradeable_held_missing_returns():
    with pytest.raises(DataContractError) as error:
        sanitize_execution_returns(
            np.array([np.nan]),
            np.array([1.0]),
            np.array([True]),
            np.array(["listed"], dtype=object),
        )

    assert error.value.code == "ERR_EXECUTION_RETURN_MISSING"


def test_missing_execution_return_policy():
    info = {}
    sanitized = sanitize_execution_returns(
        np.array([0.01, np.nan, np.nan]),
        np.array([0.4, 0.0, 0.6]),
        np.array([True, True, False]),
        np.array(["listed", "listed", "missing_return"], dtype=object),
        asset_ids=["asset_a", "asset_b", "asset_c"],
        info=info,
    )
    np.testing.assert_allclose(sanitized, np.array([0.01, 0.0, 0.0]))
    assert info["return_imputation"] == [{"asset_id": "asset_c", "reason": "missing_return", "value": 0.0}]

    with pytest.raises(DataContractError) as error:
        sanitize_execution_returns(
            np.array([np.nan]),
            np.array([1.0]),
            np.array([True]),
            np.array(["listed"], dtype=object),
        )
    assert error.value.code == "ERR_EXECUTION_RETURN_MISSING"


def test_drift_weights_normalizes_clamped_gross_returns():
    result = drift_weights(np.array([0.5, 0.5]), np.array([0.10, -0.50]))

    np.testing.assert_allclose(result, np.array([0.55 / 0.80, 0.25 / 0.80]))


def test_drift_weights_zeroes_cash_return_when_enabled():
    result = drift_weights(np.array([0.5, 0.5]), np.array([0.10, 0.90]), cash_enabled=True)

    np.testing.assert_allclose(result, np.array([0.55 / 1.05, 0.50 / 1.05]))


def test_drift_weights_rejects_zero_gross_nav():
    with pytest.raises(DataContractError) as error:
        drift_weights(np.array([1.0]), np.array([-1.0]))

    assert error.value.code == "ERR_EXECUTION_INVALID_NAV"


def test_two_stage_nav_formula():
    config = deepcopy(DEFAULT_CONFIG)
    config["cost_model"]["proportional_cost"] = 0.01
    config["cost_model"]["fixed_cost"] = 0.0
    config["cost_model"]["slippage"] = 0.0
    config["cost_model"]["market_impact_enabled"] = False
    core = PortfolioExecutionCore(config)
    decision_weights = np.array([0.6, 0.4])
    target_weights = np.array([0.5, 0.5])
    execution_state = ExecutionMarketState(
        decision_date=pd.Timestamp("2024-01-02"),
        execution_date=pd.Timestamp("2024-01-03"),
        next_valuation_date=pd.Timestamp("2024-01-03"),
        execution_price_type="open",
        execution_price=np.array([11.0, 19.0]),
        tradeable_mask_at_execution=np.array([True, True]),
        availability_reason_at_execution=np.array(["listed", "listed"], dtype=object),
        return_from_decision_to_execution=np.array([0.10, -0.05]),
        holding_simple_return=np.array([0.02, 0.03]),
        amount_at_execution=np.array([1000.0, 2000.0]),
        volume_at_execution=np.array([10.0, 20.0]),
        adv20_at_execution=np.array([1000000.0, 1000000.0]),
        volatility_20d_at_execution=np.array([0.02, 0.03]),
    )
    portfolio_state = PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=100000000.0,
        current_weights=decision_weights,
    )
    pre_drift = np.array([0.6 * 1.10, 0.4 * 0.95]) / 1.04
    turnover = 0.5 * float(np.sum(np.abs(target_weights - pre_drift)))
    transaction_cost = 0.01 * turnover
    nav_execution = 1.04
    nav_after_cost = nav_execution * (1.0 - transaction_cost)
    post_execution_return = float(np.dot(target_weights, np.array([0.02, 0.03])))
    nav_next = nav_after_cost * (1.0 + post_execution_return)

    result = core.execute_step(
        decision_weights,
        target_weights,
        execution_state,
        portfolio_state,
        rebalance_action=1,
        rebalance_intensity=1.0,
    )

    np.testing.assert_allclose(result.pre_execution_drifted_weights, pre_drift)
    np.testing.assert_allclose(result.executed_weights, target_weights)
    assert result.pre_execution_return == pytest.approx(0.04)
    assert result.post_execution_return == pytest.approx(post_execution_return)
    assert result.gross_return == pytest.approx((1.0 + 0.04) * (1.0 + post_execution_return) - 1.0)
    assert result.turnover == pytest.approx(turnover)
    assert result.transaction_cost == pytest.approx(transaction_cost)
    assert result.transaction_cost_on_initial_nav == pytest.approx(nav_execution * transaction_cost)
    assert result.estimated_turnover is None
    assert result.estimated_cost is None
    assert result.realized_turnover == pytest.approx(turnover)
    assert result.realized_cost == pytest.approx(transaction_cost)
    assert result.net_return == pytest.approx(nav_next - 1.0)
    assert result.portfolio_log_return == pytest.approx(np.log(nav_next))
    assert result.nav_execution == pytest.approx(nav_execution)
    assert result.nav_after_cost == pytest.approx(nav_after_cost)
    assert result.nav_next == pytest.approx(nav_next)
    assert portfolio_state.nav == pytest.approx(nav_next)
    assert portfolio_state.portfolio_value == pytest.approx(100000000.0 * nav_next)
    np.testing.assert_allclose(portfolio_state.previous_executed_weights, target_weights)
    np.testing.assert_allclose(
        portfolio_state.current_weights,
        np.array([0.5 * 1.02, 0.5 * 1.03]) / (1.0 + post_execution_return),
    )


def test_portfolio_state_tracks_episode_max_drawdown():
    config = deepcopy(DEFAULT_CONFIG)
    config["cost_model"]["proportional_cost"] = 0.0
    config["cost_model"]["fixed_cost"] = 0.0
    config["cost_model"]["slippage"] = 0.0
    config["cost_model"]["market_impact_enabled"] = False
    core = PortfolioExecutionCore(config)
    portfolio_state = PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=1000.0,
        current_weights=np.array([1.0]),
    )
    first_state = ExecutionMarketState(
        decision_date=pd.Timestamp("2024-01-02"),
        execution_date=pd.Timestamp("2024-01-03"),
        next_valuation_date=pd.Timestamp("2024-01-03"),
        execution_price_type="open",
        execution_price=np.array([1.0]),
        tradeable_mask_at_execution=np.array([True]),
        availability_reason_at_execution=np.array(["listed"], dtype=object),
        return_from_decision_to_execution=np.array([0.0]),
        holding_simple_return=np.array([-0.10]),
        amount_at_execution=np.array([1000.0]),
        volume_at_execution=np.array([100.0]),
        adv20_at_execution=np.array([1000.0]),
        volatility_20d_at_execution=np.array([0.02]),
    )
    core.execute_step(np.array([1.0]), np.array([1.0]), first_state, portfolio_state, rebalance_action=0)
    assert portfolio_state.current_drawdown_abs == pytest.approx(0.10)
    assert portfolio_state.max_drawdown_abs == pytest.approx(0.10)

    second_state = ExecutionMarketState(
        decision_date=pd.Timestamp("2024-01-03"),
        execution_date=pd.Timestamp("2024-01-04"),
        next_valuation_date=pd.Timestamp("2024-01-04"),
        execution_price_type="open",
        execution_price=np.array([1.0]),
        tradeable_mask_at_execution=np.array([True]),
        availability_reason_at_execution=np.array(["listed"], dtype=object),
        return_from_decision_to_execution=np.array([0.0]),
        holding_simple_return=np.array([0.20]),
        amount_at_execution=np.array([1000.0]),
        volume_at_execution=np.array([100.0]),
        adv20_at_execution=np.array([1000.0]),
        volatility_20d_at_execution=np.array([0.02]),
    )
    core.execute_step(np.array([1.0]), np.array([1.0]), second_state, portfolio_state, rebalance_action=0)
    assert portfolio_state.current_drawdown_abs == pytest.approx(0.0)
    assert portfolio_state.max_drawdown_abs == pytest.approx(0.10)


def test_two_stage_nav_invalid_nav_raises():
    core = PortfolioExecutionCore(_config("next_open"))
    execution_state = ExecutionMarketState(
        decision_date=pd.Timestamp("2024-01-02"),
        execution_date=pd.Timestamp("2024-01-03"),
        next_valuation_date=pd.Timestamp("2024-01-03"),
        execution_price_type="open",
        execution_price=np.ones(1),
        tradeable_mask_at_execution=np.array([True]),
        availability_reason_at_execution=np.array(["listed"], dtype=object),
        return_from_decision_to_execution=np.array([-1.0]),
        holding_simple_return=np.array([0.0]),
        amount_at_execution=np.ones(1),
        volume_at_execution=np.ones(1),
        adv20_at_execution=np.ones(1),
        volatility_20d_at_execution=np.ones(1),
    )

    with pytest.raises(DataContractError) as error:
        core.execute_step(
            np.array([1.0]),
            np.array([1.0]),
            execution_state,
            PortfolioState(date=pd.Timestamp("2024-01-02"), nav=1.0, portfolio_value=1000.0, current_weights=np.array([1.0])),
        )

    assert error.value.code == "ERR_EXECUTION_INVALID_NAV"


def test_partial_rebalance_and_t_plus_one_freeze():
    decision_weights = np.array([0.8, 0.2])
    target_weights = np.array([0.2, 0.8])
    expected_by_policy = {
        "report_only": np.array([0.7, 0.3]),
        "project_executed": np.array([0.6, 0.4]),
        "force_full_rebalance": np.array([0.4, 0.6]),
    }
    for policy, expected in expected_by_policy.items():
        config = _execution_config()
        config["constraints"]["max_weight"] = 0.6
        config["constraints"]["partial_rebalance_post_check_policy"] = policy
        result = PortfolioExecutionCore(config).execute_step(
            decision_weights,
            target_weights,
            _execution_state_for(np.array([True, True])),
            PortfolioState(
                date=pd.Timestamp("2024-01-02"),
                nav=1.0,
                portfolio_value=100000000.0,
                current_weights=decision_weights,
            ),
            rebalance_intensity=0.25,
        )
        np.testing.assert_allclose(result.executed_weights, expected)
        assert any(
            record.get("policy") == policy and record.get("constraint") == "max_weight"
            for record in result.info["constraint_violations"]
        )

    config = _execution_config()
    result = PortfolioExecutionCore(config).execute_step(
        np.array([0.5, 0.5, 0.0]),
        np.array([0.2, 0.3, 0.5]),
        _execution_state_for(np.array([True, True, False])),
        PortfolioState(
            date=pd.Timestamp("2024-01-02"),
            nav=1.0,
            portfolio_value=100000000.0,
            current_weights=np.array([0.5, 0.5, 0.0]),
        ),
    )
    assert result.executed_weights[2] == pytest.approx(0.0)

    result = PortfolioExecutionCore(config).execute_step(
        np.array([0.2, 0.5, 0.3]),
        np.array([0.4, 0.6, 0.0]),
        _execution_state_for(np.array([True, True, False])),
        PortfolioState(
            date=pd.Timestamp("2024-01-02"),
            nav=1.0,
            portfolio_value=100000000.0,
            current_weights=np.array([0.2, 0.5, 0.3]),
        ),
    )
    assert result.executed_weights[2] == pytest.approx(0.3)

    t_plus_config = _execution_config()
    t_plus_config["execution_model"]["t_plus_one"] = True
    t_plus_config["execution_model"]["t_plus_one_position_tracking"] = "sellable_mask"
    portfolio_state = PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=100000000.0,
        current_weights=np.array([0.6, 0.4]),
        last_buy_date_per_asset=np.array([pd.Timestamp("2024-01-03"), None], dtype=object),
    )
    result = PortfolioExecutionCore(t_plus_config).execute_step(
        np.array([0.6, 0.4]),
        np.array([0.2, 0.8]),
        _execution_state_for(np.array([True, True])),
        portfolio_state,
    )

    np.testing.assert_allclose(result.executed_weights, np.array([0.6, 0.4]))
    assert portfolio_state.sellable_mask.tolist() == [False, True]
    np.testing.assert_allclose(portfolio_state.frozen_weight, np.array([0.6, 0.0]))
    assert any(record["constraint"] == "t_plus_one" for record in result.info["constraint_violations"])

    portfolio_state = PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=100000000.0,
        current_weights=np.array([0.5, 0.5, 0.0]),
        last_buy_date_per_asset=np.array([None, None, None], dtype=object),
    )
    result = PortfolioExecutionCore(t_plus_config).execute_step(
        np.array([0.5, 0.5, 0.0]),
        np.array([0.25, 0.25, 0.5]),
        _execution_state_for(np.array([True, True, True])),
        portfolio_state,
    )
    assert portfolio_state.last_buy_date_per_asset[2] == pd.Timestamp("2024-01-03")
    assert portfolio_state.sellable_mask.tolist() == [True, True, False]
    np.testing.assert_allclose(portfolio_state.frozen_weight, np.array([0.0, 0.0, result.executed_weights[2]]))


def test_next_close_delayed_returns():
    config = _config("next_close")
    config["execution_model"]["delayed_action_execution"] = True
    core = PortfolioExecutionCore(config)
    state = core.build_execution_market_state(
        _market_dataset_bundle(),
        pending_action=PendingAction(
            decision_date=pd.Timestamp("2024-01-02"),
            execution_date=pd.Timestamp("2024-01-03"),
            next_valuation_date=pd.Timestamp("2024-01-04"),
            target_weights=np.array([0.5, 0.5]),
            candidate_weights=np.array([0.5, 0.5]),
            rebalance_action=1,
            rebalance_intensity=1.0,
            execution_price="next_close",
            execution_price_type="close",
        ),
    )

    assert state.execution_date == pd.Timestamp("2024-01-03")
    assert state.next_valuation_date == pd.Timestamp("2024-01-04")
    assert state.execution_price_type == "close"
    np.testing.assert_allclose(state.execution_price, np.array([12.0, 21.0]))
    np.testing.assert_allclose(state.return_from_decision_to_execution, np.array([0.20, 0.05]))
    np.testing.assert_allclose(state.holding_simple_return, np.array([15.0 / 12.0 - 1.0, 18.0 / 21.0 - 1.0]))
    assert state.cost_observation_date == pd.Timestamp("2024-01-03")
    assert state.cost_observation_timing == "execution_observed"
    assert core.execution_manifest_flags["delayed_action_execution"] is True


def test_same_close_debug_execution_sets_idealized_flags():
    config = _config("next_open")
    config["execution_model"]["same_close_idealized_execution_enabled"] = True
    core = PortfolioExecutionCore(config)
    state = core.build_execution_market_state(_market_dataset_bundle(), pd.Timestamp("2024-01-02"))

    assert state.execution_date == pd.Timestamp("2024-01-02")
    assert state.next_valuation_date == pd.Timestamp("2024-01-03")
    assert state.execution_price_type == "close"
    np.testing.assert_allclose(state.return_from_decision_to_execution, np.zeros(2))
    np.testing.assert_allclose(state.holding_simple_return, np.array([0.20, 0.05]))
    assert core.execution_manifest_flags["idealized_execution"] is True
    assert core.execution_manifest_flags["same_close_idealized_execution_enabled"] is True


def _config(execution_price: str):
    config = deepcopy(DEFAULT_CONFIG)
    config["execution_model"]["execution_price"] = execution_price
    return config


def _execution_config():
    config = _config("next_open")
    config["cost_model"]["proportional_cost"] = 0.0
    config["cost_model"]["fixed_cost"] = 0.0
    config["cost_model"]["slippage"] = 0.0
    config["cost_model"]["market_impact_enabled"] = False
    return config


def _execution_state_for(tradeable_mask: np.ndarray):
    n_assets = len(tradeable_mask)
    return ExecutionMarketState(
        decision_date=pd.Timestamp("2024-01-02"),
        execution_date=pd.Timestamp("2024-01-03"),
        next_valuation_date=pd.Timestamp("2024-01-03"),
        execution_price_type="open",
        execution_price=np.ones(n_assets),
        tradeable_mask_at_execution=tradeable_mask,
        availability_reason_at_execution=np.where(tradeable_mask, "listed", "suspended").astype(object),
        return_from_decision_to_execution=np.zeros(n_assets),
        holding_simple_return=np.zeros(n_assets),
        amount_at_execution=np.ones(n_assets),
        volume_at_execution=np.ones(n_assets),
        adv20_at_execution=np.ones(n_assets),
        volatility_20d_at_execution=np.ones(n_assets),
    )


def _market_dataset_bundle():
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
        ]
    )
    columns = ["510300.SH", "159915.SZ"]
    close = pd.DataFrame(
        [[10.0, 20.0], [12.0, 21.0], [15.0, 18.0]],
        index=dates,
        columns=columns,
    )
    wide = {
        "open": pd.DataFrame([[9.5, 19.5], [11.0, 18.0], [12.5, 20.0]], index=dates, columns=columns),
        "close": close,
        "amount": pd.DataFrame([[1000.0, 2000.0], [1100.0, 2100.0], [1200.0, 2200.0]], index=dates, columns=columns),
        "vol": pd.DataFrame([[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]], index=dates, columns=columns),
        "log_return": np.log(close / close.shift(1)).fillna(0.0),
    }
    availability_mask = pd.DataFrame(True, index=dates, columns=columns)
    availability_reason = pd.DataFrame("listed", index=dates, columns=columns)
    return MarketDatasetBundle(
        asset_universe=pd.DataFrame({"ts_code": columns}),
        panel=pd.DataFrame(),
        wide=wide,
        metrics_features=None,
        feature_cols=[],
        auxiliary_target_cols=[],
        availability_mask=availability_mask,
        availability_reason=availability_reason,
        data_manifest={"canonical_asset_order": columns},
    )
