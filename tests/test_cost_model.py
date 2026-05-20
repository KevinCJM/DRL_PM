from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError
from src.envs.cost_model import CostModel
from src.envs.state import ExecutionMarketState, PortfolioState


def test_empirical_market_impact_formula():
    config = deepcopy(DEFAULT_CONFIG)
    config["cost_model"]["proportional_cost"] = 0.001
    config["cost_model"]["fixed_cost"] = 0.0001
    config["cost_model"]["slippage"] = 0.0002
    config["cost_model"]["market_impact_enabled"] = True
    config["cost_model"]["market_impact_coef"] = 0.10
    config["cost_model"]["adv_eps"] = 1000000.0
    config["cost_model"]["volatility_eps"] = 1.0e-8

    prev_weights = np.array([0.20, 0.30, 0.50])
    target_weights = np.array([0.40, 0.10, 0.50])
    portfolio_value = 100000000.0
    execution_market_state = _execution_state(
        adv20=np.array([10000000.0, 2000000.0, 5000000.0]),
        sigma20=np.array([0.02, 0.03, 0.04]),
    )
    portfolio_state = _portfolio_state(prev_weights, portfolio_value)

    result = CostModel(config).estimate(prev_weights, target_weights, execution_market_state, portfolio_state)

    trade_weight = np.abs(target_weights - prev_weights)
    turnover = 0.5 * np.sum(trade_weight)
    liquidity_ratio = trade_weight * portfolio_value / np.maximum(execution_market_state.adv20_at_execution, 1000000.0)
    per_asset_market_impact = 0.10 * trade_weight * execution_market_state.volatility_20d_at_execution * np.sqrt(liquidity_ratio)
    expected_market_impact = float(np.sum(per_asset_market_impact))
    expected_proportional = 0.001 * turnover
    expected_slippage = 0.0002 * turnover
    expected_fixed = 0.0001

    assert result.turnover == pytest.approx(turnover)
    assert result.proportional_cost == pytest.approx(expected_proportional)
    assert result.fixed_cost == pytest.approx(expected_fixed)
    assert result.slippage_cost == pytest.approx(expected_slippage)
    assert result.market_impact_cost == pytest.approx(expected_market_impact)
    assert result.total_transaction_cost == pytest.approx(
        expected_proportional + expected_fixed + expected_slippage + expected_market_impact
    )
    np.testing.assert_allclose(result.per_asset_trade_weight, trade_weight)
    np.testing.assert_allclose(result.per_asset_market_impact_cost, per_asset_market_impact)


def test_cost_model_zero_turnover_returns_all_zero():
    config = deepcopy(DEFAULT_CONFIG)
    config["cost_model"]["fixed_cost"] = 0.01
    config["cost_model"]["proportional_cost"] = 0.01
    config["cost_model"]["slippage"] = 0.01
    weights = np.array([0.25, 0.75])

    result = CostModel(config).estimate(weights, weights, _execution_state(n_assets=2), _portfolio_state(weights))

    assert result.turnover == 0.0
    assert result.total_transaction_cost == 0.0
    assert result.proportional_cost == 0.0
    assert result.fixed_cost == 0.0
    assert result.slippage_cost == 0.0
    assert result.market_impact_cost == 0.0
    np.testing.assert_array_equal(result.per_asset_trade_weight, np.zeros(2))
    np.testing.assert_array_equal(result.per_asset_market_impact_cost, np.zeros(2))


def test_market_impact_uses_adv_and_sigma_fallbacks():
    config = deepcopy(DEFAULT_CONFIG)
    config["cost_model"]["proportional_cost"] = 0.0
    config["cost_model"]["fixed_cost"] = 0.0
    config["cost_model"]["slippage"] = 0.0
    config["cost_model"]["market_impact_coef"] = 0.2
    config["cost_model"]["adv_eps"] = 1000000.0
    config["cost_model"]["volatility_eps"] = 0.05
    prev_weights = np.array([0.0, 1.0])
    target_weights = np.array([0.5, 0.5])
    execution_market_state = _execution_state(
        adv20=np.array([0.0, np.nan]),
        sigma20=np.array([np.nan, -0.1]),
    )

    result = CostModel(config).estimate(prev_weights, target_weights, execution_market_state, _portfolio_state(prev_weights))

    trade_weight = np.array([0.5, 0.5])
    expected = 0.2 * trade_weight * 0.05 * np.sqrt(trade_weight * 100000000.0 / 1000000.0)
    np.testing.assert_allclose(result.per_asset_market_impact_cost, expected)
    assert result.info["adv20_fallback_index"] == [0, 1]
    assert result.info["volatility_20d_fallback_index"] == [0, 1]


def test_market_impact_requires_portfolio_value():
    with pytest.raises(DataContractError) as error:
        CostModel(DEFAULT_CONFIG).estimate(
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
            _execution_state(n_assets=2),
            SimpleNamespace(portfolio_value=None),
        )

    assert error.value.code == "ERR_COST_PORTFOLIO_VALUE_REQUIRED"


def test_calibrated_mode_does_not_use_empirical_formula_before_fit():
    config = deepcopy(DEFAULT_CONFIG)
    config["cost_model"]["mode"] = "calibrated"
    prev_weights = np.array([0.0, 1.0])
    target_weights = np.array([0.5, 0.5])

    with pytest.raises(DataContractError) as error:
        CostModel(config).estimate(
            prev_weights=prev_weights,
            target_weights=target_weights,
            execution_market_state=_execution_state(n_assets=2),
            portfolio_state=_portfolio_state(prev_weights),
        )

    assert error.value.code == "ERR_COST_CALIBRATION_NOT_FITTED"


def test_calibrated_cost_report_schema(tmp_path):
    empirical_config = deepcopy(DEFAULT_CONFIG)
    empirical_report_path = tmp_path / "empirical_cost_calibration_report.csv"
    empirical_report = CostModel(empirical_config).fit_calibration(report_path=empirical_report_path)

    expected_columns = [
        "amount_bucket",
        "turnover_rate_bucket",
        "sigma20_bucket",
        "sample_count",
        "realized_bps_mean",
        "realized_bps_median",
        "fallback_used",
        "fallback_reason",
        "status",
    ]
    assert empirical_report.columns.tolist() == expected_columns
    assert empirical_report["status"].tolist() == ["not_applicable"]
    assert empirical_report_path.exists()

    calibrated_config = deepcopy(DEFAULT_CONFIG)
    calibrated_config["cost_model"]["mode"] = "calibrated"
    calibrated_config["cost_model"]["proportional_cost"] = 0.0
    calibrated_config["cost_model"]["fixed_cost"] = 0.0
    calibrated_config["cost_model"]["slippage"] = 0.0
    calibrated_config["cost_model"]["market_impact_enabled"] = False
    calibrated_config["cost_model"]["calibration"]["min_bucket_samples"] = 3
    train_samples = pd.DataFrame(
        {
            "amount": [1000.0, 1000.0, 1000.0, 1000.0],
            "turnover_rate": [0.5, 0.5, 0.5, 0.5],
            "sigma20": [0.02, 0.02, 0.02, 0.02],
            "realized_bps": [10.0, 12.0, 14.0, 16.0],
        }
    )
    model = CostModel(calibrated_config)
    calibrated_report_path = tmp_path / "calibrated_cost_calibration_report.csv"
    calibrated_report = model.fit_calibration(train_samples, report_path=calibrated_report_path)
    assert calibrated_report.columns.tolist() == expected_columns
    assert calibrated_report["status"].tolist() == ["fitted"]
    assert calibrated_report["sample_count"].tolist() == [4]
    assert calibrated_report_path.exists()

    prev_weights = np.array([0.0, 1.0])
    target_weights = np.array([0.5, 0.5])
    result = model.estimate(prev_weights, target_weights, _execution_state(n_assets=2), _portfolio_state(prev_weights))
    assert result.market_impact_cost == pytest.approx(2 * 13.0 * 0.5 / 10000.0)
    assert result.info["calibration_fallback_used"] is False

    fallback_config = deepcopy(calibrated_config)
    fallback_config["cost_model"]["calibration"]["min_bucket_samples"] = 10
    fallback_model = CostModel(fallback_config)
    fallback_report = fallback_model.fit_calibration(train_samples, report_path=tmp_path / "fallback_cost_calibration_report.csv")
    assert fallback_report["status"].tolist() == ["insufficient_sample"]
    assert fallback_report["fallback_reason"].tolist() == ["sample_count_below_min_bucket_samples"]
    fallback_result = fallback_model.estimate(prev_weights, target_weights, _execution_state(n_assets=2), _portfolio_state(prev_weights))
    assert fallback_result.market_impact_cost == 0.0
    assert fallback_result.info["calibration_fallback_reason"] == ["empirical_default", "empirical_default"]


def test_calibrated_bucket_uses_market_turnover_rate_not_trade_weight():
    config = deepcopy(DEFAULT_CONFIG)
    config["cost_model"]["mode"] = "calibrated"
    config["cost_model"]["proportional_cost"] = 0.0
    config["cost_model"]["fixed_cost"] = 0.0
    config["cost_model"]["slippage"] = 0.0
    config["cost_model"]["market_impact_enabled"] = False
    config["cost_model"]["calibration"]["min_bucket_samples"] = 2
    train_samples = pd.DataFrame(
        {
            "amount": [1000.0] * 6,
            "turnover_rate": [0.01, 0.02, 0.03, 0.40, 0.50, 0.60],
            "sigma20": [0.02] * 6,
            "realized_bps": [10.0, 10.0, 20.0, 20.0, 90.0, 90.0],
        }
    )
    model = CostModel(config)
    model.fit_calibration(train_samples)
    prev_weights = np.array([0.0, 1.0])
    target_weights = np.array([0.5, 0.5])
    execution_state = _execution_state(n_assets=2)
    execution_state.turnover_rate_at_execution = np.array([0.015, 0.015])

    result = model.estimate(prev_weights, target_weights, execution_state, _portfolio_state(prev_weights))

    assert result.market_impact_cost == pytest.approx(2 * 10.0 * 0.5 / 10000.0)
    assert result.info["turnover_rate_source"] == "turnover_rate_at_execution"
    assert result.info["calibration_fallback_used"] is False


def _execution_state(
    n_assets: int = 3,
    *,
    adv20: np.ndarray | None = None,
    sigma20: np.ndarray | None = None,
) -> ExecutionMarketState:
    if adv20 is not None:
        n_assets = len(adv20)
    if sigma20 is not None:
        n_assets = len(sigma20)
    date = pd.Timestamp("2024-01-02")
    values = np.ones(n_assets, dtype=float)
    return ExecutionMarketState(
        decision_date=date,
        execution_date=date + pd.Timedelta(days=1),
        next_valuation_date=date + pd.Timedelta(days=1),
        execution_price_type="open",
        execution_price=values,
        tradeable_mask_at_execution=np.ones(n_assets, dtype=bool),
        availability_reason_at_execution=None,
        return_from_decision_to_execution=np.zeros(n_assets, dtype=float),
        holding_simple_return=np.zeros(n_assets, dtype=float),
        amount_at_execution=values * 1000.0,
        volume_at_execution=values * 100.0,
        adv20_at_execution=values * 10000000.0 if adv20 is None else adv20,
        volatility_20d_at_execution=values * 0.02 if sigma20 is None else sigma20,
    )


def _portfolio_state(weights: np.ndarray, portfolio_value: float = 100000000.0) -> PortfolioState:
    return PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=portfolio_value,
        current_weights=weights,
    )
