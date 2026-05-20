from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.baselines.base_strategy import BaseStrategy
from src.baselines.equal_weight import EqualWeightStrategy
from src.config import DEFAULT_CONFIG, ConfigLoader
from src.data.loader import DataContractError, MarketDatasetBundle, load_market_dataset
from src.data.splits import SplitSpec, create_split
from src.envs.backtest_engine import (
    DAILY_COSTS_COLUMNS,
    DAILY_REBALANCE_COLUMNS,
    DAILY_RETURNS_COLUMNS,
    DAILY_TURNOVER_COLUMNS,
    DAILY_WEIGHTS_COLUMNS,
    BacktestResult,
    BacktestEngine,
    _write_outputs,
)
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


def test_default_data_equal_weight_backtest_smoke():
    config = ConfigLoader.load("configs/main_model.yaml")
    dataset = load_market_dataset(config)
    split = create_split(dataset.wide["close"].index, config)

    result = BacktestEngine(config).run(
        dataset,
        split,
        EqualWeightStrategy(config),
        segment="test",
    )

    assert not result.daily_returns.empty
    assert not result.daily_costs.empty
    assert result.daily_returns["net_return"].notna().all()
    assert result.daily_returns["nav"].notna().all()
    assert result.metrics["n_steps"] == pytest.approx(float(len(result.daily_returns)))
    assert result.run_manifest["execution_price"] == "next_open"


class FixedWeightStrategy(BaseStrategy):
    fit_required = True

    def __init__(self, target_weights):
        super().__init__()
        self.target_weights = np.asarray(target_weights, dtype=float)
        self.fit_calls = 0
        self.compute_calls = 0
        self.fit_segments = []

    def fit(self, train_data=None, validation_data=None):
        self.fit_calls += 1
        self.fit_segments.append((train_data["segment"], validation_data["segment"]))
        return super().fit(train_data, validation_data)

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self.validate_decision_market_state(decision_market_state)
        self.validate_portfolio_state(portfolio_state)
        self.compute_calls += 1
        return PortfolioAction(
            self.target_weights,
            1,
            1.0,
            {"q_hold": 0.1, "q_rebalance": 0.2, "q_gap": 0.1},
        )


class FailingStrategy(BaseStrategy):
    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self.validate_decision_market_state(decision_market_state)
        self.validate_portfolio_state(portfolio_state)
        raise RuntimeError("optimizer_failed")


class HoldSignalStrategy(BaseStrategy):
    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self.validate_decision_market_state(decision_market_state)
        self.validate_portfolio_state(portfolio_state)
        return PortfolioAction(np.array([0.5, 0.5]), 0, 0.0)


class BadContractStrategy(BaseStrategy):
    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ):
        self.validate_decision_market_state(decision_market_state)
        self.validate_portfolio_state(portfolio_state)
        return np.array([0.5, 0.5])


def test_backtest_next_open_outputs_daily_records():
    strategy = FixedWeightStrategy([0.5, 0.5])
    result = BacktestEngine(_config(initial_build_cost=True)).run(
        _market_dataset_bundle(),
        _split(),
        strategy,
        segment="test",
    )

    assert list(result.daily_returns.columns) == DAILY_RETURNS_COLUMNS
    assert list(result.daily_weights.columns) == DAILY_WEIGHTS_COLUMNS
    assert list(result.daily_turnover.columns) == DAILY_TURNOVER_COLUMNS
    assert list(result.daily_rebalance.columns) == DAILY_REBALANCE_COLUMNS
    assert list(result.daily_costs.columns) == DAILY_COSTS_COLUMNS
    assert len(result.daily_returns) == 3
    assert len(result.daily_weights) == 6
    assert result.daily_returns["date"].tolist() == result.daily_returns["next_valuation_date"].tolist()
    assert result.daily_returns["execution_price_type"].eq("open").all()
    assert result.daily_returns["split"].eq("test").all()
    assert result.daily_returns["seed"].eq(42).all()
    assert result.daily_returns["fold_id"].eq("fixed").all()
    assert result.daily_returns.loc[0, "decision_date"] == pd.Timestamp("2024-01-02")
    assert result.daily_returns.loc[0, "execution_date"] == pd.Timestamp("2024-01-03")
    assert result.daily_turnover.loc[0, "turnover"] == 0.5
    assert result.daily_costs.loc[0, "total_transaction_cost"] > 0.0
    assert result.run_manifest["execution_price"] == "next_open"
    assert result.run_manifest["portfolio_initial_capital_currency"] == 100000000.0
    assert strategy.fit_calls == 1
    assert strategy.fit_segments == [("train", "validation")]
    assert strategy.compute_calls == 3


def test_backtest_result_default_sidecar_and_write_outputs_preserve_it(tmp_path):
    portfolio_state = PortfolioState(
        date=pd.Timestamp("2024-01-02"),
        nav=1.0,
        portfolio_value=1.0,
        current_weights=np.array([1.0]),
    )
    result = BacktestResult(
        daily_returns=pd.DataFrame(columns=DAILY_RETURNS_COLUMNS),
        daily_weights=pd.DataFrame(columns=DAILY_WEIGHTS_COLUMNS),
        daily_turnover=pd.DataFrame(columns=DAILY_TURNOVER_COLUMNS),
        daily_rebalance=pd.DataFrame(columns=DAILY_REBALANCE_COLUMNS),
        daily_costs=pd.DataFrame(columns=DAILY_COSTS_COLUMNS),
        metrics={},
        run_manifest={},
        portfolio_state=portfolio_state,
    )

    assert result.baseline_daily_diagnostics.empty

    diagnostics = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-03"),
                "decision_date": pd.Timestamp("2024-01-02"),
                "execution_date": pd.Timestamp("2024-01-03"),
                "model_name": "model",
                "paper_model_id": "model",
                "seed": 42,
                "fold_id": "fixed",
            }
        ]
    )
    result_with_sidecar = BacktestResult(
        daily_returns=result.daily_returns,
        daily_weights=result.daily_weights,
        daily_turnover=result.daily_turnover,
        daily_rebalance=result.daily_rebalance,
        daily_costs=result.daily_costs,
        metrics=result.metrics,
        run_manifest=result.run_manifest,
        portfolio_state=result.portfolio_state,
        baseline_daily_diagnostics=diagnostics,
    )

    written = _write_outputs(result_with_sidecar, tmp_path)

    pd.testing.assert_frame_equal(written.baseline_daily_diagnostics, diagnostics)


def test_backtest_forced_initial_build_records_actual_rebalance_action():
    result = BacktestEngine(_config(initial_build_cost=True)).run(
        _market_dataset_bundle(),
        _split(),
        HoldSignalStrategy(),
        segment="test",
    )

    assert result.daily_turnover.loc[0, "turnover"] == 0.5
    assert result.daily_turnover.loc[0, "rebalance_action"] == 1
    assert result.daily_rebalance.loc[0, "rebalance_action"] == 1


def test_backtest_initial_build_cost_false_waives_first_cost_only():
    result = BacktestEngine(_config(initial_build_cost=False)).run(
        _market_dataset_bundle(),
        _split(),
        FixedWeightStrategy([0.5, 0.5]),
        segment="test",
    )

    assert result.daily_turnover.loc[0, "turnover"] == 0.5
    assert result.daily_costs.loc[0, "total_transaction_cost"] == 0.0
    assert result.daily_costs.loc[1:, "total_transaction_cost"].gt(0.0).any()
    assert result.run_manifest["initial_build_cost"] is False


def test_backtest_strategy_failure_falls_back_to_equal_weight():
    result = BacktestEngine(_config(initial_build_cost=True)).run(
        _market_dataset_bundle(),
        _split(),
        FailingStrategy(),
        segment="test",
    )

    assert result.daily_rebalance.loc[0, "fallback_reason"] == "RuntimeError"
    first_weights = result.daily_weights[result.daily_weights["date"] == pd.Timestamp("2024-01-03")].set_index("asset_id")
    assert first_weights.loc["510300.SH", "weight"] == 0.5
    assert first_weights.loc["159915.SZ", "weight"] == 0.5


def test_backtest_strategy_contract_error_is_not_fallback():
    with pytest.raises(DataContractError) as error:
        BacktestEngine(_config(initial_build_cost=True)).run(
            _market_dataset_bundle(),
            _split(),
            BadContractStrategy(),
            segment="test",
        )

    assert error.value.code == "ERR_STRATEGY_ACTION_CONTRACT"


def test_pending_action_queue_alignment():
    result = BacktestEngine(_delayed_config()).run(
        _market_dataset_bundle(),
        _split(),
        FixedWeightStrategy([0.5, 0.5]),
        segment="test",
    )

    assert len(result.daily_returns) == 1
    assert result.daily_returns.loc[0, "decision_date"] == pd.Timestamp("2024-01-02")
    assert result.daily_returns.loc[0, "execution_date"] == pd.Timestamp("2024-01-03")
    assert result.daily_returns.loc[0, "next_valuation_date"] == pd.Timestamp("2024-01-04")
    assert result.daily_returns.loc[0, "date"] == pd.Timestamp("2024-01-04")
    assert result.daily_returns.loc[0, "execution_price_type"] == "close"
    assert result.daily_rebalance.loc[0, "q_hold"] == 0.1
    assert result.daily_rebalance.loc[0, "q_rebalance"] == 0.2
    assert result.daily_rebalance.loc[0, "q_gap"] == 0.1
    assert result.run_manifest["execution_price"] == "next_close"
    assert result.run_manifest["execution_price_type"] == "close"
    assert result.run_manifest["delayed_action_execution"] is True
    assert result.run_manifest["pending_action_truncation_count"] == 1
    assert result.run_manifest["pending_action_truncation_reason"] == "ERR_EXECUTION_DATE_OUT_OF_RANGE"


def _config(initial_build_cost: bool):
    config = deepcopy(DEFAULT_CONFIG)
    config["execution_model"]["execution_price"] = "next_open"
    config["execution_model"]["delayed_action_execution"] = False
    config["execution_model"]["initial_build_cost"] = initial_build_cost
    config["cost_model"]["proportional_cost"] = 0.01
    config["cost_model"]["fixed_cost"] = 0.0
    config["cost_model"]["slippage"] = 0.0
    config["cost_model"]["market_impact_enabled"] = False
    config["rebalance"]["mode"] = "daily"
    return config


def _delayed_config():
    config = _config(initial_build_cost=True)
    config["execution_model"]["execution_price"] = "next_close"
    config["execution_model"]["delayed_action_execution"] = True
    return config


def _split():
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),
        ]
    )
    return SplitSpec(
        train_dates=dates[:2],
        validation_dates=dates[1:3],
        test_dates=dates,
        fold_id="fixed",
        test_last_decision_date=pd.Timestamp("2024-01-04"),
    )


def _market_dataset_bundle():
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),
        ]
    )
    columns = ["510300.SH", "159915.SZ"]
    close = pd.DataFrame(
        [[10.0, 20.0], [12.0, 21.0], [13.0, 20.0], [14.0, 22.0]],
        index=dates,
        columns=columns,
    )
    wide = {
        "open": pd.DataFrame(
            [[9.5, 19.5], [11.0, 18.0], [12.5, 20.5], [13.5, 21.0]],
            index=dates,
            columns=columns,
        ),
        "close": close,
        "amount": pd.DataFrame(
            [[1000.0, 2000.0], [1100.0, 2100.0], [1200.0, 2200.0], [1300.0, 2300.0]],
            index=dates,
            columns=columns,
        ),
        "vol": pd.DataFrame(
            [[10.0, 20.0], [11.0, 21.0], [12.0, 22.0], [13.0, 23.0]],
            index=dates,
            columns=columns,
        ),
        "log_return": np.log(close / close.shift(1)).fillna(0.0),
        "turnover_rate": pd.DataFrame(0.01, index=dates, columns=columns),
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
        data_manifest={"canonical_asset_order": columns, "amount_is_proxy": True},
    )
