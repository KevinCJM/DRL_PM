from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.data.leakage_checks import assert_decision_visibility_contract
from src.data.loader import DataContractError, MarketDatasetBundle
from src.data.splits import SplitSpec
from src.envs.backtest_engine import BacktestEngine
from src.envs.portfolio_execution_core import PortfolioExecutionCore
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv
from src.envs.state import PortfolioAction


def test_gymnasium_reset_step_signature():
    env = PortfolioRebalanceEnv(
        _market_dataset_bundle(),
        _split(),
        config=_config(initial_build_cost=True),
        segment="test",
    )

    obs, info = env.reset()
    assert {
        "market_image",
        "current_weights",
        "availability_mask",
        "adv20_at_decision",
        "volatility_20d_at_decision",
        "amount_at_decision",
        "turnover_rate_at_decision",
        "portfolio_value",
    }.issubset(obs)
    assert env.observation_space.contains(obs)
    assert info["decision_date"] == pd.Timestamp("2024-01-02")

    obs, reward, terminated, truncated, info = env.step(_action())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["execution_date"] == pd.Timestamp("2024-01-03")

    env.step(_action())
    obs, reward, terminated, truncated, info = env.step(_action())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is True
    assert info["next_valuation_date"] == pd.Timestamp("2024-01-05")


def test_gymnasium_action_validation_errors():
    env = PortfolioRebalanceEnv(
        _market_dataset_bundle(),
        _split(),
        config=_config(initial_build_cost=True),
        segment="test",
    )
    env.reset()

    with pytest.raises(DataContractError) as non_finite_error:
        env.step({"weights": np.array([np.nan, 0.5]), "rebalance": 1})
    assert non_finite_error.value.code == "ERR_ACTION_NON_FINITE"

    with pytest.raises(DataContractError) as shape_error:
        env.step({"weights": np.array([0.5, 0.25, 0.25]), "rebalance": 1})
    assert shape_error.value.code == "ERR_ACTION_SHAPE_MISMATCH"


def test_execution_only_field_leakage():
    bundle = _market_dataset_bundle()
    assert_decision_visibility_contract(
        observation={"market_image": np.zeros((1, 3, 2)), "availability_mask": np.ones(2)},
        market_image=["log_return", "amount_at_decision"],
        gate_input=["close_at_decision", "available_mask_at_decision"],
        feature_window=["log_return", "volume_at_decision"],
    )

    with pytest.raises(DataContractError) as gate_error:
        assert_decision_visibility_contract(gate_input=["amount_after_execution"])
    assert gate_error.value.code == "ERR_LEAKAGE_EXECUTION_FIELD"
    assert "amount_after_execution" in str(gate_error.value)

    with pytest.raises(DataContractError) as available_t_plus_1_error:
        assert_decision_visibility_contract(gate_input=["available_mask_t_plus_1"])
    assert available_t_plus_1_error.value.code == "ERR_LEAKAGE_EXECUTION_FIELD"

    with pytest.raises(DataContractError) as available_after_execution_error:
        assert_decision_visibility_contract(gate_input=["available_mask_after_execution"])
    assert available_after_execution_error.value.code == "ERR_LEAKAGE_EXECUTION_FIELD"

    with pytest.raises(DataContractError) as env_feature_error:
        PortfolioRebalanceEnv(
            replace(bundle, feature_cols=["holding_simple_return"]),
            _split(),
            config=_config(initial_build_cost=True),
            segment="test",
        )
    assert env_feature_error.value.code == "ERR_LEAKAGE_EXECUTION_FIELD"

    with pytest.raises(DataContractError) as env_image_error:
        PortfolioRebalanceEnv(
            bundle,
            _split(),
            config=_config(initial_build_cost=True),
            segment="test",
            market_image_dataset=_LeakingMarketImageDataset(),
        )
    assert env_image_error.value.code == "ERR_LEAKAGE_EXECUTION_FIELD"

    with pytest.raises(DataContractError) as backtest_error:
        BacktestEngine(_config(initial_build_cost=True)).run(
            replace(bundle, feature_cols=["return_from_decision_to_execution"]),
            _split(),
            _FixedStrategy(),
            segment="test",
        )
    assert backtest_error.value.code == "ERR_LEAKAGE_EXECUTION_FIELD"
    assert "return_from_decision_to_execution" in str(backtest_error.value)


def test_env_and_backtest_use_injected_portfolio_execution_core():
    config = _config(initial_build_cost=True)
    core = _CountingExecutionCore(config)

    env = PortfolioRebalanceEnv(
        _market_dataset_bundle(),
        _split(),
        config=config,
        segment="test",
        execution_core=core,
    )
    env.reset()
    env.step(_action())
    env_build_calls = core.build_calls
    env_execute_calls = core.execute_calls

    assert env.execution_core is core
    assert env_build_calls == 1
    assert env_execute_calls == 1

    engine = BacktestEngine(config, execution_core=core)
    engine.run(_market_dataset_bundle(), _split(), _FixedStrategy(), segment="test")

    assert engine.execution_core is core
    assert core.build_calls > env_build_calls
    assert core.execute_calls > env_execute_calls


def test_delayed_env_queues_before_execution():
    config = _config(initial_build_cost=True)
    config["execution_model"]["execution_price"] = "next_close"
    config["execution_model"]["delayed_action_execution"] = True
    core = _CountingExecutionCore(config)
    env = PortfolioRebalanceEnv(
        _market_dataset_bundle(),
        _split(),
        config=config,
        segment="test",
        execution_core=core,
    )
    env.reset()

    _, reward, terminated, truncated, info = env.step(_action())
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["delayed_action_queued"] is True
    assert info["delayed_action_execution"] is True
    assert info["execution_price_type"] == "close"
    assert info["pending_execution_date"] == pd.Timestamp("2024-01-03")
    assert core.execute_calls == 0

    _, reward, terminated, truncated, info = env.step(_action())
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["decision_date"] == pd.Timestamp("2024-01-02")
    assert info["delayed_action_execution"] is True
    assert info["execution_price_type"] == "close"
    assert info["execution_date"] == pd.Timestamp("2024-01-03")
    assert info["next_valuation_date"] == pd.Timestamp("2024-01-04")
    assert info["pending_execution_date"] == pd.Timestamp("2024-01-04")
    assert core.execute_calls == 1


class _FixedStrategy:
    def compute_target_weights(self, decision_market_state, portfolio_state):
        return PortfolioAction(np.array([0.5, 0.5]), 1, 1.0)


class _CountingExecutionCore(PortfolioExecutionCore):
    def __init__(self, config):
        super().__init__(config)
        self.build_calls = 0
        self.execute_calls = 0

    def build_execution_market_state(self, *args, **kwargs):
        self.build_calls += 1
        return super().build_execution_market_state(*args, **kwargs)

    def execute_step(self, *args, **kwargs):
        self.execute_calls += 1
        return super().execute_step(*args, **kwargs)


class _LeakingMarketImageDataset:
    feature_cols = ["volume_t_plus_1"]

    def __getitem__(self, index):
        return np.zeros((1, 3, 2), dtype=np.float32)


def _action():
    return {"weights": np.array([0.5, 0.5]), "rebalance": 1, "rebalance_intensity": 1.0}


def _config(initial_build_cost: bool):
    config = deepcopy(DEFAULT_CONFIG)
    config["env"]["window_size"] = 3
    config["execution_model"]["execution_price"] = "next_open"
    config["execution_model"]["delayed_action_execution"] = False
    config["execution_model"]["initial_build_cost"] = initial_build_cost
    config["cost_model"]["proportional_cost"] = 0.01
    config["cost_model"]["fixed_cost"] = 0.0
    config["cost_model"]["slippage"] = 0.0
    config["cost_model"]["market_impact_enabled"] = False
    config["rebalance"]["mode"] = "daily"
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
