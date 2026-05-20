from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    BuyAndHoldStrategy,
    EqualWeightStrategy,
    FixedRatioStrategy,
    HRPStrategy,
    InverseVolatilityStrategy,
    MarkowitzMaxSharpeStrategy,
    MarkowitzMeanVarianceStrategy,
    MarkowitzMinVarianceStrategy,
    MarkowitzStrategy,
    MinimumDrawdownStrategy,
    MomentumStrategy,
    NativeBernoulliGatedPPOBaselineStrategy,
    NativeCNNPPOBaselineStrategy,
    NativeDQNTemplateStrategy,
    NativeEIIEStrategy,
    NativePPOBaselineStrategy,
    PGPortfolioEIIEStrategy,
    RiskEvaluationStrategy,
    RiskParityStrategy,
)
from src.config import DEFAULT_CONFIG
from src.data.loader import MarketDatasetBundle
from src.data.splits import SplitSpec
from src.envs.backtest_engine import BacktestEngine
from src.envs.state import PortfolioAction


REQUIRED_DAILY_RETURNS_COLUMNS = {
    "date",
    "decision_date",
    "execution_date",
    "execution_price_type",
    "next_valuation_date",
    "split",
    "seed",
    "fold_id",
    "model_name",
    "pre_execution_return",
    "post_execution_return",
    "gross_return",
    "transaction_cost",
    "transaction_cost_on_initial_nav",
    "net_return",
    "portfolio_log_return",
    "nav",
}
REQUIRED_DAILY_WEIGHTS_COLUMNS = {"date", "split", "seed", "fold_id", "model_name", "asset_id", "weight"}
REQUIRED_DAILY_TURNOVER_COLUMNS = {
    "date",
    "decision_date",
    "execution_date",
    "execution_price_type",
    "next_valuation_date",
    "split",
    "seed",
    "fold_id",
    "model_name",
    "turnover",
    "rebalance_action",
    "rebalance_intensity",
    "average_holding_period",
}
REQUIRED_DAILY_REBALANCE_COLUMNS = {
    "date",
    "decision_date",
    "execution_date",
    "execution_price_type",
    "next_valuation_date",
    "split",
    "seed",
    "fold_id",
    "model_name",
    "rebalance_action",
    "rebalance_intensity",
    "estimated_turnover",
    "realized_turnover",
    "turnover",
    "estimated_cost",
    "realized_cost",
    "q_hold",
    "q_rebalance",
    "q_gap",
}
REQUIRED_DAILY_COSTS_COLUMNS = {
    "date",
    "decision_date",
    "execution_date",
    "execution_price_type",
    "next_valuation_date",
    "split",
    "seed",
    "fold_id",
    "model_name",
    "proportional_cost",
    "fixed_cost",
    "slippage_cost",
    "market_impact_cost",
    "total_transaction_cost",
    "estimated_cost",
    "realized_cost",
    "turnover",
}
REQUIRED_ARTIFACT_COLUMNS = {
    "daily_returns": REQUIRED_DAILY_RETURNS_COLUMNS,
    "daily_weights": REQUIRED_DAILY_WEIGHTS_COLUMNS,
    "daily_turnover": REQUIRED_DAILY_TURNOVER_COLUMNS,
    "daily_rebalance": REQUIRED_DAILY_REBALANCE_COLUMNS,
    "daily_costs": REQUIRED_DAILY_COSTS_COLUMNS,
}
TRADITIONAL_MODEL_NAMES = [
    "fixed_ratio",
    "equal_weight",
    "buy_and_hold",
    "markowitz",
    "traditional_markowitz_mean_variance",
    "markowitz_min_variance",
    "markowitz_max_sharpe",
    "risk_parity",
    "inverse_volatility",
    "minimum_drawdown",
    "risk_evaluation",
    "hrp",
    "momentum",
]
DEEP_MODEL_NAMES = [
    "ppo_baseline",
    "cnn_ppo_baseline",
    "bernoulli_gated_ppo",
    "dqn_only",
    "eiie",
]


def test_traditional_baseline_output_schema(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    strategies = _traditional_strategies(dataset.data_manifest["canonical_asset_order"])
    assert [model_name for model_name, _ in strategies] == TRADITIONAL_MODEL_NAMES

    for model_name, strategy in strategies:
        result = BacktestEngine(config).run(
            dataset,
            split,
            strategy,
            segment="test",
            output_dir=tmp_path / model_name,
        )

        _assert_required_columns(result.daily_returns, REQUIRED_DAILY_RETURNS_COLUMNS)
        _assert_required_columns(result.daily_weights, REQUIRED_DAILY_WEIGHTS_COLUMNS)
        _assert_required_columns(result.daily_turnover, REQUIRED_DAILY_TURNOVER_COLUMNS)
        _assert_required_columns(result.daily_rebalance, REQUIRED_DAILY_REBALANCE_COLUMNS)
        _assert_required_columns(result.daily_costs, REQUIRED_DAILY_COSTS_COLUMNS)
        assert set(result.artifact_paths) == {
            "daily_returns",
            "daily_weights",
            "daily_turnover",
            "daily_rebalance",
            "daily_costs",
        }
        for artifact_name, path in result.artifact_paths.items():
            assert path.exists()
            persisted = pd.read_csv(path)
            _assert_required_columns(persisted, REQUIRED_ARTIFACT_COLUMNS[artifact_name])

        for frame in (
            result.daily_returns,
            result.daily_weights,
            result.daily_turnover,
            result.daily_rebalance,
            result.daily_costs,
        ):
            assert not frame.empty
            assert frame["split"].eq("test").all()
            assert frame["seed"].eq(7).all()
            assert frame["fold_id"].eq("fixed").all()
            assert frame["model_name"].eq(model_name).all()

        for frame in (
            result.daily_returns,
            result.daily_turnover,
            result.daily_rebalance,
            result.daily_costs,
        ):
            assert frame["date"].tolist() == frame["next_valuation_date"].tolist()
        assert result.daily_returns["execution_price_type"].eq("open").all()
        assert result.daily_weights.groupby("date")["asset_id"].nunique().eq(4).all()
        np.testing.assert_allclose(
            result.daily_weights.groupby("date")["weight"].sum().to_numpy(dtype=float),
            np.ones(result.daily_weights["date"].nunique(), dtype=float),
            atol=1.0e-8,
        )


def test_deep_baseline_output_schema(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    strategies = _deep_strategies(dataset.data_manifest["canonical_asset_order"])
    assert [model_name for model_name, _ in strategies] == DEEP_MODEL_NAMES

    for model_name, strategy in strategies:
        assert getattr(strategy, "fit_required", False) is True
        result = BacktestEngine(config).run(
            dataset,
            split,
            strategy,
            segment="test",
            output_dir=tmp_path / model_name,
        )
        assert strategy.is_fitted is True
        training_result = getattr(strategy, "training_result", None)
        assert training_result is not None
        assert training_result["status"] == "completed"
        assert training_result["training_algorithm"] == "supervised_execution_aligned_proxy"
        assert training_result["baseline_family"] == "neural_proxy"
        assert training_result["rl_training"] is False
        assert training_result["platform_native_rl_training"] is False
        assert training_result["native_rl_training"] is False
        assert training_result["proxy_training"] is True
        assert training_result["external_original_implementation"] is False
        assert training_result["rankable_in_unified_table"] is False
        assert training_result["configured_current_weight_mode"] == "rolling_equal_weight"
        assert training_result["effective_current_weight_mode"] == "rolling_equal_weight_proxy"
        assert training_result["current_weight_mode"] == "rolling_equal_weight_proxy"
        assert training_result["execution_path_proxy"] is True
        assert training_result["pending_action_queue_simulated"] is False
        assert training_result["sample_count"] > 0
        if model_name == "bernoulli_gated_ppo":
            assert strategy.gate_training_result["status"] == "completed"
            assert strategy.gate_training_result["configured_current_weight_mode"] == "rolling_equal_weight"
            assert strategy.gate_training_result["effective_current_weight_mode"] == "rolling_equal_weight_proxy"
            assert strategy.gate_training_result["sample_count"] > 0

        _assert_required_columns(result.daily_returns, REQUIRED_DAILY_RETURNS_COLUMNS)
        _assert_required_columns(result.daily_weights, REQUIRED_DAILY_WEIGHTS_COLUMNS)
        _assert_required_columns(result.daily_turnover, REQUIRED_DAILY_TURNOVER_COLUMNS)
        _assert_required_columns(result.daily_rebalance, REQUIRED_DAILY_REBALANCE_COLUMNS)
        _assert_required_columns(result.daily_costs, REQUIRED_DAILY_COSTS_COLUMNS)
        assert set(result.artifact_paths) == {
            "daily_returns",
            "daily_weights",
            "daily_turnover",
            "daily_rebalance",
            "daily_costs",
        }
        for artifact_name, path in result.artifact_paths.items():
            assert path.exists()
            persisted = pd.read_csv(path)
            _assert_required_columns(persisted, REQUIRED_ARTIFACT_COLUMNS[artifact_name])

        for frame in (
            result.daily_returns,
            result.daily_weights,
            result.daily_turnover,
            result.daily_rebalance,
            result.daily_costs,
        ):
            assert not frame.empty
            assert frame["split"].eq("test").all()
            assert frame["seed"].eq(7).all()
            assert frame["fold_id"].eq("fixed").all()
            assert frame["model_name"].eq(model_name).all()

        for frame in (
            result.daily_returns,
            result.daily_turnover,
            result.daily_rebalance,
            result.daily_costs,
        ):
            assert frame["date"].tolist() == frame["next_valuation_date"].tolist()
        assert result.daily_returns["execution_price_type"].eq("open").all()
        assert result.daily_weights.groupby("date")["asset_id"].nunique().eq(4).all()
        np.testing.assert_allclose(
            result.daily_weights.groupby("date")["weight"].sum().to_numpy(dtype=float),
            np.ones(result.daily_weights["date"].nunique(), dtype=float),
            atol=1.0e-8,
        )


def test_backtest_engine_fails_when_native_training_status_failed():
    class FailingStrategy:
        fit_required = True
        strategy_name = "failing_native"
        is_fitted = False
        training_result = None

        def fit(self, train_data=None, validation_data=None):
            self.training_result = {"status": "failed_no_finite_validation_metric"}
            self.is_fitted = False
            return self

        def compute_target_weights(self, decision_market_state, portfolio_state):
            return PortfolioAction(portfolio_state.current_weights.copy(), 0, 0.0, {})

    with pytest.raises(Exception, match="ERR_STRATEGY_TRAINING_FAILED"):
        BacktestEngine(_config()).run(_market_dataset_bundle(), _split(), FailingStrategy(), segment="test")


def test_native_ppo_updates_parameters_and_writes_daily_outputs(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    config.update(
        {
            "n_assets": 4,
            "n_features": 4,
            "window_size": 4,
            "latent_dim": 16,
            "encoder": {"type": "mlp"},
            "ppo": {"rollout_steps": 3, "minibatch_size": 2, "update_epochs": 1},
            "baselines": {"native_rl": {"epochs": 1}},
            "baseline_run_dir": str(tmp_path / "native_ppo"),
        }
    )
    strategy = NativePPOBaselineStrategy(config)
    before = [parameter.detach().clone() for parameter in strategy.agent.actor.parameters()]

    result = BacktestEngine(config).run(
        dataset,
        split,
        strategy,
        segment="test",
        output_dir=tmp_path / "ppo_native_outputs",
    )

    assert strategy.is_fitted is True
    assert strategy.training_result is not None
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["training_algorithm"] == "ppo_clipped_gae"
    assert strategy.training_result["rl_training"] is True
    assert strategy.training_result["platform_native_rl_training"] is True
    assert strategy.training_result["proxy_training"] is False
    assert strategy.training_result["evaluated_checkpoint_path"] == strategy.training_result["checkpoint_best_path"]
    assert (tmp_path / "native_ppo" / "checkpoints" / "ppo_native" / "best.pt").exists()
    assert not result.daily_returns.empty
    assert result.daily_returns["model_name"].eq("ppo_native").all()
    after = list(strategy.agent.actor.parameters())
    assert any(not np.allclose(left.cpu().numpy(), right.detach().cpu().numpy()) for left, right in zip(before, after))


def test_native_cnn_ppo_uses_cnn_encoder(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    config.update(
        {
            "n_assets": 4,
            "n_features": 4,
            "window_size": 4,
            "latent_dim": 16,
            "encoder": {"type": "cnn", "cnn_channels": [4]},
            "ppo": {"rollout_steps": 3, "minibatch_size": 2, "update_epochs": 1},
            "baselines": {"native_rl": {"epochs": 1}},
            "baseline_run_dir": str(tmp_path / "native_cnn_ppo"),
        }
    )

    strategy = NativeCNNPPOBaselineStrategy(config)
    result = BacktestEngine(config).run(dataset, split, strategy, segment="test")

    assert strategy.training_result is not None
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["platform_native_rl_training"] is True
    assert result.daily_returns["model_name"].eq("cnn_ppo_native").all()


def test_bernoulli_gate_on_policy_log_prob_has_gradient(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    config.update(
        {
            "n_assets": 4,
            "n_features": 4,
            "window_size": 4,
            "latent_dim": 16,
            "encoder": {"type": "cnn", "cnn_channels": [4]},
            "ppo": {"clip_ratio": 0.2, "entropy_coef": 0.01, "value_coef": 0.5},
            "baselines": {"native_rl": {"epochs": 1}},
            "baseline_run_dir": str(tmp_path / "native_bernoulli"),
        }
    )
    strategy = NativeBernoulliGatedPPOBaselineStrategy(config)
    before = [parameter.detach().clone() for parameter in strategy.gate.parameters()]

    result = BacktestEngine(config).run(dataset, split, strategy, segment="test")

    assert strategy.is_fitted is True
    assert strategy.training_result is not None
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["training_algorithm"] == "bernoulli_gated_ppo_on_policy"
    assert strategy.training_result["gate_training"] == "on_policy_bernoulli"
    assert strategy.training_result["platform_native_rl_training"] is True
    assert strategy.training_history["gate_grad_norm"].max() > 0.0
    assert {"p_rebalance_mean", "gate_entropy", "rebalance_frequency"}.issubset(strategy.training_history.columns)
    assert (tmp_path / "native_bernoulli" / "checkpoints" / "bernoulli_gated_ppo_native" / "best.pt").exists()
    assert result.daily_returns["model_name"].eq("bernoulli_gated_ppo_native").all()
    after = list(strategy.gate.parameters())
    assert any(not np.allclose(left.cpu().numpy(), right.detach().cpu().numpy()) for left, right in zip(before, after))


def test_dqn_template_native_uses_replay_target_double_dqn(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    config.update(
        {
            "n_assets": 4,
            "n_features": 4,
            "window_size": 4,
            "latent_dim": 16,
            "encoder": {"type": "mlp"},
            "dqn": {
                "batch_size": 2,
                "warmup_steps": 1,
                "target_update_interval": 1,
                "double_dqn": True,
                "use_prioritized_replay": False,
                "use_n_step": False,
            },
            "baselines": {"native_rl": {"epochs": 1}},
            "baseline_run_dir": str(tmp_path / "native_dqn"),
        }
    )
    strategy = NativeDQNTemplateStrategy(config)
    before = [parameter.detach().clone() for parameter in strategy.agent.online_network.parameters()]

    result = BacktestEngine(config).run(dataset, split, strategy, segment="test")

    assert strategy.is_fitted is True
    assert strategy.training_result is not None
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["training_algorithm"] == "double_dqn_template_selector"
    assert strategy.training_result["platform_native_rl_training"] is True
    assert strategy.training_result["gradient_updates"] > 0
    assert len(strategy.agent.replay_buffer) > 0
    assert (tmp_path / "native_dqn" / "checkpoints" / "dqn_template_native" / "best.pt").exists()
    assert result.daily_returns["model_name"].eq("dqn_template_native").all()
    after = list(strategy.agent.online_network.parameters())
    assert any(not np.allclose(left.cpu().numpy(), right.detach().cpu().numpy()) for left, right in zip(before, after))


def test_native_ppo_honors_native_rl_step_budgets(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    config.update(
        {
            "n_assets": 4,
            "n_features": 4,
            "window_size": 4,
            "latent_dim": 16,
            "encoder": {"type": "mlp"},
            "baselines": {"native_rl": {"epochs": 1, "max_train_steps": 2, "max_validation_steps": 2}},
            "ppo": {"rollout_steps": 8, "minibatch_size": 2, "update_epochs": 1},
            "baseline_run_dir": str(tmp_path / "native_ppo_budgeted"),
        }
    )
    strategy = NativePPOBaselineStrategy(config)

    BacktestEngine(config).run(dataset, split, strategy, segment="test")

    assert strategy.training_result is not None
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["env_steps"] == 2
    assert strategy.training_result["max_train_steps"] == 2
    assert strategy.training_result["max_validation_steps"] == 2
    assert strategy.training_history["max_train_steps"].iloc[0] == 2
    assert strategy.training_history["max_validation_steps"].iloc[0] == 2


def test_eiie_native_honors_native_rl_step_budgets(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    config.update(
        {
            "n_assets": 4,
            "n_features": 4,
            "window_size": 4,
            "baselines": {"native_rl": {"epochs": 1, "max_train_steps": 2, "max_validation_steps": 2}},
            "eiie_native": {"learning_rate": 0.01, "turnover_penalty": 0.01},
            "baseline_run_dir": str(tmp_path / "native_eiie_budgeted"),
        }
    )
    strategy = NativeEIIEStrategy(config)

    BacktestEngine(config).run(dataset, split, strategy, segment="test")

    assert strategy.training_result is not None
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["env_steps"] == 2
    assert strategy.training_result["gradient_updates"] == 2
    assert strategy.training_result["max_train_steps"] == 2
    assert strategy.training_result["max_validation_steps"] == 2


def test_eiie_native_trains_with_pvm_and_writes_daily_outputs(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    config.update(
        {
            "n_assets": 4,
            "n_features": 4,
            "window_size": 4,
            "baselines": {"native_rl": {"epochs": 1}},
            "eiie_native": {"learning_rate": 0.01, "turnover_penalty": 0.01},
            "baseline_run_dir": str(tmp_path / "native_eiie"),
        }
    )
    strategy = NativeEIIEStrategy(config)
    before = [parameter.detach().clone() for parameter in strategy.evaluator.parameters()]

    result = BacktestEngine(config).run(dataset, split, strategy, segment="test")

    assert strategy.is_fitted is True
    assert strategy.training_result is not None
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["training_algorithm"] == "eiie_policy_gradient_pvm"
    assert strategy.training_result["portfolio_vector_memory"] is True
    assert strategy.training_result["pre_execution_return_in_actor_loss"] is False
    assert strategy.training_result["pre_execution_return_used_for_drift"] is True
    assert strategy.training_result["pre_execution_return_in_observation"] is False
    assert strategy.training_result["platform_native_rl_training"] is True
    assert (tmp_path / "native_eiie" / "checkpoints" / "eiie_native" / "best.pt").exists()
    assert result.daily_returns["model_name"].eq("eiie_native").all()
    after = list(strategy.evaluator.parameters())
    assert any(not np.allclose(left.cpu().numpy(), right.detach().cpu().numpy()) for left, right in zip(before, after))


def test_pgportfolio_eiie_osbl_samples_train_only(tmp_path):
    dataset = _market_dataset_bundle()
    split = _split()
    config = _config()
    config.update(
        {
            "n_assets": 4,
            "n_features": 4,
            "window_size": 4,
            "baselines": {"native_rl": {"epochs": 1}},
            "optimizer": {"learning_rate": 0.01},
            "pgportfolio_eiie_native": {
                "osbl_batch_size": 2,
                "osbl_batches_per_epoch": 2,
                "turnover_penalty": 0.01,
                "seed": 11,
            },
            "baseline_run_dir": str(tmp_path / "pgportfolio_eiie"),
        }
    )
    strategy = PGPortfolioEIIEStrategy(config)
    before = [parameter.detach().clone() for parameter in strategy.evaluator.parameters()]

    result = BacktestEngine(config).run(dataset, split, strategy, segment="test")

    assert strategy.is_fitted is True
    assert strategy.training_result is not None
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["training_algorithm"] == "pgportfolio_eiie_osbl"
    assert strategy.training_result["portfolio_vector_memory"] is True
    assert strategy.training_result["online_stochastic_batch_learning"] is True
    assert strategy.training_result["clean_room_reimplementation"] is True
    assert strategy.training_result["source_code_vendored"] is False
    assert strategy.training_result["platform_native_rl_training"] is True
    assert strategy.training_result["osbl_sample_count"] == 4
    assert strategy.training_history["online_stochastic_batch_learning"].eq(True).all()
    assert strategy.training_history["osbl_batch_count"].iloc[0] == 2
    assert (tmp_path / "pgportfolio_eiie" / "checkpoints" / "pgportfolio_eiie_native" / "best.pt").exists()
    assert result.daily_returns["model_name"].eq("pgportfolio_eiie_native").all()
    assert set(pd.to_datetime(strategy.osbl_sampled_dates)).issubset(set(pd.to_datetime(split.train_dates)))
    assert strategy.pvm_update_trace
    assert {record["date"] for record in strategy.pvm_update_trace}.issubset(set(pd.to_datetime(split.train_dates)))
    after = list(strategy.evaluator.parameters())
    assert any(not np.allclose(left.cpu().numpy(), right.detach().cpu().numpy()) for left, right in zip(before, after))


def _deep_strategies(asset_ids: list[str]):
    from src.baselines.bernoulli_gated_ppo import BernoulliGatedPPOStrategy
    from src.baselines.cnn_ppo_baseline import CNNPPOBaselineStrategy
    from src.baselines.dqn_only import DQNOnlyStrategy
    from src.baselines.eiie import EIIEStrategy
    from src.baselines.ppo_baseline import PPOBaselineStrategy

    n_assets = len(asset_ids)
    n_features = 4  # Match _market_dataset_bundle
    window_size = 4

    base_config = {
        "n_assets": n_assets,
        "n_features": n_features,
        "window_size": window_size,
        "latent_dim": 64,
        "encoder": {"type": "cnn"},
        "ppo": {"enabled": True},
        "dqn": {"enabled": True},
        "dqn_only": {"templates": ["hold", "equal_weight"]},
    }

    return [
        ("ppo_baseline", PPOBaselineStrategy(base_config)),
        ("cnn_ppo_baseline", CNNPPOBaselineStrategy(base_config)),
        ("bernoulli_gated_ppo", BernoulliGatedPPOStrategy(base_config)),
        ("dqn_only", DQNOnlyStrategy(base_config)),
        ("eiie", EIIEStrategy(base_config)),
    ]

def _traditional_strategies(asset_ids: list[str]):
    return [
        (
            "fixed_ratio",
            FixedRatioStrategy(
                {
                    "fixed_ratio": {
                        "asset_ids": asset_ids,
                        "asset_weights": {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
                    }
                }
            ),
        ),
        ("equal_weight", EqualWeightStrategy()),
        ("buy_and_hold", BuyAndHoldStrategy({"buy_and_hold": {"initial_weight_mode": "equal_weight"}})),
        ("markowitz", MarkowitzStrategy({"markowitz": {"lookback_window": 4, "covariance_shrinkage": 0.2}})),
        (
            "traditional_markowitz_mean_variance",
            MarkowitzMeanVarianceStrategy({"markowitz": {"lookback_window": 4, "covariance_shrinkage": 0.2}}),
        ),
        (
            "markowitz_min_variance",
            MarkowitzMinVarianceStrategy({"markowitz": {"lookback_window": 4, "covariance_shrinkage": 0.2}}),
        ),
        (
            "markowitz_max_sharpe",
            MarkowitzMaxSharpeStrategy({"markowitz": {"lookback_window": 4, "covariance_shrinkage": 0.2}}),
        ),
        (
            "risk_parity",
            RiskParityStrategy(
                {"risk_parity": {"lookback_window": 4, "covariance_shrinkage": 0.2, "volatility_floor": 1.0e-4}}
            ),
        ),
        ("inverse_volatility", InverseVolatilityStrategy({"inverse_volatility": {"volatility_floor": 1.0e-4}})),
        ("minimum_drawdown", MinimumDrawdownStrategy({"minimum_drawdown": {"lookback_window": 4}})),
        ("risk_evaluation", RiskEvaluationStrategy({"risk_evaluation": {"lookback_window": 4}})),
        ("hrp", HRPStrategy({"hrp": {"lookback_window": 4, "covariance_shrinkage": 0.2}})),
        (
            "momentum",
            MomentumStrategy(
                {"momentum": {"lookback_window": 4, "top_k": 2, "threshold": -1.0, "weight_mode": "equal"}}
            ),
        ),
    ]


def _assert_required_columns(frame: pd.DataFrame, required_columns: set[str]) -> None:
    assert required_columns.issubset(set(frame.columns))


def _config():
    config = deepcopy(DEFAULT_CONFIG)
    config["experiment"]["type"] = "baseline_comparison"
    config["execution_model"]["execution_price"] = "next_open"
    config["execution_model"]["delayed_action_execution"] = False
    config["execution_model"]["initial_build_cost"] = True
    config["cost_model"]["proportional_cost"] = 0.0
    config["cost_model"]["fixed_cost"] = 0.0
    config["cost_model"]["slippage"] = 0.0
    config["cost_model"]["market_impact_enabled"] = False
    config["rebalance"]["mode"] = "daily"
    config["env"]["window_size"] = 4
    config["reproducibility"]["seed"] = 7
    return config


def _split():
    dates = _dates()
    return SplitSpec(
        train_dates=dates[:5],
        validation_dates=dates[4:7],
        test_dates=dates[4:],
        fold_id="fixed",
        test_last_decision_date=dates[-2],
    )


def _market_dataset_bundle():
    dates = _dates()
    columns = ["A", "B", "C", "D"]
    log_returns = pd.DataFrame(
        [
            [0.000, 0.000, 0.000, 0.000],
            [0.010, 0.003, -0.002, 0.006],
            [0.012, 0.004, -0.001, 0.005],
            [-0.004, 0.005, 0.002, 0.004],
            [0.013, 0.006, -0.002, 0.007],
            [0.011, 0.003, 0.001, 0.006],
            [-0.003, 0.004, 0.002, 0.005],
            [0.014, 0.005, -0.001, 0.007],
            [0.010, 0.006, 0.002, 0.006],
        ],
        index=dates,
        columns=columns,
    )
    close = 10.0 * np.exp(log_returns.cumsum())
    open_price = close.shift(1).fillna(close.iloc[0]) * 1.001
    amount = pd.DataFrame(
        np.array(
            [
                [1000.0, 1500.0, 2000.0, 2500.0],
                [1100.0, 1510.0, 2010.0, 2510.0],
                [1200.0, 1520.0, 2020.0, 2520.0],
                [1300.0, 1530.0, 2030.0, 2530.0],
                [1400.0, 1540.0, 2040.0, 2540.0],
                [1500.0, 1550.0, 2050.0, 2550.0],
                [1600.0, 1560.0, 2060.0, 2560.0],
                [1700.0, 1570.0, 2070.0, 2570.0],
                [1800.0, 1580.0, 2080.0, 2580.0],
            ],
            dtype=float,
        ),
        index=dates,
        columns=columns,
    )
    wide = {
        "open": open_price,
        "close": close,
        "amount": amount,
        "vol": amount / 100.0,
        "log_return": log_returns,
        "turnover_rate": pd.DataFrame(
            np.linspace(0.01, 0.04, len(dates) * len(columns)).reshape(len(dates), len(columns)),
            index=dates,
            columns=columns,
        ),
    }
    availability_mask = pd.DataFrame(True, index=dates, columns=columns)
    availability_reason = pd.DataFrame("listed", index=dates, columns=columns)
    return MarketDatasetBundle(
        asset_universe=pd.DataFrame({"ts_code": columns}),
        panel=pd.DataFrame(),
        wide=wide,
        metrics_features=None,
        feature_cols=["open", "close", "amount", "vol"],
        auxiliary_target_cols=[],
        availability_mask=availability_mask,
        availability_reason=availability_reason,
        data_manifest={"canonical_asset_order": columns, "amount_is_proxy": True},
    )


def _dates() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=9, freq="B")
