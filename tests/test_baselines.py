import numpy as np
import pandas as pd
import pytest
import torch
from copy import deepcopy
from src.data.loader import MarketDatasetBundle
from src.envs.state import DecisionMarketState, PortfolioState, PortfolioAction

def _mock_decision_market_state(n_assets, n_features, window_size):
    return DecisionMarketState(
        decision_date=pd.Timestamp("2024-01-01"),
        available_mask_at_decision=np.ones(n_assets, dtype=bool),
        availability_reason_at_decision=np.array(["listed"] * n_assets),
        close_at_decision=np.ones(n_assets),
        log_return_at_decision=np.zeros(n_assets),
        log_return_window=np.zeros((window_size, n_assets)),
        amount_at_decision=np.ones(n_assets) * 1000,
        volume_at_decision=np.ones(n_assets) * 10,
        adv20_at_decision=np.ones(n_assets) * 1000,
        volatility_20d_at_decision=np.ones(n_assets) * 0.02,
        turnover_rate_at_decision=np.ones(n_assets) * 0.01,
        feature_window=np.zeros((n_features, window_size, n_assets)),
        market_image=np.zeros((n_features, window_size, n_assets))
    )

def _mock_portfolio_state(n_assets):
    return PortfolioState(
        date=pd.Timestamp("2024-01-01"),
        nav=1.0,
        portfolio_value=1e8,
        current_weights=np.ones(n_assets) / n_assets
    )

def test_ppo_baseline_action_contract():
    from src.baselines.ppo_baseline import PPOBaselineStrategy
    from src.baselines.cnn_ppo_baseline import CNNPPOBaselineStrategy
    from src.config import DEFAULT_CONFIG
    n_assets = 10
    n_features = 5
    window_size = 20
    latent_dim = 64

    config = {
        "n_assets": n_assets,
        "n_features": n_features,
        "window_size": window_size,
        "latent_dim": latent_dim,
        "encoder": {"type": "mlp"},
        "ppo": {"enabled": True}
    }

    strategy = PPOBaselineStrategy(config)
    assert not hasattr(strategy.model, "gate")
    assert strategy.model.uses_fixed_mlp_state is True
    assert strategy.model.encoder.input_dim == n_features * window_size * n_assets + 2 * n_assets

    # Mock states
    decision_market_state = _mock_decision_market_state(n_assets, n_features, window_size)
    decision_market_state.available_mask_at_decision = np.array(
        [True, False, True, True, False, True, True, True, True, True]
    )
    portfolio_state = _mock_portfolio_state(n_assets)

    action = strategy.compute_target_weights(decision_market_state, portfolio_state)

    assert isinstance(action, PortfolioAction)
    assert action.target_weights.shape == (n_assets,)
    assert np.allclose(action.target_weights.sum(), 1.0)
    assert action.target_weights[1] == 0.0
    assert action.target_weights[4] == 0.0
    assert action.rebalance_action == 1
    assert action.rebalance_intensity == 1.0
    assert action.action_info["scheduler_controlled"] is True
    assert action.action_info["constraint_controlled"] is True

    default_like_config = deepcopy(DEFAULT_CONFIG)
    default_like_config.update(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": latent_dim,
        }
    )
    default_strategy = PPOBaselineStrategy(default_like_config)
    assert default_strategy.model.uses_fixed_mlp_state is True
    assert default_strategy.model.encoder.input_dim == n_features * window_size * n_assets + 2 * n_assets

    cnn_strategy = CNNPPOBaselineStrategy(config)
    assert cnn_strategy.model.uses_fixed_mlp_state is False
    cnn_action = cnn_strategy.compute_target_weights(decision_market_state, portfolio_state)
    assert isinstance(cnn_action, PortfolioAction)
    assert cnn_action.action_info["strategy"] == "cnn_ppo_baseline"
    assert np.allclose(cnn_action.target_weights.sum(), 1.0)
    assert cnn_action.target_weights[1] == 0.0
    assert cnn_action.target_weights[4] == 0.0


def test_ppo_proxy_holds_when_turnover_below_threshold():
    from src.baselines.cnn_ppo_baseline import CNNPPOBaselineStrategy
    from src.baselines.ppo_baseline import PPOBaselineStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    config = {
        "n_assets": n_assets,
        "n_features": n_features,
        "window_size": window_size,
        "latent_dim": 8,
        "encoder": {"type": "mlp"},
        "ppo_baseline": {"rebalance_turnover_threshold": 0.05},
        "cnn_ppo_baseline": {"rebalance_turnover_threshold": 0.05},
    }
    for strategy in (
        PPOBaselineStrategy(config),
        CNNPPOBaselineStrategy({**config, "encoder": {"type": "cnn", "cnn_channels": [4]}}),
    ):
        with torch.no_grad():
            for parameter in strategy.model.actor.parameters():
                parameter.zero_()

        action = strategy.compute_target_weights(
            _mock_decision_market_state(n_assets, n_features, window_size),
            _mock_portfolio_state(n_assets),
        )

        assert action.rebalance_action == 0
        assert action.rebalance_intensity == 0.0
        assert action.action_info["estimated_turnover"] == pytest.approx(0.0)
        assert action.action_info["forced_hold_reason"] == "below_rebalance_turnover_threshold"


def test_gated_and_dqn_only_baseline_contract():
    from src.baselines.bernoulli_gated_ppo import BernoulliGatedPPOStrategy
    from src.baselines.dqn_only import DQNOnlyStrategy

    n_assets = 10
    n_features = 5
    window_size = 20

    config = {
        "n_assets": n_assets,
        "n_features": n_features,
        "window_size": window_size,
        "latent_dim": 64,
        "encoder": {"type": "mlp"},
        "ppo": {"enabled": True},
        "dqn": {"enabled": True},
        "dqn_only": {"templates": ["hold", "equal_weight"]}
    }

    # Bernoulli Gated
    gated_strategy = BernoulliGatedPPOStrategy(config)
    action = gated_strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )
    assert isinstance(action, PortfolioAction)
    assert action.action_info["strategy"] == "bernoulli_gated_ppo"
    assert action.action_info["gate_action"] in {0, 1}
    assert np.isfinite(action.action_info["gate_log_prob"])
    assert np.isfinite(action.action_info["candidate_log_prob"])
    assert "execution" not in " ".join(action.action_info["gate_input_fields"])

    # DQN Only
    dqn_strategy = DQNOnlyStrategy(config)
    action = dqn_strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )
    assert isinstance(action, PortfolioAction)
    assert action.action_info["strategy"] == "dqn_only"
    assert action.action_info["template_chosen"] in {"hold", "equal_weight"}
    assert action.action_info["target_source"] == "template"
    assert action.action_info["q_values"].shape == (2,)
    assert "execution" not in " ".join(action.action_info["gate_input_fields"])

    hold_only_strategy = DQNOnlyStrategy({**config, "dqn_only": {"templates": ["hold"]}})
    hold_action = hold_only_strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )
    assert hold_action.action_info["template_chosen"] == "hold"
    assert hold_action.rebalance_action == 0
    assert hold_action.rebalance_intensity == 0.0

    equal_weight_strategy = DQNOnlyStrategy({**config, "dqn_only": {"templates": ["equal_weight"]}})
    equal_weight_state = _mock_decision_market_state(n_assets, n_features, window_size)
    equal_weight_state.available_mask_at_decision = np.array(
        [True, False, True, True, False, True, True, True, True, True]
    )
    action = equal_weight_strategy.compute_target_weights(equal_weight_state, _mock_portfolio_state(n_assets))
    assert action.action_info["template_chosen"] == "equal_weight"
    assert np.allclose(action.target_weights.sum(), 1.0)
    assert action.target_weights[1] == 0.0
    assert action.target_weights[4] == 0.0

    with pytest.raises(ValueError, match="ERR_DQN_ONLY_TEMPLATE_NOT_IMPLEMENTED"):
        DQNOnlyStrategy({**config, "dqn_only": {"templates": ["hold", "momentum"]}})


def test_bernoulli_proxy_gate_respects_turnover_threshold():
    from src.baselines.bernoulli_gated_ppo import BernoulliGatedPPOStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = BernoulliGatedPPOStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "mlp"},
            "bernoulli_gated_ppo": {"rebalance_turnover_threshold": 0.05},
        }
    )
    with torch.no_grad():
        for parameter in strategy.model.parameters():
            parameter.zero_()
        for parameter in strategy.gate.parameters():
            parameter.zero_()
        strategy.gate.advantage_net.bias[1] = 10.0

    action = strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )

    assert action.action_info["raw_gate_action"] == 1
    assert action.action_info["raw_model_requested_rebalance"] is True
    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["gate_action"] == 0
    assert action.action_info["estimated_turnover"] == pytest.approx(0.0)
    assert action.action_info["forced_hold_reason"] == "below_rebalance_turnover_threshold"


def test_dqn_only_hold_template_reports_final_target_turnover_zero():
    from src.baselines.dqn_only import DQNOnlyStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = DQNOnlyStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "mlp"},
            "dqn_only": {"templates": ["hold", "equal_weight"]},
        }
    )
    with torch.no_grad():
        for parameter in strategy.encoder.parameters():
            parameter.zero_()
        for parameter in strategy.gate.parameters():
            parameter.zero_()
        strategy.gate.advantage_net.bias[0] = 10.0
    portfolio = PortfolioState(
        date=pd.Timestamp("2024-01-01"),
        nav=1.0,
        portfolio_value=1e8,
        current_weights=np.array([0.70, 0.10, 0.10, 0.10]),
        step_index=4,
    )

    action = strategy.compute_target_weights(_mock_decision_market_state(n_assets, n_features, window_size), portfolio)

    assert action.action_info["template_chosen"] == "hold"
    assert action.action_info["raw_gate_action"] == 0
    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["estimated_turnover"] == pytest.approx(0.0)
    assert action.action_info["forced_hold_reason"] == "model_chosen_hold"


def test_dqn_only_hold_template_initializes_empty_portfolio():
    from src.baselines.dqn_only import DQNOnlyStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = DQNOnlyStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "mlp"},
            "dqn_only": {"templates": ["hold", "equal_weight"]},
        }
    )
    with torch.no_grad():
        for parameter in strategy.encoder.parameters():
            parameter.zero_()
        for parameter in strategy.gate.parameters():
            parameter.zero_()
        strategy.gate.advantage_net.bias[0] = 10.0
    portfolio = PortfolioState(
        date=pd.Timestamp("2024-01-01"),
        nav=1.0,
        portfolio_value=1e8,
        current_weights=np.zeros(n_assets),
        step_index=0,
    )

    action = strategy.compute_target_weights(_mock_decision_market_state(n_assets, n_features, window_size), portfolio)

    assert action.action_info["template_chosen"] == "hold"
    assert action.action_info["raw_gate_action"] == 0
    assert action.action_info["raw_model_requested_rebalance"] is False
    assert action.action_info["first_trade"] is True
    assert action.rebalance_action == 1
    assert action.rebalance_intensity == 1.0
    assert action.action_info["gate_action"] == 1
    assert action.action_info["estimated_turnover"] == pytest.approx(0.5)
    assert action.action_info["forced_hold_reason"] is None
    np.testing.assert_allclose(action.target_weights, np.ones(n_assets) / n_assets, atol=1.0e-6)


def test_dqn_only_rebalance_template_respects_turnover_threshold():
    from src.baselines.dqn_only import DQNOnlyStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = DQNOnlyStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "mlp"},
            "dqn_only": {
                "templates": ["equal_weight"],
                "rebalance_turnover_threshold": 0.05,
            },
        }
    )
    with torch.no_grad():
        for parameter in strategy.encoder.parameters():
            parameter.zero_()
        for parameter in strategy.gate.parameters():
            parameter.zero_()

    action = strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )

    assert action.action_info["template_chosen"] == "equal_weight"
    assert action.action_info["raw_gate_action"] == 1
    assert action.action_info["raw_model_requested_rebalance"] is True
    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["gate_action"] == 0
    assert action.action_info["estimated_turnover"] == pytest.approx(0.0)
    assert action.action_info["forced_hold_reason"] == "below_rebalance_turnover_threshold"


def test_deep_baseline_training_targets_next_open_holding_return():
    from src.baselines.deep_training import collect_training_batch, deep_baseline_training_config, training_summary
    from src.config import DEFAULT_CONFIG

    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    asset_ids = ["A", "B"]
    close = pd.DataFrame(
        [[10.0, 20.0], [11.0, 22.0], [12.0, 24.0], [13.0, 26.0]],
        index=dates,
        columns=asset_ids,
    )
    open_price = pd.DataFrame(
        [[10.0, 20.0], [10.5, 20.0], [10.0, 20.0], [12.5, 25.0]],
        index=dates,
        columns=asset_ids,
    )
    log_return = pd.DataFrame(9.0, index=dates, columns=asset_ids)
    availability = pd.DataFrame(True, index=dates, columns=asset_ids)
    dataset = MarketDatasetBundle(
        asset_universe=pd.DataFrame({"ts_code": asset_ids}),
        panel=pd.DataFrame(),
        wide={"open": open_price, "close": close, "log_return": log_return},
        metrics_features=None,
        feature_cols=[],
        auxiliary_target_cols=[],
        availability_mask=availability,
        availability_reason=None,
        data_manifest={"canonical_asset_order": asset_ids},
    )
    batch = collect_training_batch(
        {"dataset": dataset, "dates": pd.DatetimeIndex([dates[1]]), "config": deepcopy(DEFAULT_CONFIG)},
        n_features=1,
        window_size=2,
        n_assets=2,
        device=torch.device("cpu"),
        max_samples=8,
    )

    assert batch is not None
    np.testing.assert_allclose(
        batch.future_returns.cpu().numpy()[0],
        np.array([0.2, 0.2], dtype=np.float32),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        batch.current_weights.cpu().numpy()[0],
        np.array([0.4878049, 0.5121951], dtype=np.float32),
        atol=1.0e-6,
    )
    equal_weight_config = deepcopy(DEFAULT_CONFIG)
    equal_weight_config["baselines"]["deep_training"]["current_weight_mode"] = "equal_weight"
    equal_weight_batch = collect_training_batch(
        {"dataset": dataset, "dates": pd.DatetimeIndex([dates[1]]), "config": equal_weight_config},
        n_features=1,
        window_size=2,
        n_assets=2,
        device=torch.device("cpu"),
        max_samples=8,
    )
    assert equal_weight_batch is not None
    np.testing.assert_allclose(
        equal_weight_batch.current_weights.cpu().numpy()[0],
        np.array([0.5, 0.5], dtype=np.float32),
        atol=1.0e-6,
    )
    summary = training_summary("completed", current_weight_mode="equal_weight")
    assert summary["configured_current_weight_mode"] == "equal_weight"
    assert summary["effective_current_weight_mode"] == "equal_weight"
    assert summary["current_weight_mode"] == "equal_weight"
    assert summary["baseline_family"] == "neural_proxy"
    assert summary["rl_training"] is False
    assert summary["platform_native_rl_training"] is False
    assert summary["proxy_training"] is True
    assert summary["external_original_implementation"] is False
    assert summary["rankable_in_unified_table"] is False
    assert summary["execution_path_proxy"] is True
    assert summary["pending_action_queue_simulated"] is False

    invalid_config = deepcopy(DEFAULT_CONFIG)
    invalid_config["baselines"]["deep_training"]["current_weight_mode"] = "equal_weigth"
    with pytest.raises(ValueError, match="ERR_DEEP_BASELINE_CURRENT_WEIGHT_MODE_INVALID"):
        deep_baseline_training_config(invalid_config)


def test_proxy_baselines_are_not_marked_platform_native_rl():
    from src.baselines.deep_training import training_summary

    summary = training_summary("completed", current_weight_mode="rolling_equal_weight")

    assert summary["baseline_family"] == "neural_proxy"
    assert summary["rl_training"] is False
    assert summary["platform_native_rl_training"] is False
    assert summary["proxy_training"] is True
    assert summary["rankable_in_unified_table"] is False


def test_ppo_proxy_training_skips_no_gradient_minibatches(monkeypatch):
    import src.baselines.ppo_baseline as ppo_baseline
    from src.baselines.deep_training import DeepBaselineTrainingBatch

    config = {
        "n_assets": 2,
        "n_features": 1,
        "window_size": 2,
        "latent_dim": 8,
        "encoder": {"type": "mlp"},
        "baselines": {
            "deep_training": {
                "enabled": True,
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 1.0e-3,
                "max_samples": 64,
            }
        },
    }
    batch_size = 64
    market_image = torch.randn(batch_size, 1, 2, 2)
    availability_mask = torch.ones(batch_size, 2, dtype=torch.bool)
    availability_mask[:32, 1] = False
    current_weights = torch.zeros(batch_size, 2)
    current_weights[:32, 0] = 1.0
    current_weights[32:] = 0.5
    batch = DeepBaselineTrainingBatch(
        market_image=market_image,
        availability_mask=availability_mask,
        current_weights=current_weights,
        equal_weights=current_weights.clone(),
        future_returns=torch.randn(batch_size, 2) * 0.01,
    )
    monkeypatch.setattr(ppo_baseline, "collect_training_batch", lambda *args, **kwargs: batch)

    model = ppo_baseline.PPOBaselineModel(config)
    summary = ppo_baseline._train_ppo_baseline_model(model, {"dataset": object()}, config, torch.device("cpu"))

    assert summary["status"] == "completed"
    assert summary["skipped_no_gradient_minibatches"] == 1
    assert summary["gradient_updates"] == 1


def test_eiie_baseline_masking():
    from src.baselines.eiie import EIIEStrategy

    n_assets = 4
    n_features = 5
    window_size = 20

    config = {
        "n_assets": n_assets,
        "n_features": n_features,
        "window_size": window_size
    }

    strategy = EIIEStrategy(config)

    # Only asset 0 and 2 are available
    mask = np.array([True, False, True, False])
    state = _mock_decision_market_state(n_assets, n_features, window_size)
    state.available_mask_at_decision = mask

    action = strategy.compute_target_weights(state, _mock_portfolio_state(n_assets))

    assert action.target_weights[1] == 0.0
    assert action.target_weights[3] == 0.0
    assert np.allclose(action.target_weights.sum(), 1.0)
    assert np.isneginf(action.action_info["scores"][1])
    assert np.isneginf(action.action_info["scores"][3])
    assert np.allclose(action.action_info["previous_weights"], np.ones(n_assets) / n_assets)
    assert "execution" not in " ".join(action.action_info["score_input_fields"])


def test_eiie_native_uses_pvm_previous_weights():
    from src.baselines.native_eiie import NativeEIIEStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = NativeEIIEStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
        }
    )
    with torch.no_grad():
        conv = strategy.evaluator[0]
        linear = strategy.evaluator[-1]
        conv.weight.zero_()
        conv.bias.zero_()
        conv.weight[:, n_features, :].fill_(1.0)
        linear.weight.fill_(1.0)
        linear.bias.zero_()
    state = _mock_decision_market_state(n_assets, n_features, window_size)
    portfolio_left = PortfolioState(
        date=pd.Timestamp("2024-01-01"),
        nav=1.0,
        portfolio_value=1e8,
        current_weights=np.array([0.70, 0.10, 0.10, 0.10]),
    )
    portfolio_right = PortfolioState(
        date=pd.Timestamp("2024-01-01"),
        nav=1.0,
        portfolio_value=1e8,
        current_weights=np.array([0.10, 0.10, 0.10, 0.70]),
    )

    action_left = strategy.compute_target_weights(state, portfolio_left)
    action_right = strategy.compute_target_weights(state, portfolio_right)

    assert action_left.action_info["portfolio_vector_memory"] is True
    assert "previous_weights" in action_left.action_info["score_input_fields"]
    assert not np.allclose(action_left.target_weights, action_right.target_weights)


def test_eiie_previous_weights_use_current_drifted_position():
    from src.baselines.eiie import _previous_weights

    portfolio = PortfolioState(
        date=pd.Timestamp("2024-01-01"),
        nav=1.0,
        portfolio_value=1e8,
        current_weights=np.array([0.65, 0.15, 0.10, 0.10]),
        previous_executed_weights=np.array([0.25, 0.25, 0.25, 0.25]),
    )

    np.testing.assert_allclose(_previous_weights(portfolio), portfolio.current_weights)


def test_eiie_native_holds_when_turnover_below_threshold():
    from src.baselines.native_eiie import NativeEIIEStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = NativeEIIEStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "eiie_native": {"rebalance_turnover_threshold": 0.05},
        }
    )
    with torch.no_grad():
        for parameter in strategy.evaluator.parameters():
            parameter.zero_()
    state = _mock_decision_market_state(n_assets, n_features, window_size)
    portfolio = PortfolioState(
        date=pd.Timestamp("2024-01-01"),
        nav=1.0,
        portfolio_value=1e8,
        current_weights=np.ones(n_assets) / n_assets,
        step_index=3,
    )

    action = strategy.compute_target_weights(state, portfolio)

    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["estimated_turnover"] == pytest.approx(0.0)
    assert action.action_info["forced_hold_reason"] == "below_rebalance_turnover_threshold"


def test_eiie_native_rebalances_when_turnover_exceeds_threshold():
    from src.baselines.native_eiie import NativeEIIEStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = NativeEIIEStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "eiie_native": {"rebalance_turnover_threshold": 0.01},
        }
    )
    with torch.no_grad():
        conv = strategy.evaluator[0]
        linear = strategy.evaluator[-1]
        conv.weight.zero_()
        conv.bias.zero_()
        conv.weight[:, n_features, :].fill_(1.0)
        linear.weight.fill_(1.0)
        linear.bias.zero_()
    state = _mock_decision_market_state(n_assets, n_features, window_size)
    portfolio = PortfolioState(
        date=pd.Timestamp("2024-01-01"),
        nav=1.0,
        portfolio_value=1e8,
        current_weights=np.array([0.70, 0.10, 0.10, 0.10]),
        step_index=3,
    )

    action = strategy.compute_target_weights(state, portfolio)

    assert action.rebalance_action == 1
    assert action.rebalance_intensity == 1.0
    assert action.action_info["estimated_turnover"] > 0.01


def test_native_ppo_holds_when_turnover_below_threshold():
    from src.baselines.native_ppo import NativePPOBaselineStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = NativePPOBaselineStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "mlp"},
            "ppo_native": {"rebalance_turnover_threshold": 0.05},
        }
    )
    with torch.no_grad():
        for parameter in strategy.agent.actor.parameters():
            parameter.zero_()

    action = strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )

    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["estimated_turnover"] == pytest.approx(0.0)
    assert action.action_info["forced_hold_reason"] == "below_rebalance_turnover_threshold"


def test_native_cnn_ppo_holds_when_turnover_below_threshold():
    from src.baselines.native_ppo import NativeCNNPPOBaselineStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = NativeCNNPPOBaselineStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "cnn", "cnn_channels": [4]},
            "cnn_ppo_native": {"rebalance_turnover_threshold": 0.05},
        }
    )
    with torch.no_grad():
        for parameter in strategy.agent.actor.parameters():
            parameter.zero_()

    action = strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )

    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["estimated_turnover"] == pytest.approx(0.0)


def test_native_bernoulli_gate_respects_turnover_threshold():
    from src.baselines.native_bernoulli_gated_ppo import NativeBernoulliGatedPPOBaselineStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = NativeBernoulliGatedPPOBaselineStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "mlp"},
            "bernoulli_gated_ppo_native": {"rebalance_turnover_threshold": 0.05},
        }
    )
    with torch.no_grad():
        for parameter in strategy.actor.parameters():
            parameter.zero_()
        for parameter in strategy.gate.parameters():
            parameter.zero_()
        strategy.gate.net[-1].bias.fill_(10.0)

    action = strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )

    assert action.action_info["raw_bernoulli_gate_action"] == 1
    assert action.action_info["raw_model_requested_rebalance"] is True
    assert action.action_info["raw_action"] == 1
    assert action.action_info["raw_rho"] == pytest.approx(1.0)
    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["gate_action"] == 0
    assert action.action_info["estimated_turnover"] == pytest.approx(0.0)
    assert action.action_info["forced_hold_reason"] == "below_rebalance_turnover_threshold"


def test_pgportfolio_pvm_updates_current_sample_only():
    from src.baselines.pgportfolio_eiie import _apply_pvm_updates, _initial_pvm

    samples = [
        {"date": pd.Timestamp("2024-01-01"), "mask": np.array([True, True])},
        {"date": pd.Timestamp("2024-01-02"), "mask": np.array([True, True])},
    ]
    pvm = _initial_pvm(samples)
    trace = []

    _apply_pvm_updates(samples, pvm, [(0, np.array([0.8, 0.2]), pd.Timestamp("2024-01-01"))], trace)

    np.testing.assert_allclose(pvm[0], np.array([0.8, 0.2]))
    np.testing.assert_allclose(pvm[1], np.array([0.5, 0.5]))
    assert trace == [
        {
            "date": pd.Timestamp("2024-01-01"),
            "sample_index": 0,
            "updated_sample_state": True,
        }
    ]


def test_pgportfolio_pvm_persists_for_same_sample_dates():
    from src.baselines.pgportfolio_eiie import PGPortfolioEIIEStrategy

    strategy = PGPortfolioEIIEStrategy({"n_assets": 2, "n_features": 1, "window_size": 2})
    samples = [
        {"date": pd.Timestamp("2024-01-01"), "mask": np.array([True, True])},
        {"date": pd.Timestamp("2024-01-02"), "mask": np.array([True, True])},
    ]

    pvm = strategy._pvm_for_samples(samples)
    pvm[0] = np.array([0.9, 0.1])

    assert strategy._pvm_for_samples(samples) is pvm
    np.testing.assert_allclose(strategy._pvm_for_samples(samples)[0], np.array([0.9, 0.1]))


def test_pgportfolio_osbl_nonpermed_batch_is_contiguous():
    from src.baselines.pgportfolio_eiie import osbl_sample_indices

    batches = osbl_sample_indices(
        20,
        4,
        2,
        np.random.default_rng(7),
        sample_bias=1.0,
        is_permed=False,
    )

    assert len(batches) == 2
    for batch in batches:
        assert np.diff(batch).tolist() == [1, 1, 1]


def test_dqn_template_invalid_action_mask_blocks_bootstrap_q():
    from src.baselines.native_dqn_template import MaskedTemplateDQNAgent

    class FixedQ(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

        def forward(self, latent, candidate_weights, current_weights, estimated_turnover, estimated_cost):
            return self.bias.view(1, -1).repeat(latent.shape[0], 1)

    agent = MaskedTemplateDQNAgent(
        FixedQ(),
        FixedQ(),
        config={"dqn": {"batch_size": 1, "use_prioritized_replay": False, "use_n_step": False}},
    )
    batch = {
        "state_tp1": [
            {
                "latent": np.zeros(3, dtype=np.float32),
                "current_weights": np.ones(3, dtype=np.float32) / 3.0,
                "valid_action_mask": np.array([True, False, True, False, True, True, False, True]),
            }
        ],
        "candidate_weights_t": np.ones((1, 3), dtype=np.float32) / 3.0,
        "current_weights_t": np.ones((1, 3), dtype=np.float32) / 3.0,
        "estimated_turnover_t": np.array([0.0], dtype=np.float32),
        "estimated_cost_t": np.array([0.0], dtype=np.float32),
    }

    masked = agent._q_values(batch, next_state=True, target=True)

    assert masked[0, 1].item() < -1.0e8
    assert masked[0, 6].item() < -1.0e8
    assert masked[0, 5].item() == pytest.approx(5.0)


def test_dqn_invalid_auxiliary_target_has_no_env_transition_bootstrap():
    from src.agents.dqn_agent import DQNAgent

    class FixedQ(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.as_tensor(values, dtype=torch.float32))

        def forward(self, latent, candidate_weights, current_weights, estimated_turnover, estimated_cost):
            return self.values.view(1, -1).repeat(latent.shape[0], 1)

    online = FixedQ([0.0, 0.0])
    target = FixedQ([100.0, 100.0])
    agent = DQNAgent(
        online,
        target,
        config={"dqn": {"batch_size": 1, "use_prioritized_replay": False, "use_n_step": False}},
    )
    batch = {
        "state_t": [{"latent": np.zeros(2, dtype=np.float32)}],
        "state_tp1": [{"latent": np.zeros(2, dtype=np.float32)}],
        "candidate_weights_t": np.ones((1, 2), dtype=np.float32) / 2.0,
        "executed_weights_t": np.ones((1, 2), dtype=np.float32) / 2.0,
        "current_weights_t": np.ones((1, 2), dtype=np.float32) / 2.0,
        "gate_action_t": np.array([1], dtype=np.int64),
        "estimated_turnover_t": np.array([0.0], dtype=np.float32),
        "estimated_cost_t": np.array([0.0], dtype=np.float32),
        "reward_t": np.array([-1.25], dtype=np.float32),
        "terminated_t": np.array([False]),
        "truncated_t": np.array([False]),
        "bootstrap_mask_t": np.array([0.0], dtype=np.float32),
        "invalid_action_t": np.array([True]),
    }

    stats = agent.update(batch)

    assert stats["target_mean"] == pytest.approx(-1.25)


def test_dqn_template_all_actions_generate_valid_weights():
    from src.baselines.native_dqn_template import template_weights_from_observation

    returns = np.array(
        [
            [0.010, 0.020, 0.015, 0.005],
            [0.012, 0.018, 0.014, 0.006],
            [0.009, 0.021, 0.016, 0.004],
            [0.013, 0.019, 0.017, 0.007],
            [0.011, 0.022, 0.018, 0.005],
        ],
        dtype=np.float32,
    )
    observation = {
        "availability_mask": np.array([True, True, True, True]),
        "market_image": returns[np.newaxis, :, :],
        "volatility_20d_at_decision": np.array([0.20, 0.15, 0.12, 0.18], dtype=np.float32),
    }

    templates = template_weights_from_observation(observation, {"dqn_template": {"momentum_top_k": 2}})

    assert templates.weights.shape == (8, 4)
    assert templates.valid_mask.tolist() == [True] * 8
    np.testing.assert_allclose(templates.weights.sum(axis=1), np.ones(8), atol=1.0e-6)
    assert np.all(templates.weights >= 0.0)


def test_dqn_template_invalid_action_records_penalty_or_fallback():
    from src.baselines.native_dqn_template import template_weights_from_observation

    observation = {
        "availability_mask": np.array([True, True, True]),
        "market_image": np.zeros((1, 1, 3), dtype=np.float32),
        "volatility_20d_at_decision": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }

    templates = template_weights_from_observation(observation, {})

    assert bool(templates.valid_mask[0]) is True
    assert bool(templates.valid_mask[1]) is True
    assert bool(templates.valid_mask[2]) is False
    np.testing.assert_allclose(templates.weights[2], templates.weights[1], atol=1.0e-6)


def test_dqn_template_hold_action_returns_rebalance_zero():
    from src.baselines.native_dqn_template import NativeDQNTemplateStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = NativeDQNTemplateStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "mlp"},
            "dqn": {"use_prioritized_replay": False, "use_n_step": False},
        }
    )
    with torch.no_grad():
        for parameter in strategy.agent.online_network.parameters():
            parameter.zero_()
        strategy.agent.online_network.net[-1].bias[0] = 10.0

    portfolio = PortfolioState(
        date=pd.Timestamp("2024-01-01"),
        nav=1.0,
        portfolio_value=1e8,
        current_weights=np.array([0.40, 0.30, 0.20, 0.10]),
        step_index=3,
    )
    action = strategy.compute_target_weights(_mock_decision_market_state(n_assets, n_features, window_size), portfolio)

    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["template_chosen"] == "hold"
    assert action.action_info["gate_action"] == 0
    assert action.action_info["gate_action_index"] == 0
    np.testing.assert_allclose(action.target_weights, portfolio.current_weights, atol=1.0e-6)


def test_dqn_template_gate_action_is_binary_template_index_is_separate(monkeypatch):
    from src.baselines import native_dqn_template as module
    from src.baselines.native_dqn_template import NativeDQNTemplateStrategy, TemplateWeights

    n_assets = 4
    n_features = 3
    window_size = 5

    def fake_templates(observation, config):
        weights = np.tile(np.ones(n_assets, dtype=np.float32) / n_assets, (8, 1))
        weights[2] = np.array([0.70, 0.10, 0.10, 0.10], dtype=np.float32)
        valid_mask = np.zeros(8, dtype=bool)
        valid_mask[2] = True
        return TemplateWeights(weights, valid_mask)

    monkeypatch.setattr(module, "template_weights_from_observation", fake_templates)
    strategy = NativeDQNTemplateStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "mlp"},
            "dqn": {"use_prioritized_replay": False, "use_n_step": False},
        }
    )
    with torch.no_grad():
        for parameter in strategy.agent.online_network.parameters():
            parameter.zero_()
        strategy.agent.online_network.net[-1].bias[2] = 10.0

    action = strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )

    assert action.rebalance_action == 1
    assert action.action_info["gate_action"] == 1
    assert action.action_info["gate_action_index"] == 2
    assert action.action_info["template_action_index"] == 2


def test_dqn_template_replay_records_invalid_auxiliary_fields():
    from src.baselines.native_dqn_template import MaskedTemplateDQNAgent, _replay_item

    class FixedQ(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

        def forward(self, latent, candidate_weights, current_weights, estimated_turnover, estimated_cost):
            return self.bias.view(1, -1).repeat(latent.shape[0], 1)

    agent = MaskedTemplateDQNAgent(
        FixedQ(),
        FixedQ(),
        config={"dqn": {"batch_size": 1, "use_prioritized_replay": False, "use_n_step": False}},
    )
    observation = {
        "latent": np.zeros(3, dtype=np.float32),
        "current_weights": np.ones(3, dtype=np.float32) / 3.0,
        "valid_action_mask": np.array([True, False, True, False, True, True, False, True]),
    }
    item = _replay_item(
        observation,
        observation,
        np.ones(3, dtype=np.float32) / 3.0,
        1,
        -1.0,
        True,
        True,
        {"decision_date": pd.Timestamp("2024-01-01")},
        0.0,
        agent,
        invalid_action=True,
        bootstrap_mask=0.0,
        next_state_source="none",
    )

    assert item.invalid_action_t is True
    assert item.bootstrap_mask_t == 0.0
    assert item.next_state_source_t == "none"
    assert item.q_reference_t == pytest.approx(0.0)
    assert item.q_selected_t == pytest.approx(1.0)
    assert item.q_selected_minus_reference_t == pytest.approx(1.0)


def test_dqn_template_real_transitions_use_n_step_buffer():
    from src.baselines.native_dqn_template import NativeDQNTemplateStrategy

    class FakeEnv:
        config = {
            "dqn_template": {"momentum_top_k": 1},
        }

        def __init__(self):
            self.index = 0
            self.observations = [
                _dqn_template_observation(),
                _dqn_template_observation(scale=2.0),
                _dqn_template_observation(scale=3.0),
            ]

        def reset(self):
            self.index = 0
            return self.observations[0], {}

        def step(self, action):
            self.index += 1
            truncated = self.index >= 2
            info = {
                "decision_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=self.index - 1),
                "execution_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=self.index - 1),
                "next_valuation_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=self.index - 1),
                "execution_price_type": "open",
                "delayed_action_execution": False,
                "executed_weights": action["weights"],
                "realized_turnover": action["estimated_turnover"],
            }
            return self.observations[self.index], 0.1, False, truncated, info

    strategy = NativeDQNTemplateStrategy(
        {
            "n_assets": 3,
            "n_features": 1,
            "window_size": 3,
            "latent_dim": 3,
            "encoder": {"type": "mlp"},
            "dqn": {
                "batch_size": 99,
                "warmup_steps": 99,
                "use_prioritized_replay": False,
                "use_n_step": True,
                "n_steps": 2,
            },
            "dqn_template": {"invalid_action_penalty": 1.0},
        }
    )

    strategy._train_epoch(FakeEnv(), update_threshold=99)
    real_items = [item for item in strategy.agent.replay_buffer.items if not item.invalid_action_t]

    assert real_items
    assert max(item.n_steps for item in real_items) == 2


def test_dqn_template_train_epoch_honors_max_steps():
    from src.baselines.native_dqn_template import NativeDQNTemplateStrategy

    class FakeEnv:
        config = {
            "dqn_template": {"momentum_top_k": 1},
        }

        def __init__(self):
            self.index = 0
            self.observations = [
                _dqn_template_observation(),
                _dqn_template_observation(scale=2.0),
                _dqn_template_observation(scale=3.0),
            ]

        def reset(self):
            self.index = 0
            return self.observations[0], {}

        def step(self, action):
            self.index += 1
            info = {
                "decision_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=self.index - 1),
                "execution_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=self.index - 1),
                "next_valuation_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=self.index - 1),
                "execution_price_type": "open",
                "executed_weights": action["weights"],
                "realized_turnover": action["estimated_turnover"],
            }
            return self.observations[self.index], 0.1, False, False, info

    strategy = NativeDQNTemplateStrategy(
        {
            "n_assets": 3,
            "n_features": 1,
            "window_size": 3,
            "latent_dim": 3,
            "encoder": {"type": "mlp"},
            "dqn": {
                "batch_size": 99,
                "warmup_steps": 99,
                "use_prioritized_replay": False,
                "use_n_step": False,
            },
        }
    )

    reward_total, step_count, update_stats = strategy._train_epoch(FakeEnv(), update_threshold=99, max_steps=1)

    assert reward_total == pytest.approx(0.1)
    assert step_count == 1
    assert update_stats == []


def test_dqn_template_warmup_counts_real_transitions_not_invalid_auxiliary(monkeypatch):
    from src.baselines import native_dqn_template as module
    from src.baselines.native_dqn_template import NativeDQNTemplateStrategy, TemplateWeights

    class FakeEnv:
        config = {
            "dqn_template": {"momentum_top_k": 1},
        }

        def __init__(self):
            self.index = 0
            self.observations = [
                _dqn_template_observation(),
                _dqn_template_observation(scale=2.0),
                _dqn_template_observation(scale=3.0),
            ]

        def reset(self):
            self.index = 0
            return self.observations[0], {}

        def step(self, action):
            self.index += 1
            truncated = self.index >= 2
            info = {
                "decision_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=self.index - 1),
                "execution_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=self.index - 1),
                "next_valuation_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=self.index - 1),
                "execution_price_type": "open",
                "executed_weights": action["weights"],
                "realized_turnover": action["estimated_turnover"],
            }
            return self.observations[self.index], 0.1, False, truncated, info

    def fake_templates(observation, config):
        weights = np.tile(np.ones(3, dtype=np.float32) / 3.0, (8, 1))
        return TemplateWeights(weights, np.array([True, False, False, False, False, False, False, False]))

    monkeypatch.setattr(module, "template_weights_from_observation", fake_templates)
    strategy = NativeDQNTemplateStrategy(
        {
            "n_assets": 3,
            "n_features": 1,
            "window_size": 3,
            "latent_dim": 3,
            "encoder": {"type": "mlp"},
            "dqn": {
                "batch_size": 1,
                "warmup_steps": 3,
                "use_prioritized_replay": False,
                "use_n_step": False,
            },
            "dqn_template": {"invalid_action_penalty": 1.0},
        }
    )

    _, step_count, update_stats = strategy._train_epoch(FakeEnv(), update_threshold=3)

    assert step_count == 2
    assert len(strategy.agent.replay_buffer) > 3
    assert strategy._real_transition_count == 2
    assert update_stats == []


def test_dqn_template_train_epoch_honors_max_gradient_updates():
    from src.baselines.native_dqn_template import NativeDQNTemplateStrategy

    class FakeEnv:
        config = {
            "dqn_template": {"momentum_top_k": 1},
        }

        def __init__(self):
            self.index = 0
            self.observations = [
                _dqn_template_observation(),
                _dqn_template_observation(scale=2.0),
                _dqn_template_observation(scale=3.0),
                _dqn_template_observation(scale=4.0),
            ]

        def reset(self):
            self.index = 0
            return self.observations[0], {}

        def step(self, action):
            self.index += 1
            truncated = self.index >= 3
            info = {
                "decision_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=self.index - 1),
                "execution_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=self.index - 1),
                "next_valuation_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=self.index - 1),
                "execution_price_type": "open",
                "executed_weights": action["weights"],
                "realized_turnover": action["estimated_turnover"],
            }
            return self.observations[self.index], 0.1, False, truncated, info

    strategy = NativeDQNTemplateStrategy(
        {
            "n_assets": 3,
            "n_features": 1,
            "window_size": 3,
            "latent_dim": 3,
            "encoder": {"type": "mlp"},
            "dqn": {
                "batch_size": 1,
                "warmup_steps": 0,
                "use_prioritized_replay": False,
                "use_n_step": False,
                "target_update_interval": 1,
            },
        }
    )

    _, step_count, update_stats = strategy._train_epoch(FakeEnv(), update_threshold=1, max_gradient_updates=1)

    assert step_count == 3
    assert len(update_stats) == 1


def test_eiie_sequential_samples_applies_max_before_market_image(monkeypatch):
    from src.baselines import native_eiie as module

    dates = pd.bdate_range("2024-01-01", periods=5)
    frame = pd.DataFrame(
        np.ones((len(dates), 3), dtype=np.float32) * 0.01,
        index=dates,
        columns=["a", "b", "c"],
    )
    calls: list[pd.Timestamp] = []

    def fake_components(payload, n_assets):
        return {"pre_execution_returns": frame, "holding_returns": frame}

    def fake_market_image(dataset, market_image_dataset, asset_order, date, n_features, window_size):
        calls.append(pd.Timestamp(date))
        return np.ones((n_features, window_size, len(asset_order)), dtype=np.float32)

    monkeypatch.setattr(module, "execution_aligned_return_component_frames", fake_components)
    monkeypatch.setattr(module, "_asset_order", lambda dataset, n_assets: ["a", "b", "c"])
    monkeypatch.setattr(module, "_availability_mask", lambda dataset, asset_order, decision_date: np.ones(3, dtype=bool))
    monkeypatch.setattr(module, "_market_image", fake_market_image)

    samples = module._sequential_samples({"dataset": object(), "dates": dates}, 1, 3, 3, max_samples=2)

    assert len(samples) == 2
    assert len(calls) == 2


def _dqn_template_observation(scale: float = 1.0):
    return {
        "availability_mask": np.array([True, True, True]),
        "market_image": np.ones((1, 3, 3), dtype=np.float32) * scale,
        "current_weights": np.ones(3, dtype=np.float32) / 3.0,
        "volatility_20d_at_decision": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "adv20_at_decision": np.ones(3, dtype=np.float32),
        "amount_at_decision": np.ones(3, dtype=np.float32),
        "turnover_rate_at_decision": np.ones(3, dtype=np.float32),
        "portfolio_value": np.asarray(1.0, dtype=np.float32),
    }


def test_bernoulli_gate_hold_returns_rebalance_zero():
    from src.baselines.native_bernoulli_gated_ppo import NativeBernoulliGatedPPOBaselineStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = NativeBernoulliGatedPPOBaselineStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "cnn", "cnn_channels": [4]},
        }
    )
    with torch.no_grad():
        for parameter in strategy.gate.parameters():
            parameter.zero_()
        strategy.gate.net[-1].bias.fill_(-20.0)

    action = strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )

    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    np.testing.assert_allclose(action.target_weights, np.ones(n_assets) / n_assets)
    assert action.action_info["p_rebalance"] < 1.0e-6


def test_bernoulli_daily_activity_threshold_does_not_force_low_probability_rebalance():
    from src.baselines.native_bernoulli_gated_ppo import NativeBernoulliGatedPPOBaselineStrategy

    n_assets = 4
    n_features = 3
    window_size = 5
    strategy = NativeBernoulliGatedPPOBaselineStrategy(
        {
            "n_assets": n_assets,
            "n_features": n_features,
            "window_size": window_size,
            "latent_dim": 8,
            "encoder": {"type": "cnn", "cnn_channels": [4]},
            "execution_activity": {
                "protocol": "daily_gate_with_cost_constraint",
                "activity_gate_enforced": True,
                "min_model_rebalance_hit_rate": 0.05,
            },
        }
    )
    with torch.no_grad():
        for parameter in strategy.gate.parameters():
            parameter.zero_()
        strategy.gate.net[-1].bias.fill_(-0.5)

    action = strategy.compute_target_weights(
        _mock_decision_market_state(n_assets, n_features, window_size),
        _mock_portfolio_state(n_assets),
    )

    assert action.action_info["p_rebalance"] == pytest.approx(0.3775407, rel=1.0e-5)
    assert action.action_info["deterministic_gate_threshold"] == pytest.approx(0.5)
    assert action.rebalance_action == 0
