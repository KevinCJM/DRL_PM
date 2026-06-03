from copy import deepcopy

import numpy as np
import pandas as pd

from src.baselines.cage_eiie import CageEIIEFixedRho50Strategy, CageEIIEMultilevelGateStrategy, CageEIIENoCvarStrategy
from src.baselines.cage_common import gate_scoring_config
from src.baselines.gt_rcpo_lite import GTRCPOLiteStrategy
from src.config import ConfigLoader, DEFAULT_CONFIG, PROJECT_ROOT
from src.envs.state import DecisionMarketState, PortfolioState
from src.experiments.pipeline import _new_model_artifacts
from src.experiments.registry import DEEP_BASELINE_CLASSES
from src.experiments.run_experiment import NATIVE_HPO_MODEL_NAMES


def test_cage_outputs_candidate_weights_and_rho_without_premixing():
    strategy = CageEIIEFixedRho50Strategy(_strategy_config())
    strategy._candidate_weights = lambda _state, _portfolio: np.array([0.2, 0.8])
    strategy.set_decision_context(scheduler_allowed_rebalance=True, scheduler_pre_allowed=True, first_trade=False)

    action = strategy.compute_target_weights(_decision_state(), _portfolio_state())

    np.testing.assert_allclose(action.target_weights, np.array([0.2, 0.8]))
    assert action.rebalance_action == 1
    assert action.rebalance_intensity == 0.5
    assert action.action_info["execution_weight_mode"] == "candidate_plus_rho_execution_core"
    assert action.action_info["model_extension_id"] == "core13_v2_p12_p13_20260524"
    np.testing.assert_allclose(action.target_weights, np.array([0.2, 0.8]))
    assert not np.allclose(action.target_weights, np.array([0.4, 0.6]))


def test_scheduler_blocked_day_keeps_cage_raw_intent_for_engine_diagnostics():
    strategy = CageEIIEFixedRho50Strategy(_strategy_config())
    strategy._candidate_weights = lambda _state, _portfolio: np.array([0.2, 0.8])
    strategy.set_decision_context(scheduler_allowed_rebalance=False, scheduler_pre_allowed=False, first_trade=False)

    action = strategy.compute_target_weights(_decision_state(), _portfolio_state())

    np.testing.assert_allclose(action.target_weights, np.array([0.2, 0.8]))
    assert action.rebalance_action == 1
    assert action.rebalance_intensity == 0.5
    assert action.action_info["raw_rho"] == 0.5
    assert action.action_info["raw_model_requested_rebalance"] is True
    assert action.action_info["forced_hold_reason"] is None


def test_cage_no_cvar_variant_ignores_configured_cvar_penalty():
    strategy = CageEIIENoCvarStrategy(_strategy_config())
    strategy.config["cage_eiie"]["lambda_cvar"] = 999.0

    rho, _scores, _components, _reason = strategy._rho_action(
        scheduler_allowed=True,
        first_trade=False,
        estimated_turnover=0.0,
        estimated_cost=0.0,
        expected_return=0.01,
        expected_alpha_horizon=0.01,
        cvar_loss_5=1.0,
        drawdown=0.0,
    )

    assert rho == 1.0


def test_gt_rcpo_lite_keeps_raw_intent_on_scheduler_blocked_day():
    config = _strategy_config()
    config["gt_rcpo_lite"].update(
        {
            "lambda_turnover": 0.0,
            "lambda_cost": 0.0,
            "lambda_cvar": 0.0,
            "lambda_dd": 0.0,
        }
    )
    strategy = GTRCPOLiteStrategy(config)
    strategy.set_decision_context(scheduler_allowed_rebalance=False, scheduler_pre_allowed=False, first_trade=False)

    action = strategy.compute_target_weights(_decision_state(), _portfolio_state())

    assert action.rebalance_action == 1
    assert action.rebalance_intensity == 1.0
    assert action.action_info["raw_rho"] == 1.0
    assert action.action_info["raw_model_requested_rebalance"] is True
    assert action.action_info["graph_feature_mode"] == "decision_visible_rolling_correlation"
    assert action.action_info["scheduler_allowed_rebalance"] is False
    assert action.action_info["forced_hold_reason"] is None
    np.testing.assert_allclose(action.target_weights.sum(), 1.0)


def test_gt_rcpo_lite_uses_normalized_alpha_gate_after_initial_build():
    config = _strategy_config()
    strategy = GTRCPOLiteStrategy(config)
    strategy.set_decision_context(scheduler_allowed_rebalance=True, scheduler_pre_allowed=True, first_trade=False)

    action = strategy.compute_target_weights(_decision_state(), _portfolio_state())

    assert action.rebalance_action == 1
    assert action.rebalance_intensity > 0.0
    assert action.action_info["gate_scoring_mode"] == "normalized"
    assert action.action_info["expected_alpha_horizon"] > 0.0
    assert action.action_info["gate_score_components"] != "{}"


def test_gt_rcpo_lite_partial_rho_holds_when_executed_turnover_below_threshold():
    config = _strategy_config()
    config["execution_activity"]["model_rebalance_turnover_threshold"] = 0.02
    strategy = GTRCPOLiteStrategy(config)
    strategy._candidate_weights = lambda _state: np.array([0.61, 0.39])
    strategy._rho_action = lambda **_kwargs: (0.5, {"0": 0.0, "0.5": 1.0}, {}, None)
    strategy.set_decision_context(scheduler_allowed_rebalance=True, scheduler_pre_allowed=True, first_trade=False)

    action = strategy.compute_target_weights(_decision_state(), _portfolio_state())

    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["raw_gate_requested_rebalance"] is True
    assert action.action_info["raw_model_requested_rebalance"] is False
    assert action.action_info["raw_rho"] == 0.5
    assert action.action_info["rho"] == 0.0
    assert action.action_info["threshold_turnover_estimate"] < action.action_info["rebalance_turnover_threshold"]
    assert action.action_info["forced_hold_reason"] == "below_rebalance_turnover_threshold"


def test_cage_normalized_gate_uses_hpo_top_level_lambda_turnover():
    low_config = _strategy_config()
    low_config["hpo"]["search_space"] = {"cage_eiie.lambda_turnover": {"type": "float", "low": 0.5, "high": 4.0}}
    low_config["cage_eiie"]["gate_scoring"]["mode"] = "normalized"
    low_config["cage_eiie"]["lambda_turnover"] = 0.5
    low_rho, *_ = CageEIIEMultilevelGateStrategy(low_config)._rho_action(
        scheduler_allowed=True,
        first_trade=False,
        estimated_turnover=0.4,
        estimated_cost=0.0,
        expected_return=0.0,
        expected_alpha_horizon=0.003,
        cvar_loss_5=0.0,
        drawdown=0.0,
    )

    high_config = deepcopy(low_config)
    high_config["cage_eiie"]["lambda_turnover"] = 4.0
    high_rho, *_ = CageEIIEMultilevelGateStrategy(high_config)._rho_action(
        scheduler_allowed=True,
        first_trade=False,
        estimated_turnover=0.4,
        estimated_cost=0.0,
        expected_return=0.0,
        expected_alpha_horizon=0.003,
        cvar_loss_5=0.0,
        drawdown=0.0,
    )

    assert low_rho > 0.0
    assert high_rho == 0.0


def test_gt_rcpo_lite_normalized_gate_uses_hpo_top_level_lambda_turnover():
    low_config = _strategy_config()
    low_config["hpo"]["search_space"] = {"gt_rcpo_lite.lambda_turnover": {"type": "float", "low": 0.5, "high": 4.0}}
    low_config["gt_rcpo_lite"]["lambda_turnover"] = 0.5
    low_rho, *_ = GTRCPOLiteStrategy(low_config)._rho_action(
        scheduler_allowed=True,
        first_trade=False,
        expected_return=0.0,
        expected_alpha_horizon=0.003,
        estimated_turnover=0.4,
        estimated_cost=0.0,
        cvar_loss_5=0.0,
        drawdown=0.0,
    )

    high_config = deepcopy(low_config)
    high_config["gt_rcpo_lite"]["lambda_turnover"] = 4.0
    high_rho, *_ = GTRCPOLiteStrategy(high_config)._rho_action(
        scheduler_allowed=True,
        first_trade=False,
        expected_return=0.0,
        expected_alpha_horizon=0.003,
        estimated_turnover=0.4,
        estimated_cost=0.0,
        cvar_loss_5=0.0,
        drawdown=0.0,
    )

    assert low_rho > 0.0
    assert high_rho == 0.0


def test_normalized_gate_uses_hpo_top_level_shape_controls():
    config = _strategy_config()
    config["hpo"]["search_space"] = {
        "gt_rcpo_lite.alpha_scale": {"type": "float", "low": 0.0003, "high": 0.003},
        "gt_rcpo_lite.alpha_activation_threshold": {"type": "float", "low": 0.05, "high": 0.5},
        "gt_rcpo_lite.hold_opportunity_penalty": {"type": "float", "low": -0.8, "high": 0.0},
        "gt_rcpo_lite.cost_budget": {"type": "float", "low": 0.00005, "high": 0.002},
    }
    config["gt_rcpo_lite"].update(
        {
            "alpha_scale": 0.0007,
            "alpha_activation_threshold": 0.11,
            "hold_opportunity_penalty": -0.55,
            "cost_budget": 0.0003,
        }
    )

    gate = gate_scoring_config(config, "gt_rcpo_lite")

    assert gate["alpha_scale"] == 0.0007
    assert gate["alpha_activation_threshold"] == 0.11
    assert gate["hold_opportunity_penalty"] == -0.55
    assert gate["cost_budget_per_trade"] == 0.0003


def test_p12_p13_configs_load_and_models_are_registered():
    paths = [
        "p12_cage_eiie_smoke.yaml",
        "p12_cage_eiie_pilot.yaml",
        "p12_cage_eiie_ablation.yaml",
        "p12_cage_eiie_formal_seed_runner.yaml",
        "p12_cage_eiie_formal_comparison.yaml",
        "p12_cage_eiie_joint_light_pilot.yaml",
        "p12_cage_eiie_distributional_pilot.yaml",
        "p12_cage_eiie_fixed_rho_ablation.yaml",
        "p13_gt_rcpo_lite_smoke.yaml",
        "p13_gt_rcpo_lite_pilot.yaml",
        "p13_gt_rcpo_lite_formal_seed_runner.yaml",
        "p13_gt_rcpo_lite_formal_comparison.yaml",
    ]
    for name in paths:
        config = ConfigLoader.load(PROJECT_ROOT / "configs" / "paper" / name)
        assert config["protocol"]["protocol_id"] == "core13_v2_full_reset_20260522"
        assert config["new_model_protocol"]["model_extension_id"] == "core13_v2_p12_p13_20260524"
        assert config["new_model_protocol"]["test_used_for_model_selection"] is False
        assert config["training"]["checkpoint_include_replay_buffer"] is False

    expected = {
        "cage_eiie_frozen_gate",
        "cage_eiie_multilevel_gate",
        "cage_eiie_distributional",
        "cage_eiie_no_cvar",
        "cage_eiie_distributional_no_cvar",
        "cage_eiie_joint_light",
        "cage_eiie_fixed_rho_25",
        "cage_eiie_fixed_rho_50",
        "cage_eiie_fixed_rho_75",
        "graph_transformer_risk_constrained_actor_critic_lite",
        "gt_rcpo_lite",
    }
    assert expected.issubset(set(DEEP_BASELINE_CLASSES))
    assert expected.issubset(NATIVE_HPO_MODEL_NAMES)


def test_normalized_gate_hpo_configs_cover_activity_controls():
    required_cage = {
        "cage_eiie.lambda_turnover",
        "cage_eiie.lambda_cost",
        "cage_eiie.lambda_cvar",
        "cage_eiie.lambda_drawdown",
        "cage_eiie.turnover_budget",
        "cage_eiie.cost_budget",
        "cage_eiie.cvar_loss_budget",
        "cage_eiie.drawdown_budget",
        "cage_eiie.alpha_scale",
        "cage_eiie.alpha_activation_threshold",
        "cage_eiie.hold_opportunity_penalty",
    }
    required_gt = {
        "gt_rcpo_lite.lambda_turnover",
        "gt_rcpo_lite.lambda_cost",
        "gt_rcpo_lite.lambda_cvar",
        "gt_rcpo_lite.lambda_drawdown",
        "gt_rcpo_lite.turnover_budget",
        "gt_rcpo_lite.cost_budget",
        "gt_rcpo_lite.cvar_loss_budget",
        "gt_rcpo_lite.drawdown_budget",
        "gt_rcpo_lite.alpha_scale",
        "gt_rcpo_lite.alpha_activation_threshold",
        "gt_rcpo_lite.hold_opportunity_penalty",
    }
    configs = {
        "p12_cage_eiie_pilot.yaml": required_cage,
        "p12_cage_eiie_formal_seed_runner.yaml": required_cage,
        "p12_cage_eiie_distributional_pilot.yaml": required_cage,
        "p12_cage_eiie_joint_light_pilot.yaml": required_cage,
        "p13_gt_rcpo_lite_pilot.yaml": required_gt,
        "p13_gt_rcpo_lite_formal_seed_runner.yaml": required_gt,
    }

    for name, required in configs.items():
        config = ConfigLoader.load(PROJECT_ROOT / "configs" / "paper" / name)
        assert required.issubset(set(config["hpo"]["search_space"])), name


def test_new_model_artifact_derivation_from_diagnostics():
    diagnostics = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "decision_date": "2024-01-02",
                "execution_date": "2024-01-03",
                "model_name": "cage_eiie_distributional",
                "paper_model_id": "cage_eiie_distributional",
                "seed": 42,
                "fold_id": "fixed",
                "gate_action": 1,
                "gate_action_index": 2,
                "rho": 0.5,
                "candidate_weights_json": "[0.2,0.8]",
                "executed_weights_json": "[0.4,0.6]",
                "estimated_turnover": 0.2,
                "realized_turnover": 0.1,
                "estimated_cost": 0.001,
                "realized_cost": 0.0005,
                "CVaR_loss_5": 0.01,
                "drawdown": 0.02,
                "model_extension_id": "core13_v2_p12_p13_20260524",
            }
        ]
    )
    payload = {
        "baseline_daily_diagnostics": diagnostics,
        "daily_returns": pd.DataFrame(
            [
                {"date": "2024-01-03", "model_name": "cage_eiie_distributional", "seed": 42, "fold_id": "fixed", "net_return": 0.01}
            ]
        ),
        "canonical_asset_order": ["A", "B"],
    }

    artifacts = _new_model_artifacts(payload, config=DEFAULT_CONFIG)

    assert not artifacts["gate_actions"].empty
    assert not artifacts["cage_eiie_candidate_weights"].empty
    assert not artifacts["cage_final_weights"].empty
    assert not artifacts["risk_metrics"].empty
    assert artifacts["new_model_sidecar_manifest"]["model_extension_id"] == "core13_v2_p12_p13_20260524"


def _strategy_config():
    config = deepcopy(DEFAULT_CONFIG)
    config["n_assets"] = 2
    config["n_features"] = 1
    config["window_size"] = 3
    config["cost_model"]["market_impact_enabled"] = False
    config["rankability"]["rankable_in_unified_table"] = False
    return config


def _decision_state():
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    returns = np.array([[0.00, 0.00], [0.01, -0.01], [0.02, 0.00]], dtype=float)
    return DecisionMarketState(
        decision_date=dates[-1],
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
