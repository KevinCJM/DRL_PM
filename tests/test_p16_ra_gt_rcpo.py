from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from src.agents.constrained_actor_critic_agent import ConstrainedActorCriticAgent, agent_config_from_mapping
from src.baselines.deep_training import DeepBaselineTrainingBatch
from src.baselines.risk_aware_gt_rcpo import RiskAwareGTRCPOStrategy, _formal_training_budget_status, _gate_scoring_config
from src.config import ConfigLoader, DEFAULT_CONFIG, PROJECT_ROOT
from src.envs.state import DecisionMarketState, PortfolioState
from src.experiments.pipeline import _new_model_artifacts
from src.experiments.registry import DEEP_BASELINE_CLASSES, ExperimentRegistry
from src.experiments.run_experiment import NATIVE_HPO_MODEL_NAMES
from src.models.risk_aware_graph_transformer import (
    RA_GT_RCPO_MODEL_EXTENSION_ID,
    RA_GT_RCPO_MODEL_NAME,
    build_risk_aware_graph_transformer,
)


def test_p16_model_trains_with_real_gradient_updates():
    config = _strategy_config()
    model = build_risk_aware_graph_transformer(config, model_name=RA_GT_RCPO_MODEL_NAME)
    agent_config = agent_config_from_mapping(config, section=config["ra_gt_rcpo"])
    agent = ConstrainedActorCriticAgent(model, config=agent_config, device=torch.device("cpu"))

    history, stats = agent.train_offline(_training_batch())

    assert int(stats["gradient_updates"]) > 0
    assert not history.empty
    assert float(history["validation_metric"].iloc[-1]) == float(history["validation_metric"].iloc[-1])


def test_p16_action_emits_raw_rho_before_scheduler_finalization():
    config = _strategy_config()
    config["ra_gt_rcpo"]["rho_policy"] = "straight_through_gumbel_softmax_v1"
    strategy = RiskAwareGTRCPOStrategy(config)
    strategy._agent.select_action = lambda *_args: {
        "candidate_weights": np.array([0.2, 0.8]),
        "raw_rho": 0.5,
        "rho": 0.5,
        "rho_action_index": 2,
        "rho_probs": [0.0, 0.0, 1.0, 0.0, 0.0],
        "rho_logits": [-10.0, -10.0, 10.0, -10.0, -10.0],
        "rho_entropy": 0.0,
        "rho_expected": 0.5,
        "graph_density": 0.0,
        "mean_abs_correlation": 0.0,
        "value_return": 0.0,
        "value_cost": 0.0,
        "value_drawdown": 0.0,
        "value_cvar_loss": 0.0,
    }
    strategy.set_decision_context(scheduler_allowed_rebalance=False, scheduler_pre_allowed=False, first_trade=False)

    action = strategy.compute_target_weights(_decision_state(), _portfolio_state())

    assert action.rebalance_action == 1
    assert action.rebalance_intensity == 0.5
    assert action.action_info["raw_rho"] == 0.5
    assert action.action_info["raw_model_requested_rebalance"] is True
    assert action.action_info["execution_weight_mode"] == "candidate_plus_rho_execution_core"
    assert action.action_info["model_extension_id"] == RA_GT_RCPO_MODEL_EXTENSION_ID
    assert action.action_info["graph_feature_mode"] == "decision_visible_rolling_correlation"
    np.testing.assert_allclose(action.target_weights.sum(), 1.0)


def test_p16_partial_rho_holds_when_executed_turnover_below_threshold():
    config = _strategy_config()
    config["ra_gt_rcpo"]["rho_policy"] = "straight_through_gumbel_softmax_v1"
    config["execution_activity"]["model_rebalance_turnover_threshold"] = 0.02
    strategy = RiskAwareGTRCPOStrategy(config)
    strategy._agent.select_action = lambda *_args: {
        "candidate_weights": np.array([0.61, 0.39]),
        "raw_rho": 0.5,
        "rho": 0.5,
        "rho_action_index": 2,
        "rho_probs": [0.0, 0.0, 1.0, 0.0, 0.0],
        "rho_logits": [-10.0, -10.0, 10.0, -10.0, -10.0],
        "rho_entropy": 0.0,
        "rho_expected": 0.5,
        "rho_eval_mode": "argmax",
        "rho_eval_entropy_normalized": 0.0,
        "rho_eval_used_expected": False,
        "graph_density": 0.0,
        "mean_abs_correlation": 0.0,
        "value_return": 0.0,
        "value_cost": 0.0,
        "value_drawdown": 0.0,
        "value_cvar_loss": 0.0,
    }
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


def test_p16_learned_rho_zero_uses_normalized_gate_activity_fallback():
    config = _strategy_config()
    config["ra_gt_rcpo"].update(
        {
            "rho_policy": "straight_through_gumbel_softmax_v1",
            "alpha_scale": 0.001,
            "alpha_activation_threshold": 0.01,
            "hold_opportunity_penalty": -0.5,
            "lambda_turnover": 0.0,
            "lambda_cost": 0.0,
            "lambda_cvar": 0.0,
            "lambda_drawdown": 0.0,
            "average_turnover_per_step_budget": 1.0,
            "average_cost_per_step_budget": 1.0,
            "cvar_loss_budget": 1.0,
            "drawdown_budget": 1.0,
        }
    )
    config["execution_activity"]["model_rebalance_turnover_threshold"] = 0.0
    strategy = RiskAwareGTRCPOStrategy(config)
    strategy._agent.select_action = lambda *_args: {
        "candidate_weights": np.array([0.95, 0.05]),
        "raw_rho": 0.0,
        "rho": 0.0,
        "rho_action_index": 0,
        "rho_probs": [1.0, 0.0, 0.0, 0.0, 0.0],
        "rho_logits": [10.0, -10.0, -10.0, -10.0, -10.0],
        "rho_entropy": 0.0,
        "rho_expected": 0.0,
        "rho_eval_mode": "argmax",
        "rho_eval_entropy_normalized": 0.0,
        "rho_eval_used_expected": False,
        "graph_density": 0.0,
        "mean_abs_correlation": 0.0,
        "value_return": 0.0,
        "value_cost": 0.0,
        "value_drawdown": 0.0,
        "value_cvar_loss": 0.0,
    }
    strategy.set_decision_context(scheduler_allowed_rebalance=True, scheduler_pre_allowed=True, first_trade=False)

    action = strategy.compute_target_weights(_decision_state(), _portfolio_state())

    assert action.rebalance_action == 1
    assert action.action_info["agent_raw_rho"] == 0.0
    assert action.action_info["raw_rho"] > 0.0
    assert action.action_info["raw_model_requested_rebalance"] is True
    assert action.action_info["learned_rho_policy_adjustment_reason"] == "normalized_gate_activity_fallback"


def test_p16_normalized_gate_uses_hpo_top_level_constraints():
    config = _strategy_config()
    config["hpo"]["search_space"] = {
        "ra_gt_rcpo.lambda_turnover": {"type": "float", "low": 0.1, "high": 5.0},
        "ra_gt_rcpo.lambda_cost": {"type": "float", "low": 1.0, "high": 30.0},
        "ra_gt_rcpo.lambda_cvar": {"type": "float", "low": 0.0, "high": 2.0},
        "ra_gt_rcpo.lambda_drawdown": {"type": "float", "low": 0.0, "high": 2.0},
        "ra_gt_rcpo.average_turnover_per_step_budget": {"type": "float", "low": 0.005, "high": 0.08},
        "ra_gt_rcpo.average_cost_per_step_budget": {"type": "float", "low": 0.00005, "high": 0.001},
        "ra_gt_rcpo.cvar_loss_budget": {"type": "float", "low": 0.005, "high": 0.04},
        "ra_gt_rcpo.drawdown_budget": {"type": "float", "low": 0.05, "high": 0.15},
        "ra_gt_rcpo.alpha_scale": {"type": "float", "low": 0.0003, "high": 0.003},
        "ra_gt_rcpo.alpha_activation_threshold": {"type": "float", "low": 0.05, "high": 0.5},
        "ra_gt_rcpo.hold_opportunity_penalty": {"type": "float", "low": -0.8, "high": 0.0},
    }
    config["ra_gt_rcpo"].update(
        {
            "lambda_turnover": 3.5,
            "lambda_cost": 17.0,
            "lambda_cvar": 1.25,
            "lambda_drawdown": 1.5,
            "average_turnover_per_step_budget": 0.02,
            "average_cost_per_step_budget": 0.00007,
            "cvar_loss_budget": 0.033,
            "drawdown_budget": 0.12,
            "alpha_scale": 0.0008,
            "alpha_activation_threshold": 0.14,
            "hold_opportunity_penalty": -0.45,
        }
    )

    gate = _gate_scoring_config(config, "ra_gt_rcpo")

    assert gate["lambda_turnover"] == 3.5
    assert gate["lambda_cost"] == 17.0
    assert gate["lambda_cvar"] == 1.25
    assert gate["lambda_drawdown"] == 1.5
    assert gate["turnover_budget_per_trade"] == 0.02
    assert gate["cost_budget_per_trade"] == 0.00007
    assert gate["cvar_budget"] == 0.033
    assert gate["drawdown_budget"] == 0.12
    assert gate["alpha_scale"] == 0.0008
    assert gate["alpha_activation_threshold"] == 0.14
    assert gate["hold_opportunity_penalty"] == -0.45


def test_p16_configs_load_and_models_are_registered():
    paths = [
        "p16_ra_gt_rcpo_smoke.yaml",
        "p16_ra_gt_rcpo_pilot.yaml",
        "p16_ra_gt_rcpo_ablation.yaml",
        "p16_ra_gt_rcpo_formal_seed_runner.yaml",
        "p16_ra_gt_rcpo_formal_comparison.yaml",
    ]
    for name in paths:
        config = ConfigLoader.load(PROJECT_ROOT / "configs" / "paper" / name)
        assert config["protocol"]["protocol_id"] == "core13_v2_full_reset_20260522"
        assert config["new_model_protocol"]["model_extension_id"] == RA_GT_RCPO_MODEL_EXTENSION_ID
        assert config["new_model_protocol"]["test_used_for_model_selection"] is False
        assert config["training"]["checkpoint_include_replay_buffer"] is False

    expected = {
        RA_GT_RCPO_MODEL_NAME,
        "ra_gt_rcpo_no_graph",
        "ra_gt_rcpo_no_transformer",
        "ra_gt_rcpo_no_cvar_constraint",
        "ra_gt_rcpo_no_cost_constraint",
        "ra_gt_rcpo_no_turnover_constraint",
        "ra_gt_rcpo_mlp_actor_critic",
    }
    assert expected.issubset(set(DEEP_BASELINE_CLASSES))
    assert expected.issubset(NATIVE_HPO_MODEL_NAMES)
    cfg = ConfigLoader.load(PROJECT_ROOT / "configs/paper/p16_ra_gt_rcpo_smoke.yaml")
    assert type(ExperimentRegistry().create_experiment(cfg)).__name__ == "BaselineComparisonExperiment"
    required = {
        "ra_gt_rcpo.lambda_turnover",
        "ra_gt_rcpo.lambda_cost",
        "ra_gt_rcpo.lambda_cvar",
        "ra_gt_rcpo.lambda_drawdown",
        "ra_gt_rcpo.average_turnover_per_step_budget",
        "ra_gt_rcpo.average_cost_per_step_budget",
        "ra_gt_rcpo.cvar_loss_budget",
        "ra_gt_rcpo.drawdown_budget",
        "ra_gt_rcpo.alpha_scale",
        "ra_gt_rcpo.alpha_activation_threshold",
        "ra_gt_rcpo.hold_opportunity_penalty",
    }
    for name in ("p16_ra_gt_rcpo_pilot.yaml", "p16_ra_gt_rcpo_formal_seed_runner.yaml", "p16_ra_gt_rcpo_ablation.yaml"):
        config = ConfigLoader.load(PROJECT_ROOT / "configs" / "paper" / name)
        assert required.issubset(set(config["hpo"]["search_space"])), name


def test_p16_formal_budget_gate_only_applies_to_formal_runs():
    config = _strategy_config()
    config["ra_gt_rcpo"]["rho_policy"] = "straight_through_gumbel_softmax_v1"
    config["execution_activity"]["activity_gate_enforced"] = True

    assert _formal_training_budget_status(config, config["ra_gt_rcpo"], env_steps=32, gradient_updates=8) == "completed"

    config["rankability"]["rankable_in_unified_table"] = True
    config["rankability"]["diagnostic_status"] = "formal"

    assert (
        _formal_training_budget_status(config, config["ra_gt_rcpo"], env_steps=32, gradient_updates=8)
        == "failed_insufficient_training_budget"
    )


def test_p16_sidecar_artifacts_are_separate_from_p12_p13_extension():
    diagnostics = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "decision_date": "2024-01-02",
                "execution_date": "2024-01-03",
                "model_name": RA_GT_RCPO_MODEL_NAME,
                "paper_model_id": RA_GT_RCPO_MODEL_NAME,
                "seed": 42,
                "fold_id": "fixed",
                "rho": 0.5,
                "rebalance_intensity": 0.5,
                "scheduler_allowed_rebalance": True,
                "estimated_turnover": 0.2,
                "realized_turnover": 0.1,
                "estimated_cost": 0.001,
                "realized_cost": 0.0005,
                "CVaR_loss_5": 0.01,
                "max_drawdown_loss": 0.02,
                "lambda_turnover": 2.0,
                "lambda_cost": 10.0,
                "lambda_cvar": 0.35,
                "lambda_drawdown": 0.25,
                "graph_feature_mode": "decision_visible_rolling_correlation",
                "graph_density": 0.4,
                "mean_abs_correlation": 0.2,
                "constraint_violation_count": 1,
                "candidate_weights_json": "[0.2,0.8]",
                "executed_weights_json": "[0.3,0.7]",
                "model_extension_id": RA_GT_RCPO_MODEL_EXTENSION_ID,
            },
            {
                "date": "2024-01-03",
                "decision_date": "2024-01-02",
                "execution_date": "2024-01-03",
                "model_name": "cage_eiie_joint_light",
                "paper_model_id": "cage_eiie_joint_light",
                "seed": 42,
                "fold_id": "fixed",
                "rho": 0.25,
                "rebalance_intensity": 0.25,
                "candidate_weights_json": "[0.5,0.5]",
                "executed_weights_json": "[0.5,0.5]",
                "model_extension_id": "core13_v2_p12_p13_20260524",
            }
        ]
    )
    config = _strategy_config()
    config["new_model_protocol"]["model_extension_id"] = RA_GT_RCPO_MODEL_EXTENSION_ID
    payload = {
        "baseline_daily_diagnostics": diagnostics,
        "baseline_training_history": pd.DataFrame([{"model_name": RA_GT_RCPO_MODEL_NAME, "gradient_updates": 1}]),
        "daily_returns": pd.DataFrame(
            [{"date": "2024-01-03", "model_name": RA_GT_RCPO_MODEL_NAME, "seed": 42, "fold_id": "fixed", "net_return": 0.01}]
        ),
        "canonical_asset_order": ["A", "B"],
    }

    artifacts = _new_model_artifacts(payload, config=config)

    assert not artifacts["ra_gt_rcpo_daily_diagnostics"].empty
    assert set(artifacts["gate_actions"]["paper_model_id"]) == {RA_GT_RCPO_MODEL_NAME}
    assert not artifacts["ra_gt_rcpo_constraint_multipliers"].empty
    assert not artifacts["ra_gt_rcpo_graph_diagnostics"].empty
    assert not artifacts["ra_gt_rcpo_risk_decomposition"].empty
    assert artifacts["new_model_sidecar_manifest"]["model_extension_id"] == RA_GT_RCPO_MODEL_EXTENSION_ID


def _strategy_config():
    config = deepcopy(DEFAULT_CONFIG)
    config["n_assets"] = 2
    config["n_features"] = 1
    config["window_size"] = 3
    config["cost_model"]["market_impact_enabled"] = False
    config["rankability"]["rankable_in_unified_table"] = False
    config["new_model_protocol"]["model_extension_id"] = RA_GT_RCPO_MODEL_EXTENSION_ID
    config["ra_gt_rcpo"]["model_dim"] = 16
    config["ra_gt_rcpo"]["attention_heads"] = 2
    config["ra_gt_rcpo"]["batch_size"] = 2
    return config


def _training_batch():
    market_image = torch.tensor(
        [
            [[[0.00, 0.00], [0.01, -0.01], [0.02, 0.00]]],
            [[[0.00, 0.00], [-0.01, 0.01], [0.00, 0.02]]],
            [[[0.01, 0.00], [0.02, -0.02], [0.01, 0.01]]],
            [[[0.00, 0.01], [0.00, 0.02], [-0.01, 0.01]]],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones((4, 2), dtype=torch.bool)
    current = torch.full((4, 2), 0.5, dtype=torch.float32)
    equal = current.clone()
    future = torch.tensor([[0.01, -0.005], [-0.002, 0.01], [0.015, -0.01], [0.0, 0.012]], dtype=torch.float32)
    return DeepBaselineTrainingBatch(market_image, mask, current, equal, future)


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
