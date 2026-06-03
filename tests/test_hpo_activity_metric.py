from copy import deepcopy

import pandas as pd

from src.config import DEFAULT_CONFIG
from src.experiments.pipeline import objective_metric
from src.experiments.run_experiment import (
    _activity_trial_failure_reason,
    _apply_hpo_final_activity_status,
    _best_hpo_model_payload,
    _hpo_model_final_comparison,
)


def test_objective_penalizes_turnover_underuse():
    config = _active_config()
    low = _result(hit_rate=0.0, turnover_per_opportunity=0.0)
    active = _result(hit_rate=0.10, turnover_per_opportunity=0.010)

    low_value = objective_metric(low, "validation_return_risk_cost_constrained", config=config)
    active_value = objective_metric(active, "validation_return_risk_cost_constrained", config=config)

    assert active_value > low_value


def test_low_activity_trial_not_selected_as_best():
    config = _active_config()

    reason = _activity_trial_failure_reason(_result(hit_rate=0.0, turnover_per_opportunity=0.0), config)

    assert reason == "failed_low_trade_activity"


def test_platform_native_scope_matches_native_rl_family_alias():
    config = _active_config()
    result = _result(hit_rate=0.0, turnover_per_opportunity=0.0)
    result["baseline_daily_diagnostics"] = pd.DataFrame([{"baseline_family": "native_rl"}])

    reason = _activity_trial_failure_reason(result, config)

    assert reason == "failed_low_trade_activity"


def test_platform_native_scope_matches_main_model_without_diagnostics():
    config = _active_config()
    result = _result(hit_rate=0.0, turnover_per_opportunity=0.0)
    result["baseline_daily_diagnostics"] = pd.DataFrame()
    result["main_comparison"] = pd.DataFrame(
        [{"model_name": "full_dqn_gated_multitask_cnn_ppo", "baseline_family": "model"}]
    )
    result["model_name"] = "full_dqn_gated_multitask_cnn_ppo"

    reason = _activity_trial_failure_reason(result, config)

    assert reason == "failed_low_trade_activity"


def test_learned_rho_entropy_collapse_has_specific_failure_reason():
    config = _active_config()
    config["ra_gt_rcpo"]["rho_policy"] = "straight_through_gumbel_softmax_v1"
    result = _result(hit_rate=0.0, turnover_per_opportunity=0.0)
    result["baseline_daily_diagnostics"]["rho_entropy"] = [0.01]

    reason = _activity_trial_failure_reason(result, config)

    assert reason == "failed_rho_policy_collapsed"


def test_monthly_gate_does_not_hard_fail_activity_trial():
    config = _active_config()
    config["execution_activity"]["protocol"] = "monthly_gate"
    config["execution_activity"]["scheduler_blocks_model_actions"] = True

    assert _activity_trial_failure_reason(_result(hit_rate=0.0, turnover_per_opportunity=0.0), config) is None


def test_final_activity_failure_marks_hpo_payload_non_rankable():
    config = _active_config()
    payload = _result(hit_rate=0.0, turnover_per_opportunity=0.0)
    payload.update(
        {
            "status": "completed",
            "model_name": "full_dqn_gated_multitask_cnn_ppo",
            "hpo_model_name": "full_dqn_gated_multitask_cnn_ppo",
            "best_trial_number": 3,
            "best_value": 1.0,
            "main_comparison": pd.DataFrame(
                [
                    {
                        "model_name": "full_dqn_gated_multitask_cnn_ppo",
                        "baseline_family": "model",
                        "rankable_in_unified_table": True,
                        "model_rebalance_hit_rate": 0.0,
                        "non_initial_turnover_per_opportunity": 0.0,
                        "average_turnover": 0.0,
                    }
                ]
            ),
        }
    )

    _apply_hpo_final_activity_status(payload, config)
    comparison = _hpo_model_final_comparison([payload])

    assert payload["rankable_in_unified_table"] is False
    assert payload["diagnostic_status"] == "activity_diagnostic"
    assert payload["final_activity_failure_reason"] == "failed_low_trade_activity"
    assert bool(comparison.loc[0, "rankable_in_unified_table"]) is False
    assert bool(comparison.loc[0, "paper_included"]) is False
    assert comparison.loc[0, "reason"] == "failed_low_trade_activity"


def test_equal_budget_best_payload_ignores_non_rankable_final_activity_winner():
    selected = _best_hpo_model_payload(
        [
            {
                "status": "completed",
                "hpo_model_name": "low_activity_model",
                "best_value": 10.0,
                "rankable_in_unified_table": False,
                "final_activity_failure_reason": "failed_low_trade_activity",
            },
            {
                "status": "completed",
                "hpo_model_name": "active_model",
                "best_value": 1.0,
            },
        ],
        "maximize",
    )

    assert selected["hpo_model_name"] == "active_model"


def _active_config():
    config = deepcopy(DEFAULT_CONFIG)
    config["execution_activity"].update(
        {
            "protocol": "daily_gate_with_cost_constraint",
            "scheduler_blocks_model_actions": False,
            "activity_gate_enforced": True,
        }
    )
    config["hpo"]["activity_constraints"]["enabled"] = True
    return config


def _result(*, hit_rate: float, turnover_per_opportunity: float):
    return {
        "daily_returns": pd.DataFrame(
            [
                {"net_return": 0.01, "nav": 1.01},
                {"net_return": -0.002, "nav": 1.00798},
                {"net_return": 0.006, "nav": 1.01402788},
            ]
        ),
        "metrics": {
            "model_rebalance_hit_rate": hit_rate,
            "non_initial_turnover_per_opportunity": turnover_per_opportunity,
            "average_turnover": 0.01,
            "total_transaction_cost": 0.001,
        },
        "baseline_daily_diagnostics": pd.DataFrame([{"baseline_family": "new_model_extension"}]),
    }
