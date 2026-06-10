from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

import src.experiments.run_experiment as run_experiment
from scripts.recover_exp05_p7_formal_seed import _best_trial as _p7_recovery_best_trial
from src.config import DEFAULT_CONFIG
from src.experiments.pipeline import objective_metric
from src.experiments.registry import ExperimentContext, HPOExperiment
from src.experiments.run_experiment import (
    _activity_hpo_trial_hard_fail_enabled,
    _activity_trial_failure_reason,
    _apply_hpo_final_activity_status,
    _best_hpo_trial,
    _best_hpo_model_payload,
    _hpo_model_final_comparison,
    _selected_hpo_report_trials,
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


def test_activity_trial_failures_are_soft_by_default_with_explicit_hard_fail_escape_hatch():
    config = _active_config()

    assert _activity_trial_failure_reason(_result(hit_rate=0.0, turnover_per_opportunity=0.0), config) == "failed_low_trade_activity"
    assert _activity_hpo_trial_hard_fail_enabled(config) is False

    config["hpo"]["activity_constraints"]["hard_fail_trials"] = True

    assert _activity_hpo_trial_hard_fail_enabled(config) is True


def test_hpo_final_selection_prefers_activity_passed_trials():
    failed_high_value = SimpleNamespace(
        number=0,
        value=10.0,
        params={},
        user_attrs={"activity_failure_reason": "failed_low_trade_activity"},
    )
    passed_low_value = SimpleNamespace(number=1, value=1.0, params={}, user_attrs={})

    best = _best_hpo_trial([failed_high_value, passed_low_value], "maximize")
    selected = _selected_hpo_report_trials([failed_high_value, passed_low_value], "maximize")

    assert best.number == 1
    assert selected["best"].number == 1


def test_hpo_final_reports_retry_validation_best_when_final_activity_fails(monkeypatch):
    config = _active_config()
    experiment = SimpleNamespace(config=config)
    validation_best = SimpleNamespace(number=0, value=10.0, params={"x": 0}, user_attrs={})
    validation_second = SimpleNamespace(number=1, value=5.0, params={"x": 1}, user_attrs={})

    def fake_final_test(_experiment, trial, _final_split):
        if trial.number == 0:
            return _result(hit_rate=0.0, turnover_per_opportunity=0.0)
        return _result(hit_rate=0.2, turnover_per_opportunity=0.01)

    monkeypatch.setattr(run_experiment, "_run_hpo_final_test", fake_final_test)

    reports = run_experiment._run_hpo_final_reports(
        experiment,
        [validation_best, validation_second],
        "maximize",
        "test",
    )

    assert reports["best"]["trial_number"] == 1
    assert reports["best"]["validation_value"] == 5.0
    assert _activity_trial_failure_reason(reports["best"]["result"], config) is None


def test_hpo_final_reports_rejects_failed_final_when_validation_activity_passed(monkeypatch):
    config = _active_config()
    experiment = SimpleNamespace(config=config)
    validation_best = SimpleNamespace(number=0, value=10.0, params={"x": 0}, user_attrs={})
    validation_second = SimpleNamespace(number=1, value=5.0, params={"x": 1}, user_attrs={})

    monkeypatch.setattr(
        run_experiment,
        "_run_hpo_final_test",
        lambda *_args, **_kwargs: _result(hit_rate=0.0, turnover_per_opportunity=0.0),
    )

    with pytest.raises(RuntimeError, match="ERR_HPO_FINAL_ACTIVITY_NO_PASSING_TRIAL"):
        run_experiment._run_hpo_final_reports(
            experiment,
            [validation_best, validation_second],
            "maximize",
            "test",
        )


def test_hpo_final_reports_allows_soft_failure_when_no_validation_activity_passed(monkeypatch):
    config = _active_config()
    experiment = SimpleNamespace(config=config)
    validation_failed = SimpleNamespace(
        number=0,
        value=10.0,
        params={"x": 0},
        user_attrs={"activity_failure_reason": "failed_low_trade_activity"},
    )

    monkeypatch.setattr(
        run_experiment,
        "_run_hpo_final_test",
        lambda *_args, **_kwargs: _result(hit_rate=0.0, turnover_per_opportunity=0.0),
    )

    reports = run_experiment._run_hpo_final_reports(
        experiment,
        [validation_failed],
        "maximize",
        "test",
    )

    assert reports["best"]["trial_number"] == 0
    assert reports["best"]["selection_activity_failure_reason"] == "failed_low_trade_activity"
    assert reports["best"]["activity_failure_reason"] == "failed_low_trade_activity"
    assert _activity_trial_failure_reason(reports["best"]["result"], config) == "failed_low_trade_activity"


def test_validation_activity_failed_final_pass_is_non_rankable(monkeypatch):
    config = _active_config()
    experiment = SimpleNamespace(config=config)
    validation_failed = SimpleNamespace(
        number=0,
        value=10.0,
        params={"x": 0},
        user_attrs={"activity_failure_reason": "failed_low_trade_activity"},
    )

    monkeypatch.setattr(
        run_experiment,
        "_run_hpo_final_test",
        lambda *_args, **_kwargs: _result(hit_rate=0.2, turnover_per_opportunity=0.01),
    )

    reports = run_experiment._run_hpo_final_reports(
        experiment,
        [validation_failed],
        "maximize",
        "test",
    )
    table = run_experiment._hpo_final_report_table(
        reports,
        model_name="full_dqn_gated_multitask_cnn_ppo",
        study_name="selection_failed",
        final_split="test",
        best_trial_number=0,
        best_value=10.0,
        trial_count=1,
        selection_split="validation",
        config=config,
    )
    payload = _result(hit_rate=0.2, turnover_per_opportunity=0.01)
    payload.update(
        {
            "status": "completed",
            "model_name": "full_dqn_gated_multitask_cnn_ppo",
            "hpo_model_name": "full_dqn_gated_multitask_cnn_ppo",
            "best_trial_number": 0,
            "best_value": 10.0,
            "hpo_final_reports": run_experiment._hpo_final_report_summaries(reports),
            "hpo_final_reports_table": table,
            "main_comparison": pd.DataFrame(
                [
                    {
                        "model_name": "full_dqn_gated_multitask_cnn_ppo",
                        "baseline_family": "model",
                        "rankable_in_unified_table": True,
                        "paper_included": True,
                        "model_rebalance_hit_rate": 0.2,
                        "non_initial_turnover_per_opportunity": 0.01,
                        "average_turnover": 0.01,
                    }
                ]
            ),
        }
    )

    _apply_hpo_final_activity_status(payload, config)
    comparison = _hpo_model_final_comparison([payload])

    assert reports["best"]["selection_activity_failure_reason"] == "failed_low_trade_activity"
    assert reports["best"]["final_activity_failure_reason"] == ""
    assert table.loc[0, "selection_activity_failure_reason"] == "failed_low_trade_activity"
    assert table.loc[0, "final_activity_failure_reason"] == "failed_low_trade_activity"
    assert bool(table.loc[0, "rankable_in_unified_table"]) is False
    assert payload["selection_activity_failure_reason"] == "failed_low_trade_activity"
    assert payload["final_activity_failure_reason"] == "failed_low_trade_activity"
    assert bool(comparison.loc[0, "rankable_in_unified_table"]) is False
    assert bool(comparison.loc[0, "paper_included"]) is False
    assert comparison.loc[0, "reason"] == "failed_low_trade_activity"


def test_p7_recovery_best_trial_prefers_activity_passed_trials():
    failed_high_value = SimpleNamespace(
        number=0,
        value=10.0,
        params={},
        user_attrs={"activity_failure_reason": "failed_low_trade_activity"},
    )
    passed_low_value = SimpleNamespace(number=1, value=1.0, params={}, user_attrs={})

    best = _p7_recovery_best_trial([failed_high_value, passed_low_value], "maximize")

    assert best.number == 1


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


def test_run_hpo_single_keeps_low_activity_trial_complete(tmp_path, monkeypatch):
    config = _active_config()
    config["output"]["run_name"] = "soft_activity_hpo"
    config["hpo"].update(
        {
            "n_trials": 1,
            "metric": "validation_return_risk_cost_constrained",
            "objective": "validation_return_risk_cost_constrained",
            "search_space": {},
            "study_name": "soft_activity_hpo",
        }
    )
    run_dir = tmp_path / "soft_activity_hpo"
    context = ExperimentContext(
        config=config,
        execution_core=None,
        cost_model=None,
        constraint_manager=None,
        output_schema={},
        run_dir=run_dir,
    )
    experiment = HPOExperiment(context, "hyperparameter_sweep", "hpo_trials", hpo_enabled=True)
    low_activity = _result(hit_rate=0.0, turnover_per_opportunity=0.0)
    low_activity.update(
        {
            "status": "completed",
            "objective_value": -1.0,
            "validation_metric": -1.0,
            "rankable_in_unified_table": True,
        }
    )
    monkeypatch.setattr(run_experiment, "_run_hpo_trial", lambda *_args, **_kwargs: dict(low_activity))
    monkeypatch.setattr(run_experiment, "_run_hpo_final_reports", lambda *_args, **_kwargs: {"best": {"trial_number": 0, "validation_value": -1.0, "params": {}, "result": dict(low_activity)}})
    monkeypatch.setattr(run_experiment, "_write_best_trial_config_snapshot", lambda *_args, **_kwargs: None)

    payload = run_experiment._run_hpo_single(experiment)

    trial = payload["hpo_trials"].iloc[0]
    assert trial["state"] == "complete"
    assert trial["activity_failure_reason"] == "failed_low_trade_activity"
    assert pd.isna(trial["fail_reason"]) or trial["fail_reason"] == ""
    assert payload["status"] == "completed"
    assert payload["final_activity_failure_reason"] == "failed_low_trade_activity"


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


def test_equal_budget_best_payload_allows_activity_passed_pilot_payload():
    selected = _best_hpo_model_payload(
        [
            {
                "status": "completed",
                "hpo_model_name": "pilot_activity_passed",
                "best_value": 10.0,
                "rankable_in_unified_table": False,
                "diagnostic_status": "diagnostic",
                "final_activity_failure_reason": "",
            },
            {
                "status": "completed",
                "hpo_model_name": "formal_lower_value",
                "best_value": 1.0,
                "rankable_in_unified_table": True,
                "final_activity_failure_reason": "",
            },
        ],
        "maximize",
    )

    assert selected["hpo_model_name"] == "pilot_activity_passed"


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
