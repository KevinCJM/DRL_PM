from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments import pipeline as experiment_pipeline
from src.experiments import run_experiment
from src.baselines import hybrid_dqn_optimizer_reimplementation as hybrid_optimizer_module
from src.baselines.base_strategy import BaseStrategy
from src.baselines.hybrid_dqn_optimizer_reimplementation import (
    HYBRID_DQN_OPTIMIZER_ALIAS,
    HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
    HYBRID_DQN_OPTIMIZER_CHILD_SPECS,
    HYBRID_DQN_SIGNAL_ACTION_DIM,
    HYBRID_DQN_SIGNAL_ACTION_EXCLUDE,
    HYBRID_DQN_SIGNAL_ACTION_INCLUDE,
    HYBRID_DQN_SIGNAL_ACTION_NEUTRAL,
    HybridDQNOptimizerEqualWeightStrategy,
    HybridDQNOptimizerMarkowitzMeanVarianceStrategy,
    HybridDQNOptimizerMinimumVarianceStrategy,
    HybridDQNOptimizerReimplementationStrategy,
    HybridDQNOptimizerRiskParityStrategy,
    HybridDQNOptimizerSharpeMaximizationStrategy,
)
from src.baselines.ppo_dqn_hierarchical_reimplementation import (
    PPO_DQN_HIERARCHY_ACTION_DIM,
    PPO_DQN_HIERARCHY_HOLD_ACTION,
    PPO_DQN_HIERARCHY_ACTION_NAMES,
    PPODQNHierarchicalReimplementationStrategy,
)
from src.config import ConfigLoader
from src.data.loader import DataContractError
from src.data.splits import SplitSpec
from src.envs.backtest_engine import BacktestEngine, _build_decision_market_state
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.experiments.paper_aggregate import aggregate_paper_results
from src.experiments.registry import ExperimentContext, HPOExperiment
from src.experiments.pipeline import _comparison_rows, _paired_return_payload, result_mapping
from src.experiments.run_experiment import _hpo_model_final_frame
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import EXTRA_METRIC_FRAME_OUTPUTS, write_run_outputs


def test_training_env_preserves_related_work_action_info(
    sample_market_dataset_bundle,
    sample_split_spec,
    sample_config,
):
    config = deepcopy(sample_config)
    config["env"]["window_size"] = 3
    config["window_size"] = 3
    config["execution_model"]["delayed_action_execution"] = False
    config["execution_model"]["initial_build_cost"] = True
    config["rebalance"]["mode"] = "daily"

    env = PortfolioRebalanceEnv(
        sample_market_dataset_bundle,
        sample_split_spec,
        config=config,
        segment="test",
    )
    env.reset()

    action_info = {
        "paper_model_id": "ppo_dqn_hierarchical_reimplementation",
        "hierarchy_action": 2,
        "hierarchy_action_name": "blend_current_ppo_50",
        "ppo_actor_update_mask": 1,
        "ppo_attribution_weight": 0.5,
        "platform_adapted_surrogate": True,
        "child_model_name": "hybrid_dqn_optimizer_equal_weight",
        "baseline_family": "native_rl_reimplementation",
        "optimizer_name": "equal_weight",
        "include_count": 1,
        "exclude_count": 0,
        "neutral_count": 1,
        "selected_asset_count": 1,
        "optimizer_asset_count": 2,
        "optimizer_status": "fallback_equal_weight",
        "fallback_reason": "candidate_pool_equal_weight",
        "factorized_q": True,
        "portfolio_level_reward_shared": True,
        "counterfactual_asset_reward": False,
        "platform_adapted_approximation": True,
    }
    action = {
        "weights": np.array([0.5, 0.5], dtype=float),
        "rebalance": 1,
        "rebalance_intensity": 1.0,
        **action_info,
    }

    _, _, _, _, info = env.step(action)

    for key, expected in action_info.items():
        assert info[key] == expected


def test_backtest_collects_baseline_daily_diagnostics(
    sample_market_dataset_bundle,
    sample_trade_dates,
    sample_config,
):
    split = SplitSpec(
        train_dates=sample_trade_dates[:2],
        validation_dates=sample_trade_dates[2:4],
        test_dates=sample_trade_dates,
        fold_id="fixed",
        test_last_decision_date=sample_trade_dates[-2],
    )
    config = _related_work_config(sample_config)
    strategy = _RelatedWorkDiagnosticsStrategy()

    immediate = BacktestEngine(config).run(sample_market_dataset_bundle, split, strategy, segment="test")
    _assert_related_work_sidecar(immediate.baseline_daily_diagnostics)

    delayed_config = deepcopy(config)
    delayed_config["execution_model"]["execution_price"] = "next_close"
    delayed_config["execution_model"]["delayed_action_execution"] = True
    delayed = BacktestEngine(delayed_config).run(sample_market_dataset_bundle, split, strategy, segment="test")
    _assert_related_work_sidecar(delayed.baseline_daily_diagnostics)

    payload = result_mapping(
        immediate,
        config=config,
        artifacts=_artifact_stub(sample_market_dataset_bundle, split),
        status="completed",
        model_name=strategy.strategy_name,
    )
    pd.testing.assert_frame_equal(payload["baseline_daily_diagnostics"], immediate.baseline_daily_diagnostics)

    hpo_final = _hpo_model_final_frame(
        [{"hpo_model_name": strategy.strategy_name, "best_trial_number": 3, "baseline_daily_diagnostics": immediate.baseline_daily_diagnostics}],
        "baseline_daily_diagnostics",
    )
    _assert_related_work_sidecar(hpo_final)
    assert hpo_final["hpo_model_name"].eq(strategy.strategy_name).all()
    assert hpo_final["best_trial_number"].eq(3).all()


def test_hybrid_child_failure_status_rules(tmp_path, monkeypatch):
    completed_child = HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES[0]
    failed_child = HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES[1]

    def make_experiment(run_name, run_mode=None):
        config = {
            "experiment": {"type": "hyperparameter_sweep"},
            "output": {"run_name": run_name},
            "hpo": {
                "enabled": True,
                "equal_budget_across_models": True,
                "direction": "maximize",
            },
        }
        if run_mode is not None:
            config["hpo"]["run_mode"] = run_mode
        context = ExperimentContext(
            config=config,
            execution_core=None,
            cost_model=None,
            constraint_manager=None,
            output_schema={},
            run_dir=tmp_path / run_name,
        )
        return HPOExperiment(context, "hyperparameter_sweep", "hpo_trials", hpo_enabled=True)

    def fake_run_hpo_single(child_experiment):
        model_name = child_experiment.active_model_name
        if model_name == failed_child:
            return {
                "status": "failed_no_finite_validation_metric",
                "reason": "validation_metric_non_finite",
                "best_value": float("nan"),
                "hpo_trials": pd.DataFrame(columns=run_experiment.HPO_TRIAL_COLUMNS),
            }
        return {
            "status": "completed",
            "hpo_model_name": model_name,
            "study_name": child_experiment.config["hpo"]["study_name"],
            "best_trial_number": 0,
            "best_value": 1.0,
            "best_params": {},
            "trial_count": 1,
            "metrics": {"sharpe": 1.0},
            "hpo_trials": pd.DataFrame(
                [
                    {
                        **{column: "" for column in run_experiment.HPO_TRIAL_COLUMNS},
                        "model_name": model_name,
                        "state": "complete",
                        "objective_value": 1.0,
                    }
                ]
            ),
        }

    monkeypatch.setattr(run_experiment, "_run_hpo_single", fake_run_hpo_single)

    formal = run_experiment._run_equal_budget_hpo(
        make_experiment("hybrid_formal"),
        [completed_child, failed_child],
    )
    assert formal["status"] == "failed"
    assert formal["failed_child_model_id"] == failed_child
    assert formal["reason"] == "validation_metric_non_finite"
    assert formal["rankable_in_unified_table"] is False
    assert formal.get("diagnostic_status") != "partial_diagnostic"

    diagnostic = run_experiment._run_equal_budget_hpo(
        make_experiment("hybrid_diagnostic", run_mode="diagnostic"),
        [completed_child, failed_child],
    )
    assert diagnostic["status"] == "completed"
    assert diagnostic["diagnostic_status"] == "partial_diagnostic"
    assert diagnostic["failed_child_model_id"] == failed_child
    assert diagnostic["reason"] == "validation_metric_non_finite"
    assert diagnostic["rankable_in_unified_table"] is False
    assert diagnostic["best_model_name"] == completed_child
    assert [item["status"] for item in diagnostic["hpo_model_results"]] == [
        "completed",
        "failed_no_finite_validation_metric",
    ]

    def fake_skipped_hpo_single(child_experiment):
        model_name = child_experiment.active_model_name
        return {
            "status": "skipped",
            "reason": f"{model_name}_smoke_budget_skipped",
            "best_value": float("nan"),
            "hpo_trials": pd.DataFrame(columns=run_experiment.HPO_TRIAL_COLUMNS),
        }

    monkeypatch.setattr(run_experiment, "_run_hpo_single", fake_skipped_hpo_single)

    all_skipped = run_experiment._run_equal_budget_hpo(
        make_experiment("hybrid_smoke_all_skipped", run_mode="smoke"),
        [completed_child, failed_child],
    )
    assert all_skipped["status"] == "completed"
    assert all_skipped["diagnostic_status"] == "partial_diagnostic"
    assert all_skipped["failed_child_model_id"] == completed_child
    assert all_skipped["reason"] == f"{completed_child}_smoke_budget_skipped"
    assert all_skipped["rankable_in_unified_table"] is False
    assert all_skipped["best_model_name"] is None
    assert [item["status"] for item in all_skipped["hpo_model_results"]] == ["skipped", "skipped"]


def test_formal_hpo_seed_and_child_budget_contract(tmp_path, monkeypatch):
    formal_config = ConfigLoader.load("configs/paper/hpo_equal_budget_related_work.yaml")
    assert formal_config["long_running"] is True

    children = HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES[:2]
    config = {
        "long_running": True,
        "experiment": {"type": "hyperparameter_sweep"},
        "output": {"run_name": "formal_hpo"},
        "reproducibility": {"seed": 11, "seeds": [11, 22]},
        "hpo": {
            "enabled": True,
            "equal_budget_across_models": True,
            "native_only": True,
            "n_trials_per_model": 2,
            "direction": "maximize",
            "trainable_models": list(children),
        },
    }
    context = ExperimentContext(
        config=config,
        execution_core=None,
        cost_model=None,
        constraint_manager=None,
        output_schema={},
        run_dir=tmp_path / "formal_hpo",
    )
    experiment = HPOExperiment(context, "hyperparameter_sweep", "hpo_trials", hpo_enabled=True)
    calls = []

    def fake_run_hpo_single(child_experiment):
        model_name = child_experiment.active_model_name
        seed = int(child_experiment.config["reproducibility"]["seed"])
        calls.append(
            {
                "model_name": model_name,
                "seed": seed,
                "hpo_seed": int(child_experiment.config["hpo"]["seed"]),
                "n_trials_per_model": int(child_experiment.config["hpo"]["n_trials_per_model"]),
                "run_dir": str(child_experiment.context.run_dir.relative_to(tmp_path)),
                "study_name": child_experiment.config["hpo"]["study_name"],
            }
        )
        score = 1.0 + seed / 100.0 + list(children).index(model_name) / 10.0
        trials = pd.DataFrame([{column: "" for column in run_experiment.HPO_TRIAL_COLUMNS}], dtype=object)
        trials.loc[0, "model_name"] = model_name
        trials.loc[0, "study_name"] = child_experiment.config["hpo"]["study_name"]
        trials.loc[0, "trial_number"] = 0
        trials.loc[0, "seed"] = seed
        trials.loc[0, "state"] = "complete"
        trials.loc[0, "objective_value"] = score
        return {
            "status": "completed",
            "hpo_model_name": model_name,
            "study_name": child_experiment.config["hpo"]["study_name"],
            "best_trial_number": 0,
            "best_value": score,
            "best_params": {},
            "trial_count": 2,
            "best_checkpoint_path": str(child_experiment.context.run_dir / "checkpoints" / "best.pt"),
            "evaluated_checkpoint_path": str(child_experiment.context.run_dir / "final_test_best" / "best.pt"),
            "metrics": {"sharpe": score},
            "hpo_trials": trials,
        }

    monkeypatch.setattr(run_experiment, "_run_hpo_single", fake_run_hpo_single)

    result = run_experiment.run_hpo(experiment)

    assert result["status"] == "completed"
    assert result["long_running"] is True
    assert result["hpo_mode"] == "per_seed_independent_hpo"
    assert result["hpo_seed_count"] == 2
    assert len(calls) == 4
    assert [(call["seed"], call["model_name"]) for call in calls] == [
        (11, children[0]),
        (11, children[1]),
        (22, children[0]),
        (22, children[1]),
    ]
    assert {call["hpo_seed"] for call in calls} == {11, 22}
    assert {call["n_trials_per_model"] for call in calls} == {2}
    assert all(call["study_name"].startswith("formal_hpo_hpo_seed_") for call in calls)
    assert all(f"hpo_seed_{call['seed']}" in call["run_dir"] for call in calls)

    trials = result["hpo_trials"]
    assert trials.groupby("seed")["model_name"].apply(list).to_dict() == {
        11: list(children),
        22: list(children),
    }
    comparison = result["hpo_model_final_comparison"]
    assert comparison.groupby("seed")["hpo_model_name"].apply(list).to_dict() == {
        11: list(children),
        22: list(children),
    }

    artifacts = write_run_outputs(result, tmp_path / "formal_outputs", config=config)
    manifest = json.loads(artifacts["run_manifest"].read_text(encoding="utf-8"))
    assert manifest["long_running"] is True

    shared_dqn = _comparison_rows(
        {
            children[0]: {
                "diagnostic_status": "diagnostic_shared_dqn",
                "rankable_in_unified_table": True,
            }
        }
    )
    assert shared_dqn.loc[0, "diagnostic_status"] == "diagnostic_shared_dqn"
    assert bool(shared_dqn.loc[0, "rankable_in_unified_table"]) is False


@pytest.mark.parametrize(
    "status",
    [
        "failed_missing_train_data",
        "failed_no_finite_validation_metric",
        "failed_missing_best_checkpoint",
        "failed_checkpoint_load",
        "failed_no_valid_action",
        "failed_no_valid_optimizer_result",
        "deferred_variant",
        "needs_paper_confirmation",
    ],
)
def test_fit_required_failure_blocks_backtest(
    status,
    sample_market_dataset_bundle,
    sample_trade_dates,
    sample_config,
):
    split = SplitSpec(
        train_dates=sample_trade_dates[:2],
        validation_dates=sample_trade_dates[2:4],
        test_dates=sample_trade_dates,
        fold_id="fixed",
        test_last_decision_date=sample_trade_dates[-2],
    )
    strategy = _FailedFitRequiredRelatedWorkStrategy(status)

    with pytest.raises(DataContractError, match="ERR_STRATEGY_TRAINING_FAILED"):
        BacktestEngine(_related_work_config(sample_config)).run(
            sample_market_dataset_bundle,
            split,
            strategy,
            segment="test",
        )

    assert strategy.fit_calls == 1
    assert strategy.compute_calls == 0
    assert strategy.is_fitted is False
    assert strategy.training_result["status"] == status


def test_related_work_missing_paper_model_id_blocks_rankable_artifacts():
    model_name = "ppo_dqn_hierarchical_reimplementation"
    baseline_comparison = _comparison_rows({model_name: {"cumulative_return": 0.1}})
    assert bool(baseline_comparison.loc[0, "rankable_in_unified_table"]) is True

    comparison = _comparison_rows(
        {model_name: {"cumulative_return": 0.1}},
        training_summary_rows=[
            {
                "model_name": model_name,
                "status": "completed",
                "rankable_in_unified_table": False,
                "diagnostic_status": "missing_paper_model_id",
                "reason": "missing_paper_model_id",
            }
        ],
    )

    assert bool(comparison.loc[0, "rankable_in_unified_table"]) is False
    assert comparison.loc[0, "diagnostic_status"] == "missing_paper_model_id"

    returns_by_model = {
        model_name: pd.DataFrame({"date": pd.date_range("2024-01-02", periods=2), "net_return": [0.01, 0.02]}),
        "equal_weight": pd.DataFrame({"date": pd.date_range("2024-01-02", periods=2), "net_return": [0.0, 0.01]}),
    }
    assert _paired_return_payload(
        returns_by_model,
        {},
        training_summary_rows=[
            {
                "model_name": model_name,
                "rankable_in_unified_table": False,
            }
        ],
    ) == {}


def test_paper_diagnostic_comparison_excludes_main_rankings(tmp_path):
    formal_run = tmp_path / "results" / "formal_run"
    smoke_run = tmp_path / "results" / "p0_native_baseline_smoke"
    pilot_run = tmp_path / "results" / "hpo_equal_budget_native_pilot"
    daily_only_smoke_run = tmp_path / "results" / "daily_only_smoke"
    for run_dir, run_name in (
        (formal_run, "formal_run"),
        (smoke_run, "p0_native_baseline_smoke"),
        (pilot_run, "hpo_equal_budget_native_pilot"),
        (daily_only_smoke_run, "daily_only_smoke"),
    ):
        (run_dir / "metrics").mkdir(parents=True)
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "logs" / "run_manifest.json").write_text(
            json.dumps({"run_name": run_name, "experiment_type": "baseline_comparison", "seed": 42}),
            encoding="utf-8",
        )
    pd.DataFrame(
        [
            {"model_name": "ppo_dqn_hierarchical_reimplementation", "rankable_in_unified_table": True, "sharpe": 1.2},
            {"model_name": "equal_weight", "rankable_in_unified_table": True, "sharpe": 0.8},
            {
                "model_name": "pgportfolio_original_external",
                "rankable_in_unified_table": False,
                "external_original_implementation": True,
                "sharpe": 2.0,
            },
            {
                "model_name": "partial_hybrid_smoke",
                "paper_model_id": "partial_hybrid_smoke",
                "rankable_in_unified_table": False,
                "diagnostic_status": "partial_diagnostic",
                "reason": "child_failed",
                "sharpe": 3.0,
            },
            {
                "model_name": "shared_dqn_diagnostic",
                "paper_model_id": "hybrid_dqn_optimizer_risk_parity",
                "rankable_in_unified_table": True,
                "diagnostic_status": "diagnostic_shared_dqn",
                "sharpe": 4.0,
            },
            {
                "model_name": "incomplete_seed_grid_model",
                "rankable_in_unified_table": True,
                "seed_grid_complete": False,
                "reason": "incomplete_seed_grid",
                "sharpe": 4.5,
            },
        ]
    ).to_csv(formal_run / "metrics" / "baseline_comparison.csv", index=False)
    pd.DataFrame(
        [{"model_name": "smoke_model", "rankable_in_unified_table": True, "sharpe": 5.0}]
    ).to_csv(smoke_run / "metrics" / "baseline_comparison.csv", index=False)
    pd.DataFrame(
        [{"model_name": "pilot_model", "rankable_in_unified_table": True, "sharpe": 6.0}]
    ).to_csv(pilot_run / "metrics" / "baseline_comparison.csv", index=False)
    dates = pd.date_range("2024-01-02", periods=25, freq="D").strftime("%Y-%m-%d").tolist()
    daily_rows = []
    for model_name, net_return in (
        ("ppo_dqn_hierarchical_reimplementation", 0.01),
        ("equal_weight", 0.005),
        ("partial_hybrid_smoke", 0.02),
        ("shared_dqn_diagnostic", 0.03),
        ("incomplete_seed_grid_model", 0.035),
    ):
        daily_rows.extend({"date": date, "model_name": model_name, "net_return": net_return} for date in dates)
    pd.DataFrame(daily_rows).to_csv(formal_run / "metrics" / "daily_returns.csv", index=False)
    pd.DataFrame({"date": dates, "model_name": ["smoke_model"] * len(dates), "net_return": [0.04] * len(dates)}).to_csv(
        smoke_run / "metrics" / "daily_returns.csv",
        index=False,
    )
    pd.DataFrame({"date": dates, "model_name": ["pilot_model"] * len(dates), "net_return": [0.05] * len(dates)}).to_csv(
        pilot_run / "metrics" / "daily_returns.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "date": dates * 2,
            "model_name": ["daily_only_smoke_model"] * len(dates)
            + ["ppo_dqn_hierarchical_reimplementation"] * len(dates),
            "net_return": [0.06] * len(dates) + [0.50] * len(dates),
        }
    ).to_csv(
        daily_only_smoke_run / "metrics" / "daily_returns.csv",
        index=False,
    )

    outputs = aggregate_paper_results(
        [formal_run, smoke_run, pilot_run, daily_only_smoke_run],
        tmp_path / "paper",
        benchmark_model="equal_weight",
        paper_group_id="main_fixed",
    )

    assert outputs["paper_diagnostic_comparison"] == tmp_path / "paper" / "paper_diagnostic_comparison.csv"
    main = pd.read_csv(outputs["paper_main_comparison"])
    diagnostic = pd.read_csv(outputs["paper_diagnostic_comparison"])
    seed_summary = pd.read_csv(outputs["paper_seed_summary"])
    paired = pd.read_csv(outputs["paper_paired_statistics"])

    assert set(main["paper_model_id"]) == {"ppo_dqn_hierarchical_reimplementation", "equal_weight"}
    expected_diagnostic = {
        "pgportfolio_original_external",
        "partial_hybrid_smoke",
        "hybrid_dqn_optimizer_risk_parity",
        "incomplete_seed_grid_model",
        "smoke_model",
        "pilot_model",
        "daily_only_smoke_model",
    }
    assert expected_diagnostic.issubset(set(diagnostic["paper_model_id"]))
    assert diagnostic["rankable_in_unified_table"].map(lambda value: str(value).lower() == "false").all()
    assert diagnostic["reason"].astype(str).str.strip().ne("").all()
    assert expected_diagnostic.isdisjoint(set(seed_summary["paper_model_id"]))
    assert expected_diagnostic.isdisjoint(set(paired["model_name"].dropna()))
    model_stats = paired.loc[paired["model_name"].eq("ppo_dqn_hierarchical_reimplementation")].copy()
    assert not model_stats.empty
    assert pd.to_numeric(model_stats["n_obs"], errors="coerce").dropna().eq(len(dates)).all()

    diagnostic_benchmark_outputs = aggregate_paper_results(
        [formal_run, smoke_run, pilot_run, daily_only_smoke_run],
        tmp_path / "paper_diagnostic_benchmark",
        benchmark_model="partial_hybrid_smoke",
        paper_group_id="main_fixed",
    )
    diagnostic_benchmark_paired = pd.read_csv(diagnostic_benchmark_outputs["paper_paired_statistics"])
    assert set(diagnostic_benchmark_paired["status"]) == {"not_applicable"}


def test_closest_hybrid_figure_source_contract(tmp_path):
    formal_run = tmp_path / "results" / "formal_hpo"
    legacy_run = tmp_path / "results" / "original_paper_report"
    for run_dir, run_name in ((formal_run, "formal_hpo"), (legacy_run, "original_paper_report")):
        (run_dir / "metrics").mkdir(parents=True)
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "logs" / "run_manifest.json").write_text(
            json.dumps({"run_name": run_name, "experiment_type": "baseline_comparison", "seed": 42}),
            encoding="utf-8",
        )
    trainable_ids = ("ppo_dqn_hierarchical_reimplementation", *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES)
    rows = []
    for index, paper_model_id in enumerate(trainable_ids):
        rows.append(
            {
                "model_name": paper_model_id,
                "paper_model_id": paper_model_id,
                "child_model_name": paper_model_id,
                "baseline_family": "native_rl_reimplementation",
                "training_algorithm": (
                    "ppo_dqn_hierarchical_reimplementation"
                    if paper_model_id == "ppo_dqn_hierarchical_reimplementation"
                    else "factorized_dqn_signal_plus_portfolio_optimizer"
                ),
                "algorithm_fidelity": "platform_adapted",
                "rankable_in_unified_table": True,
                "sharpe": 1.0 + index,
                "cumulative_return": 0.1 + index * 0.01,
            }
        )
    rows.extend(
        [
            {
                "model_name": "hybrid_dqn_optimizer_reimplementation",
                "paper_model_id": "hybrid_dqn_optimizer_reimplementation",
                "baseline_family": "native_rl_reimplementation",
                "training_algorithm": "factorized_dqn_signal_plus_portfolio_optimizer",
                "algorithm_fidelity": "platform_adapted",
                "rankable_in_unified_table": True,
                "sharpe": 99.0,
            },
            {
                "model_name": "original_report_metric",
                "paper_model_id": "original_report_metric",
                "rankable_in_unified_table": True,
                "sharpe": 88.0,
            },
        ]
    )
    pd.DataFrame(rows).to_csv(formal_run / "metrics" / "baseline_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES[0],
                "paper_model_id": HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES[0],
                "rankable_in_unified_table": True,
                "sharpe": 77.0,
            }
        ]
    ).to_csv(legacy_run / "metrics" / "baseline_comparison.csv", index=False)

    outputs = aggregate_paper_results([formal_run, legacy_run], tmp_path / "paper", benchmark_model=trainable_ids[0])

    assert "closest_hybrid_figure_source" in EXTRA_METRIC_FRAME_OUTPUTS
    assert outputs["closest_hybrid_figure_source"] == tmp_path / "paper" / "closest_hybrid_figure_source.csv"
    figure_source = pd.read_csv(outputs["closest_hybrid_figure_source"])
    assert set(figure_source["paper_model_id"]) == set(trainable_ids)
    assert set(figure_source["baseline_family"]) == {"native_rl_reimplementation"}
    assert set(figure_source["algorithm_fidelity"]) == {"platform_adapted"}
    assert figure_source["rankable_in_unified_table"].map(lambda value: str(value).lower() == "true").all()
    assert {"algorithm_fidelity", "baseline_family", "training_algorithm", "rankable_in_unified_table"}.issubset(
        figure_source.columns
    )
    assert "seed_summary_mean_sharpe" in figure_source.columns
    assert "hybrid_dqn_optimizer_reimplementation" not in set(figure_source["paper_model_id"])
    assert "original_report_metric" not in set(figure_source["paper_model_id"])
    assert "original_paper_report" not in set(figure_source["source_run"])


def test_hybrid_child_specs_are_unique():
    expected = {
        "hybrid_dqn_optimizer_equal_weight": "equal_weight",
        "hybrid_dqn_optimizer_markowitz_mean_variance": "markowitz_mean_variance",
        "hybrid_dqn_optimizer_minimum_variance": "minimum_variance",
        "hybrid_dqn_optimizer_sharpe_maximization": "sharpe_maximization",
        "hybrid_dqn_optimizer_risk_parity": "risk_parity",
    }
    strategy_classes = (
        HybridDQNOptimizerEqualWeightStrategy,
        HybridDQNOptimizerMarkowitzMeanVarianceStrategy,
        HybridDQNOptimizerMinimumVarianceStrategy,
        HybridDQNOptimizerSharpeMaximizationStrategy,
        HybridDQNOptimizerRiskParityStrategy,
    )

    assert tuple(HYBRID_DQN_OPTIMIZER_CHILD_SPECS) == HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES
    assert {name: spec.optimizer_name for name, spec in HYBRID_DQN_OPTIMIZER_CHILD_SPECS.items()} == expected

    for strategy_class in strategy_classes:
        strategy = strategy_class({"optimizer_name": "unsupported_optimizer"})
        spec = HYBRID_DQN_OPTIMIZER_CHILD_SPECS[strategy_class.strategy_name]
        assert strategy.paper_model_id == strategy_class.strategy_name
        assert strategy.child_model_name == strategy_class.strategy_name
        assert strategy.optimizer_name == spec.optimizer_name
        assert strategy.training_result["paper_model_id"] == strategy_class.strategy_name
        assert strategy.training_result["child_model_name"] == strategy_class.strategy_name
        assert strategy.training_result["optimizer_name"] == spec.optimizer_name

    alias = HybridDQNOptimizerReimplementationStrategy({"optimizer_name": "unsupported_optimizer"})
    assert alias.strategy_name == HYBRID_DQN_OPTIMIZER_ALIAS
    assert alias.paper_model_id == HYBRID_DQN_OPTIMIZER_ALIAS
    assert alias.child_model_name is None
    assert alias.optimizer_name == "hybrid_dqn_optimizer"


def test_hybrid_signal_shape_and_unavailable_mask():
    config = {
        "n_assets": 3,
        "n_features": 2,
        "window_size": 4,
        "latent_dim": 8,
        "encoder": {"type": "mlp", "dropout": 0.0},
        "dqn": {"hidden_dims": [8], "dropout": 0.0},
        "device": {"mode": "cpu"},
    }
    strategy = HybridDQNOptimizerEqualWeightStrategy(config)
    market_image = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3) / 100.0
    q_values = strategy._asset_action_q_values({"market_image": market_image})

    assert strategy.asset_signal_q_network.action_dim == HYBRID_DQN_SIGNAL_ACTION_DIM
    assert strategy.target_asset_signal_q_network.action_dim == HYBRID_DQN_SIGNAL_ACTION_DIM
    assert q_values.shape == (3, HYBRID_DQN_SIGNAL_ACTION_DIM)
    assert np.isfinite(q_values).all()
    assert {HYBRID_DQN_SIGNAL_ACTION_EXCLUDE, HYBRID_DQN_SIGNAL_ACTION_NEUTRAL, HYBRID_DQN_SIGNAL_ACTION_INCLUDE} == {
        0,
        1,
        2,
    }

    forced_q_values = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.0, -1.0, 10.0],
            [0.0, np.nan, 9.0],
        ],
        dtype=np.float32,
    )
    available_mask = np.array([True, False, True], dtype=bool)
    masked = strategy._mask_asset_action_q_values(forced_q_values, available_mask)
    actions = strategy._select_asset_signal_actions(forced_q_values, available_mask)

    assert masked.shape == (3, HYBRID_DQN_SIGNAL_ACTION_DIM)
    assert np.isfinite(masked).all()
    assert actions.tolist() == [
        HYBRID_DQN_SIGNAL_ACTION_INCLUDE,
        HYBRID_DQN_SIGNAL_ACTION_EXCLUDE,
        HYBRID_DQN_SIGNAL_ACTION_EXCLUDE,
    ]
    assert int(masked[1].argmax()) == HYBRID_DQN_SIGNAL_ACTION_EXCLUDE
    assert int(masked[2].argmax()) == HYBRID_DQN_SIGNAL_ACTION_EXCLUDE


def test_hybrid_fill_can_use_available_exclude_assets():
    strategy = HybridDQNOptimizerEqualWeightStrategy(
        {
            "n_assets": 3,
            "n_features": 1,
            "window_size": 2,
            "latent_dim": 4,
            "encoder": {"type": "mlp", "dropout": 0.0},
            "dqn": {"hidden_dims": [4], "dropout": 0.0},
            "device": {"mode": "cpu"},
        }
    )
    q_values = np.array(
        [
            [0.0, 0.1, 2.0],
            [3.0, 1.0, 2.5],
            [2.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = strategy._select_candidate_assets(q_values, np.array([True, True, True], dtype=bool))

    assert result["asset_signal_actions"].tolist() == [
        HYBRID_DQN_SIGNAL_ACTION_INCLUDE,
        HYBRID_DQN_SIGNAL_ACTION_EXCLUDE,
        HYBRID_DQN_SIGNAL_ACTION_EXCLUDE,
    ]
    assert result["candidate_asset_indices"].tolist() == [0, 1]
    assert result["selected_asset_count"] == 1
    assert result["optimizer_asset_count"] == 2
    assert result["include_count"] == 1
    assert result["exclude_count"] == 2
    assert result["neutral_count"] == 0


def test_hybrid_optimizer_parameter_pins():
    config = {
        "n_assets": 4,
        "n_features": 1,
        "window_size": 2,
        "latent_dim": 4,
        "encoder": {"type": "mlp", "dropout": 0.0},
        "dqn": {"hidden_dims": [4], "dropout": 0.0},
        "device": {"mode": "cpu"},
        "markowitz": {
            "lookback_window": 5,
            "min_observations": 2,
            "covariance_shrinkage": 0.9,
            "risk_free_rate": 0.03,
            "lambda_risk": 9.0,
            "optimizer_maxiter": 3,
        },
        "risk_parity": {
            "lookback_window": 5,
            "min_observations": 2,
            "covariance_shrinkage": 0.9,
            "optimizer_maxiter": 3,
        },
    }
    class _TrappedState:
        @property
        def log_return_window(self):
            raise AssertionError("log_return_window read")

    equal_result = HybridDQNOptimizerEqualWeightStrategy(config)._compute_optimizer_weights(
        _TrappedState(),
        np.array([0, 2], dtype=np.int64),
    )
    np.testing.assert_allclose(equal_result["weights"], np.array([0.5, 0.5], dtype=np.float32))
    assert equal_result["optimizer_parameters"] == {"optimizer_name": "equal_weight"}

    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.01, size=(80, 4)).astype(np.float32)
    state = DecisionMarketState(
        decision_date=pd.Timestamp("2024-01-01"),
        available_mask_at_decision=np.ones(4, dtype=bool),
        availability_reason_at_decision=np.array(["listed"] * 4, dtype=object),
        close_at_decision=np.ones(4, dtype=float),
        log_return_at_decision=np.zeros(4, dtype=float),
        log_return_window=returns,
        amount_at_decision=np.ones(4, dtype=float),
        volume_at_decision=np.ones(4, dtype=float),
        adv20_at_decision=np.ones(4, dtype=float),
        volatility_20d_at_decision=np.ones(4, dtype=float) * 0.02,
        turnover_rate_at_decision=np.ones(4, dtype=float) * 0.01,
        feature_window=np.zeros((1, 2, 4), dtype=float),
        market_image=np.zeros((1, 2, 4), dtype=float),
    )
    expected_parameters = {
        HybridDQNOptimizerMarkowitzMeanVarianceStrategy: {
            "optimizer_name": "markowitz_mean_variance",
            "lookback_window": 252,
            "min_observations": 60,
            "covariance_shrinkage": 0.1,
            "optimizer_maxiter": 200,
            "risk_free_rate": 0.0,
            "lambda_risk": 1.0,
        },
        HybridDQNOptimizerMinimumVarianceStrategy: {
            "optimizer_name": "minimum_variance",
            "lookback_window": 252,
            "min_observations": 60,
            "covariance_shrinkage": 0.1,
            "optimizer_maxiter": 200,
        },
        HybridDQNOptimizerSharpeMaximizationStrategy: {
            "optimizer_name": "sharpe_maximization",
            "lookback_window": 252,
            "min_observations": 60,
            "covariance_shrinkage": 0.1,
            "optimizer_maxiter": 200,
            "risk_free_rate": 0.0,
        },
        HybridDQNOptimizerRiskParityStrategy: {
            "optimizer_name": "risk_parity",
            "lookback_window": 252,
            "min_observations": 60,
            "covariance_shrinkage": 0.1,
            "optimizer_maxiter": 300,
        },
    }

    for strategy_class, parameters in expected_parameters.items():
        result = strategy_class(config)._compute_optimizer_weights(state, np.array([0, 1, 3], dtype=np.int64))
        assert result["success"] is True
        assert result["optimizer_status"] == "success"
        assert result["optimizer_parameters"] == parameters
        assert result["candidate_asset_indices"].tolist() == [0, 1, 3]
        assert result["weights"].shape == (3,)
        assert np.isfinite(result["weights"]).all()
        assert float(result["weights"].sum()) == pytest.approx(1.0)


def test_hybrid_optimizer_window_matches_decision_visible_slice(
    sample_market_dataset_bundle,
    sample_trade_dates,
    sample_asset_order,
    sample_config,
):
    config = _related_work_config(sample_config)
    config["env"]["window_size"] = 3
    decision_date = sample_trade_dates[3]

    state = _build_decision_market_state(sample_market_dataset_bundle, decision_date, config)

    expected = sample_market_dataset_bundle.wide["log_return"].loc[
        sample_trade_dates[1:4],
        sample_asset_order,
    ]
    np.testing.assert_allclose(state.log_return_window, expected.to_numpy(dtype=float))
    future_row = sample_market_dataset_bundle.wide["log_return"].loc[sample_trade_dates[4], sample_asset_order]
    assert not np.isclose(state.log_return_window[-1], future_row.to_numpy(dtype=float)).all()


def test_hybrid_optimizer_fallback_reasons(monkeypatch):
    config = {
        "n_assets": 3,
        "n_features": 1,
        "window_size": 2,
        "latent_dim": 4,
        "encoder": {"type": "mlp", "dropout": 0.0},
        "dqn": {"hidden_dims": [4], "dropout": 0.0},
        "device": {"mode": "cpu"},
    }

    def decision_state(available_mask, returns):
        return DecisionMarketState(
            decision_date=pd.Timestamp("2024-01-01"),
            available_mask_at_decision=np.asarray(available_mask, dtype=bool),
            availability_reason_at_decision=np.array(["listed"] * 3, dtype=object),
            close_at_decision=np.ones(3, dtype=float),
            log_return_at_decision=np.zeros(3, dtype=float),
            log_return_window=np.asarray(returns, dtype=np.float32),
            amount_at_decision=np.ones(3, dtype=float),
            volume_at_decision=np.ones(3, dtype=float),
            adv20_at_decision=np.ones(3, dtype=float),
            volatility_20d_at_decision=np.ones(3, dtype=float) * 0.02,
            turnover_rate_at_decision=np.ones(3, dtype=float) * 0.01,
            feature_window=np.zeros((1, 2, 3), dtype=float),
            market_image=np.zeros((1, 2, 3), dtype=float),
        )

    rng = np.random.default_rng(11)
    full_history = rng.normal(0.001, 0.01, size=(80, 3)).astype(np.float32)
    strategy = HybridDQNOptimizerMarkowitzMeanVarianceStrategy(config)

    no_available = strategy._compute_optimizer_weights(
        decision_state([False, False, False], full_history),
        np.array([0, 1], dtype=np.int64),
    )
    assert no_available["status"] == "failed_no_valid_action"
    assert no_available["reason"] == "no_available_asset"
    assert no_available["success"] is False

    single_available = strategy._compute_optimizer_weights(
        decision_state([False, True, False], full_history),
        np.array([1], dtype=np.int64),
    )
    np.testing.assert_allclose(single_available["target_weights"], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert single_available["optimizer_status"] == "fallback_single_available_asset"
    assert single_available["fallback_reason"] == "single_available_asset"

    insufficient = strategy._compute_optimizer_weights(
        decision_state([True, True, True], full_history[:10]),
        np.array([0, 1], dtype=np.int64),
    )
    np.testing.assert_allclose(insufficient["target_weights"], np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert insufficient["success"] is True
    assert insufficient["optimizer_success"] is False
    assert insufficient["optimizer_status"] == "failed_insufficient_history"
    assert insufficient["fallback_reason"] == "insufficient_history"
    assert insufficient["fallback_source"] == "candidate_pool_equal_weight"

    empty_pool = strategy._compute_optimizer_weights(
        decision_state([True, True, True], full_history),
        np.array([], dtype=np.int64),
    )
    np.testing.assert_allclose(empty_pool["target_weights"], np.full(3, 1.0 / 3.0, dtype=np.float32))
    assert empty_pool["optimizer_status"] == "failed_no_candidate_asset"
    assert empty_pool["fallback_reason"] == "no_candidate_asset"
    assert empty_pool["fallback_source"] == "all_available_equal_weight"

    singular = strategy._compute_optimizer_weights(
        decision_state([True, True, True], np.tile(np.array([0.01, 0.01, 0.02], dtype=np.float32), (80, 1))),
        np.array([0, 1], dtype=np.int64),
    )
    np.testing.assert_allclose(singular["target_weights"], np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert singular["optimizer_status"] == "failed_singular_covariance"
    assert singular["fallback_reason"] == "singular_covariance"
    assert singular["fallback_source"] == "candidate_pool_equal_weight"

    with monkeypatch.context() as patched:
        def fake_failed_optimizer(*args, **kwargs):
            return hybrid_optimizer_module.PortfolioOptimizationResult(
                weights=np.zeros(2, dtype=np.float32),
                success=False,
                fallback_reason="optimizer_failed",
            )

        patched.setattr(hybrid_optimizer_module, "optimize_long_only_portfolio", fake_failed_optimizer)
        optimizer_failed = strategy._compute_optimizer_weights(
            decision_state([True, True, True], full_history),
            np.array([0, 1], dtype=np.int64),
        )
    np.testing.assert_allclose(optimizer_failed["target_weights"], np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert optimizer_failed["optimizer_status"] == "failed_optimizer_failed"
    assert optimizer_failed["fallback_reason"] == "optimizer_failed"
    assert optimizer_failed["fallback_source"] == "candidate_pool_equal_weight"

    with monkeypatch.context() as patched:
        def fake_non_finite_optimizer(*args, **kwargs):
            return hybrid_optimizer_module.PortfolioOptimizationResult(
                weights=np.array([np.nan, np.nan], dtype=np.float32),
                success=True,
            )

        patched.setattr(hybrid_optimizer_module, "optimize_long_only_portfolio", fake_non_finite_optimizer)
        non_finite = strategy._compute_optimizer_weights(
            decision_state([True, True, True], full_history),
            np.array([0, 1], dtype=np.int64),
        )
    np.testing.assert_allclose(non_finite["target_weights"], np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert non_finite["optimizer_status"] == "failed_non_finite_weights"
    assert non_finite["fallback_reason"] == "non_finite_optimizer_weights"
    assert non_finite["fallback_source"] == "candidate_pool_equal_weight"

    projection_config = {**config, "constraints": {"turnover_limit": 0.01}}
    projection_failed = HybridDQNOptimizerMarkowitzMeanVarianceStrategy(projection_config)._compute_optimizer_weights(
        decision_state([True, True, True], full_history),
        np.array([0, 1], dtype=np.int64),
    )
    assert projection_failed["status"] == "failed_no_valid_optimizer_result"
    assert projection_failed["optimizer_status"] == "failed_constraint_projection"
    assert projection_failed["fallback_reason"] == "constraint_projection_failed"
    assert projection_failed["fallback_source"] == "candidate_pool_equal_weight"
    assert projection_failed["success"] is False


def test_hybrid_each_child_trains_independently(
    sample_market_dataset_bundle,
    sample_trade_dates,
    sample_config,
    tmp_path,
    monkeypatch,
):
    validation_metric_calls = []

    def fake_objective_metric(payload, metric):
        assert metric == "validation_sharpe_minus_drawdown_turnover_penalty"
        daily_returns = payload["daily_returns"]
        assert {"net_return", "nav"}.issubset(daily_returns.columns)
        assert daily_returns["split"].eq("validation").all()
        assert "average_turnover" in payload["metrics"]
        validation_metric_calls.append(daily_returns[["net_return", "nav"]].copy())
        return 7.25

    monkeypatch.setattr(experiment_pipeline, "objective_metric", fake_objective_metric)
    config = _related_work_config(sample_config)
    config.update(
        {
            "n_assets": 2,
            "n_features": 1,
            "window_size": 3,
            "latent_dim": 4,
            "encoder": {"type": "mlp", "dropout": 0.0},
            "dqn": {
                "hidden_dims": [4],
                "batch_size": 1,
                "warmup_steps": 0,
                "target_update_interval": 1,
                "per_enabled": False,
                "use_n_step": False,
                "detach_encoder": True,
            },
            "baselines": {
                "native_rl": {
                    "epochs": 1,
                    "max_train_steps": 2,
                    "max_validation_steps": 2,
                    "max_gradient_updates_per_epoch": 2,
                },
                "checkpoint_dir": str(tmp_path / "hybrid_children"),
            },
            "device": {"mode": "cpu"},
        }
    )
    config["feature_matrix"]["window_size"] = 3
    train_payload = {
        "dataset": sample_market_dataset_bundle,
        "dates": sample_trade_dates[:4],
        "config": config,
    }
    validation_payload = {
        "dataset": sample_market_dataset_bundle,
        "dates": sample_trade_dates[2:5],
        "config": config,
    }
    required_history = {
        "epoch",
        "env_steps",
        "gradient_updates",
        "train_reward",
        "validation_metric",
        "loss",
        "status",
        "include_count",
        "neutral_count",
        "exclude_count",
        "selected_asset_count",
        "optimizer_asset_count",
        "optimizer_fallback_count",
    }
    strategy_classes = (
        HybridDQNOptimizerEqualWeightStrategy,
        HybridDQNOptimizerMarkowitzMeanVarianceStrategy,
        HybridDQNOptimizerMinimumVarianceStrategy,
        HybridDQNOptimizerSharpeMaximizationStrategy,
        HybridDQNOptimizerRiskParityStrategy,
    )
    checkpoint_best_paths = set()

    for strategy_class in strategy_classes:
        strategy = strategy_class(config)
        before_signal = [parameter.detach().clone() for parameter in strategy.asset_signal_q_network.parameters()]
        strategy.fit(train_payload, validation_payload)

        assert strategy.is_fitted is True
        assert strategy.training_result["status"] == "completed"
        assert strategy.training_result["paper_model_id"] == strategy_class.strategy_name
        assert strategy.training_result["child_model_name"] == strategy_class.strategy_name
        assert strategy.training_result["baseline_family"] == "native_rl_reimplementation"
        assert strategy.training_result["training_algorithm"] == "factorized_dqn_signal_plus_portfolio_optimizer"
        assert strategy.training_result["rl_training"] is True
        assert strategy.training_result["platform_native_rl_training"] is True
        assert strategy.training_result["proxy_training"] is False
        assert strategy.training_result["external_original_implementation"] is False
        assert strategy.training_result["rankable_in_unified_table"] is True
        assert strategy.training_result["clean_room_reimplementation"] is True
        assert strategy.training_result["algorithm_fidelity"] == "platform_adapted"
        assert strategy.training_result["factorized_q"] is True
        assert strategy.training_result["portfolio_level_reward_shared"] is True
        assert strategy.training_result["counterfactual_asset_reward"] is False
        assert strategy.training_result["platform_adapted_approximation"] is True
        assert np.isfinite(strategy.training_result["best_validation_metric"])
        assert strategy.training_result["evaluated_checkpoint_path"] == strategy.training_result["checkpoint_best_path"]
        assert strategy.training_result["checkpoint_best_path"] is not None
        assert strategy.training_result["checkpoint_last_path"] is not None
        assert strategy_class.strategy_name in strategy.training_result["checkpoint_best_path"]
        assert Path(strategy.training_result["checkpoint_best_path"]).exists()
        assert Path(strategy.training_result["checkpoint_last_path"]).exists()
        assert strategy._checkpoint_loaded_path == strategy.training_result["checkpoint_best_path"]
        checkpoint_best_paths.add(strategy.training_result["checkpoint_best_path"])
        assert strategy.training_result["env_steps"] == 2
        assert strategy.training_result["gradient_updates"] > 0
        assert required_history.issubset(strategy.training_history.columns)
        assert strategy.training_history["status"].eq("completed").all()
        assert np.isfinite(pd.to_numeric(strategy.training_history["validation_metric"], errors="coerce")).all()
        assert int(strategy.training_history["optimizer_fallback_count"].iloc[0]) >= 0
        payload = load_checkpoint(strategy.training_result["checkpoint_best_path"], device="cpu", restore_rng_state=False)
        assert payload["best_validation_metric"] == pytest.approx(strategy.training_result["best_validation_metric"])
        assert strategy.training_result["best_validation_metric"] == pytest.approx(7.25)
        after_signal = list(strategy.asset_signal_q_network.parameters())
        assert any(not torch.allclose(left, right.detach()) for left, right in zip(before_signal, after_signal, strict=True))

    assert len(checkpoint_best_paths) == len(strategy_classes)
    assert len(validation_metric_calls) == len(strategy_classes)
    missing = HybridDQNOptimizerEqualWeightStrategy(config)
    missing.fit(None, validation_payload)
    assert missing.is_fitted is False
    assert missing.training_result["status"] == "failed_missing_train_data"
    assert missing.training_result["checkpoint_best_path"] is None

    alias = HybridDQNOptimizerReimplementationStrategy(config)
    alias.fit(train_payload, validation_payload)
    assert alias.is_fitted is False
    assert alias.training_result["status"] == "deferred_variant"
    assert alias.training_result["reason"] == "orchestration_alias_only"
    assert alias.training_result["rankable_in_unified_table"] is False
    assert alias.training_result["checkpoint_best_path"] is None
    assert not (tmp_path / "hybrid_children" / "checkpoints" / HYBRID_DQN_OPTIMIZER_ALIAS / "best.pt").exists()

    def fake_non_finite_objective_metric(payload, metric):
        assert metric == "validation_sharpe_minus_drawdown_turnover_penalty"
        return float("nan")

    monkeypatch.setattr(experiment_pipeline, "objective_metric", fake_non_finite_objective_metric)
    non_finite_config = deepcopy(config)
    non_finite_config["baselines"] = deepcopy(config["baselines"])
    non_finite_config["baselines"]["checkpoint_dir"] = str(tmp_path / "hybrid_non_finite")
    non_finite = HybridDQNOptimizerEqualWeightStrategy(non_finite_config)
    non_finite.fit({**train_payload, "config": non_finite_config}, {**validation_payload, "config": non_finite_config})
    assert non_finite.is_fitted is False
    assert non_finite.training_result["status"] == "failed_no_finite_validation_metric"
    assert non_finite.training_result["checkpoint_best_path"] is None
    assert not (
        tmp_path / "hybrid_non_finite" / "checkpoints" / "hybrid_dqn_optimizer_equal_weight" / "best.pt"
    ).exists()

    mixed_metric_values = iter([7.25, float("nan")])

    def fake_mixed_objective_metric(payload, metric):
        assert metric == "validation_sharpe_minus_drawdown_turnover_penalty"
        return next(mixed_metric_values)

    monkeypatch.setattr(experiment_pipeline, "objective_metric", fake_mixed_objective_metric)
    mixed_config = deepcopy(config)
    mixed_config["baselines"] = deepcopy(config["baselines"])
    mixed_config["baselines"]["native_rl"] = deepcopy(config["baselines"]["native_rl"])
    mixed_config["baselines"]["native_rl"]["epochs"] = 2
    mixed_config["baselines"]["checkpoint_dir"] = str(tmp_path / "hybrid_mixed_non_finite")
    mixed = HybridDQNOptimizerEqualWeightStrategy(mixed_config)
    mixed.fit({**train_payload, "config": mixed_config}, {**validation_payload, "config": mixed_config})
    assert mixed.is_fitted is False
    assert mixed.training_result["status"] == "failed_no_finite_validation_metric"
    assert mixed.training_result["checkpoint_best_path"] is None
    assert mixed.training_result["evaluated_checkpoint_path"] is None
    assert mixed.training_history["status"].tolist() == ["completed", "failed_no_finite_validation_metric"]
    assert not (
        tmp_path / "hybrid_mixed_non_finite" / "checkpoints" / "hybrid_dqn_optimizer_equal_weight" / "best.pt"
    ).exists()


def test_hybrid_action_info_contract():
    config = {
        "n_assets": 3,
        "n_features": 1,
        "window_size": 2,
        "latent_dim": 4,
        "encoder": {"type": "mlp", "dropout": 0.0},
        "dqn": {"hidden_dims": [4], "dropout": 0.0},
        "device": {"mode": "cpu"},
    }
    q_values = np.array(
        [
            [0.0, 0.1, 3.0],
            [3.0, 1.0, 2.5],
            [0.0, 0.0, 9.0],
        ],
        dtype=np.float32,
    )
    decision_state = DecisionMarketState(
        decision_date=pd.Timestamp("2024-01-05"),
        available_mask_at_decision=np.array([True, True, False], dtype=bool),
        availability_reason_at_decision=np.array(["listed", "listed", "suspended"], dtype=object),
        close_at_decision=np.ones(3, dtype=float),
        log_return_at_decision=np.zeros(3, dtype=float),
        log_return_window=np.random.default_rng(17).normal(0.001, 0.01, size=(80, 3)).astype(np.float32),
        amount_at_decision=np.ones(3, dtype=float),
        volume_at_decision=np.ones(3, dtype=float),
        adv20_at_decision=np.ones(3, dtype=float),
        volatility_20d_at_decision=np.ones(3, dtype=float) * 0.02,
        turnover_rate_at_decision=np.ones(3, dtype=float) * 0.01,
        feature_window=np.zeros((1, 2, 3), dtype=float),
        market_image=np.zeros((1, 2, 3), dtype=float),
    )
    portfolio_state = PortfolioState(
        date=pd.Timestamp("2024-01-05"),
        nav=1.0,
        portfolio_value=1000000.0,
        current_weights=np.array([0.2, 0.8, 0.0], dtype=float),
    )

    strategy = HybridDQNOptimizerEqualWeightStrategy(config)
    strategy.is_fitted = True
    strategy._asset_action_q_values = lambda decision_input: q_values

    action = strategy.compute_target_weights(decision_state, portfolio_state)
    action_info = action.action_info

    assert isinstance(action, PortfolioAction)
    np.testing.assert_allclose(action.target_weights, np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert float(action.target_weights.sum()) == pytest.approx(1.0)
    assert np.all(action.target_weights[~decision_state.available_mask_at_decision] == 0.0)
    assert action_info["paper_model_id"] == "hybrid_dqn_optimizer_equal_weight"
    assert action_info["child_model_name"] == "hybrid_dqn_optimizer_equal_weight"
    assert action_info["baseline_family"] == "native_rl_reimplementation"
    assert action_info["optimizer_name"] == "equal_weight"
    assert action_info["include_count"] == 1
    assert action_info["neutral_count"] == 0
    assert action_info["exclude_count"] == 2
    assert action_info["selected_asset_count"] == 1
    assert action_info["optimizer_asset_count"] == 2
    assert action_info["optimizer_status"] == "success"
    assert action_info["fallback_reason"] is None

    signal_denominator = action_info["include_count"] + action_info["neutral_count"] + action_info["exclude_count"]
    assert action_info["include_count"] / signal_denominator == pytest.approx(1.0 / 3.0)
    assert action_info["optimizer_name"] == "equal_weight"
    assert (action_info["optimizer_status"] != "success" or action_info["fallback_reason"] is not None) is False

    hold_portfolio_state = PortfolioState(
        date=pd.Timestamp("2024-01-05"),
        nav=1.0,
        portfolio_value=1000000.0,
        current_weights=np.array([0.5, 0.5, 0.0], dtype=float),
        step_index=4,
    )
    hold_action = strategy.compute_target_weights(decision_state, hold_portfolio_state)

    np.testing.assert_allclose(hold_action.target_weights, np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert hold_action.rebalance_action == 0
    assert hold_action.rebalance_intensity == 0.0
    assert hold_action.action_info["estimated_turnover"] == pytest.approx(0.0)
    assert hold_action.action_info["forced_hold_reason"] == "below_rebalance_turnover_threshold"

    fallback_strategy = HybridDQNOptimizerMarkowitzMeanVarianceStrategy(config)
    fallback_strategy.is_fitted = True
    fallback_strategy._asset_action_q_values = lambda decision_input: q_values
    insufficient_history_state = DecisionMarketState(
        **{
            **decision_state.__dict__,
            "log_return_window": decision_state.log_return_window[:10],
        }
    )

    fallback_action = fallback_strategy.compute_target_weights(insufficient_history_state, portfolio_state)
    fallback_info = fallback_action.action_info

    np.testing.assert_allclose(fallback_action.target_weights, np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert fallback_info["paper_model_id"] == "hybrid_dqn_optimizer_markowitz_mean_variance"
    assert fallback_info["child_model_name"] == "hybrid_dqn_optimizer_markowitz_mean_variance"
    assert fallback_info["optimizer_name"] == "markowitz_mean_variance"
    assert fallback_info["selected_asset_count"] == 1
    assert fallback_info["optimizer_asset_count"] == 2
    assert fallback_info["optimizer_status"] == "failed_insufficient_history"
    assert fallback_info["fallback_reason"] == "insufficient_history"
    assert (fallback_info["optimizer_status"] != "success" or fallback_info["fallback_reason"] is not None) is True


def test_validation_penalized_sharpe_alias():
    result = {
        "daily_returns": pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=4),
                "net_return": [0.01, -0.005, 0.02, 0.0],
                "nav": [1.01, 1.00495, 1.025049, 1.025049],
            }
        ),
        "metrics": {"average_turnover": 0.03},
    }

    alias_value = experiment_pipeline.objective_metric(result, "validation_penalized_sharpe")
    canonical_value = experiment_pipeline.objective_metric(
        result,
        "validation_sharpe_minus_drawdown_turnover_penalty",
    )

    assert experiment_pipeline.VALIDATION_METRIC_ALIASES == {
        "validation_penalized_sharpe": "validation_sharpe_minus_drawdown_turnover_penalty"
    }
    assert alias_value == canonical_value
    with pytest.raises(ValueError, match="ERR_EXPERIMENT_METRIC_MISSING"):
        experiment_pipeline.objective_metric({"metrics": {}}, "unknown_metric")
    with pytest.raises(ValueError, match="ERR_EXPERIMENT_METRIC_MISSING"):
        experiment_pipeline.objective_metric({"metrics": {"cumulative_return": 0.123}}, "unknown_metric")


def test_ppo_dqn_initializes_six_hierarchy_actions():
    config = {
        "n_assets": 3,
        "n_features": 2,
        "window_size": 4,
        "latent_dim": 8,
        "encoder": {"type": "mlp", "dropout": 0.0},
        "ppo": {
            "actor_hidden_dims": [8],
            "critic_hidden_dims": [8],
            "min_alpha": 1.0e-3,
        },
        "dqn": {
            "hidden_dims": [8],
            "batch_size": 2,
            "warmup_steps": 0,
            "target_update_interval": 1,
            "use_n_step": False,
            "per_enabled": False,
        },
        "optimizer": {
            "ppo_lr": 1.0e-4,
            "dqn_lr": 1.0e-4,
        },
        "device": {"mode": "cpu"},
    }

    strategy = PPODQNHierarchicalReimplementationStrategy(config)
    market_image = torch.ones(2, 2, 4, 3, dtype=torch.float32, device=strategy.device)
    availability_mask = torch.tensor(
        [[True, True, True], [True, False, True]],
        dtype=torch.bool,
        device=strategy.device,
    )

    latent = strategy.encoder(market_image)
    candidate_weights = strategy.ppo_actor(latent, availability_mask, deterministic=True)
    critic_values = strategy.ppo_critic(latent)
    q_values = strategy.hierarchy_q_network(
        latent,
        candidate_weights,
        candidate_weights,
        torch.zeros(2, 1, device=strategy.device),
        torch.zeros(2, 1, device=strategy.device),
    )

    assert strategy.device.type == "cpu"
    assert strategy.hierarchy_q_network.action_dim == PPO_DQN_HIERARCHY_ACTION_DIM
    assert strategy.target_hierarchy_q_network.action_dim == PPO_DQN_HIERARCHY_ACTION_DIM
    assert candidate_weights.shape == (2, 3)
    assert torch.allclose(candidate_weights.sum(dim=1), torch.ones(2, device=strategy.device))
    assert candidate_weights[1, 1].item() == 0.0
    assert critic_values.shape == (2, 1)
    assert q_values.shape == (2, PPO_DQN_HIERARCHY_ACTION_DIM)
    assert strategy.dqn_agent.online_network is strategy.hierarchy_q_network
    assert strategy.dqn_agent.target_network is strategy.target_hierarchy_q_network
    assert strategy.dqn_agent.replay_buffer is strategy.replay_buffer
    assert strategy.training_result["status"] == "not_started"


def test_ppo_dqn_resolves_all_hierarchy_actions():
    strategy = PPODQNHierarchicalReimplementationStrategy(
        {
            "n_assets": 3,
            "n_features": 1,
            "window_size": 2,
            "latent_dim": 4,
            "encoder": {"type": "mlp", "dropout": 0.0},
            "ppo": {"actor_hidden_dims": [4], "critic_hidden_dims": [4]},
            "dqn": {"hidden_dims": [4], "per_enabled": False, "use_n_step": False},
            "device": {"mode": "cpu"},
        }
    )
    available_mask = np.array([True, True, False], dtype=bool)
    candidate_raw = np.array([2.0, 3.0, 999.0], dtype=float)
    current_raw = np.array([0.2, 0.8, 999.0], dtype=float)
    normalized = strategy._mask_normalize_weights(candidate_raw, available_mask)

    assert normalized["valid"] is True
    np.testing.assert_allclose(normalized["weights"], np.array([0.4, 0.6, 0.0], dtype=np.float32))

    expected = {
        0: (np.array([0.4, 0.6, 0.0], dtype=np.float32), "use_ppo_candidate", 1, 1.0, False),
        1: (np.array([0.25, 0.75, 0.0], dtype=np.float32), "blend_current_ppo_25", 1, 0.25, True),
        2: (np.array([0.30, 0.70, 0.0], dtype=np.float32), "blend_current_ppo_50", 1, 0.50, True),
        3: (np.array([0.35, 0.65, 0.0], dtype=np.float32), "blend_current_ppo_75", 1, 0.75, True),
        4: (np.array([0.50, 0.50, 0.0], dtype=np.float32), "fallback_equal_weight", 0, 0.0, False),
        5: (np.array([0.20, 0.80, 0.0], dtype=np.float32), "hold", 0, 0.0, False),
    }

    for action, (weights, name, actor_mask, attribution_weight, surrogate) in expected.items():
        resolved = strategy._resolve_hierarchy_action(action, candidate_raw, current_raw, available_mask)
        action_info = resolved["action_info"]
        np.testing.assert_allclose(resolved["target_weights"], weights, rtol=1.0e-7, atol=1.0e-7)
        assert resolved["target_weights"][2] == 0.0
        assert float(resolved["target_weights"].sum()) == pytest.approx(1.0)
        assert action_info["hierarchy_action"] == action
        assert action_info["hierarchy_action_name"] == name
        assert action_info["ppo_actor_update_mask"] == actor_mask
        assert action_info["ppo_attribution_weight"] == pytest.approx(attribution_weight)
        assert action_info["platform_adapted_surrogate"] is surrogate


def test_ppo_dqn_policy_hold_action_does_not_rebalance():
    config = {
        "n_assets": 3,
        "n_features": 1,
        "window_size": 2,
        "latent_dim": 4,
        "encoder": {"type": "mlp", "dropout": 0.0},
        "ppo": {"actor_hidden_dims": [4], "critic_hidden_dims": [4]},
        "dqn": {"hidden_dims": [4], "per_enabled": False, "use_n_step": False},
        "device": {"mode": "cpu"},
    }
    strategy = PPODQNHierarchicalReimplementationStrategy(config)
    strategy.is_fitted = True

    class _HoldQ(torch.nn.Module):
        def forward(self, latent, candidate_weights, current_weights, estimated_turnover, estimated_cost):
            q_values = torch.zeros((latent.shape[0], PPO_DQN_HIERARCHY_ACTION_DIM), dtype=torch.float32, device=latent.device)
            q_values[:, PPO_DQN_HIERARCHY_HOLD_ACTION] = 1.0
            return q_values

    strategy.hierarchy_q_network = _HoldQ()
    decision_state = DecisionMarketState(
        decision_date=pd.Timestamp("2024-01-05"),
        available_mask_at_decision=np.array([True, True, False], dtype=bool),
        availability_reason_at_decision=np.array(["listed", "listed", "suspended"], dtype=object),
        close_at_decision=np.ones(3, dtype=float),
        log_return_at_decision=np.zeros(3, dtype=float),
        log_return_window=np.zeros((20, 3), dtype=np.float32),
        amount_at_decision=np.ones(3, dtype=float),
        volume_at_decision=np.ones(3, dtype=float),
        adv20_at_decision=np.ones(3, dtype=float),
        volatility_20d_at_decision=np.ones(3, dtype=float) * 0.02,
        turnover_rate_at_decision=np.ones(3, dtype=float) * 0.01,
        feature_window=np.zeros((1, 2, 3), dtype=float),
        market_image=np.zeros((1, 2, 3), dtype=float),
    )
    portfolio_state = PortfolioState(
        date=pd.Timestamp("2024-01-05"),
        nav=1.0,
        portfolio_value=1000000.0,
        current_weights=np.array([0.2, 0.8, 0.0], dtype=float),
        step_index=3,
    )

    action = strategy.compute_target_weights(decision_state, portfolio_state)

    np.testing.assert_allclose(action.target_weights, portfolio_state.current_weights.astype(np.float32))
    assert action.rebalance_action == 0
    assert action.rebalance_intensity == 0.0
    assert action.action_info["hierarchy_action"] == PPO_DQN_HIERARCHY_HOLD_ACTION
    assert action.action_info["hierarchy_action_name"] == "hold"
    assert action.action_info["forced_hold_reason"] == "model_chosen_hold"


def test_ppo_dqn_invalid_candidate_failure_and_fallback():
    strategy = PPODQNHierarchicalReimplementationStrategy(
        {
            "n_assets": 3,
            "n_features": 1,
            "window_size": 2,
            "latent_dim": 4,
            "encoder": {"type": "mlp", "dropout": 0.0},
            "ppo": {"actor_hidden_dims": [4], "critic_hidden_dims": [4]},
            "dqn": {"hidden_dims": [4], "per_enabled": False, "use_n_step": False},
            "device": {"mode": "cpu"},
        }
    )
    available_mask = np.array([True, True, False], dtype=bool)
    invalid_candidate = np.array([0.8, -0.2, 0.4], dtype=float)
    current_weights = np.array([0.25, 0.75, 0.0], dtype=float)

    for action in (0, 1, 2, 3):
        resolved = strategy._resolve_hierarchy_action(action, invalid_candidate, current_weights, available_mask)
        action_info = resolved["action_info"]
        assert resolved["status"] == "failed_no_valid_action"
        assert resolved["reason"] == "invalid_candidate_weights"
        np.testing.assert_allclose(resolved["target_weights"], np.zeros(3, dtype=np.float32))
        assert action_info["hierarchy_action"] == action
        assert action_info["candidate_weights_valid"] is False
        assert action_info["candidate_invalid_reason"] == "negative_weights"
        assert action_info["ppo_actor_update_mask"] == 0
        assert action_info["ppo_attribution_weight"] == pytest.approx(0.0)
        assert action_info["platform_adapted_surrogate"] is False

    fallback = strategy._resolve_hierarchy_action(4, invalid_candidate, current_weights, available_mask)
    fallback_info = fallback["action_info"]
    np.testing.assert_allclose(fallback["target_weights"], np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert "status" not in fallback
    assert fallback_info["hierarchy_action_name"] == "fallback_equal_weight"
    assert fallback_info["candidate_weights_valid"] is False
    assert fallback_info["candidate_invalid_reason"] == "negative_weights"
    assert fallback_info["fallback_reason"] == "invalid_candidate_weights_equal_weight"
    assert fallback_info["ppo_actor_update_mask"] == 0
    assert fallback_info["ppo_attribution_weight"] == pytest.approx(0.0)

    no_available = strategy._resolve_hierarchy_action(
        4,
        np.array([0.4, 0.6, 0.0], dtype=float),
        current_weights,
        np.array([False, False, False], dtype=bool),
    )
    no_available_info = no_available["action_info"]
    assert no_available["status"] == "failed_no_valid_action"
    assert no_available["reason"] == "no_available_asset"
    np.testing.assert_allclose(no_available["target_weights"], np.zeros(3, dtype=np.float32))
    assert no_available_info["candidate_weights_valid"] is False
    assert no_available_info["candidate_invalid_reason"] == "no_available_asset"
    assert no_available_info["ppo_actor_update_mask"] == 0
    assert no_available_info["ppo_attribution_weight"] == pytest.approx(0.0)


def test_ppo_dqn_action_selection_sanitizes_invalid_weights_before_hierarchy_q(sample_config, monkeypatch):
    config = _related_work_config(sample_config)
    config.update(
        {
            "n_assets": 3,
            "n_features": 1,
            "window_size": 2,
            "latent_dim": 4,
            "encoder": {"type": "mlp", "dropout": 0.0},
            "ppo": {"actor_hidden_dims": [4], "critic_hidden_dims": [4]},
            "dqn": {"hidden_dims": [4], "dropout": 0.0},
            "device": {"mode": "cpu"},
        }
    )
    strategy = PPODQNHierarchicalReimplementationStrategy(config)

    class _InvalidCandidateDistribution:
        @property
        def mean(self):
            return torch.tensor([[float("nan"), -0.2, 0.4]], dtype=torch.float32, device=strategy.device)

        def sample(self):
            return self.mean

        def log_prob(self, _value):
            raise AssertionError("invalid candidate log_prob should be skipped")

    class _FallbackQ(torch.nn.Module):
        def forward(self, latent, candidate_weights, current_weights, estimated_turnover, estimated_cost):
            assert torch.isfinite(latent).all()
            assert torch.isfinite(candidate_weights).all()
            assert torch.isfinite(current_weights).all()
            assert torch.isfinite(estimated_turnover).all()
            assert torch.isfinite(estimated_cost).all()
            return torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=latent.device)

    monkeypatch.setattr(strategy.ppo_actor, "get_distribution", lambda _latent, _mask: _InvalidCandidateDistribution())
    strategy.hierarchy_q_network = _FallbackQ()
    observation = {
        "market_image": np.zeros((1, 2, 3), dtype=np.float32),
        "current_weights": np.array([float("nan"), 0.5, 0.5], dtype=np.float32),
        "availability_mask": np.array([1, 1, 0], dtype=np.int8),
    }

    decision = strategy._select_training_action(observation, deterministic=True)

    resolved = decision["resolved"]
    action_info = resolved["action_info"]
    np.testing.assert_allclose(resolved["target_weights"], np.array([0.5, 0.5, 0.0], dtype=np.float32))
    assert action_info["hierarchy_action"] == 4
    assert action_info["candidate_weights_valid"] is False
    assert action_info["candidate_invalid_reason"] == "non_finite_weights"
    assert action_info["fallback_reason"] == "invalid_candidate_weights_equal_weight"
    assert action_info["ppo_actor_update_mask"] == 0
    assert action_info["ppo_attribution_weight"] == pytest.approx(0.0)
    assert torch.equal(decision["candidate_log_prob"], torch.zeros(1, dtype=torch.float32, device=strategy.device))


def test_ppo_dqn_fit_uses_real_transition_history(
    sample_market_dataset_bundle,
    sample_trade_dates,
    sample_config,
    tmp_path,
):
    config = _related_work_config(sample_config)
    config.update(
        {
            "n_assets": 2,
            "n_features": 1,
            "window_size": 3,
            "latent_dim": 4,
            "encoder": {"type": "mlp", "dropout": 0.0},
            "ppo": {"actor_hidden_dims": [4], "critic_hidden_dims": [4]},
            "dqn": {
                "hidden_dims": [4],
                "batch_size": 1,
                "warmup_steps": 0,
                "target_update_interval": 1,
                "per_enabled": False,
                "use_n_step": False,
            },
            "optimizer": {"ppo_lr": 1.0e-3, "dqn_lr": 1.0e-3},
            "baselines": {
                "native_rl": {
                    "epochs": 1,
                    "max_train_steps": 2,
                    "max_validation_steps": 2,
                    "max_gradient_updates_per_epoch": 2,
                }
            },
            "device": {"mode": "cpu"},
            "baseline_run_dir": str(tmp_path / "ppo_dqn_fit"),
        }
    )
    config["feature_matrix"]["window_size"] = 3
    train_payload = {
        "dataset": sample_market_dataset_bundle,
        "dates": sample_trade_dates[:4],
        "config": config,
    }
    validation_payload = {
        "dataset": sample_market_dataset_bundle,
        "dates": sample_trade_dates[2:5],
        "config": config,
    }
    strategy = PPODQNHierarchicalReimplementationStrategy(config)
    before_q = [parameter.detach().clone() for parameter in strategy.hierarchy_q_network.parameters()]
    before_critic = [parameter.detach().clone() for parameter in strategy.ppo_critic.parameters()]

    strategy.fit(train_payload, validation_payload)

    assert strategy.is_fitted is True
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["rankable_in_unified_table"] is True
    assert np.isfinite(strategy.training_result["best_validation_metric"])
    assert strategy.training_result["checkpoint_best_path"] is not None
    assert strategy.training_result["checkpoint_last_path"] is not None
    assert strategy.training_result["evaluated_checkpoint_path"] == strategy.training_result["checkpoint_best_path"]
    assert (tmp_path / "ppo_dqn_fit" / "checkpoints" / "ppo_dqn_hierarchical_reimplementation" / "best.pt").exists()
    assert strategy.training_result["env_steps"] == 2
    assert strategy.training_result["gradient_updates"] > 0
    assert len(strategy.replay_buffer) > 0
    assert {"epoch", "env_steps", "gradient_updates", "train_reward", "validation_metric", "loss", "status"}.issubset(
        strategy.training_history.columns
    )
    assert np.isfinite(pd.to_numeric(strategy.training_history["validation_metric"], errors="coerce")).all()
    assert strategy.training_history["status"].eq("completed").all()
    item = strategy.replay_buffer.items[0]
    assert item.next_state_source_t == "env"
    assert isinstance(item.state_tp1, dict)
    assert "market_image" in item.state_tp1
    assert item.decision_date_next == sample_trade_dates[1]
    assert item.execution_date_next == sample_trade_dates[2]
    assert item.next_valuation_date_next == sample_trade_dates[2]
    assert item.split_boundary_t is False
    assert item.gate_action_t in range(PPO_DQN_HIERARCHY_ACTION_DIM)
    assert np.isfinite(item.reward_t)
    after_q = list(strategy.hierarchy_q_network.parameters())
    after_critic = list(strategy.ppo_critic.parameters())
    assert any(not torch.allclose(left, right.detach()) for left, right in zip(before_q, after_q, strict=True))
    assert any(not torch.allclose(left, right.detach()) for left, right in zip(before_critic, after_critic, strict=True))

    missing = PPODQNHierarchicalReimplementationStrategy(config)
    missing.fit(None, validation_payload)
    assert missing.is_fitted is False
    assert missing.training_result["status"] == "failed_missing_train_data"
    assert missing.training_result["checkpoint_best_path"] is None
    assert missing.training_result["checkpoint_last_path"] is None


def test_ppo_dqn_loads_best_checkpoint_before_test(
    sample_market_dataset_bundle,
    sample_trade_dates,
    sample_config,
    tmp_path,
    monkeypatch,
):
    validation_metric_calls = []

    def fake_objective_metric(payload, metric):
        assert metric == "validation_sharpe_minus_drawdown_turnover_penalty"
        daily_returns = payload["daily_returns"]
        assert {"net_return", "nav"}.issubset(daily_returns.columns)
        assert daily_returns["split"].eq("validation").all()
        assert "average_turnover" in payload["metrics"]
        validation_metric_calls.append(daily_returns[["net_return", "nav"]].copy())
        return 7.25

    monkeypatch.setattr(experiment_pipeline, "objective_metric", fake_objective_metric)
    config = _related_work_config(sample_config)
    config.update(
        {
            "n_assets": 2,
            "n_features": 1,
            "window_size": 3,
            "latent_dim": 4,
            "encoder": {"type": "mlp", "dropout": 0.0},
            "ppo": {"actor_hidden_dims": [4], "critic_hidden_dims": [4]},
            "dqn": {
                "hidden_dims": [4],
                "batch_size": 1,
                "warmup_steps": 0,
                "target_update_interval": 1,
                "per_enabled": False,
                "use_n_step": False,
            },
            "optimizer": {"ppo_lr": 1.0e-3, "dqn_lr": 1.0e-3},
            "baselines": {
                "native_rl": {
                    "epochs": 2,
                    "max_train_steps": 2,
                    "max_validation_steps": 2,
                    "max_gradient_updates_per_epoch": 2,
                }
            },
            "device": {"mode": "cpu"},
            "baseline_run_dir": str(tmp_path / "ppo_dqn_checkpoint"),
        }
    )
    config["feature_matrix"]["window_size"] = 3
    split = SplitSpec(
        train_dates=sample_trade_dates[:3],
        validation_dates=sample_trade_dates[2:5],
        test_dates=sample_trade_dates[4:],
        fold_id="fixed",
    )
    strategy = PPODQNHierarchicalReimplementationStrategy(config)

    result = BacktestEngine(config).run(sample_market_dataset_bundle, split, strategy, segment="test")

    best_path = strategy.training_result["checkpoint_best_path"]
    last_path = strategy.training_result["checkpoint_last_path"]
    assert strategy.training_result["status"] == "completed"
    assert strategy.training_result["evaluated_checkpoint_path"] == best_path
    assert best_path is not None and last_path is not None
    assert (tmp_path / "ppo_dqn_checkpoint" / "checkpoints" / "ppo_dqn_hierarchical_reimplementation" / "best.pt").exists()
    assert (tmp_path / "ppo_dqn_checkpoint" / "checkpoints" / "ppo_dqn_hierarchical_reimplementation" / "last.pt").exists()
    payload = load_checkpoint(best_path, device="cpu", restore_rng_state=False)
    assert payload["best_validation_metric"] == pytest.approx(strategy.training_result["best_validation_metric"])
    assert strategy.training_result["best_validation_metric"] == pytest.approx(7.25)
    assert validation_metric_calls
    assert not result.daily_returns.empty
    assert result.daily_returns["model_name"].eq("ppo_dqn_hierarchical_reimplementation").all()


def test_ppo_dqn_checkpoint_failure_statuses(
    sample_market_dataset_bundle,
    sample_trade_dates,
    sample_config,
    tmp_path,
    monkeypatch,
):
    config = _related_work_config(sample_config)
    config.update(
        {
            "n_assets": 2,
            "n_features": 1,
            "window_size": 3,
            "latent_dim": 4,
            "encoder": {"type": "mlp", "dropout": 0.0},
            "ppo": {"actor_hidden_dims": [4], "critic_hidden_dims": [4]},
            "dqn": {
                "hidden_dims": [4],
                "batch_size": 1,
                "warmup_steps": 0,
                "target_update_interval": 1,
                "per_enabled": False,
                "use_n_step": False,
            },
            "optimizer": {"ppo_lr": 1.0e-3, "dqn_lr": 1.0e-3},
            "baselines": {
                "native_rl": {
                    "epochs": 1,
                    "max_train_steps": 2,
                    "max_validation_steps": 2,
                    "max_gradient_updates_per_epoch": 2,
                }
            },
            "device": {"mode": "cpu"},
        }
    )
    config["feature_matrix"]["window_size"] = 3
    train_payload = {
        "dataset": sample_market_dataset_bundle,
        "dates": sample_trade_dates[:4],
        "config": config,
    }
    short_validation_payload = {
        "dataset": sample_market_dataset_bundle,
        "dates": sample_trade_dates[2:3],
        "config": config,
    }
    missing_validation = PPODQNHierarchicalReimplementationStrategy(config)
    missing_validation.fit(train_payload, short_validation_payload)
    assert missing_validation.is_fitted is False
    assert missing_validation.training_result["status"] == "failed_no_finite_validation_metric"
    assert missing_validation.training_result["checkpoint_best_path"] is None

    validation_payload = {
        "dataset": sample_market_dataset_bundle,
        "dates": sample_trade_dates[2:5],
        "config": config,
    }
    missing_best = PPODQNHierarchicalReimplementationStrategy(config)
    missing_best.fit(train_payload, validation_payload)
    assert missing_best.is_fitted is False
    assert missing_best.training_result["status"] == "failed_missing_best_checkpoint"
    assert missing_best.training_result["rankable_in_unified_table"] is False
    assert missing_best.training_result["checkpoint_best_path"] is None

    corrupt_config = deepcopy(config)
    corrupt_config["baseline_run_dir"] = str(tmp_path / "corrupt_checkpoint")

    def raise_load(self, path):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr(PPODQNHierarchicalReimplementationStrategy, "_load_checkpoint_state", raise_load)
    failed_load = PPODQNHierarchicalReimplementationStrategy(corrupt_config)
    failed_load.fit({**train_payload, "config": corrupt_config}, {**validation_payload, "config": corrupt_config})
    assert failed_load.is_fitted is False
    assert failed_load.training_result["status"] == "failed_checkpoint_load"
    assert failed_load.training_result["checkpoint_best_path"] is not None


def test_ppo_dqn_action_info_contract(
    sample_market_dataset_bundle,
    sample_trade_dates,
    sample_config,
    tmp_path,
):
    config = _related_work_config(sample_config)
    config.update(
        {
            "n_assets": 2,
            "n_features": 1,
            "window_size": 3,
            "latent_dim": 4,
            "encoder": {"type": "mlp", "dropout": 0.0},
            "ppo": {"actor_hidden_dims": [4], "critic_hidden_dims": [4]},
            "dqn": {
                "hidden_dims": [4],
                "batch_size": 1,
                "warmup_steps": 0,
                "target_update_interval": 1,
                "per_enabled": False,
                "use_n_step": False,
            },
            "optimizer": {"ppo_lr": 1.0e-3, "dqn_lr": 1.0e-3},
            "baselines": {
                "native_rl": {
                    "epochs": 1,
                    "max_train_steps": 2,
                    "max_validation_steps": 2,
                    "max_gradient_updates_per_epoch": 2,
                }
            },
            "device": {"mode": "cpu"},
            "baseline_run_dir": str(tmp_path / "ppo_dqn_action_info"),
        }
    )
    config["feature_matrix"]["window_size"] = 3
    split = SplitSpec(
        train_dates=sample_trade_dates[:3],
        validation_dates=sample_trade_dates[2:5],
        test_dates=sample_trade_dates[4:],
        fold_id="fixed",
    )
    required = {
        "paper_model_id",
        "hierarchy_action",
        "hierarchy_action_name",
        "ppo_actor_update_mask",
        "ppo_attribution_weight",
        "platform_adapted_surrogate",
    }
    strategy = PPODQNHierarchicalReimplementationStrategy(config)
    strategy.fit(
        {"dataset": sample_market_dataset_bundle, "dates": split.train_dates, "config": config},
        {"dataset": sample_market_dataset_bundle, "dates": split.validation_dates, "config": config},
    )
    decision_state = _build_decision_market_state(sample_market_dataset_bundle, sample_trade_dates[4], config)
    portfolio_state = PortfolioState(
        date=sample_trade_dates[4],
        nav=1.0,
        portfolio_value=1000000.0,
        current_weights=np.array([0.5, 0.5], dtype=float),
    )

    action = strategy.compute_target_weights(decision_state, portfolio_state)

    assert isinstance(action, PortfolioAction)
    assert required.issubset(action.action_info)
    assert "gate_action" not in action.action_info
    assert "rebalance_gate" not in action.action_info
    assert action.action_info["paper_model_id"] == "ppo_dqn_hierarchical_reimplementation"
    assert action.action_info["hierarchy_action"] in PPO_DQN_HIERARCHY_ACTION_NAMES
    assert action.action_info["hierarchy_action_name"] == PPO_DQN_HIERARCHY_ACTION_NAMES[action.action_info["hierarchy_action"]]
    assert action.action_info["ppo_actor_update_mask"] in {0, 1}
    assert np.isfinite(action.action_info["ppo_attribution_weight"])
    assert isinstance(action.action_info["platform_adapted_surrogate"], bool)
    assert action.target_weights.shape == (2,)
    assert np.isfinite(action.target_weights).all()
    assert np.all(action.target_weights[~decision_state.available_mask_at_decision] == 0.0)

    result = BacktestEngine(config).run(
        sample_market_dataset_bundle,
        split,
        PPODQNHierarchicalReimplementationStrategy(config),
        segment="test",
    )
    assert required.issubset(result.baseline_daily_diagnostics.columns)
    assert "gate_action" not in result.baseline_daily_diagnostics.columns
    assert "rebalance_gate" not in result.baseline_daily_diagnostics.columns
    assert not result.daily_weights.empty


def test_ppo_dqn_metadata_contract(
    sample_market_dataset_bundle,
    sample_trade_dates,
    sample_config,
    tmp_path,
):
    config = deepcopy(sample_config)
    config["model"] = {
        "encoder_type": "cnn",
        "n_assets": 2,
        "n_features": 1,
        "window_size": 3,
        "latent_dim": 8,
        "dqn": {"hidden_dims": [8], "batch_size": 1, "warmup_steps": 1, "target_update_interval": 1},
        "ppo": {"hidden_dims": [8]},
    }
    config["env"]["window_size"] = 3
    config["window_size"] = 3
    config["execution_model"]["delayed_action_execution"] = False
    config["execution_model"]["initial_build_cost"] = True
    config["cost_model"]["market_impact_enabled"] = False
    config["rebalance"]["mode"] = "daily"
    config["baselines"] = {
        "native_rl": {"epochs": 1, "max_train_steps": 1, "max_validation_steps": 2, "max_gradient_updates_per_epoch": 1},
        "checkpoint_dir": str(tmp_path / "ppo_dqn_metadata"),
    }
    split = SplitSpec(
        train_dates=sample_trade_dates[:3],
        validation_dates=sample_trade_dates[2:5],
        test_dates=sample_trade_dates[4:],
        fold_id="fixed",
    )
    model_name = "ppo_dqn_hierarchical_reimplementation"
    strategy = PPODQNHierarchicalReimplementationStrategy(config)
    result = BacktestEngine(config).run(sample_market_dataset_bundle, split, strategy, segment="test")
    summary_row, _ = experiment_pipeline._baseline_training_artifacts(
        model_name,
        strategy,
        result.baseline_daily_diagnostics,
    )
    comparison = _comparison_rows({model_name: dict(result.metrics)}, training_summary_rows=[summary_row])
    row = comparison.iloc[0]

    assert strategy.training_result["status"] == "completed"
    assert strategy.is_fitted is True
    assert bool(strategy.training_result["proxy_training"]) is False
    assert bool(strategy.training_result["external_original_implementation"]) is False
    assert bool(strategy.training_result["rankable_in_unified_table"]) is True
    assert bool(summary_row["clean_room_reimplementation"]) is True
    assert summary_row["baseline_family"] == "native_rl_reimplementation"
    assert summary_row["algorithm_fidelity"] == "platform_adapted"
    assert summary_row["dqn_role"] == "high_level_action_selector"
    assert bool(summary_row["proxy_training"]) is False
    assert bool(summary_row["external_original_implementation"]) is False
    assert bool(row["rankable_in_unified_table"]) is True
    assert row["baseline_family"] == "native_rl_reimplementation"
    assert row["training_algorithm"] == model_name
    assert bool(row["clean_room_reimplementation"]) is True
    assert row["algorithm_fidelity"] == "platform_adapted"
    assert row["dqn_role"] == "high_level_action_selector"
    counts = [int(row[f"hierarchy_action_{index}_count"]) for index in range(PPO_DQN_HIERARCHY_ACTION_DIM)]
    assert sum(counts) == len(result.baseline_daily_diagnostics)
    assert json.loads(row["hierarchy_action_distribution"]) == {
        str(index): counts[index] for index in range(PPO_DQN_HIERARCHY_ACTION_DIM)
    }
    aggregate_diagnostics = pd.concat(
        [result.baseline_daily_diagnostics, result.baseline_daily_diagnostics],
        ignore_index=True,
    )
    aggregate_result = {
        "metrics": dict(result.metrics),
        "baseline_daily_diagnostics": aggregate_diagnostics,
        "baseline_training_summary": pd.DataFrame([summary_row]),
        "baseline_comparison": comparison.copy(),
    }
    experiment_pipeline._refresh_ppo_dqn_aggregate_comparison(aggregate_result, model_name)
    aggregate_row = aggregate_result["baseline_comparison"].iloc[0]
    aggregate_counts = [
        int(aggregate_row[f"hierarchy_action_{index}_count"]) for index in range(PPO_DQN_HIERARCHY_ACTION_DIM)
    ]
    assert sum(aggregate_counts) == len(aggregate_diagnostics)

    deferred_config = deepcopy(config)
    deferred_config["dqn_role"] = "rebalance_gate"
    deferred = PPODQNHierarchicalReimplementationStrategy(deferred_config)
    deferred.fit(
        {"dataset": sample_market_dataset_bundle, "dates": split.train_dates, "config": deferred_config},
        {"dataset": sample_market_dataset_bundle, "dates": split.validation_dates, "config": deferred_config},
    )
    deferred_summary, _ = experiment_pipeline._baseline_training_artifacts(model_name, deferred)
    deferred_comparison = _comparison_rows({model_name: {"cumulative_return": 0.0}}, training_summary_rows=[deferred_summary])
    deferred_row = deferred_comparison.iloc[0]
    assert deferred.training_result["status"] == "deferred_variant"
    assert deferred.is_fitted is False
    assert bool(deferred.training_result["rankable_in_unified_table"]) is False
    assert bool(deferred_row["rankable_in_unified_table"]) is False
    assert deferred_row["status"] == "deferred_variant"
    assert deferred_row["dqn_role"] == "rebalance_gate"


def test_related_work_metadata_flows_to_comparison():
    ppo_model_name = "ppo_dqn_hierarchical_reimplementation"
    ppo_comparison = _comparison_rows({ppo_model_name: {"cumulative_return": 0.10}})
    ppo_row = ppo_comparison.iloc[0]
    assert ppo_row["paper_model_id"] == ppo_model_name
    assert ppo_row["child_model_name"] == ppo_model_name
    assert ppo_row["baseline_family"] == "native_rl_reimplementation"
    assert ppo_row["training_algorithm"] == ppo_model_name
    assert ppo_row["dqn_role"] == "high_level_action_selector"
    assert bool(ppo_row["platform_adapted_surrogate"]) is False
    assert bool(ppo_row["rankable_in_unified_table"]) is True

    child_model_name = "hybrid_dqn_optimizer_equal_weight"
    hybrid_comparison = _comparison_rows({child_model_name: {"cumulative_return": 0.20}})
    hybrid_row = hybrid_comparison.iloc[0]
    assert hybrid_row["paper_model_id"] == child_model_name
    assert hybrid_row["child_model_name"] == child_model_name
    assert hybrid_row["baseline_family"] == "native_rl_reimplementation"
    assert hybrid_row["training_algorithm"] == "factorized_dqn_signal_plus_portfolio_optimizer"
    assert hybrid_row["optimizer_name"] == "equal_weight"
    assert hybrid_row["optimizer_allocation_method"] == "equal_weight"
    assert bool(hybrid_row["factorized_q"]) is True
    assert bool(hybrid_row["portfolio_level_reward_shared"]) is True
    assert bool(hybrid_row["counterfactual_asset_reward"]) is False
    assert bool(hybrid_row["platform_adapted_approximation"]) is True
    assert bool(hybrid_row["rankable_in_unified_table"]) is True

    config = {
        "n_assets": 3,
        "n_features": 1,
        "window_size": 2,
        "latent_dim": 4,
        "encoder": {"type": "mlp", "dropout": 0.0},
        "dqn": {"hidden_dims": [4], "dropout": 0.0},
        "device": {"mode": "cpu"},
    }
    strategy = HybridDQNOptimizerEqualWeightStrategy(config)
    training_history = pd.DataFrame(
        [
            {
                "epoch": 0,
                "env_steps": 2,
                "gradient_updates": 2,
                "validation_metric": 1.25,
                "include_count": 3,
                "neutral_count": 1,
                "exclude_count": 2,
                "selected_asset_count": 2,
                "optimizer_asset_count": 2,
                "optimizer_fallback_count": 1,
                "status": "completed",
            }
        ]
    )
    strategy.training_history = training_history
    strategy.training_result = {
        "model_name": HYBRID_DQN_OPTIMIZER_ALIAS,
        "paper_model_id": child_model_name,
        "child_model_name": child_model_name,
        "baseline_family": "native_rl_reimplementation",
        "status": "completed",
        "training_algorithm": "factorized_dqn_signal_plus_portfolio_optimizer",
        "optimizer_name": "equal_weight",
        "rl_training": True,
        "platform_native_rl_training": True,
        "proxy_training": False,
        "external_original_implementation": False,
        "rankable_in_unified_table": True,
        "clean_room_reimplementation": True,
        "algorithm_fidelity": "platform_adapted",
        "factorized_q": True,
        "portfolio_level_reward_shared": True,
        "counterfactual_asset_reward": False,
        "platform_adapted_approximation": True,
        "checkpoint_best_path": "/tmp/best.pt",
        "checkpoint_last_path": "/tmp/last.pt",
        "evaluated_checkpoint_path": "/tmp/best.pt",
        "best_validation_metric": 1.25,
        "env_steps": 2,
        "gradient_updates": 2,
        "training_history": training_history,
    }
    diagnostics = pd.DataFrame(
        [
            {
                "paper_model_id": child_model_name,
                "child_model_name": child_model_name,
                "optimizer_name": "equal_weight",
                "include_count": 1,
                "neutral_count": 1,
                "exclude_count": 1,
                "optimizer_status": "success",
                "fallback_reason": None,
            },
            {
                "paper_model_id": child_model_name,
                "child_model_name": child_model_name,
                "optimizer_name": "equal_weight",
                "include_count": 2,
                "neutral_count": 0,
                "exclude_count": 1,
                "optimizer_status": "failed_insufficient_history",
                "fallback_reason": "insufficient_history",
            },
        ]
    )

    summary_row, history_frame = experiment_pipeline._baseline_training_artifacts(
        child_model_name,
        strategy,
        diagnostics,
    )
    assert summary_row["model_name"] == child_model_name
    assert summary_row["paper_model_id"] == child_model_name
    assert summary_row["checkpoint_best_path"] == "/tmp/best.pt"
    assert summary_row["evaluated_checkpoint_path"] == "/tmp/best.pt"
    assert summary_row["dqn_signal_include_rate"] == pytest.approx(0.50)
    assert summary_row["optimizer_allocation_method"] == "equal_weight"
    assert summary_row["optimizer_fallback_rate"] == pytest.approx(0.50)
    assert history_frame is not None
    assert "include_count" in history_frame.columns

    hybrid_comparison = _comparison_rows(
        {child_model_name: {"cumulative_return": 0.20}},
        training_summary_rows=[summary_row],
    )
    hybrid_row = hybrid_comparison.iloc[0]
    assert hybrid_row["checkpoint_best_path"] == "/tmp/best.pt"
    assert hybrid_row["best_validation_metric"] == pytest.approx(1.25)
    assert hybrid_row["dqn_signal_include_rate"] == pytest.approx(0.50)
    assert hybrid_row["optimizer_fallback_rate"] == pytest.approx(0.50)

    failed_comparison = _comparison_rows(
        {child_model_name: {"cumulative_return": 0.0}},
        training_summary_rows=[
            {
                "model_name": child_model_name,
                "paper_model_id": child_model_name,
                "status": "failed_no_finite_validation_metric",
                "rankable_in_unified_table": False,
            }
        ],
    )
    assert failed_comparison.iloc[0]["status"] == "failed_no_finite_validation_metric"
    assert bool(failed_comparison.iloc[0]["rankable_in_unified_table"]) is False

    mismatch_diagnostics = diagnostics.assign(
        paper_model_id="hybrid_dqn_optimizer_risk_parity",
        child_model_name="hybrid_dqn_optimizer_risk_parity",
    )
    mismatch_summary, _ = experiment_pipeline._baseline_training_artifacts(
        child_model_name,
        strategy,
        mismatch_diagnostics,
    )
    assert pd.isna(mismatch_summary["dqn_signal_include_rate"])
    assert pd.isna(mismatch_summary["optimizer_fallback_rate"])

    no_identity_summary, _ = experiment_pipeline._baseline_training_artifacts(
        child_model_name,
        strategy,
        diagnostics.drop(columns=["paper_model_id", "child_model_name"]),
    )
    assert pd.isna(no_identity_summary["dqn_signal_include_rate"])
    assert pd.isna(no_identity_summary["optimizer_fallback_rate"])

    child_only_summary, _ = experiment_pipeline._baseline_training_artifacts(
        child_model_name,
        strategy,
        diagnostics.drop(columns=["paper_model_id"]),
    )
    assert pd.isna(child_only_summary["dqn_signal_include_rate"])
    assert pd.isna(child_only_summary["optimizer_fallback_rate"])

    conflicting_identity = diagnostics.assign(child_model_name="hybrid_dqn_optimizer_risk_parity")
    conflict_summary, _ = experiment_pipeline._baseline_training_artifacts(
        child_model_name,
        strategy,
        conflicting_identity,
    )
    assert pd.isna(conflict_summary["dqn_signal_include_rate"])
    assert pd.isna(conflict_summary["optimizer_fallback_rate"])

    aggregate_result = {
        "metrics": {"cumulative_return": 0.20},
        "baseline_daily_diagnostics": diagnostics,
        "baseline_training_summary": pd.DataFrame([summary_row]),
        "baseline_comparison": hybrid_comparison.copy(),
    }
    experiment_pipeline._refresh_ppo_dqn_aggregate_comparison(aggregate_result, child_model_name)
    aggregate_row = aggregate_result["baseline_comparison"].iloc[0]
    assert aggregate_row["paper_model_id"] == child_model_name
    assert aggregate_row["optimizer_name"] == "equal_weight"
    assert aggregate_row["dqn_signal_include_rate"] == pytest.approx(0.50)
    assert aggregate_row["optimizer_fallback_rate"] == pytest.approx(0.50)

    blocked_summary = {"rankable_in_unified_table": True}
    experiment_pipeline._block_rankable_without_paper_model_id(
        child_model_name,
        pd.DataFrame([{"paper_model_id": child_model_name}, {"paper_model_id": None}]),
        blocked_summary,
    )
    assert bool(blocked_summary["rankable_in_unified_table"]) is False
    assert blocked_summary["diagnostic_status"] == "missing_paper_model_id"

    mismatch_summary = {"rankable_in_unified_table": True}
    experiment_pipeline._block_rankable_without_paper_model_id(
        child_model_name,
        mismatch_diagnostics,
        mismatch_summary,
    )
    assert bool(mismatch_summary["rankable_in_unified_table"]) is False

    valid_summary = {"rankable_in_unified_table": True}
    experiment_pipeline._block_rankable_without_paper_model_id(
        child_model_name,
        diagnostics,
        valid_summary,
    )
    assert bool(valid_summary["rankable_in_unified_table"]) is True


class _RelatedWorkDiagnosticsStrategy(BaseStrategy):
    strategy_name = "ppo_dqn_hierarchical_reimplementation"

    def compute_target_weights(self, decision_market_state, portfolio_state):
        self.validate_decision_market_state(decision_market_state)
        self.validate_portfolio_state(portfolio_state)
        return PortfolioAction(
            np.array([0.5, 0.5], dtype=float),
            1,
            1.0,
            {
                "paper_model_id": self.strategy_name,
                "hierarchy_action": 2,
                "hierarchy_action_name": "blend_current_ppo_50",
                "ppo_actor_update_mask": 1,
                "ppo_attribution_weight": 0.5,
                "platform_adapted_surrogate": True,
            },
        )


class _FailedFitRequiredRelatedWorkStrategy(BaseStrategy):
    strategy_name = "ppo_dqn_hierarchical_reimplementation"
    fit_required = True

    def __init__(self, status):
        super().__init__({})
        self.status = str(status)
        self.fit_calls = 0
        self.compute_calls = 0
        self.training_result = {"status": "not_started"}

    def fit(self, train_data=None, validation_data=None):
        self.fit_calls += 1
        self.is_fitted = False
        self.training_result = {
            "model_name": self.strategy_name,
            "paper_model_id": self.strategy_name,
            "status": self.status,
            "rankable_in_unified_table": False,
        }
        return self

    def compute_target_weights(self, decision_market_state, portfolio_state):
        self.compute_calls += 1
        return PortfolioAction(np.array([0.5, 0.5], dtype=float), 1, 1.0)


def _related_work_config(sample_config):
    config = deepcopy(sample_config)
    config["env"]["window_size"] = 3
    config["execution_model"]["execution_price"] = "next_open"
    config["execution_model"]["delayed_action_execution"] = False
    config["execution_model"]["initial_build_cost"] = True
    config["cost_model"]["market_impact_enabled"] = False
    config["rebalance"]["mode"] = "daily"
    return config


def _assert_related_work_sidecar(frame):
    required = {"date", "decision_date", "execution_date", "model_name", "paper_model_id", "seed", "fold_id"}
    assert required.issubset(frame.columns)
    assert not frame.empty
    assert frame["paper_model_id"].eq("ppo_dqn_hierarchical_reimplementation").all()
    assert frame["model_name"].eq("ppo_dqn_hierarchical_reimplementation").all()
    assert frame["fold_id"].eq("fixed").all()
    assert frame["hierarchy_action"].eq(2).all()
    assert frame["platform_adapted_surrogate"].eq(True).all()


def _artifact_stub(dataset, split):
    feature_matrix = SimpleNamespace(
        provenance=pd.DataFrame(),
        feature_group_summary=pd.DataFrame(),
        metrics_factory_audit_sample=pd.DataFrame(),
        feature_cols=[],
    )
    market_image_dataset = _MarketImageDatasetStub(len(dataset.wide["close"].index))
    return {
        "dataset": dataset,
        "feature_matrix": feature_matrix,
        "feature_reduction": SimpleNamespace(is_fitted=False),
        "split": split,
        "market_image_dataset": market_image_dataset,
        "feature_config": {"feature_matrix": {"input_matrix_id": "test"}},
        "requested_feature_config": {"feature_matrix": {"input_matrix_id": "test"}},
        "feature_fallback_reason": None,
        "reduced_train_rows": 0,
    }


class _MarketImageDatasetStub:
    feature_cols = ["log_return"]

    def __init__(self, size):
        self.size = int(size)

    def __len__(self):
        return self.size
