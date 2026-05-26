import csv
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import src.experiments.run_experiment as run_experiment
import src.experiments.pipeline as pipeline
from src.config import ConfigError, ConfigLoader, DEFAULT_CONFIG, PROJECT_ROOT, VALID_EXPERIMENT_TYPES
from src.data.loader import MarketDatasetBundle
from src.data.splits import SplitSpec
from src.envs.constraint_manager import ConstraintManager
from src.envs.cost_model import CostModel
from src.envs.portfolio_execution_core import PortfolioExecutionCore
from src.experiments.registry import (
    AblationExperiment,
    BaselineComparisonExperiment,
    ExperimentRegistry,
    FullReproductionExperiment,
    HPOExperiment,
    HYBRID_DQN_OPTIMIZER_ALIAS,
    HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
    MainModelExperiment,
    ModuleAnalysisExperiment,
    SensitivityExperiment,
    WalkForwardExperiment,
    _ablation_variants,
    _baseline_factories,
    _sensitivity_variants,
)
from src.experiments.aggregate_results import aggregate_walk_forward
from src.experiments.pipeline import _model_class
from src.experiments.paper_full import run_paper_full
from src.experiments.run_all import FULL_REPRODUCTION_SEQUENCE, run_experiment_matrix
from src.models.distributional_cvar_gated_ppo import DistributionalCVaRGatedPPO
from src.models.partial_rebalance_gated_ppo import PartialRebalanceGatedPPO
from src.models.preference_conditioned_gated_ppo import PreferenceConditionedGatedPPO
from src.models.uncertainty_aware_gated_ppo import UncertaintyAwareGatedPPO


def test_registry_all_experiment_types(tmp_path):
    expected_classes = {
        "main_model": MainModelExperiment,
        "baseline_comparison": BaselineComparisonExperiment,
        "ablation": AblationExperiment,
        "input_matrix_ablation": AblationExperiment,
        "pca_ablation": AblationExperiment,
        "kernel_size_ablation": AblationExperiment,
        "reward_ablation": AblationExperiment,
        "walk_forward": WalkForwardExperiment,
        "transaction_cost_sensitivity": SensitivityExperiment,
        "asset_universe_sensitivity": SensitivityExperiment,
        "market_regime": SensitivityExperiment,
        "seed_stability": SensitivityExperiment,
        "hyperparameter_sweep": HPOExperiment,
        "auxiliary_task_sensitivity": SensitivityExperiment,
        "rebalance_frequency_analysis": SensitivityExperiment,
        "preference_conditioned_analysis": ModuleAnalysisExperiment,
        "uncertainty_analysis": ModuleAnalysisExperiment,
        "distributional_cvar_analysis": ModuleAnalysisExperiment,
        "partial_rebalance_analysis": ModuleAnalysisExperiment,
        "full_reproduction": FullReproductionExperiment,
    }
    expected_outputs = {
        "main_model": "main_comparison",
        "baseline_comparison": "baseline_comparison",
        "ablation": "ablation_results",
        "input_matrix_ablation": "input_matrix_ablation_results",
        "pca_ablation": "PCA_ablation_results",
        "kernel_size_ablation": "kernel_size_ablation_results",
        "reward_ablation": "reward_ablation_results",
        "walk_forward": "walk_forward_results",
        "transaction_cost_sensitivity": "transaction_cost_sensitivity",
        "asset_universe_sensitivity": "asset_universe_sensitivity",
        "market_regime": "market_regime_results",
        "seed_stability": "seed_stability",
        "hyperparameter_sweep": "hpo_trials",
        "auxiliary_task_sensitivity": "auxiliary_task_sensitivity",
        "rebalance_frequency_analysis": "rebalance_frequency_analysis",
        "preference_conditioned_analysis": "preference_conditioned_results",
        "uncertainty_analysis": "uncertainty_results",
        "distributional_cvar_analysis": "distributional_cvar_results",
        "partial_rebalance_analysis": "partial_rebalance_results",
        "full_reproduction": "full_reproduction_summary",
    }
    registry = ExperimentRegistry()

    created = {}
    for experiment_type in sorted(VALID_EXPERIMENT_TYPES):
        config = _config(tmp_path, experiment_type)
        experiment = registry.create_experiment(config, device="cpu", run_dir=tmp_path / experiment_type)
        created[experiment_type] = experiment

        assert isinstance(experiment, expected_classes[experiment_type])
        assert experiment.experiment_type == experiment_type
        assert experiment.output_name == expected_outputs[experiment_type]
        assert isinstance(experiment.execution_core, PortfolioExecutionCore)
        assert isinstance(experiment.cost_model, CostModel)
        assert isinstance(experiment.constraint_manager, ConstraintManager)
        assert experiment.execution_core.cost_model is experiment.cost_model
        assert experiment.execution_core.constraint_manager is experiment.constraint_manager
        assert experiment.output_schema is registry.output_schema
        assert set(experiment.output_schema) == {
            "daily_returns",
            "daily_weights",
            "daily_turnover",
            "daily_rebalance",
            "daily_costs",
        }

    assert set(created) == set(VALID_EXPERIMENT_TYPES)
    assert set(created["baseline_comparison"].baselines) == set(DEFAULT_CONFIG["baselines"]["traditional"]) | set(
        DEFAULT_CONFIG["baselines"]["deep"]
    )

    native_config = _config(tmp_path, "baseline_comparison")
    native_config["baselines"]["traditional"] = []
    native_config["baselines"]["deep"] = []
    native_config["baselines"]["native"] = [
        "ppo_native",
        "cnn_ppo_native",
        "bernoulli_gated_ppo_native",
        "dqn_template_native",
        "eiie_native",
        "pgportfolio_eiie_native",
    ]
    native_experiment = registry.create_experiment(native_config, device="cpu", run_dir=tmp_path / "native_baselines")
    assert set(native_experiment.baselines) == {
        "ppo_native",
        "cnn_ppo_native",
        "bernoulli_gated_ppo_native",
        "dqn_template_native",
        "eiie_native",
        "pgportfolio_eiie_native",
    }

    hpo_config = _config(tmp_path, "main_model")
    hpo_config["hpo"]["enabled"] = True
    assert isinstance(registry.create_experiment(hpo_config), HPOExperiment)

    invalid_config = _config(tmp_path, "main_model")
    invalid_config["experiment"]["type"] = "unknown"
    with pytest.raises(ValueError, match="ERR_EXPERIMENT_UNKNOWN_TYPE"):
        registry.create_experiment(invalid_config)


def test_baseline_factories_expand_hybrid_dqn_alias(tmp_path):
    def factory_names(deep=(), native_rl=(), native=()):
        config = _config(tmp_path, "baseline_comparison")
        config["baselines"]["traditional"] = []
        config["baselines"]["deep"] = list(deep)
        config["baselines"]["native_rl"] = {"enabled_models": list(native_rl)}
        config["baselines"]["native"] = list(native)
        config["baselines"]["external"] = []
        config["baselines"]["external_pgportfolio"] = {"enabled": False}
        return list(_baseline_factories(config))

    expected = list(HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES)
    assert factory_names(deep=[HYBRID_DQN_OPTIMIZER_ALIAS]) == expected
    assert factory_names(native_rl=[HYBRID_DQN_OPTIMIZER_ALIAS]) == expected
    assert factory_names(native=[HYBRID_DQN_OPTIMIZER_ALIAS]) == expected
    assert (
        factory_names(
            deep=[
                HYBRID_DQN_OPTIMIZER_ALIAS,
                HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES[0],
            ],
            native_rl=[HYBRID_DQN_OPTIMIZER_ALIAS],
            native=[HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES[1]],
        )
        == expected
    )

    invalid_config = _config(tmp_path, "baseline_comparison")
    invalid_config["baselines"]["traditional"] = []
    invalid_config["baselines"]["deep"] = ["unknown_deep_baseline"]
    invalid_config["baselines"]["native_rl"] = {"enabled_models": []}
    invalid_config["baselines"]["native"] = []
    invalid_config["baselines"]["external"] = []
    invalid_config["baselines"]["external_pgportfolio"] = {"enabled": False}
    with pytest.raises(KeyError, match="unknown_deep_baseline"):
        _baseline_factories(invalid_config)


def test_run_experiment_smoke(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, "main_model")

    run_dir = run_experiment.main(
        [
            "--config",
            str(config_path),
            "--seed",
            "123",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "results"),
            "--run-name",
            "smoke",
        ]
    )

    assert run_dir == tmp_path / "results" / "smoke"
    snapshot = yaml.safe_load((run_dir / "logs" / "config_snapshot.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "logs" / "run_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "logs" / "experiment_result.json").read_text(encoding="utf-8"))
    assert snapshot["reproducibility"]["seed"] == 123
    assert snapshot["device"]["mode"] == "cpu"
    assert snapshot["output"]["root"] == str(tmp_path / "results")
    assert snapshot["output"]["run_name"] == "smoke"
    assert manifest["status"] == "success"
    assert manifest["run_id"] == "smoke"
    assert manifest["experiment_type"] == "main_model"
    assert manifest["config_hash"] == snapshot["config_hash"]
    assert manifest["execution_price"] == "next_open"
    assert manifest["execution_price_type"] == "open"
    assert result["status"] == "completed"
    assert result["experiment_type"] == "main_model"
    assert result["training_status"] == "completed"
    assert result["checkpoint_count"] >= 1
    assert result["daily_returns"]["rows"] > 0
    assert result["daily_costs"]["rows"] > 0
    assert (run_dir / "metrics" / "daily_returns.csv").stat().st_size > 0
    assert (run_dir / "metrics" / "daily_costs.csv").stat().st_size > 0
    assert (run_dir / "checkpoints" / "last.pt").stat().st_size > 0
    assert _registry_status(tmp_path / "run_registry.sqlite", "smoke") == "success"

    hpo_config_path = _write_config(tmp_path, "main_model", hpo_enabled=True, run_name="hpo")
    called = {}

    def fake_run_hpo(experiment):
        called["experiment"] = experiment
        return {"status": "completed", "experiment_type": experiment.experiment_type, "hpo_marker": "called"}

    monkeypatch.setattr(run_experiment, "run_hpo", fake_run_hpo)
    hpo_run_dir = run_experiment.main(["--config", str(hpo_config_path), "--run-name", "hpo"])

    assert isinstance(called["experiment"], HPOExperiment)
    hpo_result = json.loads((hpo_run_dir / "logs" / "experiment_result.json").read_text(encoding="utf-8"))
    assert hpo_result["status"] == "completed"
    assert hpo_result["hpo_marker"] == "called"

    fail_config_path = _write_config(tmp_path, "main_model", run_name="failed")

    def fail_run(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(MainModelExperiment, "run", fail_run)
    with pytest.raises(RuntimeError, match="boom"):
        run_experiment.main(["--config", str(fail_config_path), "--run-name", "failed"])

    failed_run_dir = tmp_path / "results" / "failed"
    failed_manifest = json.loads((failed_run_dir / "logs" / "run_manifest.json").read_text(encoding="utf-8"))
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["failure_state"]["error_type"] == "RuntimeError"
    assert _registry_status(tmp_path / "run_registry.sqlite", "failed") == "failed"


def test_hpo_without_trial_runner_fails():
    class NoTrialRunner:
        experiment_type = "hyperparameter_sweep"

    with pytest.raises(NotImplementedError, match="ERR_HPO_TRIAL_RUNNER_NOT_IMPLEMENTED"):
        run_experiment._run_hpo_trial(NoTrialRunner(), object(), "train", "validation")


def test_hpo_trial_missing_metric_fails(tmp_path, monkeypatch):
    config_path = _write_config(
        tmp_path,
        "main_model",
        hpo_enabled=True,
        run_name="hpo_missing_metric",
        hpo_overrides={"n_trials": 1, "metric": "validation_metric", "equal_budget_across_models": False},
    )

    def fake_trial(experiment, trial, train_split, validation_split):
        return {"status": "completed"}

    def fail_final_test(experiment, best_trial, final_split):
        raise AssertionError("final test should not run without completed HPO trials")

    monkeypatch.setattr(run_experiment, "_run_hpo_trial", fake_trial)
    monkeypatch.setattr(run_experiment, "_run_hpo_final_test", fail_final_test)

    with pytest.raises(RuntimeError, match="ERR_HPO_NO_COMPLETED_TRIAL"):
        run_experiment.main(["--config", str(config_path), "--run-name", "hpo_missing_metric"])

    trials_path = tmp_path / "results" / "hpo_missing_metric" / "logs" / "hpo_trials.csv"
    with trials_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["state"] == "fail"
    assert "ERR_HPO_METRIC_MISSING" in rows[0]["fail_reason"]


def test_checkpoint_resume_not_marked_success(tmp_path):
    config = _config(tmp_path, "main_model")
    config["training"]["checkpoint_load_path"] = str(tmp_path / "checkpoint.pt")
    experiment = ExperimentRegistry().create_experiment(config, device="cpu", run_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="ERR_CHECKPOINT_NOT_FOUND"):
        experiment.run()


def test_pipeline_checkpoint_honors_replay_buffer_config(tmp_path, monkeypatch):
    calls = []

    def fake_save_checkpoint(agent, path, **kwargs):
        calls.append(kwargs)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"checkpoint")
        return Path(path)

    monkeypatch.setattr(pipeline, "save_checkpoint", fake_save_checkpoint)
    agent = SimpleNamespace(last_epoch=0, global_step=3, best_validation_metric=0.4, history=[])

    path = pipeline._save_last_checkpoint(
        agent,
        {"last": tmp_path / "last.pt"},
        {"training": {"checkpoint_include_replay_buffer": False}},
        env=None,
    )

    assert path == tmp_path / "last.pt"
    assert calls[0]["include_replay_buffer"] is False


def test_hpo_trial_isolation(tmp_path, monkeypatch):
    config_path = _write_config(
        tmp_path,
        "main_model",
        hpo_enabled=True,
        run_name="hpo_isolation",
        hpo_overrides={
            "study_name": "hpo_isolation_study",
            "n_trials": 3,
            "direction": "maximize",
            "metric": "validation_metric",
            "seed": 999,
            "pruner_warmup_trials": 1,
            "pruner_warmup_steps": 2,
            "equal_budget_across_models": False,
        },
    )
    trial_calls = []
    final_calls = []

    def fake_trial(experiment, trial, train_split, validation_split):
        trial.suggest_float("learning_rate", 0.0001, 0.01)
        trial_calls.append(
            {
                "trial_number": trial.number,
                "train_split": train_split,
                "validation_split": validation_split,
            }
        )
        return {
            "validation_metric": float(trial.number),
            "objective_value": float(trial.number),
        }

    def fake_final_test(experiment, best_trial, final_split):
        final_calls.append(
            {
                "best_trial_number": best_trial.number,
                "final_split": final_split,
                "trial_count_at_final": len(trial_calls),
            }
        )
        return {
            "status": "completed",
            "split": final_split,
            "daily_returns": pd.DataFrame({"date": ["2024-01-02"], "net_return": [0.01], "nav": [1.01]}),
            "daily_costs": pd.DataFrame({"date": ["2024-01-02"], "total_transaction_cost": [0.001]}),
        }

    monkeypatch.setattr(run_experiment, "_run_hpo_trial", fake_trial)
    monkeypatch.setattr(run_experiment, "_run_hpo_final_test", fake_final_test)

    run_dir = run_experiment.main(["--config", str(config_path), "--run-name", "hpo_isolation"])

    assert trial_calls == [
        {"trial_number": 0, "train_split": "train", "validation_split": "validation"},
        {"trial_number": 1, "train_split": "train", "validation_split": "validation"},
        {"trial_number": 2, "train_split": "train", "validation_split": "validation"},
    ]
    assert final_calls == [
        {"best_trial_number": 2, "final_split": "test", "trial_count_at_final": 3},
        {"best_trial_number": 1, "final_split": "test", "trial_count_at_final": 3},
        {"best_trial_number": 0, "final_split": "test", "trial_count_at_final": 3},
    ]

    with (run_dir / "logs" / "hpo_trials.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert reader.fieldnames == list(run_experiment.HPO_TRIAL_COLUMNS)
    assert [int(row["trial_number"]) for row in rows] == [0, 1, 2]
    assert {row["state"] for row in rows} == {"complete"}
    assert {row["seed"] for row in rows} == {"999"}
    assert [float(row["validation_metric"]) for row in rows] == [0.0, 1.0, 2.0]
    assert [float(row["objective_value"]) for row in rows] == [0.0, 1.0, 2.0]
    assert all(json.loads(row["params_json"]) for row in rows)
    assert all(row["fail_reason"] == "" for row in rows)

    manifest = json.loads((run_dir / "logs" / "run_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "logs" / "experiment_result.json").read_text(encoding="utf-8"))
    snapshot = yaml.safe_load((run_dir / "logs" / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert manifest["best_trial_number"] == 2
    assert result["best_trial_number"] == 2
    assert result["daily_returns"]["rows"] == 1
    assert result["daily_costs"]["rows"] == 1
    assert result["final_result"]["daily_returns"]["rows"] == 1
    assert [item["rank_label"] for item in result["hpo_final_reports"]] == ["best", "median", "worst"]
    hpo_report_rows = pd.read_csv(run_dir / "metrics" / "hpo_final_reports_table.csv")
    assert hpo_report_rows["rank_label"].tolist() == ["best", "median", "worst"]
    assert pd.read_csv(run_dir / "metrics" / "daily_returns.csv").shape[0] == 1
    assert snapshot["hpo"]["best_trial_number"] == 2
    assert "learning_rate" in snapshot["hpo"]["best_params"]


def test_hpo_equal_budget_runs_all_trainable_models(tmp_path, monkeypatch):
    config = _config(tmp_path, "main_model")
    config["hpo"]["enabled"] = True
    config["hpo"]["equal_budget_across_models"] = True
    config["hpo"]["trainable_models"] = ["full_dqn_gated_multitask_cnn_ppo", "ppo_baseline"]
    config["hpo"]["direction"] = "maximize"
    config["hpo"]["search_space"] = {
        "ppo_lr": {"type": "float", "low": 0.0001, "high": 0.001, "log": True, "rationale": "paper budget"},
    }
    config["output"]["run_name"] = "hpo_equal_budget"
    experiment = ExperimentRegistry().create_experiment(config, device="cpu", run_dir=tmp_path / "hpo_equal_budget")
    calls = []

    def fake_run_hpo_single(child_experiment):
        model_name = child_experiment.active_model_name
        calls.append(
            {
                "model_name": model_name,
                "run_dir": child_experiment.context.run_dir.name,
                "equal_budget": child_experiment.config["hpo"]["equal_budget_across_models"],
            }
        )
        score = 2.0 if model_name == "ppo_baseline" else 1.0
        trials = pd.DataFrame(
            [
                {
                    column: ""
                    for column in run_experiment.HPO_TRIAL_COLUMNS
                }
            ],
            dtype=object,
        )
        trials.loc[0, "model_name"] = model_name
        trials.loc[0, "study_name"] = child_experiment.config["hpo"]["study_name"]
        trials.loc[0, "trial_number"] = 0
        trials.loc[0, "state"] = "complete"
        trials.loc[0, "objective_value"] = score
        return {
            "status": "completed",
            "hpo_model_name": model_name,
            "study_name": child_experiment.config["hpo"]["study_name"],
            "best_trial_number": 0,
            "best_value": score,
            "best_params": {"learning_rate": 0.001},
            "trial_count": 1,
            "hpo_trials": trials,
            "hpo_final_reports_table": pd.DataFrame(
                [
                    {
                        "model_name": model_name,
                        "hpo_model_name": model_name,
                        "rank_label": label,
                        "trial_number": idx,
                        "validation_value": score - idx,
                        "status": "completed",
                        "sharpe": score - idx * 0.1,
                    }
                    for idx, label in enumerate(("best", "median", "worst"))
                ]
            ),
            "metrics": {"sharpe": score},
            "daily_returns": pd.DataFrame({"date": ["2024-01-02"], "net_return": [0.01], "nav": [1.01]}),
            "daily_costs": pd.DataFrame({"date": ["2024-01-02"], "total_transaction_cost": [0.001]}),
        }

    monkeypatch.setattr(run_experiment, "_run_hpo_single", fake_run_hpo_single)

    result = run_experiment.run_hpo(experiment)

    assert calls == [
        {
            "model_name": "full_dqn_gated_multitask_cnn_ppo",
            "run_dir": "hpo_full_dqn_gated_multitask_cnn_ppo",
            "equal_budget": False,
        },
        {"model_name": "ppo_baseline", "run_dir": "hpo_ppo_baseline", "equal_budget": False},
    ]
    assert result["status"] == "completed"
    assert result["hpo_mode"] == "equal_budget_across_models"
    assert result["best_model_name"] == "ppo_baseline"
    assert result["trainable_model_count"] == 2
    assert result["hpo_model_final_comparison"]["model_name"].tolist() == [
        "full_dqn_gated_multitask_cnn_ppo",
        "ppo_baseline",
    ]
    assert set(result["hpo_model_final_daily_returns"]["hpo_model_name"]) == {
        "full_dqn_gated_multitask_cnn_ppo",
        "ppo_baseline",
    }
    final_reports = result["hpo_model_final_reports"]
    assert set(final_reports["hpo_model_name"]) == {"full_dqn_gated_multitask_cnn_ppo", "ppo_baseline"}
    assert final_reports.groupby("hpo_model_name")["rank_label"].apply(list).to_dict() == {
        "full_dqn_gated_multitask_cnn_ppo": ["best", "median", "worst"],
        "ppo_baseline": ["best", "median", "worst"],
    }
    assert [item["model_name"] for item in result["hpo_model_results"]] == [
        "full_dqn_gated_multitask_cnn_ppo",
        "ppo_baseline",
    ]
    persisted_trials = pd.read_csv(tmp_path / "hpo_equal_budget" / "logs" / "hpo_trials.csv")
    assert set(persisted_trials["model_name"]) == {"full_dqn_gated_multitask_cnn_ppo", "ppo_baseline"}
    search_manifest = pd.read_csv(tmp_path / "hpo_equal_budget" / "logs" / "hpo_search_space_manifest.csv")
    assert set(search_manifest["model_name"]) == {"full_dqn_gated_multitask_cnn_ppo", "ppo_baseline"}
    assert search_manifest["param_name"].tolist() == ["ppo_lr", "ppo_lr"]
    assert search_manifest["log_scale"].tolist() == [True, True]


def test_hpo_explicit_trainable_models_are_not_filtered_by_native_only(tmp_path):
    config = _config(tmp_path, "main_model")
    config["hpo"]["trainable_models"] = [
        "full_dqn_gated_multitask_cnn_ppo",
        HYBRID_DQN_OPTIMIZER_ALIAS,
        HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES[0],
        "ppo_proxy",
        "ppo_baseline",
        "pgportfolio_original_external",
        "pgportfolio_eiie_native",
        "equal_weight",
    ]
    config["hpo"]["native_only"] = True

    assert run_experiment._hpo_trainable_models(config) == [
        "full_dqn_gated_multitask_cnn_ppo",
        *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
        "ppo_proxy",
        "ppo_baseline",
        "pgportfolio_original_external",
        "pgportfolio_eiie_native",
        "equal_weight",
    ]
    assert all(run_experiment._is_native_hpo_trainable_model(name) for name in HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES)
    assert run_experiment._is_native_hpo_trainable_model(HYBRID_DQN_OPTIMIZER_ALIAS) is False


def test_hpo_default_trainable_models_expand_hybrid_dqn_alias(tmp_path):
    config = _config(tmp_path, "main_model")
    config["hpo"]["trainable_models"] = []
    config["hpo"]["native_only"] = True
    config["baselines"]["deep"] = [HYBRID_DQN_OPTIMIZER_ALIAS, HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES[0]]

    assert run_experiment._hpo_trainable_models(config) == [
        "full_dqn_gated_multitask_cnn_ppo",
        *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
    ]


def test_hpo_default_trainable_models_include_native_rl_enabled_models(tmp_path):
    config = _config(tmp_path, "main_model")
    config["hpo"]["trainable_models"] = []
    config["hpo"]["native_only"] = True
    config["baselines"]["deep"] = []
    config["baselines"]["native_rl"] = {
        "enabled_models": [
            "ppo_dqn_hierarchical_reimplementation",
            HYBRID_DQN_OPTIMIZER_ALIAS,
        ]
    }

    assert run_experiment._hpo_trainable_models(config) == [
        "full_dqn_gated_multitask_cnn_ppo",
        "ppo_dqn_hierarchical_reimplementation",
        *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
    ]


def test_m6_t5_formal_hpo_config_uses_child_budget_units():
    config = ConfigLoader.load(PROJECT_ROOT / "configs/paper/hpo_equal_budget_related_work.yaml")

    assert run_experiment._hpo_trainable_models(config) == [
        "ppo_dqn_hierarchical_reimplementation",
        *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
    ]
    assert HYBRID_DQN_OPTIMIZER_ALIAS not in config["hpo"]["trainable_models"]
    assert config["hpo"]["n_trials_per_model"] == 50
    assert config["long_running"] is True


def test_hpo_native_only_fails_for_inferred_scope_when_no_rankable_native_model(tmp_path):
    config = _config(tmp_path, "main_model")
    config["hpo"]["trainable_models"] = []
    config["model"]["name"] = "equal_weight"
    config["baselines"]["deep"] = ["ppo_proxy"]
    config["baselines"]["native_rl"] = {"enabled_models": []}
    config["hpo"]["native_only"] = True

    with pytest.raises(ConfigError, match="ERR_HPO_NO_NATIVE_TRAINABLE_MODEL"):
        run_experiment._hpo_trainable_models(config)


def test_hpo_native_baseline_trial_gets_checkpoint_run_dir(tmp_path, monkeypatch):
    config = _config(tmp_path, "main_model")
    config["hpo"]["enabled"] = True
    config["hpo"]["metric"] = "validation_metric"
    experiment = ExperimentRegistry().create_experiment(config, device="cpu", run_dir=tmp_path / "hpo_native")
    active_split = SplitSpec(
        pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        pd.DatetimeIndex(["2024-01-04"]),
        pd.DatetimeIndex(["2024-01-05"]),
        fold_id="fold_native",
    )
    setattr(experiment, "active_model_name", "dqn_template_native")
    setattr(experiment, "active_split", active_split)
    calls = []

    def fake_run_strategy_backtest(
        config,
        strategy_factory,
        *,
        model_name,
        segment="test",
        run_dir=None,
        split_override=None,
    ):
        calls.append(
            {
                "model_name": model_name,
                "segment": segment,
                "run_dir": run_dir,
                "split_override": split_override,
            }
        )
        return {"status": "completed", "metrics": {"validation_metric": 1.0}}

    monkeypatch.setattr("src.experiments.registry.run_strategy_backtest", fake_run_strategy_backtest)

    result = experiment.run_trial(SimpleNamespace(number=3, params={}), "train", "validation")

    assert result["status"] == "completed"
    assert result["hpo_model_name"] == "dqn_template_native"
    assert calls == [
        {
            "model_name": "dqn_template_native",
            "segment": "validation",
            "run_dir": str(tmp_path / "hpo_native" / "trial_3"),
            "split_override": active_split,
        }
    ]


def test_hpo_native_baseline_final_test_uses_active_split(tmp_path, monkeypatch):
    config = _config(tmp_path, "main_model")
    config["hpo"]["enabled"] = True
    experiment = ExperimentRegistry().create_experiment(config, device="cpu", run_dir=tmp_path / "hpo_native_final")
    active_split = SplitSpec(
        pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        pd.DatetimeIndex(["2024-01-04"]),
        pd.DatetimeIndex(["2024-01-05"]),
        fold_id="fold_final",
    )
    setattr(experiment, "active_model_name", "pgportfolio_eiie_native")
    setattr(experiment, "active_split", active_split)
    setattr(experiment, "final_test_label", "fold_final_test")
    calls = []

    def fake_run_strategy_backtest(
        config,
        strategy_factory,
        *,
        model_name,
        segment="test",
        run_dir=None,
        split_override=None,
    ):
        calls.append(
            {
                "model_name": model_name,
                "segment": segment,
                "run_dir": run_dir,
                "split_override": split_override,
            }
        )
        return {"status": "completed", "metrics": {"cumulative_return": 0.1}}

    monkeypatch.setattr("src.experiments.registry.run_strategy_backtest", fake_run_strategy_backtest)

    result = experiment.run_final_test(SimpleNamespace(number=7, params={}), "test")

    assert result["final_split"] == "test"
    assert result["best_trial_number"] == 7
    assert calls == [
        {
            "model_name": "pgportfolio_eiie_native",
            "segment": "test",
            "run_dir": str(tmp_path / "hpo_native_final" / "fold_final_test"),
            "split_override": active_split,
        }
    ]


def test_hpo_deep_baseline_reuses_pipeline_artifacts_for_model_params(tmp_path, monkeypatch):
    config = _config(tmp_path, "main_model")
    config["hpo"]["enabled"] = True
    config["hpo"]["metric"] = "validation_metric"
    experiment = ExperimentRegistry().create_experiment(config, device="cpu", run_dir=tmp_path / "hpo_cached")
    setattr(experiment, "active_model_name", "cage_eiie_frozen_gate")
    cached_artifacts = {"sentinel": object()}
    artifact_build_calls = []
    backtest_artifacts = []

    def fake_build_pipeline_artifacts(config, split_override=None):
        artifact_build_calls.append(split_override)
        return cached_artifacts

    def fake_run_strategy_backtest(
        config,
        strategy_factory,
        *,
        model_name,
        segment="test",
        run_dir=None,
        split_override=None,
        artifacts=None,
    ):
        assert model_name == "cage_eiie_frozen_gate"
        assert split_override is None
        backtest_artifacts.append(artifacts)
        return {"status": "completed", "metrics": {"validation_metric": 1.0}}

    monkeypatch.setattr("src.experiments.registry.build_pipeline_artifacts", fake_build_pipeline_artifacts)
    monkeypatch.setattr("src.experiments.registry.run_strategy_backtest", fake_run_strategy_backtest)

    trial_params = {"cage_eiie.lambda_turnover": 1.0}
    experiment.run_trial(SimpleNamespace(number=0, params=trial_params), "train", "validation")
    experiment.run_trial(SimpleNamespace(number=1, params=trial_params), "train", "validation")

    assert artifact_build_calls == [None]
    assert backtest_artifacts == [cached_artifacts, cached_artifacts]


def test_walk_forward_hpo_runs_independent_hpo_per_fold(tmp_path, monkeypatch):
    config = _config(tmp_path, "walk_forward")
    config["hpo"]["enabled"] = True
    config["hpo"]["equal_budget_across_models"] = False
    config["output"]["run_name"] = "wf_hpo"
    experiment = ExperimentRegistry().create_experiment(config, device="cpu", run_dir=tmp_path / "wf_hpo")
    dates = pd.date_range("2024-01-02", periods=6, freq="D")
    split_1 = SplitSpec(dates[:2], dates[2:3], dates[3:4], fold_id="fold_1")
    split_2 = SplitSpec(dates[1:3], dates[3:4], dates[4:5], fold_id="fold_2")
    calls = []

    monkeypatch.setattr(
        run_experiment,
        "load_market_dataset",
        lambda config: MarketDatasetBundle(
            asset_universe=pd.DataFrame(),
            panel=pd.DataFrame(),
            wide={"close": pd.DataFrame(index=dates)},
            metrics_features=None,
            feature_cols=[],
            auxiliary_target_cols=[],
            availability_mask=pd.DataFrame(),
            availability_reason=None,
            data_manifest={},
        ),
    )
    monkeypatch.setattr(run_experiment, "create_split", lambda trade_dates, config: [split_1, split_2])

    def fake_run_hpo_single(fold_experiment):
        fold = getattr(fold_experiment, "active_split")
        calls.append(
            {
                "fold_id": fold.fold_id,
                "run_dir": fold_experiment.context.run_dir.name,
                "study_name": fold_experiment.config["hpo"]["study_name"],
            }
        )
        return {
            "status": "completed",
            "study_name": fold_experiment.config["hpo"]["study_name"],
            "best_trial_number": 1,
            "best_value": 0.5,
            "best_params": {"lr": 0.001},
            "trial_count": 2,
            "daily_returns": pd.DataFrame(
                {"date": [fold.test_dates[0]], "net_return": [0.01], "fold_id": [fold.fold_id]}
            ),
            "daily_costs": pd.DataFrame({"date": [fold.test_dates[0]], "total_transaction_cost": [0.001]}),
        }

    monkeypatch.setattr(run_experiment, "_run_hpo_single", fake_run_hpo_single)

    result = run_experiment.run_hpo(experiment)

    assert calls == [
        {"fold_id": "fold_1", "run_dir": "hpo_fold_1", "study_name": "wf_hpo_hpo_fold_1"},
        {"fold_id": "fold_2", "run_dir": "hpo_fold_2", "study_name": "wf_hpo_hpo_fold_2"},
    ]
    assert result["status"] == "completed"
    assert result["fold_count"] == 2
    assert len(result["fold_hpo_results"]) == 2
    assert result["daily_returns"]["fold_id"].tolist() == ["fold_1", "fold_2"]


def test_walk_forward_hpo_equal_budget_per_fold_and_aggregates_trials(tmp_path, monkeypatch):
    config = _config(tmp_path, "walk_forward")
    config["hpo"]["enabled"] = True
    config["hpo"]["equal_budget_across_models"] = True
    config["hpo"]["trainable_models"] = ["full_dqn_gated_multitask_cnn_ppo", "ppo_baseline"]
    config["output"]["run_name"] = "wf_equal_hpo"
    experiment = ExperimentRegistry().create_experiment(config, device="cpu", run_dir=tmp_path / "wf_equal_hpo")
    dates = pd.date_range("2024-01-02", periods=6, freq="D")
    split_1 = SplitSpec(dates[:2], dates[2:3], dates[3:4], fold_id="fold_1")
    split_2 = SplitSpec(dates[1:3], dates[3:4], dates[4:5], fold_id="fold_2")
    calls = []

    monkeypatch.setattr(
        run_experiment,
        "load_market_dataset",
        lambda config: MarketDatasetBundle(
            asset_universe=pd.DataFrame(),
            panel=pd.DataFrame(),
            wide={"close": pd.DataFrame(index=dates)},
            metrics_features=None,
            feature_cols=[],
            auxiliary_target_cols=[],
            availability_mask=pd.DataFrame(),
            availability_reason=None,
            data_manifest={},
        ),
    )
    monkeypatch.setattr(run_experiment, "create_split", lambda trade_dates, config: [split_1, split_2])

    def fake_run_equal_budget_hpo(fold_experiment, model_names):
        fold = getattr(fold_experiment, "active_split")
        calls.append(
            {
                "fold_id": fold.fold_id,
                "run_dir": fold_experiment.context.run_dir.name,
                "models": tuple(model_names),
            }
        )
        trials = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "fold_id": fold.fold_id,
                    "study_name": f"{fold.fold_id}_{model_name}",
                    "trial_number": index,
                    "seed": 7,
                    "params_json": "{}",
                    "state": "complete",
                    "objective_value": float(index),
                    "validation_metric": float(index),
                    "train_start": "",
                    "train_end": "",
                    "duration_sec": "",
                    "pruned_step": "",
                    "fail_reason": "",
                }
                for index, model_name in enumerate(model_names)
            ]
        )
        return {
            "status": "completed",
            "hpo_mode": "equal_budget_across_models",
            "best_model_name": "ppo_baseline",
            "hpo_model_results": [{"model_name": model_name, "status": "completed"} for model_name in model_names],
            "best_trial_number": 1,
            "best_value": 1.0,
            "best_params": {"lr": 0.001},
            "trial_count": len(trials),
            "hpo_trials": trials,
            "daily_returns": pd.DataFrame(
                {"date": [fold.test_dates[0]], "net_return": [0.01], "fold_id": [fold.fold_id]}
            ),
            "daily_costs": pd.DataFrame({"date": [fold.test_dates[0]], "total_transaction_cost": [0.001]}),
        }

    monkeypatch.setattr(run_experiment, "_run_equal_budget_hpo", fake_run_equal_budget_hpo)

    result = run_experiment.run_hpo(experiment)

    assert calls == [
        {
            "fold_id": "fold_1",
            "run_dir": "hpo_fold_1",
            "models": ("full_dqn_gated_multitask_cnn_ppo", "ppo_baseline"),
        },
        {
            "fold_id": "fold_2",
            "run_dir": "hpo_fold_2",
            "models": ("full_dqn_gated_multitask_cnn_ppo", "ppo_baseline"),
        },
    ]
    assert result["status"] == "completed"
    assert result["daily_returns"]["fold_id"].tolist() == ["fold_1", "fold_2"]
    assert "best_model_name" not in result
    assert "best_trial_number" not in result
    assert result["hpo_trials"]["fold_id"].tolist() == ["fold_1", "fold_1", "fold_2", "fold_2"]
    assert set(result["hpo_trials"]["model_name"]) == {"full_dqn_gated_multitask_cnn_ppo", "ppo_baseline"}
    assert result["fold_hpo_results"][0]["best_model_name"] == "ppo_baseline"
    assert result["fold_hpo_results"][0]["hpo_model_results"] == [
        {"model_name": "full_dqn_gated_multitask_cnn_ppo", "status": "completed"},
        {"model_name": "ppo_baseline", "status": "completed"},
    ]
    persisted_trials = pd.read_csv(tmp_path / "wf_equal_hpo" / "logs" / "hpo_trials.csv")
    assert persisted_trials["fold_id"].tolist() == ["fold_1", "fold_1", "fold_2", "fold_2"]


def test_ablation_single_switch_guard(tmp_path):
    registry = ExperimentRegistry()

    input_matrix_config = _config(tmp_path, "input_matrix_ablation")
    input_matrix_config["feature_matrix"]["input_matrix_id"] = "M0"
    input_matrix_config["training"]["checkpoint_include_replay_buffer"] = False
    input_matrix_experiment = registry.create_experiment(input_matrix_config)
    assert input_matrix_experiment.ablation_id == "input_matrix_ablation.feature_matrix.input_matrix_id"
    assert input_matrix_experiment.changed_key_path == "feature_matrix.input_matrix_id"

    metadata_config = _config(tmp_path, "input_matrix_ablation")
    metadata_config["feature_matrix"]["input_matrix_id"] = "M0"
    metadata_config["data_governance"].update(
        {
            "return_source": "adj_nav",
            "valuation_source": "adj_nav",
            "reward_return_source": "adj_nav",
            "metrics_return_source": "adj_nav",
            "execution_price_source": "ohlcv",
            "valuation_execution_split": True,
            "reward_valuation_split": True,
        }
    )
    metadata_experiment = registry.create_experiment(metadata_config)
    assert metadata_experiment.ablation_id == "input_matrix_ablation.feature_matrix.input_matrix_id"
    assert metadata_experiment.changed_key_path == "feature_matrix.input_matrix_id"

    reward_config = _config(tmp_path, "reward_ablation")
    reward_config["reward_ablation"]["enabled"] = True
    reward_config["reward"]["mode"] = "A0_raw_simple_return"
    reward_config["reward"]["risk_penalty_enabled"] = False
    reward_experiment = registry.create_experiment(reward_config)
    assert reward_experiment.changed_key_path == "reward"
    assert reward_experiment.cost_model.cost_config["proportional_cost"] == DEFAULT_CONFIG["cost_model"]["proportional_cost"]

    multi_family_config = _config(tmp_path, "ablation")
    multi_family_config["ppo"]["enabled"] = False
    multi_family_config["dqn"]["enabled"] = False
    with pytest.raises(ValueError, match="ERR_EXPERIMENT_ABLATION_NOT_SINGLE_SWITCH"):
        registry.create_experiment(multi_family_config)

    removed_cost_config = _config(tmp_path, "reward_ablation")
    removed_cost_config["reward_ablation"]["enabled"] = True
    removed_cost_config["cost_model"]["proportional_cost"] = 0.0
    removed_cost_config["cost_model"]["slippage"] = 0.0
    removed_cost_config["cost_model"]["market_impact_enabled"] = False
    with pytest.raises(ValueError, match="ERR_EXPERIMENT_ABLATION_NOT_SINGLE_SWITCH"):
        registry.create_experiment(removed_cost_config)


def test_specialized_experiment_model_factory(tmp_path, monkeypatch):
    captured = []

    def fake_run_trained_model_experiment(config, *, model_name, run_dir=None, **kwargs):
        captured.append({"model_name": model_name, "config_model": config["model"]["name"]})
        return {
            "status": "completed",
            "model_name": model_name,
            "metrics": {"sharpe": 1.0},
            "main_comparison": pd.DataFrame(
                [
                    {"model_name": model_name, "status": "completed", "sharpe": 1.0},
                    {"model_name": "equal_weight", "status": "completed", "sharpe": 0.5},
                ]
            ),
        }

    monkeypatch.setattr("src.experiments.registry.run_trained_model_experiment", fake_run_trained_model_experiment)
    registry = ExperimentRegistry()
    for experiment_type, expected_model, expected_class in (
        ("preference_conditioned_analysis", "preference_conditioned_gated_ppo", PreferenceConditionedGatedPPO),
        ("uncertainty_analysis", "uncertainty_aware_gated_ppo", UncertaintyAwareGatedPPO),
        ("distributional_cvar_analysis", "distributional_cvar_gated_ppo", DistributionalCVaRGatedPPO),
        ("partial_rebalance_analysis", "partial_rebalance_gated_ppo", PartialRebalanceGatedPPO),
    ):
        experiment = registry.create_experiment(_config(tmp_path, experiment_type), device="cpu", run_dir=tmp_path)
        result = experiment.run()
        assert result["model_name"] == expected_model
        assert result[experiment.output_name]["module_name"].eq(experiment.module_name).all()
        assert result[experiment.output_name]["analysis_name"].eq(experiment.output_name).all()
        assert _model_class({"model": {"name": expected_model}}) is expected_class

    assert [item["config_model"] for item in captured] == [
        "preference_conditioned_gated_ppo",
        "uncertainty_aware_gated_ppo",
        "distributional_cvar_gated_ppo",
        "partial_rebalance_gated_ppo",
    ]


def test_result_mapping_overrides_strategy_daily_model_name(monkeypatch):
    monkeypatch.setattr(pipeline, "artifact_payload", lambda artifacts: {})
    backtest_result = SimpleNamespace(
        metrics={"cumulative_return": 0.1},
        daily_returns=pd.DataFrame({"date": ["2024-01-02"], "model_name": ["generic_strategy"], "net_return": [0.01]}),
        daily_weights=pd.DataFrame({"date": ["2024-01-02"], "model_name": ["generic_strategy"], "asset_id": ["A"], "weight": [1.0]}),
        daily_turnover=pd.DataFrame({"date": ["2024-01-02"], "model_name": ["generic_strategy"], "turnover": [0.0]}),
        daily_rebalance=pd.DataFrame({"date": ["2024-01-02"], "model_name": ["generic_strategy"], "rebalance_action": [0]}),
        daily_costs=pd.DataFrame({"date": ["2024-01-02"], "model_name": ["generic_strategy"], "total_transaction_cost": [0.0]}),
        run_manifest={},
    )

    payload = pipeline.result_mapping(
        backtest_result,
        config={},
        artifacts={},
        status="completed",
        model_name="uncertainty_aware_gated_ppo",
    )

    for key in ("daily_returns", "daily_weights", "daily_turnover", "daily_rebalance", "daily_costs"):
        assert payload[key]["model_name"].tolist() == ["uncertainty_aware_gated_ppo"]


def test_specialized_dqn_gate_modules_match_policy_forward_gate():
    partial_config = _minimal_model_config("partial_rebalance_gated_ppo")
    partial_config["partial_rebalance"] = {
        "enabled": True,
        "mode": "discrete_dqn",
        "discrete_rho_values": [0.0, 0.25, 1.0],
    }
    partial_agent = pipeline.build_hybrid_agent(partial_config)

    assert partial_agent.dqn_agent is not None
    assert partial_agent.dqn_agent.online_network is not partial_agent.policy_model.gate
    assert partial_agent.dqn_agent.online_network.output_dim == 3

    uncertainty_config = _minimal_model_config("uncertainty_aware_gated_ppo")
    uncertainty_config["uncertainty"] = {"enabled": True, "method": "multi_head", "n_heads": 2}
    uncertainty_agent = pipeline.build_hybrid_agent(uncertainty_config)

    assert uncertainty_agent.dqn_agent is not None
    assert uncertainty_agent.dqn_agent.online_network is not uncertainty_agent.policy_model.gate
    assert uncertainty_agent.dqn_agent.online_network.output_dim == 2


def test_dqn_and_auxiliary_ablation_switches_disable_training_components():
    no_dqn_config = _minimal_model_config("full_dqn_gated_multitask_cnn_ppo")
    no_dqn_config["dqn"]["enabled"] = False
    no_dqn_agent = pipeline.build_hybrid_agent(no_dqn_config)

    assert no_dqn_agent.dqn_agent is None
    assert no_dqn_agent.ppo_agent.gate_network is None
    assert no_dqn_agent.policy_model.dqn_gate_enabled is False
    action_info = no_dqn_agent.ppo_agent.select_action(
        {
            "market_image": [[[0.01, 0.02], [0.03, 0.04]]],
            "availability_mask": [True, True],
            "current_weights": [0.5, 0.5],
        },
        deterministic=True,
    )
    assert action_info["gate_action"] == 1
    assert action_info["rebalance_intensity"] == 1.0

    no_aux_config = _minimal_model_config("full_dqn_gated_multitask_cnn_ppo")
    no_aux_config["auxiliary"]["enabled"] = False
    no_aux_agent = pipeline.build_hybrid_agent(no_aux_config)

    assert no_aux_agent.auxiliary_heads is None
    assert no_aux_agent.policy_model.aux_heads is None


def test_generic_component_ablation_variants_have_stable_paper_ids(tmp_path):
    no_dqn_config = _config(tmp_path, "ablation")
    no_dqn_config["dqn"]["enabled"] = False
    assert [item["variant_id"] for item in _ablation_variants(no_dqn_config, "ablation")] == [
        "full_model",
        "without_dqn_gate",
    ]

    no_aux_config = _config(tmp_path, "ablation")
    no_aux_config["auxiliary"]["enabled"] = False
    assert [item["variant_id"] for item in _ablation_variants(no_aux_config, "ablation")] == [
        "full_model",
        "without_auxiliary",
    ]

    kernel_config = _config(tmp_path, "kernel_size_ablation")
    kernel_variants = _ablation_variants(kernel_config, "kernel_size_ablation")
    assert [item["variant_value"] for item in kernel_variants] == ["1x1", "1x3", "3x3", "5x3", "11x3", "21x3"]
    assert [
        (
            item["config"]["model"]["encoder"]["kernel_size_time"],
            item["config"]["model"]["encoder"]["kernel_size_asset"],
        )
        for item in kernel_variants
    ] == [(1, 1), (1, 3), (3, 3), (5, 3), (11, 3), (21, 3)]


def test_matrix_experiments_generate_multiple_child_variants(tmp_path, monkeypatch):
    captured = []

    def fake_run_trained_variant_matrix(config, *, model_name, matrix_name, variants, run_dir=None):
        variant_list = list(variants)
        captured.append({"matrix_name": matrix_name, "variant_ids": [item["variant_id"] for item in variant_list]})
        return {
            "status": "completed",
            "model_name": model_name,
            "child_run_count": len(variant_list),
            matrix_name: pd.DataFrame({"variant_id": [item["variant_id"] for item in variant_list]}),
        }

    monkeypatch.setattr("src.experiments.registry.run_trained_variant_matrix", fake_run_trained_variant_matrix)
    registry = ExperimentRegistry()

    ablation_result = registry.create_experiment(_config(tmp_path, "input_matrix_ablation")).run()
    kernel_result = registry.create_experiment(_config(tmp_path, "kernel_size_ablation")).run()
    sensitivity_result = registry.create_experiment(_config(tmp_path, "transaction_cost_sensitivity")).run()

    assert ablation_result["child_run_count"] == 8
    assert kernel_result["child_run_count"] == 6
    assert sensitivity_result["child_run_count"] == 5
    assert captured[0]["matrix_name"] == "input_matrix_ablation_results"
    assert captured[0]["variant_ids"] == [
        "input_matrix_M0",
        "input_matrix_M1",
        "input_matrix_M2",
        "input_matrix_M3",
        "input_matrix_M4",
        "input_matrix_M5",
        "input_matrix_M6",
        "input_matrix_M7",
    ]
    assert captured[1]["matrix_name"] == "kernel_size_ablation_results"
    assert captured[1]["variant_ids"] == [
        "kernel_single_day_1x1",
        "kernel_single_day_cross_asset_1x3",
        "kernel_short_3x3",
        "kernel_week_5x3",
        "kernel_long_11x3",
        "kernel_long_21x3",
    ]
    assert captured[2]["matrix_name"] == "transaction_cost_sensitivity"


def test_asset_universe_sensitivity_generates_asset_pool_variants(tmp_path, monkeypatch):
    asset_universe_path = tmp_path / "asset_universe.csv"
    pd.DataFrame(
        [
            {"ts_code": "510300.SH", "pool": "equity", "status": "ok"},
            {"ts_code": "511010.SH", "pool": "bond", "status": "ok"},
            {"ts_code": "513030.SH", "pool": "global_equity", "status": "missing"},
        ]
    ).to_csv(asset_universe_path, index=False)
    config = _config(tmp_path, "asset_universe_sensitivity")
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    config["data"]["asset_universe_path"] = str(asset_universe_path)

    variants = _sensitivity_variants(config, "asset_universe_sensitivity")

    assert [item["variant_id"] for item in variants] == ["asset_pool_all", "asset_pool_bond", "asset_pool_equity"]
    assert variants[0]["config"]["data"]["asset_universe_pools"] == []
    assert variants[0]["config"]["data"]["asset_universe_assets"] == []
    assert variants[1]["config"]["data"]["asset_universe_pools"] == ["bond"]
    assert variants[2]["config"]["data"]["asset_universe_pools"] == ["equity"]

    relative_config = _config(tmp_path, "asset_universe_sensitivity")
    relative_config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    relative_config["data"]["asset_universe_path"] = asset_universe_path.name
    monkeypatch.chdir(tmp_path)
    relative_variants = _sensitivity_variants(relative_config, "asset_universe_sensitivity")
    relative_ids = [item["variant_id"] for item in relative_variants]
    assert "asset_pool_equity" in relative_ids
    assert "common_history_on" not in relative_ids

    denied_config = _config(tmp_path, "asset_universe_sensitivity")
    denied_config["security"]["path_whitelist"] = [str(tmp_path)]
    with pytest.raises(ConfigError, match="ERR_SECURITY_PATH_DENIED"):
        _sensitivity_variants(denied_config, "asset_universe_sensitivity")


def test_walk_forward_all_oos_aggregation(tmp_path):
    fold_results = [
        {
            "fold_id": "fold_1",
            "daily_returns": pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "fold_id": ["fold_1", "fold_1"],
                    "split": ["test", "test"],
                    "net_return": [0.01, -0.02],
                }
            ),
            "daily_turnover": pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "fold_id": ["fold_1", "fold_1"],
                    "turnover": [0.10, 0.20],
                }
            ),
            "daily_costs": pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "fold_id": ["fold_1", "fold_1"],
                    "total_transaction_cost": [0.001, 0.002],
                }
            ),
        },
        {
            "fold_id": "fold_2",
            "daily_returns": pd.DataFrame(
                {
                    "date": ["2024-01-03", "2024-01-04"],
                    "fold_id": ["fold_2", "fold_2"],
                    "split": ["test", "test"],
                    "net_return": [0.03, 0.04],
                }
            ),
            "daily_turnover": pd.DataFrame(
                {
                    "date": ["2024-01-03", "2024-01-04"],
                    "fold_id": ["fold_2", "fold_2"],
                    "turnover": [0.30, 0.40],
                }
            ),
            "daily_costs": pd.DataFrame(
                {
                    "date": ["2024-01-03", "2024-01-04"],
                    "fold_id": ["fold_2", "fold_2"],
                    "total_transaction_cost": [0.003, 0.004],
                }
            ),
        },
    ]

    result = aggregate_walk_forward(fold_results, run_dir=tmp_path)

    all_oos = result["all_oos_daily_returns"]
    assert all_oos["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert all_oos["fold_id"].tolist() == ["fold_1", "fold_1", "fold_2"]
    assert all_oos["net_return"].tolist() == [0.01, -0.02, 0.04]
    assert result["duplicate_oos_date_count"] == 1

    walk_forward_results = result["walk_forward_results"]
    assert walk_forward_results["fold_id"].tolist() == ["fold_1", "fold_2", "all_oos"]
    all_oos_row = walk_forward_results[walk_forward_results["fold_id"] == "all_oos"].iloc[0]
    assert all_oos_row["duplicate_oos_date_count"] == 1
    assert all_oos_row["n_steps"] == 3.0
    assert all_oos_row["cumulative_return"] == pytest.approx((1.01 * 0.98 * 1.04) - 1.0)
    assert all_oos_row["hit_ratio"] == pytest.approx(2 / 3)
    assert all_oos_row["turnover"] == pytest.approx((0.10 + 0.20 + 0.40) / 3)
    assert all_oos_row["total_transaction_cost"] == pytest.approx(0.001 + 0.002 + 0.004)

    output = pd.read_csv(tmp_path / "metrics" / "walk_forward_results.csv")
    assert output["fold_id"].tolist() == ["fold_1", "fold_2", "all_oos"]
    assert int(output.loc[output["fold_id"] == "all_oos", "duplicate_oos_date_count"].iloc[0]) == 1


def test_paper_full_dry_run_writes_manifest(tmp_path):
    output_root = tmp_path / "paper_runs"
    aggregate_dir = tmp_path / "paper_tables"

    result = run_paper_full(
        profile="pilot",
        output_root=output_root,
        run_prefix="PAPER_TEST",
        aggregate_output_dir=aggregate_dir,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["run_count"] == 3
    assert [Path(item["config_path"]).name for item in result["runs"]] == [
        "p0_native_baseline_smoke.yaml",
        "hpo_equal_budget_native_pilot.yaml",
        "baseline_comparison_native.yaml",
    ]
    manifest = json.loads((aggregate_dir / "PAPER_TEST_paper_full_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"
    assert manifest["profile"] == "pilot"

    formal = run_paper_full(
        profile="formal",
        output_root=output_root,
        run_prefix="PAPER_TEST",
        aggregate_output_dir=aggregate_dir,
        dry_run=True,
    )
    formal_names = [Path(item["config_path"]).name for item in formal["runs"]]
    assert formal_names[0] == "main_model.yaml"
    assert "hpo_equal_budget_main_native.yaml" in formal_names


def test_paper_full_aggregates_only_scoped_main_tables(tmp_path, monkeypatch):
    output_root = tmp_path / "paper_runs"
    aggregate_dir = tmp_path / "paper_tables"

    def fake_run(command, check):
        assert check is True
        run_name = command[command.index("--run-name") + 1]
        output = Path(command[command.index("--output") + 1])
        config_stem = Path(command[command.index("--config") + 1]).stem
        run_dir = output / run_name
        metrics_dir = run_dir / "metrics"
        logs_dir = run_dir / "logs"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        experiment_type = "hyperparameter_sweep" if "hpo" in config_stem else "baseline_comparison"
        if config_stem == "main_model":
            experiment_type = "main_model"
        (logs_dir / "run_manifest.json").write_text(
            json.dumps({"run_name": run_name, "experiment_type": experiment_type, "seed": 42}),
            encoding="utf-8",
        )
        if config_stem == "hpo_equal_budget_main_native":
            comparison_file = "hpo_model_final_comparison.csv"
            returns_file = "hpo_model_final_daily_returns.csv"
            model_name = "full_dqn_gated_multitask_cnn_ppo"
        elif config_stem == "baseline_comparison_native":
            comparison_file = "baseline_comparison.csv"
            returns_file = "daily_returns.csv"
            model_name = "equal_weight"
        elif config_stem == "main_model":
            comparison_file = "main_comparison.csv"
            returns_file = "daily_returns.csv"
            model_name = "full_dqn_gated_multitask_cnn_ppo"
        else:
            return SimpleNamespace(returncode=0)
        pd.DataFrame(
            [{"model_name": model_name, "rankable_in_unified_table": True, "sharpe": 1.0}]
        ).to_csv(metrics_dir / comparison_file, index=False)
        pd.DataFrame(
            {"date": ["2024-01-02"], "model_name": [model_name], "net_return": [0.01]}
        ).to_csv(metrics_dir / returns_file, index=False)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("src.experiments.paper_full.subprocess.run", fake_run)

    result = run_paper_full(
        profile="formal",
        output_root=output_root,
        run_prefix="PAPER_TEST",
        aggregate_output_dir=aggregate_dir,
        dry_run=False,
    )

    assert set(result["aggregate_outputs"]) == {
        "main_hpo",
        "main_fixed",
        "p2_input_pca",
        "p3_components",
        "p4_reward",
        "p5_cost_rebalance",
        "p6_robustness",
        "p8_modules",
    }
    assert (aggregate_dir / "main_hpo" / "paper_main_comparison.csv").exists()
    assert (aggregate_dir / "main_fixed" / "paper_main_comparison.csv").exists()
    assert (aggregate_dir / "p3_components" / "paper_main_comparison.csv").exists()
    assert (aggregate_dir / "p6_robustness" / "paper_main_comparison.csv").exists()
    assert not (aggregate_dir / "paper_main_comparison.csv").exists()
    hpo_main = pd.read_csv(aggregate_dir / "main_hpo" / "paper_main_comparison.csv")
    assert {"full_dqn_gated_multitask_cnn_ppo", "equal_weight"}.issubset(set(hpo_main["paper_model_id"]))
    fixed_stats = pd.read_csv(aggregate_dir / "main_fixed" / "paper_paired_statistics.csv")
    assert "full_dqn_gated_multitask_cnn_ppo" in set(fixed_stats["model_name"].dropna())


def test_walk_forward_rebuilds_fold_specific_artifacts(tmp_path, monkeypatch):
    config = _config(tmp_path, "walk_forward")
    dates = pd.date_range("2024-01-02", periods=8, freq="D")
    splits = [
        SplitSpec(dates[:3], dates[3:4], dates[4:5], fold_id="fold_1"),
        SplitSpec(dates[1:4], dates[4:5], dates[5:6], fold_id="fold_2"),
    ]
    artifact_calls = []

    def fake_build_pipeline_artifacts(config, split_override=None):
        artifact_calls.append(None if split_override is None else split_override.fold_id)
        return {"split": split_override, "all_splits": splits}

    def fake_train_and_backtest(config, artifacts, *, split, model_name, test_split, run_dir):
        assert artifacts["split"] is split
        frame = pd.DataFrame({"date": [split.test_dates[0]], "net_return": [0.01], "fold_id": [split.fold_id]})
        turnover = pd.DataFrame({"date": [split.test_dates[0]], "turnover": [0.1], "fold_id": [split.fold_id]})
        costs = pd.DataFrame(
            {"date": [split.test_dates[0]], "total_transaction_cost": [0.001], "fold_id": [split.fold_id]}
        )
        result = SimpleNamespace(
            daily_returns=frame,
            daily_weights=pd.DataFrame({"date": [split.test_dates[0]], "fold_id": [split.fold_id]}),
            daily_turnover=turnover,
            daily_rebalance=pd.DataFrame({"date": [split.test_dates[0]], "fold_id": [split.fold_id]}),
            daily_costs=costs,
            metrics={"cumulative_return": 0.01},
        )
        return {"status": "completed", "training_status": "completed"}, result

    monkeypatch.setattr(pipeline, "_splits_for_config", lambda config: splits)
    monkeypatch.setattr(pipeline, "build_pipeline_artifacts", fake_build_pipeline_artifacts)
    monkeypatch.setattr(pipeline, "_train_and_backtest", fake_train_and_backtest)

    result = pipeline.run_trained_walk_forward_experiment(config, model_name="model", run_dir=str(tmp_path))

    assert artifact_calls == ["fold_1", "fold_2"]
    assert result["fold_count"] == 2
    assert result["daily_returns"]["fold_id"].tolist() == ["fold_1", "fold_2"]


def test_backtest_walk_forward_rebuilds_fold_specific_artifacts(tmp_path, monkeypatch):
    config = _config(tmp_path, "walk_forward")
    dates = pd.date_range("2024-01-02", periods=8, freq="D")
    splits = [
        SplitSpec(dates[:3], dates[3:4], dates[4:5], fold_id="fold_1"),
        SplitSpec(dates[1:4], dates[4:5], dates[5:6], fold_id="fold_2"),
    ]
    artifact_calls = []

    def fake_build_pipeline_artifacts(config, split_override=None):
        artifact_calls.append(None if split_override is None else split_override.fold_id)
        return {"split": split_override, "all_splits": splits}

    def fake_run_backtest_with_artifacts(config, artifacts, strategy_factory, *, model_name, split, segment, run_dir=None):
        assert artifacts["split"] is split
        assert run_dir is not None and str(split.fold_id) in str(run_dir)
        frame = pd.DataFrame({"date": [split.test_dates[0]], "net_return": [0.01], "fold_id": [split.fold_id]})
        result = SimpleNamespace(
            daily_returns=frame,
            daily_weights=pd.DataFrame({"date": [split.test_dates[0]], "fold_id": [split.fold_id]}),
            daily_turnover=pd.DataFrame({"date": [split.test_dates[0]], "turnover": [0.1], "fold_id": [split.fold_id]}),
            daily_rebalance=pd.DataFrame({"date": [split.test_dates[0]], "fold_id": [split.fold_id]}),
            daily_costs=pd.DataFrame(
                {"date": [split.test_dates[0]], "total_transaction_cost": [0.001], "fold_id": [split.fold_id]}
            ),
            metrics={"cumulative_return": 0.01},
        )
        return {"device": "cpu"}, result

    monkeypatch.setattr(pipeline, "_splits_for_config", lambda config: splits)
    monkeypatch.setattr(pipeline, "build_pipeline_artifacts", fake_build_pipeline_artifacts)
    monkeypatch.setattr(pipeline, "_run_backtest_with_artifacts", fake_run_backtest_with_artifacts)
    monkeypatch.setattr(pipeline, "artifact_payload", lambda artifacts: {})

    result = pipeline.run_walk_forward_backtest(config, lambda cfg: object(), model_name="baseline", run_dir=str(tmp_path))

    assert artifact_calls == ["fold_1", "fold_2"]
    assert result["daily_returns"]["fold_id"].tolist() == ["fold_1", "fold_2"]


def test_run_all_matrix_builds_lineage(tmp_path):
    config = _config(tmp_path, "full_reproduction")
    config["output"]["run_name"] = "full_reproduction"
    run_dir = tmp_path / "matrix"

    registry = _FakeMatrixRegistry()
    result = run_experiment_matrix(config, registry=registry, run_dir=run_dir)

    assert result["status"] == "completed"
    assert result["run_sequence"] == list(FULL_REPRODUCTION_SEQUENCE)
    assert len(result["lineage"]) == len(FULL_REPRODUCTION_SEQUENCE)
    assert registry.created == list(FULL_REPRODUCTION_SEQUENCE)
    assert (run_dir / "logs" / "lineage.json").exists()


def test_run_all_matrix_rejects_partial_child(tmp_path):
    config = _config(tmp_path, "full_reproduction")
    config["output"]["run_name"] = "full_reproduction"

    with pytest.raises(RuntimeError, match="ERR_EXPERIMENT_MATRIX_CHILD_NOT_COMPLETED"):
        run_experiment_matrix(
            config,
            registry=_PartialMatrixRegistry(),
            run_dir=tmp_path / "partial_matrix",
            experiment_sequence=("main_model",),
        )


def test_run_all_matrix_resumes_completed_child(tmp_path):
    config = _config(tmp_path, "full_reproduction")
    config["output"]["run_name"] = "full_reproduction"
    config["full_reproduction"] = {"resume_completed_children": True}
    run_dir = tmp_path / "resume_matrix"
    child_logs = run_dir / "full_reproduction.01_main_model" / "logs"
    child_logs.mkdir(parents=True)
    (child_logs / "experiment_result.json").write_text(
        '{"status": "completed", "experiment_type": "main_model"}',
        encoding="utf-8",
    )

    registry = _FakeMatrixRegistry()
    result = run_experiment_matrix(
        config,
        registry=registry,
        run_dir=run_dir,
        experiment_sequence=("main_model", "baseline_comparison"),
    )

    assert result["status"] == "completed"
    assert registry.created == ["main_model", "baseline_comparison"]
    assert result["lineage"][0]["resumed_from_completed_child"] is True
    assert result["lineage"][1]["resumed_from_completed_child"] is False


def test_trainer_importable():
    from src.experiments.trainer import Trainer

    assert Trainer.__name__ == "Trainer"


class _FakeMatrixExperiment:
    def __init__(self, experiment_type):
        self.experiment_type = experiment_type
        self.output_name = f"{experiment_type}_output"

    def run(self):
        return {"status": "completed", "experiment_type": self.experiment_type}


class _FakeMatrixRegistry:
    def __init__(self):
        self.created = []

    def create_experiment(self, config, device=None, run_dir=None):
        experiment_type = config["experiment"]["type"]
        self.created.append(experiment_type)
        return _FakeMatrixExperiment(experiment_type)


class _PartialMatrixRegistry:
    def create_experiment(self, config, device=None, run_dir=None):
        return _PartialMatrixExperiment(config["experiment"]["type"])


class _PartialMatrixExperiment(_FakeMatrixExperiment):
    def run(self):
        return {"status": "partial", "experiment_type": self.experiment_type}


def _config(tmp_path, experiment_type):
    config = deepcopy(DEFAULT_CONFIG)
    config["experiment"]["type"] = experiment_type
    config["output"]["root"] = str(tmp_path)
    config["output"]["run_name"] = experiment_type
    return config


def _minimal_model_config(model_name):
    return {
        "device": {"mode": "cpu"},
        "n_features": 1,
        "window_size": 2,
        "n_assets": 2,
        "latent_dim": 4,
        "model": {"name": model_name, "dropout": 0.0},
        "encoder": {"type": "mlp", "dropout": 0.0},
        "ppo": {"hidden_dims": [8], "rollout_steps": 1, "minibatch_size": 1, "update_epochs": 1},
        "dqn": {"hidden_dims": [8], "dropout": 0.0, "batch_size": 1, "warmup_steps": 0},
        "optimizer": {},
        "training": {"epochs": 1},
        "evaluation": {"validation_episodes": 1},
        "output": {},
        "registry": {},
        "auxiliary": {},
    }


def _write_config(tmp_path, experiment_type, hpo_enabled=False, run_name="smoke", hpo_overrides=None):
    path = tmp_path / f"{run_name}.yaml"
    hpo_config = {"enabled": hpo_enabled}
    if hpo_overrides:
        hpo_config.update(hpo_overrides)
    payload = {
        "experiment": {"type": experiment_type},
        "hpo": hpo_config,
        "device": {"mode": "cpu"},
        "output": {"root": str(tmp_path / "results"), "run_name": run_name},
        "registry": {"enabled": True, "path": str(tmp_path / "run_registry.sqlite")},
        "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _registry_status(path, run_id):
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
