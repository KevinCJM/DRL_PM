import json
from copy import deepcopy

import pandas as pd
import pytest
import yaml

from src.config import DEFAULT_CONFIG, PROJECT_ROOT
from src.envs.backtest_engine import DAILY_REBALANCE_COLUMNS
from src.experiments.paper_aggregate import aggregate_paper_results
from src.utils.logger import (
    COST_CALIBRATION_COLUMNS,
    DAILY_OUTPUT_SCHEMAS,
    FEATURE_GROUP_SUMMARY_COLUMNS,
    FEATURE_PROVENANCE_COLUMNS,
    FEATURE_SELECTION_COLUMNS,
    BASELINE_DAILY_DIAGNOSTICS_COLUMNS,
    BASELINE_TRAINING_HISTORY_COLUMNS,
    BASELINE_TRAINING_SUMMARY_COLUMNS,
    HPO_MODEL_FINAL_DAILY_DIAGNOSTICS_COLUMNS,
    HPO_TRIAL_COLUMNS,
    METRICS_FACTORY_AUDIT_COLUMNS,
    SEED_AGGREGATE_COLUMNS,
    write_run_outputs,
)
from src.utils.stats import STATISTICS_SUMMARY_COLUMNS


def test_daily_output_schema(tmp_path):
    result = {
        "daily_returns": pd.DataFrame(
            {
                "decision_date": ["2024-01-02"],
                "execution_date": ["2024-01-03"],
                "execution_price_type": ["open"],
                "next_valuation_date": ["2024-01-03"],
                "split": ["test"],
                "seed": [42],
                "fold_id": ["fixed"],
                "model_name": ["model"],
                "pre_execution_return": [0.01],
                "post_execution_return": [0.02],
                "gross_return": [0.0302],
                "transaction_cost": [0.001],
                "transaction_cost_on_initial_nav": [0.00101],
                "net_return": [0.029168],
                "portfolio_log_return": [0.028751],
                "nav": [1.029168],
                "reward": [99.0],
            }
        ),
        "daily_weights": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "split": ["test"],
                "seed": [42],
                "fold_id": ["fixed"],
                "model_name": ["model"],
                "asset_id": ["510300.SH"],
                "weight": [1.0],
            }
        ),
        "daily_turnover": pd.DataFrame(
            {
                "date": ["wrong-date"],
                "decision_date": ["2024-01-02"],
                "execution_date": ["2024-01-03"],
                "execution_price_type": ["open"],
                "next_valuation_date": ["2024-01-03"],
                "split": ["test"],
                "seed": [42],
                "fold_id": ["fixed"],
                "model_name": ["model"],
                "turnover": [0.25],
                "rebalance_action": [1],
                "rebalance_intensity": [1.0],
                "average_holding_period": [4.0],
            }
        ),
        "daily_rebalance": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "decision_date": ["2024-01-02"],
                "execution_date": ["2024-01-03"],
                "execution_price_type": ["open"],
                "next_valuation_date": ["2024-01-03"],
                "split": ["test"],
                "seed": [42],
                "fold_id": ["fixed"],
                "model_name": ["model"],
                "rebalance_action": [1],
                "rebalance_intensity": [1.0],
                "estimated_turnover": [0.20],
                "realized_turnover": [0.25],
                "turnover": [0.25],
                "estimated_cost": [0.0008],
                "realized_cost": [0.001],
                "q_hold": [0.1],
                "q_rebalance": [0.2],
                "q_gap": [0.1],
                "fallback_reason": ["invalid_candidate_weights_equal_weight"],
                "paper_model_id": ["ppo_dqn_hierarchical_reimplementation"],
                "hierarchy_action": [4],
                "optimizer_name": ["risk_parity"],
                "optimizer_status": ["fallback_equal_weight"],
            }
        ),
        "daily_costs": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "decision_date": ["2024-01-02"],
                "execution_date": ["2024-01-03"],
                "execution_price_type": ["open"],
                "next_valuation_date": ["2024-01-03"],
                "split": ["test"],
                "seed": [42],
                "fold_id": ["fixed"],
                "model_name": ["model"],
                "proportional_cost": [0.0005],
                "fixed_cost": [0.0],
                "slippage_cost": [0.0002],
                "market_impact_cost": [0.0003],
                "total_transaction_cost": [0.001],
                "estimated_cost": [0.0008],
                "realized_cost": [0.001],
                "turnover": [0.25],
            }
        ),
        "output_name": "main_comparison",
        "main_comparison": pd.DataFrame(
            {
                "model_name": ["model", "equal_weight"],
                "role": ["model", "benchmark"],
                "cumulative_return": [0.029168, 0.01],
            }
        ),
        "baseline_daily_diagnostics": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "decision_date": ["2024-01-02"],
                "execution_date": ["2024-01-03"],
                "model_name": ["ppo_dqn_hierarchical_reimplementation"],
                "paper_model_id": ["ppo_dqn_hierarchical_reimplementation"],
                "seed": [42],
                "fold_id": ["fixed"],
                "hierarchy_action": [4],
                "optimizer_name": ["risk_parity"],
                "optimizer_status": ["fallback_equal_weight"],
                "fallback_reason": ["invalid_candidate_weights_equal_weight"],
            }
        ),
    }

    artifacts = write_run_outputs(result, tmp_path)

    expected_rebalance_schema = []
    for column in DAILY_REBALANCE_COLUMNS:
        expected_rebalance_schema.append(column)
        if column == "model_name":
            expected_rebalance_schema.append("variant_id")
    assert list(DAILY_OUTPUT_SCHEMAS["daily_rebalance"]) == expected_rebalance_schema

    for name, columns in DAILY_OUTPUT_SCHEMAS.items():
        output = pd.read_csv(artifacts[name])
        assert list(output.columns) == list(columns)
        if "next_valuation_date" in output.columns:
            assert output["date"].tolist() == output["next_valuation_date"].tolist()

    rebalance = pd.read_csv(tmp_path / "metrics" / "daily_rebalance.csv")
    diagnostic_only_columns = {
        "paper_model_id",
        "hierarchy_action",
        "hierarchy_action_name",
        "ppo_actor_update_mask",
        "ppo_attribution_weight",
        "platform_adapted_surrogate",
        "child_model_name",
        "baseline_family",
        "optimizer_name",
        "include_count",
        "exclude_count",
        "neutral_count",
        "selected_asset_count",
        "optimizer_asset_count",
        "optimizer_status",
    }
    assert diagnostic_only_columns.isdisjoint(DAILY_REBALANCE_COLUMNS)
    assert diagnostic_only_columns.isdisjoint(DAILY_OUTPUT_SCHEMAS["daily_rebalance"])
    assert diagnostic_only_columns.isdisjoint(rebalance.columns)
    assert {"q_hold", "q_rebalance", "q_gap"}.issubset(rebalance.columns)
    assert rebalance.loc[0, "fallback_reason"] == "invalid_candidate_weights_equal_weight"
    sidecar = pd.read_csv(artifacts["baseline_daily_diagnostics"])
    assert {"paper_model_id", "hierarchy_action", "optimizer_name", "optimizer_status"}.issubset(sidecar.columns)
    assert sidecar.loc[0, "paper_model_id"] == "ppo_dqn_hierarchical_reimplementation"

    metrics = pd.read_csv(tmp_path / "metrics" / "metrics.csv")
    assert metrics.loc[0, "n_steps"] == 1.0
    assert metrics.loc[0, "cumulative_return"] == pytest.approx(0.029168)
    assert metrics.loc[0, "turnover"] == pytest.approx(0.25)
    assert metrics.loc[0, "total_transaction_cost"] == pytest.approx(0.001)
    for split in ("train", "validation", "test"):
        split_metrics = pd.read_csv(tmp_path / "metrics" / f"{split}_metrics.csv")
        assert split_metrics.loc[0, "status"] == "not_applicable"
    matrix = pd.read_csv(tmp_path / "metrics" / "main_comparison.csv")
    assert matrix["model_name"].tolist() == ["model", "equal_weight"]
    assert artifacts["main_comparison"] == tmp_path / "metrics" / "main_comparison.csv"


def test_baseline_training_summary_and_checkpoint_path_are_persisted(tmp_path):
    result = {
        "daily_returns": pd.DataFrame(
            {"next_valuation_date": ["2024-01-03"], "execution_price_type": ["open"], "net_return": [0.01], "nav": [1.01]}
        ),
        "daily_weights": pd.DataFrame({"date": ["2024-01-03"], "asset_id": ["A"], "weight": [1.0]}),
        "baseline_training_summary": pd.DataFrame(
            {
                "model_name": ["ppo_native"],
                "paper_model_id": ["ppo_dqn_hierarchical_reimplementation"],
                "child_model_name": ["ppo_dqn_hierarchical_reimplementation"],
                "baseline_family": ["native_rl"],
                "status": ["completed"],
                "training_algorithm": ["ppo_clipped_gae"],
                "rl_training": [True],
                "platform_native_rl_training": [True],
                "proxy_training": [False],
                "external_original_implementation": [False],
                "rankable_in_unified_table": [True],
                "algorithm_fidelity": ["platform_adapted"],
                "dqn_role": ["high_level_action_selector"],
                "optimizer_name": ["risk_parity"],
                "platform_adapted_surrogate": [True],
                "platform_adapted_approximation": [True],
                "checkpoint_best_path": [str(tmp_path / "best.pt")],
                "checkpoint_last_path": [str(tmp_path / "last.pt")],
                "evaluated_checkpoint_path": [str(tmp_path / "best.pt")],
                "best_validation_metric": [0.1],
                "env_steps": [8],
                "gradient_updates": [2],
            }
        ),
        "baseline_training_history": pd.DataFrame(
            {
                "model_name": ["ppo_native"],
                "epoch": [0],
                "step": [1],
                "env_steps": [8],
                "gradient_updates": [2],
                "train_reward": [0.01],
                "validation_metric": [0.1],
                "loss": [0.2],
                "include_count": [2],
                "neutral_count": [1],
                "exclude_count": [1],
                "selected_asset_count": [2],
                "optimizer_asset_count": [3],
                "optimizer_fallback_count": [1],
                "ppo_actor_update_mask_rate": [0.75],
                "ppo_attribution_weight_mean": [0.5],
                "platform_adapted_surrogate": [True],
                "status": ["completed"],
            }
        ),
    }

    artifacts = write_run_outputs(result, tmp_path)

    summary = pd.read_csv(artifacts["baseline_training_summary"])
    history = pd.read_csv(artifacts["baseline_training_history"])
    required_summary_columns = {
        "paper_model_id",
        "child_model_name",
        "algorithm_fidelity",
        "dqn_role",
        "optimizer_name",
        "platform_adapted_surrogate",
        "platform_adapted_approximation",
    }
    required_history_columns = {
        "include_count",
        "neutral_count",
        "exclude_count",
        "selected_asset_count",
        "optimizer_asset_count",
        "optimizer_fallback_count",
        "ppo_actor_update_mask_rate",
        "ppo_attribution_weight_mean",
        "platform_adapted_surrogate",
    }
    assert required_summary_columns.issubset(BASELINE_TRAINING_SUMMARY_COLUMNS)
    assert required_summary_columns.issubset(summary.columns)
    assert required_history_columns.issubset(BASELINE_TRAINING_HISTORY_COLUMNS)
    assert required_history_columns.issubset(history.columns)
    assert summary.loc[0, "evaluated_checkpoint_path"] == str(tmp_path / "best.pt")
    assert bool(summary.loc[0, "platform_native_rl_training"]) is True
    assert summary.loc[0, "paper_model_id"] == "ppo_dqn_hierarchical_reimplementation"
    assert summary.loc[0, "algorithm_fidelity"] == "platform_adapted"
    assert bool(summary.loc[0, "platform_adapted_surrogate"]) is True
    assert bool(summary.loc[0, "platform_adapted_approximation"]) is True
    assert history.loc[0, "validation_metric"] == pytest.approx(0.1)
    assert history.loc[0, "include_count"] == 2
    assert history.loc[0, "neutral_count"] == 1
    assert history.loc[0, "exclude_count"] == 1
    assert history.loc[0, "optimizer_fallback_count"] == 1
    assert history.loc[0, "ppo_actor_update_mask_rate"] == pytest.approx(0.75)
    assert history.loc[0, "ppo_attribution_weight_mean"] == pytest.approx(0.5)
    assert bool(history.loc[0, "platform_adapted_surrogate"]) is True


def test_write_run_outputs_persists_hpo_model_final_tables(tmp_path):
    result = {
        "daily_returns": pd.DataFrame(
            {"next_valuation_date": ["2024-01-03"], "execution_price_type": ["open"], "net_return": [0.01], "nav": [1.01]}
        ),
        "daily_weights": pd.DataFrame({"date": ["2024-01-03"], "asset_id": ["A"], "weight": [1.0]}),
        "hpo_model_final_comparison": pd.DataFrame(
            {"model_name": ["ppo_native"], "best_value": [0.2], "sharpe": [1.1]}
        ),
        "hpo_model_final_reports": pd.DataFrame(
            {"model_name": ["ppo_native"], "rank_label": ["best"], "trial_number": [2], "sharpe": [1.1]}
        ),
        "hpo_model_final_daily_returns": pd.DataFrame(
            {"date": ["2024-01-03"], "model_name": ["ppo_native"], "net_return": [0.02], "nav": [1.02]}
        ),
        "hpo_model_final_daily_diagnostics": pd.DataFrame(
            {
                "hpo_model_name": ["ppo_dqn_hierarchical_reimplementation"],
                "date": ["2024-01-03"],
                "decision_date": ["2024-01-02"],
                "execution_date": ["2024-01-03"],
                "model_name": ["ppo_dqn_hierarchical_reimplementation"],
                "paper_model_id": ["ppo_dqn_hierarchical_reimplementation"],
                "seed": [42],
                "fold_id": ["fixed"],
                "hierarchy_action": [2],
                "platform_adapted_surrogate": [True],
                "best_trial_number": [2],
                "best_value": [0.2],
            }
        ),
    }

    artifacts = write_run_outputs(result, tmp_path)

    assert artifacts["hpo_model_final_comparison"] == tmp_path / "metrics" / "hpo_model_final_comparison.csv"
    assert artifacts["hpo_model_final_reports"] == tmp_path / "metrics" / "hpo_model_final_reports.csv"
    assert artifacts["hpo_model_final_daily_returns"] == tmp_path / "metrics" / "hpo_model_final_daily_returns.csv"
    assert artifacts["hpo_model_final_daily_diagnostics"] == tmp_path / "metrics" / "hpo_model_final_daily_diagnostics.csv"
    final_comparison = pd.read_csv(artifacts["hpo_model_final_comparison"])
    final_reports = pd.read_csv(artifacts["hpo_model_final_reports"])
    final_returns = pd.read_csv(artifacts["hpo_model_final_daily_returns"])
    final_diagnostics = pd.read_csv(artifacts["hpo_model_final_daily_diagnostics"])
    assert final_comparison.loc[0, "model_name"] == "ppo_native"
    assert final_reports.loc[0, "rank_label"] == "best"
    assert final_returns.loc[0, "net_return"] == pytest.approx(0.02)
    assert list(final_diagnostics.columns[: len(HPO_MODEL_FINAL_DAILY_DIAGNOSTICS_COLUMNS)]) == list(
        HPO_MODEL_FINAL_DAILY_DIAGNOSTICS_COLUMNS
    )
    assert final_diagnostics.loc[0, "hpo_model_name"] == "ppo_dqn_hierarchical_reimplementation"
    assert final_diagnostics.loc[0, "paper_model_id"] == "ppo_dqn_hierarchical_reimplementation"


def test_related_work_daily_diagnostics_written(tmp_path):
    diagnostics = pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "decision_date": "2024-01-02",
                "execution_date": "2024-01-03",
                "model_name": "ppo_dqn_hierarchical_reimplementation",
                "paper_model_id": "ppo_dqn_hierarchical_reimplementation",
                "seed": 42,
                "fold_id": "fixed",
                "hierarchy_action": 2,
                "platform_adapted_surrogate": True,
            }
        ]
    )
    hpo_diagnostics = diagnostics.copy()
    hpo_diagnostics["hpo_model_name"] = "ppo_dqn_hierarchical_reimplementation"
    result = {
        "daily_returns": pd.DataFrame(
            {"next_valuation_date": ["2024-01-03"], "execution_price_type": ["open"], "net_return": [0.01], "nav": [1.01]}
        ),
        "daily_weights": pd.DataFrame({"date": ["2024-01-03"], "asset_id": ["A"], "weight": [1.0]}),
        "baseline_daily_diagnostics": diagnostics,
        "hpo_model_final_daily_diagnostics": hpo_diagnostics,
    }

    artifacts = write_run_outputs(result, tmp_path)

    assert artifacts["baseline_daily_diagnostics"] == tmp_path / "metrics" / "baseline_daily_diagnostics.csv"
    assert artifacts["hpo_model_final_daily_diagnostics"] == tmp_path / "metrics" / "hpo_model_final_daily_diagnostics.csv"
    baseline = pd.read_csv(artifacts["baseline_daily_diagnostics"])
    hpo = pd.read_csv(artifacts["hpo_model_final_daily_diagnostics"])
    assert list(baseline.columns[: len(BASELINE_DAILY_DIAGNOSTICS_COLUMNS)]) == list(BASELINE_DAILY_DIAGNOSTICS_COLUMNS)
    assert list(hpo.columns[: len(HPO_MODEL_FINAL_DAILY_DIAGNOSTICS_COLUMNS)]) == list(HPO_MODEL_FINAL_DAILY_DIAGNOSTICS_COLUMNS)
    assert baseline.loc[0, "paper_model_id"] == "ppo_dqn_hierarchical_reimplementation"
    assert hpo.loc[0, "paper_model_id"] == "ppo_dqn_hierarchical_reimplementation"
    assert hpo.loc[0, "hpo_model_name"] == "ppo_dqn_hierarchical_reimplementation"


def test_write_run_outputs_preserves_variant_id_in_daily_outputs(tmp_path):
    result = {
        "daily_returns": pd.DataFrame(
            {
                "next_valuation_date": ["2024-01-03"],
                "execution_price_type": ["open"],
                "model_name": ["model"],
                "variant_id": ["input_matrix_M7"],
                "net_return": [0.01],
                "nav": [1.01],
            }
        ),
        "daily_weights": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "model_name": ["model"],
                "variant_id": ["input_matrix_M7"],
                "asset_id": ["510300.SH"],
                "weight": [1.0],
            }
        ),
        "daily_turnover": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "model_name": ["model"],
                "variant_id": ["input_matrix_M7"],
                "turnover": [0.1],
            }
        ),
        "daily_rebalance": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "model_name": ["model"],
                "variant_id": ["input_matrix_M7"],
                "rebalance_action": [1],
            }
        ),
        "daily_costs": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "model_name": ["model"],
                "variant_id": ["input_matrix_M7"],
                "total_transaction_cost": [0.001],
            }
        ),
    }

    artifacts = write_run_outputs(result, tmp_path)

    for key in ("daily_returns", "daily_weights", "daily_turnover", "daily_rebalance", "daily_costs"):
        output = pd.read_csv(artifacts[key])
        assert output.loc[0, "variant_id"] == "input_matrix_M7"


def test_paper_aggregate_generates_main_stats_and_seed_summary(tmp_path):
    run_dir = tmp_path / "results" / "run_s42"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(
        json.dumps({"run_name": "run_s42", "seed": 42}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"model_name": "model", "rankable_in_unified_table": True, "sharpe": 1.2},
            {"model_name": "equal_weight", "rankable_in_unified_table": True, "sharpe": 0.8},
            {"model_name": "ppo_proxy", "rankable_in_unified_table": False, "sharpe": 2.0},
            {
                "model_name": "hybrid_dqn_optimizer_equal_weight",
                "paper_model_id": "hybrid_dqn_optimizer_equal_weight",
                "rankable_in_unified_table": True,
                "sharpe": 1.4,
            },
            {
                "model_name": "hybrid_dqn_optimizer_reimplementation",
                "paper_model_id": "hybrid_dqn_optimizer_reimplementation",
                "rankable_in_unified_table": True,
                "sharpe": 9.0,
            },
            {
                "model_name": "shared_dqn_diagnostic",
                "paper_model_id": "hybrid_dqn_optimizer_risk_parity",
                "rankable_in_unified_table": True,
                "diagnostic_status": "diagnostic_shared_dqn",
                "sharpe": 8.0,
            },
            {
                "model_name": "partial_hybrid_smoke",
                "paper_model_id": "partial_hybrid_smoke",
                "rankable_in_unified_table": False,
                "diagnostic_status": "partial_diagnostic",
                "sharpe": 7.0,
            },
            {
                "model_name": "runtime_alias",
                "paper_model_id": "explicit_paper_id",
                "rankable_in_unified_table": True,
                "variant_id": "ignored_variant",
                "sharpe": 1.1,
            },
        ]
    ).to_csv(metrics_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
            "model_name": ["model", "model", "equal_weight", "equal_weight", "ppo_proxy", "ppo_proxy"],
            "split": ["test"] * 6,
            "net_return": [0.02, 0.01, 0.01, 0.0, 0.03, 0.03],
        }
    ).to_csv(metrics_dir / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results([run_dir], tmp_path / "paper", benchmark_model="equal_weight")

    main = pd.read_csv(outputs["paper_main_comparison"])
    stats = pd.read_csv(outputs["paper_paired_statistics"])
    seed_summary = pd.read_csv(outputs["paper_seed_summary"])
    assert set(outputs) == {
        "paper_main_comparison",
        "paper_diagnostic_comparison",
        "paper_paired_statistics",
        "paper_seed_summary",
        "closest_hybrid_figure_source",
    }
    assert "model" in set(main["model_name"])
    assert "model" in set(main["paper_model_id"])
    assert "hybrid_dqn_optimizer_equal_weight" in set(main["paper_model_id"])
    assert "explicit_paper_id" in set(main["paper_model_id"])
    assert "ignored_variant" not in set(main["paper_model_id"])
    assert "hybrid_dqn_optimizer_reimplementation" not in set(main["model_name"])
    assert "hybrid_dqn_optimizer_reimplementation" not in set(main["paper_model_id"])
    assert "hybrid_dqn_optimizer_risk_parity" not in set(main["paper_model_id"])
    assert "partial_hybrid_smoke" not in set(main["paper_model_id"])
    assert "ppo_proxy" not in set(main["paper_model_id"])
    assert "ppo_proxy" not in set(stats["model_name"].dropna())
    assert "sharpe" in set(seed_summary["metric_name"])
    assert "hybrid_dqn_optimizer_reimplementation" not in set(seed_summary["paper_model_id"])


def test_paper_aggregate_filters_hybrid_alias_from_daily_only_main(tmp_path):
    run_dir = tmp_path / "results" / "daily_only"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(json.dumps({"run_name": "daily_only", "seed": 42}), encoding="utf-8")
    child_ids = [
        "hybrid_dqn_optimizer_equal_weight",
        "hybrid_dqn_optimizer_markowitz_mean_variance",
        "hybrid_dqn_optimizer_minimum_variance",
        "hybrid_dqn_optimizer_sharpe_maximization",
        "hybrid_dqn_optimizer_risk_parity",
    ]
    rows = [
        {
            "date": "2024-01-02",
            "model_name": "hybrid_dqn_optimizer_reimplementation",
            "paper_model_id": "hybrid_dqn_optimizer_reimplementation",
            "rankable_in_unified_table": True,
            "net_return": 0.01,
        },
        {
            "date": "2024-01-02",
            "model_name": "non_rankable_reference",
            "paper_model_id": "non_rankable_reference",
            "rankable_in_unified_table": False,
            "net_return": 0.02,
        },
        {
            "date": "2024-01-02",
            "model_name": "shared_dqn_diagnostic",
            "paper_model_id": "shared_dqn_diagnostic",
            "rankable_in_unified_table": True,
            "diagnostic_status": "diagnostic_shared_dqn",
            "net_return": 0.03,
        },
        {
            "date": "2024-01-02",
            "model_name": "partial_hybrid_smoke",
            "paper_model_id": "partial_hybrid_smoke",
            "rankable_in_unified_table": True,
            "diagnostic_status": "partial_diagnostic",
            "net_return": 0.04,
        },
    ]
    rows.extend(
        {
            "date": "2024-01-02",
            "model_name": "hybrid_dqn_optimizer_reimplementation",
            "paper_model_id": child_id,
            "rankable_in_unified_table": True,
            "net_return": 0.05,
        }
        for child_id in child_ids
    )
    pd.DataFrame(
        rows
    ).to_csv(metrics_dir / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results([run_dir], tmp_path / "paper", benchmark_model="hybrid_dqn_optimizer_equal_weight")

    main = pd.read_csv(outputs["paper_main_comparison"])
    assert set(main["paper_model_id"]) == set(child_ids)
    assert "hybrid_dqn_optimizer_reimplementation" not in set(main["paper_model_id"])
    assert "non_rankable_reference" not in set(main["paper_model_id"])
    assert "shared_dqn_diagnostic" not in set(main["paper_model_id"])
    assert "partial_hybrid_smoke" not in set(main["paper_model_id"])


def test_paper_aggregate_supports_multiple_benchmarks_and_seed_metric_whitelist(tmp_path):
    run_dir = tmp_path / "results" / "run_s42"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(json.dumps({"run_name": "run_s42", "seed": 42}), encoding="utf-8")
    pd.DataFrame(
        [
            {"model_name": "main", "rankable_in_unified_table": True, "baseline_family": "model", "sharpe": 1.2, "best_trial_number": 3},
            {"model_name": "equal_weight", "rankable_in_unified_table": True, "baseline_family": "traditional", "sharpe": 0.4, "best_trial_number": 0},
            {"model_name": "cnn_ppo_native", "rankable_in_unified_table": True, "baseline_family": "native_rl", "sharpe": 0.9, "best_trial_number": 1},
        ]
    ).to_csv(metrics_dir / "main_comparison.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"] * 3,
            "model_name": ["main", "main", "equal_weight", "equal_weight", "cnn_ppo_native", "cnn_ppo_native"],
            "split": ["test"] * 6,
            "net_return": [0.03, 0.01, 0.01, 0.0, 0.02, 0.01],
        }
    ).to_csv(metrics_dir / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results(
        [run_dir],
        tmp_path / "paper",
        benchmark_models=["equal_weight", "cnn_ppo_native"],
        seed_metric_columns=["sharpe"],
    )

    stats = pd.read_csv(outputs["paper_paired_statistics"])
    seed_summary = pd.read_csv(outputs["paper_seed_summary"])
    assert {"equal_weight", "cnn_ppo_native"}.issubset(set(stats["benchmark_name"].dropna()))
    assert set(seed_summary["metric_name"]) == {"sharpe"}
    assert "best_trial_number" not in set(seed_summary["metric_name"])


def test_paper_aggregate_uses_source_run_and_variant_identity(tmp_path):
    run_dir = tmp_path / "results" / "P3_without_dqn_gate"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(
        json.dumps({"run_name": "P3_without_dqn_gate", "experiment_type": "ablation", "seed": 42}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"variant_id": "full_model", "sharpe": 1.1},
            {"variant_id": "without_dqn_gate", "sharpe": 0.6},
        ]
    ).to_csv(metrics_dir / "ablation_results.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"] * 2,
            "model_name": ["full_dqn_gated_multitask_cnn_ppo"] * 4,
            "variant_id": ["full_model", "full_model", "without_dqn_gate", "without_dqn_gate"],
            "split": ["test"] * 4,
            "net_return": [0.03, 0.01, 0.01, 0.0],
        }
    ).to_csv(metrics_dir / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results([run_dir], tmp_path / "paper", benchmark_model="without_dqn_gate")

    main = pd.read_csv(outputs["paper_main_comparison"])
    stats = pd.read_csv(outputs["paper_paired_statistics"])
    seed_summary = pd.read_csv(outputs["paper_seed_summary"])
    assert {"full_model", "without_dqn_gate"} == set(main["paper_model_id"])
    assert set(stats["model_name"].dropna()) == {"full_model"}
    assert set(seed_summary["paper_model_id"]) == {"full_model", "without_dqn_gate"}


def test_paper_aggregate_prefers_hpo_model_final_comparison(tmp_path):
    run_dir = tmp_path / "results" / "P7_hpo"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(
        json.dumps({"run_name": "P7_hpo", "experiment_type": "hyperparameter_sweep", "seed": 42}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"model_name": "ppo_native", "rankable_in_unified_table": True, "sharpe": 0.5},
            {"model_name": "equal_weight", "rankable_in_unified_table": True, "sharpe": 0.2},
        ]
    ).to_csv(metrics_dir / "main_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"model_name": "ppo_native", "rankable_in_unified_table": True, "sharpe": 1.1},
            {"model_name": "cnn_ppo_native", "rankable_in_unified_table": True, "sharpe": 0.9},
        ]
    ).to_csv(metrics_dir / "hpo_model_final_comparison.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"] * 2,
            "model_name": ["ppo_native", "ppo_native", "cnn_ppo_native", "cnn_ppo_native"],
            "net_return": [0.02, 0.01, 0.01, 0.0],
        }
    ).to_csv(metrics_dir / "hpo_model_final_daily_returns.csv", index=False)

    outputs = aggregate_paper_results([run_dir], tmp_path / "paper", benchmark_model="cnn_ppo_native")

    main = pd.read_csv(outputs["paper_main_comparison"])
    assert set(main["paper_model_id"]) == {"ppo_native", "cnn_ppo_native"}
    assert main["paper_model_id"].tolist().count("ppo_native") == 1
    assert set(main["source_file"]) == {"hpo_model_final_comparison.csv"}


def test_paper_aggregate_paired_stats_can_share_group_across_runs(tmp_path):
    dates = pd.date_range("2024-01-02", periods=25, freq="D").strftime("%Y-%m-%d").tolist()
    main_run = tmp_path / "results" / "EXP01_main"
    baseline_run = tmp_path / "results" / "EXP04_baselines"
    for run_dir, run_name, experiment_type in (
        (main_run, "EXP01_main", "main_model"),
        (baseline_run, "EXP04_baselines", "baseline_comparison"),
    ):
        (run_dir / "metrics").mkdir(parents=True)
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "logs" / "run_manifest.json").write_text(
            json.dumps({"run_name": run_name, "experiment_type": experiment_type, "seed": 42}),
            encoding="utf-8",
        )
    pd.DataFrame(
        [{"model_name": "full_dqn_gated_multitask_cnn_ppo", "rankable_in_unified_table": True, "sharpe": 1.0}]
    ).to_csv(main_run / "metrics" / "main_comparison.csv", index=False)
    pd.DataFrame(
        [{"model_name": "cnn_ppo_native", "rankable_in_unified_table": True, "sharpe": 0.7}]
    ).to_csv(baseline_run / "metrics" / "baseline_comparison.csv", index=False)
    pd.DataFrame(
        {"date": dates, "model_name": ["full_dqn_gated_multitask_cnn_ppo"] * len(dates), "net_return": [0.01] * len(dates)}
    ).to_csv(main_run / "metrics" / "daily_returns.csv", index=False)
    pd.DataFrame(
        {"date": dates, "model_name": ["cnn_ppo_native"] * len(dates), "net_return": [0.005] * len(dates)}
    ).to_csv(baseline_run / "metrics" / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results(
        [main_run, baseline_run],
        tmp_path / "paper",
        benchmark_model="cnn_ppo_native",
        paper_group_id="main_fixed",
    )

    stats = pd.read_csv(outputs["paper_paired_statistics"])
    assert "full_dqn_gated_multitask_cnn_ppo" in set(stats["model_name"].dropna())
    assert set(stats["benchmark_name"].dropna()) == {"cnn_ppo_native"}
    assert set(stats["paper_group"].dropna()) == {"main_fixed|42"}


def test_paper_aggregate_dedupes_duplicate_daily_models_in_shared_group(tmp_path):
    dates = pd.date_range("2024-01-02", periods=25, freq="D").strftime("%Y-%m-%d").tolist()
    main_run = tmp_path / "results" / "EXP01_main"
    baseline_run = tmp_path / "results" / "EXP04_baselines"
    for run_dir, run_name, experiment_type in (
        (main_run, "EXP01_main", "main_model"),
        (baseline_run, "EXP04_baselines", "baseline_comparison"),
    ):
        (run_dir / "metrics").mkdir(parents=True)
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "logs" / "run_manifest.json").write_text(
            json.dumps({"run_name": run_name, "experiment_type": experiment_type, "seed": 42}),
            encoding="utf-8",
        )
    pd.DataFrame(
        [
            {"model_name": "full_dqn_gated_multitask_cnn_ppo", "rankable_in_unified_table": True, "sharpe": 1.0},
            {"model_name": "equal_weight", "rankable_in_unified_table": True, "sharpe": 0.4},
        ]
    ).to_csv(main_run / "metrics" / "main_comparison.csv", index=False)
    pd.DataFrame(
        [{"model_name": "equal_weight", "rankable_in_unified_table": True, "sharpe": 0.5}]
    ).to_csv(baseline_run / "metrics" / "baseline_comparison.csv", index=False)
    pd.DataFrame(
        {
            "date": dates * 2,
            "model_name": ["full_dqn_gated_multitask_cnn_ppo"] * len(dates) + ["equal_weight"] * len(dates),
            "net_return": [0.01] * len(dates) + [0.005] * len(dates),
        }
    ).to_csv(main_run / "metrics" / "daily_returns.csv", index=False)
    pd.DataFrame(
        {"date": dates, "model_name": ["equal_weight"] * len(dates), "net_return": [0.004] * len(dates)}
    ).to_csv(baseline_run / "metrics" / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results(
        [main_run, baseline_run],
        tmp_path / "paper",
        benchmark_model="equal_weight",
        paper_group_id="main_fixed",
    )

    stats = pd.read_csv(outputs["paper_paired_statistics"])
    assert set(stats["model_name"].dropna()) == {"full_dqn_gated_multitask_cnn_ppo"}
    assert pd.to_numeric(stats["n_obs"], errors="coerce").dropna().eq(len(dates)).all()


def test_paper_aggregate_dedupes_duplicate_comparison_models_in_shared_group(tmp_path):
    main_run = tmp_path / "results" / "EXP01_main"
    baseline_run = tmp_path / "results" / "EXP04_baselines"
    for run_dir, run_name, experiment_type in (
        (main_run, "EXP01_main", "main_model"),
        (baseline_run, "EXP04_baselines", "baseline_comparison"),
    ):
        (run_dir / "metrics").mkdir(parents=True)
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "logs" / "run_manifest.json").write_text(
            json.dumps({"run_name": run_name, "experiment_type": experiment_type, "seed": 42}),
            encoding="utf-8",
        )
    pd.DataFrame(
        [
            {"model_name": "full_dqn_gated_multitask_cnn_ppo", "rankable_in_unified_table": True, "sharpe": 1.0},
            {"model_name": "equal_weight", "rankable_in_unified_table": True, "sharpe": 0.4},
        ]
    ).to_csv(main_run / "metrics" / "main_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"model_name": "equal_weight", "rankable_in_unified_table": True, "sharpe": 0.5},
            {"model_name": "cnn_ppo_native", "rankable_in_unified_table": True, "sharpe": 0.7},
        ]
    ).to_csv(baseline_run / "metrics" / "baseline_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-02", "model_name": "full_dqn_gated_multitask_cnn_ppo", "net_return": 0.01},
            {"date": "2024-01-02", "model_name": "equal_weight", "net_return": 0.005},
        ]
    ).to_csv(main_run / "metrics" / "daily_returns.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-02", "model_name": "equal_weight", "net_return": 0.004},
            {"date": "2024-01-02", "model_name": "cnn_ppo_native", "net_return": 0.006},
        ]
    ).to_csv(baseline_run / "metrics" / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results(
        [main_run, baseline_run],
        tmp_path / "paper",
        benchmark_model="equal_weight",
        paper_group_id="main_fixed",
    )

    main = pd.read_csv(outputs["paper_main_comparison"])
    assert main.duplicated(["paper_group_id", "paper_model_id", "seed"]).sum() == 0
    equal_weight = main.loc[main["paper_model_id"].eq("equal_weight")].iloc[0]
    assert equal_weight["source_file"] == "main_comparison.csv"
    assert equal_weight["sharpe"] == pytest.approx(0.4)


def test_paper_aggregate_treats_missing_rankable_as_included_for_matrix_rows(tmp_path):
    run_dir = tmp_path / "results" / "P2_input_matrix"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(
        json.dumps({"run_name": "P2_input_matrix", "experiment_type": "input_matrix_ablation", "seed": 42}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"variant_id": "input_matrix_M0", "status": "completed", "sharpe": 0.7},
            {"variant_id": "input_matrix_M6", "status": "completed", "sharpe": 1.0},
        ]
    ).to_csv(metrics_dir / "input_matrix_ablation_results.csv", index=False)
    dates = pd.date_range("2024-01-02", periods=25, freq="D").strftime("%Y-%m-%d").tolist()
    pd.DataFrame(
        {
            "date": dates * 2,
            "model_name": ["full_dqn_gated_multitask_cnn_ppo"] * (len(dates) * 2),
            "variant_id": ["input_matrix_M0"] * len(dates) + ["input_matrix_M6"] * len(dates),
            "net_return": [0.004] * len(dates) + [0.006] * len(dates),
        }
    ).to_csv(metrics_dir / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results(
        [run_dir],
        tmp_path / "paper",
        benchmark_model="input_matrix_M0",
        paper_group_id="p2_input_pca",
    )

    main = pd.read_csv(outputs["paper_main_comparison"])
    stats = pd.read_csv(outputs["paper_paired_statistics"])
    assert main["paper_included"].all()
    assert set(main["paper_model_id"]) == {"input_matrix_M0", "input_matrix_M6"}
    assert set(stats["model_name"].dropna()) == {"input_matrix_M6"}


def test_paper_aggregate_infers_walk_forward_identity_from_experiment_result(tmp_path):
    run_dir = tmp_path / "results" / "P6_walk_forward"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(
        json.dumps({"run_name": "P6_walk_forward", "experiment_type": "walk_forward", "seed": 42}),
        encoding="utf-8",
    )
    (logs_dir / "experiment_result.json").write_text(
        json.dumps({"status": "completed", "model_name": "full_dqn_gated_multitask_cnn_ppo"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"fold_id": "0", "n_steps": 25, "sharpe": 0.8},
            {"fold_id": "all_oos", "n_steps": 50, "sharpe": 1.1},
        ]
    ).to_csv(metrics_dir / "walk_forward_results.csv", index=False)
    dates = pd.date_range("2024-01-02", periods=25, freq="D").strftime("%Y-%m-%d").tolist()
    pd.DataFrame(
        {
            "date": dates,
            "fold_id": ["0"] * len(dates),
            "model_name": ["full_dqn_gated_multitask_cnn_ppo"] * len(dates),
            "net_return": [0.006] * len(dates),
        }
    ).to_csv(metrics_dir / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results(
        [run_dir],
        tmp_path / "paper",
        benchmark_model="full_dqn_gated_multitask_cnn_ppo",
        paper_group_id="p6_robustness",
    )

    main = pd.read_csv(outputs["paper_main_comparison"])
    assert "unknown_model" not in set(main["paper_model_id"])
    assert set(main["paper_model_id"]) == {"full_dqn_gated_multitask_cnn_ppo"}
    assert set(main["status"]) == {"completed"}
    assert set(main["fold_id"].astype(str)) == {"0", "all_oos"}


def test_paper_aggregate_maps_p3_main_model_to_full_model_label(tmp_path):
    main_run = tmp_path / "results" / "EXP01_main"
    ablation_run = tmp_path / "results" / "P3_without_dqn_gate"
    for run_dir, run_name, experiment_type in (
        (main_run, "EXP01_main", "main_model"),
        (ablation_run, "P3_without_dqn_gate", "ablation"),
    ):
        (run_dir / "metrics").mkdir(parents=True)
        (run_dir / "logs").mkdir(parents=True)
        (run_dir / "logs" / "run_manifest.json").write_text(
            json.dumps({"run_name": run_name, "experiment_type": experiment_type, "seed": 42}),
            encoding="utf-8",
        )
    pd.DataFrame(
        [{"model_name": "full_dqn_gated_multitask_cnn_ppo", "rankable_in_unified_table": True, "sharpe": 1.0}]
    ).to_csv(main_run / "metrics" / "main_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"variant_id": "full_model", "status": "completed", "sharpe": 0.9},
            {"variant_id": "without_dqn_gate", "status": "completed", "sharpe": 0.8},
        ]
    ).to_csv(ablation_run / "metrics" / "ablation_results.csv", index=False)
    dates = pd.date_range("2024-01-02", periods=25, freq="D").strftime("%Y-%m-%d").tolist()
    pd.DataFrame(
        {
            "date": dates,
            "model_name": ["full_dqn_gated_multitask_cnn_ppo"] * len(dates),
            "net_return": [0.006] * len(dates),
        }
    ).to_csv(main_run / "metrics" / "daily_returns.csv", index=False)
    pd.DataFrame(
        {
            "date": dates * 2,
            "model_name": ["full_dqn_gated_multitask_cnn_ppo"] * (len(dates) * 2),
            "variant_id": ["full_model"] * len(dates) + ["without_dqn_gate"] * len(dates),
            "net_return": [0.005] * len(dates) + [0.004] * len(dates),
        }
    ).to_csv(ablation_run / "metrics" / "daily_returns.csv", index=False)

    outputs = aggregate_paper_results(
        [main_run, ablation_run],
        tmp_path / "paper",
        benchmark_model="without_dqn_gate",
        paper_group_id="p3_components",
    )

    main = pd.read_csv(outputs["paper_main_comparison"])
    assert set(main["paper_model_id"]) == {"full_model", "without_dqn_gate"}
    assert main["paper_model_id"].tolist().count("full_model") == 1


def test_run_manifest_frozen_fields(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["config_hash"] = "config-hash"
    config["output"]["run_name"] = "manifest_test"
    config["reproducibility"]["seed"] = 7

    result = {
        "daily_returns": pd.DataFrame(
            {
                "next_valuation_date": ["2024-01-03"],
                "execution_price_type": ["open"],
                "fold_id": ["fixed"],
                "net_return": [0.01],
            }
        ),
        "daily_weights": pd.DataFrame(
            {
                "date": ["2024-01-03", "2024-01-03"],
                "asset_id": ["510300.SH", "159915.SZ"],
                "weight": [0.6, 0.4],
            }
        ),
    }

    artifacts = write_run_outputs(
        result,
        tmp_path,
        config=config,
        config_path="configs/default.yaml",
        command="python -m src.experiments.run_experiment",
    )

    manifest = json.loads((tmp_path / "logs" / "run_manifest.json").read_text(encoding="utf-8"))
    expected_fields = {
        "run_id",
        "timestamp",
        "command",
        "config_path",
        "run_name",
        "config_hash",
        "data_path",
        "data_hash",
        "split_id",
        "asset_universe_hash",
        "device",
        "python_executable",
        "python_version",
        "package_versions",
        "git_commit_if_available",
        "code_version",
        "created_at",
        "execution_model",
        "data_governance",
        "portfolio_initial_nav",
        "portfolio_initial_capital_currency",
        "portfolio_currency",
        "execution_price",
        "execution_price_type",
        "delayed_action_execution",
        "same_close_idealized_execution_enabled",
        "idealized_execution",
        "strict_no_lookahead_execution",
        "t_plus_one",
        "initial_build_cost",
        "amount_is_proxy",
        "metrics_factory_enabled",
        "turnover_rate_all_missing",
        "best_trial_number",
        "seed",
        "fold_id",
    }
    assert expected_fields.issubset(manifest.keys())
    assert manifest["execution_model"] == config["execution_model"]
    assert manifest["data_governance"] == config["data_governance"]
    assert manifest["portfolio_initial_nav"] == config["portfolio"]["initial_nav"]
    assert manifest["portfolio_initial_capital_currency"] == config["portfolio"]["initial_capital_currency"]
    assert manifest["portfolio_currency"] == "CNY"
    assert manifest["execution_price"] == "next_open"
    assert manifest["execution_price_type"] == "open"
    assert manifest["delayed_action_execution"] is False
    assert manifest["same_close_idealized_execution_enabled"] is False
    assert manifest["idealized_execution"] is False
    assert manifest["strict_no_lookahead_execution"] is True
    assert manifest["t_plus_one"] is False
    assert manifest["initial_build_cost"] is True
    assert manifest["amount_is_proxy"] is True
    assert manifest["metrics_factory_enabled"] is True
    assert manifest["turnover_rate_all_missing"] is False
    assert manifest["seed"] == 7
    assert manifest["fold_id"] == "fixed"

    assert artifacts["run_manifest"] == tmp_path / "logs" / "run_manifest.json"
    assert (tmp_path / "logs" / "asset_list.txt").read_text(encoding="utf-8").splitlines() == ["510300.SH", "159915.SZ"]
    assert json.loads((tmp_path / "logs" / "data_split.json").read_text(encoding="utf-8"))["mode"] == "fixed"
    assert yaml.safe_load((tmp_path / "logs" / "config_snapshot.yaml").read_text(encoding="utf-8"))["config_hash"] == "config-hash"
    assert json.loads((tmp_path / "logs" / "pca_report.json").read_text(encoding="utf-8"))["status"] == "not_applicable"
    assert pd.read_csv(tmp_path / "logs" / "hpo_trials.csv").loc[0, "state"] == "not_applicable"


def test_disabled_special_outputs_keep_frozen_schema(tmp_path):
    result = {
        "daily_returns": pd.DataFrame(
            {
                "next_valuation_date": ["2024-01-03"],
                "execution_price_type": ["open"],
                "fold_id": ["fixed"],
                "net_return": [0.01],
            }
        ),
        "daily_weights": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "asset_id": ["510300.SH"],
                "weight": [1.0],
            }
        ),
    }

    artifacts = write_run_outputs(result, tmp_path, config=deepcopy(DEFAULT_CONFIG))

    assert yaml.safe_load((tmp_path / "logs" / "selected_input_matrix.yaml").read_text(encoding="utf-8"))["status"] == "not_applicable"
    assert json.loads((tmp_path / "logs" / "input_matrix_feature_groups.json").read_text(encoding="utf-8"))["status"] == "not_applicable"
    assert json.loads((tmp_path / "logs" / "pca_report.json").read_text(encoding="utf-8"))["status"] == "not_applicable"
    assert json.loads((tmp_path / "logs" / "fixed_ratio_weights.json").read_text(encoding="utf-8"))["status"] == "not_applicable"

    _assert_csv_schema(tmp_path / "logs" / "feature_provenance_report.csv", FEATURE_PROVENANCE_COLUMNS)
    _assert_csv_schema(tmp_path / "logs" / "feature_group_summary.csv", FEATURE_GROUP_SUMMARY_COLUMNS)
    _assert_csv_schema(tmp_path / "logs" / "feature_selection_report.csv", FEATURE_SELECTION_COLUMNS, "status")
    _assert_csv_schema(tmp_path / "logs" / "metrics_factory_audit_sample.csv", METRICS_FACTORY_AUDIT_COLUMNS, "status")
    _assert_csv_schema(tmp_path / "logs" / "cost_calibration_report.csv", COST_CALIBRATION_COLUMNS, "status")
    _assert_csv_schema(tmp_path / "logs" / "hpo_trials.csv", HPO_TRIAL_COLUMNS, "state")
    _assert_csv_schema(tmp_path / "logs" / "seed_aggregate_summary.csv", SEED_AGGREGATE_COLUMNS)
    _assert_csv_schema(tmp_path / "metrics" / "statistics_summary.csv", STATISTICS_SUMMARY_COLUMNS, "status")
    assert artifacts["statistics_summary"] == tmp_path / "metrics" / "statistics_summary.csv"


def test_output_paths_are_under_run_dir(tmp_path):
    run_dir = tmp_path / "results" / "safe_run"
    result = {
        "daily_returns": pd.DataFrame(
            {
                "next_valuation_date": ["2024-01-03"],
                "execution_price_type": ["open"],
                "fold_id": ["fixed"],
                "net_return": [0.01],
            }
        ),
        "daily_weights": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "asset_id": ["510300.SH"],
                "weight": [1.0],
            }
        ),
    }

    artifacts = write_run_outputs(result, run_dir)

    for artifact_path in artifacts.values():
        assert artifact_path.resolve().is_relative_to(run_dir.resolve())
        assert not artifact_path.resolve().is_relative_to((PROJECT_ROOT / "data" / "processed").resolve())
        assert not artifact_path.resolve().is_relative_to((PROJECT_ROOT / "data" / "metrics_factory").resolve())
        assert not artifact_path.resolve().is_relative_to((PROJECT_ROOT / "data" / "reports").resolve())

    with pytest.raises(ValueError, match="ERR_SECURITY_PATH_DENIED"):
        write_run_outputs(result, run_dir, registry_path=tmp_path / "run_registry.sqlite")

    blocked_run_dir = PROJECT_ROOT / "data" / "processed" / f"blocked_{tmp_path.name}"
    with pytest.raises(ValueError, match="ERR_SECURITY_PATH_DENIED"):
        write_run_outputs(result, blocked_run_dir)
    assert not blocked_run_dir.exists()


def _assert_csv_schema(path, columns, status_column=None):
    frame = pd.read_csv(path)
    assert list(frame.columns) == list(columns)
    assert len(frame) == 1
    if status_column is not None:
        assert frame.loc[0, status_column] == "not_applicable"
