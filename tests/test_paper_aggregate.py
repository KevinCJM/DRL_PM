import json

import pandas as pd

from src.experiments.paper_aggregate import aggregate_paper_results


HYBRID_CHILDREN = (
    "hybrid_dqn_optimizer_equal_weight",
    "hybrid_dqn_optimizer_markowitz_mean_variance",
    "hybrid_dqn_optimizer_minimum_variance",
    "hybrid_dqn_optimizer_sharpe_maximization",
    "hybrid_dqn_optimizer_risk_parity",
)


def _run_dir(tmp_path, name, rows, daily_rows=None):
    run_dir = tmp_path / "results" / name
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(
        json.dumps({"run_name": name, "experiment_type": "baseline_comparison", "seed": 42}),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(metrics_dir / "baseline_comparison.csv", index=False)
    if daily_rows is not None:
        pd.DataFrame(daily_rows).to_csv(metrics_dir / "daily_returns.csv", index=False)
    return run_dir


def test_paper_aggregate_filters_hybrid_alias_and_diagnostics(tmp_path):
    dates = pd.date_range("2024-01-02", periods=25, freq="D").strftime("%Y-%m-%d")
    run_dir = _run_dir(
        tmp_path,
        "formal_related_work",
        [
            {
                "model_name": "ppo_dqn_hierarchical_reimplementation",
                "paper_model_id": "ppo_dqn_hierarchical_reimplementation",
                "rankable_in_unified_table": True,
                "baseline_family": "native_rl_reimplementation",
                "training_algorithm": "ppo_dqn_hierarchical_reimplementation",
                "algorithm_fidelity": "platform_adapted",
                "sharpe": 1.0,
            },
            {
                "model_name": "hybrid_dqn_optimizer_reimplementation",
                "paper_model_id": "hybrid_dqn_optimizer_reimplementation",
                "rankable_in_unified_table": True,
                "baseline_family": "native_rl_reimplementation",
                "training_algorithm": "factorized_dqn_signal_plus_portfolio_optimizer",
                "algorithm_fidelity": "platform_adapted",
                "sharpe": 99.0,
            },
            {
                "model_name": "shared_dqn_diagnostic",
                "paper_model_id": "hybrid_dqn_optimizer_risk_parity",
                "rankable_in_unified_table": True,
                "diagnostic_status": "diagnostic_shared_dqn",
                "reason": "shared_dqn_multi_optimizer_final_test",
                "sharpe": 2.0,
            },
            *[
                {
                    "model_name": child_id,
                    "paper_model_id": child_id,
                    "rankable_in_unified_table": True,
                    "baseline_family": "native_rl_reimplementation",
                    "training_algorithm": "factorized_dqn_signal_plus_portfolio_optimizer",
                    "algorithm_fidelity": "platform_adapted",
                    "sharpe": 0.8,
                }
                for child_id in HYBRID_CHILDREN
            ],
        ],
        [
            {"date": date, "model_name": "ppo_dqn_hierarchical_reimplementation", "net_return": 0.01}
            for date in dates
        ]
        + [{"date": date, "model_name": HYBRID_CHILDREN[0], "net_return": 0.005} for date in dates],
    )

    outputs = aggregate_paper_results([run_dir], tmp_path / "paper", benchmark_model=HYBRID_CHILDREN[0])

    main = pd.read_csv(outputs["paper_main_comparison"])
    diagnostic = pd.read_csv(outputs["paper_diagnostic_comparison"])
    assert "hybrid_dqn_optimizer_reimplementation" not in set(main["paper_model_id"])
    assert {"ppo_dqn_hierarchical_reimplementation", *HYBRID_CHILDREN}.issubset(set(main["paper_model_id"]))
    assert {"hybrid_dqn_optimizer_reimplementation", "hybrid_dqn_optimizer_risk_parity"}.issubset(
        set(diagnostic["paper_model_id"])
    )
    assert diagnostic["rankable_in_unified_table"].map(lambda value: str(value).lower() == "false").all()
    assert diagnostic["reason"].astype(str).str.strip().ne("").all()
    reasons = dict(zip(diagnostic["paper_model_id"], diagnostic["reason"], strict=False))
    assert reasons["hybrid_dqn_optimizer_reimplementation"] == "hybrid_dqn_optimizer_alias"
    assert reasons["hybrid_dqn_optimizer_risk_parity"] == "shared_dqn_multi_optimizer_final_test"


def test_paper_aggregate_formal_filter_records_diagnostics(tmp_path):
    formal_run = _run_dir(
        tmp_path,
        "formal_run",
        [{"model_name": "main", "rankable_in_unified_table": True, "sharpe": 1.0, "reason": None}],
        [{"date": "2024-01-02", "model_name": "main", "net_return": 0.01}],
    )
    diagnostic_run = _run_dir(
        tmp_path,
        "diagnostic_run",
        [{"model_name": "legacy", "rankable_in_unified_table": True, "sharpe": 2.0, "reason": None}],
        [{"date": "2024-01-02", "model_name": "legacy", "net_return": 0.02}],
    )
    manifest = {
        "protocol_id": "core13_v2_full_reset_20260522",
        "data_cutoff_date": "2026-05-20",
        "data_mode": "availability_mask",
        "return_source": "adj_nav",
        "valuation_source": "adj_nav",
        "reward_return_source": "adj_nav",
        "metrics_return_source": "adj_nav",
        "execution_price_source": "ohlcv",
        "valuation_execution_split": True,
        "reward_valuation_split": True,
        "rankable_in_unified_table": True,
        "diagnostic_status": "formal",
        "availability_mask_contract_passed": True,
        "unavailable_asset_weight_abs_max": 0.0,
        "frozen_or_imputed_valuation_count": 0,
        "daily_returns_finite": True,
        "daily_nav_finite": True,
        "run_name": "formal_run",
        "experiment_type": "baseline_comparison",
        "seed": 42,
    }
    (formal_run / "logs" / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    diagnostic_manifest = dict(manifest)
    diagnostic_manifest.update({"run_name": "diagnostic_run", "protocol_id": "legacy17"})
    (diagnostic_run / "logs" / "run_manifest.json").write_text(json.dumps(diagnostic_manifest), encoding="utf-8")

    outputs = aggregate_paper_results(
        [formal_run, diagnostic_run],
        tmp_path / "paper",
        required_protocol_id="core13_v2_full_reset_20260522",
        required_data_cutoff_date="2026-05-20",
        require_formal_manifest=True,
        require_availability_mask_contract=True,
    )

    main = pd.read_csv(outputs["paper_main_comparison"])
    diagnostic = pd.read_csv(outputs["paper_diagnostic_comparison"])
    status = json.loads(outputs["diagnostic_status"].read_text(encoding="utf-8"))
    manifest = json.loads(outputs["paper_aggregate_manifest"].read_text(encoding="utf-8"))
    assert set(main["paper_model_id"]) == {"main"}
    assert "legacy" in set(diagnostic["paper_model_id"])
    assert "protocol_mismatch" in set(diagnostic["reason"])
    assert status["status"] == "completed_with_diagnostics"
    assert manifest["formal_filter"]["require_formal_manifest"] is True
    assert outputs["source_run_dirs"].read_text(encoding="utf-8").count("\n") == 2


def test_paper_aggregate_excludes_explicit_paper_models(tmp_path):
    run_dir = _run_dir(
        tmp_path,
        "formal_run",
        [
            {"model_name": "promoted_model", "rankable_in_unified_table": True, "sharpe": 1.0},
            {"model_name": "unpromoted_model", "rankable_in_unified_table": True, "sharpe": 9.0},
        ],
        [
            {"date": "2024-01-02", "model_name": "promoted_model", "net_return": 0.01},
            {"date": "2024-01-03", "model_name": "promoted_model", "net_return": 0.02},
            {"date": "2024-01-02", "model_name": "unpromoted_model", "net_return": 0.03},
            {"date": "2024-01-03", "model_name": "unpromoted_model", "net_return": 0.04},
        ],
    )

    outputs = aggregate_paper_results(
        [run_dir],
        tmp_path / "paper",
        benchmark_model="promoted_model",
        exclude_models=["unpromoted_model"],
    )

    main = pd.read_csv(outputs["paper_main_comparison"])
    seed_summary = pd.read_csv(outputs["paper_seed_summary"])
    manifest = json.loads(outputs["paper_aggregate_manifest"].read_text(encoding="utf-8"))
    assert set(main["paper_model_id"]) == {"promoted_model"}
    assert set(seed_summary["paper_model_id"]) == {"promoted_model"}
    assert manifest["exclude_models"] == ["unpromoted_model"]


def test_paper_aggregate_demotes_active_hpo_final_low_activity(tmp_path):
    run_dir = tmp_path / "results" / "formal_hpo_low_activity"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    metrics_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_name": "formal_hpo_low_activity",
                "experiment_type": "hyperparameter_sweep",
                "seed": 42,
                "execution_activity": {
                    "protocol": "daily_gate_with_cost_constraint",
                    "activity_gate_enforced": True,
                    "min_model_rebalance_hit_rate": 0.05,
                    "max_model_rebalance_hit_rate": 0.6,
                    "min_non_initial_turnover_per_opportunity": 0.002,
                    "max_average_turnover": 0.03,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model_name": "full_dqn_gated_multitask_cnn_ppo",
                "hpo_model_name": "full_dqn_gated_multitask_cnn_ppo",
                "baseline_family": "new_model_extension",
                "seed": 42,
                "rankable_in_unified_table": True,
                "model_rebalance_hit_rate": 0.0,
                "non_initial_turnover_per_opportunity": 0.0,
                "average_turnover": 0.0,
                "sharpe": 2.0,
            },
            {
                "model_name": "full_dqn_gated_multitask_cnn_ppo",
                "hpo_model_name": "full_dqn_gated_multitask_cnn_ppo",
                "baseline_family": "new_model_extension",
                "seed": 123,
                "rankable_in_unified_table": True,
                "model_rebalance_hit_rate": 0.08,
                "non_initial_turnover_per_opportunity": 0.004,
                "average_turnover": 0.01,
                "sharpe": 1.7,
            },
            {
                "model_name": "equal_weight",
                "baseline_family": "traditional",
                "seed": 42,
                "rankable_in_unified_table": True,
                "sharpe": 0.5,
            },
        ]
    ).to_csv(metrics_dir / "hpo_model_final_comparison.csv", index=False)

    outputs = aggregate_paper_results([run_dir], tmp_path / "paper", benchmark_model="equal_weight")

    main = pd.read_csv(outputs["paper_main_comparison"])
    diagnostic = pd.read_csv(outputs["paper_diagnostic_comparison"])
    assert set(main["paper_model_id"]) == {"equal_weight"}
    target_diag = diagnostic.loc[diagnostic["paper_model_id"].eq("full_dqn_gated_multitask_cnn_ppo")]
    assert target_diag.shape[0] == 2
    assert set(target_diag["reason"]) == {"failed_low_trade_activity", "failed_low_trade_activity_group"}


def test_closest_hybrid_figure_source_uses_only_rankable_platform_rows(tmp_path):
    trainable_ids = ("ppo_dqn_hierarchical_reimplementation", *HYBRID_CHILDREN)
    rows = [
        {
            "model_name": paper_model_id,
            "paper_model_id": paper_model_id,
            "baseline_family": "native_rl_reimplementation",
            "training_algorithm": (
                "ppo_dqn_hierarchical_reimplementation"
                if paper_model_id == "ppo_dqn_hierarchical_reimplementation"
                else "factorized_dqn_signal_plus_portfolio_optimizer"
            ),
            "algorithm_fidelity": "platform_adapted",
            "rankable_in_unified_table": True,
            "sharpe": 1.0,
        }
        for paper_model_id in trainable_ids
    ]
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
    run_dir = _run_dir(tmp_path, "formal_hpo", rows)

    outputs = aggregate_paper_results([run_dir], tmp_path / "paper", benchmark_model=trainable_ids[0])

    figure_source = pd.read_csv(outputs["closest_hybrid_figure_source"])
    assert set(figure_source["paper_model_id"]) == set(trainable_ids)
    assert set(figure_source["baseline_family"]) == {"native_rl_reimplementation"}
    assert set(figure_source["algorithm_fidelity"]) == {"platform_adapted"}
    assert "hybrid_dqn_optimizer_reimplementation" not in set(figure_source["paper_model_id"])
    assert "original_report_metric" not in set(figure_source["paper_model_id"])


def test_paired_statistics_skips_misaligned_dates_with_reason(tmp_path):
    benchmark_id = "ppo_dqn_hierarchical_reimplementation"
    model_id = HYBRID_CHILDREN[0]
    benchmark_dates = pd.date_range("2024-01-02", periods=25, freq="D").strftime("%Y-%m-%d")
    model_dates = pd.date_range("2024-03-01", periods=25, freq="D").strftime("%Y-%m-%d")
    run_dir = _run_dir(
        tmp_path,
        "formal_misaligned",
        [
            {
                "model_name": benchmark_id,
                "paper_model_id": benchmark_id,
                "baseline_family": "native_rl_reimplementation",
                "training_algorithm": "ppo_dqn_hierarchical_reimplementation",
                "algorithm_fidelity": "platform_adapted",
                "rankable_in_unified_table": True,
                "sharpe": 1.0,
            },
            {
                "model_name": model_id,
                "paper_model_id": model_id,
                "baseline_family": "native_rl_reimplementation",
                "training_algorithm": "factorized_dqn_signal_plus_portfolio_optimizer",
                "algorithm_fidelity": "platform_adapted",
                "rankable_in_unified_table": True,
                "sharpe": 0.8,
            },
        ],
        [{"date": date, "model_name": benchmark_id, "net_return": 0.01} for date in benchmark_dates]
        + [{"date": date, "model_name": model_id, "net_return": 0.005} for date in model_dates],
    )

    outputs = aggregate_paper_results([run_dir], tmp_path / "paper", benchmark_model=benchmark_id)

    paired = pd.read_csv(outputs["paper_paired_statistics"])
    model_stats = paired.loc[paired["model_name"].eq(model_id)]
    assert not model_stats.empty
    assert "pass" not in set(model_stats["status"])
    skipped = model_stats.loc[model_stats["status"].eq("skipped")]
    assert not skipped.empty
    assert set(skipped["skip_reason"]) == {"insufficient_samples"}
    assert skipped["n_obs"].eq(0).all()
