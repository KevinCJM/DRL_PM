import json
from pathlib import Path

import pandas as pd

import src.experiments.formal_readiness as formal_readiness


def test_formal_readiness_date_normalization():
    assert formal_readiness._normalize_date_token("20260520") == "2026-05-20"
    assert formal_readiness._normalize_date_token("2026-05-20") == "2026-05-20"


def test_formal_readiness_writes_no_go_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(formal_readiness, "PROJECT_ROOT", tmp_path)
    reset = tmp_path / "results/protocol_reset/core13_v2_full_reset_20260522"
    reset.mkdir(parents=True)
    (reset / "protocol_reset_manifest.json").write_text(
        json.dumps(
            {
                "new_protocol_id": "core13_v2_full_reset_20260522",
                "discard_previous_results": True,
                "forbid_checkpoint_reuse": True,
                "forbid_hpo_reuse": True,
            }
        ),
        encoding="utf-8",
    )

    outputs = formal_readiness.audit_formal_readiness(root=tmp_path, output_dir="results/full")

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    rows = pd.read_csv(outputs["csv"])
    assert payload["status"] == "no_go"
    assert rows.loc[rows["requirement"].eq("protocol_reset_manifest"), "status"].iloc[0] == "pass"
    assert rows["status"].eq("fail").any()


def test_formal_readiness_accepts_required_diagnostic_groups(tmp_path):
    for group in formal_readiness.FORMAL_PAPER_TABLE_GROUPS:
        group_dir = tmp_path / "results/paper_tables" / group
        group_dir.mkdir(parents=True)
        (group_dir / "paper_aggregate_manifest.json").write_text(
            json.dumps(
                {
                    "formal_filter": {
                        "required_protocol_id": formal_readiness.PROTOCOL_ID,
                        "required_data_cutoff_date": formal_readiness.DATA_CUTOFF_DATE,
                        "require_formal_manifest": True,
                        "require_availability_mask_contract": True,
                    }
                }
            ),
            encoding="utf-8",
        )
        if group == "main_hpo_5seed":
            main_rows = [
                {
                    "paper_model_id": "full_dqn_gated_multitask_cnn_ppo",
                    "diagnostic_status": "formal",
                    "rankable_in_unified_table": True,
                    "source_run": f"{formal_readiness.P1_MAIN_HPO_RUN_PREFIX}_s{seed}",
                    "source_file": "baseline_comparison.csv",
                }
                for seed in formal_readiness.SEEDS
            ]
        elif group == "main_hpo_plus_p9":
            main_rows = [
                {
                    "paper_model_id": "full_dqn_gated_multitask_cnn_ppo",
                    "diagnostic_status": "formal",
                    "rankable_in_unified_table": True,
                    "source_run": f"{formal_readiness.P1_MAIN_HPO_RUN_PREFIX}_s{seed}",
                    "source_file": "baseline_comparison.csv",
                }
                for seed in formal_readiness.SEEDS
            ]
            main_rows.extend(
                {
                    "paper_model_id": "ppo_dqn_hierarchical_reimplementation",
                    "diagnostic_status": "formal",
                    "rankable_in_unified_table": True,
                    "source_run": f"EXP09_P9_formal_hpo_related_work_s{seed}",
                    "source_file": "hpo_model_final_comparison.csv",
                }
                for seed in formal_readiness.SEEDS
            )
        else:
            main_rows = [
                {
                    "paper_model_id": "full_dqn_gated_multitask_cnn_ppo",
                    "diagnostic_status": "formal",
                    "rankable_in_unified_table": True,
                }
            ]
        pd.DataFrame(main_rows).to_csv(group_dir / "paper_main_comparison.csv", index=False)
        pd.DataFrame([{"metric_name": "sharpe", "mean": 1.0}]).to_csv(group_dir / "paper_seed_summary.csv", index=False)
        pd.DataFrame([{"model_name": "x", "benchmark_name": "y", "n_obs": 25}]).to_csv(
            group_dir / "paper_paired_statistics.csv", index=False
        )

    for group in formal_readiness.DIAGNOSTIC_PAPER_TABLE_GROUPS:
        group_dir = tmp_path / "results/paper_tables" / group
        group_dir.mkdir(parents=True)
        (group_dir / "paper_aggregate_manifest.json").write_text('{"status": "diagnostic"}', encoding="utf-8")
        (group_dir / "diagnostic_status.json").write_text(
            json.dumps(
                {
                    "status": "diagnostic_complete",
                    "source_run_dir_count": 1,
                    "source_run_manifest_count": 1,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame([{"paper_model_id": "diagnostic_variant"}]).to_csv(
            group_dir / "paper_main_comparison.csv", index=False
        )
        pd.DataFrame([{"metric_name": "sharpe", "mean": 1.0}]).to_csv(group_dir / "paper_seed_summary.csv", index=False)
        pd.DataFrame([{"reason": "not_applicable"}]).to_csv(group_dir / "paper_paired_statistics.csv", index=False)

    rows = formal_readiness._audit_paper_tables(
        tmp_path,
        formal_readiness.PROTOCOL_ID,
        formal_readiness.DATA_CUTOFF_DATE,
    )

    assert all(row["status"] == "pass" for row in rows)
    assert any(row["requirement"] == "diagnostic_paper_table_group:p2_input_pca" for row in rows)


def test_formal_readiness_rejects_raw_exp05_as_main_hpo_5seed_source(tmp_path):
    group_dir = tmp_path / "results/paper_tables/main_hpo_5seed"
    group_dir.mkdir(parents=True)
    (group_dir / "paper_aggregate_manifest.json").write_text(
        json.dumps(
            {
                "formal_filter": {
                    "required_protocol_id": formal_readiness.PROTOCOL_ID,
                    "required_data_cutoff_date": formal_readiness.DATA_CUTOFF_DATE,
                    "require_formal_manifest": True,
                    "require_availability_mask_contract": True,
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "paper_model_id": "full_dqn_gated_multitask_cnn_ppo",
                "diagnostic_status": "formal",
                "rankable_in_unified_table": True,
                "source_run": f"EXP05_P7_formal_hpo_main_native_s{seed}",
                "source_file": "hpo_model_final_comparison.csv",
            }
            for seed in formal_readiness.SEEDS
        ]
    ).to_csv(group_dir / "paper_main_comparison.csv", index=False)
    pd.DataFrame([{"metric_name": "sharpe", "mean": 1.0}]).to_csv(group_dir / "paper_seed_summary.csv", index=False)
    pd.DataFrame([{"model_name": "x", "benchmark_name": "y", "n_obs": 25}]).to_csv(
        group_dir / "paper_paired_statistics.csv", index=False
    )

    rows = formal_readiness._audit_paper_tables(
        tmp_path,
        formal_readiness.PROTOCOL_ID,
        formal_readiness.DATA_CUTOFF_DATE,
    )

    target = next(
        row for row in rows if row["requirement"] == "formal_paper_table_group:main_hpo_5seed_requires_p1_from_hpo_sources"
    )
    assert target["status"] == "fail"


def test_formal_readiness_rejects_main_hpo_plus_p9_without_p1_sources(tmp_path):
    group_dir = tmp_path / "results/paper_tables/main_hpo_plus_p9"
    group_dir.mkdir(parents=True)
    (group_dir / "paper_aggregate_manifest.json").write_text(
        json.dumps(
            {
                "formal_filter": {
                    "required_protocol_id": formal_readiness.PROTOCOL_ID,
                    "required_data_cutoff_date": formal_readiness.DATA_CUTOFF_DATE,
                    "require_formal_manifest": True,
                    "require_availability_mask_contract": True,
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "paper_model_id": "full_dqn_gated_multitask_cnn_ppo",
                "diagnostic_status": "formal",
                "rankable_in_unified_table": True,
                "source_run": f"EXP05_P7_formal_hpo_main_native_s{seed}",
                "source_file": "hpo_model_final_comparison.csv",
            }
            for seed in formal_readiness.SEEDS
        ]
        + [
            {
                "paper_model_id": "ppo_dqn_hierarchical_reimplementation",
                "diagnostic_status": "formal",
                "rankable_in_unified_table": True,
                "source_run": f"EXP09_P9_formal_hpo_related_work_s{seed}",
                "source_file": "hpo_model_final_comparison.csv",
            }
            for seed in formal_readiness.SEEDS
        ]
    ).to_csv(group_dir / "paper_main_comparison.csv", index=False)
    pd.DataFrame([{"metric_name": "sharpe", "mean": 1.0}]).to_csv(group_dir / "paper_seed_summary.csv", index=False)
    pd.DataFrame([{"model_name": "x", "benchmark_name": "y", "n_obs": 25}]).to_csv(
        group_dir / "paper_paired_statistics.csv", index=False
    )

    rows = formal_readiness._audit_paper_tables(
        tmp_path,
        formal_readiness.PROTOCOL_ID,
        formal_readiness.DATA_CUTOFF_DATE,
    )

    target = next(
        row for row in rows if row["requirement"] == "formal_paper_table_group:main_hpo_plus_p9_requires_p1_and_p9_sources"
    )
    assert target["status"] == "fail"


def test_formal_readiness_audits_p16_hpo_config(tmp_path):
    config_dir = tmp_path / "configs/paper"
    config_dir.mkdir(parents=True)
    (config_dir / "p16_ra_gt_rcpo_formal_seed_runner.yaml").write_text(
        f"""
experiment:
  type: hyperparameter_sweep
long_running: false
data_governance:
  return_source: adj_nav
  execution_price_source: ohlcv
feature_matrix:
  input_matrix_id: M6
feature_reduction:
  pca:
    enabled: true
    explained_variance: 0.95
baselines:
  native_rl:
    epochs: 8
    max_train_steps: 2048
    max_validation_steps: 512
    max_gradient_updates_per_epoch: 64
rebalance:
  mode: daily
execution_activity:
  protocol: daily_gate_with_cost_constraint
  scheduler_blocks_model_actions: false
  activity_gate_enforced: true
  turnover_optimization_protocol_id: turnover_active_v1
hpo:
  enabled: true
  equal_budget_across_models: true
  n_trials_per_model: 50
  metric: validation_return_risk_cost_constrained
  objective: validation_return_risk_cost_constrained
  trainable_models:
  - risk_aware_graph_transformer_constrained_actor_critic
  activity_constraints:
    enabled: true
    scope_activity_protocols:
    - daily_gate_with_cost_constraint
    min_model_rebalance_hit_rate: 0.05
    min_non_initial_turnover_per_opportunity: 0.002
ra_gt_rcpo:
  enabled: true
  rho_policy: straight_through_gumbel_softmax_v1
training:
  epochs: 8
  max_train_steps: 2048
  max_validation_steps: 512
  checkpoint_include_replay_buffer: false
protocol:
  protocol_id: core13_v2_full_reset_20260522
  data_cutoff_date: '2026-05-20'
reproducibility:
  seeds: [42]
security:
  path_whitelist:
  - /Users/chenjunming/Desktop/DRL_PM
  - {tmp_path}
""",
        encoding="utf-8",
    )

    rows = formal_readiness._audit_hpo_config(config_dir / "p16_ra_gt_rcpo_formal_seed_runner.yaml", "p16")

    assert all(row["status"] == "pass" for row in rows)
    assert any(row["requirement"] == "hpo_equal_budget_50:p16_ra_gt_rcpo_formal_seed_runner.yaml" for row in rows)
    assert any(row["requirement"] == "active_execution_activity:p16_ra_gt_rcpo_formal_seed_runner.yaml" for row in rows)
    assert any(row["requirement"] == "p16_learned_rho_policy:p16_ra_gt_rcpo_formal_seed_runner.yaml" for row in rows)
    assert any(row["requirement"] == "p16_effective_training_budget:p16_ra_gt_rcpo_formal_seed_runner.yaml" for row in rows)


def test_formal_readiness_fails_p16_insufficient_effective_budget(tmp_path):
    config_dir = tmp_path / "configs/paper"
    config_dir.mkdir(parents=True)
    (config_dir / "p16_ra_gt_rcpo_formal_seed_runner.yaml").write_text(
        f"""
experiment:
  type: hyperparameter_sweep
long_running: false
data_governance:
  return_source: adj_nav
  execution_price_source: ohlcv
feature_matrix:
  input_matrix_id: M6
feature_reduction:
  pca:
    enabled: true
    explained_variance: 0.95
baselines:
  native_rl:
    epochs: 2
    max_train_steps: 128
    max_validation_steps: 128
    max_gradient_updates_per_epoch: 16
rebalance:
  mode: daily
execution_activity:
  protocol: daily_gate_with_cost_constraint
  scheduler_blocks_model_actions: false
  activity_gate_enforced: true
  turnover_optimization_protocol_id: turnover_active_v1
hpo:
  enabled: true
  equal_budget_across_models: true
  n_trials_per_model: 50
  metric: validation_return_risk_cost_constrained
  objective: validation_return_risk_cost_constrained
  trainable_models:
  - risk_aware_graph_transformer_constrained_actor_critic
  activity_constraints:
    enabled: true
    scope_activity_protocols:
    - daily_gate_with_cost_constraint
    min_model_rebalance_hit_rate: 0.05
    min_non_initial_turnover_per_opportunity: 0.002
ra_gt_rcpo:
  enabled: true
  rho_policy: straight_through_gumbel_softmax_v1
training:
  epochs: 2
  max_train_steps: 128
  max_validation_steps: 128
  checkpoint_include_replay_buffer: false
protocol:
  protocol_id: core13_v2_full_reset_20260522
  data_cutoff_date: '2026-05-20'
reproducibility:
  seeds: [42]
security:
  path_whitelist:
  - /Users/chenjunming/Desktop/DRL_PM
  - {tmp_path}
""",
        encoding="utf-8",
    )

    rows = formal_readiness._audit_hpo_config(config_dir / "p16_ra_gt_rcpo_formal_seed_runner.yaml", "p16")
    budget_rows = [
        row
        for row in rows
        if row["requirement"] == "p16_effective_training_budget:p16_ra_gt_rcpo_formal_seed_runner.yaml"
    ]

    assert len(budget_rows) == 1
    assert budget_rows[0]["status"] == "fail"
    assert '"estimated_gradient_updates": 8' in budget_rows[0]["detail"]


def test_formal_readiness_accepts_p13_declared_skipped(tmp_path):
    reference_dir = tmp_path / "results/paper_tables/p12_p13_validation_references"
    promotion_dir = tmp_path / "results/paper_tables/p12_p13_promotion_gate"
    reference_dir.mkdir(parents=True)
    promotion_dir.mkdir(parents=True)
    (reference_dir / "validation_reference_manifest.json").write_text(
        json.dumps({"selection_split": "validation", "test_used_for_model_selection": False}),
        encoding="utf-8",
    )
    (promotion_dir / "promotion_gate_manifest.json").write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "test_used_for_model_selection": False,
                "p13_evaluated": False,
            }
        ),
        encoding="utf-8",
    )
    required_models = [
        "eiie_native",
        "full_dqn_gated_multitask_cnn_ppo",
        "ppo_dqn_hierarchical_reimplementation",
        "cnn_ppo_native",
        "pgportfolio_eiie_native",
    ]
    reference_rows = [{"model_name": model} for model in required_models]
    pd.DataFrame(reference_rows).to_csv(reference_dir / "validation_reference_comparison.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(reference_dir / "validation_reference_daily_returns.csv", index=False)
    pd.DataFrame([{"model_name": "cage_eiie_frozen_gate"}]).to_csv(
        reference_dir / "validation_selection_report.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "phase": "P12",
                "model_name": "cage_eiie_frozen_gate",
                "promotion_gate_passed": True,
            }
        ]
    ).to_csv(promotion_dir / "promotion_gate_report.csv", index=False)

    rows = formal_readiness._audit_p12_p13_validation_gate(tmp_path)

    assert all(row["status"] == "pass" for row in rows)
    assert any(row["requirement"] == "p13_promotion_gate_decided_or_skipped" for row in rows)


def test_formal_readiness_uses_config_snapshot_when_manifest_missing_active_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(formal_readiness, "FORMAL_MAIN_RUN_PREFIXES", ("EXP05_P7_formal_hpo_main_native",))
    monkeypatch.setattr(formal_readiness, "SEEDS", (42,))
    run_dir = tmp_path / "results/EXP05_P7_formal_hpo_main_native_s42"
    logs_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    logs_dir.mkdir(parents=True)
    metrics_dir.mkdir(parents=True)
    (logs_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol_id": formal_readiness.PROTOCOL_ID,
                "diagnostic_status": "formal",
                "rankable_in_unified_table": True,
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / "config_snapshot.yaml").write_text(
        """
protocol:
  protocol_id: core13_v2_full_reset_20260522
rankability:
  diagnostic_status: formal
  rankable_in_unified_table: true
rebalance:
  mode: daily
execution_activity:
  protocol: daily_gate_with_cost_constraint
  turnover_optimization_protocol_id: turnover_active_v1
  scheduler_blocks_model_actions: false
  activity_gate_enforced: true
hpo:
  metric: validation_return_risk_cost_constrained
  objective: validation_return_risk_cost_constrained
  activity_constraints:
    enabled: true
""",
        encoding="utf-8",
    )
    pd.DataFrame([{"model_name": "full_dqn_gated_multitask_cnn_ppo"}]).to_csv(
        metrics_dir / "hpo_model_final_comparison.csv",
        index=False,
    )
    pd.DataFrame([{"date": "2026-05-20", "daily_return": 0.01}]).to_csv(
        metrics_dir / "hpo_model_final_daily_returns.csv",
        index=False,
    )
    pd.DataFrame([{"model_name": "full_dqn_gated_multitask_cnn_ppo", "param_name": "ppo_lr"}]).to_csv(
        logs_dir / "hpo_search_space_manifest.csv",
        index=False,
    )

    rows = formal_readiness._audit_formal_seed_runs(tmp_path, formal_readiness.PROTOCOL_ID)

    target = next(
        row
        for row in rows
        if row["requirement"] == "formal_seed_run:EXP05_P7_formal_hpo_main_native_s42"
    )
    assert target["status"] == "pass"


def test_formal_readiness_audits_p16_final_table_contract(tmp_path):
    final_dir = tmp_path / "results/paper_tables/p16_ra_gt_rcpo_final"
    final_dir.mkdir(parents=True)
    (final_dir / "paper_aggregate_manifest.json").write_text(
        json.dumps(
            {
                "formal_filter": {
                    "required_protocol_id": formal_readiness.PROTOCOL_ID,
                    "required_data_cutoff_date": formal_readiness.DATA_CUTOFF_DATE,
                    "require_formal_manifest": True,
                    "require_availability_mask_contract": True,
                }
            }
        ),
        encoding="utf-8",
    )
    main_rows = [
        {
            "paper_model_id": formal_readiness.P16_PRIMARY_MODEL_ID,
            "source_run": f"EXP35_P16_formal_ra_gt_rcpo_s{seed}",
            "source_file": "hpo_model_final_comparison.csv",
            "seed": seed,
            "baseline_family": "new_model_extension",
            "diagnostic_status": "formal",
            "rankable_in_unified_table": True,
        }
        for seed in formal_readiness.SEEDS
    ]
    main_rows.extend(
        {
            "paper_model_id": model,
            "source_run": "EXP36_P1_fixed_deterministic_formal_export",
            "source_file": "baseline_comparison.csv",
            "seed": 42,
            "baseline_family": "traditional",
            "deterministic_baseline": True,
            "n_independent_seeds": 1,
            "diagnostic_status": "formal",
            "rankable_in_unified_table": True,
        }
        for model in formal_readiness.P16_DETERMINISTIC_BASELINES
    )
    pd.DataFrame(main_rows).to_csv(final_dir / "paper_main_comparison.csv", index=False)
    pd.DataFrame([{"paper_model_id": formal_readiness.P16_PRIMARY_MODEL_ID, "metric_name": "cumulative_return"}]).to_csv(
        final_dir / "paper_seed_summary.csv",
        index=False,
    )
    pd.DataFrame([{"model_name": formal_readiness.P16_PRIMARY_MODEL_ID, "benchmark_name": "risk_parity"}]).to_csv(
        final_dir / "paper_paired_statistics.csv",
        index=False,
    )

    rows = formal_readiness._audit_p16_final_table(
        final_dir,
        formal_readiness.PROTOCOL_ID,
        formal_readiness.DATA_CUTOFF_DATE,
        promoted=True,
    )

    assert all(row["status"] == "pass" for row in rows)

    broken = pd.DataFrame(main_rows)
    broken.loc[broken["paper_model_id"].eq("risk_parity"), "n_independent_seeds"] = 5
    broken.to_csv(final_dir / "paper_main_comparison.csv", index=False)

    rows = formal_readiness._audit_p16_final_table(
        final_dir,
        formal_readiness.PROTOCOL_ID,
        formal_readiness.DATA_CUTOFF_DATE,
        promoted=True,
    )

    deterministic_row = next(
        row for row in rows if row["requirement"] == "p16_final_table_deterministic_baseline_1seed"
    )
    assert deterministic_row["status"] == "fail"
