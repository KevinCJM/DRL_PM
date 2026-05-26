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
        pd.DataFrame(
            [
                {
                    "paper_model_id": "full_dqn_gated_multitask_cnn_ppo",
                    "diagnostic_status": "formal",
                    "rankable_in_unified_table": True,
                }
            ]
        ).to_csv(group_dir / "paper_main_comparison.csv", index=False)
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
hpo:
  enabled: true
  equal_budget_across_models: true
  n_trials_per_model: 50
  trainable_models:
  - risk_aware_graph_transformer_constrained_actor_critic
training:
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
