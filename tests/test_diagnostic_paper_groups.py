import json

import pandas as pd

from src.experiments.diagnostic_paper_groups import audit_diagnostic_paper_groups


def test_diagnostic_group_audit_marks_missing_source_runs_pending(tmp_path):
    group = tmp_path / "results/paper_tables/p2_input_pca"
    group.mkdir(parents=True)
    pd.DataFrame([{"model_name": "main"}]).to_csv(group / "paper_main_comparison.csv", index=False)
    pd.DataFrame([{"model_name": "main", "metric_name": "sharpe"}]).to_csv(
        group / "paper_seed_summary.csv",
        index=False,
    )
    pd.DataFrame([{"model_name": "main", "status": "skipped"}]).to_csv(
        group / "paper_paired_statistics.csv",
        index=False,
    )

    outputs = audit_diagnostic_paper_groups(root=tmp_path, groups=["p2_input_pca"])

    payload = json.loads((group / "diagnostic_status.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(outputs["csv"])
    assert payload["status"] == "aggregation_pending"
    assert payload["reason"] == "missing_source_run_dirs"
    assert payload["rankable_in_unified_table"] is False
    assert (group / "source_run_dirs.txt").exists()
    assert audit.loc[0, "status"] == "aggregation_pending"


def test_diagnostic_group_audit_accepts_complete_diagnostic_group(tmp_path):
    run_dir = tmp_path / "results/EXP10_P2_input_matrix_s42"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs/run_manifest.json").write_text(
        json.dumps({"run_name": "EXP10_P2_input_matrix_s42", "diagnostic_status": "diagnostic"}),
        encoding="utf-8",
    )
    group = tmp_path / "results/paper_tables/p2_input_pca"
    group.mkdir(parents=True)
    (group / "source_run_dirs.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    pd.DataFrame([{"model_name": "main"}]).to_csv(group / "paper_main_comparison.csv", index=False)
    pd.DataFrame([{"model_name": "main", "metric_name": "sharpe"}]).to_csv(
        group / "paper_seed_summary.csv",
        index=False,
    )
    (group / "not_applicable_reason.txt").write_text("single diagnostic run has no paired test\n", encoding="utf-8")

    outputs = audit_diagnostic_paper_groups(root=tmp_path, groups=["p2_input_pca"])

    payload = json.loads((group / "diagnostic_status.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(outputs["csv"])
    assert payload["status"] == "diagnostic_complete"
    assert payload["source_run_manifest_count"] == 1
    assert audit.loc[0, "status"] == "diagnostic_complete"
