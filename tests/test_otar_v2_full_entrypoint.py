from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.config import ConfigLoader, PROJECT_ROOT
from src.experiments import otar_v2_full
from src.experiments import pipeline
from src.experiments import run_experiment


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "otar_base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
                "output": {"root": str(tmp_path / "outputs")},
                "device": {"mode": "cpu"},
                "registry": {"path": str(tmp_path / "run_registry.sqlite")},
                "protocol": {"protocol_id": "otar_v2_s0_20260605", "data_cutoff_date": "2026-05-20"},
                "rankability": {"rankable_in_unified_table": False, "diagnostic_status": "diagnostic"},
                "training": {"checkpoint_include_replay_buffer": False},
                "rebalance": {"mode": "daily"},
                "execution_activity": {
                    "protocol": "daily_gate_with_cost_constraint",
                    "scheduler_blocks_model_actions": False,
                    "activity_gate_enforced": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_completed_child(child_dir: Path, *, child_config_hash: str = "hash") -> Path:
    logs_dir = child_dir / "logs"
    metrics_dir = child_dir / "metrics"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "experiment_result.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (logs_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "rankable_in_unified_table": True,
                "diagnostic_status": "formal",
                "protocol_id": "otar_v2_s0_20260605",
                "data_cutoff_date": "2026-05-20",
                "config_hash": child_config_hash,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"model_name": "model", "rankable_in_unified_table": True, "sharpe": 1.0}]).to_csv(
        metrics_dir / "hpo_model_final_comparison.csv",
        index=False,
    )
    pd.DataFrame([{"date": "2024-01-02", "model_name": "model", "net_return": 0.01}]).to_csv(
        metrics_dir / "hpo_model_final_daily_returns.csv",
        index=False,
    )
    return child_dir


def _write_aggregate_manifest(output_dir: Path, rows: int = 1) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "paper_aggregate_manifest.json"
    manifest_path.write_text(
        json.dumps({"row_counts": {"paper_main_comparison": rows}}),
        encoding="utf-8",
    )
    main_path = output_dir / "paper_main_comparison.csv"
    pd.DataFrame([{"model_name": "model"}] if rows else []).to_csv(main_path, index=False)
    return {"paper_aggregate_manifest": manifest_path, "paper_main_comparison": main_path}


def test_otar_v2_full_dry_run_writes_formal_plan(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(otar_v2_full, "validate_protocol", lambda post_freeze: {"status": "success"})
    monkeypatch.setattr(otar_v2_full, "_assert_frozen_config", lambda config_path: None)

    result = otar_v2_full.run_otar_v2_full_pipeline(
        config_path=config_path,
        output_root=tmp_path / "outputs",
        run_name="OTAR_FULL_DRY",
        dry_run=True,
    )

    run_dir = tmp_path / "outputs" / "OTAR_FULL_DRY"
    assert result["status"] == "dry_run"
    assert (run_dir / "logs" / "otar_v2_full_manifest.json").exists()
    plan = json.loads((run_dir / "logs" / "formal_matrix_plan.json").read_text(encoding="utf-8"))
    assert plan["full_run_count"] == 105
    snapshot = yaml.safe_load((run_dir / "logs" / "config_snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot["rankability"]["rankable_in_unified_table"] is True
    assert snapshot["rankability"]["diagnostic_status"] == "formal"
    assert snapshot["experiment"]["type"] == "otar_formal_matrix"


def test_otar_v2_full_runs_formal_then_aggregate(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(otar_v2_full, "validate_protocol", lambda post_freeze: calls.append("s0") or {"status": "success"})
    monkeypatch.setattr(otar_v2_full, "_assert_frozen_config", lambda config_path: None)

    def fake_formal(config, *, matrix_path, run_dir, max_runs, device, resume_completed):
        calls.append("formal")
        assert config["rankability"]["rankable_in_unified_table"] is True
        assert config["rankability"]["diagnostic_status"] == "formal"
        assert resume_completed is True
        plan = otar_v2_full._planned_child_runs(Path(matrix_path), config, Path(run_dir), formal_max_runs=max_runs)
        row = plan["child_runs"][0]
        child_dir = _write_completed_child(Path(row["child_run_dir"]), child_config_hash=row["child_config_hash"])
        return {
            "status": "completed",
            "run_count": 1,
            "lineage": [{"status": "completed", "child_run_id": "child", "run_dir": str(child_dir)}],
        }

    def fake_aggregate(run_dirs, output_dir, **kwargs):
        calls.append("aggregate")
        assert len(run_dirs) == 1
        assert kwargs["required_protocol_id"] == "otar_v2_s0_20260605"
        assert kwargs["required_data_cutoff_date"] == "2026-05-20"
        assert kwargs["require_formal_manifest"] is True
        assert kwargs["require_availability_mask_contract"] is True
        return _write_aggregate_manifest(Path(output_dir), rows=1)

    monkeypatch.setattr(otar_v2_full, "run_otar_formal_matrix", fake_formal)
    monkeypatch.setattr(otar_v2_full, "aggregate_paper_results", fake_aggregate)

    result = otar_v2_full.run_otar_v2_full_pipeline(
        config_path=config_path,
        output_root=tmp_path / "outputs",
        run_name="OTAR_FULL",
        formal_max_runs=1,
        resume_completed_children=True,
    )

    assert calls == ["s0", "formal", "aggregate"]
    assert result["status"] == "completed"
    assert result["aggregate_status"] == "completed"


def test_otar_v2_full_allows_zero_formal_runs_when_aggregate_skipped(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(otar_v2_full, "validate_protocol", lambda post_freeze: {"status": "success"})
    monkeypatch.setattr(otar_v2_full, "_assert_frozen_config", lambda config_path: None)
    monkeypatch.setattr(
        otar_v2_full,
        "run_otar_formal_matrix",
        lambda *args, **kwargs: {"status": "completed", "run_count": 0, "lineage": []},
    )

    result = otar_v2_full.run_otar_v2_full_pipeline(
        config_path=config_path,
        output_root=tmp_path / "outputs",
        run_name="OTAR_ZERO",
        formal_max_runs=0,
        skip_aggregate=True,
    )

    assert result["status"] == "completed"
    assert result["child_run_dirs"] == []
    assert result["aggregate_status"] == "skipped"


def test_otar_v2_full_rejects_noncanonical_matrix_when_s0_enabled(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    custom_matrix = tmp_path / "matrix.yaml"
    custom_matrix.write_text("protocol_id: custom\n", encoding="utf-8")
    monkeypatch.setattr(otar_v2_full, "validate_protocol", lambda post_freeze: {"status": "success"})
    monkeypatch.setattr(otar_v2_full, "_assert_frozen_config", lambda config_path: None)

    with pytest.raises(RuntimeError, match="ERR_OTAR_V2_FULL_UNFROZEN_MATRIX"):
        otar_v2_full.run_otar_v2_full_pipeline(
            config_path=config_path,
            matrix_path=custom_matrix,
            output_root=tmp_path / "outputs",
            run_name="OTAR_BAD_MATRIX",
            dry_run=True,
        )


def test_otar_v2_full_rejects_unfrozen_config_when_s0_enabled(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(otar_v2_full, "validate_protocol", lambda post_freeze: {"status": "success"})

    with pytest.raises(RuntimeError, match="ERR_OTAR_V2_FULL_UNFROZEN_CONFIG"):
        otar_v2_full.run_otar_v2_full_pipeline(
            config_path=config_path,
            output_root=tmp_path / "outputs",
            run_name="OTAR_BAD_CONFIG",
            dry_run=True,
        )


def test_otar_v2_full_skip_formal_requires_completed_child_artifacts(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(otar_v2_full, "validate_protocol", lambda post_freeze: {"status": "success"})
    monkeypatch.setattr(otar_v2_full, "_assert_frozen_config", lambda config_path: None)
    run_dir = tmp_path / "outputs" / "OTAR_SKIP_FORMAL"
    base = otar_v2_full._load_formal_base_config(
        config_path,
        output_root=tmp_path / "outputs",
        run_name="OTAR_SKIP_FORMAL",
        device=None,
    )
    plan = otar_v2_full._planned_child_runs(
        PROJECT_ROOT / otar_v2_full.CANONICAL_FORMAL_MATRIX,
        base,
        run_dir,
        formal_max_runs=1,
    )
    child_dir = Path(plan["child_runs"][0]["child_run_dir"])
    (child_dir / "logs").mkdir(parents=True)
    (child_dir / "logs" / "experiment_result.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "otar_formal_lineage.json").write_text(
        json.dumps({"status": "completed", "lineage": [{"status": "completed", "child_run_id": "child", "run_dir": str(child_dir)}]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ERR_OTAR_V2_FULL_CHILD_ARTIFACT_INCOMPLETE"):
        otar_v2_full.run_otar_v2_full_pipeline(
            config_path=config_path,
            output_root=tmp_path / "outputs",
            run_name="OTAR_SKIP_FORMAL",
            formal_max_runs=1,
            skip_formal_runs=True,
        )


def test_otar_v2_full_skip_formal_rejects_stale_child_hash(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(otar_v2_full, "validate_protocol", lambda post_freeze: {"status": "success"})
    monkeypatch.setattr(otar_v2_full, "_assert_frozen_config", lambda config_path: None)
    run_dir = tmp_path / "outputs" / "OTAR_STALE_SKIP"
    base = otar_v2_full._load_formal_base_config(
        config_path,
        output_root=tmp_path / "outputs",
        run_name="OTAR_STALE_SKIP",
        device=None,
    )
    plan = otar_v2_full._planned_child_runs(
        PROJECT_ROOT / otar_v2_full.CANONICAL_FORMAL_MATRIX,
        base,
        run_dir,
        formal_max_runs=1,
    )
    child_dir = _write_completed_child(Path(plan["child_runs"][0]["child_run_dir"]), child_config_hash="stale")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "otar_formal_lineage.json").write_text(
        json.dumps({"status": "completed", "lineage": [{"status": "completed", "child_run_id": "child", "run_dir": str(child_dir)}]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ERR_OTAR_V2_FULL_CHILD_LINEAGE_PLAN_MISMATCH"):
        otar_v2_full.run_otar_v2_full_pipeline(
            config_path=config_path,
            output_root=tmp_path / "outputs",
            run_name="OTAR_STALE_SKIP",
            formal_max_runs=1,
            skip_formal_runs=True,
        )


def test_otar_v2_full_rejects_empty_aggregate_main_table(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(otar_v2_full, "validate_protocol", lambda post_freeze: {"status": "success"})
    monkeypatch.setattr(otar_v2_full, "_assert_frozen_config", lambda config_path: None)

    def fake_formal(config, *, matrix_path, run_dir, max_runs, device, resume_completed):
        plan = otar_v2_full._planned_child_runs(Path(matrix_path), config, Path(run_dir), formal_max_runs=max_runs)
        row = plan["child_runs"][0]
        child_dir = _write_completed_child(Path(row["child_run_dir"]), child_config_hash=row["child_config_hash"])
        return {"status": "completed", "lineage": [{"status": "completed", "child_run_id": "child", "run_dir": str(child_dir)}]}

    monkeypatch.setattr(otar_v2_full, "run_otar_formal_matrix", fake_formal)
    monkeypatch.setattr(
        otar_v2_full,
        "aggregate_paper_results",
        lambda run_dirs, output_dir, **kwargs: _write_aggregate_manifest(Path(output_dir), rows=0),
    )

    with pytest.raises(RuntimeError, match="ERR_OTAR_V2_FULL_AGGREGATE_EMPTY_MAIN_TABLE"):
        otar_v2_full.run_otar_v2_full_pipeline(
            config_path=config_path,
            output_root=tmp_path / "outputs",
            run_name="OTAR_EMPTY_AGG",
            formal_max_runs=1,
        )


def test_otar_formal_matrix_resume_rejects_stale_child(tmp_path):
    config_path = _write_config(tmp_path)
    base = otar_v2_full._load_formal_base_config(
        config_path,
        output_root=tmp_path / "outputs",
        run_name="OTAR_RESUME",
        device="cpu",
    )
    parent = tmp_path / "formal_parent"
    first = pipeline.expand_otar_formal_matrix(PROJECT_ROOT / otar_v2_full.CANONICAL_FORMAL_MATRIX, base)[0]
    run_name = first["output"]["run_name"]
    child_dir = parent / f"001_{run_name}"
    (child_dir / "logs").mkdir(parents=True)
    (child_dir / "logs" / "experiment_result.json").write_text(
        json.dumps({"status": "completed", "metrics": {"sharpe": 1.0}}),
        encoding="utf-8",
    )
    (child_dir / "logs" / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "config_hash": "stale",
                "protocol_id": first["protocol"]["protocol_id"],
                "data_cutoff_date": first["protocol"]["data_cutoff_date"],
                "ablation_id": first["experiment"]["ablation_id"],
                "asset_universe_id": first["protocol"]["asset_universe_id"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="ERR_OTAR_FORMAL_RESUME_STALE_CHILD"):
        pipeline.run_otar_formal_matrix(
            base,
            matrix_path=PROJECT_ROOT / otar_v2_full.CANONICAL_FORMAL_MATRIX,
            run_dir=parent,
            max_runs=1,
            resume_completed=True,
        )


def test_run_experiment_formal_matrix_legacy_call_still_works(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path)

    def fake_formal(config, *, matrix_path, run_dir, max_runs, device):
        return {
            "status": "completed",
            "run_count": 0,
            "lineage": [],
            "otar_formal_matrix_summary": pd.DataFrame(),
            "output_name": "otar_formal_matrix_summary",
        }

    monkeypatch.setattr(run_experiment, "run_otar_formal_matrix", fake_formal)

    run_dir = run_experiment.main(
        [
            "--config",
            str(config_path),
            "--formal-matrix",
            str(PROJECT_ROOT / otar_v2_full.CANONICAL_FORMAL_MATRIX),
            "--formal-max-runs",
            "0",
            "--output",
            str(tmp_path / "outputs"),
            "--run-name",
            "LEGACY_FORMAL",
        ]
    )

    assert (run_dir / "logs" / "run_manifest.json").exists()
