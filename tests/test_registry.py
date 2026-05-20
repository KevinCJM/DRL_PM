import json
import sqlite3
from copy import deepcopy

import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.utils.logger import mark_run_failed, mark_run_status, write_run_outputs


def test_run_registry(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config["config_hash"] = "registry-config-hash"
    config["output"]["run_name"] = "registry_test"
    result = {
        "metrics": {"cumulative_return": 0.12, "sharpe": 1.5},
        "daily_returns": pd.DataFrame(
            {
                "next_valuation_date": ["2024-01-03"],
                "execution_price_type": ["open"],
                "fold_id": ["fixed"],
                "net_return": [0.12],
            }
        ),
        "daily_weights": pd.DataFrame(
            {
                "date": ["2024-01-03"],
                "asset_id": ["510300.SH"],
                "weight": [1.0],
            }
        ),
        "lineage": [{"parent_run_id": "parent_run", "relation": "derived_from"}],
    }

    artifacts = write_run_outputs(result, tmp_path, config=config)

    registry_path = tmp_path / "logs" / "run_registry.sqlite"
    assert artifacts["run_registry"] == registry_path
    with sqlite3.connect(registry_path) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"runs", "artifacts", "metrics", "lineage"}.issubset(tables)

        run = connection.execute("SELECT * FROM runs WHERE run_id = ?", ("registry_test",)).fetchone()
        assert run["status"] == "success"
        assert run["run_dir"] == str(tmp_path)
        assert run["started_at"] is not None
        assert run["completed_at"] is not None
        assert json.loads(run["manifest_json"])["config_hash"] == "registry-config-hash"

        artifact = connection.execute(
            "SELECT * FROM artifacts WHERE run_id = ? AND artifact_name = ?",
            ("registry_test", "run_manifest"),
        ).fetchone()
        assert artifact["path"] == str(tmp_path / "logs" / "run_manifest.json")
        assert artifact["sha256"] is not None
        assert artifact["size_bytes"] > 0
        assert artifact["created_at"] is not None

        metric = connection.execute(
            "SELECT metric_value FROM metrics WHERE run_id = ? AND metric_name = ? AND split = ?",
            ("registry_test", "cumulative_return", "all"),
        ).fetchone()
        assert metric["metric_value"] == pytest.approx(0.12)

        lineage = connection.execute("SELECT * FROM lineage WHERE run_id = ?", ("registry_test",)).fetchone()
        assert lineage["parent_run_id"] == "parent_run"
        assert lineage["relation"] == "derived_from"


def test_run_registry_failed_status(tmp_path):
    registry_path = tmp_path / "run_registry.sqlite"

    mark_run_status("running", registry_path, run_id="failed_run")
    mark_run_failed({"message": "boom"}, registry_path, run_id="failed_run")

    with sqlite3.connect(registry_path) as connection:
        connection.row_factory = sqlite3.Row
        run = connection.execute("SELECT * FROM runs WHERE run_id = ?", ("failed_run",)).fetchone()
        tables = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"runs", "artifacts", "metrics", "lineage"}.issubset(tables)
        assert run["status"] == "failed"
        assert run["fail_reason"] == "boom"
        assert json.loads(run["failure_state_json"])["message"] == "boom"
