from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
import src.experiments.external_baselines as external_baselines
from src.experiments.external_baselines import (
    external_pgportfolio_config,
    import_external_pgportfolio_outputs,
    run_external_pgportfolio_baseline,
    validate_external_pgportfolio_import,
)
from src.experiments.pipeline import _baseline_metadata, _comparison_rows, _paired_return_payload
from src.experiments.registry import _baseline_factories


def test_external_pgportfolio_disabled_by_default():
    config = deepcopy(DEFAULT_CONFIG)

    assert external_pgportfolio_config(config).get("enabled") is False
    assert "pgportfolio_original_external" not in _baseline_factories(config)


def test_external_pgportfolio_out_of_scope_is_skipped():
    config = deepcopy(DEFAULT_CONFIG)
    config["baselines"]["external_pgportfolio"] = {
        "enabled": True,
        "repo_path": "/tmp/PGPortfolio",
        "repo_whitelist": ["external/PGPortfolio"],
    }

    result = run_external_pgportfolio_baseline(config, {})
    summary = result["baseline_training_summary"]

    assert result["status"] == "skipped_out_of_scope"
    assert summary.loc[0, "status"] == "skipped_out_of_scope"
    assert bool(summary.loc[0, "platform_native_rl_training"]) is False
    assert bool(summary.loc[0, "external_original_implementation"]) is True
    assert bool(summary.loc[0, "rankable_in_unified_table"]) is False
    assert "out_of_scope" in summary.loc[0, "out_of_scope_edge"]


def test_external_pgportfolio_absolute_config_whitelist_escape_is_ignored():
    config = deepcopy(DEFAULT_CONFIG)
    config["baselines"]["external_pgportfolio"] = {
        "enabled": True,
        "repo_path": "/tmp/PGPortfolio",
        "repo_whitelist": ["/tmp"],
    }

    result = run_external_pgportfolio_baseline(config, {})

    assert result["status"] == "skipped_out_of_scope"


def test_external_pgportfolio_subprocess_failure_fails_child_run(tmp_path, monkeypatch):
    monkeypatch.setattr(external_baselines, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(external_baselines, "EXTERNAL_PGPORTFOLIO_REPO_ROOT", tmp_path / "external" / "PGPortfolio")
    repo = tmp_path / "external" / "PGPortfolio"
    repo.mkdir(parents=True)
    config = deepcopy(DEFAULT_CONFIG)
    config["baselines"]["external_pgportfolio"] = {
        "enabled": True,
        "repo_path": str(repo),
        "command": [sys.executable, "-c", "import sys; sys.exit(3)"],
        "timeout_seconds": 30,
    }

    result = run_external_pgportfolio_baseline(config, {}, run_dir=tmp_path / "run")
    summary = result["baseline_training_summary"]

    assert result["status"] == "failed"
    assert summary.loc[0, "fail_reason"] == "external_process_nonzero"
    assert int(summary.loc[0, "returncode"]) == 3
    assert (tmp_path / "run" / "external_pgportfolio" / "stderr.txt").exists()


def test_external_pgportfolio_subprocess_success_imports_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(external_baselines, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(external_baselines, "EXTERNAL_PGPORTFOLIO_REPO_ROOT", tmp_path / "external" / "PGPortfolio")
    repo = tmp_path / "external" / "PGPortfolio"
    repo.mkdir(parents=True)
    result_csv = tmp_path / "external_results.csv"
    command = (
        "from pathlib import Path; "
        f"Path({str(result_csv)!r}).write_text("
        "'date,nav,net_return,A,B\\n2024-01-01,1.0,0.0,0.5,0.5\\n2024-01-02,1.01,0.01,0.4,0.6\\n', "
        "encoding='utf-8')"
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["baselines"]["external_pgportfolio"] = {
        "enabled": True,
        "repo_path": str(repo),
        "import_results_csv": str(result_csv),
        "command": [sys.executable, "-c", command],
        "timeout_seconds": 30,
    }
    artifacts = {"dataset": SimpleNamespace(data_manifest={"canonical_asset_order": ["A", "B"]})}

    result = run_external_pgportfolio_baseline(config, artifacts, run_dir=tmp_path / "run")

    assert result["status"] == "completed"
    assert result["daily_returns"]["model_name"].eq("pgportfolio_original_external").all()
    assert result["daily_weights"].groupby("date")["weight"].sum().round(10).eq(1.0).all()
    assert bool(result["baseline_training_summary"].loc[0, "platform_native_rl_training"]) is False
    assert result["baseline_training_summary"].loc[0, "repo_path"] == str(repo)
    assert result["baseline_training_summary"].loc[0, "external_repo"] == "https://github.com/ZhengyaoJiang/PGPortfolio"
    assert "external_results.csv" in result["baseline_training_summary"].loc[0, "import_results_csv"]


def test_external_pgportfolio_is_not_rankable_in_unified_table():
    metadata = _baseline_metadata("pgportfolio_original_external")

    assert metadata["baseline_family"] == "external_original"
    assert metadata["rl_training"] is True
    assert metadata["platform_native_rl_training"] is False
    assert metadata["external_original_implementation"] is True
    assert metadata["rankable_in_unified_table"] is False
    assert metadata["cost_model_shared"] is False


def test_baseline_comparison_separates_proxy_native_external_rows():
    frame = _comparison_rows(
        {
            "ppo_baseline": {"cumulative_return": 0.01},
            "pgportfolio_eiie_native": {"cumulative_return": 0.02},
            "pgportfolio_original_external": {"cumulative_return": 0.03},
        }
    )
    rows = frame.set_index("model_name")

    assert rows.loc["ppo_baseline", "baseline_family"] == "neural_proxy"
    assert bool(rows.loc["ppo_baseline", "rankable_in_unified_table"]) is False
    assert rows.loc["pgportfolio_eiie_native", "baseline_family"] == "native_rl"
    assert rows.loc["pgportfolio_eiie_native", "training_algorithm"] == "pgportfolio_eiie_osbl"
    assert bool(rows.loc["pgportfolio_eiie_native", "rankable_in_unified_table"]) is True
    assert rows.loc["pgportfolio_original_external", "baseline_family"] == "external_original"
    assert bool(rows.loc["pgportfolio_original_external", "platform_native_rl_training"]) is False
    assert bool(rows.loc["pgportfolio_original_external", "rankable_in_unified_table"]) is False


def test_statistics_payload_excludes_proxy_and_external_rows():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")

    def returns(value):
        return pd.DataFrame({"date": dates, "net_return": [value, value, value]})

    payload = _paired_return_payload(
        {
            "equal_weight": returns(0.0),
            "ppo_baseline": returns(0.1),
            "pgportfolio_eiie_native": returns(0.02),
            "pgportfolio_original_external": returns(0.03),
        },
        {"statistics": {"primary_benchmark": "equal_weight"}},
    )

    assert set(payload["model_returns"]) == {"pgportfolio_eiie_native"}
    assert set(payload["benchmark_returns"]) == {"equal_weight"}


def test_external_import_weights_align_test_universe():
    valid = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "nav": [1.0, 1.01],
            "net_return": [0.0, 0.01],
            "A": [0.5, 0.4],
            "B": [0.3, 0.4],
            "C": [0.2, 0.2],
        }
    )

    assert validate_external_pgportfolio_import(valid, ["A", "B", "C"])["status"] == "completed"

    missing = valid.drop(columns=["C"])
    assert validate_external_pgportfolio_import(missing, ["A", "B", "C"])["status"] == "failed_missing_asset_weights"

    extra = valid.assign(D=[0.0, 0.0])
    assert validate_external_pgportfolio_import(extra, ["A", "B", "C"])["status"] == "failed_extra_asset_weights"

    cash = valid.assign(cash=[0.0, 0.0])
    assert validate_external_pgportfolio_import(cash, ["A", "B", "C"])["status"] == "failed_cash_weight_column"

    bad_sum = valid.copy()
    bad_sum.loc[0, "A"] = 0.8
    assert validate_external_pgportfolio_import(bad_sum, ["A", "B", "C"])["status"] == "failed_weight_sum_tolerance"

    bad_nav = valid.copy().astype({"nav": object})
    bad_nav.loc[0, "nav"] = "bad"
    assert validate_external_pgportfolio_import(bad_nav, ["A", "B", "C"])["status"] == "failed_nav_numeric"

    nan_nav = valid.copy()
    nan_nav.loc[0, "nav"] = None
    assert validate_external_pgportfolio_import(nan_nav, ["A", "B", "C"])["status"] == "failed_nav_nan"

    bad_return = valid.copy().astype({"net_return": object})
    bad_return.loc[0, "net_return"] = "bad"
    assert validate_external_pgportfolio_import(bad_return, ["A", "B", "C"])["status"] == "failed_net_return_numeric"

    inf_nav = valid.copy()
    inf_nav.loc[0, "nav"] = np.inf
    assert validate_external_pgportfolio_import(inf_nav, ["A", "B", "C"])["status"] == "failed_nav_non_finite"

    inf_return = valid.copy()
    inf_return.loc[0, "net_return"] = np.inf
    assert (
        validate_external_pgportfolio_import(inf_return, ["A", "B", "C"])["status"]
        == "failed_net_return_non_finite"
    )

    outside = valid.copy()
    outside.loc[0, "date"] = "1900-01-01"
    assert (
        validate_external_pgportfolio_import(
            outside,
            ["A", "B", "C"],
            test_dates=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )["status"]
        == "failed_date_outside_test_split"
    )

    missing_date = valid.iloc[:1].copy()
    assert (
        validate_external_pgportfolio_import(
            missing_date,
            ["A", "B", "C"],
            test_dates=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )["status"]
        == "failed_missing_test_dates"
    )

    availability = pd.DataFrame(
        {
            "A": [True, True],
            "B": [False, True],
            "C": [True, True],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    assert (
        validate_external_pgportfolio_import(
            valid,
            ["A", "B", "C"],
            test_dates=availability.index,
            availability_mask=availability,
        )["status"]
        == "failed_unavailable_asset_nonzero_weight"
    )

    weights_column = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "nav": [1.0, 1.01],
            "net_return": [0.0, 0.01],
            "weights": ['{"A": 0.5, "B": 0.3, "C": 0.2}', '{"A": 0.4, "B": 0.4, "C": 0.2}'],
        }
    )
    assert validate_external_pgportfolio_import(weights_column, ["A", "B", "C"])["status"] == "completed"


def test_external_import_outputs_use_frozen_daily_schema():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "nav": [1.0, 1.01],
            "net_return": [0.0, 0.01],
            "A": [0.5, 0.4],
            "B": [0.5, 0.6],
        }
    )

    payload = import_external_pgportfolio_outputs(frame, ["A", "B"])

    assert payload["status"] == "completed"
    assert {"decision_date", "execution_date", "execution_price_type", "net_return", "nav"}.issubset(
        payload["daily_returns"].columns
    )
    assert {"estimated_cost", "realized_cost", "total_transaction_cost"}.issubset(payload["daily_costs"].columns)
    assert payload["daily_returns"]["execution_price_type"].eq("external").all()
    assert payload["daily_weights"].groupby("date")["weight"].sum().round(10).eq(1.0).all()


def test_baseline_comparison_persists_checkpoint_training_fields():
    frame = _comparison_rows(
        {"dqn_template_native": {"cumulative_return": 0.01}},
        training_summary_rows=[
            {
                "model_name": "dqn_template_native",
                "checkpoint_best_path": "best.pt",
                "checkpoint_last_path": "last.pt",
                "evaluated_checkpoint_path": "best.pt",
                "env_steps": 12,
                "gradient_updates": 3,
                "best_validation_metric": 0.4,
            }
        ],
    )
    row = frame.iloc[0]

    assert row["checkpoint_best_path"] == "best.pt"
    assert row["evaluated_checkpoint_path"] == "best.pt"
    assert int(row["env_steps"]) == 12
    assert int(row["gradient_updates"]) == 3
