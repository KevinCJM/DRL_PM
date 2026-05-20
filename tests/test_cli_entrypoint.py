from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from src.config import ConfigError, DEFAULT_CONFIG, PROJECT_ROOT
from src.experiments.run_experiment import main


def write_config(tmp_path, output_root):
    config = {
        "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
        "output": {"root": str(output_root)},
        "device": {"mode": "cpu"},
        "registry": {"path": str(tmp_path / "run_registry.sqlite")},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_run_experiment_creates_config_snapshot(tmp_path):
    config_path = write_config(tmp_path, tmp_path / "results")

    run_dir = main(
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
            "smoke.run-1",
        ]
    )

    snapshot_path = run_dir / "logs" / "config_snapshot.yaml"
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert run_dir == tmp_path / "results" / "smoke.run-1"
    assert snapshot["reproducibility"]["seed"] == 123
    assert snapshot["device"]["mode"] == "cpu"
    assert snapshot["output"]["root"] == str(tmp_path / "results")
    assert snapshot["output"]["run_name"] == "smoke.run-1"
    assert "config_hash" in snapshot
    assert (run_dir / "metrics/daily_returns.csv").stat().st_size > 0
    assert (run_dir / "metrics/daily_costs.csv").stat().st_size > 0
    assert (run_dir / "logs/run_manifest.json").stat().st_size > 0
    assert (run_dir / "checkpoints/last.pt").stat().st_size > 0
    assert not (PROJECT_ROOT / "data/processed/smoke.run-1").exists()
    assert not (PROJECT_ROOT / "data/metrics_factory/smoke.run-1").exists()
    assert not (PROJECT_ROOT / "data/reports/smoke.run-1").exists()

    with pytest.raises(ConfigError) as error:
        main(["--config", str(config_path), "--run-name", "../bad"])
    assert error.value.code == "ERR_OUTPUT_INVALID_RUN_NAME"

    with pytest.raises(ConfigError) as dot_error:
        main(["--config", str(config_path), "--run-name", "."])
    assert dot_error.value.code == "ERR_OUTPUT_INVALID_RUN_NAME"

    with pytest.raises(ConfigError) as output_error:
        main(["--config", str(config_path), "--output", str(PROJECT_ROOT / "data/processed")])
    assert output_error.value.code == "ERR_SECURITY_PATH_DENIED"

    with pytest.raises(ConfigError) as run_dir_error:
        main(["--config", str(config_path), "--output", str(PROJECT_ROOT / "data"), "--run-name", "processed"])
    assert run_dir_error.value.code == "ERR_SECURITY_PATH_DENIED"


def test_experiment_alias_matches_config(tmp_path):
    config_path = write_config(tmp_path, tmp_path / "results")
    common_args = [
        "--seed",
        "999",
        "--device",
        "cpu",
        "--output",
        str(tmp_path / "results"),
    ]

    config_run_dir = main(["--config", str(config_path), *common_args, "--run-name", "config_run"])
    experiment_run_dir = main(["--experiment", str(config_path), *common_args, "--run-name", "experiment_run"])

    config_snapshot = yaml.safe_load((config_run_dir / "logs/config_snapshot.yaml").read_text(encoding="utf-8"))
    experiment_snapshot = yaml.safe_load(
        (experiment_run_dir / "logs/config_snapshot.yaml").read_text(encoding="utf-8")
    )
    comparable_config = deepcopy(config_snapshot)
    comparable_experiment = deepcopy(experiment_snapshot)
    comparable_config["output"]["run_name"] = "same"
    comparable_experiment["output"]["run_name"] = "same"
    comparable_config.pop("config_hash")
    comparable_experiment.pop("config_hash")

    assert comparable_config == comparable_experiment
    assert config_snapshot["output"]["run_name"] == "config_run"
    assert experiment_snapshot["output"]["run_name"] == "experiment_run"
    assert DEFAULT_CONFIG["output"]["root"] == "results"


def test_run_experiment_always_writes_fixed_config_snapshot_path(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "security": {"path_whitelist": [str(PROJECT_ROOT), str(tmp_path)]},
                "output": {"root": str(tmp_path / "results")},
                "registry": {"path": str(tmp_path / "run_registry.sqlite")},
                "device": {"mode": "cpu"},
                "logging": {"save_config_snapshot": False, "log_dir": "custom_logs"},
            }
        ),
        encoding="utf-8",
    )

    run_dir = main(["--config", str(config_path), "--run-name", "fixed_logs"])

    assert (run_dir / "logs/config_snapshot.yaml").exists()
    assert not (run_dir / "custom_logs/config_snapshot.yaml").exists()
