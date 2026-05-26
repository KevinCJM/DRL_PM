import json

import pandas as pd

import src.experiments.run_ledger as run_ledger


def test_build_experiment_run_ledger_writes_required_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(run_ledger, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "logs" / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_name": "EXP03_P7_hpo_smoke_main_native_s42",
                "config_path": "configs/paper/hpo_equal_budget_main_native_pilot.yaml",
                "seed": 42,
                "data_mode": "availability_mask",
                "protocol_id": "core13_v2_full_reset_20260522",
                "diagnostic_status": "diagnostic",
                "rankable_in_unified_table": False,
                "status": "completed",
                "timestamp": "2026-05-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"model_name": "ppo_native", "rankable_in_unified_table": False, "sharpe": 0.5}]
    ).to_csv(run_dir / "metrics" / "hpo_model_final_comparison.csv", index=False)
    pd.DataFrame([{"date": "2024-01-02", "model_name": "ppo_native", "net_return": 0.01}]).to_csv(
        run_dir / "metrics" / "hpo_model_final_daily_returns.csv",
        index=False,
    )

    outputs = run_ledger.build_experiment_run_ledger([run_dir], tmp_path / "ledger")

    ledger = pd.read_csv(outputs["csv"])
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert ledger.loc[0, "phase"] == "p7"
    assert ledger.loc[0, "run_name"] == "EXP03_P7_hpo_smoke_main_native_s42"
    assert json.loads(ledger.loc[0, "model_scope"]) == ["ppo_native"]
    assert payload["row_count"] == 1
