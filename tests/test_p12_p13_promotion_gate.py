import json

import pandas as pd
import pytest

import scripts.evaluate_p12_p13_promotion_gate as gate


def test_p12_promotion_gate_does_not_force_p13(tmp_path, monkeypatch):
    reference_dir = tmp_path / "refs"
    reference_dir.mkdir()
    pd.DataFrame(
        [
            _reference("eiie_native", cumulative_return=0.10, turnover=0.20, cost=0.010, mdd=0.05, cvar=0.02),
            _reference(
                "full_dqn_gated_multitask_cnn_ppo",
                cumulative_return=0.08,
                turnover=0.18,
                cost=0.009,
                mdd=0.06,
                cvar=0.03,
            ),
        ]
    ).to_csv(reference_dir / "validation_reference_comparison.csv", index=False)
    calls = []

    def fake_evaluate_candidates(*, config_path, run_dir, output_dir, model_names):
        calls.append((str(config_path), str(run_dir), tuple(model_names)))
        return [
            pd.DataFrame(
                [
                    {
                        "model_name": "cage_eiie_frozen_gate",
                        "cumulative_return": 0.12,
                        "turnover_mean": 0.19,
                        "transaction_cost_total": 0.009,
                        "max_drawdown_loss": 0.04,
                        "CVaR_loss_5": 0.015,
                        "validation_return_cost_risk_utility": 0.11,
                        "daily_returns_finite": True,
                        "daily_nav_finite": True,
                    }
                ]
            )
        ]

    monkeypatch.setattr(gate, "_evaluate_candidates", fake_evaluate_candidates)

    paths = gate.evaluate_promotion_gate(
        p12_config="p12.yaml",
        p12_run_dir="p12_run",
        reference_dir=reference_dir,
        output_dir=tmp_path / "gate",
    )

    report = pd.read_csv(paths["gate_report"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert calls[0][2] == gate.P12_MODELS
    assert set(report["phase"]) == {"P12"}
    assert manifest["p13_evaluated"] is False
    assert manifest["p13_config"] is None
    assert manifest["p13_run_dir"] is None


def test_p12_promotion_gate_rejects_incomplete_p13_args(tmp_path):
    with pytest.raises(ValueError, match="ERR_PROMOTION_GATE_P13_ARGS_INCOMPLETE"):
        gate.evaluate_promotion_gate(
            p12_config="p12.yaml",
            p12_run_dir="p12_run",
            reference_dir=tmp_path / "refs",
            output_dir=tmp_path / "gate",
            p13_config="p13.yaml",
        )


def _reference(model_name, *, cumulative_return, turnover, cost, mdd, cvar):
    return {
        "model_name": model_name,
        "cumulative_return": cumulative_return,
        "turnover_mean": turnover,
        "transaction_cost_total": cost,
        "max_drawdown_loss": mdd,
        "CVaR_loss_5": cvar,
    }
