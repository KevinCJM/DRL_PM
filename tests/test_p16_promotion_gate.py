import pandas as pd

from scripts.evaluate_p16_promotion_gate import evaluate_promotion_gate
from scripts.generate_p16_validation_references import validation_comparison


def test_p16_validation_comparison_outputs_required_fields():
    daily_returns = pd.DataFrame(
        [
            {"date": "2024-01-01", "model_name": "eiie_native", "seed": 42, "net_return": 0.01, "nav": 1.01},
            {"date": "2024-01-02", "model_name": "eiie_native", "seed": 42, "net_return": -0.005, "nav": 1.00495},
        ]
    )
    daily_turnover = pd.DataFrame(
        [
            {"date": "2024-01-01", "model_name": "eiie_native", "turnover": 0.1},
            {"date": "2024-01-02", "model_name": "eiie_native", "turnover": 0.2},
        ]
    )
    daily_costs = pd.DataFrame(
        [
            {"date": "2024-01-01", "model_name": "eiie_native", "total_transaction_cost": 0.001},
            {"date": "2024-01-02", "model_name": "eiie_native", "total_transaction_cost": 0.002},
        ]
    )

    comparison = validation_comparison(
        pd.DataFrame([{"model_name": "eiie_native", "paper_model_id": "eiie_native"}]),
        daily_returns=daily_returns,
        daily_turnover=daily_turnover,
        daily_costs=daily_costs,
        model_names=("eiie_native",),
    )

    row = comparison.iloc[0]
    assert row["split"] == "validation"
    assert bool(row["rankable_reference"])
    assert row["average_cost_per_step"] == 0.0015
    assert "validation_utility" in comparison.columns


def test_p16_promotion_gate_reads_pilot_outputs(tmp_path):
    reference_dir = tmp_path / "refs"
    pilot_dir = tmp_path / "pilot"
    (reference_dir).mkdir()
    (pilot_dir / "metrics").mkdir(parents=True)
    (pilot_dir / "logs").mkdir()
    pd.DataFrame(
        [
            _reference("eiie_native", cumulative_return=0.10, cvar=0.02, mdd=0.05, turnover=0.20, cost=0.0005, utility=0.06),
            _reference("ppo_native", cumulative_return=0.08, cvar=0.03, mdd=0.06, turnover=0.30, cost=0.0006, utility=0.04),
            _reference("cage_eiie_joint_light", cumulative_return=0.11, cvar=0.025, mdd=0.055, turnover=0.18, cost=0.0005, utility=0.05),
        ]
    ).to_csv(reference_dir / "validation_reference_comparison.csv", index=False)
    model = "risk_aware_graph_transformer_constrained_actor_critic"
    pd.DataFrame([{"model_name": model, "paper_model_id": model}]).to_csv(
        pilot_dir / "metrics/hpo_model_final_comparison.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {"date": "2024-01-01", "model_name": model, "seed": 42, "net_return": 0.07, "nav": 1.07},
            {"date": "2024-01-02", "model_name": model, "seed": 42, "net_return": 0.03, "nav": 1.1021},
        ]
    ).to_csv(pilot_dir / "metrics/hpo_model_final_daily_returns.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-01", "model_name": model, "turnover": 0.05},
            {"date": "2024-01-02", "model_name": model, "turnover": 0.06},
        ]
    ).to_csv(pilot_dir / "metrics/hpo_model_final_daily_turnover.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-01", "model_name": model, "total_transaction_cost": 0.0001},
            {"date": "2024-01-02", "model_name": model, "total_transaction_cost": 0.0001},
        ]
    ).to_csv(pilot_dir / "metrics/hpo_model_final_daily_costs.csv", index=False)
    pd.DataFrame(
        [
            {"model_name": model, "state": "complete"},
            {"model_name": model, "state": "complete"},
        ]
    ).to_csv(pilot_dir / "logs/hpo_trials.csv", index=False)

    paths = evaluate_promotion_gate(
        pilot_run_dir=pilot_dir,
        reference_dir=reference_dir,
        output_dir=tmp_path / "gate",
        model_names=(model,),
        configured_average_cost_per_step_budget=0.001,
    )

    report = pd.read_csv(paths["gate_report"])
    assert bool(report["condition_failed_trial_rate_le_20pct"].iloc[0])
    assert bool(report["condition_finite_artifact_rate_1"].iloc[0])
    assert (tmp_path / "gate/validation_reference_comparison.csv").exists()


def _reference(model_name, *, cumulative_return, cvar, mdd, turnover, cost, utility):
    return {
        "model_name": model_name,
        "paper_model_id": model_name,
        "split": "validation",
        "cumulative_return": cumulative_return,
        "CVaR_loss_5": cvar,
        "max_drawdown_loss": mdd,
        "average_turnover": turnover,
        "average_cost_per_step": cost,
        "total_transaction_cost": cost * 2,
        "validation_utility": utility,
    }
