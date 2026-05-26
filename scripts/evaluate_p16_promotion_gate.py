from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_p16_validation_references import MODEL_EXTENSION_ID, validation_comparison  # noqa: E402
from src.utils.logger import save_json_atomic  # noqa: E402


P16_MODELS = (
    "risk_aware_graph_transformer_constrained_actor_critic",
    "ra_gt_rcpo_no_graph",
    "ra_gt_rcpo_no_transformer",
    "ra_gt_rcpo_no_cvar_constraint",
    "ra_gt_rcpo_no_cost_constraint",
    "ra_gt_rcpo_no_turnover_constraint",
    "ra_gt_rcpo_mlp_actor_critic",
)
DEFAULT_OUTPUT_DIR = "results/paper_tables/p16_promotion_gate"


def evaluate_promotion_gate(
    *,
    pilot_run_dir: str | Path,
    reference_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_names: Sequence[str] = P16_MODELS,
    configured_average_cost_per_step_budget: float = 0.001,
    cost_budget_tolerance: float = 1.0e-6,
) -> dict[str, Path]:
    pilot = Path(pilot_run_dir)
    reference = Path(reference_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    reference_comparison = pd.read_csv(reference / "validation_reference_comparison.csv")
    candidate_comparison = _candidate_comparison_from_pilot(pilot, model_names)
    trial_stats = _trial_stats(pilot / "logs" / "hpo_trials.csv")
    report = promotion_report(
        candidate_comparison,
        reference_comparison,
        trial_stats=trial_stats,
        configured_average_cost_per_step_budget=configured_average_cost_per_step_budget,
        cost_budget_tolerance=cost_budget_tolerance,
    )
    paths = {
        "candidate_comparison": output / "promotion_candidate_comparison.csv",
        "gate_report": output / "promotion_gate_report.csv",
        "reference_comparison": output / "validation_reference_comparison.csv",
        "manifest": output / "promotion_gate_manifest.json",
    }
    candidate_comparison.to_csv(paths["candidate_comparison"], index=False)
    report.to_csv(paths["gate_report"], index=False)
    shutil.copyfile(reference / "validation_reference_comparison.csv", paths["reference_comparison"])
    save_json_atomic(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_extension_id": MODEL_EXTENSION_ID,
            "selection_split": "validation",
            "test_used_for_model_selection": False,
            "pilot_run_dir": str(pilot_run_dir),
            "reference_dir": str(reference_dir),
            "configured_average_cost_per_step_budget": configured_average_cost_per_step_budget,
            "cost_budget_tolerance": cost_budget_tolerance,
            "outputs": {key: str(path) for key, path in paths.items()},
        },
        paths["manifest"],
    )
    return paths


def promotion_report(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    trial_stats: Mapping[str, Mapping[str, float]],
    configured_average_cost_per_step_budget: float,
    cost_budget_tolerance: float,
) -> pd.DataFrame:
    ref = _records(reference)
    rows = []
    for candidate in candidates.to_dict("records"):
        model_name = str(candidate.get("model_name"))
        stats = trial_stats.get(model_name, {})
        rows.append(
            _gate_row(
                candidate,
                ref,
                failed_trial_rate=float(stats.get("failed_trial_rate", np.nan)),
                finite_artifact_rate=float(stats.get("finite_artifact_rate", 1.0)),
                configured_average_cost_per_step_budget=configured_average_cost_per_step_budget,
                cost_budget_tolerance=cost_budget_tolerance,
            )
        )
    return pd.DataFrame(rows)


def _gate_row(
    candidate: Mapping[str, Any],
    ref: Mapping[str, Mapping[str, Any]],
    *,
    failed_trial_rate: float,
    finite_artifact_rate: float,
    configured_average_cost_per_step_budget: float,
    cost_budget_tolerance: float,
) -> dict[str, Any]:
    eiie = ref.get("eiie_native", {})
    ppo = ref.get("ppo_native", {})
    cage = ref.get("cage_eiie_joint_light", {})
    return_ge_eiie = _num(candidate, "cumulative_return") >= _num(eiie, "cumulative_return") - 0.01
    cvar_le_eiie = _num(candidate, "CVaR_loss_5") <= _num(eiie, "CVaR_loss_5")
    mdd_le_eiie = _num(candidate, "max_drawdown_loss") <= _num(eiie, "max_drawdown_loss")
    turnover_le_ppo = _num(candidate, "average_turnover") <= _num(ppo, "average_turnover")
    cost_budget_ok = _num(candidate, "average_cost_per_step") <= float(configured_average_cost_per_step_budget) + float(cost_budget_tolerance)
    cage_utility_ok = _num(candidate, "validation_utility") >= _num(cage, "validation_utility")
    cage_return_ok = _num(candidate, "cumulative_return") >= _num(cage, "cumulative_return") - 0.01
    cage_improvement_count = int(
        sum(
            [
                _num(candidate, "CVaR_loss_5") < _num(cage, "CVaR_loss_5"),
                _num(candidate, "max_drawdown_loss") < _num(cage, "max_drawdown_loss"),
                _num(candidate, "average_turnover") < _num(cage, "average_turnover"),
            ]
        )
    )
    cage_alternative_ok = cage_utility_ok or (cage_return_ok and cage_improvement_count >= 2)
    failed_rate_ok = np.isfinite(failed_trial_rate) and failed_trial_rate <= 0.20
    finite_rate_ok = np.isfinite(finite_artifact_rate) and finite_artifact_rate == 1.0
    passed = all([failed_rate_ok, finite_rate_ok, return_ge_eiie, cvar_le_eiie, mdd_le_eiie, turnover_le_ppo, cost_budget_ok, cage_alternative_ok])
    return {
        "model_name": candidate.get("model_name"),
        "selection_split": "validation",
        "test_used_for_model_selection": False,
        "cumulative_return": candidate.get("cumulative_return"),
        "CVaR_loss_5": candidate.get("CVaR_loss_5"),
        "max_drawdown_loss": candidate.get("max_drawdown_loss"),
        "average_turnover": candidate.get("average_turnover"),
        "average_cost_per_step": candidate.get("average_cost_per_step"),
        "total_transaction_cost": candidate.get("total_transaction_cost"),
        "validation_utility": candidate.get("validation_utility"),
        "failed_trial_rate": failed_trial_rate,
        "finite_artifact_rate": finite_artifact_rate,
        "condition_failed_trial_rate_le_20pct": bool(failed_rate_ok),
        "condition_finite_artifact_rate_1": bool(finite_rate_ok),
        "condition_return_ge_eiie_minus_1pct": bool(return_ge_eiie),
        "condition_cvar_le_eiie": bool(cvar_le_eiie),
        "condition_mdd_le_eiie": bool(mdd_le_eiie),
        "condition_turnover_le_ppo": bool(turnover_le_ppo),
        "condition_average_cost_per_step_budget": bool(cost_budget_ok),
        "condition_cage_utility_or_pareto": bool(cage_alternative_ok),
        "cage_risk_cost_improvement_count": cage_improvement_count,
        "promotion_gate_passed": bool(passed),
        "blocking_reason": "" if passed else "P16 validation promotion conditions not met",
        "model_extension_id": MODEL_EXTENSION_ID,
    }


def _candidate_comparison_from_pilot(pilot: Path, model_names: Sequence[str]) -> pd.DataFrame:
    daily_returns = _read_csv(pilot / "metrics" / "hpo_model_final_daily_returns.csv")
    daily_turnover = _read_csv(pilot / "metrics" / "hpo_model_final_daily_turnover.csv")
    daily_costs = _read_csv(pilot / "metrics" / "hpo_model_final_daily_costs.csv")
    comparison = _read_csv(pilot / "metrics" / "hpo_model_final_comparison.csv")
    return validation_comparison(
        comparison,
        daily_returns=daily_returns,
        daily_turnover=daily_turnover,
        daily_costs=daily_costs,
        model_names=model_names,
    )


def _trial_stats(path: Path) -> dict[str, dict[str, float]]:
    frame = _read_csv(path)
    if frame.empty or "model_name" not in frame.columns:
        return {}
    result: dict[str, dict[str, float]] = {}
    for model_name, group in frame.groupby(frame["model_name"].astype(str), sort=False):
        states = group.get("state", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
        total = len(group)
        failed = int(states.isin({"fail", "failed"}).sum())
        completed = int(states.isin({"complete", "completed"}).sum())
        result[str(model_name)] = {
            "failed_trial_rate": float(failed / total) if total else np.nan,
            "finite_artifact_rate": 1.0 if completed > 0 else 0.0,
        }
    return result


def _records(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "model_name" not in frame.columns:
        return {}
    return {str(row["model_name"]): row for row in frame.to_dict("records")}


def _num(record: Mapping[str, Any], key: str) -> float:
    try:
        return float(record.get(key, np.nan))
    except (TypeError, ValueError):
        return np.nan


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate P16 validation promotion gate.")
    parser.add_argument("--pilot-run-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--average-cost-per-step-budget", type=float, default=0.001)
    parser.add_argument("--cost-budget-tolerance", type=float, default=1.0e-6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    return evaluate_promotion_gate(
        pilot_run_dir=args.pilot_run_dir,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        configured_average_cost_per_step_budget=args.average_cost_per_step_budget,
        cost_budget_tolerance=args.cost_budget_tolerance,
    )


if __name__ == "__main__":
    main()
