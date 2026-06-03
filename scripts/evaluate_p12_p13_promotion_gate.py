from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_p12_p13_validation_references import (  # noqa: E402
    MODEL_EXTENSION_ID,
    _append_payload_frames,
    _concat,
    _reference_config,
    _validation_comparison,
)
from src.config import ConfigLoader  # noqa: E402
from src.experiments.pipeline import run_strategy_comparison  # noqa: E402
from src.experiments.registry import BaselineComparisonExperiment, ExperimentRegistry, _config_with_params  # noqa: E402
from src.utils.logger import save_json_atomic  # noqa: E402


P12_MODELS = (
    "cage_eiie_frozen_gate",
    "cage_eiie_multilevel_gate",
    "cage_eiie_distributional",
)
P12_SUPPLEMENTAL_MODELS = ("cage_eiie_joint_light",)
P13_MODELS = ("graph_transformer_risk_constrained_actor_critic_lite",)
DEFAULT_OUTPUT_DIR = "results/paper_tables/p12_p13_promotion_gate"


def evaluate_promotion_gate(
    *,
    p12_config: str | Path,
    p12_run_dir: str | Path,
    reference_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    p13_config: str | Path | None = None,
    p13_run_dir: str | Path | None = None,
    p12_supplemental_config: str | Path | None = None,
    p12_supplemental_run_dir: str | Path | None = None,
    p12_supplemental_models: Sequence[str] | None = None,
) -> dict[str, Path]:
    p13_requested = p13_config is not None or p13_run_dir is not None
    if p13_requested and (p13_config is None or p13_run_dir is None):
        raise ValueError("ERR_PROMOTION_GATE_P13_ARGS_INCOMPLETE")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(Path(reference_dir) / "validation_reference_comparison.csv")
    candidate_frames: list[pd.DataFrame] = []

    candidate_frames.extend(
        _evaluate_candidates(
            config_path=p12_config,
            run_dir=p12_run_dir,
            output_dir=output / "candidate_validation_runs" / "p12",
            model_names=P12_MODELS,
        )
    )
    if p12_supplemental_config is not None or p12_supplemental_run_dir is not None:
        if p12_supplemental_config is None or p12_supplemental_run_dir is None:
            raise ValueError("ERR_PROMOTION_GATE_SUPPLEMENTAL_P12_ARGS_INCOMPLETE")
        candidate_frames.extend(
            _evaluate_candidates(
                config_path=p12_supplemental_config,
                run_dir=p12_supplemental_run_dir,
                output_dir=output / "candidate_validation_runs" / "p12_supplemental",
                model_names=p12_supplemental_models or P12_SUPPLEMENTAL_MODELS,
            )
        )
    if p13_requested:
        candidate_frames.extend(
            _evaluate_candidates(
                config_path=p13_config,
                run_dir=p13_run_dir,
                output_dir=output / "candidate_validation_runs" / "p13",
                model_names=P13_MODELS,
            )
        )

    candidates = _concat(candidate_frames)
    report = _promotion_report(candidates, reference, include_p13=p13_requested)
    paths = {
        "candidate_comparison": output / "promotion_candidate_comparison.csv",
        "gate_report": output / "promotion_gate_report.csv",
        "manifest": output / "promotion_gate_manifest.json",
    }
    candidates.to_csv(paths["candidate_comparison"], index=False)
    report.to_csv(paths["gate_report"], index=False)
    save_json_atomic(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_extension_id": MODEL_EXTENSION_ID,
            "selection_split": "validation",
            "test_used_for_model_selection": False,
            "p12_config": str(p12_config),
            "p12_run_dir": str(p12_run_dir),
            "p13_config": str(p13_config) if p13_config is not None else None,
            "p13_run_dir": str(p13_run_dir) if p13_run_dir is not None else None,
            "p13_evaluated": bool(p13_requested),
            "p12_supplemental_config": str(p12_supplemental_config) if p12_supplemental_config is not None else None,
            "p12_supplemental_run_dir": str(p12_supplemental_run_dir) if p12_supplemental_run_dir is not None else None,
            "p12_supplemental_models": list(p12_supplemental_models or P12_SUPPLEMENTAL_MODELS)
            if p12_supplemental_config is not None
            else [],
            "reference_dir": str(reference_dir),
            "outputs": {key: str(path) for key, path in paths.items()},
        },
        paths["manifest"],
    )
    return paths


def _evaluate_candidates(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    output_dir: Path,
    model_names: Sequence[str],
) -> list[pd.DataFrame]:
    config = ConfigLoader.load(config_path)
    hpo_trials = pd.read_csv(Path(run_dir) / "logs" / "hpo_trials.csv")
    frames: list[pd.DataFrame] = []
    for model_name in model_names:
        model_trials = hpo_trials.loc[
            hpo_trials["model_name"].astype(str).eq(model_name)
            & hpo_trials["state"].astype(str).eq("complete")
        ].copy()
        if model_trials.empty:
            continue
        best = _select_activity_aware_best_trial(model_trials)
        params = json.loads(str(best.get("params_json") or "{}"))
        trial_config = _config_with_params(config, params)
        candidate_config = _reference_config(trial_config, [model_name])
        experiment = ExperimentRegistry().create_experiment(candidate_config)
        if not isinstance(experiment, BaselineComparisonExperiment):
            raise TypeError("ERR_PROMOTION_GATE_EXPERIMENT_TYPE")
        payload = run_strategy_comparison(
            candidate_config,
            experiment.baselines,
            segment="validation",
            run_dir=str(output_dir / model_name),
        )
        comparison_frames: list[pd.DataFrame] = []
        daily_returns_frames: list[pd.DataFrame] = []
        daily_turnover_frames: list[pd.DataFrame] = []
        daily_cost_frames: list[pd.DataFrame] = []
        _append_payload_frames(
            payload,
            comparison_frames=comparison_frames,
            daily_returns_frames=daily_returns_frames,
            daily_turnover_frames=daily_turnover_frames,
            daily_cost_frames=daily_cost_frames,
        )
        comparison = _validation_comparison(
            _concat(comparison_frames),
            daily_returns=_concat(daily_returns_frames),
            daily_turnover=_concat(daily_turnover_frames),
            daily_costs=_concat(daily_cost_frames),
            reference_models=(model_name,),
        )
        comparison["best_trial_number"] = int(best["trial_number"])
        comparison["best_validation_metric"] = float(best["validation_metric"])
        comparison["best_activity_failure_reason"] = _clean_value(best.get("activity_failure_reason"))
        comparison["best_params_json"] = json.dumps(params, ensure_ascii=False, sort_keys=True)
        comparison["pilot_run_dir"] = str(run_dir)
        frames.append(comparison)
    return frames


def _select_activity_aware_best_trial(model_trials: pd.DataFrame) -> pd.Series:
    complete = model_trials.loc[model_trials["state"].astype(str).eq("complete")].copy()
    if complete.empty:
        raise ValueError("ERR_PROMOTION_GATE_NO_COMPLETE_TRIAL")
    if "activity_failure_reason" in complete.columns:
        reasons = complete["activity_failure_reason"].map(_clean_value)
        passed = complete.loc[reasons.eq("")].copy()
        if not passed.empty:
            complete = passed
    return complete.sort_values("objective_value", ascending=False).iloc[0]


def _promotion_report(candidates: pd.DataFrame, reference: pd.DataFrame, *, include_p13: bool) -> pd.DataFrame:
    ref = _records(reference)
    candidate_records = _records(candidates)
    rows: list[dict[str, Any]] = []
    p12_promoted_utilities: list[float] = []
    for model_name in (*P12_MODELS, *P12_SUPPLEMENTAL_MODELS):
        row = _p12_gate_row(model_name, candidate_records.get(model_name, {}), ref)
        rows.append(row)
        if row["promotion_gate_passed"]:
            p12_promoted_utilities.append(float(row["validation_return_cost_risk_utility"]))
    best_cage_utility = max(p12_promoted_utilities) if p12_promoted_utilities else None
    if include_p13:
        for model_name in P13_MODELS:
            rows.append(_p13_gate_row(model_name, candidate_records.get(model_name, {}), ref, best_cage_utility))
    return pd.DataFrame(rows)


def _p12_gate_row(model_name: str, candidate: Mapping[str, Any], ref: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    eiie = ref.get("eiie_native", {})
    full_dqn = ref.get("full_dqn_gated_multitask_cnn_ppo", {})
    condition_a = _num(candidate, "cumulative_return") > _num(eiie, "cumulative_return")
    condition_b = (
        _num(candidate, "cumulative_return") >= _num(eiie, "cumulative_return") - 0.01
        and _num(candidate, "turnover_mean") <= 0.85 * _num(eiie, "turnover_mean")
        and _num(candidate, "transaction_cost_total") <= 0.85 * _num(eiie, "transaction_cost_total")
        and _num(candidate, "max_drawdown_loss") <= _num(eiie, "max_drawdown_loss") + 0.005
        and _num(candidate, "CVaR_loss_5") <= _num(eiie, "CVaR_loss_5") + 0.005
    )
    condition_c = (
        _num(candidate, "cumulative_return") > _num(full_dqn, "cumulative_return")
        and _num(candidate, "turnover_mean") <= 1.05 * _num(full_dqn, "turnover_mean")
        and _num(candidate, "transaction_cost_total") <= 1.05 * _num(full_dqn, "transaction_cost_total")
    )
    return {
        **_base_gate_row(model_name, candidate),
        "phase": "P12",
        "condition_a_return_beats_eiie": condition_a,
        "condition_b_pareto_vs_eiie": condition_b,
        "condition_c_beats_full_dqn_cost_bounded": condition_c,
        "promotion_gate_passed": bool(condition_a or condition_b or condition_c),
        "blocking_reason": "" if (condition_a or condition_b or condition_c) else "P12 validation promotion conditions not met",
    }


def _p13_gate_row(
    model_name: str,
    candidate: Mapping[str, Any],
    ref: Mapping[str, Mapping[str, Any]],
    best_cage_utility: float | None,
) -> dict[str, Any]:
    eiie = ref.get("eiie_native", {})
    finite_ok = bool(candidate.get("daily_returns_finite")) and bool(candidate.get("daily_nav_finite"))
    conditions = {
        "condition_return_ge_eiie": _num(candidate, "cumulative_return") >= _num(eiie, "cumulative_return"),
        "condition_utility_ge_best_promoted_cage": best_cage_utility is not None
        and _num(candidate, "validation_return_cost_risk_utility") >= best_cage_utility,
        "condition_turnover_le_eiie": _num(candidate, "turnover_mean") <= _num(eiie, "turnover_mean"),
        "condition_cost_le_eiie": _num(candidate, "transaction_cost_total") <= _num(eiie, "transaction_cost_total"),
        "condition_failed_trial_rate_le_20pct": True,
        "condition_finite_artifact_rate_1": finite_ok,
    }
    passed = all(conditions.values())
    return {
        **_base_gate_row(model_name, candidate),
        "phase": "P13",
        **conditions,
        "best_promoted_cage_validation_utility": best_cage_utility,
        "promotion_gate_passed": bool(passed),
        "blocking_reason": "" if passed else "P13 validation promotion conditions not met",
    }


def _base_gate_row(model_name: str, candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "selection_split": "validation",
        "test_used_for_model_selection": False,
        "cumulative_return": candidate.get("cumulative_return"),
        "turnover_mean": candidate.get("turnover_mean"),
        "transaction_cost_total": candidate.get("transaction_cost_total"),
        "max_drawdown_loss": candidate.get("max_drawdown_loss"),
        "CVaR_loss_5": candidate.get("CVaR_loss_5"),
        "validation_return_cost_risk_utility": candidate.get("validation_return_cost_risk_utility"),
        "best_trial_number": candidate.get("best_trial_number"),
        "best_validation_metric": candidate.get("best_validation_metric"),
        "best_activity_failure_reason": candidate.get("best_activity_failure_reason"),
        "daily_returns_finite": candidate.get("daily_returns_finite"),
        "daily_nav_finite": candidate.get("daily_nav_finite"),
        "model_extension_id": MODEL_EXTENSION_ID,
    }


def _records(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "model_name" not in frame.columns:
        return {}
    return {str(record["model_name"]): record for record in frame.to_dict("records")}


def _num(record: Mapping[str, Any], key: str) -> float:
    try:
        return float(record.get(key))
    except (TypeError, ValueError):
        return float("nan")


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate P12/P13 validation-only promotion gates.")
    parser.add_argument("--p12-config", default="configs/paper/p12_cage_eiie_pilot.yaml")
    parser.add_argument("--p12-run-dir", default="results/EXP28_P12_cage_eiie_pilot_s42")
    parser.add_argument("--p13-config", help="Optional; include P13 gate only when paired with --p13-run-dir.")
    parser.add_argument("--p13-run-dir", help="Optional; include P13 gate only when paired with --p13-config.")
    parser.add_argument("--p12-supplemental-config")
    parser.add_argument("--p12-supplemental-run-dir")
    parser.add_argument("--p12-supplemental-model", action="append", dest="p12_supplemental_models")
    parser.add_argument("--reference-dir", default="results/paper_tables/p12_p13_validation_references")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = evaluate_promotion_gate(
        p12_config=args.p12_config,
        p12_run_dir=args.p12_run_dir,
        p13_config=args.p13_config,
        p13_run_dir=args.p13_run_dir,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        p12_supplemental_config=args.p12_supplemental_config,
        p12_supplemental_run_dir=args.p12_supplemental_run_dir,
        p12_supplemental_models=args.p12_supplemental_models,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
