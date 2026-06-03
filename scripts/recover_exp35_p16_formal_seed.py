from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigLoader
from src.experiments.registry import ExperimentRegistry, HPOExperiment, _config_with_params
from src.experiments.run_experiment import (
    _HPOTrialFailure,
    _activity_audit_values,
    _activity_hpo_trial_hard_fail_enabled,
    _activity_trial_failure_reason,
    _apply_orchestration_metadata,
    _assert_completed_result,
    _attach_single_model_hpo_final_outputs,
    _config_for_output_snapshot,
    _config_path,
    _create_run_dir,
    _hpo_int,
    _hpo_manifest_model_names,
    _hpo_model_final_comparison,
    _hpo_model_final_reports,
    _hpo_pruner_int,
    _mapping,
    _read_hpo_trials_csv,
    _refresh_new_model_artifacts,
    _registry_path,
    _result_mapping,
    _result_summary,
    _run_hpo_final_reports,
    _run_hpo_trial,
    _safe_hpo_model_key,
    _selected_hpo_report_trials,
    _utc_now,
    _write_best_trial_config_snapshot,
    _write_experiment_outputs,
    _write_hpo_search_space_manifest,
    _write_hpo_trials_csv,
    _write_run_manifest,
)
from src.utils.device import get_device
from src.utils.logger import mark_run_failed, mark_run_status, save_json_atomic, save_yaml_atomic, write_run_outputs
from src.utils.seed import set_global_seed


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover or resume interrupted EXP35 P16 formal seed runs.")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", dest="config", help="Path to experiment config YAML.")
    config_group.add_argument("--experiment", dest="experiment", help="Alias for --config.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _root_ready(run_dir: Path) -> bool:
    manifest_path = run_dir / "logs" / "run_manifest.json"
    comparison_path = run_dir / "metrics" / "hpo_model_final_comparison.csv"
    if not manifest_path.exists() or not comparison_path.exists():
        return False
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return (
        isinstance(manifest, dict)
        and manifest.get("status") == "success"
        and manifest.get("diagnostic_status") == "formal"
        and manifest.get("rankable_in_unified_table") is True
    )


def _active_model_name(experiment: HPOExperiment) -> str:
    model_names = _hpo_manifest_model_names(experiment.config, experiment)
    if len(model_names) != 1:
        raise RuntimeError(f"ERR_P16_RECOVERY_EXPECTS_SINGLE_MODEL: {model_names}")
    model_name = str(model_names[0])
    setattr(experiment, "active_model_name", model_name)
    return model_name


def _trial_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _search_space_distributions(config: Mapping[str, Any], model_name: str) -> dict[str, optuna.distributions.BaseDistribution]:
    hpo_config = _mapping(config.get("hpo"))
    search_space = _mapping(hpo_config.get("search_space"))
    distributions: dict[str, optuna.distributions.BaseDistribution] = {}
    for raw_name, raw_spec in search_space.items():
        name = str(raw_name)
        spec = _mapping(raw_spec)
        model_scope = spec.get("models") or spec.get("model_names")
        if isinstance(model_scope, Sequence) and not isinstance(model_scope, (str, bytes)):
            scoped = {str(item) for item in model_scope}
            if model_name not in scoped:
                continue
        if "choices" in spec:
            distributions[name] = optuna.distributions.CategoricalDistribution(list(spec.get("choices") or []))
            continue
        param_type = str(spec.get("type", "float"))
        if param_type == "int":
            distributions[name] = optuna.distributions.IntDistribution(
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
            continue
        distributions[name] = optuna.distributions.FloatDistribution(
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    return distributions


def _trial_params(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("params_json")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, str) and not raw.strip():
        return {}
    parsed = json.loads(str(raw))
    return parsed if isinstance(parsed, Mapping) else {}


def _trial_state(row: Mapping[str, Any]) -> optuna.trial.TrialState | None:
    raw = str(row.get("state") or "").strip().lower()
    if raw == "complete":
        return optuna.trial.TrialState.COMPLETE
    if raw == "fail":
        return optuna.trial.TrialState.FAIL
    if raw == "pruned":
        return optuna.trial.TrialState.PRUNED
    return None


def _frozen_trial(
    row: Mapping[str, Any],
    distributions: Mapping[str, optuna.distributions.BaseDistribution],
) -> optuna.trial.FrozenTrial | None:
    state = _trial_state(row)
    if state is None:
        return None
    params = _trial_params(row)
    trial_distributions = {name: dist for name, dist in distributions.items() if name in params}
    kwargs: dict[str, Any] = {
        "state": state,
        "params": params,
        "distributions": trial_distributions,
        "user_attrs": {
            "train_start": row.get("train_start"),
            "train_end": row.get("train_end"),
            "duration_sec": row.get("duration_sec"),
            "fail_reason": row.get("fail_reason"),
        },
    }
    if state == optuna.trial.TrialState.COMPLETE:
        kwargs["value"] = float(row["objective_value"])
    elif state == optuna.trial.TrialState.PRUNED:
        step = row.get("pruned_step")
        value = row.get("validation_metric")
        if step not in ("", None) and not pd.isna(step) and value not in ("", None) and not pd.isna(value):
            kwargs["intermediate_values"] = {int(step): float(value)}
    return optuna.trial.create_trial(**kwargs)


def _existing_trial_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.fillna("").to_dict("records")
    return [{str(key): value for key, value in record.items()} for record in records]


def _resume_or_run_single(experiment: HPOExperiment, *, dry_run: bool = False) -> dict[str, Any]:
    config = experiment.config
    hpo_config = _mapping(config.get("hpo"))
    run_dir = experiment.context.run_dir
    if run_dir is None:
        raise RuntimeError("ERR_EXPERIMENT_OUTPUT_DIR_MISSING")

    model_name = _active_model_name(experiment)
    seed = _hpo_int(hpo_config.get("seed"), config["reproducibility"]["seed"])
    n_trials = _hpo_int(hpo_config.get("n_trials"), hpo_config.get("n_trials_per_model", 1))
    timeout = hpo_config.get("timeout")
    if timeout is None:
        timeout = hpo_config.get("timeout_per_model_seconds")
    direction = str(hpo_config.get("direction") or "maximize")
    metric = str(hpo_config.get("metric") or hpo_config.get("objective") or "validation_metric")
    study_name = str(hpo_config.get("study_name") or f"{config['output']['run_name']}_hpo")
    storage = hpo_config.get("storage")
    train_split = "train"
    validation_split = str(hpo_config.get("selection_split") or "validation")
    final_split = str(hpo_config.get("final_report_split") or "test")
    trials_path = run_dir / "logs" / "hpo_trials.csv"

    _write_hpo_search_space_manifest(config, _hpo_manifest_model_names(config, experiment), run_dir / "logs" / "hpo_search_space_manifest.csv")

    existing_frame = _trial_rows(trials_path)
    existing_rows = _existing_trial_records(existing_frame)
    distributions = _search_space_distributions(config, model_name)
    recovered_trial_count = 0

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=_hpo_pruner_int(hpo_config, "warmup_trials", "n_startup_trials", "pruner_warmup_trials"),
        n_warmup_steps=_hpo_pruner_int(hpo_config, "warmup_steps", "n_warmup_steps", "pruner_warmup_steps"),
    )
    study_kwargs = {
        "study_name": study_name,
        "direction": direction,
        "sampler": sampler,
        "pruner": pruner,
    }
    if storage:
        study_kwargs["storage"] = storage
        study_kwargs["load_if_exists"] = True
    study = optuna.create_study(**study_kwargs)

    if existing_rows and not storage:
        for record in existing_rows:
            frozen = _frozen_trial(record, distributions)
            if frozen is None:
                continue
            study.add_trial(frozen)
            recovered_trial_count += 1

    remaining_trials = max(0, n_trials - len(existing_rows))
    if dry_run:
        return {
            "status": "completed",
            "recovery_plan": {
                "recovered_trial_rows": len(existing_rows),
                "recovered_optuna_trials": recovered_trial_count,
                "remaining_trials": remaining_trials,
                "study_name": study_name,
                "model_name": model_name,
            },
        }

    trial_rows = list(existing_rows)

    def objective(trial: optuna.Trial) -> float:
        train_start_clock = time.perf_counter()
        row: dict[str, Any] = {
            "model_name": model_name,
            "fold_id": "",
            "study_name": study_name,
            "trial_number": trial.number,
            "seed": seed,
            "params_json": "{}",
            "state": "",
            "objective_value": "",
            "validation_metric": "",
            "train_start": _utc_now(),
            "train_end": "",
            "duration_sec": "",
            "pruned_step": "",
            "fail_reason": "",
        }
        try:
            trial_result = _result_mapping(_run_hpo_trial(experiment, trial, train_split, validation_split))
            validation_metric = float(trial_result.get(metric, trial_result.get("validation_metric")))
            objective_value = float(trial_result.get("objective_value", validation_metric))
            row.update(_activity_audit_values(trial_result))
            activity_failure = _activity_trial_failure_reason(trial_result, config)
            row["activity_failure_reason"] = activity_failure or ""
            if activity_failure and _activity_hpo_trial_hard_fail_enabled(config):
                row["fail_reason"] = activity_failure
                raise _HPOTrialFailure(activity_failure)
            row["state"] = "complete"
            row["objective_value"] = objective_value
            row["validation_metric"] = validation_metric
            return objective_value
        except optuna.TrialPruned:
            row["state"] = "pruned"
            row["pruned_step"] = "" if trial.last_step is None else trial.last_step
            raise
        except Exception as exc:
            row["state"] = "fail"
            row["fail_reason"] = str(exc)
            if isinstance(exc, _HPOTrialFailure):
                raise
            raise _HPOTrialFailure(str(exc)) from exc
        finally:
            row["params_json"] = json.dumps(dict(trial.params), ensure_ascii=False, sort_keys=True)
            row["train_end"] = _utc_now()
            row["duration_sec"] = round(time.perf_counter() - train_start_clock, 6)
            trial_rows.append(row)
            _write_hpo_trials_csv(trial_rows, trials_path)

    if not trials_path.exists():
        _write_hpo_trials_csv(trial_rows, trials_path)
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, timeout=timeout, catch=(_HPOTrialFailure,))

    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        raise RuntimeError(f"ERR_HPO_NO_COMPLETED_TRIAL: {study_name}")

    best_trial = study.best_trial
    _write_best_trial_config_snapshot(experiment, best_trial, run_dir)
    final_reports = _run_hpo_final_reports(experiment, complete_trials, direction, final_split)
    final_result = dict(final_reports["best"]["result"])
    if str(final_result.get("status", "unknown")) != "completed":
        raise RuntimeError(f"ERR_HPO_FINAL_RESULT_NOT_COMPLETED: status={final_result.get('status', 'unknown')}")

    payload = dict(final_result)
    payload.update(
        {
            "status": "completed",
            "experiment_type": experiment.experiment_type,
            "study_name": study_name,
            "direction": direction,
            "metric": metric,
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_params": dict(best_trial.params),
            "trial_count": len(trial_rows),
            "selection_split": validation_split,
            "final_report_split": final_split,
            "hpo_trials_path": str(trials_path),
            "hpo_trials": _read_hpo_trials_csv(trials_path),
            "hpo_final_reports": [
                {
                    "rank_label": label,
                    "trial_number": report.get("trial_number"),
                    "validation_value": report.get("validation_value"),
                    "params": report.get("params", {}),
                    "result": _result_summary(_result_mapping(report.get("result", {}))),
                }
                for label, report in final_reports.items()
            ],
            "final_result": _result_summary(final_result),
            "hpo_model_name": model_name,
            "recovery_metadata": {
                "mode": "single_model_resume",
                "recovered_trial_rows": len(existing_rows),
                "recovered_optuna_trials": recovered_trial_count,
                "remaining_trials_started": remaining_trials,
                "selected_report_trials": {
                    label: int(getattr(trial, "number", -1))
                    for label, trial in _selected_hpo_report_trials(complete_trials, direction).items()
                },
            },
        }
    )
    from src.experiments.run_experiment import _hpo_final_report_table

    payload["hpo_final_reports_table"] = _hpo_final_report_table(
        final_reports,
        model_name=model_name,
        study_name=study_name,
        final_split=final_split,
        best_trial_number=best_trial.number,
        best_value=best_trial.value,
        trial_count=len(trial_rows),
        selection_split=validation_split,
    )
    _refresh_new_model_artifacts(payload, config)
    _attach_single_model_hpo_final_outputs(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> Path | None:
    args = _parse_cli(argv)
    config_path = _config_path(args)
    config = ConfigLoader.load(config_path, cli_overrides=args)
    set_global_seed(config["reproducibility"]["seed"], config["reproducibility"])
    device = get_device(config["device"])
    run_dir = _create_run_dir(config)

    if _root_ready(run_dir):
        if args.dry_run:
            print(json.dumps({"already_ready": True, "remaining_trials": 0}, ensure_ascii=False, sort_keys=True))
        return run_dir

    registry_path = _registry_path(config)
    run_id = str(config["output"]["run_name"])
    registry = ExperimentRegistry()
    experiment = None
    try:
        if not args.dry_run:
            save_yaml_atomic(config, run_dir / "logs" / "config_snapshot.yaml")
            mark_run_status("running", registry_path, run_id)
        experiment = registry.create_experiment(config=config, device=device, run_dir=run_dir)
        if not isinstance(experiment, HPOExperiment):
            raise RuntimeError("ERR_P16_RECOVERY_NOT_HPO_EXPERIMENT")
        result = _resume_or_run_single(experiment, dry_run=args.dry_run)
        if args.dry_run:
            print(json.dumps(result["recovery_plan"], ensure_ascii=False, sort_keys=True))
            return None
        _apply_orchestration_metadata(result, config)
        _assert_completed_result(result)
        result_path = _write_experiment_outputs(result, run_dir)
        output_config = _config_for_output_snapshot(config, result)
        artifacts = write_run_outputs(
            result,
            run_dir,
            config=output_config,
            config_path=config_path,
            command=" ".join(["python", "scripts/recover_exp35_p16_formal_seed.py", *map(str, argv or sys.argv[1:])]),
            manifest_overrides={
                "status": "success",
                "experiment_type": getattr(experiment, "experiment_type", config["experiment"]["type"]),
                "output_name": getattr(experiment, "output_name", None),
                "result_path": str(result_path),
            },
        )
    except Exception as exc:
        failure_state = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "experiment_type": getattr(experiment, "experiment_type", config["experiment"]["type"]),
        }
        save_json_atomic(failure_state, run_dir / "logs" / "p16_recovery_failure.json")
        _write_run_manifest(config, experiment, run_dir, "failed", failure_state=failure_state, device=device)
        mark_run_failed(failure_state, registry_path, run_id)
        raise

    mark_run_status("success", registry_path, run_id, {"manifest_path": str(artifacts["run_manifest"])})
    return run_dir


if __name__ == "__main__":
    main(sys.argv[1:])
