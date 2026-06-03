from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigLoader
from src.experiments.registry import ExperimentRegistry, HPOExperiment
from src.experiments.run_experiment import (
    _HPOTrialFailure,
    _activity_trial_failure_reason,
    _apply_orchestration_metadata,
    _assert_completed_result,
    _attach_single_model_hpo_final_outputs,
    _best_hpo_model_payload,
    _config_for_output_snapshot,
    _config_path,
    _create_run_dir,
    _hpo_int,
    _hpo_model_final_comparison,
    _hpo_model_final_frame,
    _hpo_model_final_reports,
    _hpo_model_summary,
    _hpo_pruner_int,
    _hpo_trainable_models,
    _mapping,
    _read_hpo_trials_csv,
    _refresh_new_model_artifacts,
    _registry_path,
    _result_mapping,
    _run_hpo_final_reports,
    _run_hpo_trial,
    _run_hpo_single,
    _safe_hpo_model_key,
    _selected_hpo_report_trials,
    _utc_now,
    _write_best_trial_config_snapshot,
    _write_experiment_outputs,
    _write_hpo_trials_csv,
    _write_run_manifest,
)
from src.utils.device import get_device
from src.utils.logger import mark_run_failed, mark_run_status, save_json_atomic, save_yaml_atomic, write_run_outputs
from src.utils.seed import set_global_seed


FINISHED = "finished"
MISSING = "missing"
INTERRUPTED = "interrupted"
FAILED_NO_COMPLETE = "failed_no_complete"


def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover or rerun interrupted EXP05 P7 formal seed runs.")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", dest="config", help="Path to experiment config YAML.")
    config_group.add_argument("--experiment", dest="experiment", help="Alias for --config.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _trial_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def _model_state(model_dir: Path) -> str:
    frame = _trial_rows(model_dir / "logs" / "hpo_trials.csv")
    if frame.empty:
        return MISSING
    if len(frame) < 50:
        return INTERRUPTED
    complete_count = int(frame["state"].astype(str).eq("complete").sum()) if "state" in frame.columns else 0
    if complete_count <= 0:
        return FAILED_NO_COMPLETE
    return FINISHED


def _archive_dir(path: Path) -> Path:
    suffix = 1
    target = path.with_name(f"{path.name}.interrupted")
    while target.exists():
        suffix += 1
        target = path.with_name(f"{path.name}.interrupted_{suffix}")
    path.rename(target)
    return target


def _model_experiment(root_experiment: HPOExperiment, model_name: str) -> tuple[HPOExperiment, dict[str, Any], Path]:
    config = root_experiment.config
    run_dir = root_experiment.context.run_dir
    if run_dir is None:
        raise RuntimeError("ERR_EXPERIMENT_OUTPUT_DIR_MISSING")
    model_key = _safe_hpo_model_key(model_name)
    model_config = deepcopy(dict(config))
    model_config.setdefault("output", {})
    model_config["output"]["run_name"] = f"{config['output']['run_name']}_{model_key}"
    model_config.setdefault("hpo", {})
    model_config["hpo"]["equal_budget_across_models"] = False
    model_config["hpo"]["study_name"] = str(
        model_config["hpo"].get("study_name") or f"{config['output']['run_name']}_hpo"
    ) + f"_{model_key}"
    model_dir = run_dir / f"hpo_{model_key}"
    model_context = replace(root_experiment.context, config=model_config, run_dir=model_dir)
    model_experiment = HPOExperiment(
        model_context,
        root_experiment.experiment_type,
        root_experiment.output_name,
        hpo_enabled=root_experiment.hpo_enabled,
    )
    setattr(model_experiment, "active_model_name", model_name)
    if hasattr(root_experiment, "active_split"):
        setattr(model_experiment, "active_split", getattr(root_experiment, "active_split"))
    return model_experiment, model_config, model_dir


def _complete_trials(frame: pd.DataFrame) -> list[SimpleNamespace]:
    if frame.empty:
        return []
    complete = frame.loc[frame["state"].astype(str).eq("complete")].copy()
    trials: list[SimpleNamespace] = []
    for row in complete.itertuples(index=False):
        params_json = getattr(row, "params_json", "{}")
        params = json.loads(params_json) if isinstance(params_json, str) and params_json else {}
        objective_value = float(getattr(row, "objective_value"))
        trial_number = int(getattr(row, "trial_number"))
        trials.append(SimpleNamespace(number=trial_number, value=objective_value, params=dict(params)))
    return trials


def _best_trial(trials: Sequence[SimpleNamespace], direction: str) -> SimpleNamespace:
    reverse = str(direction).lower() != "minimize"
    return sorted(trials, key=lambda item: float(item.value), reverse=reverse)[0]


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


def _resume_or_run_interrupted_model_payload(root_experiment: HPOExperiment, model_name: str) -> dict[str, Any]:
    model_experiment, model_config, model_dir = _model_experiment(root_experiment, model_name)
    hpo_cfg = _mapping(model_config.get("hpo"))
    seed = _hpo_int(hpo_cfg.get("seed"), model_config["reproducibility"]["seed"])
    n_trials = _hpo_int(hpo_cfg.get("n_trials"), hpo_cfg.get("n_trials_per_model", 1))
    timeout = hpo_cfg.get("timeout")
    if timeout is None:
        timeout = hpo_cfg.get("timeout_per_model_seconds")
    direction = str(hpo_cfg.get("direction") or "maximize")
    metric = str(hpo_cfg.get("metric") or hpo_cfg.get("objective") or "validation_metric")
    validation_split = str(hpo_cfg.get("selection_split") or "validation")
    final_split = str(hpo_cfg.get("final_report_split") or "test")
    study_name = str(hpo_cfg.get("study_name") or f"{root_experiment.config['output']['run_name']}_hpo_{_safe_hpo_model_key(model_name)}")
    trials_path = model_dir / "logs" / "hpo_trials.csv"

    existing_frame = _trial_rows(trials_path)
    existing_rows = _existing_trial_records(existing_frame)
    distributions = _search_space_distributions(model_config, model_name)
    recovered_trial_count = 0

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=_hpo_pruner_int(hpo_cfg, "warmup_trials", "n_startup_trials", "pruner_warmup_trials"),
        n_warmup_steps=_hpo_pruner_int(hpo_cfg, "warmup_steps", "n_warmup_steps", "pruner_warmup_steps"),
    )
    study_kwargs = {
        "study_name": study_name,
        "direction": direction,
        "sampler": sampler,
        "pruner": pruner,
    }
    storage = hpo_cfg.get("storage")
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

    trial_rows = list(existing_rows)
    remaining_trials = max(0, n_trials - len(existing_rows))

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
            trial_result = _result_mapping(_run_hpo_trial(model_experiment, trial, "train", validation_split))
            validation_metric = float(trial_result.get(metric, trial_result.get("validation_metric")))
            objective_value = float(trial_result.get("objective_value", validation_metric))
            activity_failure = _activity_trial_failure_reason(trial_result, model_config)
            if activity_failure:
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
        raise RuntimeError(f"ERR_P7_RECOVERY_NO_COMPLETED_TRIAL: {model_name}")

    best_trial = study.best_trial
    _write_best_trial_config_snapshot(model_experiment, best_trial, model_dir)
    final_reports = _run_hpo_final_reports(model_experiment, complete_trials, direction, final_split)
    final_result = dict(final_reports["best"]["result"])
    if str(final_result.get("status", "unknown")) != "completed":
        raise RuntimeError(f"ERR_HPO_FINAL_RESULT_NOT_COMPLETED: {model_name}")

    payload = dict(final_result)
    payload.update(
        {
            "status": "completed",
            "experiment_type": model_experiment.experiment_type,
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
                    "result": report.get("result", {}),
                }
                for label, report in final_reports.items()
            ],
            "final_result": _result_mapping(final_result),
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
    _refresh_new_model_artifacts(payload, model_config)
    _attach_single_model_hpo_final_outputs(payload)
    return payload


def _recover_completed_model_payload(root_experiment: HPOExperiment, model_name: str) -> dict[str, Any]:
    model_experiment, model_config, model_dir = _model_experiment(root_experiment, model_name)
    trials_path = model_dir / "logs" / "hpo_trials.csv"
    trial_frame = _trial_rows(trials_path)
    complete_trials = _complete_trials(trial_frame)
    if not complete_trials:
        raise RuntimeError(f"ERR_P7_RECOVERY_NO_COMPLETED_TRIAL: {model_name}")
    hpo_cfg = _mapping(model_config.get("hpo"))
    direction = str(hpo_cfg.get("direction") or "maximize")
    metric = str(hpo_cfg.get("metric") or hpo_cfg.get("objective") or "validation_metric")
    validation_split = str(hpo_cfg.get("selection_split") or "validation")
    final_split = str(hpo_cfg.get("final_report_split") or "test")
    study_name = (
        str(trial_frame["study_name"].dropna().iloc[0])
        if "study_name" in trial_frame.columns and not trial_frame["study_name"].dropna().empty
        else str(hpo_cfg.get("study_name") or f"{root_experiment.config['output']['run_name']}_hpo_{_safe_hpo_model_key(model_name)}")
    )
    best_trial = _best_trial(complete_trials, direction)
    _write_best_trial_config_snapshot(model_experiment, best_trial, model_dir)
    final_reports = _run_hpo_final_reports(model_experiment, complete_trials, direction, final_split)
    final_result = dict(final_reports["best"]["result"])
    if str(final_result.get("status", "unknown")) != "completed":
        raise RuntimeError(f"ERR_HPO_FINAL_RESULT_NOT_COMPLETED: {model_name}")
    payload = dict(final_result)
    payload.update(
        {
            "status": "completed",
            "experiment_type": model_experiment.experiment_type,
            "study_name": study_name,
            "direction": direction,
            "metric": metric,
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_params": dict(best_trial.params),
            "trial_count": len(trial_frame),
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
                    "result": report.get("result", {}),
                }
                for label, report in final_reports.items()
            ],
            "final_result": _result_mapping(final_result),
            "hpo_model_name": model_name,
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
        trial_count=len(trial_frame),
        selection_split=validation_split,
    )
    _refresh_new_model_artifacts(payload, model_config)
    _attach_single_model_hpo_final_outputs(payload)
    return payload


def _failed_model_payload(
    *,
    root_experiment: HPOExperiment,
    model_name: str,
    trial_frame: pd.DataFrame,
    reason: str,
    recovery_mode: str,
    recovered_trial_count: int | None = None,
    remaining_trials_started: int | None = None,
) -> dict[str, Any]:
    model_experiment, model_config, model_dir = _model_experiment(root_experiment, model_name)
    hpo_cfg = _mapping(model_config.get("hpo"))
    study_name = (
        str(trial_frame["study_name"].dropna().iloc[0])
        if "study_name" in trial_frame.columns and not trial_frame["study_name"].dropna().empty
        else str(hpo_cfg.get("study_name") or f"{root_experiment.config['output']['run_name']}_hpo_{_safe_hpo_model_key(model_name)}")
    )
    payload: dict[str, Any] = {
        "status": "failed",
        "experiment_type": model_experiment.experiment_type,
        "study_name": study_name,
        "direction": str(hpo_cfg.get("direction") or "maximize"),
        "metric": str(hpo_cfg.get("metric") or hpo_cfg.get("objective") or "validation_metric"),
        "best_trial_number": None,
        "best_value": float("nan"),
        "best_params": {},
        "trial_count": len(trial_frame),
        "selection_split": str(hpo_cfg.get("selection_split") or "validation"),
        "final_report_split": str(hpo_cfg.get("final_report_split") or "test"),
        "hpo_trials_path": str(model_dir / "logs" / "hpo_trials.csv"),
        "hpo_trials": trial_frame.copy(),
        "hpo_final_reports": [],
        "hpo_model_name": model_name,
        "rankable_in_unified_table": False,
        "reason": reason,
        "fail_reason": reason,
        "recovery_metadata": {
            "mode": recovery_mode,
            "recovered_trial_rows": recovered_trial_count if recovered_trial_count is not None else len(trial_frame),
            "remaining_trials_started": remaining_trials_started if remaining_trials_started is not None else 0,
        },
    }
    return payload


def _aggregate_root_result(
    *,
    root_experiment: HPOExperiment,
    payloads: Sequence[Mapping[str, Any]],
    recovered_models: Sequence[str],
    rerun_models: Sequence[str],
) -> dict[str, Any]:
    if not payloads:
        raise RuntimeError("ERR_P7_RECOVERY_EMPTY_PAYLOADS")
    direction = str(_mapping(root_experiment.config.get("hpo")).get("direction") or "maximize")
    best_payload = _best_hpo_model_payload(payloads, direction)
    result = dict(best_payload)
    result["status"] = "completed"
    result["hpo_mode"] = "equal_budget_across_models"
    result["best_model_name"] = best_payload.get("hpo_model_name")
    result["trainable_model_count"] = len(payloads)
    result["hpo_model_results"] = [_hpo_model_summary(payload) for payload in payloads]
    result["hpo_model_final_comparison"] = _hpo_model_final_comparison(payloads)
    result["hpo_model_final_reports"] = _hpo_model_final_reports(payloads)
    for frame_name in ("daily_returns", "daily_weights", "daily_turnover", "daily_rebalance", "daily_costs"):
        final_frame = _hpo_model_final_frame(payloads, frame_name)
        if not final_frame.empty or len(final_frame.columns) > 0:
            result[f"hpo_model_final_{frame_name}"] = final_frame
    final_diagnostics = _hpo_model_final_frame(payloads, "baseline_daily_diagnostics")
    if not final_diagnostics.empty or len(final_diagnostics.columns) > 0:
        result["hpo_model_final_daily_diagnostics"] = final_diagnostics
    result["hpo_trials"] = pd.concat(
        [payload["hpo_trials"].assign(model_name=str(payload.get("hpo_model_name"))) for payload in payloads],
        ignore_index=True,
    )
    result["recovery_metadata"] = {
        "mode": "partial_equal_budget_resume",
        "recovered_models": list(recovered_models),
        "rerun_models": list(rerun_models),
        "failed_models": [
            str(payload.get("hpo_model_name") or payload.get("model_name") or "")
            for payload in payloads
            if str(payload.get("status", "completed")) != "completed"
        ],
    }
    return result


def recover_or_run(root_experiment: HPOExperiment, *, dry_run: bool = False) -> dict[str, Any]:
    run_dir = root_experiment.context.run_dir
    if run_dir is None:
        raise RuntimeError("ERR_EXPERIMENT_OUTPUT_DIR_MISSING")
    model_names = _hpo_trainable_models(root_experiment.config)
    recovered_models: list[str] = []
    rerun_models: list[str] = []
    rerun_model_states: dict[str, str] = {}
    payloads: list[dict[str, Any]] = []

    for index, model_name in enumerate(model_names):
        model_dir = run_dir / f"hpo_{_safe_hpo_model_key(model_name)}"
        state = _model_state(model_dir)
        if state == FINISHED:
            recovered_models.append(model_name)
            continue
        rerun_models = list(model_names[index:])
        rerun_model_states[model_name] = state
        break

    for model_name in rerun_models[1:]:
        model_dir = run_dir / f"hpo_{_safe_hpo_model_key(model_name)}"
        rerun_model_states[model_name] = _model_state(model_dir)

    if dry_run:
        return {
            "status": "completed",
            "recovery_plan": {
                "recovered_models": recovered_models,
                "rerun_models": rerun_models,
                "rerun_model_states": rerun_model_states,
            },
        }

    for model_name in recovered_models:
        payloads.append(_recover_completed_model_payload(root_experiment, model_name))

    for model_name in rerun_models:
        model_experiment, _model_config, model_dir = _model_experiment(root_experiment, model_name)
        state = rerun_model_states.get(model_name, _model_state(model_dir))
        if state == INTERRUPTED:
            try:
                payload = dict(_resume_or_run_interrupted_model_payload(root_experiment, model_name))
            except RuntimeError as exc:
                if "ERR_P7_RECOVERY_NO_COMPLETED_TRIAL" not in str(exc):
                    raise
                trial_frame = _trial_rows(model_dir / "logs" / "hpo_trials.csv")
                reason = next(
                    (
                        str(value)
                        for value in reversed(trial_frame.get("fail_reason", pd.Series(dtype=str)).tolist())
                        if str(value).strip()
                    ),
                    str(exc),
                )
                payload = _failed_model_payload(
                    root_experiment=root_experiment,
                    model_name=model_name,
                    trial_frame=trial_frame,
                    reason=reason,
                    recovery_mode="single_model_resume_all_fail",
                    recovered_trial_count=len(trial_frame),
                    remaining_trials_started=0,
                )
        elif state == FAILED_NO_COMPLETE:
            trial_frame = _trial_rows(model_dir / "logs" / "hpo_trials.csv")
            reason = next(
                (
                    str(value)
                    for value in reversed(trial_frame.get("fail_reason", pd.Series(dtype=str)).tolist())
                    if str(value).strip()
                ),
                "ERR_P7_RECOVERY_NO_COMPLETED_TRIAL",
            )
            payload = _failed_model_payload(
                root_experiment=root_experiment,
                model_name=model_name,
                trial_frame=trial_frame,
                reason=reason,
                recovery_mode="single_model_preserve_all_fail",
            )
        else:
            if model_dir.exists():
                _archive_dir(model_dir)
            try:
                payload = dict(_run_hpo_single(model_experiment))
            except RuntimeError as exc:
                if "ERR_HPO_NO_COMPLETED_TRIAL" not in str(exc):
                    raise
                trial_frame = _trial_rows(model_dir / "logs" / "hpo_trials.csv")
                reason = next(
                    (
                        str(value)
                        for value in reversed(trial_frame.get("fail_reason", pd.Series(dtype=str)).tolist())
                        if str(value).strip()
                    ),
                    str(exc),
                )
                payload = _failed_model_payload(
                    root_experiment=root_experiment,
                    model_name=model_name,
                    trial_frame=trial_frame,
                    reason=reason,
                    recovery_mode="single_model_fresh_all_fail",
                )
        payload["hpo_model_name"] = model_name
        payloads.append(payload)

    result = _aggregate_root_result(
        root_experiment=root_experiment,
        payloads=payloads,
        recovered_models=recovered_models,
        rerun_models=rerun_models,
    )
    _write_hpo_trials_csv(result["hpo_trials"].to_dict("records"), run_dir / "logs" / "hpo_trials.csv")
    return result


def main(argv: Sequence[str] | None = None) -> Path | None:
    args = _parse_cli(argv)
    config_path = _config_path(args)
    config = ConfigLoader.load(config_path, cli_overrides=args)
    set_global_seed(config["reproducibility"]["seed"], config["reproducibility"])
    device = get_device(config["device"])
    run_dir = _create_run_dir(config)
    if _root_ready(run_dir):
        if args.dry_run:
            print(json.dumps({"already_ready": True, "recovered_models": [], "rerun_models": []}, ensure_ascii=False, sort_keys=True))
            return None
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
            raise RuntimeError("ERR_P7_RECOVERY_NOT_HPO_EXPERIMENT")
        result = recover_or_run(experiment, dry_run=args.dry_run)
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
            command=" ".join(["python", "scripts/recover_exp05_p7_formal_seed.py", *map(str, argv or sys.argv[1:])]),
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
        save_json_atomic(failure_state, run_dir / "logs" / "p7_recovery_failure.json")
        _write_run_manifest(config, experiment, run_dir, "failed", failure_state=failure_state, device=device)
        mark_run_failed(failure_state, registry_path, run_id)
        raise

    mark_run_status("success", registry_path, run_id, {"manifest_path": str(artifacts["run_manifest"])})
    return run_dir


if __name__ == "__main__":
    main(sys.argv[1:])
