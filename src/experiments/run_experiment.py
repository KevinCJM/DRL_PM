from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.config import ConfigError, ConfigLoader, PROJECT_ROOT, assert_path_allowed
from src.data.loader import DataContractError
from src.data.loader import load_market_dataset
from src.data.splits import create_split
from src.experiments.aggregate_results import aggregate_walk_forward
from src.experiments.registry import (
    ExperimentRegistry,
    HPOExperiment,
    HYBRID_DQN_OPTIMIZER_ALIAS,
    HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
    _expand_baseline_aliases,
)
from src.utils.device import get_device
from src.utils.logger import mark_run_failed, mark_run_status, save_json_atomic, save_yaml_atomic, write_run_outputs
from src.utils.seed import set_global_seed


VALID_RUN_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
FORBIDDEN_OUTPUT_DIRS = (
    Path("data/processed"),
    Path("data/metrics_factory"),
    Path("data/reports"),
)
PLATFORM_NATIVE_RL_FAMILY_ALIASES = frozenset(
    {
        "native_rl",
        "native_rl_reimplementation",
        "new_model_extension",
    }
)
PLATFORM_NATIVE_RL_MODEL_NAMES = frozenset(
    {
        "full_dqn_gated_multitask_cnn_ppo",
        "ppo_native",
        "cnn_ppo_native",
        "bernoulli_gated_ppo_native",
        "dqn_template_native",
        "eiie_native",
        "pgportfolio_eiie_native",
        "ppo_dqn_hierarchical_reimplementation",
        HYBRID_DQN_OPTIMIZER_ALIAS,
        *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
    }
)
HPO_ACTIVITY_AUDIT_COLUMNS = (
    "model_rebalance_hit_rate",
    "non_initial_turnover_per_opportunity",
    "average_turnover",
    "non_initial_rebalance_count",
    "raw_model_requested_rebalance_count",
    "model_chosen_hold_count",
)
HPO_TRIAL_COLUMNS = (
    "model_name",
    "fold_id",
    "study_name",
    "trial_number",
    "seed",
    "params_json",
    "state",
    "objective_value",
    "validation_metric",
    *HPO_ACTIVITY_AUDIT_COLUMNS,
    "activity_failure_reason",
    "train_start",
    "train_end",
    "duration_sec",
    "pruned_step",
    "fail_reason",
)
HPO_FINAL_REPORT_BASE_COLUMNS = (
    "model_name",
    "hpo_model_name",
    "study_name",
    "rank_label",
    "trial_number",
    "validation_value",
    "params_json",
    "final_report_split",
    "status",
    "best_trial_number",
    "best_value",
    "trial_count",
    "selection_split",
    "evaluated_checkpoint_path",
)
HPO_SEARCH_SPACE_MANIFEST_COLUMNS = (
    "model_name",
    "param_name",
    "param_type",
    "low",
    "high",
    "choices",
    "log_scale",
    "is_shared_across_models",
    "is_model_specific",
    "rationale",
)
NEW_MODEL_HPO_FINAL_FRAME_NAMES = (
    "gate_actions",
    "gate_action_summary",
    "cage_eiie_candidate_weights",
    "cage_final_weights",
    "turnover_cost_breakdown",
    "risk_metrics",
    "validation_selection_report",
    "ra_gt_rcpo_daily_diagnostics",
    "ra_gt_rcpo_constraint_multipliers",
    "ra_gt_rcpo_graph_diagnostics",
    "ra_gt_rcpo_actor_critic_training_history",
    "ra_gt_rcpo_risk_decomposition",
)
HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAME_SET = frozenset(HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES)
HYBRID_DQN_DIAGNOSTIC_RUN_MODES = {"smoke", "diagnostic"}
PROXY_HPO_MODEL_NAMES = {
    "ppo_proxy",
    "ppo_baseline",
    "cnn_ppo_proxy",
    "cnn_ppo_baseline",
    "bernoulli_gated_ppo_proxy",
    "bernoulli_gated_ppo",
    "dqn_template_proxy",
    "dqn_only",
    "eiie_proxy",
    "eiie",
}
NATIVE_HPO_MODEL_NAMES = {
    "full_dqn_gated_multitask_cnn_ppo",
    "ppo_native",
    "cnn_ppo_native",
    "bernoulli_gated_ppo_native",
    "dqn_template_native",
    "eiie_native",
    "pgportfolio_eiie_native",
    "ppo_dqn_hierarchical_reimplementation",
    "cage_eiie_frozen_gate",
    "cage_eiie_multilevel_gate",
    "cage_eiie_distributional",
    "cage_eiie_no_cvar",
    "cage_eiie_distributional_no_cvar",
    "cage_eiie_joint_light",
    "cage_eiie_fixed_rho_25",
    "cage_eiie_fixed_rho_50",
    "cage_eiie_fixed_rho_75",
    "graph_transformer_risk_constrained_actor_critic_lite",
    "gt_rcpo_lite",
    "risk_aware_graph_transformer_constrained_actor_critic",
    "ra_gt_rcpo_no_graph",
    "ra_gt_rcpo_no_transformer",
    "ra_gt_rcpo_no_cvar_constraint",
    "ra_gt_rcpo_no_cost_constraint",
    "ra_gt_rcpo_no_turnover_constraint",
    "ra_gt_rcpo_mlp_actor_critic",
    *HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES,
}
NON_NATIVE_HPO_MODEL_NAMES = PROXY_HPO_MODEL_NAMES | {
    "pgportfolio_original_external",
    "fixed_ratio",
    "equal_weight",
    "buy_and_hold",
    "markowitz",
    "traditional_markowitz_mean_variance",
    "markowitz_min_variance",
    "markowitz_max_sharpe",
    "risk_parity",
    "inverse_volatility",
    "minimum_drawdown",
    "risk_evaluation",
    "hrp",
    "momentum",
}


class _HPOTrialFailure(RuntimeError):
    pass


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one asset allocation experiment.")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", dest="config", help="Path to experiment config YAML.")
    config_group.add_argument("--experiment", dest="experiment", help="Alias for --config.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument("--run-name")
    return parser.parse_args(argv)


def _config_path(args: argparse.Namespace) -> str:
    return args.config or args.experiment


def _validate_run_name(run_name: str) -> None:
    if not run_name or run_name in {".", ".."} or not VALID_RUN_NAME.fullmatch(run_name):
        raise ConfigError(
            "ERR_OUTPUT_INVALID_RUN_NAME",
            "output.run_name",
            "ERR_OUTPUT_INVALID_RUN_NAME: output.run_name",
        )


def _create_run_dir(config: dict) -> Path:
    output_root = assert_path_allowed(
        config["output"]["root"],
        config["security"]["path_whitelist"],
        "output.root",
    )
    for forbidden in FORBIDDEN_OUTPUT_DIRS:
        forbidden_path = (PROJECT_ROOT / forbidden).resolve()
        if output_root == forbidden_path or output_root.is_relative_to(forbidden_path):
            raise ConfigError("ERR_SECURITY_PATH_DENIED", "output.root", "ERR_SECURITY_PATH_DENIED: output.root")
    run_name = config["output"]["run_name"]
    _validate_run_name(run_name)
    run_dir = output_root / run_name
    for forbidden in FORBIDDEN_OUTPUT_DIRS:
        forbidden_path = (PROJECT_ROOT / forbidden).resolve()
        if run_dir == forbidden_path or run_dir.is_relative_to(forbidden_path):
            raise ConfigError("ERR_SECURITY_PATH_DENIED", "output.root", "ERR_SECURITY_PATH_DENIED: output.root")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def run_hpo(experiment: HPOExperiment) -> Mapping[str, Any]:
    if _hpo_per_seed_enabled(experiment.config) and not hasattr(experiment, "active_hpo_seed"):
        result = dict(_run_per_seed_hpo(experiment))
    elif experiment.experiment_type == "walk_forward":
        result = dict(_run_walk_forward_hpo(experiment))
    elif _hpo_equal_budget_enabled(experiment.config):
        models = _hpo_trainable_models(experiment.config)
        if len(models) > 1 and not getattr(experiment, "active_model_name", None):
            result = dict(_run_equal_budget_hpo(experiment, models))
        else:
            if len(models) == 1 and not getattr(experiment, "active_model_name", None):
                setattr(experiment, "active_model_name", models[0])
            result = dict(_run_hpo_single(experiment))
    else:
        result = dict(_run_hpo_single(experiment))
    _apply_orchestration_metadata(result, experiment.config)
    return result


def _run_per_seed_hpo(experiment: HPOExperiment) -> Mapping[str, Any]:
    config = experiment.config
    run_dir = experiment.context.run_dir
    if run_dir is None:
        raise ConfigError("ERR_EXPERIMENT_OUTPUT_DIR_MISSING", "output.root", "ERR_EXPERIMENT_OUTPUT_DIR_MISSING")
    _write_hpo_search_space_manifest(config, _hpo_manifest_model_names(config), run_dir / "logs" / "hpo_search_space_manifest.csv")
    seed_payloads: list[dict[str, Any]] = []
    for seed in _hpo_seed_values(config):
        seed_key = _safe_hpo_model_key(f"seed_{seed}")
        seed_config = deepcopy(dict(config))
        seed_config.setdefault("reproducibility", {})
        seed_config["reproducibility"]["seed"] = int(seed)
        seed_config.setdefault("hpo", {})
        seed_config["hpo"]["seed"] = int(seed)
        seed_config.setdefault("output", {})
        seed_config["output"]["run_name"] = f"{config['output']['run_name']}_{seed_key}"
        seed_config["hpo"]["study_name"] = str(
            seed_config["hpo"].get("study_name") or f"{config['output']['run_name']}_hpo"
        ) + f"_{seed_key}"
        seed_context = replace(experiment.context, config=seed_config, run_dir=run_dir / f"hpo_{seed_key}")
        seed_experiment = HPOExperiment(
            seed_context,
            experiment.experiment_type,
            experiment.output_name,
            hpo_enabled=experiment.hpo_enabled,
        )
        setattr(seed_experiment, "active_hpo_seed", int(seed))
        if hasattr(experiment, "active_split"):
            setattr(seed_experiment, "active_split", getattr(experiment, "active_split"))
        payload = dict(run_hpo(seed_experiment))
        payload["seed"] = int(seed)
        seed_payloads.append(payload)
        if str(payload.get("status", "completed")) != "completed":
            result = _failed_seed_hpo_result(config, seed_payloads, payload, int(seed))
            _write_seed_hpo_trials(result, run_dir)
            return result
    result = _combined_seed_hpo_result(config, seed_payloads)
    _write_seed_hpo_trials(result, run_dir)
    return result


def _run_equal_budget_hpo(experiment: HPOExperiment, model_names: Sequence[str]) -> Mapping[str, Any]:
    config = experiment.config
    run_dir = experiment.context.run_dir
    if run_dir is None:
        raise ConfigError("ERR_EXPERIMENT_OUTPUT_DIR_MISSING", "output.root", "ERR_EXPERIMENT_OUTPUT_DIR_MISSING")
    _write_hpo_search_space_manifest(config, model_names, run_dir / "logs" / "hpo_search_space_manifest.csv")
    direction = str(_mapping(config.get("hpo")).get("direction") or "maximize")
    diagnostic_run = _is_explicit_hybrid_diagnostic_run(config)
    parent_rows: list[pd.DataFrame] = []
    model_payloads: list[dict[str, Any]] = []
    child_failures: list[dict[str, Any]] = []
    for model_name in model_names:
        if str(model_name) == HYBRID_DQN_OPTIMIZER_ALIAS:
            raise ConfigError(
                "ERR_HPO_HYBRID_ALIAS_NOT_EXPANDED",
                "hpo.trainable_models",
                "ERR_HPO_HYBRID_ALIAS_NOT_EXPANDED: hpo.trainable_models",
            )
        model_key = _safe_hpo_model_key(model_name)
        model_config = deepcopy(dict(config))
        model_config.setdefault("output", {})
        model_config["output"]["run_name"] = f"{config['output']['run_name']}_{model_key}"
        model_config.setdefault("hpo", {})
        model_config["hpo"]["equal_budget_across_models"] = False
        model_config["hpo"]["study_name"] = str(
            model_config["hpo"].get("study_name") or f"{config['output']['run_name']}_hpo"
        ) + f"_{model_key}"
        model_context = replace(experiment.context, config=model_config, run_dir=run_dir / f"hpo_{model_key}")
        model_experiment = HPOExperiment(
            model_context,
            experiment.experiment_type,
            experiment.output_name,
            hpo_enabled=experiment.hpo_enabled,
        )
        setattr(model_experiment, "active_model_name", model_name)
        if hasattr(experiment, "active_split"):
            setattr(model_experiment, "active_split", getattr(experiment, "active_split"))
        child_failure_handled = False
        try:
            payload = dict(_run_hpo_single(model_experiment))
        except Exception as exc:
            if not diagnostic_run and not _is_hybrid_optimizer_child(model_name):
                raise
            failure = _hybrid_child_failure(model_name, exc)
            if not diagnostic_run:
                return _hybrid_required_child_failed_result(
                    config,
                    run_dir,
                    parent_rows,
                    model_payloads,
                    failure,
                )
            payload = _hybrid_child_failure_payload(failure)
            child_failures.append(failure)
            child_failure_handled = True
        payload["hpo_model_name"] = model_name
        if not child_failure_handled and _is_failed_hpo_child_payload(payload):
            failure = _hybrid_child_failure(model_name, payload)
            if not diagnostic_run and _is_hybrid_optimizer_child(model_name):
                return _hybrid_required_child_failed_result(
                    config,
                    run_dir,
                    parent_rows,
                    model_payloads,
                    failure,
                )
            if not diagnostic_run and not _is_hybrid_optimizer_child(model_name):
                raise RuntimeError(str(failure.get("reason") or failure.get("child_status") or "failed"))
            payload.update(_hybrid_child_failure_payload(failure))
            child_failures.append(failure)
        model_payloads.append(payload)
        trials = payload.get("hpo_trials")
        if isinstance(trials, pd.DataFrame):
            trial_frame = trials.copy()
            trial_frame["model_name"] = model_name
            if "fold_id" not in trial_frame.columns:
                trial_frame["fold_id"] = _active_fold_id(experiment)
            parent_rows.append(trial_frame)

    if not model_payloads:
        raise RuntimeError("ERR_HPO_NO_TRAINABLE_MODEL")
    partial_diagnostic_without_best = False
    try:
        best_payload = _best_hpo_model_payload(model_payloads, direction)
    except RuntimeError as exc:
        if not (
            diagnostic_run
            and child_failures
            and str(exc) in {"ERR_HPO_NO_COMPLETED_TRIAL", "ERR_HPO_NO_RANKABLE_COMPLETED_TRIAL"}
        ):
            raise
        best_payload = dict(model_payloads[0])
        partial_diagnostic_without_best = True
    result = dict(best_payload)
    result["status"] = "completed"
    result["hpo_mode"] = "equal_budget_across_models"
    result["best_model_name"] = None if partial_diagnostic_without_best else best_payload.get("hpo_model_name")
    result["trainable_model_count"] = len(model_payloads)
    result["hpo_model_results"] = [_hpo_model_summary(payload) for payload in model_payloads]
    result["hpo_model_final_comparison"] = _hpo_model_final_comparison(model_payloads)
    result["hpo_model_final_reports"] = _hpo_model_final_reports(model_payloads)
    for frame_name in ("daily_returns", "daily_weights", "daily_turnover", "daily_rebalance", "daily_costs"):
        final_frame = _hpo_model_final_frame(model_payloads, frame_name)
        if not final_frame.empty or len(final_frame.columns) > 0:
            result[f"hpo_model_final_{frame_name}"] = final_frame
    final_diagnostics = _hpo_model_final_frame(model_payloads, "baseline_daily_diagnostics")
    if not final_diagnostics.empty or len(final_diagnostics.columns) > 0:
        result["hpo_model_final_daily_diagnostics"] = final_diagnostics
    for frame_name in NEW_MODEL_HPO_FINAL_FRAME_NAMES:
        final_frame = _hpo_model_final_frame(model_payloads, frame_name)
        if not final_frame.empty or len(final_frame.columns) > 0:
            result[f"hpo_model_final_{frame_name}"] = final_frame
    result["hpo_trials"] = pd.concat(parent_rows, ignore_index=True) if parent_rows else pd.DataFrame(columns=HPO_TRIAL_COLUMNS)
    _write_hpo_trials_csv(result["hpo_trials"].to_dict("records"), run_dir / "logs" / "hpo_trials.csv")
    if child_failures:
        _apply_hybrid_partial_diagnostic_result(result, child_failures[0])
    return result


def _is_hybrid_optimizer_child(model_name: Any) -> bool:
    return str(model_name) in HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAME_SET


def _is_explicit_hybrid_diagnostic_run(config: Mapping[str, Any]) -> bool:
    for section_name in ("hpo", "experiment", "diagnostic", "smoke"):
        section = _mapping(config.get(section_name))
        for key in ("run_mode", "mode"):
            value = section.get(key)
            if value is not None and str(value).lower() in HYBRID_DQN_DIAGNOSTIC_RUN_MODES:
                return True
        if section_name in HYBRID_DQN_DIAGNOSTIC_RUN_MODES and section.get("enabled") is True:
            return True
    for key in ("run_mode", "mode"):
        value = config.get(key)
        if value is not None and str(value).lower() in HYBRID_DQN_DIAGNOSTIC_RUN_MODES:
            return True
    return False


def _is_failed_hpo_child_payload(payload: Mapping[str, Any]) -> bool:
    status = payload.get("status")
    return status is not None and str(status) != "completed"


def _hybrid_child_failure(model_name: Any, failure: Any) -> dict[str, Any]:
    if isinstance(failure, Mapping):
        status = str(failure.get("status") or "failed")
        reason = failure.get("reason") or failure.get("fail_reason") or failure.get("message") or status
    else:
        status = "failed"
        reason = str(failure) or type(failure).__name__
    return {
        "failed_child_model_id": str(model_name),
        "child_status": status,
        "reason": str(reason),
    }


def _hybrid_child_failure_payload(failure: Mapping[str, Any], *, partial_diagnostic: bool = True) -> dict[str, Any]:
    child_model_id = str(failure["failed_child_model_id"])
    payload = {
        "status": str(failure.get("child_status") or "failed"),
        "model_name": child_model_id,
        "hpo_model_name": child_model_id,
        "paper_model_id": child_model_id,
        "child_model_name": child_model_id,
        "best_trial_number": None,
        "best_value": float("nan"),
        "trial_count": 0,
        "rankable_in_unified_table": False,
        "failed_child_model_id": child_model_id,
        "reason": str(failure.get("reason") or failure.get("child_status") or "failed"),
        "hpo_trials": pd.DataFrame(columns=HPO_TRIAL_COLUMNS),
    }
    if partial_diagnostic:
        payload["diagnostic_status"] = "partial_diagnostic"
    return payload


def _hybrid_required_child_failed_result(
    config: Mapping[str, Any],
    run_dir: Path,
    parent_rows: Sequence[pd.DataFrame],
    model_payloads: Sequence[Mapping[str, Any]],
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    failure_payload = _hybrid_child_failure_payload(failure, partial_diagnostic=False)
    payloads = [*model_payloads, failure_payload]
    hpo_trials = pd.concat(parent_rows, ignore_index=True) if parent_rows else pd.DataFrame(columns=HPO_TRIAL_COLUMNS)
    _write_hpo_trials_csv(hpo_trials.to_dict("records"), run_dir / "logs" / "hpo_trials.csv")
    return {
        "status": "failed",
        "hpo_mode": "equal_budget_across_models",
        "best_model_name": None,
        "trainable_model_count": len(payloads),
        "hpo_model_results": [_hpo_model_summary(payload) for payload in payloads],
        "hpo_model_final_comparison": _hpo_model_final_comparison(payloads),
        "hpo_model_final_reports": _hpo_model_final_reports(payloads),
        "hpo_trials": hpo_trials,
        "rankable_in_unified_table": False,
        "failed_child_model_id": str(failure["failed_child_model_id"]),
        "reason": str(failure.get("reason") or "failed"),
        "output_run_name": _mapping(config.get("output")).get("run_name"),
    }


def _apply_hybrid_partial_diagnostic_result(result: dict[str, Any], failure: Mapping[str, Any]) -> None:
    result["status"] = "completed"
    result["diagnostic_status"] = "partial_diagnostic"
    result["rankable_in_unified_table"] = False
    result["failed_child_model_id"] = str(failure["failed_child_model_id"])
    result["reason"] = str(failure.get("reason") or "failed")


def _hpo_per_seed_enabled(config: Mapping[str, Any]) -> bool:
    return bool(config.get("long_running") is True and len(_hpo_seed_values(config)) > 1)


def _hpo_seed_values(config: Mapping[str, Any]) -> list[int]:
    reproducibility = _mapping(config.get("reproducibility"))
    values = reproducibility.get("seeds")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        seeds = [int(value) for value in values]
        return _dedupe_ints(seeds)
    seed = reproducibility.get("seed")
    return [] if seed is None else [int(seed)]


def _dedupe_ints(values: Sequence[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        item = int(value)
        if item not in result:
            result.append(item)
    return result


def _combined_seed_hpo_result(config: Mapping[str, Any], seed_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not seed_payloads:
        raise RuntimeError("ERR_HPO_NO_SEED")
    result: dict[str, Any] = {
        "status": "completed",
        "long_running": bool(config.get("long_running") is True),
        "hpo_mode": "per_seed_independent_hpo",
        "hpo_seed_count": len(seed_payloads),
        "best_model_name": None,
        "hpo_seed_results": [_hpo_seed_summary(payload) for payload in seed_payloads],
    }
    first_trainable_count = seed_payloads[0].get("trainable_model_count")
    if first_trainable_count is not None:
        result["trainable_model_count"] = first_trainable_count
    for frame_name in (
        "hpo_trials",
        "hpo_model_final_comparison",
        "hpo_model_final_reports",
        "hpo_model_final_daily_returns",
        "hpo_model_final_daily_weights",
        "hpo_model_final_daily_turnover",
        "hpo_model_final_daily_rebalance",
        "hpo_model_final_daily_costs",
        "hpo_model_final_daily_diagnostics",
        "hpo_model_final_gate_actions",
        "hpo_model_final_gate_action_summary",
        "hpo_model_final_cage_eiie_candidate_weights",
        "hpo_model_final_cage_final_weights",
        "hpo_model_final_turnover_cost_breakdown",
        "hpo_model_final_risk_metrics",
        "hpo_model_final_validation_selection_report",
        "hpo_model_final_ra_gt_rcpo_daily_diagnostics",
        "hpo_model_final_ra_gt_rcpo_constraint_multipliers",
        "hpo_model_final_ra_gt_rcpo_graph_diagnostics",
        "hpo_model_final_ra_gt_rcpo_actor_critic_training_history",
        "hpo_model_final_ra_gt_rcpo_risk_decomposition",
    ):
        frame = _hpo_seed_frame(seed_payloads, frame_name)
        if not frame.empty or len(frame.columns) > 0:
            result[frame_name] = frame
    return result


def _failed_seed_hpo_result(
    config: Mapping[str, Any],
    seed_payloads: Sequence[Mapping[str, Any]],
    failed_payload: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    result = _combined_seed_hpo_result(config, seed_payloads)
    result["status"] = "failed"
    result["failed_seed"] = int(seed)
    result["reason"] = str(failed_payload.get("reason") or failed_payload.get("status") or "failed")
    return result


def _hpo_seed_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": payload.get("seed"),
        "status": payload.get("status"),
        "hpo_mode": payload.get("hpo_mode"),
        "best_model_name": payload.get("best_model_name", payload.get("hpo_model_name")),
        "trainable_model_count": payload.get("trainable_model_count"),
        "hpo_model_results": payload.get("hpo_model_results"),
    }


def _hpo_seed_frame(payloads: Sequence[Mapping[str, Any]], frame_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for payload in payloads:
        frame = payload.get(frame_name)
        if not isinstance(frame, pd.DataFrame):
            continue
        item = frame.copy()
        seed = payload.get("seed")
        if "seed" not in item.columns or item["seed"].isna().all():
            item["seed"] = seed
        frames.append(item)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _write_seed_hpo_trials(result: Mapping[str, Any], run_dir: Path) -> None:
    trials = result.get("hpo_trials")
    if isinstance(trials, pd.DataFrame):
        _write_hpo_trials_csv(trials.to_dict("records"), run_dir / "logs" / "hpo_trials.csv")


def _apply_orchestration_metadata(result: dict[str, Any], config: Mapping[str, Any]) -> None:
    result["long_running"] = bool(config.get("long_running") is True)


def _refresh_new_model_artifacts(result: dict[str, Any], config: Mapping[str, Any]) -> None:
    from src.experiments.pipeline import _new_model_artifacts

    result.update(_new_model_artifacts(result, config=config))


def _hpo_equal_budget_enabled(config: Mapping[str, Any]) -> bool:
    hpo_config = _mapping(config.get("hpo"))
    return bool(hpo_config.get("equal_budget_across_models") is True)


def _hpo_trainable_models(config: Mapping[str, Any]) -> list[str]:
    hpo_config = _mapping(config.get("hpo"))
    explicit = hpo_config.get("trainable_models")
    if explicit:
        return _dedupe_strings(_expand_baseline_aliases(explicit))
    model_config = _mapping(config.get("model"))
    main_model = str(model_config.get("name", "full_dqn_gated_multitask_cnn_ppo"))
    baseline_config = _mapping(config.get("baselines"))
    deep_baselines = baseline_config.get("deep", ())
    native_config = baseline_config.get("native_rl")
    native_models = list(native_config.get("enabled_models", ())) if isinstance(native_config, Mapping) else []
    native_models.extend(list(baseline_config.get("native", ()) or ()))
    models = _dedupe_strings(_expand_baseline_aliases([main_model, *list(deep_baselines or ()), *native_models]))
    return _filter_hpo_proxy_models(models, hpo_config)


def _filter_hpo_proxy_models(models: Sequence[str], hpo_config: Mapping[str, Any]) -> list[str]:
    if hpo_config.get("native_only") is not True:
        return list(models)
    filtered = [
        str(model_name)
        for model_name in models
        if _is_native_hpo_trainable_model(str(model_name))
    ]
    if not filtered:
        raise ConfigError(
            "ERR_HPO_NO_NATIVE_TRAINABLE_MODEL",
            "hpo.trainable_models",
            "ERR_HPO_NO_NATIVE_TRAINABLE_MODEL: hpo.trainable_models",
        )
    return filtered


def _is_native_hpo_trainable_model(model_name: str) -> bool:
    if model_name in NON_NATIVE_HPO_MODEL_NAMES:
        return False
    return model_name in NATIVE_HPO_MODEL_NAMES


def _best_hpo_model_payload(payloads: Sequence[Mapping[str, Any]], direction: str) -> dict[str, Any]:
    reverse = str(direction).lower() != "minimize"
    finite_payloads = [item for item in payloads if math.isfinite(float(item.get("best_value", float("nan"))))]
    rankable_payloads = [item for item in finite_payloads if _hpo_payload_rankable(item)]
    ordered = sorted(rankable_payloads, key=lambda item: float(item.get("best_value", float("nan"))), reverse=reverse)
    if not ordered:
        if finite_payloads:
            raise RuntimeError("ERR_HPO_NO_RANKABLE_COMPLETED_TRIAL")
        raise RuntimeError("ERR_HPO_NO_COMPLETED_TRIAL")
    best = ordered[0]
    return dict(best)


def _hpo_payload_rankable(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("status", "completed")) != "completed":
        return False
    if _clean_text(payload.get("final_activity_failure_reason")):
        return False
    if payload.get("rankable_in_unified_table") is not None and not _truthy(payload.get("rankable_in_unified_table")):
        return False
    diagnostic_status = _clean_text(payload.get("diagnostic_status"))
    return diagnostic_status not in {"partial_diagnostic", "diagnostic_shared_dqn", "activity_diagnostic"}


def _hpo_model_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_name": payload.get("hpo_model_name"),
        "study_name": payload.get("study_name"),
        "best_trial_number": payload.get("best_trial_number"),
        "best_value": payload.get("best_value"),
        "best_params": payload.get("best_params"),
        "trial_count": payload.get("trial_count"),
        "hpo_trials_path": payload.get("hpo_trials_path"),
        "best_checkpoint_path": payload.get("best_checkpoint_path"),
        "last_checkpoint_path": payload.get("last_checkpoint_path"),
        "evaluated_checkpoint_path": payload.get("evaluated_checkpoint_path"),
        "best_validation_metric": payload.get("best_validation_metric"),
        "hpo_final_report_count": _hpo_final_report_count(payload),
        "status": payload.get("status"),
        "rankable_in_unified_table": payload.get("rankable_in_unified_table"),
        "diagnostic_status": payload.get("diagnostic_status"),
        "final_activity_passed": payload.get("final_activity_passed"),
        "final_activity_failure_reason": payload.get("final_activity_failure_reason"),
    }


def _hpo_model_final_comparison(payloads: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        model_name = str(payload.get("hpo_model_name") or payload.get("model_name") or "")
        if not model_name:
            continue
        comparison = _first_frame(
            payload.get("main_comparison"),
            payload.get("baseline_comparison"),
        )
        if comparison is not None and not comparison.empty:
            selected = comparison.loc[comparison["model_name"].astype(str).eq(model_name)].copy()
            if selected.empty:
                selected = comparison.head(1).copy()
            for record in selected.to_dict("records"):
                record.setdefault("model_name", model_name)
                record["hpo_model_name"] = model_name
                record["best_trial_number"] = payload.get("best_trial_number")
                record["best_value"] = payload.get("best_value")
                rows.append(_annotate_hpo_final_record(record, payload, model_name=model_name))
            continue
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
        rows.append(
            _annotate_hpo_final_record(
                {
                    "model_name": model_name,
                    "hpo_model_name": model_name,
                    "status": payload.get("status", "completed"),
                    "best_trial_number": payload.get("best_trial_number"),
                    "best_value": payload.get("best_value"),
                    **{str(key): value for key, value in dict(metrics).items()},
                },
                payload,
                model_name=model_name,
            )
        )
    return pd.DataFrame(rows)


def _annotate_hpo_final_record(record: Mapping[str, Any], payload: Mapping[str, Any], *, model_name: str) -> dict[str, Any]:
    result = dict(record)
    result.setdefault("model_name", model_name)
    result.setdefault("hpo_model_name", model_name)
    for key, value in _activity_audit_values(payload, model_name=model_name).items():
        result.setdefault(key, value)
    failure = _clean_text(payload.get("final_activity_failure_reason"))
    rankable = _hpo_payload_rankable(payload)
    result.setdefault("final_activity_passed", failure == "")
    result.setdefault("final_activity_failure_reason", failure)
    if failure or not rankable:
        result["rankable_in_unified_table"] = False
        result["paper_included"] = False
        if not _clean_text(result.get("diagnostic_status")):
            result["diagnostic_status"] = _clean_text(payload.get("diagnostic_status")) or "activity_diagnostic"
        if not _clean_text(result.get("reason")):
            result["reason"] = failure or _clean_text(payload.get("reason")) or _clean_text(result.get("diagnostic_status")) or "non_rankable"
    else:
        result.setdefault("rankable_in_unified_table", True)
        result.setdefault("paper_included", True)
    return result


def _hpo_model_final_frame(payloads: Sequence[Mapping[str, Any]], frame_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for payload in payloads:
        frame = payload.get(frame_name)
        if not isinstance(frame, pd.DataFrame):
            continue
        model_name = str(payload.get("hpo_model_name") or payload.get("model_name") or "")
        item = frame.copy()
        if model_name:
            item["hpo_model_name"] = model_name
            if "model_name" not in item.columns or item["model_name"].isna().all():
                item["model_name"] = model_name
        item["best_trial_number"] = payload.get("best_trial_number")
        item["best_value"] = payload.get("best_value")
        frames.append(item)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _hpo_model_final_reports(payloads: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for payload in payloads:
        model_name = str(payload.get("hpo_model_name") or payload.get("model_name") or "")
        table = payload.get("hpo_final_reports_table")
        if isinstance(table, pd.DataFrame):
            frame = table.copy()
        else:
            frame = _hpo_final_report_summary_table(
                payload.get("hpo_final_reports"),
                model_name=model_name,
                study_name=payload.get("study_name"),
                final_split=payload.get("final_report_split"),
            )
        if frame.empty and len(frame.columns) == 0:
            continue
        if model_name:
            if "hpo_model_name" not in frame.columns or frame["hpo_model_name"].isna().all():
                frame["hpo_model_name"] = model_name
            if "model_name" not in frame.columns or frame["model_name"].isna().all():
                frame["model_name"] = model_name
        frame["best_trial_number"] = payload.get("best_trial_number")
        frame["best_value"] = payload.get("best_value")
        frame["trial_count"] = payload.get("trial_count")
        frame["selection_split"] = payload.get("selection_split")
        frames.append(frame)
    if frames:
        return _order_hpo_final_report_columns(pd.concat(frames, ignore_index=True, sort=False))
    return pd.DataFrame(columns=HPO_FINAL_REPORT_BASE_COLUMNS)


def _first_frame(*values: Any) -> pd.DataFrame | None:
    for value in values:
        if isinstance(value, pd.DataFrame):
            return value
    return None


def _activity_audit_values(result: Mapping[str, Any], *, model_name: str | None = None) -> dict[str, Any]:
    values: dict[str, Any] = {}
    metrics = _mapping(result.get("metrics"))
    for column in HPO_ACTIVITY_AUDIT_COLUMNS:
        if column in metrics:
            values[column] = metrics[column]
    comparison = _first_frame(result.get("main_comparison"), result.get("baseline_comparison"))
    if comparison is not None and not comparison.empty:
        selected = comparison
        target_model = model_name or result.get("hpo_model_name") or result.get("model_name")
        if target_model and "model_name" in comparison.columns:
            match = comparison.loc[comparison["model_name"].astype(str).eq(str(target_model))]
            if not match.empty:
                selected = match
        record = selected.iloc[0].to_dict()
        for column in HPO_ACTIVITY_AUDIT_COLUMNS:
            if column in record and column not in values:
                values[column] = record[column]
    return values


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _safe_hpo_model_key(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name)).strip("._") or "model"


def _dedupe_strings(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value)
        if item and item not in result:
            result.append(item)
    return result


def _activity_trial_failure_reason(result: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    hpo_cfg = _mapping(config.get("hpo"))
    constraints = _mapping(hpo_cfg.get("activity_constraints"))
    activity = _mapping(config.get("execution_activity"))
    if constraints.get("enabled") is not True or activity.get("activity_gate_enforced") is not True:
        return None
    protocol = str(activity.get("protocol", "monthly_gate"))
    scope_protocols = {str(item) for item in constraints.get("scope_activity_protocols", [])}
    if protocol not in scope_protocols:
        return None
    families = _result_baseline_families(result)
    scope_families = {str(item) for item in constraints.get("scope_baseline_families", [])}
    if not families or families.isdisjoint(scope_families):
        return None
    metrics = _activity_audit_values(result)
    hit_rate = _safe_float(metrics.get("model_rebalance_hit_rate"), 0.0)
    turnover_per_opportunity = _safe_float(metrics.get("non_initial_turnover_per_opportunity"), 0.0)
    avg_turnover = _safe_float(metrics.get("average_turnover"), 0.0)
    min_hit_rate = float(constraints.get("min_model_rebalance_hit_rate", 0.05))
    max_hit_rate = constraints.get("max_model_rebalance_hit_rate")
    max_average_turnover = constraints.get("max_average_turnover")
    if hit_rate < min_hit_rate and _rho_policy_collapsed(result, config):
        return "failed_rho_policy_collapsed"
    if hit_rate < min_hit_rate:
        return "failed_low_trade_activity"
    if turnover_per_opportunity < float(constraints.get("min_non_initial_turnover_per_opportunity", 0.002)):
        return "failed_low_trade_activity"
    if max_hit_rate is not None and hit_rate > float(max_hit_rate):
        return "failed_high_trade_activity"
    if max_average_turnover is not None and avg_turnover > float(max_average_turnover):
        return "failed_high_trade_activity"
    return None


def _activity_hpo_trial_hard_fail_enabled(config: Mapping[str, Any]) -> bool:
    constraints = _mapping(_mapping(config.get("hpo")).get("activity_constraints"))
    return constraints.get("hard_fail_trials") is True


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _result_baseline_families(result: Mapping[str, Any]) -> set[str]:
    families: set[str] = set()
    model_names: set[str] = set()
    diagnostics = result.get("baseline_daily_diagnostics")
    if isinstance(diagnostics, pd.DataFrame) and "baseline_family" in diagnostics.columns:
        families.update(diagnostics["baseline_family"].dropna().astype(str).tolist())
        model_names.update(_frame_string_values(diagnostics, "model_name"))
        model_names.update(_frame_string_values(diagnostics, "paper_model_id"))
        if _frame_has_truthy(diagnostics, "platform_native_rl_training"):
            families.add("platform_native_rl")
    comparison = result.get("main_comparison")
    if isinstance(comparison, pd.DataFrame) and "baseline_family" in comparison.columns:
        families.update(comparison["baseline_family"].dropna().astype(str).tolist())
        model_names.update(_frame_string_values(comparison, "model_name"))
        model_names.update(_frame_string_values(comparison, "paper_model_id"))
        if _frame_has_truthy(comparison, "platform_native_rl_training"):
            families.add("platform_native_rl")
    for key in ("model_name", "hpo_model_name", "paper_model_id"):
        model_name = result.get(key)
        if model_name is not None:
            model_names.add(str(model_name))
    if families.intersection(PLATFORM_NATIVE_RL_FAMILY_ALIASES) or model_names.intersection(PLATFORM_NATIVE_RL_MODEL_NAMES):
        families.add("platform_native_rl")
    return families


def _frame_string_values(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame.columns:
        return set()
    return {str(value) for value in frame[column].dropna().tolist() if str(value).strip()}


def _frame_has_truthy(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    return any(_truthy(value) for value in frame[column].dropna().tolist())


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value) and not math.isnan(float(value))
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _rho_policy_collapsed(result: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    section = _mapping(config.get("ra_gt_rcpo"))
    if str(section.get("rho_policy", "score_rho_normalized")) not in {"learned", "straight_through_gumbel_softmax_v1"}:
        return False
    threshold = _rho_entropy_threshold(section)
    entropy = _rho_entropy_series(result)
    if entropy.empty:
        return False
    return float(entropy.mean()) < threshold


def _rho_entropy_threshold(section: Mapping[str, Any]) -> float:
    temperature = _mapping(section.get("rho_temperature"))
    return float(temperature.get("min_entropy_threshold", section.get("rho_min_entropy_threshold", 0.05)))


def _rho_entropy_series(result: Mapping[str, Any]) -> pd.Series:
    frames: list[pd.Series] = []
    diagnostics = result.get("baseline_daily_diagnostics")
    if isinstance(diagnostics, pd.DataFrame) and "rho_entropy" in diagnostics.columns:
        frames.append(pd.to_numeric(diagnostics["rho_entropy"], errors="coerce"))
    history = result.get("training_history")
    if isinstance(history, pd.DataFrame) and "rho_entropy" in history.columns:
        frames.append(pd.to_numeric(history["rho_entropy"], errors="coerce"))
    elif isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        frame = pd.DataFrame(history)
        if "rho_entropy" in frame.columns:
            frames.append(pd.to_numeric(frame["rho_entropy"], errors="coerce"))
    if not frames:
        return pd.Series(dtype=float)
    values = pd.concat(frames, ignore_index=True).dropna()
    return values.loc[values.map(math.isfinite)]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _active_fold_id(experiment: HPOExperiment) -> str:
    split = getattr(experiment, "active_split", None)
    if split is None:
        return ""
    return str(getattr(split, "fold_id", ""))


def _run_hpo_single(experiment: HPOExperiment) -> Mapping[str, Any]:
    import optuna

    config = experiment.config
    hpo_config = config.get("hpo", {})
    if not isinstance(hpo_config, Mapping):
        hpo_config = {}
    run_dir = experiment.context.run_dir
    if run_dir is None:
        raise ConfigError("ERR_EXPERIMENT_OUTPUT_DIR_MISSING", "output.root", "ERR_EXPERIMENT_OUTPUT_DIR_MISSING")

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
    trial_rows: list[dict[str, Any]] = []

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

    def objective(trial: optuna.Trial) -> float:
        train_start_clock = time.perf_counter()
        row: dict[str, Any] = {
            "model_name": str(getattr(experiment, "active_model_name", config.get("model", {}).get("name", "full_dqn_gated_multitask_cnn_ppo"))),
            "fold_id": _active_fold_id(experiment),
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
            validation_metric = _metric_value(trial_result, metric)
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
        except DataContractError as exc:
            row["state"] = "fail"
            row["fail_reason"] = str(exc)
            raise
        except Exception as exc:
            row["state"] = "fail"
            row["fail_reason"] = str(exc)
            raise _HPOTrialFailure(str(exc)) from exc
        finally:
            row["params_json"] = json.dumps(dict(trial.params), ensure_ascii=False, sort_keys=True)
            row["train_end"] = _utc_now()
            row["duration_sec"] = round(time.perf_counter() - train_start_clock, 6)
            trial_rows.append(row)
            _write_hpo_trials_csv(trial_rows, trials_path)

    _write_hpo_trials_csv(trial_rows, trials_path)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(_HPOTrialFailure,))

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
            "hpo_final_reports": _hpo_final_report_summaries(final_reports),
            "hpo_final_reports_table": _hpo_final_report_table(
                final_reports,
                model_name=str(
                    getattr(
                        experiment,
                        "active_model_name",
                        config.get("model", {}).get("name", "full_dqn_gated_multitask_cnn_ppo"),
                    )
                ),
                study_name=study_name,
                final_split=final_split,
                best_trial_number=best_trial.number,
                best_value=best_trial.value,
                trial_count=len(trial_rows),
                selection_split=validation_split,
                config=config,
            ),
            "final_result": _result_summary(final_result),
            "hpo_model_name": str(getattr(experiment, "active_model_name", config.get("model", {}).get("name", "full_dqn_gated_multitask_cnn_ppo"))),
        }
    )
    _apply_hpo_final_activity_status(payload, config)
    _refresh_new_model_artifacts(payload, config)
    _attach_single_model_hpo_final_outputs(payload)
    return payload


def _apply_hpo_final_activity_status(payload: dict[str, Any], config: Mapping[str, Any]) -> None:
    payload.update(_activity_audit_values(payload, model_name=_clean_text(payload.get("hpo_model_name")) or None))
    failure = _activity_trial_failure_reason(payload, config)
    payload["final_activity_passed"] = failure is None
    payload["final_activity_failure_reason"] = failure or ""
    if failure is None:
        return
    payload["rankable_in_unified_table"] = False
    payload["diagnostic_status"] = "activity_diagnostic"
    payload["reason"] = failure


def _attach_single_model_hpo_final_outputs(payload: dict[str, Any]) -> None:
    payloads = [payload]
    payload["hpo_model_final_comparison"] = _hpo_model_final_comparison(payloads)
    payload["hpo_model_final_reports"] = _hpo_model_final_reports(payloads)
    for frame_name in ("daily_returns", "daily_weights", "daily_turnover", "daily_rebalance", "daily_costs"):
        final_frame = _hpo_model_final_frame(payloads, frame_name)
        if not final_frame.empty or len(final_frame.columns) > 0:
            payload[f"hpo_model_final_{frame_name}"] = final_frame
    final_diagnostics = _hpo_model_final_frame(payloads, "baseline_daily_diagnostics")
    if not final_diagnostics.empty or len(final_diagnostics.columns) > 0:
        payload["hpo_model_final_daily_diagnostics"] = final_diagnostics
    for frame_name in NEW_MODEL_HPO_FINAL_FRAME_NAMES:
        final_frame = _hpo_model_final_frame(payloads, frame_name)
        if not final_frame.empty or len(final_frame.columns) > 0:
            payload[f"hpo_model_final_{frame_name}"] = final_frame


def _run_walk_forward_hpo(experiment: HPOExperiment) -> Mapping[str, Any]:
    config = experiment.config
    run_dir = experiment.context.run_dir
    if run_dir is None:
        raise ConfigError("ERR_EXPERIMENT_OUTPUT_DIR_MISSING", "output.root", "ERR_EXPERIMENT_OUTPUT_DIR_MISSING")
    _write_hpo_search_space_manifest(config, _hpo_manifest_model_names(config), run_dir / "logs" / "hpo_search_space_manifest.csv")
    dataset = load_market_dataset(config)
    splits = create_split(pd.DatetimeIndex(dataset.wide["close"].index), config)
    fold_splits = list(splits) if isinstance(splits, list) else [splits]
    if not fold_splits:
        raise RuntimeError("ERR_HPO_WALK_FORWARD_EMPTY")

    fold_payloads: list[dict[str, Any]] = []
    for index, split in enumerate(fold_splits):
        fold_id = str(getattr(split, "fold_id", f"fold_{index + 1}"))
        fold_run_dir = run_dir / f"hpo_{fold_id}"
        fold_run_dir.mkdir(parents=True, exist_ok=True)
        fold_config = deepcopy(dict(config))
        fold_config.setdefault("output", {})
        fold_config["output"]["run_name"] = f"{config['output']['run_name']}_{fold_id}"
        fold_config.setdefault("hpo", {})
        fold_config["hpo"]["study_name"] = str(
            fold_config["hpo"].get("study_name") or f"{config['output']['run_name']}_hpo"
        ) + f"_{fold_id}"
        fold_context = replace(experiment.context, config=fold_config, run_dir=fold_run_dir)
        fold_experiment = HPOExperiment(
            fold_context,
            experiment.experiment_type,
            experiment.output_name,
            hpo_enabled=experiment.hpo_enabled,
        )
        setattr(fold_experiment, "active_split", split)
        if _hpo_equal_budget_enabled(fold_config):
            model_names = _hpo_trainable_models(fold_config)
            if len(model_names) > 1 and not getattr(fold_experiment, "active_model_name", None):
                fold_payload = dict(_run_equal_budget_hpo(fold_experiment, model_names))
            else:
                fold_payload = dict(_run_hpo_single(fold_experiment))
        else:
            fold_payload = dict(_run_hpo_single(fold_experiment))
        fold_payload["fold_id"] = fold_id
        fold_payloads.append(fold_payload)

    aggregation = aggregate_walk_forward(fold_payloads, run_dir=run_dir)
    result: dict[str, Any] = {}
    for frame_name in ("daily_returns", "daily_weights", "daily_turnover", "daily_rebalance", "daily_costs"):
        frames = []
        for payload in fold_payloads:
            frame = payload.get(frame_name)
            if isinstance(frame, pd.DataFrame):
                fold_frame = frame.copy()
                if "fold_id" not in fold_frame:
                    fold_frame["fold_id"] = payload["fold_id"]
                frames.append(fold_frame)
        result[frame_name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result["status"] = "completed"
    result["experiment_type"] = experiment.experiment_type
    result["fold_count"] = aggregation["fold_count"]
    result["duplicate_oos_date_count"] = aggregation["duplicate_oos_date_count"]
    result["walk_forward_results"] = aggregation["walk_forward_results"]
    result["all_oos_daily_returns"] = aggregation["all_oos_daily_returns"]
    result["fold_hpo_results"] = [
        {
            "fold_id": payload["fold_id"],
            "study_name": payload.get("study_name"),
            "hpo_mode": payload.get("hpo_mode"),
            "best_model_name": payload.get("best_model_name", payload.get("hpo_model_name")),
            "hpo_model_results": payload.get("hpo_model_results"),
            "best_trial_number": payload.get("best_trial_number"),
            "best_value": payload.get("best_value"),
            "best_params": payload.get("best_params"),
            "trial_count": payload.get("trial_count"),
            "hpo_trials_path": payload.get("hpo_trials_path"),
        }
        for payload in fold_payloads
    ]
    hpo_trial_frames = []
    for payload in fold_payloads:
        frame = payload.get("hpo_trials")
        if isinstance(frame, pd.DataFrame):
            fold_frame = frame.copy()
            if "fold_id" not in fold_frame.columns or fold_frame["fold_id"].fillna("").eq("").all():
                fold_frame["fold_id"] = payload["fold_id"]
            hpo_trial_frames.append(fold_frame)
    result["hpo_trials"] = (
        pd.concat(hpo_trial_frames, ignore_index=True)
        if hpo_trial_frames
        else pd.DataFrame(columns=HPO_TRIAL_COLUMNS)
    )
    _write_hpo_trials_csv(result["hpo_trials"].to_dict("records"), run_dir / "logs" / "hpo_trials.csv")
    return result


def main(argv: Sequence[str] | None = None) -> Path:
    args = _parse_args(argv)
    config_path = _config_path(args)
    config = ConfigLoader.load(config_path, cli_overrides=args)
    set_global_seed(config["reproducibility"]["seed"], config["reproducibility"])
    device = get_device(config["device"])
    run_dir = _create_run_dir(config)
    save_yaml_atomic(config, run_dir / "logs" / "config_snapshot.yaml")
    registry_path = _registry_path(config)
    run_id = str(config["output"]["run_name"])
    registry = ExperimentRegistry()
    mark_run_status("running", registry_path, run_id)
    experiment = None
    try:
        experiment = registry.create_experiment(config=config, device=device, run_dir=run_dir)
        if isinstance(experiment, HPOExperiment):
            result = run_hpo(experiment)
        else:
            result = _result_mapping(experiment.run())
        _apply_orchestration_metadata(result, config)
        _assert_completed_result(result)
        result_path = _write_experiment_outputs(result, run_dir)
        output_config = _config_for_output_snapshot(config, result)
        artifacts = write_run_outputs(
            result,
            run_dir,
            config=output_config,
            config_path=config_path,
            command=_command_string(argv),
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
        _write_run_manifest(config, experiment, run_dir, "failed", failure_state=failure_state)
        mark_run_failed(failure_state, registry_path, run_id)
        raise

    manifest_path = artifacts["run_manifest"]
    mark_run_status("success", registry_path, run_id, {"manifest_path": str(manifest_path)})
    return run_dir


def _registry_path(config: Mapping[str, Any]) -> Path | None:
    registry_config = config.get("registry", {})
    if not isinstance(registry_config, Mapping) or registry_config.get("enabled") is not True:
        return None
    return assert_path_allowed(
        registry_config["path"],
        config["security"]["path_whitelist"],
        "registry.path",
    )


def _result_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    return {"status": "completed", "result": result}


def _assert_completed_result(result: Mapping[str, Any]) -> None:
    status = str(result.get("status", "completed"))
    if status != "completed":
        raise RuntimeError(f"ERR_EXPERIMENT_RESULT_NOT_COMPLETED: status={status}")


def _result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in result.items():
        if hasattr(value, "shape") and hasattr(value, "columns"):
            summary[key] = {
                "rows": int(value.shape[0]),
                "columns": [str(column) for column in value.columns],
            }
        else:
            summary[key] = value
    return summary


def _command_string(argv: Sequence[str] | None) -> str:
    if argv is None:
        import sys

        return " ".join(["python", "-m", "src.experiments.run_experiment", *sys.argv[1:]])
    return " ".join(["python", "-m", "src.experiments.run_experiment", *map(str, argv)])


def _config_for_output_snapshot(config: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = deepcopy(dict(config))
    if "best_trial_number" in result or "best_params" in result:
        hpo_config = dict(snapshot.get("hpo", {}))
        if "best_trial_number" in result:
            hpo_config["best_trial_number"] = result["best_trial_number"]
        if "best_params" in result:
            hpo_config["best_params"] = dict(result["best_params"])
        snapshot["hpo"] = hpo_config
    return snapshot


def _write_experiment_outputs(result: Mapping[str, Any], run_dir: Path) -> Path:
    return save_json_atomic(_result_summary(result), run_dir / "logs" / "experiment_result.json")


def _run_hpo_trial(experiment: HPOExperiment, trial: Any, train_split: str, validation_split: str) -> Mapping[str, Any]:
    runner = getattr(experiment, "run_trial", None)
    if callable(runner):
        return _result_mapping(
            runner(
                trial=trial,
                train_split=train_split,
                validation_split=validation_split,
            )
        )
    raise NotImplementedError(f"ERR_HPO_TRIAL_RUNNER_NOT_IMPLEMENTED: {experiment.experiment_type}")


def _run_hpo_final_test(experiment: HPOExperiment, best_trial: Any, final_split: str) -> Mapping[str, Any]:
    runner = getattr(experiment, "run_final_test", None)
    if callable(runner):
        return _result_mapping(runner(best_trial=best_trial, split=final_split))
    raise NotImplementedError(f"ERR_HPO_FINAL_TEST_NOT_IMPLEMENTED: {experiment.experiment_type}")


def _run_hpo_final_reports(
    experiment: HPOExperiment,
    complete_trials: Sequence[Any],
    direction: str,
    final_split: str,
) -> dict[str, dict[str, Any]]:
    selected = _selected_hpo_report_trials(complete_trials, direction)
    cache: dict[int, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    previous_label = getattr(experiment, "final_test_label", None)
    had_previous_label = hasattr(experiment, "final_test_label")
    try:
        for label, trial in selected.items():
            trial_number = int(getattr(trial, "number", -1))
            if trial_number not in cache:
                setattr(experiment, "final_test_label", f"final_test_{label}")
                cache[trial_number] = _result_mapping(_run_hpo_final_test(experiment, trial, final_split))
            reports[label] = {
                "trial_number": trial_number,
                "validation_value": getattr(trial, "value", None),
                "params": dict(getattr(trial, "params", {})),
                "result": cache[trial_number],
            }
    finally:
        if had_previous_label:
            setattr(experiment, "final_test_label", previous_label)
        elif hasattr(experiment, "final_test_label"):
            delattr(experiment, "final_test_label")
    return reports


def _selected_hpo_report_trials(complete_trials: Sequence[Any], direction: str) -> dict[str, Any]:
    reverse = str(direction).lower() != "minimize"
    ordered = sorted(complete_trials, key=lambda trial: float(trial.value), reverse=reverse)
    if not ordered:
        raise RuntimeError("ERR_HPO_NO_COMPLETED_TRIAL")
    median_index = len(ordered) // 2
    return {
        "best": ordered[0],
        "median": ordered[median_index],
        "worst": ordered[-1],
    }


def _hpo_final_report_summaries(final_reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, report in final_reports.items():
        rows.append(
            {
                "rank_label": label,
                "trial_number": report.get("trial_number"),
                "validation_value": report.get("validation_value"),
                "params": report.get("params", {}),
                "result": _result_summary(_result_mapping(report.get("result", {}))),
            }
        )
    return rows


def _hpo_final_report_table(
    final_reports: Mapping[str, Mapping[str, Any]],
    *,
    model_name: str,
    study_name: str,
    final_split: str,
    best_trial_number: Any,
    best_value: Any,
    trial_count: int,
    selection_split: str,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, report in final_reports.items():
        result = _result_mapping(report.get("result", {}))
        row = _hpo_final_report_base_row(
            model_name=model_name,
            hpo_model_name=model_name,
            study_name=study_name,
            rank_label=label,
            trial_number=report.get("trial_number"),
            validation_value=report.get("validation_value"),
            params=report.get("params", {}),
            final_split=final_split,
            status=result.get("status", "unknown"),
            best_trial_number=best_trial_number,
            best_value=best_value,
            trial_count=trial_count,
            selection_split=selection_split,
            evaluated_checkpoint_path=result.get("evaluated_checkpoint_path"),
        )
        row.update(_hpo_final_report_metric_values(result, model_name=model_name))
        row.update(_activity_audit_values(result, model_name=model_name))
        activity_failure = _activity_trial_failure_reason(result, config) if config is not None else None
        row["final_activity_passed"] = activity_failure is None
        row["final_activity_failure_reason"] = activity_failure or ""
        rows.append(row)
    return _order_hpo_final_report_columns(pd.DataFrame(rows))


def _hpo_final_report_summary_table(
    summaries: Any,
    *,
    model_name: str,
    study_name: Any,
    final_split: Any,
) -> pd.DataFrame:
    if not isinstance(summaries, Sequence) or isinstance(summaries, (str, bytes)):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        if not isinstance(summary, Mapping):
            continue
        result = _mapping(summary.get("result"))
        row = _hpo_final_report_base_row(
            model_name=model_name,
            hpo_model_name=model_name,
            study_name=study_name,
            rank_label=summary.get("rank_label"),
            trial_number=summary.get("trial_number"),
            validation_value=summary.get("validation_value"),
            params=summary.get("params", {}),
            final_split=final_split,
            status=result.get("status", "unknown"),
            best_trial_number=None,
            best_value=None,
            trial_count=None,
            selection_split=None,
            evaluated_checkpoint_path=result.get("evaluated_checkpoint_path"),
        )
        row.update(_hpo_final_report_metric_values(result, model_name=model_name))
        rows.append(row)
    return _order_hpo_final_report_columns(pd.DataFrame(rows))


def _hpo_final_report_base_row(
    *,
    model_name: Any,
    hpo_model_name: Any,
    study_name: Any,
    rank_label: Any,
    trial_number: Any,
    validation_value: Any,
    params: Any,
    final_split: Any,
    status: Any,
    best_trial_number: Any,
    best_value: Any,
    trial_count: Any,
    selection_split: Any,
    evaluated_checkpoint_path: Any,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "hpo_model_name": hpo_model_name,
        "study_name": study_name,
        "rank_label": rank_label,
        "trial_number": trial_number,
        "validation_value": validation_value,
        "params_json": json.dumps(params or {}, ensure_ascii=False, sort_keys=True, default=str),
        "final_report_split": final_split,
        "status": status,
        "best_trial_number": best_trial_number,
        "best_value": best_value,
        "trial_count": trial_count,
        "selection_split": selection_split,
        "evaluated_checkpoint_path": evaluated_checkpoint_path,
    }


def _hpo_final_report_metric_values(result: Mapping[str, Any], *, model_name: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    metrics = _mapping(result.get("metrics"))
    for key, value in metrics.items():
        if _is_hpo_report_scalar(value):
            values[str(key)] = value
    comparison = _first_frame(result.get("main_comparison"), result.get("baseline_comparison"))
    if comparison is not None and not comparison.empty:
        selected = comparison
        if "model_name" in comparison.columns and model_name:
            match = comparison.loc[comparison["model_name"].astype(str).eq(str(model_name))]
            if not match.empty:
                selected = match
        record = selected.iloc[0].to_dict()
        for key, value in record.items():
            if str(key) in HPO_FINAL_REPORT_BASE_COLUMNS:
                continue
            if _is_hpo_report_scalar(value) and str(key) not in values:
                values[str(key)] = value
    return values


def _is_hpo_report_scalar(value: Any) -> bool:
    if isinstance(value, Mapping):
        return False
    if isinstance(value, (list, tuple, set)):
        return False
    if hasattr(value, "shape") and hasattr(value, "columns"):
        return False
    return True


def _order_hpo_final_report_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty and len(frame.columns) == 0:
        return pd.DataFrame(columns=HPO_FINAL_REPORT_BASE_COLUMNS)
    result = frame.copy()
    for column in HPO_FINAL_REPORT_BASE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    extra = [column for column in result.columns if column not in HPO_FINAL_REPORT_BASE_COLUMNS]
    return result.loc[:, [*HPO_FINAL_REPORT_BASE_COLUMNS, *extra]]


def _hpo_final_report_count(payload: Mapping[str, Any]) -> int:
    table = payload.get("hpo_final_reports_table")
    if isinstance(table, pd.DataFrame):
        return int(table.shape[0])
    reports = payload.get("hpo_final_reports")
    if isinstance(reports, Sequence) and not isinstance(reports, (str, bytes)):
        return len(reports)
    return 0


def _metric_value(result: Mapping[str, Any], metric: str) -> float:
    for key in (metric, "validation_metric", "objective_value"):
        if key not in result:
            continue
        value = float(result[key])
        if not math.isfinite(value):
            raise ValueError(f"ERR_HPO_METRIC_NON_FINITE: {key}")
        return value
    raise ValueError(f"ERR_HPO_METRIC_MISSING: {metric}")


def _hpo_int(value: Any, default: Any) -> int:
    if value is None:
        return int(default)
    return int(value)


def _hpo_pruner_int(hpo_config: Mapping[str, Any], *keys: str) -> int:
    pruner_config = hpo_config.get("pruner")
    if isinstance(pruner_config, Mapping):
        for key in keys:
            if key in pruner_config and pruner_config[key] is not None:
                return int(pruner_config[key])
    for key in keys:
        if key in hpo_config and hpo_config[key] is not None:
            return int(hpo_config[key])
    return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_hpo_trials_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            writer = csv.DictWriter(fh, fieldnames=HPO_TRIAL_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in HPO_TRIAL_COLUMNS})
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)
        return target
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _hpo_manifest_model_names(config: Mapping[str, Any], experiment: HPOExperiment | None = None) -> list[str]:
    active_model = None if experiment is None else getattr(experiment, "active_model_name", None)
    if active_model:
        return [str(active_model)]
    if _hpo_equal_budget_enabled(config):
        return _hpo_trainable_models(config)
    model_config = _mapping(config.get("model"))
    return [str(model_config.get("name", "full_dqn_gated_multitask_cnn_ppo"))]


def _write_hpo_search_space_manifest(
    config: Mapping[str, Any],
    model_names: Sequence[str],
    path: Path,
) -> Path:
    rows = _hpo_search_space_manifest_rows(config, model_names)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            writer = csv.DictWriter(fh, fieldnames=HPO_SEARCH_SPACE_MANIFEST_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in HPO_SEARCH_SPACE_MANIFEST_COLUMNS})
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)
        return target
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _hpo_search_space_manifest_rows(config: Mapping[str, Any], model_names: Sequence[str]) -> list[dict[str, Any]]:
    hpo_config = _mapping(config.get("hpo"))
    search_space = _mapping(hpo_config.get("search_space"))
    models = _dedupe_strings(model_names)
    if not models:
        models = _hpo_manifest_model_names(config)
    rows: list[dict[str, Any]] = []
    for model_name in models:
        for param_name, raw_spec in search_space.items():
            spec = _mapping(raw_spec)
            model_scope = spec.get("models") or spec.get("model_names")
            model_specific = isinstance(model_scope, Sequence) and not isinstance(model_scope, (str, bytes))
            if model_specific and str(model_name) not in {str(item) for item in model_scope}:
                continue
            choices = spec.get("choices")
            rows.append(
                {
                    "model_name": str(model_name),
                    "param_name": str(param_name),
                    "param_type": str(spec.get("type", "")),
                    "low": spec.get("low", ""),
                    "high": spec.get("high", ""),
                    "choices": "" if choices is None else json.dumps(choices, ensure_ascii=False, sort_keys=True),
                    "log_scale": bool(spec.get("log", spec.get("log_scale", False))),
                    "is_shared_across_models": not model_specific,
                    "is_model_specific": model_specific,
                    "rationale": str(spec.get("rationale", "")),
                }
            )
    return rows


def _read_hpo_trials_csv(path: Path) -> Any:
    if not path.exists():
        return []
    import pandas as pd

    return pd.read_csv(path)


def _write_best_trial_config_snapshot(experiment: HPOExperiment, best_trial: Any, run_dir: Path) -> Path:
    config = deepcopy(experiment.config)
    hpo_config = dict(config.get("hpo", {}))
    hpo_config["best_trial_number"] = best_trial.number
    hpo_config["best_params"] = dict(best_trial.params)
    config["hpo"] = hpo_config
    return save_yaml_atomic(config, run_dir / "logs" / "config_snapshot.yaml")


def _write_run_manifest(
    config: Mapping[str, Any],
    experiment: Any,
    run_dir: Path,
    status: str,
    result: Mapping[str, Any] | None = None,
    result_path: Path | None = None,
    failure_state: Mapping[str, Any] | None = None,
    device: Any | None = None,
) -> Path:
    execution_model = config["execution_model"]
    data_governance = config["data_governance"]
    portfolio = config["portfolio"]
    protocol = config.get("protocol", {}) if isinstance(config.get("protocol"), Mapping) else {}
    rankability = config.get("rankability", {}) if isinstance(config.get("rankability"), Mapping) else {}
    data_config = config.get("data", {}) if isinstance(config.get("data"), Mapping) else {}
    data_mode = data_config.get("data_mode") or (
        "strict_common_history" if data_config.get("strict_common_history_mode") is True else "availability_mask"
    )
    manifest = {
        "run_id": config["output"]["run_name"],
        "status": status,
        "experiment_type": getattr(experiment, "experiment_type", config["experiment"]["type"]),
        "output_name": getattr(experiment, "output_name", None),
        "config_hash": config.get("config_hash"),
        "data_hash": _data_hash(config),
        "execution_model": dict(execution_model),
        "data_governance": dict(data_governance),
        "portfolio_initial_nav": float(portfolio.get("initial_nav", 1.0)),
        "portfolio_initial_capital_currency": float(portfolio.get("initial_capital_currency", 0.0)),
        "portfolio_currency": str(portfolio.get("currency", "")),
        "execution_price": execution_model.get("execution_price"),
        "execution_price_type": _execution_price_type(execution_model),
        "delayed_action_execution": bool(execution_model.get("delayed_action_execution", False)),
        "same_close_idealized_execution_enabled": bool(
            execution_model.get("same_close_idealized_execution_enabled", False)
        ),
        "idealized_execution": bool(execution_model.get("idealized_execution", False)),
        "strict_no_lookahead_execution": bool(execution_model.get("strict_no_lookahead_execution", False)),
        "t_plus_one": bool(execution_model.get("t_plus_one", False)),
        "long_running": bool(config.get("long_running") is True),
        "device": str(device) if device is not None else None,
        "result_path": None if result_path is None else str(result_path),
        "result": dict(result or {}),
        "failure_state": None if failure_state is None else dict(failure_state),
        "protocol_id": protocol.get("protocol_id"),
        "asset_universe_id": protocol.get("asset_universe_id"),
        "data_cutoff_date": protocol.get("data_cutoff_date"),
        "data_mode": data_mode,
        "data_contract": {
            "data_mode": data_mode,
            "strict_common_history_mode": bool(data_config.get("strict_common_history_mode", False)),
            "return_source": data_governance.get("return_source"),
            "valuation_source": data_governance.get("valuation_source"),
            "reward_return_source": data_governance.get("reward_return_source"),
            "metrics_return_source": data_governance.get("metrics_return_source"),
            "execution_price_source": data_governance.get("execution_price_source"),
        },
        "valuation_source": data_governance.get("valuation_source"),
        "return_source": data_governance.get("return_source"),
        "reward_return_source": data_governance.get("reward_return_source"),
        "metrics_return_source": data_governance.get("metrics_return_source"),
        "execution_price_source": data_governance.get("execution_price_source"),
        "valuation_execution_split": bool(data_governance.get("valuation_execution_split", False)),
        "reward_valuation_split": bool(data_governance.get("reward_valuation_split", False)),
        "rankability": dict(rankability),
        "rankable_in_unified_table": bool(rankability.get("rankable_in_unified_table", False)),
        "diagnostic_status": rankability.get("diagnostic_status", "diagnostic"),
    }
    if result is not None and "best_trial_number" in result:
        manifest["best_trial_number"] = result["best_trial_number"]
    return save_json_atomic(manifest, run_dir / "logs" / "run_manifest.json")


def _execution_price_type(execution_model: Mapping[str, Any]) -> str:
    return "open" if execution_model.get("execution_price") == "next_open" else "close"


def _data_hash(config: Mapping[str, Any]) -> str | None:
    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        return None
    manifest_path = data_config.get("download_manifest_path")
    if manifest_path is None:
        return None
    path = (PROJECT_ROOT / str(manifest_path)).resolve()
    if not path.exists():
        return None
    try:
        import json

        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    files = manifest.get("files")
    if isinstance(files, Mapping):
        return str(files.get(data_config.get("panel_path"), "")) or None
    return None


if __name__ == "__main__":
    main()
