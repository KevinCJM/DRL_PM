from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_PROTOCOL_ID = "core13_v2_full_reset_20260522"
MODEL_EXTENSION_ID = "core13_v2_p12_p13_20260524"
DEFAULT_CAGE_RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_GT_RCPOLITE_RHOS = (0.0, 0.25, 0.5, 1.0)
VALID_ACTIVITY_PROTOCOLS = {"monthly_gate", "weekly_gate", "daily_gate_with_cost_constraint"}


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def weights_json(weights: Any) -> str:
    array = np.asarray(weights, dtype=float).reshape(-1)
    return json.dumps([float(value) for value in array], separators=(",", ":"))


def normalize_candidate(weights: Any, mask: Any) -> np.ndarray:
    available = np.asarray(mask, dtype=bool).reshape(-1)
    result = np.asarray(weights, dtype=float).reshape(-1).copy()
    if result.shape != available.shape:
        result = np.zeros(available.shape, dtype=float)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    result[~available] = 0.0
    result[result < 0.0] = 0.0
    total = float(result.sum())
    if total > 0.0 and np.isfinite(total):
        return result / total
    fallback = np.zeros(available.shape, dtype=float)
    if available.any():
        fallback[available] = 1.0 / float(available.sum())
    return fallback


def estimate_turnover(candidate_weights: np.ndarray, current_weights: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(candidate_weights, dtype=float) - np.asarray(current_weights, dtype=float))))


def estimate_cost(config: Mapping[str, Any], turnover: float) -> float:
    cost_cfg = mapping(config.get("cost_model"))
    proportional = float(cost_cfg.get("proportional_cost", 0.0) or 0.0)
    slippage = float(cost_cfg.get("slippage", 0.0) or 0.0)
    impact = float(cost_cfg.get("market_impact_coef", 0.0) or 0.0) if cost_cfg.get("market_impact_enabled", False) else 0.0
    return float(turnover * (proportional + slippage) + impact * turnover * turnover)


def rebalance_turnover_threshold(config: Mapping[str, Any], model_key: str) -> float:
    model_config = mapping(config.get(model_key))
    for key in ("rebalance_turnover_threshold", "turnover_gate_threshold", "min_rebalance_turnover"):
        if model_config.get(key) is not None:
            return _non_negative_float(model_config[key])
    activity = mapping(config.get("execution_activity"))
    for key in ("model_rebalance_turnover_threshold", "rebalance_turnover_threshold", "turnover_gate_threshold"):
        if activity.get(key) is not None:
            return _non_negative_float(activity[key])
    rebalance = mapping(config.get("rebalance"))
    if str(rebalance.get("mode", "")) == "threshold_turnover":
        return _non_negative_float(rebalance.get("threshold_turnover", 0.0))
    return 0.0


def gate_scoring_config(config: Mapping[str, Any], section_name: str) -> dict[str, Any]:
    section = mapping(config.get(section_name))
    gate = dict(mapping(section.get("gate_scoring")))
    hpo_paths = _hpo_param_paths(config)

    _inherit_hpo_param(gate, section, hpo_paths, section_name, "lambda_turnover", "lambda_turnover")
    _inherit_hpo_param(gate, section, hpo_paths, section_name, "lambda_cost", "lambda_cost")
    _inherit_hpo_param(gate, section, hpo_paths, section_name, "lambda_cvar", "lambda_cvar")
    _inherit_hpo_param(gate, section, hpo_paths, section_name, "lambda_drawdown", "lambda_drawdown", "lambda_dd")
    _inherit_hpo_param(gate, section, hpo_paths, section_name, "cvar_budget", "cvar_loss_budget")
    _inherit_hpo_param(gate, section, hpo_paths, section_name, "drawdown_budget", "drawdown_budget")
    _inherit_hpo_param(
        gate,
        section,
        hpo_paths,
        section_name,
        "turnover_budget_per_trade",
        "average_turnover_per_step_budget",
        "turnover_budget",
    )
    _inherit_hpo_param(
        gate,
        section,
        hpo_paths,
        section_name,
        "cost_budget_per_trade",
        "average_cost_per_step_budget",
        "cost_budget",
    )

    if "cvar_budget" not in gate and "cvar_loss_budget" in section:
        gate["cvar_budget"] = section["cvar_loss_budget"]
    if "drawdown_budget" not in gate and "drawdown_budget" in section:
        gate["drawdown_budget"] = section["drawdown_budget"]
    return gate


def partial_rho_execution_decision(
    config: Mapping[str, Any],
    model_key: str,
    *,
    raw_rho: float,
    estimated_turnover: float,
    first_trade: bool,
    model_hold_reason: str | None = None,
) -> dict[str, Any]:
    raw = float(np.clip(float(raw_rho), 0.0, 1.0))
    threshold = rebalance_turnover_threshold(config, model_key)
    raw_gate_requested = bool(raw > 0.0)
    threshold_turnover = float(raw * float(estimated_turnover))
    raw_requested = bool(first_trade or (raw_gate_requested and threshold_turnover > threshold + 1.0e-12))
    rho = 1.0 if first_trade else (raw if raw_requested else 0.0)
    forced_hold_reason = None
    if rho <= 0.0:
        if raw_gate_requested and not raw_requested:
            forced_hold_reason = "below_rebalance_turnover_threshold"
        else:
            forced_hold_reason = model_hold_reason or "model_chosen_hold"
    return {
        "rho": float(rho),
        "rebalance_action": int(rho > 0.0),
        "rebalance_intensity": float(rho),
        "raw_rho": raw,
        "raw_gate_requested_rebalance": raw_gate_requested,
        "raw_model_requested_rebalance": raw_requested,
        "raw_action": int(raw_requested),
        "raw_rebalance_intensity": raw if raw_requested else 0.0,
        "rebalance_turnover_threshold": threshold,
        "threshold_turnover_estimate": threshold_turnover,
        "forced_hold_reason": forced_hold_reason,
    }


def decision_return_features(log_return_window: Any, fallback_returns: Sequence[float] | None = None) -> dict[str, float]:
    window = np.asarray(log_return_window, dtype=float)
    if window.ndim == 3:
        window = window.reshape(window.shape[-2], window.shape[-1])
    if window.ndim == 2 and window.size:
        series = np.nanmean(window, axis=1)
    else:
        series = np.asarray(list(fallback_returns or ()), dtype=float)
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    if series.size == 0:
        return {"mean": 0.0, "volatility": 0.0, "cvar_loss_5": 0.0}
    tail_n = max(1, int(math.ceil(0.05 * float(series.size))))
    sorted_returns = np.sort(series)
    cvar_return = float(np.mean(sorted_returns[:tail_n]))
    return {
        "mean": float(np.mean(series)),
        "volatility": float(np.std(series, ddof=0)),
        "cvar_loss_5": float(max(0.0, -cvar_return)),
    }


def rho_grid(config: Mapping[str, Any], section_name: str, fallback: Sequence[float]) -> tuple[float, ...]:
    section = mapping(config.get(section_name))
    raw = section.get("rho_actions") or section.get("rho_grid") or fallback
    values = sorted({max(0.0, min(1.0, float(value))) for value in raw})
    if 0.0 not in values:
        values.insert(0, 0.0)
    return tuple(values)


def choose_rho(
    *,
    rho_values: Sequence[float],
    expected_return: float,
    estimated_turnover: float,
    estimated_cost: float,
    cvar_loss_5: float,
    drawdown: float,
    lambda_turnover: float,
    lambda_cost: float,
    lambda_cvar: float,
    lambda_drawdown: float,
    cvar_loss_budget: float,
    drawdown_budget: float,
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    cvar_penalty = max(0.0, float(cvar_loss_5) - float(cvar_loss_budget))
    drawdown_penalty = max(0.0, float(drawdown) - float(drawdown_budget))
    for rho in rho_values:
        value = (
            float(rho) * float(expected_return)
            - float(lambda_turnover) * float(rho) * float(estimated_turnover)
            - float(lambda_cost) * float(rho) * float(estimated_cost)
            - float(lambda_cvar) * float(rho) * cvar_penalty
            - float(lambda_drawdown) * float(rho) * drawdown_penalty
        )
        scores[_rho_key(rho)] = float(value)
    best_key = max(scores, key=lambda key: scores[key])
    return float(best_key), scores


def score_rho_normalized(
    *,
    rho_values: Sequence[float],
    expected_alpha: float,
    estimated_turnover: float,
    estimated_cost: float,
    cvar_loss_5: float,
    drawdown: float,
    scale_config: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, float], dict[str, dict[str, float]]]:
    cfg = mapping(scale_config)
    alpha_scale = _positive_scale(cfg.get("alpha_scale", 0.001), "alpha_scale")
    turnover_scale = _positive_scale(cfg.get("turnover_scale", 0.05), "turnover_scale")
    cost_scale = _positive_scale(cfg.get("cost_scale", 0.001), "cost_scale")
    cvar_scale = _positive_scale(cfg.get("cvar_scale", 0.01), "cvar_scale")
    drawdown_scale = _positive_scale(cfg.get("drawdown_scale", 0.05), "drawdown_scale")
    alpha_score = float(expected_alpha) / alpha_scale
    alpha_activation_threshold = float(cfg.get("alpha_activation_threshold", 0.25))
    hold_opportunity_penalty = float(cfg.get("hold_opportunity_penalty", -0.20))
    turnover_budget = float(cfg.get("turnover_budget_per_trade", 0.05))
    cost_budget = float(cfg.get("cost_budget_per_trade", 0.001))
    cvar_budget = float(cfg.get("cvar_budget", cfg.get("cvar_loss_budget", 0.02)))
    drawdown_budget = float(cfg.get("drawdown_budget", 0.10))
    lambda_turnover = float(cfg.get("lambda_turnover", 0.20))
    lambda_cost = float(cfg.get("lambda_cost", 0.20))
    lambda_cvar = float(cfg.get("lambda_cvar", 0.20))
    lambda_drawdown = float(cfg.get("lambda_drawdown", cfg.get("lambda_dd", 0.20)))
    cvar_score = max(0.0, float(cvar_loss_5) - cvar_budget) / cvar_scale
    drawdown_score = max(0.0, float(drawdown) - drawdown_budget) / drawdown_scale

    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for rho in rho_values:
        rho_value = float(rho)
        turnover_excess_score = max(0.0, rho_value * float(estimated_turnover) - turnover_budget) / turnover_scale
        cost_excess_score = max(0.0, rho_value * float(estimated_cost) - cost_budget) / cost_scale
        hold_penalty = hold_opportunity_penalty if rho_value == 0.0 and alpha_score > alpha_activation_threshold else 0.0
        value = (
            rho_value * alpha_score
            - lambda_turnover * turnover_excess_score
            - lambda_cost * cost_excess_score
            - lambda_cvar * rho_value * cvar_score
            - lambda_drawdown * rho_value * drawdown_score
            + hold_penalty
        )
        key = _rho_key(rho_value)
        scores[key] = float(value)
        components[key] = {
            "alpha_score": float(alpha_score),
            "turnover_excess_score": float(turnover_excess_score),
            "cost_excess_score": float(cost_excess_score),
            "cvar_score": float(cvar_score),
            "drawdown_score": float(drawdown_score),
            "hold_opportunity_penalty": float(hold_penalty),
        }
    best_key = max(scores, key=lambda key: scores[key])
    return float(best_key), scores, components


def compute_expected_alpha_horizon(
    *,
    activity_protocol: str,
    candidate_weights: Any,
    current_weights: Any,
    mu_1d_decision_visible: Any,
    horizon_config: Mapping[str, Any] | None = None,
) -> float:
    protocol = str(activity_protocol)
    if protocol not in VALID_ACTIVITY_PROTOCOLS:
        raise ValueError(f"ERR_ACTIVITY_PROTOCOL_INVALID: {protocol}")
    cfg = mapping(horizon_config)
    default_horizons = {
        "daily_gate_with_cost_constraint": 1,
        "weekly_gate": 5,
        "monthly_gate": 21,
    }
    horizon_days = int(cfg.get("horizon_days", default_horizons[protocol]))
    alpha_cap = float(cfg.get("alpha_cap", 0.20))
    candidate = np.asarray(candidate_weights, dtype=float).reshape(-1)
    current = np.asarray(current_weights, dtype=float).reshape(-1)
    mu = np.asarray(mu_1d_decision_visible, dtype=float).reshape(-1)
    if candidate.shape != current.shape or candidate.shape != mu.shape:
        raise ValueError("ERR_EXPECTED_ALPHA_SHAPE")
    expected_alpha_1d = float(np.dot(candidate - current, np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)))
    return float(np.clip(float(horizon_days) * expected_alpha_1d, -alpha_cap, alpha_cap))


def gate_action_index(rho_values: Sequence[float], rho: float) -> int:
    values = [float(value) for value in rho_values]
    if float(rho) in values:
        return values.index(float(rho))
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(rho))))


def new_model_training_result(
    *,
    model_name: str,
    status: str,
    algorithm: str,
    training_history: pd.DataFrame,
    config: Mapping[str, Any],
    env_steps: int = 0,
    gradient_updates: int = 0,
    checkpoint_best_path: str | None = None,
    checkpoint_last_path: str | None = None,
    evaluated_checkpoint_path: str | None = None,
    best_validation_metric: float | None = None,
) -> dict[str, Any]:
    rankability = mapping(config.get("rankability"))
    return {
        "model_name": model_name,
        "paper_model_id": model_name,
        "child_model_name": model_name,
        "baseline_family": "new_model_extension",
        "status": status,
        "training_algorithm": algorithm,
        "rl_training": True,
        "platform_native_rl_training": True,
        "proxy_training": False,
        "external_original_implementation": False,
        "clean_room_reimplementation": True,
        "algorithm_fidelity": "platform_adapted",
        "rankable_in_unified_table": bool(rankability.get("rankable_in_unified_table", False)),
        "model_extension_id": MODEL_EXTENSION_ID,
        "post_hoc_development_disclosure": True,
        "test_used_for_model_selection": False,
        "data_protocol": "platform",
        "execution_protocol": "platform_backtest_engine",
        "evaluation_protocol": "unified_platform",
        "cost_model_shared": True,
        "cost_availability": "available",
        "constraint_protocol_shared": True,
        "training_history": training_history,
        "checkpoint_best_path": checkpoint_best_path,
        "checkpoint_last_path": checkpoint_last_path,
        "evaluated_checkpoint_path": evaluated_checkpoint_path,
        "best_validation_metric": best_validation_metric,
        "env_steps": int(env_steps),
        "gradient_updates": int(gradient_updates),
        "max_train_steps": _native_cfg_int(config, "max_train_steps"),
        "max_validation_steps": _native_cfg_int(config, "max_validation_steps"),
        "max_gradient_updates_per_epoch": _native_cfg_int(config, "max_gradient_updates_per_epoch"),
    }


def checkpoint_string(value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    return str(path)


def _native_cfg_int(config: Mapping[str, Any], key: str) -> int | None:
    native = mapping(mapping(config.get("baselines")).get("native_rl"))
    value = native.get(key, mapping(config.get("training")).get(key))
    return None if value is None else int(value)


def _rho_key(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _hpo_param_paths(config: Mapping[str, Any]) -> set[str]:
    hpo = mapping(config.get("hpo"))
    search_space = mapping(hpo.get("search_space"))
    best_params = mapping(hpo.get("best_params"))
    return {str(key) for key in (*search_space.keys(), *best_params.keys())}


def _inherit_hpo_param(
    gate: dict[str, Any],
    section: Mapping[str, Any],
    hpo_paths: set[str],
    section_name: str,
    gate_key: str,
    *section_keys: str,
) -> None:
    for section_key in section_keys:
        if f"{section_name}.{section_key}" in hpo_paths and section.get(section_key) is not None:
            gate[gate_key] = section[section_key]
            return


def _positive_scale(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0 or not np.isfinite(result):
        raise ValueError(f"ERR_GATE_SCORING_SCALE_INVALID: {name}")
    return result


def _non_negative_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("ERR_REBALANCE_THRESHOLD_INVALID")
    return result
