from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigLoader
from src.experiments.pipeline import run_strategy_comparison, run_trained_model_experiment
from src.experiments.registry import BaselineComparisonExperiment, ExperimentRegistry
from src.utils.logger import save_json_atomic


REFERENCE_MODELS = (
    "eiie_native",
    "full_dqn_gated_multitask_cnn_ppo",
    "ppo_dqn_hierarchical_reimplementation",
    "cnn_ppo_native",
    "pgportfolio_eiie_native",
)
NATIVE_REFERENCE_MODELS = (
    "eiie_native",
    "ppo_dqn_hierarchical_reimplementation",
    "cnn_ppo_native",
    "pgportfolio_eiie_native",
)
TRAINED_REFERENCE_MODELS = ("full_dqn_gated_multitask_cnn_ppo",)
MODEL_EXTENSION_ID = "core13_v2_p12_p13_20260524"
PROTOCOL_ID = "core13_v2_full_reset_20260522"
DEFAULT_OUTPUT_DIR = "results/paper_tables/p12_p13_validation_references"
UTILITY_WEIGHTS = {
    "alpha_mdd": 0.25,
    "alpha_cvar": 0.25,
    "alpha_turnover": 2.0,
    "alpha_cost": 10.0,
}


def generate_validation_references(
    config_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    reference_models: Sequence[str] = REFERENCE_MODELS,
) -> dict[str, Path]:
    config = ConfigLoader.load(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / "reference_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    comparison_frames: list[pd.DataFrame] = []
    daily_returns_frames: list[pd.DataFrame] = []
    daily_turnover_frames: list[pd.DataFrame] = []
    daily_cost_frames: list[pd.DataFrame] = []

    native_models = [model for model in reference_models if model in NATIVE_REFERENCE_MODELS]
    if native_models:
        native_config = _reference_config(config, native_models)
        experiment = ExperimentRegistry().create_experiment(native_config)
        if not isinstance(experiment, BaselineComparisonExperiment):
            raise TypeError("ERR_VALIDATION_REFERENCE_EXPERIMENT_TYPE")
        native_payload = run_strategy_comparison(
            native_config,
            experiment.baselines,
            segment="validation",
            run_dir=str(run_dir / "native_reference"),
        )
        _append_payload_frames(
            native_payload,
            comparison_frames=comparison_frames,
            daily_returns_frames=daily_returns_frames,
            daily_turnover_frames=daily_turnover_frames,
            daily_cost_frames=daily_cost_frames,
        )

    for model_name in [model for model in reference_models if model in TRAINED_REFERENCE_MODELS]:
        model_config = _trained_model_config(config, model_name)
        payload = run_trained_model_experiment(
            model_config,
            model_name=model_name,
            test_split="validation",
            run_dir=str(run_dir / model_name),
        )
        _append_payload_frames(
            payload,
            comparison_frames=comparison_frames,
            daily_returns_frames=daily_returns_frames,
            daily_turnover_frames=daily_turnover_frames,
            daily_cost_frames=daily_cost_frames,
        )

    daily_returns = _concat(daily_returns_frames)
    daily_turnover = _concat(daily_turnover_frames)
    daily_costs = _concat(daily_cost_frames)
    comparison = _validation_comparison(
        _concat(comparison_frames),
        daily_returns=daily_returns,
        daily_turnover=daily_turnover,
        daily_costs=daily_costs,
        reference_models=reference_models,
    )
    selection = _validation_selection_report(comparison, config_path=config_path)

    paths = {
        "comparison": output / "validation_reference_comparison.csv",
        "daily_returns": output / "validation_reference_daily_returns.csv",
        "daily_turnover": output / "validation_reference_daily_turnover.csv",
        "daily_costs": output / "validation_reference_daily_costs.csv",
        "selection_report": output / "validation_selection_report.csv",
        "manifest": output / "validation_reference_manifest.json",
    }
    comparison.to_csv(paths["comparison"], index=False)
    daily_returns.to_csv(paths["daily_returns"], index=False)
    daily_turnover.to_csv(paths["daily_turnover"], index=False)
    daily_costs.to_csv(paths["daily_costs"], index=False)
    selection.to_csv(paths["selection_report"], index=False)
    save_json_atomic(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "protocol_id": _get(config, "protocol", "protocol_id") or PROTOCOL_ID,
            "model_extension_id": MODEL_EXTENSION_ID,
            "selection_split": "validation",
            "test_used_for_model_selection": False,
            "reference_models": list(reference_models),
            "utility_weights": UTILITY_WEIGHTS,
            "outputs": {key: str(path) for key, path in paths.items()},
        },
        paths["manifest"],
    )
    return paths


def _reference_config(config: Mapping[str, Any], model_names: Sequence[str]) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    resolved["experiment"] = {"type": "baseline_comparison"}
    resolved["baselines"] = {
        "traditional": [],
        "deep": [],
        "native_rl": {
            **dict(_get(resolved, "baselines", "native_rl") or {}),
            "enabled_models": list(model_names),
        },
    }
    resolved.setdefault("hpo", {})
    resolved["hpo"]["enabled"] = False
    resolved.setdefault("new_model_protocol", {})
    resolved["new_model_protocol"]["phase"] = "P12_P13_validation_reference"
    resolved["new_model_protocol"]["selection_split"] = "validation"
    resolved["new_model_protocol"]["test_used_for_model_selection"] = False
    resolved.setdefault("rankability", {})
    resolved["rankability"]["rankable_in_unified_table"] = False
    resolved["rankability"]["diagnostic_status"] = "validation_reference"
    return resolved


def _trained_model_config(config: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    resolved = _reference_config(config, [])
    model = dict(resolved.get("model", {}))
    model["name"] = model_name
    resolved["model"] = model
    return resolved


def _append_payload_frames(
    payload: Mapping[str, Any],
    *,
    comparison_frames: list[pd.DataFrame],
    daily_returns_frames: list[pd.DataFrame],
    daily_turnover_frames: list[pd.DataFrame],
    daily_cost_frames: list[pd.DataFrame],
) -> None:
    for key in ("baseline_comparison", "main_comparison"):
        frame = _frame(payload.get(key))
        if not frame.empty:
            comparison_frames.append(frame)
    daily_returns_frames.append(_frame(payload.get("daily_returns")))
    daily_turnover_frames.append(_frame(payload.get("daily_turnover")))
    daily_cost_frames.append(_frame(payload.get("daily_costs")))


def _validation_comparison(
    comparison: pd.DataFrame,
    *,
    daily_returns: pd.DataFrame,
    daily_turnover: pd.DataFrame,
    daily_costs: pd.DataFrame,
    reference_models: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparison_by_model = _records_by_model(comparison)
    for model_name in reference_models:
        returns = _model_frame(daily_returns, model_name)
        turnover = _model_frame(daily_turnover, model_name)
        costs = _model_frame(daily_costs, model_name)
        net_returns = pd.to_numeric(returns.get("net_return"), errors="coerce").dropna()
        nav = _nav_series(returns, net_returns)
        drawdown = _drawdown(nav)
        cvar_loss_5 = _cvar_loss(net_returns, 0.05)
        cumulative_return = float(nav.iloc[-1] - 1.0) if not nav.empty else _numeric(comparison_by_model.get(model_name, {}), "cumulative_return")
        turnover_mean = _mean_numeric(turnover, "turnover", fallback=_numeric(comparison_by_model.get(model_name, {}), "average_turnover"))
        transaction_cost_total = _sum_numeric(
            costs,
            "total_transaction_cost",
            fallback=_numeric(comparison_by_model.get(model_name, {}), "total_transaction_cost"),
        )
        max_drawdown_loss = float(drawdown.max()) if not drawdown.empty else np.nan
        validation_utility = (
            cumulative_return
            - UTILITY_WEIGHTS["alpha_mdd"] * _finite_or_zero(max_drawdown_loss)
            - UTILITY_WEIGHTS["alpha_cvar"] * _finite_or_zero(cvar_loss_5)
            - UTILITY_WEIGHTS["alpha_turnover"] * _finite_or_zero(turnover_mean)
            - UTILITY_WEIGHTS["alpha_cost"] * _finite_or_zero(transaction_cost_total)
        )
        row = {
            **comparison_by_model.get(model_name, {}),
            "model_name": model_name,
            "paper_model_id": comparison_by_model.get(model_name, {}).get("paper_model_id", model_name),
            "selection_split": "validation",
            "test_used_for_model_selection": False,
            "cumulative_return": cumulative_return,
            "turnover_mean": turnover_mean,
            "average_turnover": turnover_mean,
            "transaction_cost_total": transaction_cost_total,
            "total_transaction_cost": transaction_cost_total,
            "max_drawdown_loss": max_drawdown_loss,
            "CVaR_loss_5": cvar_loss_5,
            "validation_return_cost_risk_utility": validation_utility,
            "daily_returns_finite": _finite_frame(returns, ("net_return", "nav")),
            "daily_nav_finite": _finite_frame(returns, ("nav",)),
            "n_validation_days": int(len(returns)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _validation_selection_report(comparison: pd.DataFrame, *, config_path: str | Path) -> pd.DataFrame:
    rows = []
    for record in comparison.to_dict("records"):
        rows.append(
            {
                "model_name": record.get("model_name"),
                "paper_model_id": record.get("paper_model_id", record.get("model_name")),
                "selection_split": "validation",
                "selection_metric": "validation_return_cost_risk_utility",
                "validation_return_cost_risk_utility": record.get("validation_return_cost_risk_utility"),
                "cumulative_return": record.get("cumulative_return"),
                "turnover_mean": record.get("turnover_mean"),
                "transaction_cost_total": record.get("transaction_cost_total"),
                "max_drawdown_loss": record.get("max_drawdown_loss"),
                "CVaR_loss_5": record.get("CVaR_loss_5"),
                "test_used_for_model_selection": False,
                "config_path": str(config_path),
                "model_extension_id": MODEL_EXTENSION_ID,
            }
        )
    return pd.DataFrame(rows)


def _records_by_model(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "model_name" not in frame.columns:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict("records"):
        result[str(record.get("model_name"))] = record
    return result


def _model_frame(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if frame.empty or "model_name" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["model_name"].astype(str).eq(model_name)].copy()


def _nav_series(returns: pd.DataFrame, net_returns: pd.Series) -> pd.Series:
    if "nav" in returns.columns:
        nav = pd.to_numeric(returns["nav"], errors="coerce").dropna()
        if not nav.empty:
            return nav.reset_index(drop=True)
    if net_returns.empty:
        return pd.Series(dtype=float)
    return (1.0 + net_returns.reset_index(drop=True)).cumprod()


def _drawdown(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return pd.Series(dtype=float)
    running_max = nav.cummax()
    return ((running_max - nav) / running_max).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _cvar_loss(values: pd.Series, alpha: float) -> float:
    if values.empty:
        return np.nan
    sorted_values = np.sort(values.to_numpy(dtype=float))
    count = max(1, int(np.ceil(alpha * len(sorted_values))))
    return float(max(0.0, -np.mean(sorted_values[:count])))


def _mean_numeric(frame: pd.DataFrame, column: str, *, fallback: Any = np.nan) -> float:
    if frame.empty or column not in frame.columns:
        return float(fallback) if pd.notna(fallback) else np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else (float(fallback) if pd.notna(fallback) else np.nan)


def _sum_numeric(frame: pd.DataFrame, column: str, *, fallback: Any = np.nan) -> float:
    if frame.empty or column not in frame.columns:
        return float(fallback) if pd.notna(fallback) else np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.sum()) if not values.empty else (float(fallback) if pd.notna(fallback) else np.nan)


def _numeric(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _finite_frame(frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    if frame.empty:
        return False
    for column in columns:
        if column not in frame.columns:
            return False
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.empty or not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            return False
    return True


def _finite_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _frame(value: Any) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    selected = [frame for frame in frames if isinstance(frame, pd.DataFrame) and (not frame.empty or len(frame.columns) > 0)]
    return pd.concat(selected, ignore_index=True, sort=False) if selected else pd.DataFrame()


def _get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate P12/P13 validation-only reference tables.")
    parser.add_argument("--config", default="configs/paper/p12_cage_eiie_formal_comparison.yaml")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = generate_validation_references(args.config, args.output_dir)
    print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
