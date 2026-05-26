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

from src.config import ConfigLoader  # noqa: E402
from src.experiments.pipeline import run_strategy_comparison, run_trained_model_experiment  # noqa: E402
from src.experiments.registry import BaselineComparisonExperiment, ExperimentRegistry  # noqa: E402
from src.utils.logger import save_json_atomic  # noqa: E402


REFERENCE_MODELS = (
    "eiie_native",
    "ppo_native",
    "cage_eiie_joint_light",
    "full_dqn_gated_multitask_cnn_ppo",
    "cnn_ppo_native",
    "pgportfolio_eiie_native",
    "risk_parity",
)
BASELINE_REFERENCE_MODELS = (
    "eiie_native",
    "ppo_native",
    "cage_eiie_joint_light",
    "cnn_ppo_native",
    "pgportfolio_eiie_native",
    "risk_parity",
)
TRAINED_REFERENCE_MODELS = ("full_dqn_gated_multitask_cnn_ppo",)
PROTOCOL_ID = "core13_v2_full_reset_20260522"
MODEL_EXTENSION_ID = "core13_v2_p16_ra_gt_rcpo_20260525"
DEFAULT_OUTPUT_DIR = "results/paper_tables/p16_validation_references"
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

    baseline_models = [model for model in reference_models if model in BASELINE_REFERENCE_MODELS]
    if baseline_models:
        baseline_config = _reference_config(config, baseline_models)
        experiment = ExperimentRegistry().create_experiment(baseline_config)
        if not isinstance(experiment, BaselineComparisonExperiment):
            raise TypeError("ERR_P16_VALIDATION_REFERENCE_EXPERIMENT_TYPE")
        payload = run_strategy_comparison(
            baseline_config,
            experiment.baselines,
            segment="validation",
            run_dir=str(run_dir / "baseline_reference"),
        )
        _append_payload_frames(
            payload,
            comparison_frames=comparison_frames,
            daily_returns_frames=daily_returns_frames,
            daily_turnover_frames=daily_turnover_frames,
            daily_cost_frames=daily_cost_frames,
        )

    for model_name in [model for model in reference_models if model in TRAINED_REFERENCE_MODELS]:
        payload = run_trained_model_experiment(
            _trained_model_config(config, model_name),
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
    comparison = validation_comparison(
        _concat(comparison_frames),
        daily_returns=daily_returns,
        daily_turnover=daily_turnover,
        daily_costs=daily_costs,
        model_names=reference_models,
    )
    risk_metrics = comparison.loc[
        :,
        [
            "paper_model_id",
            "model_name",
            "seed",
            "split",
            "max_drawdown_loss",
            "CVaR_loss_5",
            "volatility",
            "Sharpe",
            "Sortino",
            "Calmar",
            "source_run_dir",
        ],
    ].copy()
    selection = validation_selection_report(comparison, config_path=config_path)

    paths = {
        "comparison": output / "validation_reference_comparison.csv",
        "daily_returns": output / "validation_reference_daily_returns.csv",
        "risk_metrics": output / "validation_reference_risk_metrics.csv",
        "selection_report": output / "validation_selection_report.csv",
        "manifest": output / "validation_reference_manifest.json",
    }
    comparison.to_csv(paths["comparison"], index=False)
    daily_returns.to_csv(paths["daily_returns"], index=False)
    risk_metrics.to_csv(paths["risk_metrics"], index=False)
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


def validation_comparison(
    comparison: pd.DataFrame,
    *,
    daily_returns: pd.DataFrame,
    daily_turnover: pd.DataFrame,
    daily_costs: pd.DataFrame,
    model_names: Sequence[str],
) -> pd.DataFrame:
    comparison_by_model = _records_by_model(comparison)
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        returns = _model_frame(daily_returns, model_name)
        turnover = _model_frame(daily_turnover, model_name)
        costs = _model_frame(daily_costs, model_name)
        net_returns = pd.to_numeric(returns.get("net_return"), errors="coerce").dropna()
        nav = _nav_series(returns, net_returns)
        drawdown = _drawdown(nav)
        cvar_loss_5 = _cvar_loss(net_returns, 0.05)
        cumulative_return = float(nav.iloc[-1] - 1.0) if not nav.empty else _numeric(comparison_by_model.get(model_name, {}), "cumulative_return")
        average_turnover = _mean_numeric(turnover, "turnover", fallback=_numeric(comparison_by_model.get(model_name, {}), "average_turnover"))
        total_transaction_cost = _sum_numeric(
            costs,
            "total_transaction_cost",
            fallback=_numeric(comparison_by_model.get(model_name, {}), "total_transaction_cost"),
        )
        n_steps = int(len(returns))
        average_cost_per_step = float(total_transaction_cost / n_steps) if n_steps > 0 and np.isfinite(total_transaction_cost) else np.nan
        max_drawdown_loss = float(drawdown.max()) if not drawdown.empty else np.nan
        volatility = float(net_returns.std(ddof=0)) if not net_returns.empty else np.nan
        sharpe = _sharpe(net_returns)
        sortino = _sortino(net_returns)
        calmar = float(cumulative_return / max_drawdown_loss) if max_drawdown_loss and np.isfinite(max_drawdown_loss) else np.nan
        validation_utility = (
            cumulative_return
            - UTILITY_WEIGHTS["alpha_mdd"] * _finite_or_zero(max_drawdown_loss)
            - UTILITY_WEIGHTS["alpha_cvar"] * _finite_or_zero(cvar_loss_5)
            - UTILITY_WEIGHTS["alpha_turnover"] * _finite_or_zero(average_turnover)
            - UTILITY_WEIGHTS["alpha_cost"] * _finite_or_zero(average_cost_per_step)
        )
        row = {
            **comparison_by_model.get(model_name, {}),
            "paper_model_id": comparison_by_model.get(model_name, {}).get("paper_model_id", model_name),
            "model_name": model_name,
            "seed": comparison_by_model.get(model_name, {}).get("seed", _first_frame_value(returns, "seed")),
            "split": "validation",
            "cumulative_return": cumulative_return,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": calmar,
            "volatility": volatility,
            "max_drawdown_loss": max_drawdown_loss,
            "CVaR_loss_5": cvar_loss_5,
            "average_turnover": average_turnover,
            "average_cost_per_step": average_cost_per_step,
            "total_transaction_cost": total_transaction_cost,
            "validation_utility": validation_utility,
            "source_run_dir": comparison_by_model.get(model_name, {}).get("source_run_dir", ""),
            "rankable_reference": True,
            "daily_returns_finite": _finite_frame(returns, ("net_return", "nav")),
            "n_validation_days": n_steps,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def validation_selection_report(comparison: pd.DataFrame, *, config_path: str | Path) -> pd.DataFrame:
    rows = []
    for record in comparison.to_dict("records"):
        rows.append(
            {
                "model_name": record.get("model_name"),
                "paper_model_id": record.get("paper_model_id", record.get("model_name")),
                "selection_split": "validation",
                "selection_metric": "validation_utility",
                "validation_utility": record.get("validation_utility"),
                "cumulative_return": record.get("cumulative_return"),
                "average_turnover": record.get("average_turnover"),
                "average_cost_per_step": record.get("average_cost_per_step"),
                "total_transaction_cost": record.get("total_transaction_cost"),
                "max_drawdown_loss": record.get("max_drawdown_loss"),
                "CVaR_loss_5": record.get("CVaR_loss_5"),
                "config_path": str(config_path),
                "test_used_for_model_selection": False,
            }
        )
    return pd.DataFrame(rows)


def _reference_config(config: Mapping[str, Any], model_names: Sequence[str]) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    native_models = [model for model in model_names if model != "risk_parity"]
    traditional_models = [model for model in model_names if model == "risk_parity"]
    resolved["experiment"] = {"type": "baseline_comparison"}
    resolved["baselines"] = {
        "traditional": traditional_models,
        "deep": [],
        "native_rl": {
            **dict(_get(resolved, "baselines", "native_rl") or {}),
            "enabled_models": native_models,
        },
    }
    resolved.setdefault("hpo", {})
    resolved["hpo"]["enabled"] = False
    resolved.setdefault("new_model_protocol", {})
    resolved["new_model_protocol"]["phase"] = "P16_validation_reference"
    resolved["new_model_protocol"]["model_extension_id"] = MODEL_EXTENSION_ID
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


def _records_by_model(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "model_name" not in frame.columns:
        return {}
    return {
        str(row["model_name"]): row
        for row in frame.to_dict("records")
        if row.get("model_name") is not None
    }


def _model_frame(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if frame.empty or "model_name" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["model_name"].astype(str).eq(str(model_name))].copy()


def _nav_series(returns_frame: pd.DataFrame, net_returns: pd.Series) -> pd.Series:
    if "nav" in returns_frame.columns:
        nav = pd.to_numeric(returns_frame["nav"], errors="coerce").dropna()
        if not nav.empty:
            return nav.reset_index(drop=True)
    return (1.0 + net_returns.reset_index(drop=True)).cumprod()


def _drawdown(nav: pd.Series) -> pd.Series:
    if nav.empty:
        return pd.Series(dtype=float)
    running_max = nav.cummax()
    return ((running_max - nav) / running_max).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _cvar_loss(values: pd.Series, alpha: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    tail_n = max(1, int(np.ceil(float(alpha) * len(clean))))
    return float(max(0.0, -clean.sort_values().iloc[:tail_n].mean()))


def _sharpe(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    std = float(clean.std(ddof=0)) if not clean.empty else np.nan
    if not std or not np.isfinite(std):
        return np.nan
    return float(clean.mean() / std)


def _sortino(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    downside = clean.loc[clean < 0.0]
    std = float(downside.std(ddof=0)) if not downside.empty else np.nan
    if not std or not np.isfinite(std):
        return np.nan
    return float(clean.mean() / std)


def _mean_numeric(frame: pd.DataFrame, column: str, *, fallback: float = np.nan) -> float:
    if frame.empty or column not in frame.columns:
        return fallback
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return fallback if values.empty else float(values.mean())


def _sum_numeric(frame: pd.DataFrame, column: str, *, fallback: float = np.nan) -> float:
    if frame.empty or column not in frame.columns:
        return fallback
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return fallback if values.empty else float(values.sum())


def _numeric(record: Mapping[str, Any], column: str) -> float:
    try:
        return float(record.get(column, np.nan))
    except (TypeError, ValueError):
        return np.nan


def _finite_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _finite_frame(frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    if frame.empty:
        return False
    for column in columns:
        if column not in frame.columns:
            return False
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.empty or not np.isfinite(values).all():
            return False
    return True


def _first_frame_value(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna()
    return None if values.empty else values.iloc[0]


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    selected = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def _frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    return pd.DataFrame(value)


def _get(mapping: Mapping[str, Any], *keys: str) -> Any:
    cursor: Any = mapping
    for key in keys:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(key)
    return cursor


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate P16 validation reference bundle.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    return generate_validation_references(args.config, args.output_dir)


if __name__ == "__main__":
    main()
