from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd


WALK_FORWARD_RESULT_COLUMNS = (
    "fold_id",
    "n_steps",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "cvar",
    "turnover",
    "total_transaction_cost",
    "hit_ratio",
    "duplicate_oos_date_count",
)


def aggregate_walk_forward(
    fold_results: Sequence[Any],
    run_dir: str | Path | None = None,
    *,
    annualization_factor: int = 252,
    cvar_alpha: float = 0.05,
) -> dict[str, Any]:
    fold_metric_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    turnover_frames: list[pd.DataFrame] = []
    cost_frames: list[pd.DataFrame] = []
    for fold_order, fold_result in enumerate(fold_results):
        fold_id = _fold_id(fold_result, fold_order)
        daily_returns = _daily_returns_frame(fold_result, fold_id)
        daily_turnover = _optional_frame(_value(fold_result, "daily_turnover"))
        daily_costs = _optional_frame(_value(fold_result, "daily_costs"))
        daily_returns["_fold_order"] = fold_order
        daily_returns["_row_order"] = np.arange(len(daily_returns), dtype=int)
        daily_frames.append(daily_returns)
        turnover_frame = _oos_side_frame(daily_turnover, fold_id, fold_order)
        if turnover_frame is not None:
            turnover_frames.append(turnover_frame)
        cost_frame = _oos_side_frame(daily_costs, fold_id, fold_order)
        if cost_frame is not None:
            cost_frames.append(cost_frame)

        fold_metrics = _performance_metrics(
            daily_returns,
            annualization_factor=annualization_factor,
            cvar_alpha=cvar_alpha,
            daily_turnover=daily_turnover,
            daily_costs=daily_costs,
        )
        fold_metrics["fold_id"] = fold_id
        fold_metrics["duplicate_oos_date_count"] = 0
        fold_metric_rows.append(_result_row(fold_metrics))

    all_oos_daily_returns, duplicate_count = _all_oos_daily_returns(daily_frames)
    all_oos_daily_turnover, _ = _all_oos_daily_returns(turnover_frames)
    all_oos_daily_costs, _ = _all_oos_daily_returns(cost_frames)
    all_oos_metrics = _performance_metrics(
        all_oos_daily_returns,
        annualization_factor=annualization_factor,
        cvar_alpha=cvar_alpha,
        daily_turnover=all_oos_daily_turnover,
        daily_costs=all_oos_daily_costs,
    )
    all_oos_metrics["fold_id"] = "all_oos"
    all_oos_metrics["duplicate_oos_date_count"] = duplicate_count

    walk_forward_results = pd.DataFrame(
        [*fold_metric_rows, _result_row(all_oos_metrics)],
        columns=WALK_FORWARD_RESULT_COLUMNS,
    )
    metrics_path = None
    if run_dir is not None:
        metrics_path = _write_walk_forward_results(walk_forward_results, Path(run_dir) / "metrics" / "walk_forward_results.csv")
    return {
        "status": "completed",
        "fold_count": len(fold_results),
        "duplicate_oos_date_count": duplicate_count,
        "all_oos_daily_returns": all_oos_daily_returns.drop(columns=["_fold_order", "_row_order"], errors="ignore"),
        "walk_forward_results": walk_forward_results,
        "metrics_path": None if metrics_path is None else str(metrics_path),
    }


def _fold_id(fold_result: Any, fold_order: int) -> str:
    value = _value(fold_result, "fold_id")
    if value is not None:
        return str(value)
    daily_returns = _value(fold_result, "daily_returns")
    if isinstance(daily_returns, pd.DataFrame) and "fold_id" in daily_returns and not daily_returns.empty:
        return str(daily_returns["fold_id"].iloc[0])
    return f"fold_{fold_order + 1}"


def _daily_returns_frame(fold_result: Any, fold_id: str) -> pd.DataFrame:
    frame = _required_frame(_value(fold_result, "daily_returns"), "daily_returns")
    if "date" not in frame.columns and "next_valuation_date" in frame.columns:
        frame["date"] = frame["next_valuation_date"]
    if "date" not in frame.columns or "net_return" not in frame.columns:
        raise ValueError("ERR_WALK_FORWARD_DAILY_RETURNS_SCHEMA")
    if "fold_id" not in frame.columns:
        frame["fold_id"] = fold_id
    else:
        frame["fold_id"] = frame["fold_id"].fillna(fold_id).astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _oos_side_frame(frame: pd.DataFrame | None, fold_id: str, fold_order: int) -> pd.DataFrame | None:
    if frame is None:
        return None
    if "date" not in frame.columns and "next_valuation_date" in frame.columns:
        frame["date"] = frame["next_valuation_date"]
    if "date" not in frame.columns:
        return None
    if "fold_id" not in frame.columns:
        frame["fold_id"] = fold_id
    else:
        frame["fold_id"] = frame["fold_id"].fillna(fold_id).astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["_fold_order"] = fold_order
    frame["_row_order"] = np.arange(len(frame), dtype=int)
    return frame


def _all_oos_daily_returns(daily_frames: Sequence[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    if not daily_frames:
        return pd.DataFrame(columns=["date", "fold_id", "net_return", "_fold_order", "_row_order"]), 0
    combined = pd.concat(daily_frames, ignore_index=True, sort=False)
    duplicate_count = int(combined.duplicated(subset=["date"]).sum())
    retained = (
        combined.sort_values(["_fold_order", "_row_order"], kind="mergesort")
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    return retained, duplicate_count


def _performance_metrics(
    daily_returns: pd.DataFrame,
    *,
    annualization_factor: int,
    cvar_alpha: float,
    daily_turnover: pd.DataFrame | None = None,
    daily_costs: pd.DataFrame | None = None,
) -> dict[str, float]:
    if daily_returns.empty:
        return {
            "n_steps": 0.0,
            "cumulative_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "max_drawdown": np.nan,
            "cvar": np.nan,
            "turnover": np.nan,
            "total_transaction_cost": np.nan,
            "hit_ratio": np.nan,
        }
    returns = pd.to_numeric(daily_returns["net_return"], errors="coerce").astype(float)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n_steps = float(len(returns))
    if returns.empty:
        return _performance_metrics(pd.DataFrame(), annualization_factor=annualization_factor, cvar_alpha=cvar_alpha)

    cumulative_return = float(np.prod(1.0 + returns.to_numpy()) - 1.0)
    annualized_return = float((1.0 + cumulative_return) ** (annualization_factor / len(returns)) - 1.0)
    volatility = float(returns.std(ddof=0) * np.sqrt(annualization_factor))
    mean_return = float(returns.mean())
    sharpe = np.nan if volatility == 0.0 else float(mean_return / returns.std(ddof=0) * np.sqrt(annualization_factor))
    downside = returns[returns < 0.0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    sortino = np.nan if downside_std == 0.0 else float(mean_return / downside_std * np.sqrt(annualization_factor))
    nav = pd.Series(np.cumprod(1.0 + returns.to_numpy()))
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = abs(float(drawdown.min()))
    calmar = np.nan if max_drawdown == 0.0 else float(annualized_return / max_drawdown)
    threshold = float(returns.quantile(cvar_alpha))
    tail = returns[returns <= threshold]
    cvar = float(-tail.mean()) if not tail.empty else np.nan
    hit_ratio = float((returns > 0.0).mean())
    turnover = _turnover_value(daily_returns, daily_turnover)
    total_transaction_cost = _cost_value(daily_returns, daily_costs)
    return {
        "n_steps": n_steps,
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "cvar": cvar,
        "turnover": turnover,
        "total_transaction_cost": total_transaction_cost,
        "hit_ratio": hit_ratio,
    }


def _turnover_value(daily_returns: pd.DataFrame, daily_turnover: pd.DataFrame | None) -> float:
    if "turnover" in daily_returns:
        return float(pd.to_numeric(daily_returns["turnover"], errors="coerce").mean())
    if daily_turnover is not None and "turnover" in daily_turnover:
        return float(pd.to_numeric(daily_turnover["turnover"], errors="coerce").mean())
    return np.nan


def _cost_value(daily_returns: pd.DataFrame, daily_costs: pd.DataFrame | None) -> float:
    if "transaction_cost" in daily_returns:
        return float(pd.to_numeric(daily_returns["transaction_cost"], errors="coerce").sum())
    if "total_transaction_cost" in daily_returns:
        return float(pd.to_numeric(daily_returns["total_transaction_cost"], errors="coerce").sum())
    if daily_costs is not None and "total_transaction_cost" in daily_costs:
        return float(pd.to_numeric(daily_costs["total_transaction_cost"], errors="coerce").sum())
    return np.nan


def _result_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {column: metrics.get(column, np.nan) for column in WALK_FORWARD_RESULT_COLUMNS}


def _required_frame(value: Any, name: str) -> pd.DataFrame:
    frame = _optional_frame(value)
    if frame is None:
        raise ValueError(f"ERR_WALK_FORWARD_MISSING_{name.upper()}")
    return frame


def _optional_frame(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        return pd.read_csv(value)
    return pd.DataFrame(value).copy()


def _value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _write_walk_forward_results(frame: pd.DataFrame, path: Path) -> Path:
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
            frame.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)
        return target
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


__all__ = ["WALK_FORWARD_RESULT_COLUMNS", "aggregate_walk_forward"]
