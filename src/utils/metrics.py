from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    "n_steps",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "omega",
    "max_drawdown",
    "max_drawdown_abs",
    "max_drawdown_signed",
    "var",
    "cvar",
    "turnover",
    "average_turnover",
    "total_transaction_cost",
    "cost",
    "hit_ratio",
    "final_nav",
)


def calculate_performance_metrics(
    daily_returns: Any,
    daily_turnover: Any | None = None,
    daily_costs: Any | None = None,
    *,
    annualization_factor: int = 252,
    risk_free_rate_annual: float = 0.0,
    var_alpha: float = 0.05,
    cvar_alpha: float = 0.05,
    omega_threshold: float | None = None,
) -> dict[str, float]:
    returns_frame = _required_frame(daily_returns, "daily_returns")
    if "net_return" not in returns_frame.columns:
        raise ValueError("ERR_METRICS_MISSING_NET_RETURN")

    returns = _finite_series(returns_frame["net_return"])
    turnover_frame = _optional_frame(daily_turnover)
    costs_frame = _optional_frame(daily_costs)
    if returns.empty:
        return _empty_metrics()

    nav = pd.Series(np.cumprod(1.0 + returns.to_numpy(dtype=float)))
    final_nav = float(nav.iloc[-1])
    cumulative_return = final_nav - 1.0
    annualized_return = _annualized_return(cumulative_return, len(returns), annualization_factor)
    daily_rf = float(risk_free_rate_annual) / float(annualization_factor)
    excess_returns = returns - daily_rf
    daily_std = float(returns.std(ddof=0))
    annualized_volatility = daily_std * float(np.sqrt(annualization_factor))
    downside = excess_returns[excess_returns < 0.0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    drawdown_signed = nav / nav.cummax() - 1.0
    max_drawdown_signed = float(drawdown_signed.min())
    max_drawdown_abs = abs(max_drawdown_signed)
    var_value = _tail_loss(returns, var_alpha)
    cvar_value = _cvar_loss(returns, cvar_alpha)
    turnover = _turnover(returns_frame, turnover_frame)
    cost = _cost(returns_frame, costs_frame)

    metrics = {
        "n_steps": float(len(returns)),
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": np.nan if daily_std == 0.0 else float(excess_returns.mean() / daily_std * np.sqrt(annualization_factor)),
        "sortino": np.nan if downside_std == 0.0 else float(excess_returns.mean() / downside_std * np.sqrt(annualization_factor)),
        "calmar": np.nan if max_drawdown_abs == 0.0 else float(annualized_return / max_drawdown_abs),
        "omega": _omega_ratio(returns, daily_rf if omega_threshold is None else float(omega_threshold)),
        "max_drawdown": max_drawdown_abs,
        "max_drawdown_abs": max_drawdown_abs,
        "max_drawdown_signed": max_drawdown_signed,
        "var": var_value,
        "cvar": cvar_value,
        "turnover": turnover,
        "average_turnover": turnover,
        "total_transaction_cost": cost,
        "cost": cost,
        "hit_ratio": float((returns > 0.0).mean()),
        "final_nav": final_nav,
    }
    return {column: float(metrics[column]) for column in METRIC_COLUMNS}


def _annualized_return(cumulative_return: float, n_steps: int, annualization_factor: int) -> float:
    if n_steps <= 0:
        return np.nan
    final_nav = 1.0 + float(cumulative_return)
    if final_nav <= 0.0:
        return -1.0
    return float(final_nav ** (float(annualization_factor) / float(n_steps)) - 1.0)


def _tail_loss(returns: pd.Series, alpha: float) -> float:
    threshold = float(returns.quantile(_bounded_alpha(alpha)))
    return max(0.0, -threshold)


def _cvar_loss(returns: pd.Series, alpha: float) -> float:
    threshold = float(returns.quantile(_bounded_alpha(alpha)))
    tail = returns[returns <= threshold]
    if tail.empty:
        return np.nan
    return max(0.0, -float(tail.mean()))


def _omega_ratio(returns: pd.Series, threshold: float) -> float:
    excess = returns - float(threshold)
    gains = float(excess.clip(lower=0.0).sum())
    losses = abs(float(excess.clip(upper=0.0).sum()))
    if losses == 0.0:
        return np.inf if gains > 0.0 else np.nan
    return gains / losses


def _turnover(daily_returns: pd.DataFrame, daily_turnover: pd.DataFrame | None) -> float:
    if "turnover" in daily_returns.columns:
        return _series_mean(daily_returns["turnover"])
    if daily_turnover is not None and "turnover" in daily_turnover.columns:
        return _series_mean(daily_turnover["turnover"])
    return np.nan


def _cost(daily_returns: pd.DataFrame, daily_costs: pd.DataFrame | None) -> float:
    for column in ("transaction_cost", "total_transaction_cost"):
        if column in daily_returns.columns:
            return _series_sum(daily_returns[column])
    if daily_costs is not None:
        for column in ("total_transaction_cost", "realized_cost", "transaction_cost"):
            if column in daily_costs.columns:
                return _series_sum(daily_costs[column])
    return np.nan


def _series_mean(values: Any) -> float:
    series = _finite_series(values)
    return np.nan if series.empty else float(series.mean())


def _series_sum(values: Any) -> float:
    series = _finite_series(values)
    return np.nan if series.empty else float(series.sum())


def _finite_series(values: Any) -> pd.Series:
    series = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _bounded_alpha(alpha: float) -> float:
    return float(np.clip(float(alpha), 1.0e-12, 1.0))


def _required_frame(value: Any, name: str) -> pd.DataFrame:
    frame = _optional_frame(value)
    if frame is None:
        raise ValueError(f"ERR_METRICS_MISSING_{name.upper()}")
    return frame


def _optional_frame(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        return pd.read_csv(value)
    if isinstance(value, Mapping):
        try:
            return pd.DataFrame(dict(value))
        except ValueError:
            return pd.DataFrame([dict(value)])
    if isinstance(value, Sequence):
        return pd.DataFrame(value)
    return pd.DataFrame(value)


def _empty_metrics() -> dict[str, float]:
    result = {column: np.nan for column in METRIC_COLUMNS}
    result["n_steps"] = 0.0
    return result


__all__ = ["METRIC_COLUMNS", "calculate_performance_metrics"]
