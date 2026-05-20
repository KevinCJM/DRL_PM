from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


STATISTICS_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "benchmark_name",
    "test_name",
    "metric_name",
    "effect_size",
    "ci_lower",
    "ci_upper",
    "test_statistic",
    "raw_p_value",
    "adjusted_p_value",
    "adjustment_method",
    "significant",
    "status",
    "skip_reason",
    "n_obs",
    "fail_reason",
)
RETURN_TESTS: tuple[str, ...] = (
    "block_bootstrap",
    "hac",
    "psr",
    "dsr",
    "white_reality_check",
    "hansen_spa",
)
DM_TEST_NAME = "diebold_mariano"
_NORMAL = NormalDist()


def run_statistical_tests(
    model_returns: Any,
    benchmark_returns: Any,
    *,
    config: Mapping[str, Any] | None = None,
    auxiliary_forecast_errors: Any = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    stats_config = _stats_config(config)
    min_samples = int(stats_config.get("min_paired_samples", 20))
    alpha = float(_mapping(stats_config.get("multiple_testing")).get("alpha", 0.05))
    model_frames = _named_return_frames(model_returns, default_name="model")
    benchmark_frames = _named_return_frames(benchmark_returns, default_name="benchmark")
    rows: list[dict[str, Any]] = []
    paired_by_benchmark: dict[str, list[dict[str, Any]]] = {}

    for model_name, model_frame in model_frames.items():
        for benchmark_name, benchmark_frame in benchmark_frames.items():
            paired = _paired_returns(model_frame, benchmark_frame)
            diff = paired["diff"].to_numpy(dtype=float) if len(paired) else np.array([], dtype=float)
            pair = {
                "model_name": model_name,
                "benchmark_name": benchmark_name,
                "paired": paired,
                "diff": diff,
                "model_returns": paired["model_return"].to_numpy(dtype=float) if len(paired) else np.array([], dtype=float),
                "benchmark_returns": paired["benchmark_return"].to_numpy(dtype=float) if len(paired) else np.array([], dtype=float),
            }
            paired_by_benchmark.setdefault(benchmark_name, []).append(pair)
            if len(diff) < min_samples:
                rows.extend(_skipped_rows(model_name, benchmark_name, RETURN_TESTS, len(diff)))
            else:
                rows.append(_block_bootstrap_row(pair, stats_config))
                rows.append(_hac_row(pair, stats_config))
                rows.append(_psr_row(pair, stats_config))
                rows.append(_dsr_row(pair, stats_config, n_strategies=max(1, len(model_frames))))
                rows.append(_row(pair, "white_reality_check", "net_return_diff", effect_size=_mean(diff)))
                rows.append(_row(pair, "hansen_spa", "net_return_diff", effect_size=_mean(diff)))
            rows.append(_dm_row(pair, auxiliary_forecast_errors, stats_config, min_samples))

    for benchmark_name, pairs in paired_by_benchmark.items():
        valid_pairs = [pair for pair in pairs if len(pair["diff"]) >= min_samples]
        group_rows = _reality_check_rows(valid_pairs, stats_config)
        if group_rows:
            _replace_group_rows(rows, benchmark_name, "white_reality_check", group_rows["white_reality_check"])
            _replace_group_rows(rows, benchmark_name, "hansen_spa", group_rows["hansen_spa"])

    frame = pd.DataFrame(rows, columns=STATISTICS_SUMMARY_COLUMNS)
    frame = _apply_holm_bonferroni(frame, alpha=alpha)
    if output_path is not None:
        _write_csv_atomic(frame, output_path)
    return frame


def _block_bootstrap_row(pair: Mapping[str, Any], stats_config: Mapping[str, Any]) -> dict[str, Any]:
    diff = np.asarray(pair["diff"], dtype=float)
    try:
        bootstrap = _bootstrap_means(diff, stats_config)
        bootstrap_config = _mapping(stats_config.get("bootstrap"))
        confidence = float(bootstrap_config.get("confidence_level", stats_config.get("confidence_level", 0.95)))
    except Exception as exc:
        return _row(pair, "block_bootstrap", "net_return_diff", effect_size=_mean(diff), status="failed", fail_reason=str(exc))
    lower_q = (1.0 - confidence) / 2.0
    upper_q = 1.0 - lower_q
    effect = _mean(diff)
    raw_p = _bootstrap_p_value(bootstrap, effect)
    return _row(
        pair,
        "block_bootstrap",
        "net_return_diff",
        effect_size=effect,
        ci_lower=float(np.quantile(bootstrap, lower_q)),
        ci_upper=float(np.quantile(bootstrap, upper_q)),
        raw_p_value=raw_p,
    )


def _hac_row(pair: Mapping[str, Any], stats_config: Mapping[str, Any]) -> dict[str, Any]:
    diff = np.asarray(pair["diff"], dtype=float)
    effect = _mean(diff)
    se = _newey_west_se(diff, _hac_lags(len(diff), stats_config))
    stat = np.nan if se <= 0.0 or not np.isfinite(se) else effect / se
    raw_p = np.nan if not np.isfinite(stat) else 2.0 * (1.0 - _NORMAL.cdf(abs(float(stat))))
    return _row(pair, "hac", "net_return_diff", effect_size=effect, test_statistic=stat, raw_p_value=raw_p)


def _psr_row(pair: Mapping[str, Any], stats_config: Mapping[str, Any]) -> dict[str, Any]:
    returns = np.asarray(pair["model_returns"], dtype=float)
    benchmark_returns = np.asarray(pair["benchmark_returns"], dtype=float)
    sr = _sharpe(returns)
    benchmark_sr = _first_not_none(stats_config.get("benchmark_sharpe"), _sharpe(benchmark_returns))
    psr = _probabilistic_sharpe_ratio(returns, float(benchmark_sr))
    raw_p = np.nan if not np.isfinite(psr) else 1.0 - float(psr)
    return _row(pair, "psr", "sharpe", effect_size=sr - float(benchmark_sr), test_statistic=psr, raw_p_value=raw_p)


def _dsr_row(pair: Mapping[str, Any], stats_config: Mapping[str, Any], *, n_strategies: int) -> dict[str, Any]:
    returns = np.asarray(pair["model_returns"], dtype=float)
    benchmark_returns = np.asarray(pair["benchmark_returns"], dtype=float)
    sr = _sharpe(returns)
    benchmark_sr = _first_not_none(stats_config.get("benchmark_sharpe"), _sharpe(benchmark_returns))
    psr = _probabilistic_sharpe_ratio(returns, float(benchmark_sr))
    raw_p = np.nan if not np.isfinite(psr) else min(1.0, (1.0 - float(psr)) * float(max(1, n_strategies)))
    dsr = np.nan if not np.isfinite(raw_p) else 1.0 - raw_p
    return _row(pair, "dsr", "sharpe", effect_size=sr - float(benchmark_sr), test_statistic=dsr, raw_p_value=raw_p)


def _reality_check_rows(pairs: Sequence[Mapping[str, Any]], stats_config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if not pairs:
        return {}
    common = _common_diff_matrix(pairs)
    if common.size == 0:
        return {}
    observed = common.mean(axis=0)
    centered = common - observed.reshape(1, -1)
    bootstrap = _bootstrap_matrix_means(centered, stats_config)
    max_boot = bootstrap.max(axis=1)
    spa_source = common - np.minimum(observed, 0.0).reshape(1, -1)
    spa_boot = _bootstrap_matrix_means(spa_source - spa_source.mean(axis=0).reshape(1, -1), stats_config).max(axis=1)
    white_rows: list[dict[str, Any]] = []
    spa_rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        p_white = float(np.mean(max_boot >= observed[idx]))
        p_spa = float(np.mean(spa_boot >= max(observed[idx], 0.0)))
        white_rows.append(
            _row(pair, "white_reality_check", "net_return_diff", effect_size=float(pair["diff"].mean()), raw_p_value=p_white)
        )
        spa_rows.append(_row(pair, "hansen_spa", "net_return_diff", effect_size=float(pair["diff"].mean()), raw_p_value=p_spa))
    return {"white_reality_check": white_rows, "hansen_spa": spa_rows}


def _dm_row(pair: Mapping[str, Any], auxiliary_forecast_errors: Any, stats_config: Mapping[str, Any], min_samples: int) -> dict[str, Any]:
    dm_data = _dm_loss_diff(auxiliary_forecast_errors, pair["model_name"], pair["benchmark_name"], stats_config)
    if dm_data is None:
        return _row(pair, DM_TEST_NAME, "auxiliary_forecast_error", status="not_applicable")
    loss_diff, horizon = dm_data
    if loss_diff.size < min_samples:
        return _row(pair, DM_TEST_NAME, "auxiliary_forecast_error", status="skipped", skip_reason="insufficient_samples", n_obs=loss_diff.size)
    effect = _mean(loss_diff)
    se = _newey_west_se(loss_diff, max(0, int(horizon) - 1))
    stat = np.nan if se <= 0.0 or not np.isfinite(se) else effect / se
    raw_p = np.nan if not np.isfinite(stat) else 2.0 * (1.0 - _NORMAL.cdf(abs(float(stat))))
    return _row(pair, DM_TEST_NAME, "auxiliary_forecast_error", effect_size=effect, test_statistic=stat, raw_p_value=raw_p, n_obs=loss_diff.size)


def _paired_returns(model_frame: pd.DataFrame, benchmark_frame: pd.DataFrame) -> pd.DataFrame:
    model = _return_frame(model_frame, "model_return")
    benchmark = _return_frame(benchmark_frame, "benchmark_return")
    paired = model.merge(benchmark, on="date", how="inner").dropna(subset=["model_return", "benchmark_return"])
    paired["diff"] = paired["model_return"] - paired["benchmark_return"]
    return paired.sort_values("date").reset_index(drop=True)


def _return_frame(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
    data = _oos_frame(frame)
    if "date" not in data.columns:
        data = data.copy()
        data["date"] = data.index
    value_column = "net_return" if "net_return" in data.columns else value_name
    if value_column not in data.columns:
        raise ValueError(f"ERR_STATS_MISSING_NET_RETURN: {value_name}")
    out = data.loc[:, ["date", value_column]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_column], errors="coerce")
    return out.loc[:, ["date", value_name]].dropna(subset=["date"]).sort_values("date")


def _oos_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "split" not in frame.columns:
        return frame.copy()
    split = frame["split"].astype(str).str.lower()
    mask = split.isin({"test", "oos", "out_of_sample", "all_oos"})
    return frame.loc[mask].copy() if mask.any() else frame.copy()


def _named_return_frames(value: Any, *, default_name: str) -> dict[str, pd.DataFrame]:
    if isinstance(value, Mapping):
        frames: dict[str, pd.DataFrame] = {}
        for key, item in value.items():
            frame = _frame(item)
            if not frame.empty:
                frames[str(key)] = frame
        if frames:
            return frames
    frame = _frame(value)
    if "model_name" in frame.columns:
        return {str(name): group.copy() for name, group in frame.groupby("model_name", sort=False)}
    if "benchmark_name" in frame.columns:
        return {str(name): group.copy() for name, group in frame.groupby("benchmark_name", sort=False)}
    return {default_name: frame}


def _dm_loss_diff(auxiliary_forecast_errors: Any, model_name: str, benchmark_name: str, stats_config: Mapping[str, Any]) -> tuple[np.ndarray, int] | None:
    if auxiliary_forecast_errors is None:
        return None
    frame = _frame(auxiliary_forecast_errors)
    if frame.empty:
        return None
    horizon = int(_first_not_none(frame["horizon"].dropna().iloc[0] if "horizon" in frame.columns and frame["horizon"].notna().any() else None, stats_config.get("dm_horizon"), 1))
    if {"model_error", "benchmark_error"}.issubset(frame.columns):
        if "model_name" in frame.columns:
            frame = frame.loc[frame["model_name"].astype(str).eq(str(model_name))]
        if "benchmark_name" in frame.columns:
            frame = frame.loc[frame["benchmark_name"].astype(str).eq(str(benchmark_name))]
        model_error = pd.to_numeric(frame["model_error"], errors="coerce")
        benchmark_error = pd.to_numeric(frame["benchmark_error"], errors="coerce")
        valid = model_error.notna() & benchmark_error.notna()
        if not valid.any():
            return None
        return _loss(model_error[valid].to_numpy(dtype=float), stats_config) - _loss(benchmark_error[valid].to_numpy(dtype=float), stats_config), horizon
    if {"date", "forecast_error", "model_name"}.issubset(frame.columns):
        model = frame.loc[frame["model_name"].astype(str).eq(str(model_name)), ["date", "forecast_error"]].rename(columns={"forecast_error": "model_error"})
        benchmark = frame.loc[frame["model_name"].astype(str).eq(str(benchmark_name)), ["date", "forecast_error"]].rename(columns={"forecast_error": "benchmark_error"})
        merged = model.merge(benchmark, on="date", how="inner")
        if merged.empty:
            return None
        return _loss(pd.to_numeric(merged["model_error"], errors="coerce").to_numpy(dtype=float), stats_config) - _loss(pd.to_numeric(merged["benchmark_error"], errors="coerce").to_numpy(dtype=float), stats_config), horizon
    return None


def _loss(errors: np.ndarray, stats_config: Mapping[str, Any]) -> np.ndarray:
    if str(stats_config.get("dm_loss", "squared")).lower() == "absolute":
        return np.abs(errors)
    return errors * errors


def _bootstrap_means(values: np.ndarray, stats_config: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n_bootstrap = _n_bootstrap(stats_config)
    block_length = _block_length(len(values), stats_config)
    rng = np.random.default_rng(_seed(stats_config))
    indices = _moving_block_indices(len(values), block_length, n_bootstrap, rng)
    return values[indices].mean(axis=1)


def _bootstrap_matrix_means(values: np.ndarray, stats_config: Mapping[str, Any]) -> np.ndarray:
    n_bootstrap = _n_bootstrap(stats_config)
    block_length = _block_length(values.shape[0], stats_config)
    rng = np.random.default_rng(_seed(stats_config))
    indices = _moving_block_indices(values.shape[0], block_length, n_bootstrap, rng)
    return values[indices, :].mean(axis=1)


def _moving_block_indices(n_obs: int, block_length: int, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n_obs, size=(n_bootstrap, int(math.ceil(n_obs / block_length))))
    offsets = np.arange(block_length, dtype=int)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_obs
    return indices.reshape(n_bootstrap, -1)[:, :n_obs]


def _bootstrap_p_value(bootstrap: np.ndarray, observed: float) -> float:
    centered = bootstrap - float(np.mean(bootstrap))
    return float(np.mean(np.abs(centered) >= abs(float(observed))))


def _newey_west_se(values: np.ndarray, lags: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan
    centered = x - float(np.mean(x))
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for lag in range(1, min(int(lags), n - 1) + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    return math.sqrt(max(variance, 0.0) / n)


def _hac_lags(n_obs: int, stats_config: Mapping[str, Any]) -> int:
    hac = _mapping(stats_config.get("hac"))
    if "lags" in hac and hac["lags"] is not None:
        return max(0, int(hac["lags"]))
    return max(0, int(math.floor(4.0 * (float(n_obs) / 100.0) ** (2.0 / 9.0))))


def _sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    std = float(np.std(values, ddof=0))
    return np.nan if std == 0.0 else float(np.mean(values) / std)


def _probabilistic_sharpe_ratio(values: np.ndarray, benchmark_sharpe: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    sr = _sharpe(values)
    if n < 2 or not np.isfinite(sr):
        return np.nan
    centered = values - float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if std == 0.0:
        return np.nan
    skew = float(np.mean((centered / std) ** 3.0))
    kurt = float(np.mean((centered / std) ** 4.0))
    denom = math.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr, 1.0e-12))
    z_value = (sr - float(benchmark_sharpe)) * math.sqrt(n - 1.0) / denom
    return float(_NORMAL.cdf(z_value))


def _common_diff_matrix(pairs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    frames = []
    for pair in pairs:
        paired = pair["paired"].loc[:, ["date", "diff"]].copy()
        paired = paired.rename(columns={"diff": str(pair["model_name"])})
        frames.append(paired)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner")
    if merged.empty:
        return np.array([], dtype=float)
    return merged.drop(columns=["date"]).to_numpy(dtype=float)


def _replace_group_rows(rows: list[dict[str, Any]], benchmark_name: str, test_name: str, replacements: Sequence[dict[str, Any]]) -> None:
    replacement_map = {row["model_name"]: row for row in replacements}
    for index, row in enumerate(rows):
        if row["benchmark_name"] == benchmark_name and row["test_name"] == test_name and row["model_name"] in replacement_map:
            rows[index] = replacement_map[row["model_name"]]


def _apply_holm_bonferroni(frame: pd.DataFrame, *, alpha: float) -> pd.DataFrame:
    result = frame.copy()
    result["adjusted_p_value"] = pd.NA
    result["adjustment_method"] = ""
    result["significant"] = False
    valid = result["status"].eq("pass") & pd.to_numeric(result["raw_p_value"], errors="coerce").notna()
    for (benchmark_name, test_name), group in result.loc[valid].groupby(["benchmark_name", "test_name"], sort=False):
        ordered = group.assign(_p=pd.to_numeric(group["raw_p_value"], errors="coerce")).sort_values("_p")
        m = len(ordered)
        previous = 0.0
        for rank, (idx, item) in enumerate(ordered.iterrows(), start=1):
            adjusted = min(1.0, max(previous, float(item["_p"]) * (m - rank + 1)))
            previous = adjusted
            result.at[idx, "adjusted_p_value"] = adjusted
            result.at[idx, "adjustment_method"] = "holm_bonferroni"
            result.at[idx, "significant"] = bool(adjusted <= alpha)
    return result.loc[:, list(STATISTICS_SUMMARY_COLUMNS)]


def _skipped_rows(model_name: str, benchmark_name: str, tests: Sequence[str], n_obs: int) -> list[dict[str, Any]]:
    pair = {"model_name": model_name, "benchmark_name": benchmark_name, "diff": np.array([], dtype=float)}
    return [
        _row(pair, test, "net_return_diff" if test in {"block_bootstrap", "hac", "white_reality_check", "hansen_spa"} else "sharpe", status="skipped", skip_reason="insufficient_samples", n_obs=n_obs)
        for test in tests
    ]


def _row(
    pair: Mapping[str, Any],
    test_name: str,
    metric_name: str,
    *,
    effect_size: float = np.nan,
    ci_lower: float = np.nan,
    ci_upper: float = np.nan,
    test_statistic: float = np.nan,
    raw_p_value: float = np.nan,
    status: str = "pass",
    skip_reason: str = "",
    n_obs: int | None = None,
    fail_reason: str = "",
) -> dict[str, Any]:
    diff = np.asarray(pair.get("diff", np.array([], dtype=float)), dtype=float)
    return {
        "model_name": str(pair["model_name"]),
        "benchmark_name": str(pair["benchmark_name"]),
        "test_name": test_name,
        "metric_name": metric_name,
        "effect_size": effect_size,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "test_statistic": test_statistic,
        "raw_p_value": raw_p_value,
        "adjusted_p_value": pd.NA,
        "adjustment_method": "",
        "significant": False,
        "status": status,
        "skip_reason": skip_reason,
        "n_obs": int(diff.size if n_obs is None else n_obs),
        "fail_reason": fail_reason,
    }


def _stats_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    value = config.get("statistics", config)
    return value if isinstance(value, Mapping) else {}


def _n_bootstrap(stats_config: Mapping[str, Any]) -> int:
    bootstrap = _mapping(stats_config.get("bootstrap"))
    return max(1, int(bootstrap.get("n_bootstrap", stats_config.get("n_bootstrap", 1000))))


def _block_length(n_obs: int, stats_config: Mapping[str, Any]) -> int:
    bootstrap = _mapping(stats_config.get("bootstrap"))
    configured = bootstrap.get("block_length", stats_config.get("block_length"))
    if configured is None:
        return max(1, int(round(math.sqrt(float(max(1, n_obs))))))
    return max(1, min(int(configured), max(1, n_obs)))


def _seed(stats_config: Mapping[str, Any]) -> int | None:
    bootstrap = _mapping(stats_config.get("bootstrap"))
    seed = bootstrap.get("seed", stats_config.get("seed"))
    return None if seed is None else int(seed)


def _frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, (str, Path)):
        return pd.read_csv(value)
    if isinstance(value, Mapping):
        try:
            return pd.DataFrame(dict(value))
        except ValueError:
            return pd.DataFrame([dict(value)])
    return pd.DataFrame(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.nan if values.size == 0 else float(np.mean(values))


def _write_csv_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
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


__all__ = ["DM_TEST_NAME", "RETURN_TESTS", "STATISTICS_SUMMARY_COLUMNS", "run_statistical_tests"]
