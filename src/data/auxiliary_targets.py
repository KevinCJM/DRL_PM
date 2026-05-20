from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import DataContractError, MarketDatasetBundle


AUXILIARY_TARGET_COLUMNS = [
    "future_log_return_5d",
    "future_volatility_20d",
    "future_trend_10d",
    "future_cross_sectional_rank",
    "future_downside_volatility",
    "future_max_drawdown",
    "future_CVaR",
    "future_correlation_or_covariance",
    "masked_feature_reconstruction",
]
MASKED_RECONSTRUCTION_METADATA_COLUMNS = [
    "masked_feature_reconstruction_mask",
    "masked_feature_reconstruction_feature",
    "masked_feature_reconstruction_window_offset",
    "masked_feature_reconstruction_asset",
]


@dataclass(frozen=True)
class AuxiliaryTargetResult:
    targets: pd.DataFrame
    target_cols: list[str]
    metadata_cols: list[str]
    purged_dates: pd.DatetimeIndex


def build_auxiliary_targets(
    bundle_or_wide: MarketDatasetBundle | Mapping[str, pd.DataFrame],
    split: Any,
    config: Mapping[str, Any] | None = None,
) -> AuxiliaryTargetResult:
    wide = bundle_or_wide.wide if isinstance(bundle_or_wide, MarketDatasetBundle) else bundle_or_wide
    if "close" not in wide or "log_return" not in wide:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            "ERR_DATA_SCHEMA_MISMATCH: auxiliary targets require close and log_return",
        )

    asset_order = _asset_order(bundle_or_wide, wide)
    close = _wide_frame(wide["close"], asset_order)
    log_return = _wide_frame(wide["log_return"], asset_order)
    simple_return = _simple_return(wide, asset_order)
    segments = _date_segments(split)
    max_horizon = _max_target_horizon(config)
    valid_dates = _valid_target_dates(close.index, segments, max_horizon)

    target_frames = _target_frames(close, log_return, simple_return, config)
    metadata_frames = _masked_reconstruction_metadata_frames(log_return)
    purged_dates = pd.DatetimeIndex([date for date in close.index if date in segments and date not in set(valid_dates)])
    target_panel = _frames_to_panel(target_frames, asset_order)
    metadata_panel = _frames_to_panel(metadata_frames, asset_order)
    target_panel = target_panel.merge(metadata_panel, on=["date", "ts_code"], how="left")
    target_panel = target_panel.loc[target_panel["date"].isin(valid_dates)].reset_index(drop=True)
    return AuxiliaryTargetResult(
        targets=target_panel,
        target_cols=list(target_frames.keys()),
        metadata_cols=list(metadata_frames.keys()),
        purged_dates=purged_dates,
    )


def attach_auxiliary_targets(
    bundle: MarketDatasetBundle,
    targets: AuxiliaryTargetResult,
) -> MarketDatasetBundle:
    auxiliary_target_cols = list(
        dict.fromkeys([*bundle.auxiliary_target_cols, *targets.target_cols, *targets.metadata_cols])
    )
    return replace(bundle, auxiliary_target_cols=auxiliary_target_cols)


def _target_frames(
    close: pd.DataFrame,
    log_return: pd.DataFrame,
    simple_return: pd.DataFrame,
    config: Mapping[str, Any] | None,
) -> dict[str, pd.DataFrame]:
    cvar_alpha = _cvar_alpha(config)
    future_log_return_5d = np.log(close.shift(-5) / close)
    future_simple_20d = _future_window(simple_return, 20)
    future_close_path_20d = _future_window(close, 20, include_current=True)

    return {
        "future_log_return_5d": future_log_return_5d,
        "future_volatility_20d": _future_std(future_simple_20d, close.index, close.columns),
        "future_trend_10d": (np.log(close.shift(-10) / close) > 0.0).where(close.shift(-10).notna()).astype(float),
        "future_cross_sectional_rank": future_log_return_5d.rank(axis=1, pct=True),
        "future_downside_volatility": _future_downside_volatility(future_simple_20d, close.index, close.columns),
        "future_max_drawdown": _future_max_drawdown(future_close_path_20d, close.index, close.columns),
        "future_CVaR": _future_cvar_loss(future_simple_20d, close.index, close.columns, cvar_alpha),
        "future_correlation_or_covariance": _future_covariance_to_benchmark(future_simple_20d, close.index, close.columns),
        "masked_feature_reconstruction": log_return.copy(),
    }


def _masked_reconstruction_metadata_frames(log_return: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mask = log_return.notna().astype(float)
    feature = pd.DataFrame("log_return", index=log_return.index, columns=log_return.columns)
    window_offset = pd.DataFrame(0.0, index=log_return.index, columns=log_return.columns)
    asset = pd.DataFrame(
        {column: str(column) for column in log_return.columns},
        index=log_return.index,
    )
    return {
        "masked_feature_reconstruction_mask": mask,
        "masked_feature_reconstruction_feature": feature,
        "masked_feature_reconstruction_window_offset": window_offset,
        "masked_feature_reconstruction_asset": asset,
    }


def _future_window(frame: pd.DataFrame, horizon: int, *, include_current: bool = False) -> np.ndarray:
    start = 0 if include_current else 1
    return np.stack(
        [frame.shift(-step).to_numpy(dtype=float, copy=True) for step in range(start, horizon + 1)],
        axis=0,
    )


def _future_std(values: np.ndarray, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    valid = np.isfinite(values).all(axis=0)
    output = np.full(values.shape[1:], np.nan, dtype=float)
    if valid.any():
        output[valid] = np.std(values[:, valid], axis=0, ddof=1)
    return pd.DataFrame(output, index=index, columns=columns)


def _future_downside_volatility(values: np.ndarray, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    downside = np.where(values < 0.0, values, 0.0)
    valid = np.isfinite(values).all(axis=0)
    output = np.full(values.shape[1:], np.nan, dtype=float)
    if valid.any():
        output[valid] = np.std(downside[:, valid], axis=0, ddof=1)
    return pd.DataFrame(output, index=index, columns=columns)


def _future_max_drawdown(values: np.ndarray, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    valid = np.isfinite(values).all(axis=0)
    running_max = np.maximum.accumulate(values, axis=0)
    drawdown = 1.0 - values / running_max
    output = np.full(values.shape[1:], np.nan, dtype=float)
    if valid.any():
        output[valid] = np.max(drawdown[:, valid], axis=0)
    return pd.DataFrame(output, index=index, columns=columns)


def _future_cvar_loss(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    alpha: float,
) -> pd.DataFrame:
    valid = np.isfinite(values).all(axis=0)
    tail_count = max(1, int(np.ceil(values.shape[0] * alpha)))
    output = np.full(values.shape[1:], np.nan, dtype=float)
    if valid.any():
        sorted_returns = np.sort(values[:, valid], axis=0)
        output[valid] = -np.mean(sorted_returns[:tail_count], axis=0)
    return pd.DataFrame(output, index=index, columns=columns)


def _future_covariance_to_benchmark(values: np.ndarray, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    valid = np.isfinite(values).all(axis=0)
    output = np.full(values.shape[1:], np.nan, dtype=float)
    benchmark = np.full(values.shape[:2], np.nan, dtype=float)
    valid_benchmark = np.isfinite(values).all(axis=(0, 2))
    for date_index in np.where(valid_benchmark)[0]:
        benchmark[:, date_index] = values[:, date_index, :].mean(axis=1)
    for asset_index in range(values.shape[2]):
        valid_asset = valid[:, asset_index] & valid_benchmark
        if not valid_asset.any():
            continue
        asset_values = values[:, valid_asset, asset_index]
        benchmark_values = benchmark[:, valid_asset]
        asset_mean = asset_values.mean(axis=0)
        benchmark_mean = benchmark_values.mean(axis=0)
        output[valid_asset, asset_index] = ((asset_values - asset_mean) * (benchmark_values - benchmark_mean)).mean(axis=0)
    return pd.DataFrame(output, index=index, columns=columns)


def _frames_to_panel(frames: Mapping[str, pd.DataFrame], asset_order: Sequence[str]) -> pd.DataFrame:
    rows = None
    for name, frame in frames.items():
        aligned = frame.reindex(columns=list(asset_order))
        long = (
            aligned.rename_axis(index="date", columns="ts_code")
            .reset_index()
            .melt(id_vars="date", var_name="ts_code", value_name=name)
        )
        if rows is None:
            rows = long
        else:
            rows = rows.merge(long, on=["date", "ts_code"], how="left")
    if rows is None:
        return pd.DataFrame(columns=["date", "ts_code", *AUXILIARY_TARGET_COLUMNS])
    rows["date"] = pd.to_datetime(rows["date"])
    rows["ts_code"] = rows["ts_code"].astype(str)
    return rows[["date", "ts_code", *frames.keys()]]


def _valid_target_dates(
    dates: pd.DatetimeIndex,
    segments: Mapping[pd.Timestamp, str],
    horizon: int,
) -> pd.DatetimeIndex:
    valid_dates = []
    date_positions = {pd.Timestamp(date): position for position, date in enumerate(dates)}
    for date, segment in segments.items():
        position = date_positions.get(pd.Timestamp(date))
        if position is None:
            continue
        future_dates = dates[position + 1 : position + horizon + 1]
        if len(future_dates) == horizon and all(segments.get(pd.Timestamp(future_date)) == segment for future_date in future_dates):
            valid_dates.append(pd.Timestamp(date))
    return pd.DatetimeIndex(valid_dates)


def _date_segments(split: Any) -> dict[pd.Timestamp, str]:
    if isinstance(split, Sequence) and not isinstance(split, (str, bytes, pd.DatetimeIndex)):
        segments: dict[pd.Timestamp, str] = {}
        for item in split:
            segments.update(_date_segments(item))
        return segments

    segments = {}
    for name in ("train", "validation", "test"):
        dates = getattr(split, f"{name}_dates", None)
        if dates is None:
            continue
        segment_name = f"{getattr(split, 'fold_id', 'fixed')}:{name}"
        for date in pd.DatetimeIndex(pd.to_datetime(list(dates))):
            segments[pd.Timestamp(date)] = segment_name
    return segments


def _max_target_horizon(config: Mapping[str, Any] | None) -> int:
    return_horizons = [5]
    volatility_horizons = [20]
    if isinstance(config, Mapping):
        auxiliary = config.get("auxiliary", {})
        if isinstance(auxiliary, Mapping):
            return_horizons = [int(item) for item in auxiliary.get("future_return_horizons", return_horizons)]
            volatility_horizons = [int(item) for item in auxiliary.get("future_volatility_horizons", volatility_horizons)]
    return max([10, 20, *return_horizons, *volatility_horizons])


def _cvar_alpha(config: Mapping[str, Any] | None) -> float:
    if isinstance(config, Mapping):
        cvar_config = config.get("distributional_cvar", {})
        if isinstance(cvar_config, Mapping):
            return float(cvar_config.get("cvar_alpha", 0.05))
    return 0.05


def _simple_return(wide: Mapping[str, pd.DataFrame], asset_order: Sequence[str]) -> pd.DataFrame:
    if "pct_chg" in wide:
        return _wide_frame(wide["pct_chg"], asset_order)
    log_return = _wide_frame(wide["log_return"], asset_order)
    return np.expm1(log_return)


def _wide_frame(frame: pd.DataFrame, asset_order: Sequence[str]) -> pd.DataFrame:
    aligned = frame.copy()
    aligned.index = pd.DatetimeIndex(pd.to_datetime(aligned.index))
    aligned.columns = [str(column) for column in aligned.columns]
    return aligned.sort_index().reindex(columns=list(asset_order)).replace([np.inf, -np.inf], np.nan)


def _asset_order(
    bundle_or_wide: MarketDatasetBundle | Mapping[str, pd.DataFrame],
    wide: Mapping[str, pd.DataFrame],
) -> list[str]:
    if isinstance(bundle_or_wide, MarketDatasetBundle):
        manifest_order = bundle_or_wide.data_manifest.get("canonical_asset_order")
        if isinstance(manifest_order, list) and manifest_order:
            return [str(asset) for asset in manifest_order]
        return [str(asset) for asset in bundle_or_wide.availability_mask.columns]
    return [str(asset) for asset in wide["close"].columns]
