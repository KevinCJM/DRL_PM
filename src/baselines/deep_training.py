from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

VALID_CURRENT_WEIGHT_MODES = frozenset({"equal_weight", "rolling_equal_weight"})


@dataclass(frozen=True)
class DeepBaselineTrainingConfig:
    enabled: bool = True
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    max_samples: int = 512
    turnover_penalty: float = 0.0
    prior_blend_weight: float = 0.5
    current_weight_mode: str = "rolling_equal_weight"


@dataclass(frozen=True)
class DeepBaselineTrainingBatch:
    market_image: torch.Tensor
    availability_mask: torch.Tensor
    current_weights: torch.Tensor
    equal_weights: torch.Tensor
    future_returns: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.market_image.shape[0])


def deep_baseline_training_config(config: Mapping[str, Any]) -> DeepBaselineTrainingConfig:
    baselines = _mapping(config.get("baselines"))
    deep_training = _mapping(baselines.get("deep_training"))
    return DeepBaselineTrainingConfig(
        enabled=bool(deep_training.get("enabled", True)),
        epochs=max(0, int(deep_training.get("epochs", 1))),
        batch_size=max(1, int(deep_training.get("batch_size", 32))),
        learning_rate=float(deep_training.get("learning_rate", 1.0e-3)),
        max_samples=max(0, int(deep_training.get("max_samples", 512))),
        turnover_penalty=float(deep_training.get("turnover_penalty", 0.0)),
        prior_blend_weight=_bounded_float(deep_training.get("prior_blend_weight", 0.5), 0.0, 1.0),
        current_weight_mode=_current_weight_mode(deep_training.get("current_weight_mode", "rolling_equal_weight")),
    )


def collect_training_batch(
    train_data: Any | None,
    *,
    n_features: int,
    window_size: int,
    n_assets: int,
    device: torch.device,
    max_samples: int,
) -> DeepBaselineTrainingBatch | None:
    if max_samples == 0 or not isinstance(train_data, Mapping):
        return None
    dataset = train_data.get("dataset")
    dates = train_data.get("dates")
    if dataset is None or dates is None:
        return None
    wide = getattr(dataset, "wide", {})
    if not isinstance(wide, Mapping):
        return None
    asset_order = _asset_order(dataset, n_assets)
    close = _wide_frame(wide, "close", asset_order)
    if close is None:
        return None
    date_index = pd.DatetimeIndex(close.index)
    requested_dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    market_image_dataset = train_data.get("market_image_dataset")
    config = _mapping(train_data.get("config"))
    training_config = deep_baseline_training_config(config)
    current_weights_by_date = _current_weights_by_date(
        dataset,
        wide,
        asset_order,
        date_index,
        config,
        training_config.current_weight_mode,
    )

    samples: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    current_weights: list[np.ndarray] = []
    future_returns: list[np.ndarray] = []
    for decision_date in requested_dates:
        try:
            position = int(np.flatnonzero(date_index == pd.Timestamp(decision_date))[0])
        except IndexError:
            continue
        if position < int(window_size) - 1 or position + 1 >= len(date_index):
            continue
        mask = _availability_mask(dataset, asset_order, decision_date)
        if not mask.any():
            continue
        image = _market_image(
            dataset,
            market_image_dataset,
            asset_order,
            position,
            pd.Timestamp(decision_date),
            int(n_features),
            int(window_size),
        )
        if image is None:
            continue
        returns = _execution_aligned_future_returns(wide, asset_order, position, date_index, config)
        if returns is None:
            continue
        current = current_weights_by_date.get(pd.Timestamp(decision_date), masked_equal_weights(mask))
        samples.append(image.astype(np.float32, copy=False))
        masks.append(mask.astype(bool, copy=False))
        current_weights.append(current.astype(np.float32, copy=False))
        future_returns.append(returns)

    if not samples:
        return None
    if len(samples) > max_samples:
        indices = np.linspace(0, len(samples) - 1, max_samples, dtype=int)
        samples = [samples[int(index)] for index in indices]
        masks = [masks[int(index)] for index in indices]
        current_weights = [current_weights[int(index)] for index in indices]
        future_returns = [future_returns[int(index)] for index in indices]

    mask_array = np.stack(masks, axis=0)
    equal_weights = np.stack([masked_equal_weights(mask) for mask in mask_array], axis=0)
    current_weights_array = np.stack(current_weights, axis=0)
    return DeepBaselineTrainingBatch(
        market_image=torch.as_tensor(np.stack(samples, axis=0), dtype=torch.float32, device=device),
        availability_mask=torch.as_tensor(mask_array, dtype=torch.bool, device=device),
        current_weights=torch.as_tensor(current_weights_array, dtype=torch.float32, device=device),
        equal_weights=torch.as_tensor(equal_weights, dtype=torch.float32, device=device),
        future_returns=torch.as_tensor(np.stack(future_returns, axis=0), dtype=torch.float32, device=device),
    )


def masked_equal_weights(mask: np.ndarray) -> np.ndarray:
    available = np.asarray(mask, dtype=bool)
    weights = np.zeros(available.shape, dtype=np.float32)
    if available.any():
        weights[available] = 1.0 / float(available.sum())
    return weights


def iter_minibatches(batch: DeepBaselineTrainingBatch, batch_size: int):
    for start in range(0, batch.size, int(batch_size)):
        stop = min(start + int(batch_size), batch.size)
        yield slice(start, stop)


def training_summary(
    status: str,
    *,
    samples: int = 0,
    loss: float | None = None,
    current_weight_mode: str = "rolling_equal_weight",
) -> dict[str, Any]:
    configured_mode = _current_weight_mode(current_weight_mode)
    effective_mode = _effective_current_weight_mode(configured_mode)
    return {
        "status": status,
        "samples": int(samples),
        "sample_count": int(samples),
        "loss": None if loss is None else float(loss),
        "baseline_family": "neural_proxy",
        "training_algorithm": "supervised_execution_aligned_proxy",
        "rl_training": False,
        "platform_native_rl_training": False,
        "native_rl_training": False,
        "proxy_training": True,
        "external_original_implementation": False,
        "rankable_in_unified_table": False,
        "return_target_mode": "execution_holding_simple_return",
        "configured_current_weight_mode": configured_mode,
        "effective_current_weight_mode": effective_mode,
        "current_weight_mode": effective_mode,
        "execution_path_proxy": True,
        "pending_action_queue_simulated": False,
    }


def execution_aligned_future_return_frame(train_data: Any | None, n_assets: int) -> pd.DataFrame | None:
    if not isinstance(train_data, Mapping):
        return None
    dataset = train_data.get("dataset")
    dates = train_data.get("dates")
    if dataset is None or dates is None:
        return None
    wide = getattr(dataset, "wide", {})
    if not isinstance(wide, Mapping):
        return None
    asset_order = _asset_order(dataset, n_assets)
    close = _wide_frame(wide, "close", asset_order)
    if close is None:
        return None
    date_index = pd.DatetimeIndex(close.index)
    config = _mapping(train_data.get("config"))
    rows: dict[pd.Timestamp, np.ndarray] = {}
    for decision_date in pd.DatetimeIndex(pd.to_datetime(list(dates))):
        try:
            position = int(np.flatnonzero(date_index == pd.Timestamp(decision_date))[0])
        except IndexError:
            continue
        returns = _execution_aligned_future_returns(wide, asset_order, position, date_index, config)
        if returns is not None:
            rows[pd.Timestamp(decision_date)] = returns
    if not rows:
        return None
    return pd.DataFrame.from_dict(rows, orient="index", columns=asset_order).sort_index()


def execution_aligned_return_component_frames(
    train_data: Any | None,
    n_assets: int,
) -> dict[str, pd.DataFrame] | None:
    if not isinstance(train_data, Mapping):
        return None
    dataset = train_data.get("dataset")
    dates = train_data.get("dates")
    if dataset is None or dates is None:
        return None
    wide = getattr(dataset, "wide", {})
    if not isinstance(wide, Mapping):
        return None
    asset_order = _asset_order(dataset, n_assets)
    close = _wide_frame(wide, "close", asset_order)
    if close is None:
        return None
    date_index = pd.DatetimeIndex(close.index)
    config = _mapping(train_data.get("config"))
    pre_rows: dict[pd.Timestamp, np.ndarray] = {}
    holding_rows: dict[pd.Timestamp, np.ndarray] = {}
    for decision_date in pd.DatetimeIndex(pd.to_datetime(list(dates))):
        try:
            position = int(np.flatnonzero(date_index == pd.Timestamp(decision_date))[0])
        except IndexError:
            continue
        components = _execution_aligned_return_components(wide, asset_order, position, date_index, config)
        if components is None:
            continue
        pre_rows[pd.Timestamp(decision_date)] = components["pre_execution_returns"]
        holding_rows[pd.Timestamp(decision_date)] = components["holding_returns"]
    if not pre_rows:
        return None
    return {
        "pre_execution_returns": pd.DataFrame.from_dict(pre_rows, orient="index", columns=asset_order).sort_index(),
        "holding_returns": pd.DataFrame.from_dict(holding_rows, orient="index", columns=asset_order).sort_index(),
    }


def _market_image(
    dataset: Any,
    market_image_dataset: Any | None,
    asset_order: list[str],
    position: int,
    decision_date: pd.Timestamp,
    n_features: int,
    window_size: int,
) -> np.ndarray | None:
    if market_image_dataset is not None:
        try:
            image = np.asarray(market_image_dataset[decision_date], dtype=np.float32)
            if image.shape == (n_features, window_size, len(asset_order)):
                return image
        except Exception:
            pass
    feature_cols = [str(item) for item in getattr(dataset, "feature_cols", [])]
    frames = []
    for feature in feature_cols:
        wide = getattr(dataset, "wide", {})
        if isinstance(wide, Mapping) and feature in wide:
            frame = wide[feature].reindex(columns=asset_order)
            window = frame.iloc[position - window_size + 1 : position + 1].to_numpy(dtype=np.float32, copy=True)
            frames.append(np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0))
    if len(frames) == n_features:
        return np.stack(frames, axis=0)
    wide = getattr(dataset, "wide", {})
    if isinstance(wide, Mapping) and "log_return" in wide and n_features == 1:
        frame = wide["log_return"].reindex(columns=asset_order)
        return np.nan_to_num(
            frame.iloc[position - window_size + 1 : position + 1].to_numpy(dtype=np.float32, copy=True)[np.newaxis, :, :],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    return None


def _execution_aligned_future_returns(
    wide: Mapping[str, Any],
    asset_order: list[str],
    position: int,
    date_index: pd.DatetimeIndex,
    config: Mapping[str, Any],
) -> np.ndarray | None:
    components = _execution_aligned_return_components(wide, asset_order, position, date_index, config)
    if components is None:
        return None
    return components["holding_returns"]


def _execution_aligned_return_components(
    wide: Mapping[str, Any],
    asset_order: list[str],
    position: int,
    date_index: pd.DatetimeIndex,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray] | None:
    close = _wide_frame(wide, "close", asset_order)
    if close is None:
        return None
    execution_config = _mapping(config.get("execution_model"))
    data_governance = _mapping(config.get("data_governance"))
    execution_price = str(execution_config.get("execution_price", "next_open"))
    same_close_enabled = bool(
        execution_config.get("same_close_idealized_execution_enabled", False)
        or execution_config.get("idealized_execution", False)
        or data_governance.get("same_close_idealized_execution_enabled", False)
    )
    if same_close_enabled:
        if position + 1 >= len(date_index):
            return None
        zeros = np.zeros(len(asset_order), dtype=np.float32)
        holding = _safe_simple_return(close.iloc[position + 1], close.iloc[position])
        return {"pre_execution_returns": zeros, "holding_returns": holding}
    if execution_price == "next_open":
        open_frame = _wide_frame(wide, "open", asset_order)
        if open_frame is None or position + 1 >= len(date_index):
            return None
        pre_execution = _safe_simple_return(open_frame.iloc[position + 1], close.iloc[position])
        holding = _safe_simple_return(close.iloc[position + 1], open_frame.iloc[position + 1])
        return {"pre_execution_returns": pre_execution, "holding_returns": holding}
    if execution_price == "next_close":
        if position + 2 >= len(date_index):
            return None
        pre_execution = _safe_simple_return(close.iloc[position + 1], close.iloc[position])
        holding = _safe_simple_return(close.iloc[position + 2], close.iloc[position + 1])
        return {"pre_execution_returns": pre_execution, "holding_returns": holding}
    return None


def _current_weights_by_date(
    dataset: Any,
    wide: Mapping[str, Any],
    asset_order: list[str],
    date_index: pd.DatetimeIndex,
    config: Mapping[str, Any],
    mode: str,
) -> dict[pd.Timestamp, np.ndarray]:
    weights_by_date: dict[pd.Timestamp, np.ndarray] = {}
    if len(date_index) == 0:
        return weights_by_date
    first_mask = _availability_mask(dataset, asset_order, pd.Timestamp(date_index[0]))
    current = masked_equal_weights(first_mask)
    for position, date in enumerate(date_index):
        timestamp = pd.Timestamp(date)
        mask = _availability_mask(dataset, asset_order, timestamp)
        if mode == "equal_weight":
            weights_by_date[timestamp] = masked_equal_weights(mask)
        else:
            weights_by_date[timestamp] = _enforce_masked_weights(current, mask)
        returns = _execution_aligned_future_returns(wide, asset_order, position, date_index, config)
        if returns is None:
            continue
        rebalance_weights = masked_equal_weights(mask)
        current = _drift_weights(rebalance_weights, returns)

    return weights_by_date


def _drift_weights(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    gross = np.asarray(weights, dtype=np.float32) * (1.0 + np.asarray(returns, dtype=np.float32))
    total = float(np.nansum(gross))
    if not np.isfinite(total) or total <= 0.0:
        return np.asarray(weights, dtype=np.float32).copy()
    return (gross / total).astype(np.float32, copy=False)


def _enforce_masked_weights(weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(weights, dtype=np.float32).copy()
    available = np.asarray(mask, dtype=bool)
    result[~available] = 0.0
    total = float(np.nansum(result))
    if not np.isfinite(total) or total <= 0.0:
        return masked_equal_weights(available)
    return (result / total).astype(np.float32, copy=False)


def _wide_frame(wide: Mapping[str, Any], field: str, asset_order: list[str]) -> pd.DataFrame | None:
    frame = wide.get(field)
    if frame is None:
        return None
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    return result.sort_index().reindex(columns=asset_order)


def _safe_simple_return(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    top = numerator.to_numpy(dtype=np.float32, copy=True)
    bottom = denominator.to_numpy(dtype=np.float32, copy=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = top / bottom - 1.0
    return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _availability_mask(dataset: Any, asset_order: list[str], decision_date: pd.Timestamp) -> np.ndarray:
    availability = getattr(dataset, "availability_mask", None)
    if availability is None:
        return np.ones(len(asset_order), dtype=bool)
    return availability.reindex(columns=asset_order).loc[pd.Timestamp(decision_date)].to_numpy(dtype=bool, copy=True)


def _asset_order(dataset: Any, n_assets: int) -> list[str]:
    manifest = getattr(dataset, "data_manifest", {})
    order = manifest.get("canonical_asset_order") if isinstance(manifest, Mapping) else None
    if isinstance(order, list) and order:
        return [str(item) for item in order[: int(n_assets)]]
    availability = getattr(dataset, "availability_mask", None)
    if availability is not None:
        return [str(item) for item in availability.columns[: int(n_assets)]]
    return [str(index) for index in range(int(n_assets))]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bounded_float(value: Any, lower: float, upper: float) -> float:
    result = float(value)
    if result < lower:
        return float(lower)
    if result > upper:
        return float(upper)
    return result


def _current_weight_mode(value: Any) -> str:
    mode = str(value)
    if mode in VALID_CURRENT_WEIGHT_MODES:
        return mode
    allowed = ", ".join(sorted(VALID_CURRENT_WEIGHT_MODES))
    raise ValueError(f"ERR_DEEP_BASELINE_CURRENT_WEIGHT_MODE_INVALID: {mode!r}; allowed={allowed}")


def _effective_current_weight_mode(mode: str) -> str:
    if mode == "rolling_equal_weight":
        return "rolling_equal_weight_proxy"
    if mode == "equal_weight":
        return "equal_weight"
    raise ValueError(f"ERR_DEEP_BASELINE_CURRENT_WEIGHT_MODE_INVALID: {mode!r}")


__all__ = [
    "DeepBaselineTrainingBatch",
    "DeepBaselineTrainingConfig",
    "collect_training_batch",
    "deep_baseline_training_config",
    "execution_aligned_return_component_frames",
    "execution_aligned_future_return_frame",
    "iter_minibatches",
    "masked_equal_weights",
    "training_summary",
]
