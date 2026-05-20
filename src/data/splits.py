from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.config import assert_path_allowed
from src.data.loader import DataContractError


@dataclass(frozen=True)
class SplitSpec:
    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    fold_id: int | str
    train_last_decision_date: pd.Timestamp | None = None
    validation_last_decision_date: pd.Timestamp | None = None
    test_last_decision_date: pd.Timestamp | None = None


def create_split(
    trade_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    config: Mapping[str, Any],
) -> SplitSpec | list[SplitSpec]:
    split_config = _resolve_split_config(config)
    dates = _normalize_trade_dates(trade_dates)
    mode = split_config.get("mode", "fixed")
    if mode == "fixed":
        return _create_fixed_split(dates, split_config, config)
    if mode == "purged":
        return _create_fixed_split(dates, split_config, config, force_purge=True)
    if mode == "embargo":
        return _create_fixed_split(dates, split_config, config, force_embargo=True)
    if mode == "walk_forward":
        return _create_walk_forward_splits(dates, split_config, config)
    raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: split.mode={mode}")


def write_data_split_json(
    split: SplitSpec | Sequence[SplitSpec],
    path: str | Path,
    config: Mapping[str, Any],
) -> None:
    import json

    whitelist = config["security"]["path_whitelist"]
    output_path = assert_path_allowed(path, whitelist, "data_split_path")
    payload = split_to_dict(split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def split_to_dict(split: SplitSpec | Sequence[SplitSpec]) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(split, SplitSpec):
        return _split_spec_to_dict(split)
    return [_split_spec_to_dict(item) for item in split]


def _resolve_split_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    split_config = config.get("split", config)
    if not isinstance(split_config, Mapping):
        raise DataContractError("ERR_SPLIT_EMPTY", "ERR_SPLIT_EMPTY: split")
    return split_config


def _normalize_trade_dates(trade_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(list(trade_dates)))
    if dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise DataContractError("ERR_SPLIT_EMPTY", "ERR_SPLIT_EMPTY: trade_dates")
    return dates


def _create_fixed_split(
    dates: pd.DatetimeIndex,
    split_config: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    force_purge: bool = False,
    force_embargo: bool = False,
) -> SplitSpec:
    train_ratio = float(split_config.get("train_ratio", 0.70))
    validation_ratio = float(split_config.get("validation_ratio", 0.15))
    train_end = int(len(dates) * train_ratio)
    validation_end = train_end + int(len(dates) * validation_ratio)
    spec = SplitSpec(
        train_dates=dates[:train_end],
        validation_dates=dates[train_end:validation_end],
        test_dates=dates[validation_end:],
        fold_id="fixed",
    )
    return _finalize_split(spec, split_config, config, force_purge=force_purge, force_embargo=force_embargo)


def _create_walk_forward_splits(
    dates: pd.DatetimeIndex,
    split_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[SplitSpec]:
    walk_forward_config = split_config.get("walk_forward", {})
    train_years = int(walk_forward_config.get("train_years", 3))
    validation_months = int(walk_forward_config.get("validation_months", 6))
    test_months = int(walk_forward_config.get("test_months", 6))
    step_months = int(walk_forward_config.get("step_months", 6))

    fold_start = dates[0]
    fold_id = 0
    splits: list[SplitSpec] = []
    while fold_start < dates[-1]:
        train_end = fold_start + pd.DateOffset(years=train_years)
        validation_end = train_end + pd.DateOffset(months=validation_months)
        test_end = validation_end + pd.DateOffset(months=test_months)
        if test_end > dates[-1] + pd.Timedelta(days=1):
            break

        spec = SplitSpec(
            train_dates=dates[(dates >= fold_start) & (dates < train_end)],
            validation_dates=dates[(dates >= train_end) & (dates < validation_end)],
            test_dates=dates[(dates >= validation_end) & (dates < test_end)],
            fold_id=fold_id,
        )
        if not spec.train_dates.empty and not spec.validation_dates.empty and not spec.test_dates.empty:
            splits.append(_finalize_split(spec, split_config, config))
            fold_id += 1

        fold_start = fold_start + pd.DateOffset(months=step_months)

    if not splits:
        raise DataContractError("ERR_SPLIT_EMPTY", "ERR_SPLIT_EMPTY: walk_forward")
    return splits


def _finalize_split(
    spec: SplitSpec,
    split_config: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    force_purge: bool = False,
    force_embargo: bool = False,
) -> SplitSpec:
    purge_days = _effective_purge_days(split_config, config, force_purge)
    embargo_days = _effective_embargo_days(split_config, config, force_embargo)
    if purge_days < 0 or embargo_days < 0:
        raise DataContractError("ERR_SPLIT_EMPTY", "ERR_SPLIT_EMPTY: purge_or_embargo")
    boundary_tail_gap = max(purge_days, embargo_days)

    train_dates = _drop_tail(spec.train_dates, boundary_tail_gap)
    validation_dates = _drop_tail(_drop_head(spec.validation_dates, embargo_days), boundary_tail_gap)
    test_dates = _drop_head(spec.test_dates, embargo_days)
    finalized = SplitSpec(
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
        fold_id=spec.fold_id,
        train_last_decision_date=_last_decision_date(train_dates, config),
        validation_last_decision_date=_last_decision_date(validation_dates, config),
        test_last_decision_date=_last_decision_date(test_dates, config),
    )
    _assert_non_empty(finalized)
    return finalized


def _effective_purge_days(
    split_config: Mapping[str, Any],
    config: Mapping[str, Any],
    force_purge: bool,
) -> int:
    explicit_purge_days = int(split_config.get("purge_days", 0))
    if explicit_purge_days != 0:
        return explicit_purge_days
    if force_purge:
        auxiliary_config = config.get("auxiliary", {})
        if isinstance(auxiliary_config, Mapping):
            return int(auxiliary_config.get("purge_horizon_days", 0))
    return 0


def _effective_embargo_days(
    split_config: Mapping[str, Any],
    config: Mapping[str, Any],
    force_embargo: bool,
) -> int:
    explicit_embargo_days = int(split_config.get("embargo_days", 0))
    if explicit_embargo_days != 0:
        return explicit_embargo_days
    if force_embargo:
        auxiliary_config = config.get("auxiliary", {})
        if isinstance(auxiliary_config, Mapping):
            return int(auxiliary_config.get("purge_horizon_days", 0))
    return 0


def _drop_head(dates: pd.DatetimeIndex, n_dates: int) -> pd.DatetimeIndex:
    if n_dates == 0:
        return dates
    return dates[n_dates:]


def _drop_tail(dates: pd.DatetimeIndex, n_dates: int) -> pd.DatetimeIndex:
    if n_dates == 0:
        return dates
    return dates[:-n_dates]


def _assert_non_empty(spec: SplitSpec) -> None:
    if spec.train_dates.empty or spec.validation_dates.empty or spec.test_dates.empty:
        raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: fold_id={spec.fold_id}")
    if (
        spec.train_last_decision_date is None
        or spec.validation_last_decision_date is None
        or spec.test_last_decision_date is None
    ):
        raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: fold_id={spec.fold_id}")


def _last_decision_date(dates: pd.DatetimeIndex, config: Mapping[str, Any]) -> pd.Timestamp | None:
    execution_model = config.get("execution_model", {})
    strict_no_lookahead = False
    if isinstance(execution_model, Mapping):
        strict_no_lookahead = bool(execution_model.get("strict_no_lookahead_execution", False))
    if not strict_no_lookahead:
        return None if dates.empty else dates[-1]
    if len(dates) <= 2:
        return None
    return dates[-3]


def _split_spec_to_dict(spec: SplitSpec) -> dict[str, Any]:
    return {
        "fold_id": spec.fold_id,
        "train_dates": [date.strftime("%Y-%m-%d") for date in spec.train_dates],
        "validation_dates": [date.strftime("%Y-%m-%d") for date in spec.validation_dates],
        "test_dates": [date.strftime("%Y-%m-%d") for date in spec.test_dates],
        "train_last_decision_date": _date_or_none(spec.train_last_decision_date),
        "validation_last_decision_date": _date_or_none(spec.validation_last_decision_date),
        "test_last_decision_date": _date_or_none(spec.test_last_decision_date),
    }


def _date_or_none(date: pd.Timestamp | None) -> str | None:
    if date is None:
        return None
    return date.strftime("%Y-%m-%d")
