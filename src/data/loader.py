from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd

from src.config import PROJECT_ROOT, assert_path_allowed


REQUIRED_WIDE_PATH_KEYS: dict[str, str] = {
    "open": "wide_open_path",
    "high": "wide_high_path",
    "low": "wide_low_path",
    "close": "wide_close_path",
    "pre_close": "wide_pre_close_path",
    "pct_chg": "wide_pct_chg_path",
    "log_return": "wide_log_return_path",
    "amount": "wide_amount_path",
    "vol": "wide_vol_path",
    "turnover_rate": "wide_turnover_rate_path",
}
OPTIONAL_WIDE_PATH_KEYS: dict[str, str] = {
    "adj_nav": "wide_adj_nav_path",
}
ASSET_UNIVERSE_COLUMNS = {
    "ts_code",
    "symbol",
    "name",
    "type",
    "pool",
    "status",
    "rows",
    "first_date",
    "last_date",
    "median_amount_last_252",
    "raw_path",
}
PANEL_KEY_COLUMNS = {"trade_date", "ts_code"}
METRICS_FEATURE_KEY_COLUMNS = {"date", "ts_code"}
QUOTE_WIDE_FIELDS = ("open", "high", "low", "close", "pre_close")
RETURN_WIDE_FIELDS = ("log_return", "pct_chg")
AVAILABILITY_REASON_VALUES = {
    "listed",
    "not_listed_yet",
    "delisted",
    "suspended",
    "missing_quote",
    "missing_return",
}


class DataContractError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        paths: list[Path] | None = None,
    ) -> None:
        self.code = code
        self.paths = paths or []
        detail = message or code
        if self.paths:
            path_refs = [{"kind": "path", "ref": _path_ref(path)} for path in self.paths]
            detail = f"{code}: {json.dumps(path_refs, ensure_ascii=False)}"
        super().__init__(detail)


@dataclass(frozen=True)
class MarketDatasetBundle:
    asset_universe: pd.DataFrame
    panel: pd.DataFrame
    wide: dict[str, pd.DataFrame]
    metrics_features: pd.DataFrame | None
    feature_cols: list[str]
    auxiliary_target_cols: list[str]
    availability_mask: pd.DataFrame
    availability_reason: pd.DataFrame | None
    data_manifest: dict[str, Any]


@dataclass(frozen=True)
class RequiredDataPaths:
    asset_universe: Path
    panel: Path
    wide: dict[str, Path]
    all_metrics_features: Path | None
    download_manifest: Path
    metrics_manifest: Path

    def iter_paths(self) -> Iterator[Path]:
        yield self.asset_universe
        yield self.panel
        yield from self.wide.values()
        if self.all_metrics_features is not None:
            yield self.all_metrics_features
        yield self.download_manifest
        yield self.metrics_manifest


def _path_ref(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _metrics_factory_enabled(config: Mapping[str, Any]) -> bool:
    metrics_config = config["data"].get("metrics_factory", {})
    return bool(metrics_config.get("enabled", True))


def resolve_required_paths(config: Mapping[str, Any]) -> RequiredDataPaths:
    data_config = config["data"]
    whitelist = config["security"]["path_whitelist"]
    wide = {
        name: assert_path_allowed(data_config[key], whitelist, f"data.{key}")
        for name, key in REQUIRED_WIDE_PATH_KEYS.items()
    }
    use_optional_valuation = _valuation_split_requested(config)
    for name, key in OPTIONAL_WIDE_PATH_KEYS.items():
        value = data_config.get(key)
        if value is not None and use_optional_valuation:
            wide[name] = assert_path_allowed(value, whitelist, f"data.{key}")
    metrics_path = None
    if _metrics_factory_enabled(config):
        metrics_path = assert_path_allowed(
            data_config["all_metrics_features_path"],
            whitelist,
            "data.all_metrics_features_path",
        )
    return RequiredDataPaths(
        asset_universe=assert_path_allowed(
            data_config["asset_universe_path"],
            whitelist,
            "data.asset_universe_path",
        ),
        panel=assert_path_allowed(data_config["panel_path"], whitelist, "data.panel_path"),
        wide=wide,
        all_metrics_features=metrics_path,
        download_manifest=assert_path_allowed(
            data_config["download_manifest_path"],
            whitelist,
            "data.download_manifest_path",
        ),
        metrics_manifest=assert_path_allowed(
            data_config["metrics_manifest_path"],
            whitelist,
            "data.metrics_manifest_path",
        ),
    )


def _valuation_split_requested(config: Mapping[str, Any]) -> bool:
    governance = config.get("data_governance", {}) if isinstance(config.get("data_governance"), Mapping) else {}
    source = str(governance.get("valuation_source") or governance.get("return_source") or "").lower()
    return bool(governance.get("valuation_execution_split", False) or source == "adj_nav")


def assert_required_paths_exist(required_paths: RequiredDataPaths) -> None:
    missing = [path for path in required_paths.iter_paths() if not path.exists()]
    if missing:
        raise DataContractError("ERR_DATA_MISSING_FILE", paths=missing)


def load_asset_universe(
    asset_universe_path: Path,
    *,
    pools: Any | None = None,
    assets: Any | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if not asset_universe_path.exists():
        raise DataContractError("ERR_DATA_MISSING_FILE", paths=[asset_universe_path])

    asset_universe = pd.read_csv(asset_universe_path)
    missing_columns = sorted(ASSET_UNIVERSE_COLUMNS - set(asset_universe.columns))
    if missing_columns:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            f"ERR_DATA_SCHEMA_MISMATCH: asset_universe missing columns {missing_columns}",
        )

    ok_universe = asset_universe.loc[asset_universe["status"] == "ok"].copy()
    pool_filter = _string_filter_values(pools)
    if pool_filter:
        ok_universe = ok_universe.loc[ok_universe["pool"].astype(str).isin(pool_filter)].copy()
    asset_filter = _string_filter_values(assets)
    if asset_filter:
        ok_universe = ok_universe.loc[ok_universe["ts_code"].astype(str).isin(asset_filter)].copy()
    if ok_universe.empty:
        raise DataContractError("ERR_DATA_NO_AVAILABLE_ASSET", "ERR_DATA_NO_AVAILABLE_ASSET: asset_universe.status")

    canonical_asset_order = ok_universe["ts_code"].astype(str).tolist()
    ok_universe.attrs["canonical_asset_order"] = canonical_asset_order
    return ok_universe, canonical_asset_order


def _string_filter_values(value: Any | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    try:
        return [str(item) for item in value if str(item)]
    except TypeError:
        return [str(value)]


def load_wide_table(field: str, path: Path, canonical_asset_order: list[str]) -> pd.DataFrame:
    table = pd.read_parquet(path).copy()
    try:
        table.index = pd.DatetimeIndex(pd.to_datetime(table.index))
    except (TypeError, ValueError) as exc:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            f"ERR_DATA_SCHEMA_MISMATCH: wide_{field} index cannot convert to datetime",
        ) from exc

    table = table.sort_index()
    if table.index.hasnans or not table.index.is_unique or not table.index.is_monotonic_increasing:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            f"ERR_DATA_SCHEMA_MISMATCH: wide_{field} date index",
        )
    if table.columns.has_duplicates:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            f"ERR_DATA_SCHEMA_MISMATCH: wide_{field} duplicate columns",
        )

    missing_assets = [asset for asset in canonical_asset_order if asset not in table.columns]
    if missing_assets:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            f"ERR_DATA_SCHEMA_MISMATCH: wide_{field} missing assets {missing_assets}",
        )

    aligned = table.reindex(columns=canonical_asset_order)
    if list(aligned.columns) != canonical_asset_order:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            f"ERR_DATA_SCHEMA_MISMATCH: wide_{field} column order",
        )
    return aligned


def load_wide_tables(required_paths: RequiredDataPaths, canonical_asset_order: list[str]) -> dict[str, pd.DataFrame]:
    return {
        field: load_wide_table(field, path, canonical_asset_order)
        for field, path in required_paths.wide.items()
    }


def load_panel(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path).copy()
    missing_columns = sorted(PANEL_KEY_COLUMNS - set(panel.columns))
    if missing_columns:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            f"ERR_DATA_SCHEMA_MISMATCH: panel missing columns {missing_columns}",
        )

    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    if panel.duplicated(["trade_date", "ts_code"]).any():
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            "ERR_DATA_SCHEMA_MISMATCH: panel duplicate (trade_date, ts_code)",
        )
    return panel


def load_metrics_features(required_paths: RequiredDataPaths) -> pd.DataFrame | None:
    if required_paths.all_metrics_features is None:
        return None

    metrics_features = pd.read_parquet(required_paths.all_metrics_features).copy()
    missing_columns = sorted(METRICS_FEATURE_KEY_COLUMNS - set(metrics_features.columns))
    if missing_columns:
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            f"ERR_DATA_SCHEMA_MISMATCH: metrics_features missing columns {missing_columns}",
        )

    metrics_features["date"] = pd.to_datetime(metrics_features["date"])
    if metrics_features.duplicated(["date", "ts_code"]).any():
        raise DataContractError(
            "ERR_DATA_SCHEMA_MISMATCH",
            "ERR_DATA_SCHEMA_MISMATCH: metrics_features duplicate (date, ts_code)",
        )
    return metrics_features


def build_availability(
    asset_universe: pd.DataFrame,
    wide: dict[str, pd.DataFrame],
    canonical_asset_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_index = wide["close"].index
    availability_mask = wide["close"].notna() & wide["log_return"].notna()
    availability_mask = availability_mask.reindex(index=date_index, columns=canonical_asset_order)

    quote_missing = pd.DataFrame(False, index=date_index, columns=canonical_asset_order)
    for field in QUOTE_WIDE_FIELDS:
        quote_missing |= wide[field].reindex(index=date_index, columns=canonical_asset_order).isna()

    return_missing = pd.DataFrame(False, index=date_index, columns=canonical_asset_order)
    for field in RETURN_WIDE_FIELDS:
        return_missing |= wide[field].reindex(index=date_index, columns=canonical_asset_order).isna()

    zero_liquidity = (
        wide["vol"].reindex(index=date_index, columns=canonical_asset_order).le(0).fillna(False)
        | wide["amount"].reindex(index=date_index, columns=canonical_asset_order).le(0).fillna(False)
    )
    quote_exists = ~quote_missing
    return_exists = ~return_missing

    availability_reason = pd.DataFrame("missing_return", index=date_index, columns=canonical_asset_order)
    availability_reason = availability_reason.mask(quote_missing, "missing_quote")
    availability_reason = availability_reason.mask(
        availability_mask & quote_exists & return_exists,
        "listed",
    )
    availability_reason = availability_reason.mask(
        quote_exists & return_exists & zero_liquidity,
        "suspended",
    )

    universe_by_asset = asset_universe.set_index("ts_code", drop=False)
    for asset in canonical_asset_order:
        first_date = pd.to_datetime(universe_by_asset.loc[asset, "first_date"], errors="coerce")
        last_date = pd.to_datetime(universe_by_asset.loc[asset, "last_date"], errors="coerce")
        if pd.notna(first_date):
            availability_reason.loc[date_index < first_date, asset] = "not_listed_yet"
        if pd.notna(last_date):
            availability_reason.loc[date_index > last_date, asset] = "delisted"

    availability_mask = availability_mask & availability_reason.eq("listed")
    return availability_mask.astype(bool), availability_reason


def is_amount_proxy(wide: dict[str, pd.DataFrame], canonical_asset_order: list[str]) -> bool:
    amount = wide["amount"].reindex(columns=canonical_asset_order)
    proxy_amount = (
        wide["close"].reindex(columns=canonical_asset_order)
        * wide["vol"].reindex(columns=canonical_asset_order)
        * 100.0
    )
    comparable = amount.notna() & proxy_amount.notna()
    if not comparable.any().any():
        return False

    tolerance = 1.0e-8 + 1.0e-10 * proxy_amount.abs()
    within_tolerance = (amount - proxy_amount).abs() <= tolerance
    return bool(within_tolerance.where(comparable).stack().all())


def is_turnover_rate_all_missing(wide: dict[str, pd.DataFrame], canonical_asset_order: list[str]) -> bool:
    turnover_rate = wide["turnover_rate"].reindex(columns=canonical_asset_order)
    return bool(turnover_rate.isna().all().all())


def build_liquidity_manifest(
    wide: dict[str, pd.DataFrame],
    canonical_asset_order: list[str],
    config: Mapping[str, Any],
) -> dict[str, bool]:
    turnover_rate_all_missing = is_turnover_rate_all_missing(wide, canonical_asset_order)
    if turnover_rate_all_missing and bool(config["data_governance"].get("turnover_rate_required", False)):
        raise DataContractError(
            "ERR_DATA_TURNOVER_RATE_REQUIRED",
            "ERR_DATA_TURNOVER_RATE_REQUIRED: wide_turnover_rate",
        )

    return {
        "amount_is_proxy": is_amount_proxy(wide, canonical_asset_order),
        "turnover_rate_all_missing": turnover_rate_all_missing,
    }


def apply_date_policy(
    panel: pd.DataFrame,
    wide: dict[str, pd.DataFrame],
    metrics_features: pd.DataFrame | None,
    availability_mask: pd.DataFrame,
    availability_reason: pd.DataFrame | None,
    data_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    date_index = pd.DatetimeIndex(wide["close"].index)
    start_date = _optional_timestamp(data_config.get("start_date"), "data.start_date")
    end_date = _optional_timestamp(data_config.get("end_date"), "data.end_date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise DataContractError("ERR_DATA_NO_COMMON_HISTORY", "ERR_DATA_NO_COMMON_HISTORY: data.start_date > data.end_date")

    keep = pd.Series(True, index=date_index)
    if start_date is not None:
        keep &= date_index >= start_date
    if end_date is not None:
        keep &= date_index <= end_date

    strict_common = bool(data_config.get("strict_common_history_mode", False))
    if strict_common:
        common_available = availability_mask.reindex(index=date_index).all(axis=1)
        keep &= common_available

    selected_dates = pd.DatetimeIndex(date_index[keep.to_numpy(dtype=bool)])
    if selected_dates.empty:
        raise DataContractError("ERR_DATA_NO_COMMON_HISTORY", "ERR_DATA_NO_COMMON_HISTORY: no common available dates")

    filtered_wide = {name: frame.reindex(index=selected_dates) for name, frame in wide.items()}
    filtered_availability = availability_mask.reindex(index=selected_dates)
    filtered_reason = None if availability_reason is None else availability_reason.reindex(index=selected_dates)

    filtered_panel = panel.copy()
    if "trade_date" in filtered_panel.columns:
        filtered_panel = filtered_panel.loc[filtered_panel["trade_date"].isin(selected_dates)].copy()

    filtered_metrics = metrics_features
    if filtered_metrics is not None and "date" in filtered_metrics.columns:
        filtered_metrics = filtered_metrics.loc[filtered_metrics["date"].isin(selected_dates)].copy()

    manifest = {
        "date_start": selected_dates[0].strftime("%Y-%m-%d"),
        "date_end": selected_dates[-1].strftime("%Y-%m-%d"),
        "date_count": int(len(selected_dates)),
        "configured_start_date": _date_or_none(start_date),
        "configured_end_date": _date_or_none(end_date),
        "strict_common_history_mode": strict_common,
        "all_assets_available_each_date": bool(filtered_availability.all(axis=1).all()),
    }
    return filtered_panel, filtered_wide, filtered_metrics, filtered_availability.astype(bool), filtered_reason, manifest


def load_market_dataset(config: Mapping[str, Any]) -> MarketDatasetBundle:
    required_paths = resolve_required_paths(config)
    assert_required_paths_exist(required_paths)
    data_config = config.get("data", {}) if isinstance(config.get("data"), Mapping) else {}
    asset_universe, canonical_asset_order = load_asset_universe(
        required_paths.asset_universe,
        pools=data_config.get("asset_universe_pools"),
        assets=data_config.get("asset_universe_assets"),
    )
    wide = load_wide_tables(required_paths, canonical_asset_order)
    panel = load_panel(required_paths.panel)
    metrics_features = load_metrics_features(required_paths)
    availability_mask, availability_reason = build_availability(asset_universe, wide, canonical_asset_order)
    panel, wide, metrics_features, availability_mask, availability_reason, date_manifest = apply_date_policy(
        panel,
        wide,
        metrics_features,
        availability_mask,
        availability_reason,
        data_config,
    )
    liquidity_manifest = build_liquidity_manifest(wide, canonical_asset_order, config)
    return MarketDatasetBundle(
        asset_universe=asset_universe,
        panel=panel,
        wide=wide,
        metrics_features=metrics_features,
        feature_cols=[],
        auxiliary_target_cols=[],
        availability_mask=availability_mask,
        availability_reason=availability_reason,
        data_manifest={
            "canonical_asset_order": canonical_asset_order,
            "asset_count": len(canonical_asset_order),
            "metrics_factory_enabled": metrics_features is not None,
            "availability_reason_inference": "field_based",
            **date_manifest,
            **liquidity_manifest,
        },
    )


def _optional_timestamp(value: Any, key_path: str) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise DataContractError("ERR_DATA_NO_COMMON_HISTORY", f"ERR_DATA_NO_COMMON_HISTORY: {key_path}")
    return pd.Timestamp(timestamp)


def _date_or_none(value: pd.Timestamp | None) -> str | None:
    return None if value is None else value.strftime("%Y-%m-%d")
