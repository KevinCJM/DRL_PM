from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG, PROJECT_ROOT
from src.data.loader import MarketDatasetBundle, REQUIRED_WIDE_PATH_KEYS
from src.data.splits import SplitSpec
from src.envs.state import PortfolioState


SAMPLE_ASSET_ORDER = ["510300.SH", "159915.SZ"]


@pytest.fixture
def sample_asset_order() -> list[str]:
    return list(SAMPLE_ASSET_ORDER)


@pytest.fixture
def sample_trade_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=6)


@pytest.fixture
def sample_asset_universe(sample_asset_order: list[str], sample_trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    return _sample_asset_universe(sample_asset_order, sample_trade_dates)


@pytest.fixture
def sample_wide_tables(sample_asset_order: list[str], sample_trade_dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    return _sample_wide_tables(sample_asset_order, sample_trade_dates)


@pytest.fixture
def sample_panel(sample_wide_tables: dict[str, pd.DataFrame], sample_asset_order: list[str]) -> pd.DataFrame:
    return _panel_from_wide(sample_wide_tables, sample_asset_order)


@pytest.fixture
def sample_metrics_features(sample_asset_order: list[str], sample_trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    return _sample_metrics_features(sample_asset_order, sample_trade_dates)


@pytest.fixture
def sample_market_dataset_bundle(
    sample_asset_universe: pd.DataFrame,
    sample_panel: pd.DataFrame,
    sample_wide_tables: dict[str, pd.DataFrame],
    sample_metrics_features: pd.DataFrame,
    sample_asset_order: list[str],
) -> MarketDatasetBundle:
    return _market_dataset_bundle(
        asset_universe=sample_asset_universe,
        panel=sample_panel,
        wide=sample_wide_tables,
        metrics_features=sample_metrics_features,
        asset_order=sample_asset_order,
    )


@pytest.fixture
def sample_split_spec(sample_trade_dates: pd.DatetimeIndex) -> SplitSpec:
    return SplitSpec(
        train_dates=sample_trade_dates[:3],
        validation_dates=sample_trade_dates[3:4],
        test_dates=sample_trade_dates[4:],
        fold_id="fixed",
        train_last_decision_date=sample_trade_dates[2],
        validation_last_decision_date=sample_trade_dates[3],
        test_last_decision_date=sample_trade_dates[4],
    )


@pytest.fixture
def sample_portfolio_state(sample_trade_dates: pd.DatetimeIndex, sample_asset_order: list[str]) -> PortfolioState:
    n_assets = len(sample_asset_order)
    weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    return PortfolioState(
        date=sample_trade_dates[0],
        nav=1.0,
        portfolio_value=100000000.0,
        current_weights=weights,
        drifted_weights=weights.copy(),
        previous_executed_weights=weights.copy(),
        running_max_nav=1.0,
        current_drawdown_abs=0.0,
        rolling_returns=[],
        step_index=0,
        last_buy_date_per_asset=np.array([sample_trade_dates[0]] * n_assets, dtype=object),
        sellable_mask=np.ones(n_assets, dtype=bool),
        frozen_weight=np.zeros(n_assets, dtype=float),
    )


@pytest.fixture
def sample_config(tmp_path: Path) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    config["data"]["asset_universe_path"] = str(data_dir / "asset_universe.csv")
    config["data"]["panel_path"] = str(data_dir / "etf_lof_daily_panel.parquet")
    config["data"]["all_metrics_features_path"] = str(data_dir / "all_metrics_features.parquet")
    config["data"]["download_manifest_path"] = str(data_dir / "download_manifest.json")
    config["data"]["metrics_manifest_path"] = str(data_dir / "metrics_manifest.json")
    config["data"]["metrics_factory"]["all_metrics_features_path"] = config["data"]["all_metrics_features_path"]
    config["output"]["root"] = str(results_dir)
    config["output"]["run_name"] = "sample_run"
    config["registry"]["path"] = str(results_dir / "run_registry.sqlite")
    for field, key in REQUIRED_WIDE_PATH_KEYS.items():
        config["data"][key] = str(data_dir / f"wide_{field}.parquet")
    return config


@pytest.fixture
def market_dataset_bundle_factory(
    sample_asset_order: list[str],
    sample_trade_dates: pd.DatetimeIndex,
) -> Any:
    def factory(
        *,
        turnover_rate_all_missing: bool = False,
        no_available_asset: bool = False,
        execution_return_missing: bool = False,
        metrics_features: bool = True,
    ) -> MarketDatasetBundle:
        asset_universe = _sample_asset_universe(sample_asset_order, sample_trade_dates)
        if no_available_asset:
            asset_universe["status"] = "missing"
        wide = _sample_wide_tables(sample_asset_order, sample_trade_dates, turnover_rate_all_missing=turnover_rate_all_missing)
        if execution_return_missing:
            wide["open"].loc[sample_trade_dates[1], sample_asset_order[0]] = np.nan
        panel = _panel_from_wide(wide, sample_asset_order)
        metrics = _sample_metrics_features(sample_asset_order, sample_trade_dates) if metrics_features else None
        return _market_dataset_bundle(
            asset_universe=asset_universe,
            panel=panel,
            wide=wide,
            metrics_features=metrics,
            asset_order=sample_asset_order,
            no_available_asset=no_available_asset,
        )

    return factory


@pytest.fixture
def market_dataset_files_factory(
    tmp_path: Path,
    sample_config: dict[str, Any],
    sample_asset_order: list[str],
    sample_trade_dates: pd.DatetimeIndex,
) -> Any:
    def factory(
        *,
        turnover_rate_all_missing: bool = False,
        missing_file: str | None = None,
        no_available_asset: bool = False,
        metrics_features: bool = True,
    ) -> dict[str, Any]:
        config = deepcopy(sample_config)
        asset_universe = _sample_asset_universe(sample_asset_order, sample_trade_dates)
        if no_available_asset:
            asset_universe["status"] = "missing"
        wide = _sample_wide_tables(sample_asset_order, sample_trade_dates, turnover_rate_all_missing=turnover_rate_all_missing)
        panel = _panel_from_wide(wide, sample_asset_order)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        asset_universe.to_csv(config["data"]["asset_universe_path"], index=False)
        panel.to_parquet(config["data"]["panel_path"])
        for field, key in REQUIRED_WIDE_PATH_KEYS.items():
            if missing_file == f"wide_{field}":
                continue
            wide[field].to_parquet(config["data"][key])
        if metrics_features:
            _sample_metrics_features(sample_asset_order, sample_trade_dates).to_parquet(config["data"]["all_metrics_features_path"])
        else:
            config["data"]["metrics_factory"]["enabled"] = False
        if missing_file != "download_manifest":
            Path(config["data"]["download_manifest_path"]).write_text("{}", encoding="utf-8")
        if missing_file != "metrics_manifest":
            Path(config["data"]["metrics_manifest_path"]).write_text("{}", encoding="utf-8")
        if missing_file == "panel":
            Path(config["data"]["panel_path"]).unlink(missing_ok=True)
        if missing_file == "asset_universe":
            Path(config["data"]["asset_universe_path"]).unlink(missing_ok=True)
        if missing_file == "metrics_features":
            Path(config["data"]["all_metrics_features_path"]).unlink(missing_ok=True)

        return {
            "config": config,
            "asset_order": list(sample_asset_order),
            "asset_universe": asset_universe,
            "panel": panel,
            "wide": _copy_wide(wide),
            "metrics_features": _sample_metrics_features(sample_asset_order, sample_trade_dates) if metrics_features else None,
        }

    return factory


def _sample_asset_universe(asset_order: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": asset,
                "symbol": asset.split(".")[0],
                "name": f"asset_{index}",
                "type": "ETF",
                "pool": "sample",
                "status": "ok",
                "rows": len(dates),
                "first_date": dates[0].strftime("%Y-%m-%d"),
                "last_date": dates[-1].strftime("%Y-%m-%d"),
                "median_amount_last_252": 1000000.0 + index,
                "raw_path": f"data/raw/{asset}_daily.parquet",
            }
            for index, asset in enumerate(asset_order)
        ]
    )


def _sample_wide_tables(
    asset_order: list[str],
    dates: pd.DatetimeIndex,
    *,
    turnover_rate_all_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    base = np.arange(len(dates), dtype=float).reshape(-1, 1)
    offsets = np.arange(len(asset_order), dtype=float).reshape(1, -1)
    columns = list(asset_order)
    close = pd.DataFrame(10.0 + base + offsets * 5.0, index=dates, columns=columns)
    pre_close = close.shift(1)
    pre_close.iloc[0] = close.iloc[0] * 0.99
    open_ = pre_close * 1.002
    high = pd.concat([open_.stack(), close.stack()], axis=1).max(axis=1).unstack() + 0.1
    low = pd.concat([open_.stack(), close.stack()], axis=1).min(axis=1).unstack() - 0.1
    vol = pd.DataFrame(1000.0 + base * 10.0 + offsets * 100.0, index=dates, columns=columns)
    amount = close * vol * 100.0
    turnover_rate = pd.DataFrame(
        np.nan if turnover_rate_all_missing else 0.01,
        index=dates,
        columns=columns,
    )
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "pre_close": pre_close,
        "pct_chg": close / pre_close - 1.0,
        "log_return": np.log(close / pre_close),
        "amount": amount,
        "vol": vol,
        "turnover_rate": turnover_rate,
    }


def _panel_from_wide(wide: dict[str, pd.DataFrame], asset_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date in wide["close"].index:
        for asset in asset_order:
            row = {"trade_date": trade_date, "ts_code": asset}
            for field in REQUIRED_WIDE_PATH_KEYS:
                row[field] = wide[field].loc[trade_date, asset]
            rows.append(row)
    return pd.DataFrame(rows)


def _sample_metrics_features(asset_order: list[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date_index, date in enumerate(dates):
        for asset_index, asset in enumerate(asset_order):
            rows.append(
                {
                    "date": date,
                    "ts_code": asset,
                    "momentum_3d": float(date_index + asset_index),
                    "volatility_3d": float(0.01 + asset_index * 0.001),
                    "auxiliary_return_1d": float(0.001 * (date_index + 1)),
                }
            )
    return pd.DataFrame(rows)


def _market_dataset_bundle(
    *,
    asset_universe: pd.DataFrame,
    panel: pd.DataFrame,
    wide: dict[str, pd.DataFrame],
    metrics_features: pd.DataFrame | None,
    asset_order: list[str],
    no_available_asset: bool = False,
) -> MarketDatasetBundle:
    availability_mask = pd.DataFrame(not no_available_asset, index=wide["close"].index, columns=asset_order)
    availability_reason = pd.DataFrame(
        "missing_quote" if no_available_asset else "listed",
        index=wide["close"].index,
        columns=asset_order,
    )
    return MarketDatasetBundle(
        asset_universe=asset_universe.copy(),
        panel=panel.copy(),
        wide=_copy_wide(wide),
        metrics_features=None if metrics_features is None else metrics_features.copy(),
        feature_cols=["momentum_3d", "volatility_3d"] if metrics_features is not None else [],
        auxiliary_target_cols=["auxiliary_return_1d"] if metrics_features is not None else [],
        availability_mask=availability_mask,
        availability_reason=availability_reason,
        data_manifest={
            "canonical_asset_order": list(asset_order),
            "asset_count": len(asset_order),
            "metrics_factory_enabled": metrics_features is not None,
            "availability_reason_inference": "fixture",
            "amount_is_proxy": True,
            "turnover_rate_all_missing": bool(wide["turnover_rate"].isna().all().all()),
        },
    )


def _copy_wide(wide: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {field: frame.copy() for field, frame in wide.items()}
