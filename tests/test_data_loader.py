from copy import deepcopy
from dataclasses import fields
import inspect
import json

import numpy as np
import pandas as pd
import pytest

import src.data.loader as loader_module
import src.experiments.run_experiment as run_experiment_module
from src.config import DEFAULT_CONFIG, PROJECT_ROOT
from src.data.akshare_etf_fetcher import main as akshare_fetcher_main
from src.data.loader import (
    DataContractError,
    MarketDatasetBundle,
    assert_required_paths_exist,
    load_market_dataset,
    resolve_required_paths,
)
from src.data.preprocess import (
    build_return_fields,
    detect_amount_proxy,
    panel_to_wide_tables,
    validate_panel_schema,
)


def test_market_dataset_bundle_contract(tmp_path):
    bundle_fields = {field.name for field in fields(MarketDatasetBundle)}
    assert bundle_fields == {
        "asset_universe",
        "panel",
        "wide",
        "metrics_features",
        "feature_cols",
        "auxiliary_target_cols",
        "availability_mask",
        "availability_reason",
        "data_manifest",
    }

    frame = pd.DataFrame({"ts_code": ["510300.SH"]})
    mask = pd.DataFrame({"510300.SH": [True]})
    bundle = MarketDatasetBundle(
        asset_universe=frame,
        panel=frame,
        wide={"close": pd.DataFrame({"510300.SH": [1.0]})},
        metrics_features=None,
        feature_cols=[],
        auxiliary_target_cols=[],
        availability_mask=mask,
        availability_reason=None,
        data_manifest={"metrics_factory_enabled": False},
    )
    assert bundle.asset_universe is frame
    assert bundle.metrics_features is None
    assert bundle.data_manifest["metrics_factory_enabled"] is False

    config = deepcopy(DEFAULT_CONFIG)
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    config["data"]["asset_universe_path"] = str(tmp_path / "asset_universe.csv")
    config["data"]["panel_path"] = str(tmp_path / "panel.parquet")
    config["data"]["download_manifest_path"] = str(tmp_path / "download_manifest.json")
    config["data"]["metrics_manifest_path"] = str(tmp_path / "metrics_manifest.json")
    config["data"]["metrics_factory"]["enabled"] = False
    for key in list(config["data"]):
        if key.startswith("wide_") and key.endswith("_path"):
            config["data"][key] = str(tmp_path / f"{key}.parquet")

    required_paths = resolve_required_paths(config)
    assert required_paths.asset_universe == tmp_path / "asset_universe.csv"
    assert required_paths.panel == tmp_path / "panel.parquet"
    assert required_paths.all_metrics_features is None
    assert set(required_paths.wide) == {
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "log_return",
        "amount",
        "vol",
        "turnover_rate",
    }

    with pytest.raises(DataContractError) as error:
        assert_required_paths_exist(required_paths)
    assert error.value.code == "ERR_DATA_MISSING_FILE"
    assert '"kind": "path"' in str(error.value)

    config["data"]["metrics_factory"]["enabled"] = True
    config["data"]["all_metrics_features_path"] = str(tmp_path / "custom_metrics.parquet")
    config["data"]["metrics_factory"]["all_metrics_features_path"] = str(tmp_path / "nested_legacy_metrics.parquet")

    required_paths = resolve_required_paths(config)

    assert required_paths.all_metrics_features == tmp_path / "custom_metrics.parquet"


def test_preprocess_panel_to_wide_and_proxy_flags():
    panel = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
            "ts_code": ["510300.SH", "510300.SH", "159915.SZ", "159915.SZ"],
            "open": [1.0, 1.1, 2.0, 2.2],
            "high": [1.2, 1.3, 2.3, 2.4],
            "low": [0.9, 1.0, 1.9, 2.1],
            "close": [1.0, 1.2, 2.0, 2.4],
            "vol": [10.0, 20.0, 30.0, 40.0],
        }
    )

    validate_panel_schema(panel)
    with_returns = build_return_fields(panel)
    with_amount, amount_is_proxy = detect_amount_proxy(with_returns)
    wide = panel_to_wide_tables(panel, asset_order=["510300.SH", "159915.SZ"])

    assert amount_is_proxy is True
    amount_row = with_amount.loc[
        (with_amount["trade_date"] == pd.Timestamp("2024-01-02")) & (with_amount["ts_code"] == "510300.SH")
    ]
    assert amount_row["amount"].iloc[0] == 1000.0
    assert set(wide) == {
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "log_return",
        "amount",
        "vol",
        "turnover_rate",
    }
    assert list(wide["close"].columns) == ["510300.SH", "159915.SZ"]
    assert wide["close"].index.is_monotonic_increasing
    assert wide["pre_close"].loc[pd.Timestamp("2024-01-03"), "510300.SH"] == 1.0
    assert wide["pct_chg"].loc[pd.Timestamp("2024-01-03"), "510300.SH"] == pytest.approx(0.2)
    assert wide["log_return"].loc[pd.Timestamp("2024-01-03"), "159915.SZ"] == pytest.approx(np.log(1.2))
    assert wide["amount"].loc[pd.Timestamp("2024-01-03"), "159915.SZ"] == 9600.0
    assert wide["turnover_rate"].isna().all().all()

    bad_panel = panel.drop(columns=["high"])
    with pytest.raises(DataContractError) as error:
        validate_panel_schema(bad_panel)
    assert error.value.code == "ERR_DATA_SCHEMA_MISMATCH"


def test_akshare_fetcher_is_explicit_only(tmp_path):
    panel = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "ts_code": ["510300.SH", "510300.SH"],
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.0, 1.2],
            "vol": [10.0, 20.0],
        }
    )
    panel_path = tmp_path / "input_panel.parquet"
    processed_dir = tmp_path / "processed"
    manifest_path = tmp_path / "reports" / "data_manifest.json"
    panel.to_parquet(panel_path)

    manifest = akshare_fetcher_main(
        [
            "--panel",
            str(panel_path),
            "--processed-dir",
            str(processed_dir),
            "--manifest-path",
            str(manifest_path),
        ]
    )

    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["amount_is_proxy"] is True
    assert written_manifest["proxy_flags"]["amount"] is True
    assert written_manifest["source_range"] == {"start": "2024-01-02", "end": "2024-01-03"}
    assert written_manifest["panel_rows"] == len(panel)
    assert (processed_dir / "etf_lof_daily_panel.parquet").exists()
    assert (processed_dir / "wide_close.parquet").exists()
    assert (processed_dir / "wide_log_return.parquet").exists()
    assert list(processed_dir.glob(".*.tmp.parquet")) == []
    assert list(manifest_path.parent.glob(".*.tmp")) == []

    assert "akshare_etf_fetcher" not in inspect.getsource(loader_module)
    assert "akshare_etf_fetcher" not in inspect.getsource(run_experiment_module)


def _write_required_market_dataset_files(config, tmp_path, asset_order):
    config["data"]["panel_path"] = str(tmp_path / "panel.parquet")
    pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02") for _ in asset_order],
            "ts_code": asset_order,
        }
    ).to_parquet(config["data"]["panel_path"])

    wide_frame = pd.DataFrame(
        {asset: [1.0] for asset in asset_order},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]),
    )
    for key in list(config["data"]):
        if key.startswith("wide_") and key.endswith("_path"):
            config["data"][key] = str(tmp_path / f"{key}.parquet")
            wide_frame.to_parquet(config["data"][key])

    config["data"]["all_metrics_features_path"] = str(tmp_path / "all_metrics_features.parquet")
    pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "ts_code": [asset_order[0]]}).to_parquet(
        config["data"]["all_metrics_features_path"]
    )
    config["data"]["download_manifest_path"] = str(tmp_path / "download_manifest.json")
    config["data"]["metrics_manifest_path"] = str(tmp_path / "metrics_manifest.json")
    (tmp_path / "download_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metrics_manifest.json").write_text("{}", encoding="utf-8")


def test_asset_universe_status_ok_order(tmp_path):
    asset_universe_path = tmp_path / "asset_universe.csv"
    pd.DataFrame(
        [
            {
                "ts_code": "510050.SH",
                "symbol": "510050",
                "name": "上证50ETF",
                "type": "ETF",
                "pool": "equity",
                "status": "missing",
                "rows": 0,
                "first_date": "",
                "last_date": "",
                "median_amount_last_252": np.nan,
                "raw_path": "data/raw/510050.SH_daily.parquet",
            },
            {
                "ts_code": "159915.SZ",
                "symbol": "159915",
                "name": "创业板ETF",
                "type": "ETF",
                "pool": "equity",
                "status": "ok",
                "rows": 2,
                "first_date": "2024-01-02",
                "last_date": "2024-01-03",
                "median_amount_last_252": 1000.0,
                "raw_path": "data/raw/159915.SZ_daily.parquet",
            },
            {
                "ts_code": "511010.SH",
                "symbol": "511010",
                "name": "国债ETF",
                "type": "ETF",
                "pool": "bond",
                "status": "ok",
                "rows": 2,
                "first_date": "2024-01-02",
                "last_date": "2024-01-03",
                "median_amount_last_252": 2000.0,
                "raw_path": "data/raw/511010.SH_daily.parquet",
            },
        ]
    ).to_csv(asset_universe_path, index=False)
    config = deepcopy(DEFAULT_CONFIG)
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    config["data"]["asset_universe_path"] = str(asset_universe_path)
    _write_required_market_dataset_files(config, tmp_path, ["159915.SZ", "511010.SH"])

    bundle = load_market_dataset(config)

    assert bundle.asset_universe["ts_code"].tolist() == ["159915.SZ", "511010.SH"]
    assert bundle.asset_universe["status"].tolist() == ["ok", "ok"]
    assert bundle.asset_universe.attrs["canonical_asset_order"] == ["159915.SZ", "511010.SH"]
    assert bundle.data_manifest["canonical_asset_order"] == ["159915.SZ", "511010.SH"]
    assert list(bundle.availability_mask.columns) == ["159915.SZ", "511010.SH"]
    assert "fund_list" not in inspect.getsource(loader_module)

    pool_filtered_config = deepcopy(config)
    pool_filtered_config["data"]["asset_universe_pools"] = ["bond"]
    pool_filtered_bundle = load_market_dataset(pool_filtered_config)
    assert pool_filtered_bundle.asset_universe["ts_code"].tolist() == ["511010.SH"]
    assert pool_filtered_bundle.data_manifest["canonical_asset_order"] == ["511010.SH"]
    assert list(pool_filtered_bundle.availability_mask.columns) == ["511010.SH"]

    asset_filtered_config = deepcopy(config)
    asset_filtered_config["data"]["asset_universe_assets"] = ["159915.SZ"]
    asset_filtered_bundle = load_market_dataset(asset_filtered_config)
    assert asset_filtered_bundle.asset_universe["ts_code"].tolist() == ["159915.SZ"]
    assert asset_filtered_bundle.data_manifest["canonical_asset_order"] == ["159915.SZ"]
    assert list(asset_filtered_bundle.availability_mask.columns) == ["159915.SZ"]

    missing_required_config = deepcopy(config)
    missing_required_config["data"]["panel_path"] = str(tmp_path / "missing_panel.parquet")
    with pytest.raises(DataContractError) as missing_error:
        load_market_dataset(missing_required_config)
    assert missing_error.value.code == "ERR_DATA_MISSING_FILE"
    assert "missing_panel.parquet" in str(missing_error.value)

    missing_column_path = tmp_path / "asset_universe_missing_column.csv"
    pd.read_csv(asset_universe_path).drop(columns=["status"]).to_csv(missing_column_path, index=False)
    config["data"]["asset_universe_path"] = str(missing_column_path)
    with pytest.raises(DataContractError) as schema_error:
        load_market_dataset(config)
    assert schema_error.value.code == "ERR_DATA_SCHEMA_MISMATCH"

    no_available_path = tmp_path / "asset_universe_no_available.csv"
    no_available = pd.read_csv(asset_universe_path)
    no_available["status"] = "missing"
    no_available.to_csv(no_available_path, index=False)
    config["data"]["asset_universe_path"] = str(no_available_path)
    with pytest.raises(DataContractError) as available_error:
        load_market_dataset(config)
    assert available_error.value.code == "ERR_DATA_NO_AVAILABLE_ASSET"


def test_wide_tables_align_to_asset_order(tmp_path):
    asset_order = ["159915.SZ", "511010.SH"]
    asset_universe_path = tmp_path / "asset_universe.csv"
    pd.DataFrame(
        [
            {
                "ts_code": "159915.SZ",
                "symbol": "159915",
                "name": "创业板ETF",
                "type": "ETF",
                "pool": "equity",
                "status": "ok",
                "rows": 2,
                "first_date": "2024-01-02",
                "last_date": "2024-01-03",
                "median_amount_last_252": 1000.0,
                "raw_path": "data/raw/159915.SZ_daily.parquet",
            },
            {
                "ts_code": "511010.SH",
                "symbol": "511010",
                "name": "国债ETF",
                "type": "ETF",
                "pool": "bond",
                "status": "ok",
                "rows": 2,
                "first_date": "2024-01-02",
                "last_date": "2024-01-03",
                "median_amount_last_252": 2000.0,
                "raw_path": "data/raw/511010.SH_daily.parquet",
            },
        ]
    ).to_csv(asset_universe_path, index=False)

    config = deepcopy(DEFAULT_CONFIG)
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    config["data"]["asset_universe_path"] = str(asset_universe_path)
    _write_required_market_dataset_files(config, tmp_path, asset_order)

    source_wide = pd.DataFrame(
        {
            "511010.SH": [20.0, 10.0],
            "ignored_extra_asset": [200.0, 100.0],
            "159915.SZ": [2.0, 1.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02")]),
    )
    for key in list(config["data"]):
        if key.startswith("wide_") and key.endswith("_path"):
            source_wide.to_parquet(config["data"][key])

    bundle = load_market_dataset(config)

    assert list(bundle.wide) == [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "log_return",
        "amount",
        "vol",
        "turnover_rate",
    ]
    for table in bundle.wide.values():
        assert list(table.columns) == asset_order
        assert table.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert bundle.wide["close"].loc[pd.Timestamp("2024-01-02"), "159915.SZ"] == 1.0
    assert bundle.wide["close"].loc[pd.Timestamp("2024-01-03"), "511010.SH"] == 20.0

    missing_asset_wide = source_wide.drop(columns=["511010.SH"])
    missing_asset_wide.to_parquet(config["data"]["wide_close_path"])
    with pytest.raises(DataContractError) as missing_asset_error:
        load_market_dataset(config)
    assert missing_asset_error.value.code == "ERR_DATA_SCHEMA_MISMATCH"

    duplicate_date_wide = pd.DataFrame(
        {
            "159915.SZ": [1.0, 2.0],
            "511010.SH": [10.0, 20.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")]),
    )
    duplicate_date_wide.to_parquet(config["data"]["wide_close_path"])
    with pytest.raises(DataContractError) as duplicate_date_error:
        load_market_dataset(config)
    assert duplicate_date_error.value.code == "ERR_DATA_SCHEMA_MISMATCH"


def test_panel_and_metrics_feature_key_integrity(tmp_path):
    asset_order = ["159915.SZ", "511010.SH"]
    asset_universe_path = tmp_path / "asset_universe.csv"
    pd.DataFrame(
        [
            {
                "ts_code": "159915.SZ",
                "symbol": "159915",
                "name": "创业板ETF",
                "type": "ETF",
                "pool": "equity",
                "status": "ok",
                "rows": 2,
                "first_date": "2024-01-02",
                "last_date": "2024-01-03",
                "median_amount_last_252": 1000.0,
                "raw_path": "data/raw/159915.SZ_daily.parquet",
            },
            {
                "ts_code": "511010.SH",
                "symbol": "511010",
                "name": "国债ETF",
                "type": "ETF",
                "pool": "bond",
                "status": "ok",
                "rows": 2,
                "first_date": "2024-01-02",
                "last_date": "2024-01-03",
                "median_amount_last_252": 2000.0,
                "raw_path": "data/raw/511010.SH_daily.parquet",
            },
        ]
    ).to_csv(asset_universe_path, index=False)

    config = deepcopy(DEFAULT_CONFIG)
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    config["data"]["asset_universe_path"] = str(asset_universe_path)
    _write_required_market_dataset_files(config, tmp_path, asset_order)

    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "ts_code": ["159915.SZ", "159915.SZ"],
            "close": [1.0, 1.1],
        }
    )
    panel.to_parquet(config["data"]["panel_path"])
    metrics = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "ts_code": ["159915.SZ", "159915.SZ"],
            "feature_a": [0.1, 0.2],
        }
    )
    metrics.to_parquet(config["data"]["all_metrics_features_path"])

    bundle = load_market_dataset(config)

    assert bundle.panel["trade_date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert bundle.metrics_features is not None
    assert bundle.metrics_features["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert bundle.data_manifest["metrics_factory_enabled"] is True

    duplicate_panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    duplicate_panel.to_parquet(config["data"]["panel_path"])
    with pytest.raises(DataContractError) as panel_error:
        load_market_dataset(config)
    assert panel_error.value.code == "ERR_DATA_SCHEMA_MISMATCH"

    panel.to_parquet(config["data"]["panel_path"])
    duplicate_metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
    duplicate_metrics.to_parquet(config["data"]["all_metrics_features_path"])
    with pytest.raises(DataContractError) as metrics_error:
        load_market_dataset(config)
    assert metrics_error.value.code == "ERR_DATA_SCHEMA_MISMATCH"

    metrics.to_parquet(config["data"]["all_metrics_features_path"])
    missing_metrics_config = deepcopy(config)
    missing_metrics_path = tmp_path / "missing_metrics.parquet"
    missing_metrics_config["data"]["all_metrics_features_path"] = str(missing_metrics_path)
    with pytest.raises(DataContractError) as missing_error:
        load_market_dataset(missing_metrics_config)
    assert missing_error.value.code == "ERR_DATA_MISSING_FILE"
    assert "missing_metrics.parquet" in str(missing_error.value)

    disabled_config = deepcopy(missing_metrics_config)
    disabled_config["data"]["metrics_factory"]["enabled"] = False
    disabled_bundle = load_market_dataset(disabled_config)
    assert disabled_bundle.metrics_features is None
    assert disabled_bundle.data_manifest["metrics_factory_enabled"] is False


def test_availability_reason_is_exclusive(tmp_path):
    asset_order = ["159915.SZ"]
    asset_universe_path = tmp_path / "asset_universe.csv"
    pd.DataFrame(
        [
            {
                "ts_code": "159915.SZ",
                "symbol": "159915",
                "name": "创业板ETF",
                "type": "ETF",
                "pool": "equity",
                "status": "ok",
                "rows": 6,
                "first_date": "2024-01-03",
                "last_date": "2024-01-07",
                "median_amount_last_252": 1000.0,
                "raw_path": "data/raw/159915.SZ_daily.parquet",
            }
        ]
    ).to_csv(asset_universe_path, index=False)

    config = deepcopy(DEFAULT_CONFIG)
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    config["data"]["asset_universe_path"] = str(asset_universe_path)
    _write_required_market_dataset_files(config, tmp_path, asset_order)

    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),
            pd.Timestamp("2024-01-06"),
            pd.Timestamp("2024-01-08"),
        ]
    )
    values = {
        "open": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "high": [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        "low": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        "close": [1.0, 1.0, 1.0, np.nan, 1.0, 1.0],
        "pre_close": [0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
        "pct_chg": [0.01, 0.01, 0.01, 0.01, np.nan, 0.01],
        "log_return": [0.01, 0.01, 0.01, 0.01, np.nan, 0.01],
        "amount": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        "vol": [100.0, 100.0, 0.0, 100.0, 100.0, 100.0],
        "turnover_rate": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    for field, series in values.items():
        pd.DataFrame({"159915.SZ": series}, index=dates).to_parquet(config["data"][f"wide_{field}_path"])

    bundle = load_market_dataset(config)

    assert bundle.availability_mask["159915.SZ"].tolist() == [False, True, False, False, False, False]
    assert bundle.availability_reason["159915.SZ"].tolist() == [
        "not_listed_yet",
        "listed",
        "suspended",
        "missing_quote",
        "missing_return",
        "delisted",
    ]
    assert set(bundle.availability_reason.stack()) == {
        "listed",
        "not_listed_yet",
        "delisted",
        "suspended",
        "missing_quote",
        "missing_return",
    }
    assert bundle.data_manifest["availability_reason_inference"] == "field_based"


def test_turnover_rate_all_missing_policy(tmp_path):
    asset_order = ["159915.SZ"]
    asset_universe_path = tmp_path / "asset_universe.csv"
    pd.DataFrame(
        [
            {
                "ts_code": "159915.SZ",
                "symbol": "159915",
                "name": "创业板ETF",
                "type": "ETF",
                "pool": "equity",
                "status": "ok",
                "rows": 2,
                "first_date": "2024-01-02",
                "last_date": "2024-01-03",
                "median_amount_last_252": 8500.0,
                "raw_path": "data/raw/159915.SZ_daily.parquet",
            }
        ]
    ).to_csv(asset_universe_path, index=False)

    config = deepcopy(DEFAULT_CONFIG)
    config["security"]["path_whitelist"] = [str(PROJECT_ROOT), str(tmp_path)]
    config["data"]["asset_universe_path"] = str(asset_universe_path)
    _write_required_market_dataset_files(config, tmp_path, asset_order)

    dates = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
    values = {
        "open": [10.0, 12.0],
        "high": [10.5, 12.5],
        "low": [9.5, 11.5],
        "close": [10.0, 12.0],
        "pre_close": [9.9, 10.0],
        "pct_chg": [0.01, 0.20],
        "log_return": [0.01, 0.18],
        "amount": [5000.0, 12000.0],
        "vol": [5.0, 10.0],
        "turnover_rate": [np.nan, np.nan],
    }
    for field, series in values.items():
        pd.DataFrame({"159915.SZ": series}, index=dates).to_parquet(config["data"][f"wide_{field}_path"])

    bundle = load_market_dataset(config)

    assert bundle.data_manifest["amount_is_proxy"] is True
    assert bundle.data_manifest["turnover_rate_all_missing"] is True

    required_config = deepcopy(config)
    required_config["data_governance"]["turnover_rate_required"] = True
    with pytest.raises(DataContractError) as error:
        load_market_dataset(required_config)
    assert error.value.code == "ERR_DATA_TURNOVER_RATE_REQUIRED"
