from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.data.feature_matrix import FeatureMatrix, FeatureMatrixBuilder, MarketImageDataset, TECHNICAL_WINDOWS
from src.data.loader import DataContractError, MarketDatasetBundle


def test_g2_feature_formulas():
    dates = pd.date_range("2024-01-02", periods=130, freq="D")
    asset_order = ["510300.SH", "159915.SZ"]
    close_base = np.arange(1.0, 131.0)
    wide_close = pd.DataFrame(
        {
            "510300.SH": close_base,
            "159915.SZ": close_base * 2.0,
        },
        index=dates,
    )
    wide_open = wide_close - 0.25
    wide_high = wide_close + 0.50
    wide_low = wide_close - 0.50
    wide_pre_close = wide_close.shift(1)
    wide_pct_chg = wide_close / wide_pre_close - 1.0
    wide_log_return = np.log(wide_close / wide_pre_close)
    wide_vol = pd.DataFrame(
        {
            "510300.SH": np.arange(10.0, 140.0),
            "159915.SZ": np.arange(20.0, 150.0),
        },
        index=dates,
    )
    wide_amount = wide_close * wide_vol * 100.0
    wide_turnover_rate = pd.DataFrame(
        {
            "510300.SH": np.linspace(0.01, 0.20, len(dates)),
            "159915.SZ": np.linspace(0.02, 0.21, len(dates)),
        },
        index=dates,
    )
    bundle = MarketDatasetBundle(
        asset_universe=pd.DataFrame({"ts_code": asset_order}),
        panel=pd.DataFrame(),
        wide={
            "open": wide_open,
            "high": wide_high,
            "low": wide_low,
            "close": wide_close,
            "pre_close": wide_pre_close,
            "pct_chg": wide_pct_chg,
            "log_return": wide_log_return,
            "amount": wide_amount,
            "vol": wide_vol,
            "turnover_rate": wide_turnover_rate,
        },
        metrics_features=None,
        feature_cols=[],
        auxiliary_target_cols=["amount_ratio_20"],
        availability_mask=pd.DataFrame(True, index=dates, columns=asset_order),
        availability_reason=None,
        data_manifest={"canonical_asset_order": asset_order},
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["feature_matrix"]["input_matrix_id"] = "M2"

    result = FeatureMatrixBuilder(config).build(bundle)

    for window in TECHNICAL_WINDOWS:
        assert f"close_ma_ratio_{window}" in result.feature_cols
        assert f"momentum_log_return_{window}" in result.feature_cols
        assert f"rolling_log_return_sum_{window}" in result.feature_cols
        assert f"rolling_volatility_{window}" in result.feature_cols
        assert f"rolling_downside_vol_{window}" in result.feature_cols
        assert f"drawdown_signed_{window}" in result.feature_cols
        assert f"drawdown_abs_{window}" in result.feature_cols
        assert f"max_drawdown_abs_{window}" in result.feature_cols

    assert "amount_ratio_20" not in result.feature_cols
    assert result.feature_panel.columns[:2].tolist() == ["date", "ts_code"]
    assert result.feature_panel[["date", "ts_code"]].head(4).values.tolist() == [
        [pd.Timestamp("2024-01-02"), "510300.SH"],
        [pd.Timestamp("2024-01-02"), "159915.SZ"],
        [pd.Timestamp("2024-01-03"), "510300.SH"],
        [pd.Timestamp("2024-01-03"), "159915.SZ"],
    ]

    row = result.feature_panel.loc[
        (result.feature_panel["date"] == dates[-1]) & (result.feature_panel["ts_code"] == "510300.SH")
    ].iloc[0]
    close = wide_close["510300.SH"]
    pct_chg = wide_pct_chg["510300.SH"]
    log_return = wide_log_return["510300.SH"]
    amount = wide_amount["510300.SH"]
    vol = wide_vol["510300.SH"]
    turnover_rate = wide_turnover_rate["510300.SH"]

    assert row["close_ma_ratio_5"] == pytest.approx(close.iloc[-1] / close.rolling(5).mean().iloc[-1] - 1.0)
    assert row["ma_5_over_20_ratio"] == pytest.approx(
        close.rolling(5).mean().iloc[-1] / close.rolling(20).mean().iloc[-1] - 1.0
    )
    assert row["ma_20_over_60_ratio"] == pytest.approx(
        close.rolling(20).mean().iloc[-1] / close.rolling(60).mean().iloc[-1] - 1.0
    )
    assert row["momentum_log_return_5"] == pytest.approx(np.log(close.iloc[-1] / close.shift(5).iloc[-1]))
    assert row["rolling_log_return_sum_5"] == pytest.approx(log_return.rolling(5).sum().iloc[-1])
    assert row["rolling_volatility_5"] == pytest.approx(pct_chg.rolling(5).std().iloc[-1])
    assert row["rolling_downside_vol_5"] == pytest.approx(pct_chg.clip(upper=0.0).rolling(5).std().iloc[-1])
    assert row["drawdown_signed_5"] == pytest.approx(close.iloc[-1] / close.rolling(5).max().iloc[-1] - 1.0)
    assert row["drawdown_abs_5"] == pytest.approx(max(0.0, 1.0 - close.iloc[-1] / close.rolling(5).max().iloc[-1]))
    assert row["max_drawdown_abs_5"] == pytest.approx(
        (1.0 - close / close.rolling(5).max()).clip(lower=0.0).rolling(5).max().iloc[-1]
    )
    assert row["log1p_amount"] == pytest.approx(np.log1p(amount.iloc[-1]))
    assert row["log1p_vol"] == pytest.approx(np.log1p(vol.iloc[-1]))
    assert row["amount_ratio_20"] == pytest.approx(amount.iloc[-1] / amount.rolling(20).mean().iloc[-1] - 1.0)
    assert row["vol_ratio_20"] == pytest.approx(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1] - 1.0)
    assert row["turnover_rate"] == pytest.approx(turnover_rate.iloc[-1])
    assert row["turnover_rate_ma_20"] == pytest.approx(turnover_rate.rolling(20).mean().iloc[-1])
    assert row["high_low_over_close"] == pytest.approx(
        (wide_high["510300.SH"].iloc[-1] - wide_low["510300.SH"].iloc[-1]) / close.iloc[-1]
    )
    assert row["close_open_over_open"] == pytest.approx(
        (close.iloc[-1] - wide_open["510300.SH"].iloc[-1]) / wide_open["510300.SH"].iloc[-1]
    )


def test_metrics_factory_full_audit_statuses():
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    asset_order = ["510300.SH", "159915.SZ"]
    base = pd.DataFrame(
        {
            "510300.SH": [1.0, 1.1, 1.2],
            "159915.SZ": [2.0, 2.1, 2.2],
        },
        index=dates,
    )
    metrics_features = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ts_code": ["510300.SH", "159915.SZ", "510300.SH", "159915.SZ"],
            "clean_metric": [0.1, 0.2, 0.3, 0.4],
            "needs_warning_metric": [1.0, 1.1, 1.2, 1.3],
            "future_alpha": [2.0, 2.1, 2.2, 2.3],
            "dropped_aux_metric": [3.0, 3.1, 3.2, 3.3],
        }
    )
    bundle = MarketDatasetBundle(
        asset_universe=pd.DataFrame({"ts_code": asset_order}),
        panel=pd.DataFrame(),
        wide={
            "open": base,
            "high": base,
            "low": base,
            "close": base,
            "pre_close": base.shift(1),
            "pct_chg": base.pct_change(),
            "log_return": np.log(base / base.shift(1)),
            "amount": base * 1000.0,
            "vol": base * 10.0,
            "turnover_rate": base * 0.01,
        },
        metrics_features=metrics_features,
        feature_cols=[],
        auxiliary_target_cols=["dropped_aux_metric"],
        availability_mask=pd.DataFrame(True, index=dates, columns=asset_order),
        availability_reason=None,
        data_manifest={"canonical_asset_order": asset_order},
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["feature_matrix"]["input_matrix_id"] = "M3"

    result = FeatureMatrixBuilder(config).build(bundle)

    provenance = result.provenance.set_index("feature_name")
    assert "clean_metric" in result.feature_cols
    assert "needs_warning_metric" in result.feature_cols
    assert "future_alpha" not in result.feature_cols
    assert "dropped_aux_metric" not in result.feature_cols
    assert provenance.loc["clean_metric", "leakage_check_status"] == "pass"
    assert provenance.loc["needs_warning_metric", "leakage_check_status"] == "warning"
    assert provenance.loc["future_alpha", "leakage_check_status"] == "fail"
    assert provenance.loc["future_alpha", "drop_reason"] == "blacklist_pattern"
    assert provenance.loc["dropped_aux_metric", "leakage_check_status"] == "dropped"
    assert provenance.loc["dropped_aux_metric", "drop_reason"] == "auxiliary_target"
    assert provenance.loc["dropped_aux_metric", "is_auxiliary_target"] == True
    assert provenance.loc["future_alpha", "is_metrics_factory_feature"] == True
    assert result.feature_panel.loc[
        (result.feature_panel["date"] == dates[1]) & (result.feature_panel["ts_code"] == "159915.SZ"),
        "clean_metric",
    ].iloc[0] == pytest.approx(0.4)

    drop_warning_config = deepcopy(config)
    drop_warning_config["feature_audit"]["warning_policy"] = "drop"
    drop_warning_result = FeatureMatrixBuilder(drop_warning_config).build(bundle)
    drop_warning_provenance = drop_warning_result.provenance.set_index("feature_name")
    assert "needs_warning_metric" not in drop_warning_result.feature_cols
    assert drop_warning_provenance.loc["needs_warning_metric", "leakage_check_status"] == "warning"
    assert drop_warning_provenance.loc["needs_warning_metric", "drop_reason"] == "warning_policy_drop"


def test_feature_matrix_fail_fast_error_codes(sample_market_dataset_bundle):
    invalid_config = deepcopy(DEFAULT_CONFIG)
    invalid_config["feature_matrix"]["input_matrix_id"] = "M_UNKNOWN"
    with pytest.raises(DataContractError) as invalid_error:
        FeatureMatrixBuilder(invalid_config).build(sample_market_dataset_bundle)
    assert invalid_error.value.code == "ERR_FEATURE_MATRIX_INVALID_INPUT_MATRIX"

    missing_wide_config = deepcopy(DEFAULT_CONFIG)
    missing_wide_config["feature_matrix"]["input_matrix_id"] = "M2"
    missing_wide = dict(sample_market_dataset_bundle.wide)
    missing_wide.pop("close")
    missing_wide_bundle = replace(sample_market_dataset_bundle, wide=missing_wide)
    with pytest.raises(DataContractError) as missing_wide_error:
        FeatureMatrixBuilder(missing_wide_config).build(missing_wide_bundle)
    assert missing_wide_error.value.code == "ERR_DATA_SCHEMA_MISMATCH"

    metrics_schema_config = deepcopy(DEFAULT_CONFIG)
    metrics_schema_config["feature_matrix"]["input_matrix_id"] = "M3"
    bad_metrics_bundle = replace(
        sample_market_dataset_bundle,
        metrics_features=pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "clean_metric": [1.0]}),
    )
    with pytest.raises(DataContractError) as metrics_schema_error:
        FeatureMatrixBuilder(metrics_schema_config).build(bad_metrics_bundle)
    assert metrics_schema_error.value.code == "ERR_DATA_SCHEMA_MISMATCH"


def test_g4_cross_asset_features_are_past_only():
    dates = pd.date_range("2024-01-02", periods=260, freq="D")
    asset_order = ["510300.SH", "159915.SZ", "513100.SH"]
    steps = np.arange(len(dates), dtype=float)
    pct_chg = pd.DataFrame(
        {
            "510300.SH": 0.0022 + 0.0002 * np.sin(steps / 7.0),
            "159915.SZ": 0.0018 + 0.0003 * np.cos(steps / 9.0),
            "513100.SH": 0.0015 + 0.0004 * np.sin(steps / 5.0),
        },
        index=dates,
    )
    pct_chg.iloc[0] = 0.0
    pct_chg.loc[dates[-1], "513100.SH"] = 0.50
    availability = pd.DataFrame(True, index=dates, columns=asset_order)
    availability.loc[dates[-1], "513100.SH"] = False

    def make_bundle(returns: pd.DataFrame, mask: pd.DataFrame) -> MarketDatasetBundle:
        close = (1.0 + returns).cumprod()
        return MarketDatasetBundle(
            asset_universe=pd.DataFrame({"ts_code": asset_order}),
            panel=pd.DataFrame(),
            wide={
                "open": close.shift(1).fillna(close),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "pre_close": close.shift(1),
                "pct_chg": returns,
                "log_return": np.log(1.0 + returns),
                "amount": close * 1000.0,
                "vol": close * 10.0,
                "turnover_rate": close * 0.01,
            },
            metrics_features=None,
            feature_cols=[],
            auxiliary_target_cols=[],
            availability_mask=mask,
            availability_reason=None,
            data_manifest={"canonical_asset_order": asset_order},
        )

    config = deepcopy(DEFAULT_CONFIG)
    config["feature_matrix"]["input_matrix_id"] = "M4"
    result = FeatureMatrixBuilder(config).build(make_bundle(pct_chg, availability))

    for feature in (
        "rolling_beta_to_benchmark_20",
        "rolling_corr_to_benchmark_20",
        "rolling_cov_to_benchmark_20",
        "asset_vol_20",
        "asset_vol_over_benchmark_vol_20",
        "benchmark_return_20",
        "benchmark_return_60",
        "benchmark_vol_20",
        "benchmark_vol_percentile_252",
        "trend_signal",
        "eigen_loading_1",
        "eigen_loading_2",
        "eigen_loading_3",
    ):
        assert feature in result.feature_cols

    provenance = result.provenance.set_index("feature_name")
    assert provenance.loc["rolling_beta_to_benchmark_20", "feature_group"] == "G4"
    assert provenance.loc["rolling_beta_to_benchmark_20", "uses_cross_asset_data"] == True
    assert provenance.loc["rolling_beta_to_benchmark_20", "window"] == 20

    row = result.feature_panel.loc[
        (result.feature_panel["date"] == dates[-1]) & (result.feature_panel["ts_code"] == "510300.SH")
    ].iloc[0]
    unavailable_row = result.feature_panel.loc[
        (result.feature_panel["date"] == dates[-1]) & (result.feature_panel["ts_code"] == "513100.SH")
    ].iloc[0]
    masked_returns = pct_chg.where(availability)
    benchmark_return = masked_returns.mean(axis=1)
    unmasked_benchmark_return = pct_chg.mean(axis=1)
    expected_benchmark_return_20 = np.prod(1.0 + benchmark_return.iloc[-20:]) - 1.0
    unmasked_benchmark_return_20 = np.prod(1.0 + unmasked_benchmark_return.iloc[-20:]) - 1.0
    expected_cov = masked_returns["510300.SH"].rolling(20).cov(benchmark_return).iloc[-1]
    expected_var = benchmark_return.rolling(20).var().iloc[-1]
    expected_asset_vol = masked_returns["510300.SH"].rolling(20).std().iloc[-1]
    expected_benchmark_vol = benchmark_return.rolling(20).std().iloc[-1]
    expected_return_120 = np.prod(1.0 + benchmark_return.iloc[-120:]) - 1.0

    assert row["benchmark_return_20"] == pytest.approx(expected_benchmark_return_20)
    assert abs(row["benchmark_return_20"] - unmasked_benchmark_return_20) > 0.01
    assert row["rolling_beta_to_benchmark_20"] == pytest.approx(expected_cov / expected_var)
    assert row["rolling_cov_to_benchmark_20"] == pytest.approx(expected_cov)
    assert row["asset_vol_20"] == pytest.approx(expected_asset_vol)
    assert row["asset_vol_over_benchmark_vol_20"] == pytest.approx(expected_asset_vol / expected_benchmark_vol)
    assert row["trend_signal"] == (1.0 if expected_return_120 >= 0.10 else 0.0)
    assert pd.isna(unavailable_row["benchmark_return_20"])

    loading_rows = result.feature_panel.loc[result.feature_panel["date"] == dates[-2]].set_index("ts_code")
    for feature in ("eigen_loading_1", "eigen_loading_2", "eigen_loading_3"):
        loadings = loading_rows.loc[asset_order, feature].dropna()
        assert not loadings.empty
        assert loadings.sum() >= -1.0e-12

    truncated_result = FeatureMatrixBuilder(config).build(make_bundle(pct_chg.iloc[:-1], availability.iloc[:-1]))
    full_previous = result.feature_panel.loc[
        (result.feature_panel["date"] == dates[-2]) & (result.feature_panel["ts_code"] == "510300.SH")
    ].iloc[0]
    truncated_previous = truncated_result.feature_panel.loc[
        (truncated_result.feature_panel["date"] == dates[-2])
        & (truncated_result.feature_panel["ts_code"] == "510300.SH")
    ].iloc[0]
    for feature in ("benchmark_return_20", "rolling_beta_to_benchmark_20", "eigen_loading_1", "trend_signal"):
        assert full_previous[feature] == pytest.approx(truncated_previous[feature])


def test_feature_provenance_schema():
    dates = pd.date_range("2024-01-02", periods=130, freq="D")
    asset_order = ["510300.SH", "159915.SZ"]
    close = pd.DataFrame(
        {
            "510300.SH": np.linspace(1.0, 2.0, len(dates)),
            "159915.SZ": np.linspace(2.0, 3.0, len(dates)),
        },
        index=dates,
    )
    turnover_rate = pd.DataFrame(np.nan, index=dates, columns=asset_order)
    bundle = MarketDatasetBundle(
        asset_universe=pd.DataFrame({"ts_code": asset_order}),
        panel=pd.DataFrame(),
        wide={
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "pre_close": close.shift(1),
            "pct_chg": close.pct_change(),
            "log_return": np.log(close / close.shift(1)),
            "amount": close * 1000.0,
            "vol": close * 10.0,
            "turnover_rate": turnover_rate,
        },
        metrics_features=None,
        feature_cols=[],
        auxiliary_target_cols=[],
        availability_mask=pd.DataFrame(True, index=dates, columns=asset_order),
        availability_reason=None,
        data_manifest={"canonical_asset_order": asset_order, "turnover_rate_all_missing": True},
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["feature_matrix"]["input_matrix_id"] = "M2"

    result = FeatureMatrixBuilder(config).build(bundle)

    assert result.provenance.columns.tolist() == [
        "feature_name",
        "feature_group",
        "source_file",
        "source_family",
        "window",
        "uses_price",
        "uses_volume",
        "uses_return",
        "uses_cross_asset_data",
        "is_metrics_factory_feature",
        "is_auxiliary_target",
        "is_model_feature",
        "requires_shift",
        "shift_steps",
        "fit_scope",
        "leakage_risk_level",
        "leakage_check_status",
        "drop_reason",
    ]
    assert result.feature_group_summary.columns.tolist() == [
        "feature_group",
        "source_family",
        "n_total",
        "n_used",
        "n_dropped",
        "n_shifted",
        "n_train_only_fit",
        "n_warning",
        "n_fail",
    ]
    provenance = result.provenance.set_index("feature_name")
    assert "turnover_rate" not in result.feature_cols
    assert "turnover_rate_ma_20" not in result.feature_cols
    assert provenance.loc["turnover_rate", "leakage_check_status"] == "dropped"
    assert provenance.loc["turnover_rate", "drop_reason"] == "all_missing"
    assert provenance.loc["turnover_rate_ma_20", "leakage_check_status"] == "dropped"
    assert provenance.loc["turnover_rate_ma_20", "drop_reason"] == "all_missing"
    summary = result.feature_group_summary.set_index(["feature_group", "source_family"])
    assert summary.loc[("G2", "local_wide"), "n_dropped"] >= 2


def test_metrics_factory_audit_sample():
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    asset_order = ["510300.SH", "159915.SZ"]
    base = pd.DataFrame(
        {
            "510300.SH": [1.0, 1.1, 1.21],
            "159915.SZ": [2.0, 2.2, 2.42],
        },
        index=dates,
    )
    log_return = np.log(base / base.shift(1))
    expected_total_return = log_return["510300.SH"].iloc[1:3].sum()
    metrics_features = pd.DataFrame(
        {
            "date": [dates[2], dates[2]],
            "ts_code": ["510300.SH", "159915.SZ"],
            "TotalReturn:2d": [expected_total_return, log_return["159915.SZ"].iloc[1:3].sum()],
            "clean_metric": [1.0, 2.0],
        }
    )
    bundle = MarketDatasetBundle(
        asset_universe=pd.DataFrame({"ts_code": asset_order}),
        panel=pd.DataFrame(),
        wide={
            "open": base,
            "high": base,
            "low": base,
            "close": base,
            "pre_close": base.shift(1),
            "pct_chg": base.pct_change(),
            "log_return": log_return,
            "amount": base * 1000.0,
            "vol": base * 10.0,
            "turnover_rate": base * 0.01,
        },
        metrics_features=metrics_features,
        feature_cols=[],
        auxiliary_target_cols=[],
        availability_mask=pd.DataFrame(True, index=dates, columns=asset_order),
        availability_reason=None,
        data_manifest={"canonical_asset_order": asset_order},
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["feature_matrix"]["input_matrix_id"] = "M3"

    result = FeatureMatrixBuilder(config).build(bundle)

    assert result.metrics_factory_audit_sample.columns.tolist() == [
        "feature_name",
        "ts_code",
        "date",
        "stored_value",
        "recomputed_value",
        "abs_error",
        "status",
    ]
    sample = result.metrics_factory_audit_sample.set_index("feature_name")
    assert sample.index.tolist() == ["TotalReturn:2d"]
    assert sample.loc["TotalReturn:2d", "status"] == "pass"
    assert sample.loc["TotalReturn:2d", "stored_value"] == pytest.approx(expected_total_return)
    assert sample.loc["TotalReturn:2d", "recomputed_value"] == pytest.approx(expected_total_return)
    assert sample.loc["TotalReturn:2d", "abs_error"] == pytest.approx(0.0)
    provenance = result.provenance.set_index("feature_name")
    assert provenance.loc["clean_metric", "drop_reason"] == "not_recomputable"

    corrupted_metrics = metrics_features.copy()
    corrupted_metrics["TotalReturn:2d"] += 0.01
    corrupted_bundle = replace(bundle, metrics_features=corrupted_metrics)
    corrupted_result = FeatureMatrixBuilder(config).build(corrupted_bundle)
    failed_sample = corrupted_result.metrics_factory_audit_sample.set_index("feature_name")
    assert failed_sample.loc["TotalReturn:2d", "status"] == "fail"
    provenance = corrupted_result.provenance.set_index("feature_name")
    assert provenance.loc["TotalReturn:2d", "leakage_check_status"] == "fail"
    assert provenance.loc["TotalReturn:2d", "drop_reason"] == "metrics_factory_audit_failed"
    assert not bool(provenance.loc["TotalReturn:2d", "is_model_feature"])


def test_lazy_market_image_index_alignment():
    dates = pd.date_range("2024-01-02", periods=6, freq="D")
    asset_order = ["510300.SH", "159915.SZ"]
    rows = []
    for date_index, date in enumerate(dates):
        for asset_index, asset in enumerate(asset_order):
            rows.append(
                {
                    "date": date,
                    "ts_code": asset,
                    "factor_a": float(date_index * 10 + asset_index),
                    "factor_b": float(100 + date_index * 10 + asset_index),
                }
            )
    feature_matrix = FeatureMatrix(
        feature_panel=pd.DataFrame(rows),
        feature_cols=["factor_a", "factor_b"],
        provenance=pd.DataFrame(),
        feature_group_summary=pd.DataFrame(),
        metrics_factory_audit_sample=pd.DataFrame(),
    )

    dataset = MarketImageDataset(feature_matrix, window_size=3, asset_order=asset_order)

    assert dataset.market_images is None
    assert dataset.date_index.tolist() == dates[2:].tolist()
    assert len(dataset) == 4
    image = dataset[0]
    assert image.dtype == np.float32
    assert image.shape == (2, 3, 2)
    np.testing.assert_array_equal(
        image[0],
        np.array(
            [
                [0.0, 1.0],
                [10.0, 11.0],
                [20.0, 21.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        image[1],
        np.array(
            [
                [100.0, 101.0],
                [110.0, 111.0],
                [120.0, 121.0],
            ],
            dtype=np.float32,
        ),
    )
    image[0, 0, 0] = -999.0
    assert dataset[0][0, 0, 0] == pytest.approx(0.0)
    np.testing.assert_array_equal(dataset[pd.Timestamp("2024-01-05")], dataset[1])

    materialized = dataset.materialize()
    assert materialized.shape == (4, 2, 3, 2)
    assert dataset.market_images is not None
    np.testing.assert_array_equal(dataset[1], materialized[1])
    assert dataset.date_index.tolist() == dates[2:].tolist()

    leaking_matrix = replace(feature_matrix, feature_cols=["holding_simple_return"])
    with pytest.raises(DataContractError, match="ERR_LEAKAGE_EXECUTION_FIELD"):
        MarketImageDataset(leaking_matrix, window_size=3, asset_order=asset_order)
