from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.data.auxiliary_targets import (
    AUXILIARY_TARGET_COLUMNS,
    MASKED_RECONSTRUCTION_METADATA_COLUMNS,
    attach_auxiliary_targets,
    build_auxiliary_targets,
)
from src.data.feature_matrix import FeatureMatrixBuilder, MarketImageDataset
from src.data.loader import DataContractError, MarketDatasetBundle
from src.data.splits import SplitSpec


def test_auxiliary_targets_purge_split_boundaries():
    dates = pd.date_range("2024-01-02", periods=100, freq="D")
    steps = np.arange(len(dates), dtype=float)
    close = pd.DataFrame(
        {
            "510300.SH": np.exp(steps * 0.01),
            "159915.SZ": np.exp(steps * 0.02),
        },
        index=dates,
    )
    log_return = np.log(close / close.shift(1))
    pct_chg = close / close.shift(1) - 1.0
    wide = {"close": close, "log_return": log_return, "pct_chg": pct_chg}
    split = SplitSpec(
        train_dates=pd.DatetimeIndex(dates[:40]),
        validation_dates=pd.DatetimeIndex(dates[40:70]),
        test_dates=pd.DatetimeIndex(dates[70:]),
        fold_id="fixed",
    )

    result = build_auxiliary_targets(wide, split, DEFAULT_CONFIG)

    assert result.target_cols == AUXILIARY_TARGET_COLUMNS
    assert result.metadata_cols == MASKED_RECONSTRUCTION_METADATA_COLUMNS
    assert dates[0] in set(result.targets["date"])
    assert dates[20] not in set(result.targets["date"])
    assert dates[20] in set(result.purged_dates)

    row = result.targets.loc[
        (result.targets["date"] == dates[0]) & (result.targets["ts_code"] == "510300.SH")
    ].iloc[0]
    assert row["future_log_return_5d"] == pytest.approx(np.log(close["510300.SH"].iloc[5] / close["510300.SH"].iloc[0]))
    assert row["future_trend_10d"] == 1.0
    assert np.isfinite(row["future_volatility_20d"])
    assert np.isfinite(row["future_downside_volatility"])
    assert np.isfinite(row["future_max_drawdown"])
    assert np.isfinite(row["future_CVaR"])
    assert np.isfinite(row["future_correlation_or_covariance"])

    masked_row = result.targets.loc[
        (result.targets["date"] == dates[1]) & (result.targets["ts_code"] == "510300.SH")
    ].iloc[0]
    assert masked_row["masked_feature_reconstruction"] == pytest.approx(log_return["510300.SH"].iloc[1])
    assert masked_row["masked_feature_reconstruction_mask"] == 1.0
    assert masked_row["masked_feature_reconstruction_feature"] == "log_return"
    assert masked_row["masked_feature_reconstruction_window_offset"] == 0.0
    assert masked_row["masked_feature_reconstruction_asset"] == "510300.SH"


def test_auxiliary_targets_do_not_enter_features_or_market_image():
    dates = pd.date_range("2024-01-02", periods=30, freq="D")
    asset_order = ["510300.SH", "159915.SZ"]
    base = pd.DataFrame(
        {
            "510300.SH": np.linspace(1.0, 1.5, len(dates)),
            "159915.SZ": np.linspace(2.0, 2.6, len(dates)),
        },
        index=dates,
    )
    wide = {
        "open": base,
        "high": base * 1.01,
        "low": base * 0.99,
        "close": base,
        "pre_close": base.shift(1),
        "pct_chg": base.pct_change(),
        "log_return": np.log(base / base.shift(1)),
        "amount": base * 1000.0,
        "vol": base * 10.0,
        "turnover_rate": base * 0.01,
    }
    split = SplitSpec(
        train_dates=pd.DatetimeIndex(dates[:20]),
        validation_dates=pd.DatetimeIndex(dates[20:25]),
        test_dates=pd.DatetimeIndex(dates[25:]),
        fold_id="fixed",
    )
    bundle = MarketDatasetBundle(
        asset_universe=pd.DataFrame({"ts_code": asset_order}),
        panel=pd.DataFrame(),
        wide=wide,
        metrics_features=pd.DataFrame(
            {
                "date": [dates[0], dates[0]],
                "ts_code": asset_order,
                "clean_metric": [0.1, 0.2],
                "future_log_return_5d": [0.3, 0.4],
                "custom_aux": [0.5, 0.6],
            }
        ),
        feature_cols=[],
        auxiliary_target_cols=["custom_aux"],
        availability_mask=pd.DataFrame(True, index=dates, columns=asset_order),
        availability_reason=None,
        data_manifest={"canonical_asset_order": asset_order},
    )
    target_result = build_auxiliary_targets(bundle, split, DEFAULT_CONFIG)
    bundle_with_targets = attach_auxiliary_targets(bundle, target_result)
    config = deepcopy(DEFAULT_CONFIG)
    config["feature_matrix"]["input_matrix_id"] = "M3"

    feature_matrix = FeatureMatrixBuilder(config).build(bundle_with_targets)
    dataset = MarketImageDataset(feature_matrix, window_size=2)

    assert "custom_aux" in bundle_with_targets.auxiliary_target_cols
    assert "custom_aux" not in feature_matrix.feature_cols
    assert "custom_aux" not in dataset.feature_cols
    for target_col in [*target_result.target_cols, *target_result.metadata_cols]:
        assert target_col not in feature_matrix.feature_cols
        assert target_col not in dataset.feature_cols


def test_auxiliary_targets_missing_required_wide_fail_fast(sample_split_spec):
    with pytest.raises(DataContractError) as error:
        build_auxiliary_targets({"close": pd.DataFrame()}, sample_split_spec, DEFAULT_CONFIG)

    assert error.value.code == "ERR_DATA_SCHEMA_MISMATCH"
