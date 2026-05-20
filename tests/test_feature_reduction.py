from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG
from src.data.feature_matrix import FeatureMatrixBuilder
from src.data.feature_reduction import FeatureReductionPipeline
from src.data.loader import DataContractError
from src.data.splits import SplitSpec


def test_train_only_fit_scope():
    dates = pd.date_range("2024-01-02", periods=6, freq="D")
    panel = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "ts_code": ["510300.SH", "159915.SZ"] * len(dates),
            "feature_a": np.tile([1.0, 2.0], len(dates)) + np.repeat(np.arange(len(dates)), 2),
            "feature_b": np.tile([2.0, 4.0], len(dates)) + np.repeat(np.arange(len(dates)), 2),
            "availability_mask": [True, False] * len(dates),
            "future_log_return_5d": np.arange(len(dates) * 2, dtype=float),
            "portfolio_state": np.arange(len(dates) * 2, dtype=float) + 100.0,
        }
    )
    split = SplitSpec(
        train_dates=pd.DatetimeIndex(dates[:3]),
        validation_dates=pd.DatetimeIndex(dates[3:5]),
        test_dates=pd.DatetimeIndex(dates[5:]),
        fold_id="fixed",
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["feature_reduction"]["pca"]["fixed_components"] = 1

    pipeline = FeatureReductionPipeline(config)
    train_panel = panel[panel["date"].isin(split.train_dates)]
    validation_panel = panel[panel["date"].isin(split.validation_dates)]

    pipeline.fit(
        train_panel,
        feature_cols=[
            "feature_a",
            "feature_b",
            "availability_mask",
            "future_log_return_5d",
            "portfolio_state",
        ],
        split=split,
        auxiliary_target_cols=["future_log_return_5d"],
    )
    transformed = pipeline.transform(validation_panel)

    assert pipeline.fit_dates_.equals(split.train_dates)
    assert pipeline.input_feature_cols_ == ["feature_a", "feature_b"]
    assert pipeline.output_feature_cols_ == ["pca_component_1"]
    assert transformed.columns.tolist() == ["date", "ts_code", "availability_mask", "pca_component_1"]
    assert transformed["availability_mask"].tolist() == validation_panel["availability_mask"].tolist()
    assert "future_log_return_5d" not in transformed.columns
    assert "portfolio_state" not in transformed.columns

    with pytest.raises(DataContractError) as error:
        FeatureReductionPipeline(config).fit(
            panel,
            feature_cols=["feature_a", "feature_b"],
            split=split,
        )
    assert error.value.code == "ERR_LEAKAGE_PCA_FIT_SCOPE"


def test_feature_selection_uses_purged_mutual_information_label():
    dates = pd.date_range("2024-01-02", periods=20, freq="D")
    wide_log_return = pd.DataFrame(
        {
            "510300.SH": np.linspace(-0.03, 0.04, len(dates)),
        },
        index=dates,
    )
    future_log_return_5d = sum(wide_log_return["510300.SH"].shift(-step) for step in range(1, 6))
    panel = pd.DataFrame(
        {
            "date": dates,
            "ts_code": ["510300.SH"] * len(dates),
            "mi_driver": future_log_return_5d.fillna(0.0).to_numpy(),
            "weak_noise": np.tile([0.0, 1.0, 0.5, -0.5], 5),
            "availability_mask": True,
        }
    )
    split = SplitSpec(
        train_dates=pd.DatetimeIndex(dates),
        validation_dates=pd.DatetimeIndex([]),
        test_dates=pd.DatetimeIndex([]),
        fold_id="fixed",
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["feature_matrix"]["input_matrix_id"] = "M7"
    config["feature_reduction"]["feature_selection"]["enabled"] = True
    config["feature_reduction"]["feature_selection"]["max_features"] = 1
    config["feature_reduction"]["pca"]["fixed_components"] = 1

    pipeline = FeatureReductionPipeline(config).fit(
        panel,
        feature_cols=["mi_driver", "weak_noise"],
        split=split,
        wide_log_return=wide_log_return,
    )

    assert pipeline.selected_feature_cols_ == ["mi_driver"]
    report = pipeline.feature_selection_report_.set_index("feature_name")
    assert report.loc["mi_driver", "selected"] == True
    assert report.loc["weak_noise", "selected"] == False
    assert report["skip_reason"].fillna("").eq("").all()


def test_feature_selection_drops_non_finite_mi_targets():
    dates = pd.date_range("2024-01-02", periods=18, freq="D")
    returns = np.linspace(-0.01, 0.02, len(dates))
    returns[4] = np.inf
    wide_log_return = pd.DataFrame({"510300.SH": returns}, index=dates)
    panel = pd.DataFrame(
        {
            "date": dates,
            "ts_code": ["510300.SH"] * len(dates),
            "feature_a": np.linspace(0.0, 1.0, len(dates)),
            "feature_b": np.tile([0.0, 1.0, -1.0], 6),
            "availability_mask": True,
        }
    )
    split = SplitSpec(
        train_dates=pd.DatetimeIndex(dates),
        validation_dates=pd.DatetimeIndex([]),
        test_dates=pd.DatetimeIndex([]),
        fold_id="fixed",
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["feature_matrix"]["input_matrix_id"] = "M7"
    config["feature_reduction"]["feature_selection"]["enabled"] = True
    config["feature_reduction"]["feature_selection"]["max_features"] = 1
    config["feature_reduction"]["pca"]["fixed_components"] = 1

    pipeline = FeatureReductionPipeline(config).fit(
        panel,
        feature_cols=["feature_a", "feature_b"],
        split=split,
        wide_log_return=wide_log_return,
    )

    assert len(pipeline.selected_feature_cols_) == 1
    report = pipeline.feature_selection_report_
    assert np.isfinite(pd.to_numeric(report["mi_score"], errors="coerce").dropna()).all()


def test_feature_selection_requires_target_source():
    dates = pd.date_range("2024-01-02", periods=8, freq="D")
    panel = pd.DataFrame(
        {
            "date": dates,
            "ts_code": ["510300.SH"] * len(dates),
            "feature_a": np.arange(len(dates), dtype=float),
            "feature_b": np.arange(len(dates), dtype=float) * 2.0,
        }
    )
    split = SplitSpec(
        train_dates=pd.DatetimeIndex(dates),
        validation_dates=pd.DatetimeIndex([]),
        test_dates=pd.DatetimeIndex([]),
        fold_id="fixed",
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["feature_matrix"]["input_matrix_id"] = "M7"
    config["feature_reduction"]["feature_selection"]["enabled"] = True

    with pytest.raises(DataContractError) as error:
        FeatureReductionPipeline(config).fit(panel, feature_cols=["feature_a", "feature_b"], split=split)

    assert error.value.code == "ERR_FEATURE_SELECTION_TARGET_MISSING"


def test_feature_reduction_fail_fast_error_codes():
    dates = pd.date_range("2024-01-02", periods=6, freq="D")
    panel = pd.DataFrame(
        {
            "date": dates,
            "ts_code": ["510300.SH"] * len(dates),
            "feature_a": np.arange(len(dates), dtype=float),
            "availability_mask": True,
        }
    )
    split = SplitSpec(
        train_dates=pd.DatetimeIndex(dates[:4]),
        validation_dates=pd.DatetimeIndex(dates[4:5]),
        test_dates=pd.DatetimeIndex(dates[5:]),
        fold_id="fixed",
    )

    with pytest.raises(DataContractError) as not_fitted_error:
        FeatureReductionPipeline(DEFAULT_CONFIG).transform(panel)
    assert not_fitted_error.value.code == "ERR_FEATURE_REDUCTION_NOT_FITTED"

    with pytest.raises(DataContractError) as empty_error:
        FeatureReductionPipeline(DEFAULT_CONFIG).fit(
            panel[panel["date"].isin(split.train_dates)],
            feature_cols=["availability_mask"],
            split=split,
        )
    assert empty_error.value.code == "ERR_FEATURE_REDUCTION_EMPTY"


def test_input_matrix_m0_to_m7_composition():
    expected_groups = {
        "M0": ("G0",),
        "M1": ("G1",),
        "M2": ("G1", "G2"),
        "M3": ("G1", "G3"),
        "M4": ("G1", "G2", "G4"),
        "M5": ("G1", "G2", "G3", "G4"),
        "M6": ("G1", "G2", "G3", "G4"),
        "M7": ("G1", "G2", "G3", "G4"),
    }
    expected_reduction = {
        "M0": (False, False),
        "M1": (False, False),
        "M2": (False, False),
        "M3": (False, False),
        "M4": (False, False),
        "M5": (False, False),
        "M6": (True, False),
        "M7": (True, True),
    }

    for input_matrix_id, groups in expected_groups.items():
        config = deepcopy(DEFAULT_CONFIG)
        config["feature_matrix"]["input_matrix_id"] = input_matrix_id
        pipeline = FeatureReductionPipeline(config)
        pca_enabled, feature_selection_enabled = expected_reduction[input_matrix_id]
        assert FeatureMatrixBuilder._resolve_groups(config) == groups
        assert pipeline.config["pca"]["enabled"] == pca_enabled
        assert pipeline.config["feature_selection"]["enabled"] == feature_selection_enabled

    dates = pd.date_range("2024-01-02", periods=20, freq="D")
    panel = pd.DataFrame(
        {
            "date": dates,
            "ts_code": ["510300.SH"] * len(dates),
            "feature_a": np.linspace(1.0, 2.0, len(dates)),
            "feature_b": np.linspace(2.0, 4.0, len(dates)),
            "availability_mask": [True, False] * 10,
        }
    )
    split = SplitSpec(
        train_dates=pd.DatetimeIndex(dates),
        validation_dates=pd.DatetimeIndex([]),
        test_dates=pd.DatetimeIndex([]),
        fold_id="fixed",
    )

    m6_config = deepcopy(DEFAULT_CONFIG)
    m6_config["feature_matrix"]["input_matrix_id"] = "M6"
    m6_config["feature_reduction"]["pca"]["fixed_components"] = 1
    m6_pipeline = FeatureReductionPipeline(m6_config).fit(
        panel,
        feature_cols=["feature_a", "feature_b", "availability_mask"],
        split=split,
    )
    m6_transformed = m6_pipeline.transform(panel)
    assert m6_pipeline.selected_feature_cols_ == ["feature_a", "feature_b"]
    assert m6_transformed.columns.tolist() == ["date", "ts_code", "availability_mask", "pca_component_1"]
    assert m6_transformed["availability_mask"].tolist() == panel["availability_mask"].tolist()

    wide_log_return = pd.DataFrame({"510300.SH": np.linspace(-0.03, 0.04, len(dates))}, index=dates)
    m7_config = deepcopy(DEFAULT_CONFIG)
    m7_config["feature_matrix"]["input_matrix_id"] = "M7"
    m7_config["feature_reduction"]["feature_selection"]["max_features"] = 1
    m7_config["feature_reduction"]["pca"]["fixed_components"] = 1
    m7_pipeline = FeatureReductionPipeline(m7_config).fit(
        panel,
        feature_cols=["feature_a", "feature_b", "availability_mask"],
        split=split,
        wide_log_return=wide_log_return,
    )
    m7_transformed = m7_pipeline.transform(panel)
    assert len(m7_pipeline.selected_feature_cols_) == 1
    assert int(m7_pipeline.pca_.n_features_in_) == 1
    assert m7_transformed.columns.tolist() == ["date", "ts_code", "availability_mask", "pca_component_1"]
    assert m7_transformed["availability_mask"].tolist() == panel["availability_mask"].tolist()
