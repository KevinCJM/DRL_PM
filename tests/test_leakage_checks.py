import numpy as np
import pandas as pd
import pytest

from src.data.leakage_checks import (
    assert_no_execution_field_in_observation,
    assert_no_future_label_in_features,
    assert_pca_not_fit_on_validation_or_test,
    assert_rolling_estimator_uses_past_only,
)
from src.data.loader import DataContractError
from src.data.splits import SplitSpec


def test_future_label_feature_rejected():
    assert_no_future_label_in_features(
        ["close_ma_ratio_5", "rolling_volatility_20"],
        ["future_log_return_5d"],
    )

    with pytest.raises(DataContractError) as error:
        assert_no_future_label_in_features(
            ["close_ma_ratio_5", "future_log_return_5d"],
            ["future_log_return_5d"],
        )

    assert error.value.code == "ERR_LEAKAGE_FUTURE_LABEL"

    with pytest.raises(DataContractError) as blacklist_error:
        assert_no_future_label_in_features(["next_close_return"])
    assert blacklist_error.value.code == "ERR_LEAKAGE_FUTURE_LABEL"


def test_pca_fit_scope_must_be_train_only():
    split = SplitSpec(
        train_dates=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]),
        validation_dates=pd.DatetimeIndex(["2024-01-05"]),
        test_dates=pd.DatetimeIndex(["2024-01-08"]),
        fold_id="fixed",
    )

    assert_pca_not_fit_on_validation_or_test(
        pd.DatetimeIndex(["2024-01-02", "2024-01-04"]),
        split,
    )

    with pytest.raises(DataContractError) as error:
        assert_pca_not_fit_on_validation_or_test(
            pd.DatetimeIndex(["2024-01-02", "2024-01-05"]),
            split,
        )
    assert error.value.code == "ERR_LEAKAGE_PCA_FIT_SCOPE"

    with pytest.raises(DataContractError) as scope_error:
        assert_pca_not_fit_on_validation_or_test(
            pd.DatetimeIndex(["2024-01-02"]),
            split,
            fit_scope="all",
        )
    assert scope_error.value.code == "ERR_LEAKAGE_PCA_FIT_SCOPE"


def test_rolling_estimator_rejects_future_window():
    provenance = pd.DataFrame(
        [
            {
                "feature_name": "rolling_volatility_20",
                "date": "2024-01-05",
                "window_end_date": "2024-01-05",
                "center": False,
            }
        ]
    )
    assert_rolling_estimator_uses_past_only(provenance)

    future_window = provenance.copy()
    future_window.loc[0, "window_end_date"] = "2024-01-08"
    with pytest.raises(DataContractError) as error:
        assert_rolling_estimator_uses_past_only(future_window)
    assert error.value.code == "ERR_LEAKAGE_ROLLING_FUTURE"

    centered_window = provenance.copy()
    centered_window.loc[0, "center"] = True
    with pytest.raises(DataContractError) as center_error:
        assert_rolling_estimator_uses_past_only(centered_window)
    assert center_error.value.code == "ERR_LEAKAGE_ROLLING_FUTURE"


def test_execution_only_field_rejected_from_observation():
    assert_no_execution_field_in_observation(
        {
            "market_image": np.zeros((2, 3, 1)),
            "feature_window": ["close_at_decision", "amount_at_decision", "available_mask_at_decision"],
        }
    )

    with pytest.raises(DataContractError) as error:
        assert_no_execution_field_in_observation(
            {
                "market_image": np.zeros((2, 3, 1)),
                "feature_window": ["close_at_decision", "amount_at_execution"],
            }
        )
    assert error.value.code == "ERR_LEAKAGE_EXECUTION_FIELD"

    with pytest.raises(DataContractError) as nested_error:
        assert_no_execution_field_in_observation({"state": {"holding_simple_return": np.array([0.01])}})
    assert nested_error.value.code == "ERR_LEAKAGE_EXECUTION_FIELD"

    with pytest.raises(DataContractError) as after_execution_error:
        assert_no_execution_field_in_observation(["availability_after_execution"])
    assert after_execution_error.value.code == "ERR_LEAKAGE_EXECUTION_FIELD"
