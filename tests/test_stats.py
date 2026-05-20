import numpy as np
import pandas as pd
import pytest

from src.utils.logger import write_run_outputs
from src.utils.stats import STATISTICS_SUMMARY_COLUMNS, run_statistical_tests


def test_run_statistical_tests_inner_joins_dates_and_adjusts_pvalues(tmp_path):
    model_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]),
            "model_name": ["model_a"] * 6,
            "split": ["test"] * 6,
            "net_return": [0.020, 0.010, 0.015, -0.005, 0.012, 0.018],
        }
    )
    benchmark_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05", "2024-01-06"]),
            "benchmark_name": ["benchmark"] * 5,
            "split": ["test"] * 5,
            "net_return": [0.010, 0.004, -0.002, 0.004, 0.010],
        }
    )
    config = {
        "statistics": {
            "min_paired_samples": 4,
            "bootstrap": {"n_bootstrap": 64, "block_length": 2, "confidence_level": 0.90, "seed": 7},
            "hac": {"lags": 1},
            "multiple_testing": {"alpha": 0.05},
        }
    }

    output_path = tmp_path / "statistics_summary.csv"
    summary = run_statistical_tests(model_returns, benchmark_returns, config=config, output_path=output_path)

    assert list(summary.columns) == list(STATISTICS_SUMMARY_COLUMNS)
    assert output_path.exists()
    assert set(summary["test_name"]) == {
        "block_bootstrap",
        "hac",
        "psr",
        "dsr",
        "white_reality_check",
        "hansen_spa",
        "diebold_mariano",
    }
    return_rows = summary[summary["test_name"].ne("diebold_mariano")]
    assert return_rows["status"].eq("pass").all()
    assert return_rows["n_obs"].eq(5).all()

    bootstrap = summary.loc[summary["test_name"].eq("block_bootstrap")].iloc[0]
    expected_diff = np.array([0.010, 0.006, -0.003, 0.008, 0.008]).mean()
    assert bootstrap["effect_size"] == pytest.approx(expected_diff)
    assert np.isfinite(float(bootstrap["ci_lower"]))
    assert np.isfinite(float(bootstrap["ci_upper"]))

    adjusted = return_rows[pd.to_numeric(return_rows["raw_p_value"], errors="coerce").notna()]
    assert adjusted["adjustment_method"].eq("holm_bonferroni").all()
    assert pd.to_numeric(adjusted["adjusted_p_value"], errors="coerce").notna().all()

    dm = summary.loc[summary["test_name"].eq("diebold_mariano")].iloc[0]
    assert dm["status"] == "not_applicable"


def test_statistical_tests_skip_insufficient_samples_and_dm_uses_auxiliary_errors():
    model_returns = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "model_name": ["model_a"] * 3,
            "split": ["test"] * 3,
            "net_return": [0.010, 0.020, 0.015],
        }
    )
    benchmark_returns = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "benchmark_name": ["benchmark"] * 3,
            "split": ["test"] * 3,
            "net_return": [0.009, 0.015, 0.011],
        }
    )
    auxiliary_errors = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "model_name": ["model_a"] * 5,
            "benchmark_name": ["benchmark"] * 5,
            "horizon": [2] * 5,
            "model_error": [0.10, 0.08, 0.07, 0.06, 0.05],
            "benchmark_error": [0.12, 0.10, 0.09, 0.08, 0.07],
        }
    )

    summary = run_statistical_tests(
        model_returns,
        benchmark_returns,
        config={"statistics": {"min_paired_samples": 4, "bootstrap": {"n_bootstrap": 16, "seed": 11}}},
        auxiliary_forecast_errors=auxiliary_errors,
    )

    return_rows = summary[summary["test_name"].ne("diebold_mariano")]
    assert return_rows["status"].eq("skipped").all()
    assert return_rows["skip_reason"].eq("insufficient_samples").all()
    dm = summary.loc[summary["test_name"].eq("diebold_mariano")].iloc[0]
    assert dm["status"] == "pass"
    assert dm["metric_name"] == "auxiliary_forecast_error"
    assert dm["n_obs"] == 5
    assert dm["effect_size"] < 0.0


def test_write_run_outputs_persists_statistics_summary(tmp_path):
    result = {
        "daily_returns": pd.DataFrame({"next_valuation_date": ["2024-01-01"], "net_return": [0.01]}),
        "model_returns": pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "model_name": ["model_a"] * 5,
                "split": ["test"] * 5,
                "net_return": [0.020, 0.010, 0.015, -0.005, 0.012],
            }
        ),
        "benchmark_returns": pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "benchmark_name": ["benchmark"] * 5,
                "split": ["test"] * 5,
                "net_return": [0.010, 0.004, 0.010, -0.002, 0.004],
            }
        ),
    }

    artifacts = write_run_outputs(
        result,
        tmp_path,
        config={"statistics": {"min_paired_samples": 4, "bootstrap": {"n_bootstrap": 16, "seed": 3}}},
    )

    assert artifacts["statistics_summary"] == tmp_path / "metrics" / "statistics_summary.csv"
    persisted = pd.read_csv(artifacts["statistics_summary"])
    assert list(persisted.columns) == list(STATISTICS_SUMMARY_COLUMNS)
    assert persisted["test_name"].eq("hac").any()
