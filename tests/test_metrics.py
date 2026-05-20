import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import calculate_performance_metrics


def test_calculate_performance_metrics_uses_net_return_and_drawdown(tmp_path):
    daily_returns = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "net_return": [0.10, -0.20, -0.05, 0.02],
            "nav": [99.0, 99.0, 99.0, 99.0],
        }
    )
    daily_turnover = pd.DataFrame({"turnover": [0.10, 0.30, 0.20, 0.40]})
    daily_costs = pd.DataFrame({"total_transaction_cost": [0.001, 0.002, 0.003, 0.004]})
    returns_path = tmp_path / "daily_returns.csv"
    daily_returns.to_csv(returns_path, index=False)

    metrics = calculate_performance_metrics(returns_path, daily_turnover, daily_costs)

    returns = np.array([0.10, -0.20, -0.05, 0.02], dtype=float)
    nav = np.cumprod(1.0 + returns)
    drawdown_signed = nav / np.maximum.accumulate(nav) - 1.0
    cumulative_return = float(np.prod(1.0 + returns) - 1.0)

    assert metrics["n_steps"] == 4.0
    assert metrics["final_nav"] == pytest.approx(float(nav[-1]))
    assert metrics["cumulative_return"] == pytest.approx(cumulative_return)
    assert metrics["annualized_return"] == pytest.approx((1.0 + cumulative_return) ** (252 / 4) - 1.0)
    assert metrics["annualized_volatility"] == pytest.approx(float(returns.std(ddof=0) * np.sqrt(252)))
    assert metrics["sharpe"] == pytest.approx(float(returns.mean() / returns.std(ddof=0) * np.sqrt(252)))
    downside_std = returns[returns < 0].std(ddof=0)
    assert metrics["sortino"] == pytest.approx(float(returns.mean() / downside_std * np.sqrt(252)))
    assert metrics["max_drawdown_signed"] == pytest.approx(float(drawdown_signed.min()))
    assert metrics["max_drawdown_abs"] == pytest.approx(abs(float(drawdown_signed.min())))
    assert metrics["max_drawdown"] == pytest.approx(metrics["max_drawdown_abs"])
    assert metrics["calmar"] == pytest.approx(metrics["annualized_return"] / metrics["max_drawdown_abs"])
    assert metrics["hit_ratio"] == pytest.approx(2 / 4)
    assert metrics["turnover"] == pytest.approx(0.25)
    assert metrics["average_turnover"] == pytest.approx(0.25)
    assert metrics["total_transaction_cost"] == pytest.approx(0.010)
    assert metrics["cost"] == pytest.approx(0.010)


def test_calculate_performance_metrics_tail_and_omega():
    daily_returns = pd.DataFrame(
        {
            "net_return": [-0.10, -0.05, 0.02, 0.03],
            "turnover": [0.1, 0.2, 0.3, 0.4],
            "transaction_cost": [0.001, 0.001, 0.002, 0.002],
        }
    )

    metrics = calculate_performance_metrics(daily_returns, var_alpha=0.50, cvar_alpha=0.50)

    returns = pd.Series([-0.10, -0.05, 0.02, 0.03], dtype=float)
    threshold = float(returns.quantile(0.50))
    expected_cvar = -float(returns[returns <= threshold].mean())
    gains = float(returns.clip(lower=0.0).sum())
    losses = abs(float(returns.clip(upper=0.0).sum()))

    assert metrics["var"] == pytest.approx(max(0.0, -threshold))
    assert metrics["cvar"] == pytest.approx(expected_cvar)
    assert metrics["omega"] == pytest.approx(gains / losses)
    assert metrics["turnover"] == pytest.approx(0.25)
    assert metrics["total_transaction_cost"] == pytest.approx(0.006)


def test_calculate_performance_metrics_requires_net_return():
    with pytest.raises(ValueError, match="ERR_METRICS_MISSING_NET_RETURN"):
        calculate_performance_metrics(pd.DataFrame({"return": [0.01]}))
