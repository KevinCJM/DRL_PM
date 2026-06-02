from copy import deepcopy

import pandas as pd

from src.config import DEFAULT_CONFIG
from src.experiments.pipeline import objective_metric
from src.experiments.run_experiment import _activity_trial_failure_reason


def test_high_activity_trial_not_selected_as_best():
    config = _active_config()

    reason = _activity_trial_failure_reason(_result(hit_rate=1.0, turnover=0.010), config)

    assert reason == "failed_high_trade_activity"


def test_objective_penalizes_hit_rate_overuse():
    config = _active_config()
    active = _result(hit_rate=0.30, turnover=0.010)
    overactive = _result(hit_rate=1.0, turnover=0.010)

    active_value = objective_metric(active, "validation_return_risk_cost_constrained", config=config)
    overactive_value = objective_metric(overactive, "validation_return_risk_cost_constrained", config=config)

    assert active_value > overactive_value


def _active_config():
    config = deepcopy(DEFAULT_CONFIG)
    config["execution_activity"].update(
        {
            "protocol": "daily_gate_with_cost_constraint",
            "scheduler_blocks_model_actions": False,
            "activity_gate_enforced": True,
        }
    )
    config["hpo"]["activity_constraints"].update(
        {
            "enabled": True,
            "scope_baseline_families": ["platform_native_rl"],
            "scope_activity_protocols": ["daily_gate_with_cost_constraint"],
            "min_model_rebalance_hit_rate": 0.05,
            "max_model_rebalance_hit_rate": 0.6,
            "min_non_initial_turnover_per_opportunity": 0.002,
            "max_average_turnover": 0.030,
        }
    )
    return config


def _result(*, hit_rate: float, turnover: float):
    return {
        "daily_returns": pd.DataFrame(
            [
                {"net_return": 0.01, "nav": 1.01},
                {"net_return": -0.002, "nav": 1.00798},
                {"net_return": 0.006, "nav": 1.01402788},
            ]
        ),
        "metrics": {
            "model_rebalance_hit_rate": hit_rate,
            "non_initial_turnover_per_opportunity": turnover,
            "average_turnover": 0.01,
            "total_transaction_cost": 0.001,
        },
        "baseline_daily_diagnostics": pd.DataFrame([{"baseline_family": "platform_native_rl"}]),
    }
