import numpy as np

from src.baselines.cage_common import (
    choose_rho,
    compute_expected_alpha_horizon,
    score_rho_normalized,
)


def test_normalized_gate_selects_positive_rho_when_alpha_large():
    rho, scores, components = score_rho_normalized(
        rho_values=[0.0, 0.25, 0.5, 1.0],
        expected_alpha=0.006,
        estimated_turnover=0.04,
        estimated_cost=0.0004,
        cvar_loss_5=0.0,
        drawdown=0.0,
        scale_config={
            "alpha_scale": 0.001,
            "turnover_budget_per_trade": 0.05,
            "cost_budget_per_trade": 0.001,
            "hold_opportunity_penalty": -0.20,
        },
    )

    assert rho > 0.0
    assert scores[str(rho).rstrip("0").rstrip(".")] > scores["0"]
    assert components["0"]["hold_opportunity_penalty"] < 0.0


def test_legacy_gate_can_reproduce_hold_bias():
    rho, scores = choose_rho(
        rho_values=[0.0, 0.25, 0.5, 1.0],
        expected_return=0.001,
        estimated_turnover=0.10,
        estimated_cost=0.001,
        cvar_loss_5=0.0,
        drawdown=0.0,
        lambda_turnover=2.0,
        lambda_cost=10.0,
        lambda_cvar=0.0,
        lambda_drawdown=0.0,
        cvar_loss_budget=0.02,
        drawdown_budget=0.10,
    )

    assert rho == 0.0
    assert scores["0"] == 0.0


def test_expected_alpha_horizon_scales_by_activity_protocol():
    candidate = np.array([0.2, 0.8])
    current = np.array([0.5, 0.5])
    mu = np.array([0.0, 0.001])

    daily = compute_expected_alpha_horizon(
        activity_protocol="daily_gate_with_cost_constraint",
        candidate_weights=candidate,
        current_weights=current,
        mu_1d_decision_visible=mu,
    )
    weekly = compute_expected_alpha_horizon(
        activity_protocol="weekly_gate",
        candidate_weights=candidate,
        current_weights=current,
        mu_1d_decision_visible=mu,
    )

    assert weekly == daily * 5.0
