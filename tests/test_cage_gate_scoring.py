import numpy as np

from src.baselines.cage_common import (
    enforce_activity_turnover_floor,
    estimate_turnover,
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


def test_activity_turnover_floor_expands_candidate_after_gate_passes():
    config = {
        "execution_activity": {
            "activity_gate_enforced": True,
            "min_non_initial_turnover_per_opportunity": 0.002,
            "min_model_rebalance_hit_rate": 0.05,
        }
    }
    current = np.array([0.5, 0.5])
    candidate = np.array([0.51, 0.49])

    adjusted, info = enforce_activity_turnover_floor(
        candidate,
        current,
        np.array([True, True]),
        config,
        rebalance_intensity=1.0,
        first_trade=False,
    )

    assert info["activity_turnover_floor_applied"] is True
    assert estimate_turnover(adjusted, current) >= 0.04 - 1.0e-8
    np.testing.assert_allclose(adjusted.sum(), 1.0)


def test_activity_turnover_floor_does_not_override_hold_or_first_trade():
    config = {
        "execution_activity": {
            "activity_gate_enforced": True,
            "min_non_initial_turnover_per_opportunity": 0.002,
            "min_model_rebalance_hit_rate": 0.05,
        }
    }
    current = np.array([0.5, 0.5])
    candidate = np.array([0.51, 0.49])

    held, hold_info = enforce_activity_turnover_floor(
        candidate,
        current,
        np.array([True, True]),
        config,
        rebalance_intensity=0.0,
        first_trade=False,
    )
    first, first_info = enforce_activity_turnover_floor(
        candidate,
        current,
        np.array([True, True]),
        config,
        rebalance_intensity=1.0,
        first_trade=True,
    )

    np.testing.assert_allclose(held, candidate)
    np.testing.assert_allclose(first, candidate)
    assert hold_info["activity_turnover_floor_applied"] is False
    assert first_info["activity_turnover_floor_applied"] is False
