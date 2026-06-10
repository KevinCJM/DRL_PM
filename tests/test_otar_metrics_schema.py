"""Tests for M6-T1: RELATED_WORK_ACTION_INFO_KEYS expansion.

Verifies:
- RELATED_WORK_ACTION_INFO_KEYS contains required CQR keys
- All required gate diagnostics keys are present
"""

from __future__ import annotations

import pytest

from src.envs.portfolio_rebalance_env import RELATED_WORK_ACTION_INFO_KEYS


# ---------------------------------------------------------------------------
# Required CQR keys
# ---------------------------------------------------------------------------

class TestRELATEDWorkActionInfoKeys:
    """RELATED_WORK_ACTION_INFO_KEYS must contain all required CQR gate diagnostics keys."""

    REQUIRED_CQR_KEYS = [
        "raw_gate_action",
        "executed_gate_action",
        "predicted_5pct_quantile_executed",
        "predicted_5pct_quantile_candidate",
        "predicted_5pct_quantile_hold",
        "pred_delta_utility",
        "pred_candidate_utility",
        "pred_hold_utility",
        "pred_candidate_mean_return",
        "pred_hold_mean_return",
        "pred_candidate_lower_tail_loss",
        "pred_hold_lower_tail_loss",
        "candidate_estimated_cost",
        "hold_estimated_cost",
        "gate_margin",
        "quantile_spread_candidate",
        "quantile_spread_hold",
    ]

    REQUIRED_POLICY_KEYS = [
        "log_prob",
        "entropy",
        "alpha_min",
        "alpha_max",
        "alpha_mean",
    ]

    REQUIRED_PROJECTION_KEYS = [
        "projection_distance",
        "projection_violation_count",
    ]

    REQUIRED_MASK_KEYS = [
        "value_update_mask",
        "gate_update_mask",
    ]

    REQUIRED_COST_KEYS = [
        "estimated_turnover",
        "estimated_cost",
    ]

    def test_cqr_decision_time_keys_present(self) -> None:
        keys_set = set(RELATED_WORK_ACTION_INFO_KEYS)
        for key in self.REQUIRED_CQR_KEYS:
            assert key in keys_set, f"Missing CQR key: {key}"

    def test_policy_diagnostic_keys_present(self) -> None:
        keys_set = set(RELATED_WORK_ACTION_INFO_KEYS)
        for key in self.REQUIRED_POLICY_KEYS:
            assert key in keys_set, f"Missing policy key: {key}"

    def test_projection_diagnostic_keys_present(self) -> None:
        keys_set = set(RELATED_WORK_ACTION_INFO_KEYS)
        for key in self.REQUIRED_PROJECTION_KEYS:
            assert key in keys_set, f"Missing projection key: {key}"

    def test_mask_keys_present(self) -> None:
        keys_set = set(RELATED_WORK_ACTION_INFO_KEYS)
        for key in self.REQUIRED_MASK_KEYS:
            assert key in keys_set, f"Missing mask key: {key}"

    def test_cost_keys_present(self) -> None:
        keys_set = set(RELATED_WORK_ACTION_INFO_KEYS)
        for key in self.REQUIRED_COST_KEYS:
            assert key in keys_set, f"Missing cost key: {key}"

    def test_ppo_actor_update_mask_present(self) -> None:
        assert "ppo_actor_update_mask" in set(RELATED_WORK_ACTION_INFO_KEYS)
