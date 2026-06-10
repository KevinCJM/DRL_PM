"""Tests for M7-T1: TailCalibrator contract.

Verifies:
- calibrate() returns required keys
- calibration_status is valid
- boundary validation works
- forbidden calibration labels are rejected
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.tail_calibration import FORBIDDEN_CALIBRATION_LABELS, TailCalibrator


def _make_diagnostics_df(
    n_episodes: int = 2,
    steps_per_episode: int = 100,
    gate_action_ratio: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for ep in range(n_episodes):
        for t in range(steps_per_episode):
            gate = 1 if np.random.random() < gate_action_ratio else 0
            rows.append({
                "episode_id": f"ep_{ep}",
                "timestep": t,
                "done_t": t == steps_per_episode - 1,
                "split": "validation" if ep == 0 else "test",
                "executed_gate_action": gate,
                "predicted_5pct_quantile_executed": np.random.normal(-0.05, 0.01),
                "realized_gross_simple_return_t": np.random.normal(0.01, 0.02),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# calibrate() contract
# ---------------------------------------------------------------------------

class TestTailCalibratorContract:
    """calibrate() must return dict with required keys."""

    def test_calibrate_returns_required_keys(self) -> None:
        calibrator = TailCalibrator({"min_executed_action_count_for_calibration": {"rebalance": 5, "hold": 5}})
        df = _make_diagnostics_df(n_episodes=2, steps_per_episode=50)
        result = calibrator.calibrate(df)
        required_keys = [
            "calibration_status", "rebalance_count", "hold_count",
            "tail_coverage_error", "realized_below_quantile_frequency", "quantile_pinball_loss",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_calibration_status_valid(self) -> None:
        calibrator = TailCalibrator({"min_executed_action_count_for_calibration": {"rebalance": 5, "hold": 5}})
        df = _make_diagnostics_df(n_episodes=2, steps_per_episode=50)
        result = calibrator.calibrate(df)
        valid_statuses = {"passed", "failed", "low_action_count", "invalid"}
        assert result["calibration_status"] in valid_statuses, (
            f"Invalid status: {result['calibration_status']}"
        )

    def test_low_action_count_status(self) -> None:
        calibrator = TailCalibrator({"min_executed_action_count_for_calibration": {"rebalance": 1000, "hold": 1000}})
        df = _make_diagnostics_df(n_episodes=1, steps_per_episode=10)
        result = calibrator.calibrate(df)
        assert result["calibration_status"] in ("low_action_count", "invalid")


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------

class TestBoundaryValidation:
    """Boundary fields must be valid (timestep sequential, exactly one done_t per episode)."""

    def test_valid_boundary_passes(self) -> None:
        calibrator = TailCalibrator({"min_executed_action_count_for_calibration": {"rebalance": 5, "hold": 5}})
        df = _make_diagnostics_df(n_episodes=1, steps_per_episode=50)
        result = calibrator.calibrate(df)
        assert result["calibration_status"] != "invalid" or "boundary" not in str(result.get("reason", ""))


# ---------------------------------------------------------------------------
# Forbidden labels
# ---------------------------------------------------------------------------

class TestForbiddenCalibrationLabels:
    """calibrate() must reject forbidden calibration labels."""

    def test_forbidden_label_rejected(self) -> None:
        calibrator = TailCalibrator({"min_executed_action_count_for_calibration": {"rebalance": 5, "hold": 5}})
        df = _make_diagnostics_df(n_episodes=1, steps_per_episode=50)
        for label in FORBIDDEN_CALIBRATION_LABELS:
            result = calibrator.calibrate(df, calibration_label=label)
            assert result["calibration_status"] == "invalid", (
                f"Label '{label}' should be rejected but got status '{result['calibration_status']}'"
            )
