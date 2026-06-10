"""Tests for M3-T3: ReplayItem CQR field contract.

Verifies:
- ReplayItem has 6 new CQR fields with default None
- as_batch() includes new fields
- validate_cqr_fields() enforces gate consistency
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.buffers.replay_buffer import ReplayBuffer, ReplayItem


def _make_cqr_item(
    gate_action: int = 1,
    n_assets: int = 4,
    pre_trade_drifted: np.ndarray | None = None,
    realized_gross: float = 0.01,
) -> ReplayItem:
    weights = np.full(n_assets, 1.0 / n_assets)
    if pre_trade_drifted is None:
        pre_trade_drifted = weights.copy()
    return ReplayItem(
        state_t={},
        state_tp1={},
        decision_date_t="2024-01-01",
        execution_date_t="2024-01-02",
        next_valuation_date_t="2024-01-03",
        decision_date_next="2024-01-02",
        execution_date_next="2024-01-03",
        next_valuation_date_next="2024-01-04",
        execution_price_t="close",
        delayed_action_execution_t=False,
        candidate_weights_t=weights.copy(),
        executed_weights_t=weights.copy() if gate_action == 1 else pre_trade_drifted.copy(),
        gate_action_t=gate_action,
        rebalance_action_t=gate_action,
        rebalance_intensity_t=float(gate_action),
        estimated_turnover_t=0.0,
        realized_turnover_t=0.0,
        estimated_cost_t=0.0,
        realized_cost_t=0.0,
        reward_t=0.0,
        terminated_t=False,
        truncated_t=False,
        q_hold_t=0.5,
        q_rebalance_t=0.6,
        pre_trade_drifted_weights_t=pre_trade_drifted.copy(),
        pre_trade_drifted_weights_t_plus_1=pre_trade_drifted.copy(),
        estimated_cost_candidate_t=0.0,
        estimated_cost_hold_t=0.0,
        realized_gross_simple_return_t=realized_gross,
    )


# ---------------------------------------------------------------------------
# CQR field defaults
# ---------------------------------------------------------------------------

class TestReplayItemCQRDefaults:
    """ReplayItem CQR fields must default to None."""

    def test_cqr_fields_default_none(self) -> None:
        n_assets = 4
        weights = np.full(n_assets, 0.25)
        item = ReplayItem(
            state_t={},
            state_tp1={},
            decision_date_t="2024-01-01",
            execution_date_t="2024-01-02",
            next_valuation_date_t="2024-01-03",
            decision_date_next="2024-01-02",
            execution_date_next="2024-01-03",
            next_valuation_date_next="2024-01-04",
            execution_price_t="close",
            delayed_action_execution_t=False,
            candidate_weights_t=weights,
            executed_weights_t=weights,
            gate_action_t=1,
            rebalance_action_t=1,
            rebalance_intensity_t=1.0,
            estimated_turnover_t=0.0,
            realized_turnover_t=0.0,
            estimated_cost_t=0.0,
            realized_cost_t=0.0,
            reward_t=0.0,
            terminated_t=False,
            truncated_t=False,
            q_hold_t=0.0,
            q_rebalance_t=0.0,
        )
        assert item.pre_trade_drifted_weights_t is None
        assert item.pre_trade_drifted_weights_t_plus_1 is None
        assert item.estimated_cost_candidate_t is None
        assert item.estimated_cost_hold_t is None
        assert item.realized_gross_simple_return_t is None
        assert item.prev_weights_t is None


# ---------------------------------------------------------------------------
# as_batch with CQR fields
# ---------------------------------------------------------------------------

class TestReplayBufferAsBatchCQR:
    """as_batch() must include CQR fields when present."""

    def test_as_batch_includes_cqr_keys(self) -> None:
        buf = ReplayBuffer(capacity=10)
        buf.add(_make_cqr_item(gate_action=1))
        buf.add(_make_cqr_item(gate_action=0))
        batch = buf.as_batch()
        cqr_keys = [
            "pre_trade_drifted_weights_t",
            "pre_trade_drifted_weights_t_plus_1",
            "estimated_cost_candidate_t",
            "estimated_cost_hold_t",
            "realized_gross_simple_return_t",
        ]
        for key in cqr_keys:
            assert key in batch, f"Missing CQR key in batch: {key}"


# ---------------------------------------------------------------------------
# validate_cqr_fields
# ---------------------------------------------------------------------------

class TestValidateCQRFields:
    """validate_cqr_fields must enforce gate-action consistency."""

    def test_validate_passes_for_rebalance(self) -> None:
        buf = ReplayBuffer(capacity=10)
        item = _make_cqr_item(gate_action=1)
        buf.add(item)
        # Should not raise
        buf.validate_cqr_fields(item)

    def test_validate_passes_for_hold(self) -> None:
        buf = ReplayBuffer(capacity=10)
        item = _make_cqr_item(gate_action=0)
        buf.add(item)
        buf.validate_cqr_fields(item)

    def test_validate_fails_when_gate1_executed_mismatch(self) -> None:
        buf = ReplayBuffer(capacity=10)
        item = _make_cqr_item(gate_action=1)
        # Corrupt executed_weights to differ from candidate
        item.executed_weights_t = np.array([0.1, 0.2, 0.3, 0.4])
        buf.add(item)
        with pytest.raises(ValueError, match="ERR_CQR_GATE_CONSISTENCY"):
            buf.validate_cqr_fields(item)

    def test_validate_fails_when_missing_field(self) -> None:
        buf = ReplayBuffer(capacity=10)
        item = _make_cqr_item(gate_action=1)
        item.pre_trade_drifted_weights_t = None
        buf.add(item)
        with pytest.raises(ValueError, match="ERR_CQR_REPLAY_ITEM_MISSING_FIELD"):
            buf.validate_cqr_fields(item)
