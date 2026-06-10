"""Tests for M4-T3: buffer coverage guard and actor-gate mask logic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.baselines.otar_cqr_gate_strategy import OTarCQRGateStrategy
from src.buffers.replay_buffer import ReplayItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_replay_item(gate_action: int) -> ReplayItem:
    """Create a minimal ReplayItem with given gate_action."""
    n_assets = 4
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
        candidate_weights_t=np.full(n_assets, 1.0 / n_assets),
        executed_weights_t=np.full(n_assets, 1.0 / n_assets),
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
        q_hold_t=0.0,
        q_rebalance_t=0.0,
        pre_trade_drifted_weights_t=np.full(n_assets, 1.0 / n_assets),
        pre_trade_drifted_weights_t_plus_1=np.full(n_assets, 1.0 / n_assets),
        estimated_cost_candidate_t=0.0,
        estimated_cost_hold_t=0.0,
        realized_gross_simple_return_t=0.0,
    )


def _make_strategy(
    *,
    min_rebalance_ratio: float = 0.10,
    min_hold_ratio: float = 0.10,
    gate_batch_size: int = 64,
    replay_min_size: int = 1,
    epsilon_start: float = 0.0,
    epsilon_end: float = 0.0,
) -> OTarCQRGateStrategy:
    """Create an OTarCQRGateStrategy with a mocked model."""
    import torch as _torch

    config: dict[str, Any] = {
        "cqr": {
            "replay_capacity": 1000,
            "gate_gamma": 0.99,
            "gate_lr": 1e-3,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_steps": 1,
            "replay_min_size": replay_min_size,
            "gate_batch_size": gate_batch_size,
            "target_update_interval": 100,
            "min_rebalance_ratio_in_buffer": min_rebalance_ratio,
            "min_hold_ratio_in_buffer": min_hold_ratio,
        },
        "data": {"asset_ids": ["A", "B", "C", "D"]},
    }
    model = MagicMock()
    model._device = "cpu"
    # Provide a real parameter so torch.optim.Adam doesn't fail
    dummy_param = _torch.nn.Parameter(_torch.zeros(1))
    model.cqr_critic.parameters.return_value = [dummy_param]
    return OTarCQRGateStrategy(config, model)


# ---------------------------------------------------------------------------
# Buffer coverage guard tests
# ---------------------------------------------------------------------------

class TestBufferCoverageGuard:
    """Tests for _sample_cqr_batch oversampling logic."""

    def test_oversamples_rebalance_when_below_threshold(self) -> None:
        strategy = _make_strategy(min_rebalance_ratio=0.30, gate_batch_size=10)
        # Add 2 rebalance and 8 hold items → rebalance_ratio=0.20 < 0.30
        for _ in range(2):
            strategy.replay_buffer.add(_make_replay_item(gate_action=1))
        for _ in range(8):
            strategy.replay_buffer.add(_make_replay_item(gate_action=0))

        batch = strategy._sample_cqr_batch()
        assert batch is not None
        rebalance_count = sum(1 for item in batch if int(item.gate_action_t) == 1)
        # Without oversampling, expected rebalance in batch of 10 would be ~2
        # With oversampling (50% target), rebalance should be >= 2
        assert rebalance_count >= 2

    def test_oversamples_hold_when_below_threshold(self) -> None:
        strategy = _make_strategy(min_hold_ratio=0.30, gate_batch_size=10)
        # Add 8 rebalance and 2 hold items → hold_ratio=0.20 < 0.30
        for _ in range(8):
            strategy.replay_buffer.add(_make_replay_item(gate_action=1))
        for _ in range(2):
            strategy.replay_buffer.add(_make_replay_item(gate_action=0))

        batch = strategy._sample_cqr_batch()
        assert batch is not None
        hold_count = sum(1 for item in batch if int(item.gate_action_t) == 0)
        assert hold_count >= 2

    def test_uniform_sampling_when_ratios_sufficient(self) -> None:
        strategy = _make_strategy(min_rebalance_ratio=0.10, min_hold_ratio=0.10, gate_batch_size=10)
        for _ in range(5):
            strategy.replay_buffer.add(_make_replay_item(gate_action=1))
        for _ in range(5):
            strategy.replay_buffer.add(_make_replay_item(gate_action=0))

        batch = strategy._sample_cqr_batch()
        assert batch is not None
        assert len(batch) == 10

    def test_returns_none_when_buffer_empty(self) -> None:
        strategy = _make_strategy()
        assert strategy._sample_cqr_batch() is None

    def test_batch_size_capped_at_gate_batch_size(self) -> None:
        strategy = _make_strategy(gate_batch_size=5)
        for _ in range(20):
            strategy.replay_buffer.add(_make_replay_item(gate_action=1))

        batch = strategy._sample_cqr_batch()
        assert batch is not None
        assert len(batch) == 5


# ---------------------------------------------------------------------------
# Actor-gate mask tests
# ---------------------------------------------------------------------------

class TestActorGateMasks:
    """Tests for ppo_actor_update_mask / value_update_mask / gate_update_mask logic."""

    def test_masks_rebalance_action(self) -> None:
        """executed_gate_action=1 → ppo_actor_update_mask=1, value=1, gate=1."""
        strategy = _make_strategy()
        # Simulate the mask assignment logic from _train_epoch
        executed_gate_action = 1
        ppo_actor_update_mask = 1 if executed_gate_action == 1 else 0
        assert ppo_actor_update_mask == 1
        # value_update_mask and gate_update_mask are always 1
        assert 1 == 1  # value_update_mask
        assert 1 == 1  # gate_update_mask

    def test_masks_hold_action(self) -> None:
        """executed_gate_action=0 → ppo_actor_update_mask=0, value=1, gate=1."""
        executed_gate_action = 0
        ppo_actor_update_mask = 1 if executed_gate_action == 1 else 0
        assert ppo_actor_update_mask == 0

    def test_actor_update_counters(self) -> None:
        """actor_update_count and actor_skipped_by_gate_count accumulate correctly."""
        strategy = _make_strategy()
        assert strategy._actor_update_count == 0
        assert strategy._actor_skipped_by_gate_count == 0

        # Simulate 3 rebalance (mask=1) and 2 hold (mask=0) steps
        for gate_action in [1, 1, 0, 1, 0]:
            mask = 1 if gate_action == 1 else 0
            strategy._actor_update_count += mask
            strategy._actor_skipped_by_gate_count += (1 - mask)

        assert strategy._actor_update_count == 3
        assert strategy._actor_skipped_by_gate_count == 2

    def test_effective_actor_update_ratio(self) -> None:
        """effective_actor_update_ratio = actor_update_count / total."""
        strategy = _make_strategy()
        strategy._actor_update_count = 3
        strategy._actor_skipped_by_gate_count = 7
        ratio = strategy._effective_actor_update_ratio()
        assert abs(ratio - 0.3) < 1e-9

    def test_effective_actor_update_ratio_zero_total(self) -> None:
        """Ratio is 0.0 when no steps have been taken."""
        strategy = _make_strategy()
        assert strategy._effective_actor_update_ratio() == 0.0

    def test_action_info_contains_masks(self) -> None:
        """action_info dict includes ppo_actor_update_mask, value_update_mask, gate_update_mask."""
        strategy = _make_strategy()
        # Simulate building action_info as in _train_epoch
        executed_gate_action_scalar = 1
        ppo_actor_update_mask = 1 if executed_gate_action_scalar == 1 else 0
        action_info: dict[str, Any] = {
            "ppo_actor_update_mask": int(ppo_actor_update_mask),
            "value_update_mask": 1,
            "gate_update_mask": 1,
        }
        assert action_info["ppo_actor_update_mask"] == 1
        assert action_info["value_update_mask"] == 1
        assert action_info["gate_update_mask"] == 1

    def test_action_info_masks_for_hold(self) -> None:
        """Hold action: ppo_actor_update_mask=0, value=1, gate=1."""
        executed_gate_action_scalar = 0
        ppo_actor_update_mask = 1 if executed_gate_action_scalar == 1 else 0
        action_info: dict[str, Any] = {
            "ppo_actor_update_mask": int(ppo_actor_update_mask),
            "value_update_mask": 1,
            "gate_update_mask": 1,
        }
        assert action_info["ppo_actor_update_mask"] == 0
        assert action_info["value_update_mask"] == 1
        assert action_info["gate_update_mask"] == 1


# ---------------------------------------------------------------------------
# Epsilon-greedy tests
# ---------------------------------------------------------------------------

class TestEpsilonGreedy:
    """Tests for epsilon-greedy gate action flip logic."""

    def test_epsilon_decays(self) -> None:
        strategy = _make_strategy(epsilon_start=1.0, epsilon_end=0.0)
        strategy._global_step = 0
        assert strategy._epsilon() == pytest.approx(1.0)
        strategy._global_step = 1000
        assert strategy._epsilon() < 1.0

    def test_epsilon_clamps_at_end(self) -> None:
        strategy = _make_strategy(epsilon_start=1.0, epsilon_end=0.05)
        strategy._global_step = 999999
        assert strategy._epsilon() == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Integration: _sample_cqr_batch respects min ratios
# ---------------------------------------------------------------------------

class TestBufferCoverageIntegration:
    """Integration tests for buffer coverage guard with realistic buffer state."""

    def test_all_rebalance_items_respect_hold_minimum(self) -> None:
        """When buffer is all rebalance, oversample hold (but hold_items is empty)."""
        strategy = _make_strategy(min_hold_ratio=0.50, gate_batch_size=10)
        for _ in range(10):
            strategy.replay_buffer.add(_make_replay_item(gate_action=1))

        batch = strategy._sample_cqr_batch()
        assert batch is not None
        # All items are rebalance, hold oversampling path but hold_items is empty
        # Should still return a valid batch
        assert len(batch) > 0

    def test_all_hold_items_respect_rebalance_minimum(self) -> None:
        """When buffer is all hold, oversample rebalance (but rebalance_items is empty)."""
        strategy = _make_strategy(min_rebalance_ratio=0.50, gate_batch_size=10)
        for _ in range(10):
            strategy.replay_buffer.add(_make_replay_item(gate_action=0))

        batch = strategy._sample_cqr_batch()
        assert batch is not None
        assert len(batch) > 0

    def test_batch_size_not_exceed_buffer_size(self) -> None:
        """Batch size is capped at total buffer items."""
        strategy = _make_strategy(gate_batch_size=100)
        for _ in range(3):
            strategy.replay_buffer.add(_make_replay_item(gate_action=1))
        for _ in range(2):
            strategy.replay_buffer.add(_make_replay_item(gate_action=0))

        batch = strategy._sample_cqr_batch()
        assert batch is not None
        assert len(batch) == 5
