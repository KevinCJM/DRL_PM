import numpy as np
import pandas as pd
import pytest

from src.buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.buffers.replay_buffer import ReplayBuffer, ReplayItem
from src.buffers.rollout_buffer import RolloutBuffer, RolloutItem


def test_rollout_buffer_required_fields():
    item = _rollout_item()
    buffer = RolloutBuffer(rollout_steps=2)
    stored = buffer.add(item)

    assert len(buffer) == 1
    assert stored.decision_date == pd.Timestamp("2024-01-02")
    assert stored.execution_date == pd.Timestamp("2024-01-03")
    assert stored.next_valuation_date == pd.Timestamp("2024-01-03")
    assert stored.execution_price == "next_open"
    assert stored.delayed_action_execution is False
    assert stored.state["market_image"].shape == (2, 4, 3)
    np.testing.assert_allclose(stored.candidate_weights, np.array([0.2, 0.3, 0.5]))
    np.testing.assert_allclose(stored.executed_weights, np.array([0.25, 0.25, 0.5]))
    assert stored.log_prob == pytest.approx(-0.7)
    assert stored.value == pytest.approx(0.4)
    assert stored.decision_value == pytest.approx(0.4)
    assert stored.gate_action == 1
    assert stored.rebalance_action == 1
    assert stored.rebalance_intensity == pytest.approx(1.0)
    assert stored.reward == pytest.approx(0.02)
    assert stored.terminated is False
    assert stored.truncated is False
    assert stored.advantage is None
    assert stored.return_ is None
    assert getattr(stored, "return") is None
    assert stored.auxiliary_labels == {"future_return_5d": 0.01}

    buffer.compute_gae(last_value=0.0)
    assert np.isfinite(buffer.items[0].advantage)
    assert np.isfinite(buffer.items[0].return_)
    assert getattr(buffer.items[0], "return") == pytest.approx(buffer.items[0].return_)

    batch = buffer.as_batch()
    assert batch["candidate_weights"].shape == (1, 3)
    assert batch["executed_weights"].shape == (1, 3)
    assert batch["log_prob"].shape == (1, 1)
    assert batch["advantage"].shape == (1, 1)
    assert batch["return"].shape == (1, 1)
    assert batch["decision_date"] == [pd.Timestamp("2024-01-02")]
    assert batch["execution_price"] == ["next_open"]


def test_rollout_item_accepts_vector_feature_payloads():
    item = RolloutItem(
        **_rollout_payload(
            uncertainty_features=np.array([0.1, 0.2], dtype=np.float32),
            distributional_features=np.array([0.3, 0.4], dtype=np.float32),
        )
    )

    np.testing.assert_allclose(item.uncertainty_features["uncertainty_features_vector"], np.array([0.1, 0.2]))
    np.testing.assert_allclose(item.distributional_features["distributional_features_vector"], np.array([0.3, 0.4]))


def test_rollout_buffer_accepts_return_alias_and_enforces_capacity():
    buffer = RolloutBuffer(rollout_steps=1)
    buffer.add(_rollout_payload(return_=0.12, advantage=0.03))

    assert buffer.items[0].return_ == pytest.approx(0.12)
    assert buffer.items[0].advantage == pytest.approx(0.03)
    with pytest.raises(ValueError, match="ERR_ROLLOUT_BUFFER_FULL"):
        buffer.add(_rollout_item())


def test_rollout_buffer_gae_keeps_previous_step_bootstrap_when_next_step_terminal():
    buffer = RolloutBuffer(
        rollout_steps=2,
        gamma=1.0,
        gae_lambda=1.0,
        advantage_normalization=False,
    )
    buffer.add(_rollout_payload(reward=1.0, value=0.5, terminated=False, truncated=False))
    buffer.add(
        _rollout_payload(
            decision_date=pd.Timestamp("2024-01-03"),
            execution_date=pd.Timestamp("2024-01-04"),
            next_valuation_date=pd.Timestamp("2024-01-04"),
            reward=2.0,
            value=0.25,
            terminated=True,
            truncated=False,
        )
    )

    buffer.compute_gae(last_value=99.0)

    assert buffer.items[1].advantage == pytest.approx(1.75)
    assert buffer.items[1].return_ == pytest.approx(2.0)
    assert buffer.items[0].advantage == pytest.approx(2.5)
    assert buffer.items[0].return_ == pytest.approx(3.0)


def test_replay_buffer_date_fields_and_n_step():
    item = ReplayItem(**_replay_payload())

    assert item.decision_date_t == pd.Timestamp("2024-01-02")
    assert item.execution_date_t == pd.Timestamp("2024-01-03")
    assert item.next_valuation_date_t == pd.Timestamp("2024-01-04")
    assert item.decision_date_next == pd.Timestamp("2024-01-05")
    assert item.execution_date_next == pd.Timestamp("2024-01-06")
    assert item.next_valuation_date_next == pd.Timestamp("2024-01-07")
    assert item.execution_price_t == "next_open"
    assert item.delayed_action_execution_t is False
    assert item.estimated_cost_t == pytest.approx(0.001)
    assert item.realized_cost_t == pytest.approx(0.002)
    assert item.q_hold_t == pytest.approx(0.1)
    assert item.q_rebalance_t == pytest.approx(0.3)
    assert item.q_gap_t == pytest.approx(0.2)

    buffer = ReplayBuffer(capacity=10, gamma=1.0, n_steps=3)
    buffer.add_transition(_replay_payload(reward_t=1.0, terminated_t=False, truncated_t=False))
    assert len(buffer) == 0
    buffer.add_transition(_replay_payload(step=1, reward_t=2.0, terminated_t=True, truncated_t=False))

    assert len(buffer) == 2
    first, second = buffer.items
    assert first.reward_t == pytest.approx(3.0)
    assert first.terminated_t is True
    assert first.truncated_t is False
    assert first.n_steps == 2
    assert first.discount == pytest.approx(0.0)
    assert first.state_tp1["market_image"][0, 0] == pytest.approx(2.0)
    assert first.decision_date_next == pd.Timestamp("2024-01-06")
    assert first.execution_date_next == pd.Timestamp("2024-01-07")
    assert first.next_valuation_date_next == pd.Timestamp("2024-01-08")
    assert second.reward_t == pytest.approx(2.0)
    assert second.terminated_t is True
    assert second.discount == pytest.approx(0.0)

    split_buffer = ReplayBuffer(capacity=10, gamma=1.0, n_steps=3)
    split_buffer.add_transition(_replay_payload(reward_t=1.0, terminated_t=False, truncated_t=False))
    split_buffer.add_transition(
        _replay_payload(step=1, reward_t=2.0, terminated_t=False, truncated_t=False, split_boundary_t=True)
    )
    split_first, split_second = split_buffer.items
    assert split_first.reward_t == pytest.approx(3.0)
    assert split_first.terminated_t is False
    assert split_first.truncated_t is True
    assert split_first.split_boundary_t is True
    assert split_first.discount == pytest.approx(0.0)
    assert split_second.truncated_t is True
    assert not split_buffer.pending_items

    batch = buffer.as_batch()
    assert batch["reward_t"].shape == (2, 1)
    assert batch["candidate_weights_t"].shape == (2, 2)
    assert batch["decision_date_t"] == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert batch["execution_price_t"] == ["next_open", "next_open"]


def test_prioritized_replay_formula():
    buffer = PrioritizedReplayBuffer(
        capacity=8,
        gamma=1.0,
        n_steps=1,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        beta_anneal_steps=10,
        per_priority_eps=1.0e-6,
    )
    td_errors = np.array([0.0, 1.0, 3.0], dtype=float)
    for step, td_error in enumerate(td_errors):
        buffer.add(_replay_payload(step=step), td_error=td_error)

    expected_priorities = (np.abs(td_errors) + 1.0e-6) ** 0.6
    expected_probabilities = expected_priorities / expected_priorities.sum()
    assert PrioritizedReplayBuffer.compute_priority(1.0, 0.6, 1.0e-6) == pytest.approx((1.0 + 1.0e-6) ** 0.6)
    np.testing.assert_allclose(buffer.priorities, expected_priorities)
    np.testing.assert_allclose(buffer.sampling_probabilities(), expected_probabilities)

    indices = np.array([0, 2], dtype=np.int64)
    expected_weights = (len(buffer) * expected_probabilities[indices]) ** (-0.4)
    expected_weights = expected_weights / expected_weights.max()
    np.testing.assert_allclose(buffer.importance_sampling_weights(indices, beta=0.4), expected_weights)
    assert buffer.beta_at_step(0) == pytest.approx(0.4)
    assert buffer.beta_at_step(5) == pytest.approx(0.7)
    assert buffer.beta_at_step(10) == pytest.approx(1.0)
    assert buffer.beta_at_step(99) == pytest.approx(1.0)

    sample = buffer.sample(2, rng=np.random.default_rng(7), replace=True, beta=0.4)
    sample_indices = sample["indices"]
    sample_expected_weights = (len(buffer) * expected_probabilities[sample_indices]) ** (-0.4)
    sample_expected_weights = sample_expected_weights / sample_expected_weights.max()
    np.testing.assert_allclose(sample["is_weight"], sample_expected_weights)
    assert sample["is_weight"].max() == pytest.approx(1.0)
    np.testing.assert_allclose(sample["sampling_probability"], expected_probabilities[sample_indices])
    assert sample["batch"]["is_weight"].shape == (2, 1)

    buffer.update_priorities([1], td_errors=4.0)
    expected_priorities[1] = (4.0 + 1.0e-6) ** 0.6
    np.testing.assert_allclose(buffer.priorities, expected_priorities)


def _rollout_item():
    return RolloutItem(**_rollout_payload())


def _rollout_payload(**overrides):
    payload = {
        "decision_date": pd.Timestamp("2024-01-02"),
        "execution_date": pd.Timestamp("2024-01-03"),
        "next_valuation_date": pd.Timestamp("2024-01-03"),
        "execution_price": "next_open",
        "delayed_action_execution": False,
        "state": {"market_image": np.zeros((2, 4, 3), dtype=np.float32)},
        "candidate_weights": np.array([0.2, 0.3, 0.5], dtype=float),
        "executed_weights": np.array([0.25, 0.25, 0.5], dtype=float),
        "log_prob": -0.7,
        "value": 0.4,
        "decision_value": None,
        "gate_action": 1,
        "rebalance_action": 1,
        "rebalance_intensity": 1.0,
        "reward": 0.02,
        "terminated": False,
        "truncated": False,
        "advantage": None,
        "return_value": None,
        "auxiliary_labels": {"future_return_5d": 0.01},
        "preference_vector": np.array([0.2, 0.8], dtype=float),
        "uncertainty_features": {"candidate_variance": 0.0},
        "distributional_features": {"cvar": -0.01},
    }
    payload.update(overrides)
    return payload


def _replay_payload(step=0, **overrides):
    base_date = pd.Timestamp("2024-01-02") + pd.Timedelta(days=step)
    payload = {
        "state_t": {"market_image": np.full((2, 2), step, dtype=np.float32)},
        "state_tp1": {"market_image": np.full((2, 2), step + 1, dtype=np.float32)},
        "decision_date_t": base_date,
        "execution_date_t": base_date + pd.Timedelta(days=1),
        "next_valuation_date_t": base_date + pd.Timedelta(days=2),
        "decision_date_next": base_date + pd.Timedelta(days=3),
        "execution_date_next": base_date + pd.Timedelta(days=4),
        "next_valuation_date_next": base_date + pd.Timedelta(days=5),
        "execution_price_t": "next_open",
        "delayed_action_execution_t": False,
        "candidate_weights_t": np.array([0.4, 0.6], dtype=float),
        "executed_weights_t": np.array([0.5, 0.5], dtype=float),
        "gate_action_t": 1,
        "rebalance_action_t": 1,
        "rebalance_intensity_t": 1.0,
        "estimated_turnover_t": 0.1,
        "realized_turnover_t": 0.12,
        "estimated_cost_t": 0.001,
        "realized_cost_t": 0.002,
        "reward_t": 1.0,
        "terminated_t": False,
        "truncated_t": False,
        "q_hold_t": 0.1,
        "q_rebalance_t": 0.3,
    }
    payload.update(overrides)
    return payload
