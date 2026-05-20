import sqlite3

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.agents.dqn_agent import DQNAgent, DQNAgentConfig
from src.agents.distributional_agent import DistributionalAgent
from src.agents.hybrid_agent import HybridAgent, HybridAgentConfig
from src.agents.partial_rebalance_agent import PartialRebalanceAgent
from src.agents.ppo_agent import PPOAgent, PPOAgentConfig, _rollout_features
from src.agents.preference_agent import PreferenceAgent
from src.agents.uncertainty_agent import UncertaintyAgent
from src.buffers.rollout_buffer import RolloutBuffer
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.dqn_gate import DQNGate
from src.models.ppo_actor import PPOActor
from src.models.ppo_critic import PPOCritic


def test_ppo_update_contract():
    torch.manual_seed(7)
    agent = _agent(rollout_steps=2, minibatch_size=2, update_epochs=1)
    env = _ToyEnv()

    rollout = agent.collect_rollout(env)
    assert len(rollout) == 2
    advantages = np.asarray([item.advantage for item in rollout.items], dtype=float)
    assert advantages.mean() == pytest.approx(0.0, abs=1.0e-6)
    assert rollout.items[0].execution_price == "next_open"
    assert rollout.items[0].delayed_action_execution is False

    buffer = RolloutBuffer(rollout_steps=3, advantage_normalization=True)
    buffer.add(_rollout_payload(step=0, gate_action=0, rebalance_intensity=1.0, advantage=1.0, return_value=0.4))
    buffer.add(_rollout_payload(step=1, gate_action=1, rebalance_intensity=0.5, advantage=2.0, return_value=0.5))
    buffer.add(_rollout_payload(step=2, gate_action=None, rebalance_intensity=1.0, advantage=3.0, return_value=0.6))
    batch = buffer.as_batch(device=agent.device)

    weights = agent.actor_loss_weights(batch)
    assert torch.allclose(weights, torch.tensor([[0.0], [0.5], [1.0]]))

    losses = agent.compute_losses(batch)
    for key in ("actor_loss", "value_loss", "entropy", "clip_fraction", "actor_loss_weight_mean"):
        assert key in losses
        assert torch.isfinite(losses[key])
    assert losses["actor_loss_weight_mean"] == pytest.approx(torch.tensor(0.5))

    stats = agent.update(buffer)
    for key in ("actor_loss", "value_loss", "entropy", "clip_fraction", "grad_norm"):
        assert key in stats
        assert np.isfinite(stats[key])


def test_hybrid_agent_respects_train_and_validation_step_budgets():
    ppo = _RecordingPPOAgent()
    agent = HybridAgent(
        ppo,
        config={
            "training": {"epochs": 1, "max_train_steps": 7, "max_validation_steps": 3},
            "evaluation": {"validation_episodes": 1},
        },
    )
    env = _CountingEvalEnv(max_steps=10)

    result = agent.train(env, validation_env=env)

    assert result["status"] == "completed"
    assert ppo.rollout_max_steps == 7
    assert env.max_observed_steps == 3


def test_ppo_rollout_gate_hold_sends_rebalance_zero():
    torch.manual_seed(13)
    agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    env = _RecordingGateEnv()

    def hold_action(observation, deterministic=False):
        return {
            "candidate_weights": np.array([0.8, 0.2], dtype=np.float32),
            "log_prob": -0.1,
            "value": 0.0,
            "gate_action": 0,
            "rebalance_intensity": 0.0,
            "q_hold": 1.0,
            "q_rebalance": 0.0,
            "q_gap": -1.0,
        }

    agent.select_action = hold_action
    rollout = agent.collect_rollout(env)

    assert env.actions[0]["rebalance"] == 0
    assert env.actions[0]["rebalance_intensity"] == 0.0
    np.testing.assert_allclose(env.actions[0]["weights"], np.array([0.5, 0.5], dtype=np.float32))
    assert rollout.items[0].gate_action == 0
    assert rollout.items[0].rebalance_action == 0
    assert rollout.items[0].rebalance_intensity == 0.0
    assert rollout.items[0].uncertainty_features["q_gap"] == pytest.approx(-1.0)


def test_rollout_features_accepts_uncertainty_vector_payloads():
    features = _rollout_features(
        {
            "uncertainty_features": np.array([0.1, 0.2], dtype=np.float32),
            "estimated_turnover": 0.3,
        },
        {"uncertainty_features": torch.tensor([0.4, 0.5]), "realized_cost": 0.01},
    )

    np.testing.assert_allclose(features["policy_uncertainty_features"], np.array([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(features["env_uncertainty_features"], np.array([0.4, 0.5], dtype=np.float32))
    assert features["estimated_turnover"] == pytest.approx(0.3)
    assert features["realized_cost"] == pytest.approx(0.01)


def test_ppo_select_action_uses_policy_model_forward_and_partial_intensity():
    agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    policy_model = _ForwardPolicyModel()
    agent.policy_model = policy_model

    action_info = agent.select_action(_state(0), deterministic=True)

    assert policy_model.calls == 2
    assert action_info["gate_action"] == 1
    assert action_info["rebalance_intensity"] == pytest.approx(0.42)
    assert action_info["q_gap"] == pytest.approx(0.1)


def test_ppo_policy_model_loss_backpropagates_through_policy_forward():
    agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    policy_model = _ForwardPolicyModel()
    agent.policy_model = policy_model
    buffer = RolloutBuffer(rollout_steps=1, advantage_normalization=False)
    buffer.add(_rollout_payload(step=0, advantage=1.0, return_value=1.0, uncertainty_features={"estimated_turnover": 0.1, "estimated_cost": 0.002}))
    batch = buffer.as_batch(device=agent.device)

    losses = agent.compute_losses(batch)
    losses["loss"].backward()

    assert policy_model.calls == 1
    assert policy_model.weight.grad is not None
    assert torch.isfinite(policy_model.weight.grad)


def test_ppo_policy_model_gate_uses_real_estimated_inputs_in_training_selection():
    agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    agent.raw_config = {
        "cost_model": {
            "mode": "empirical_default",
            "proportional_cost": 0.01,
            "slippage": 0.0,
            "market_impact_enabled": False,
        }
    }
    policy_model = _CostAwarePolicyModel()
    agent.policy_model = policy_model
    state = _state(0)
    state.update(
        {
            "adv20_at_decision": np.array([1000.0, 1000.0], dtype=np.float32),
            "volatility_20d_at_decision": np.array([0.1, 0.1], dtype=np.float32),
            "amount_at_decision": np.array([100.0, 100.0], dtype=np.float32),
            "turnover_rate_at_decision": np.array([0.1, 0.1], dtype=np.float32),
            "portfolio_value": 100.0,
        }
    )

    action_info = agent.select_action(state, deterministic=False)

    assert policy_model.calls == 2
    assert policy_model.seen_turnover[0] == pytest.approx(0.0)
    assert policy_model.seen_cost[0] == pytest.approx(0.0)
    assert policy_model.seen_turnover[1] > 0.0
    assert policy_model.seen_cost[1] > 0.0
    assert action_info["q_rebalance"] > 0.0


def test_ppo_policy_model_gate_action_can_override_binary_q_gap():
    agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    agent.policy_model = _AlwaysRebalancePolicyModel()

    action_info = agent.select_action(_state(0), deterministic=True)

    assert action_info["q_gap"] < 0.0
    assert action_info["gate_action"] == 1
    assert action_info["rebalance_intensity"] == pytest.approx(0.37)


def test_hybrid_dqn_beta_selector_uses_raw_intensity_after_model_hold():
    agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    agent.policy_model = _HybridHoldRawIntensityPolicyModel()

    def rebalance_selector(q_values):
        return torch.ones(q_values.shape[0], dtype=torch.long, device=q_values.device)

    action_info = agent.select_action(_state(0), deterministic=False, gate_action_selector=rebalance_selector)

    assert action_info["gate_action"] == 1
    assert action_info["rebalance_intensity"] == pytest.approx(0.42)
    assert action_info["raw_rebalance_intensity"] == pytest.approx(0.42)


def test_hybrid_training_uses_dqn_selector_and_warmup_steps():
    torch.manual_seed(23)
    ppo_agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    dqn_agent = DQNAgent(
        DQNGate(latent_dim=4, n_assets=2, hidden_dims=(8,), dropout=0.0),
        DQNGate(latent_dim=4, n_assets=2, hidden_dims=(8,), dropout=0.0),
        config=DQNAgentConfig(batch_size=1, warmup_steps=5, n_steps=1, target_update_interval=1),
    )
    ppo_agent.gate_network = dqn_agent.online_network
    calls = []

    def select_action(q_values, step, rng=None):
        calls.append({"shape": tuple(q_values.shape), "step": step})
        return torch.zeros(q_values.shape[0], dtype=torch.long, device=q_values.device)

    dqn_agent.select_action = select_action
    agent = HybridAgent(
        ppo_agent,
        dqn_agent=dqn_agent,
        config=HybridAgentConfig(epochs=1, validation_episodes=1),
    )
    env = _RecordingGateEnv()

    result = agent.train(env, epochs=1)

    assert result["status"] == "completed"
    assert calls == [{"shape": (1, 2), "step": 0}]
    assert env.actions[0]["rebalance"] == 0
    assert agent.history[0]["dqn"]["status"] == "warmup"
    assert agent.history[0]["dqn"]["required_replay"] == 5


def test_realized_turnover_and_cost_sync_to_replay():
    torch.manual_seed(29)
    ppo_agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    dqn_agent = DQNAgent(
        DQNGate(latent_dim=4, n_assets=2, hidden_dims=(8,), dropout=0.0),
        DQNGate(latent_dim=4, n_assets=2, hidden_dims=(8,), dropout=0.0),
        config=DQNAgentConfig(batch_size=1, warmup_steps=5, n_steps=1, target_update_interval=1),
    )
    agent = HybridAgent(
        ppo_agent,
        dqn_agent=dqn_agent,
        config=HybridAgentConfig(epochs=1, validation_episodes=1),
    )

    agent.train(_RealizedInfoEnv(), epochs=1)

    replay_item = dqn_agent.replay_buffer.items[-1]
    assert replay_item.realized_turnover_t == pytest.approx(0.37)
    assert replay_item.realized_cost_t == pytest.approx(0.004)


def test_multi_action_gate_index_syncs_to_dqn_replay():
    ppo_agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    dqn_agent = DQNAgent(
        DQNGate(latent_dim=4, n_assets=2, output_dim=5, hidden_dims=(8,), dropout=0.0),
        DQNGate(latent_dim=4, n_assets=2, output_dim=5, hidden_dims=(8,), dropout=0.0),
        config=DQNAgentConfig(batch_size=1, warmup_steps=0, n_steps=1, target_update_interval=1),
    )
    agent = HybridAgent(ppo_agent, dqn_agent=dqn_agent, config=HybridAgentConfig(epochs=1))
    buffer = RolloutBuffer(rollout_steps=1, advantage_normalization=False)
    buffer.add(
        _rollout_payload(
            step=0,
            gate_action=1,
            rebalance_intensity=0.75,
            uncertainty_features={"gate_action_index": 3, "estimated_turnover": 0.2, "estimated_cost": 0.01},
        )
    )
    buffer.last_observation = _state(1)

    agent._sync_dqn_replay(buffer)

    replay_item = dqn_agent.replay_buffer.items[-1]
    assert replay_item.gate_action_t == 3


def test_double_dqn_target():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    next_q_online = torch.tensor([[0.1, 0.5], [0.9, 0.2], [0.1, 0.2]])
    next_q_target = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    terminated = torch.tensor([[False], [True], [False]])
    truncated = torch.tensor([[False], [False], [True]])

    target = DQNAgent.compute_double_dqn_target(
        rewards,
        next_q_online,
        next_q_target,
        terminated,
        truncated,
        gamma=0.5,
        n_steps=2,
    )
    assert torch.allclose(target, torch.tensor([[6.0], [2.0], [3.0]]))

    discount_target = DQNAgent.compute_double_dqn_target(
        rewards,
        next_q_online,
        next_q_target,
        terminated,
        truncated,
        discount=torch.tensor([[0.25], [0.25], [0.25]]),
    )
    assert torch.allclose(discount_target, target)

    vanilla_target = DQNAgent.compute_dqn_target(
        rewards,
        next_q_target,
        terminated,
        truncated,
        gamma=0.5,
        n_steps=2,
    )
    assert torch.allclose(vanilla_target, torch.tensor([[6.0], [2.0], [3.0]]))

    source = nn.Linear(1, 1, bias=False)
    hard_target = nn.Linear(1, 1, bias=False)
    source.weight.data.fill_(2.0)
    hard_target.weight.data.zero_()
    DQNAgent.hard_update_target_network(source, hard_target)
    assert torch.allclose(hard_target.weight, source.weight)

    soft_target = nn.Linear(1, 1, bias=False)
    source.weight.data.fill_(1.0)
    soft_target.weight.data.zero_()
    DQNAgent.soft_update_target_network(source, soft_target, tau=0.25)
    assert torch.allclose(soft_target.weight, torch.full_like(soft_target.weight, 0.25))

    agent = DQNAgent(
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        config=DQNAgentConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=20),
    )
    assert agent.epsilon_at_step(0) == pytest.approx(1.0)
    assert agent.epsilon_at_step(10) == pytest.approx(0.525)
    assert agent.epsilon_at_step(20) == pytest.approx(0.05)
    assert agent.epsilon_at_step(99) == pytest.approx(0.05)

    recorder = _PriorityRecorder()
    agent.replay_buffer = recorder
    agent.update_replay_priorities({"indices": np.array([2, 4])}, torch.tensor([[-1.5], [0.5]]))
    np.testing.assert_array_equal(recorder.indices, np.array([2, 4]))
    np.testing.assert_allclose(recorder.td_errors, np.array([1.5, 0.5]))


def test_dqn_detached_encoder_runs_without_autograd():
    class GradModeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, 1))
            self.grad_modes: list[bool] = []

        def forward(self, x):
            self.grad_modes.append(torch.is_grad_enabled())
            return x.view(x.shape[0], -1) @ self.weight

    class TinyQ(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, 2))

        def forward(self, latent, candidate_weights, current_weights, estimated_turnover, estimated_cost):
            return latent @ self.weight

    encoder = GradModeEncoder()
    agent = DQNAgent(
        TinyQ(),
        TinyQ(),
        encoder=encoder,
        target_encoder=GradModeEncoder(),
        config=DQNAgentConfig(detach_encoder=True),
    )
    batch = {
        "state_t": [{"market_image": np.ones((1, 1, 1), dtype=np.float32)}],
        "candidate_weights_t": np.array([[1.0]], dtype=np.float32),
        "executed_weights_t": np.array([[1.0]], dtype=np.float32),
        "estimated_turnover_t": np.array([0.0], dtype=np.float32),
        "estimated_cost_t": np.array([0.0], dtype=np.float32),
    }

    q_values = agent._q_values(batch, next_state=False, target=False)

    assert encoder.grad_modes == [False]
    assert q_values.requires_grad


def test_hybrid_agent_single_update_smoke():
    torch.manual_seed(11)
    ppo_agent = _agent(rollout_steps=2, minibatch_size=2, update_epochs=1)
    dqn_agent = DQNAgent(
        DQNGate(latent_dim=4, n_assets=2, hidden_dims=(8,), dropout=0.0),
        DQNGate(latent_dim=4, n_assets=2, hidden_dims=(8,), dropout=0.0),
        config=DQNAgentConfig(batch_size=1, warmup_steps=0, n_steps=1, target_update_interval=1),
    )
    auxiliary_heads = AuxiliaryHeads(
        latent_dim=4,
        n_assets=2,
        n_features=1,
        window_size=2,
        config={"tasks": ["return"], "future_return_horizons": [5], "hidden_dims": [8]},
    )
    checkpoint_payloads = []
    agent = HybridAgent(
        ppo_agent,
        dqn_agent=dqn_agent,
        auxiliary_heads=auxiliary_heads,
        config=HybridAgentConfig(epochs=1, validation_episodes=1),
        checkpoint_callback=checkpoint_payloads.append,
    )

    assert ppo_agent.config.rollout_steps == 2
    assert ppo_agent.config.minibatch_size == 2
    assert ppo_agent.config.update_epochs == 1
    assert dqn_agent.config.batch_size == 1
    assert agent.config.epochs == 1
    assert agent.config.validation_episodes == 1

    result = agent.train(_ToyEnv(truncate_at=3), validation_env=_ToyEnv(), epochs=1)

    assert result["status"] == "completed"
    assert agent.status == "completed"
    assert agent.failure_state is None
    assert len(agent.history) == 1
    assert len(dqn_agent.replay_buffer) >= 1
    assert dqn_agent.config.detach_encoder is True
    last_replay_item = dqn_agent.replay_buffer.items[-1]
    assert last_replay_item.split_boundary_t is True
    assert last_replay_item.truncated_t is True
    assert last_replay_item.discount == pytest.approx(0.0)
    assert not dqn_agent.replay_buffer.pending_items
    assert last_replay_item.state_tp1["market_image"][0, 0, 0] == pytest.approx(2.0)
    record = agent.history[0]
    assert np.isfinite(record["ppo"]["actor_loss"])
    assert np.isfinite(record["dqn"]["loss"])
    assert np.isfinite(record["auxiliary"]["total"])
    assert "validation_sharpe" in record["validation"]
    assert agent.best_checkpoint_payload is not None
    assert checkpoint_payloads

    failing_agent = HybridAgent(_agent(rollout_steps=1, minibatch_size=1, update_epochs=1))
    with pytest.raises(RuntimeError, match="boom"):
        failing_agent.train(_FailingEnv())
    assert failing_agent.status == "failed"
    assert failing_agent.failure_state["error_type"] == "RuntimeError"


def test_non_finite_training_guard(tmp_path):
    torch.manual_seed(19)
    ppo_agent = _agent(rollout_steps=1, minibatch_size=1, update_epochs=1)
    buffer = RolloutBuffer(rollout_steps=1, advantage_normalization=False)
    buffer.add(_rollout_payload(step=0))
    parameter = next(ppo_agent.encoder.parameters())

    def non_finite_losses(batch):
        loss = parameter.sum() * torch.tensor(float("nan"), device=ppo_agent.device)
        return {"loss": loss, "actor_loss": loss.detach()}

    ppo_agent.compute_losses = non_finite_losses
    with pytest.raises(ValueError, match="ERR_TRAINING_NON_FINITE_LOSS"):
        ppo_agent.update(buffer)

    dqn_agent = DQNAgent(
        nn.Linear(1, 1),
        nn.Linear(1, 1),
        config=DQNAgentConfig(batch_size=1, warmup_steps=0, target_update_interval=1),
    )
    dqn_parameter = next(dqn_agent.online_network.parameters())

    def non_finite_q_values(batch, next_state, target):
        if next_state:
            return torch.zeros(1, 2, dtype=torch.float32)
        return (dqn_parameter.sum() * torch.tensor(float("nan"))).expand(1, 2)

    dqn_agent._q_values = non_finite_q_values
    dqn_batch = {
        "state_t": [{"latent": np.zeros(1, dtype=np.float32)}],
        "state_tp1": [{"latent": np.zeros(1, dtype=np.float32)}],
        "candidate_weights_t": np.array([[0.5, 0.5]], dtype=np.float32),
        "executed_weights_t": np.array([[0.5, 0.5]], dtype=np.float32),
        "gate_action_t": np.array([0], dtype=np.int64),
        "estimated_turnover_t": np.array([0.0], dtype=np.float32),
        "estimated_cost_t": np.array([0.0], dtype=np.float32),
        "reward_t": np.array([0.0], dtype=np.float32),
        "terminated_t": np.array([True]),
        "truncated_t": np.array([False]),
    }
    with pytest.raises(ValueError, match="ERR_TRAINING_NON_FINITE_LOSS"):
        dqn_agent.update(dqn_batch)

    registry_path = tmp_path / "run_registry.sqlite"
    failing_agent = HybridAgent(
        _agent(rollout_steps=1, minibatch_size=1, update_epochs=1),
        config={
            "training": {"epochs": 1},
            "registry": {"enabled": True, "path": str(registry_path)},
            "output": {"run_name": "failed_run"},
        },
    )
    with pytest.raises(RuntimeError, match="boom"):
        failing_agent.train(_FailingEnv())

    with sqlite3.connect(registry_path) as connection:
        row = connection.execute(
            "SELECT status, fail_reason, failure_state_json FROM runs WHERE run_id = ?",
            ("failed_run",),
        ).fetchone()
    assert row is not None
    assert row[0] == "failed"
    assert "boom" in row[1]
    assert "RuntimeError" in row[2]


def test_specialized_agents_share_execution_contract():
    specialized_cases = (
        (PreferenceAgent, {"omega": [0.2, 0.2, 0.2, 0.2, 0.2]}, "preference"),
        (UncertaintyAgent, {"method": "dropout"}, "uncertainty"),
        (DistributionalAgent, {"cvar_alpha": 0.05}, "distributional_cvar"),
        (PartialRebalanceAgent, {"mode": "hybrid_dqn_beta"}, "partial_rebalance"),
    )

    for agent_cls, module_config, module_name in specialized_cases:
        torch.manual_seed(17)
        hybrid_agent = HybridAgent(
            _agent(rollout_steps=1, minibatch_size=1, update_epochs=1),
            config=HybridAgentConfig(epochs=1, validation_episodes=1),
        )
        agent = agent_cls(hybrid_agent=hybrid_agent, module_config=module_config)

        assert agent.hybrid_agent is hybrid_agent
        assert agent.uses_shared_execution_contract is True
        assert agent.shared_execution_contract()["executor"] == "HybridAgent"
        assert agent.shared_execution_contract()["module"] == module_name

        result = agent.train(_ToyEnv(truncate_at=1), validation_env=_ToyEnv(truncate_at=1), epochs=1)

        assert result["status"] == "completed"
        assert agent.status == hybrid_agent.status == "completed"
        assert agent.history is hybrid_agent.history
        assert len(agent.history) == 1

    assert PreferenceAgent(
        hybrid_agent=HybridAgent(_agent(rollout_steps=1, minibatch_size=1, update_epochs=1)),
        module_config={"omega": [1.0, 0.0, 0.0, 0.0, 0.0]},
    ).preference_omegas() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert UncertaintyAgent(
        hybrid_agent=HybridAgent(_agent(rollout_steps=1, minibatch_size=1, update_epochs=1)),
        module_config={"method": "multi_head"},
    ).uncertainty_method == "multi_head"
    assert DistributionalAgent(
        hybrid_agent=HybridAgent(_agent(rollout_steps=1, minibatch_size=1, update_epochs=1)),
        module_config={"cvar_alpha": 0.10},
    ).cvar_alpha == pytest.approx(0.10)
    assert PartialRebalanceAgent(
        hybrid_agent=HybridAgent(_agent(rollout_steps=1, minibatch_size=1, update_epochs=1)),
        module_config={"mode": "continuous_beta"},
    ).partial_rebalance_mode == "continuous_beta"


class _FlatEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, market_image):
        return self.projection(market_image.reshape(market_image.shape[0], -1))


class _PriorityRecorder:
    def __init__(self):
        self.indices = None
        self.td_errors = None

    def update_priorities(self, indices, td_errors):
        self.indices = indices
        self.td_errors = td_errors


class _ForwardPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(
        self,
        market_image,
        mask,
        current_weights,
        estimated_turnover,
        estimated_cost,
        deterministic=False,
        candidate_weights_override=None,
        rebalance_intensity_override=None,
    ):
        self.calls += 1
        batch_size, _, _, n_assets = market_image.shape
        if candidate_weights_override is None:
            candidate = torch.full((batch_size, n_assets), 1.0 / n_assets, device=market_image.device)
        else:
            candidate = candidate_weights_override.to(device=market_image.device, dtype=market_image.dtype)
        intensity = torch.full((batch_size, 1), 0.42, device=market_image.device)
        if rebalance_intensity_override is not None:
            intensity = rebalance_intensity_override.to(device=market_image.device, dtype=market_image.dtype)
        return {
            "candidate_weights": candidate,
            "log_prob": torch.full((batch_size,), -0.2, device=market_image.device) + self.weight,
            "value": torch.full((batch_size, 1), 0.3, device=market_image.device) + self.weight,
            "gate_q": torch.tensor([[0.1, 0.2]], dtype=torch.float32, device=market_image.device),
            "rebalance_intensity": intensity,
        }


class _CostAwarePolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.seen_turnover = []
        self.seen_cost = []

    def forward(
        self,
        market_image,
        mask,
        current_weights,
        estimated_turnover,
        estimated_cost,
        deterministic=False,
        candidate_weights_override=None,
        rebalance_intensity_override=None,
    ):
        self.calls += 1
        self.seen_turnover.append(float(estimated_turnover.view(-1)[0].detach().cpu()))
        self.seen_cost.append(float(estimated_cost.view(-1)[0].detach().cpu()))
        batch_size, _, _, n_assets = market_image.shape
        if candidate_weights_override is None:
            candidate = torch.tensor([[0.8, 0.2]], dtype=market_image.dtype, device=market_image.device).expand(batch_size, n_assets)
        else:
            candidate = candidate_weights_override.to(device=market_image.device, dtype=market_image.dtype)
        q_rebalance = estimated_turnover + estimated_cost
        return {
            "candidate_weights": candidate,
            "log_prob": torch.full((batch_size,), -0.1, device=market_image.device),
            "value": torch.zeros(batch_size, 1, device=market_image.device),
            "gate_q": torch.cat([torch.zeros_like(q_rebalance), q_rebalance], dim=1),
        }


class _AlwaysRebalancePolicyModel(nn.Module):
    def forward(
        self,
        market_image,
        mask,
        current_weights,
        estimated_turnover,
        estimated_cost,
        deterministic=False,
        candidate_weights_override=None,
        rebalance_intensity_override=None,
    ):
        batch_size, _, _, n_assets = market_image.shape
        if candidate_weights_override is None:
            candidate = torch.full((batch_size, n_assets), 1.0 / n_assets, device=market_image.device)
        else:
            candidate = candidate_weights_override.to(device=market_image.device, dtype=market_image.dtype)
        return {
            "candidate_weights": candidate,
            "log_prob": torch.full((batch_size,), -0.1, device=market_image.device),
            "value": torch.zeros(batch_size, 1, device=market_image.device),
            "gate_q": torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=market_image.device),
            "gate_action": torch.ones(batch_size, dtype=torch.long, device=market_image.device),
            "rebalance_intensity": torch.full((batch_size, 1), 0.37, device=market_image.device),
        }


class _HybridHoldRawIntensityPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self,
        market_image,
        mask,
        current_weights,
        estimated_turnover,
        estimated_cost,
        deterministic=False,
        candidate_weights_override=None,
        rebalance_intensity_override=None,
    ):
        self.calls += 1
        batch_size, _, _, n_assets = market_image.shape
        if candidate_weights_override is None:
            candidate = torch.full((batch_size, n_assets), 1.0 / n_assets, device=market_image.device)
        else:
            candidate = candidate_weights_override.to(device=market_image.device, dtype=market_image.dtype)
        if rebalance_intensity_override is None:
            raw_intensity = torch.full((batch_size, 1), 0.42, device=market_image.device)
        else:
            raw_intensity = rebalance_intensity_override.to(device=market_image.device, dtype=market_image.dtype)
        model_gate_action = torch.zeros(batch_size, dtype=torch.long, device=market_image.device)
        return {
            "candidate_weights": candidate,
            "log_prob": torch.full((batch_size,), -0.1, device=market_image.device),
            "joint_log_prob": torch.full((batch_size,), -0.2, device=market_image.device),
            "value": torch.zeros(batch_size, 1, device=market_image.device),
            "gate_q": torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=market_image.device),
            "gate_action": model_gate_action,
            "raw_rebalance_intensity": raw_intensity,
            "rebalance_intensity": raw_intensity * model_gate_action.float().view(-1, 1),
        }


class _ToyEnv:
    execution_config = {"execution_price": "next_open", "delayed_action_execution": False}

    def __init__(self, truncate_at=2):
        self.step_pos = 0
        self.truncate_at = truncate_at

    def reset(self):
        self.step_pos = 0
        return _state(0), {"decision_date": pd.Timestamp("2024-01-02")}

    def step(self, action):
        self.step_pos += 1
        truncated = self.step_pos >= self.truncate_at
        info = {
            "decision_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=self.step_pos - 1),
            "execution_date": pd.Timestamp("2024-01-03") + pd.Timedelta(days=self.step_pos - 1),
            "next_valuation_date": pd.Timestamp("2024-01-03") + pd.Timedelta(days=self.step_pos - 1),
            "executed_weights": np.asarray(action["weights"], dtype=float),
            "rebalance_action": 1,
            "rebalance_intensity": 1.0,
            "turnover": 0.1,
            "auxiliary_labels": {"future_log_return_5d": np.array([0.01, 0.02], dtype=np.float32)},
        }
        return _state(self.step_pos), float(self.step_pos), False, truncated, info


class _RecordingPPOAgent:
    device = torch.device("cpu")

    def __init__(self):
        self.rollout_max_steps = None

    def collect_rollout(self, env, gate_action_selector=None, max_steps=None):
        self.rollout_max_steps = max_steps
        return RolloutBuffer(rollout_steps=1)

    def update(self, rollout):
        return {"status": "updated"}

    def select_action(self, observation, deterministic=True):
        return {
            "candidate_weights": np.array([0.5, 0.5], dtype=np.float32),
            "log_prob": 0.0,
            "value": 0.0,
            "gate_action": 0,
            "rebalance_intensity": 0.0,
            "q_hold": 1.0,
            "q_rebalance": 0.0,
            "q_gap": -1.0,
        }

    def action_for_env(self, observation, action_info):
        return {
            "weights": np.array([0.5, 0.5], dtype=np.float32),
            "rebalance": int(action_info["gate_action"]),
            "rebalance_intensity": float(action_info["rebalance_intensity"]),
        }


class _CountingEvalEnv(_ToyEnv):
    def __init__(self, max_steps=10):
        super().__init__(truncate_at=max_steps)
        self.max_observed_steps = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.max_observed_steps = max(self.max_observed_steps, self.step_pos)
        return observation, reward, terminated, truncated, info


class _RecordingGateEnv(_ToyEnv):
    def __init__(self):
        super().__init__(truncate_at=1)
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        self.step_pos += 1
        info = {
            "decision_date": pd.Timestamp("2024-01-02"),
            "execution_date": pd.Timestamp("2024-01-03"),
            "next_valuation_date": pd.Timestamp("2024-01-03"),
            "executed_weights": np.asarray(action["weights"], dtype=float),
            "rebalance_action": int(action["rebalance"]),
            "rebalance_intensity": float(action["rebalance_intensity"]),
            "turnover": 0.0,
        }
        return _state(self.step_pos), 0.0, False, True, info


class _RealizedInfoEnv(_RecordingGateEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["realized_turnover"] = 0.37
        info["realized_cost"] = 0.004
        return observation, reward, terminated, truncated, info


class _FailingEnv:
    def reset(self):
        return _state(0), {}

    def step(self, action):
        raise RuntimeError("boom")


def _agent(rollout_steps=3, minibatch_size=3, update_epochs=1):
    return PPOAgent(
        _FlatEncoder(),
        PPOActor(latent_dim=4, n_assets=2, hidden_dims=(8,)),
        PPOCritic(latent_dim=4, hidden_dims=(8,)),
        config=PPOAgentConfig(
            rollout_steps=rollout_steps,
            minibatch_size=minibatch_size,
            update_epochs=update_epochs,
            advantage_normalization=True,
            max_grad_norm=0.5,
        ),
    )


def _state(step):
    return {
        "market_image": np.full((1, 2, 2), float(step), dtype=np.float32),
        "current_weights": np.array([0.5, 0.5], dtype=np.float32),
        "availability_mask": np.array([True, True]),
    }


def _rollout_payload(step, **overrides):
    payload = {
        "decision_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=step),
        "execution_date": pd.Timestamp("2024-01-03") + pd.Timedelta(days=step),
        "next_valuation_date": pd.Timestamp("2024-01-03") + pd.Timedelta(days=step),
        "execution_price": "next_open",
        "delayed_action_execution": False,
        "state": _state(step),
        "candidate_weights": np.array([0.6, 0.4], dtype=float),
        "executed_weights": np.array([0.6, 0.4], dtype=float),
        "log_prob": -0.5,
        "value": 0.1,
        "decision_value": 0.1,
        "gate_action": None,
        "rebalance_action": 1,
        "rebalance_intensity": 1.0,
        "reward": 0.1,
        "terminated": False,
        "truncated": False,
        "advantage": 1.0,
        "return_value": 0.2,
    }
    payload.update(overrides)
    return payload
