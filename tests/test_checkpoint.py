import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.agents.dqn_agent import DQNAgent, DQNAgentConfig
from src.agents.hybrid_agent import HybridAgent, HybridAgentConfig
from src.agents.ppo_agent import PPOAgent, PPOAgentConfig
from src.buffers.replay_buffer import ReplayBuffer
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.dqn_gate import DQNGate
from src.models.ppo_actor import PPOActor
from src.models.ppo_critic import PPOCritic
from src.utils.checkpoint import REQUIRED_CHECKPOINT_KEYS, build_checkpoint_payload, load_checkpoint, save_checkpoint


_DESIGN_CHECKPOINT_KEYS = (
    "encoder_state",
    "ppo_actor_state",
    "ppo_critic_state",
    "dqn_gate_state",
    "dqn_target_network_state",
    "auxiliary_head_states",
    "optimizer_states",
    "scheduler_states",
    "amp_grad_scaler_state",
    "replay_buffer_state",
    "epoch",
    "global_step",
    "best_validation_metric",
    "rng_states",
    "resolved_config",
)


def test_checkpoint_payload_schema(tmp_path):
    torch.manual_seed(5)
    hybrid_agent = _hybrid_agent()
    hybrid_agent.policy_model = nn.Sequential(nn.Linear(1, 1))
    hybrid_agent.target_policy_model = nn.Sequential(nn.Linear(1, 1))
    hybrid_agent.ppo_agent.policy_model = hybrid_agent.policy_model
    with torch.no_grad():
        hybrid_agent.policy_model[0].weight.fill_(2.0)
        hybrid_agent.policy_model[0].bias.fill_(0.5)
    hybrid_agent.best_validation_metric = 1.25
    _seed_replay(hybrid_agent.dqn_agent.replay_buffer)

    payload = build_checkpoint_payload(
        hybrid_agent,
        epoch=3,
        global_step=11,
        resolved_config={"experiment": {"type": "main_model"}, "device": {"mode": "cpu"}},
    )

    assert REQUIRED_CHECKPOINT_KEYS == ("schema_version", *_DESIGN_CHECKPOINT_KEYS)
    assert set(REQUIRED_CHECKPOINT_KEYS).issubset(payload)
    assert {
        "gate_step",
        "training_history",
        "best_checkpoint_score",
        "hybrid_status",
        "policy_model_state",
        "target_policy_model_state",
    }.issubset(payload)
    assert payload["encoder_state"]
    assert payload["ppo_actor_state"]
    assert payload["ppo_critic_state"]
    assert payload["dqn_gate_state"]
    assert payload["dqn_target_network_state"]
    assert payload["auxiliary_head_states"]
    assert set(payload["optimizer_states"]) == {"ppo", "dqn", "auxiliary"}
    assert payload["scheduler_states"] == {}
    assert payload["amp_grad_scaler_state"] is None
    assert payload["replay_buffer_state"]["buffer_type"] == "PrioritizedReplayBuffer"
    assert len(payload["replay_buffer_state"]["items"]) == 1
    assert payload["epoch"] == 3
    assert payload["global_step"] == 11
    assert payload["gate_step"] == 0
    assert payload["training_history"] == []
    assert payload["hybrid_status"] == "initialized"
    assert payload["best_validation_metric"] == pytest.approx(1.25)
    assert set(payload["rng_states"]) == {
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "environment_random_state",
    }
    assert payload["resolved_config"]["experiment"]["type"] == "main_model"

    checkpoint_path = tmp_path / "checkpoints" / "model.pt"
    save_checkpoint(payload, checkpoint_path)
    loaded = load_checkpoint(checkpoint_path, device="cpu")

    assert set(REQUIRED_CHECKPOINT_KEYS).issubset(loaded)
    assert loaded["epoch"] == 3
    assert loaded["global_step"] == 11
    assert loaded["amp_grad_scaler_state"] is None
    assert loaded["replay_buffer_state"]["items"][0]["decision_date_t"] == "2024-01-02T00:00:00"

    target_agent = _hybrid_agent()
    target_agent.policy_model = nn.Sequential(nn.Linear(1, 1))
    target_agent.target_policy_model = nn.Sequential(nn.Linear(1, 1))
    target_agent.ppo_agent.policy_model = target_agent.policy_model
    load_checkpoint(checkpoint_path, device="cpu", agent=target_agent)
    for source, restored in zip(
        hybrid_agent.ppo_agent.actor.parameters(),
        target_agent.ppo_agent.actor.parameters(),
        strict=True,
    ):
        assert torch.allclose(source, restored)
    assert len(target_agent.dqn_agent.replay_buffer) == 1
    assert target_agent.dqn_agent.global_step == 11
    assert target_agent.start_epoch == 4
    assert target_agent.global_step == 11
    assert target_agent.best_validation_metric == pytest.approx(1.25)
    assert target_agent.policy_model[0].weight.item() == pytest.approx(2.0)
    assert target_agent.policy_model[0].bias.item() == pytest.approx(0.5)

    target_agent.ppo_agent.policy_model = None
    resumed = target_agent.train(_TinyEnv(), epochs=1)
    assert resumed["status"] == "completed"
    assert resumed["history"][-1]["epoch"] == 4
    assert target_agent.global_step == 12


def test_checkpoint_payload_can_omit_replay_buffer():
    hybrid_agent = _hybrid_agent()
    _seed_replay(hybrid_agent.dqn_agent.replay_buffer)

    payload = build_checkpoint_payload(
        hybrid_agent,
        epoch=0,
        global_step=1,
        resolved_config={"training": {"checkpoint_include_replay_buffer": False}},
        include_replay_buffer=False,
    )

    assert payload["replay_buffer_state"] is None


class _FlatEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, market_image):
        return self.projection(market_image.reshape(market_image.shape[0], -1))


def _ppo_agent():
    return PPOAgent(
        _FlatEncoder(),
        PPOActor(latent_dim=4, n_assets=2, hidden_dims=(8,)),
        PPOCritic(latent_dim=4, hidden_dims=(8,)),
        config=PPOAgentConfig(rollout_steps=1, minibatch_size=1, update_epochs=1),
    )


def _hybrid_agent():
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
    return HybridAgent(
        _ppo_agent(),
        dqn_agent=dqn_agent,
        auxiliary_heads=auxiliary_heads,
        config=HybridAgentConfig(epochs=1, validation_episodes=1),
    )


def _state(step):
    return {
        "market_image": np.full((1, 2, 2), float(step), dtype=np.float32),
        "current_weights": np.array([0.5, 0.5], dtype=np.float32),
        "availability_mask": np.array([True, True]),
    }


def _seed_replay(buffer: ReplayBuffer) -> None:
    buffer.add_transition(
        state_t=_state(0),
        state_tp1=_state(1),
        decision_date_t=pd.Timestamp("2024-01-02"),
        execution_date_t=pd.Timestamp("2024-01-03"),
        next_valuation_date_t=pd.Timestamp("2024-01-03"),
        decision_date_next=pd.Timestamp("2024-01-03"),
        execution_date_next=pd.Timestamp("2024-01-04"),
        next_valuation_date_next=pd.Timestamp("2024-01-04"),
        execution_price_t="next_open",
        delayed_action_execution_t=False,
        execution_price_next="next_open",
        delayed_action_execution_next=False,
        candidate_weights_t=np.array([0.6, 0.4], dtype=float),
        executed_weights_t=np.array([0.6, 0.4], dtype=float),
        gate_action_t=1,
        rebalance_action_t=1,
        rebalance_intensity_t=1.0,
        estimated_turnover_t=0.1,
        realized_turnover_t=0.1,
        estimated_cost_t=0.001,
        realized_cost_t=0.001,
        reward_t=0.02,
        terminated_t=False,
        truncated_t=True,
        split_boundary_t=True,
        q_hold_t=0.1,
        q_rebalance_t=0.2,
        q_gap_t=0.1,
    )


class _TinyEnv:
    execution_config = {"execution_price": "next_open", "delayed_action_execution": False}

    def __init__(self):
        self.step_pos = 0

    def reset(self):
        self.step_pos = 0
        return _state(0), {}

    def step(self, action):
        self.step_pos += 1
        info = {
            "decision_date": pd.Timestamp("2024-01-02"),
            "execution_date": pd.Timestamp("2024-01-03"),
            "next_valuation_date": pd.Timestamp("2024-01-03"),
            "executed_weights": np.asarray(action["weights"], dtype=float),
            "rebalance_action": int(action.get("rebalance", 1)),
            "rebalance_intensity": float(action.get("rebalance_intensity", 1.0)),
            "turnover": 0.05,
            "realized_cost": 0.001,
        }
        return _state(self.step_pos), 0.01, False, True, info
