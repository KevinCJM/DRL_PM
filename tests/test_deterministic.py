import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.agents.hybrid_agent import HybridAgent, HybridAgentConfig
from src.agents.ppo_agent import PPOAgent, PPOAgentConfig
from src.models.ppo_actor import PPOActor
from src.models.ppo_critic import PPOCritic
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.seed import collect_rng_states, restore_rng_states, set_global_seed


def test_set_global_seed_and_backend_flags():
    config = {
        "deterministic_torch": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
    }

    set_global_seed(123, config)
    python_value = random.random()
    numpy_value = float(np.random.random())
    torch_value = float(torch.rand(1))

    set_global_seed(123, config)
    assert random.random() == pytest.approx(python_value)
    assert float(np.random.random()) == pytest.approx(numpy_value)
    assert float(torch.rand(1)) == pytest.approx(torch_value)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_checkpoint_restores_rng_and_deterministic_evaluation(tmp_path):
    set_global_seed(
        321,
        {
            "deterministic_torch": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
    )
    source_agent = _agent()
    source_env = _DeterministicEnv(seed=77)
    checkpoint_path = tmp_path / "checkpoint.pt"

    source_metrics = source_agent.evaluate(_DeterministicEnv(seed=10), deterministic=True)
    save_checkpoint(
        source_agent,
        checkpoint_path,
        epoch=2,
        global_step=5,
        best_validation_metric=source_metrics["validation_sharpe"],
        resolved_config={"reproducibility": {"seed": 321}},
        env=source_env,
    )
    expected_env_draw = source_env.next_random()

    set_global_seed(999)
    target_agent = _agent()
    target_env = _DeterministicEnv(seed=1)
    load_checkpoint(checkpoint_path, device="cpu", agent=target_agent, env=target_env)

    restored_metrics = target_agent.evaluate(_DeterministicEnv(seed=10), deterministic=True)
    assert restored_metrics == source_metrics
    assert target_env.next_random() == pytest.approx(expected_env_draw)

    states = collect_rng_states()
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = float(torch.rand(1))
    random.random()
    np.random.random()
    torch.rand(1)
    restore_rng_states(states)
    assert random.random() == pytest.approx(expected_random)
    assert float(np.random.random()) == pytest.approx(expected_numpy)
    assert float(torch.rand(1)) == pytest.approx(expected_torch)


class _FlatEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, market_image):
        return self.projection(market_image.reshape(market_image.shape[0], -1))


class _DeterministicEnv:
    execution_config = {"execution_price": "next_open", "delayed_action_execution": False}

    def __init__(self, seed=0, truncate_at=2):
        self.np_random = np.random.default_rng(seed)
        self.truncate_at = int(truncate_at)
        self.step_pos = 0

    def reset(self, seed=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.step_pos = 0
        return _state(0), {}

    def step(self, action):
        self.step_pos += 1
        return _state(self.step_pos), float(self.step_pos), False, self.step_pos >= self.truncate_at, {
            "executed_weights": np.asarray(action["weights"], dtype=float),
            "turnover": 0.1,
        }

    def get_rng_state(self):
        return self.np_random.bit_generator.state

    def set_rng_state(self, state):
        self.np_random.bit_generator.state = state

    def next_random(self):
        return float(self.np_random.random())


def _agent():
    return HybridAgent(
        PPOAgent(
            _FlatEncoder(),
            PPOActor(latent_dim=4, n_assets=2, hidden_dims=(8,)),
            PPOCritic(latent_dim=4, hidden_dims=(8,)),
            config=PPOAgentConfig(rollout_steps=1, minibatch_size=1, update_epochs=1),
        ),
        config=HybridAgentConfig(epochs=1, validation_episodes=1),
    )


def _state(step):
    return {
        "market_image": np.full((1, 2, 2), float(step), dtype=np.float32),
        "current_weights": np.array([0.5, 0.5], dtype=np.float32),
        "availability_mask": np.array([True, True]),
    }
