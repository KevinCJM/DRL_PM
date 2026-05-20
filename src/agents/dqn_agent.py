from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.training_guard import assert_finite_tensor, clip_grad_norm_checked
from src.buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.buffers.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DQNAgentConfig:
    replay_size: int = 100000
    batch_size: int = 128
    warmup_steps: int = 1000
    gamma: float = 0.99
    target_update_interval: int = 500
    soft_update_tau: float | None = None
    epsilon_start: float = 1.00
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000
    use_double_dqn: bool = True
    use_n_step: bool = True
    n_steps: int = 3
    use_prioritized_replay: bool = True
    per_priority_eps: float = 1.0e-6
    max_grad_norm: float = 0.50
    lr: float = 1.0e-4
    detach_encoder: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None = None) -> DQNAgentConfig:
        if not config:
            return cls()
        dqn_config = _mapping(config.get("dqn"))
        optimizer_config = _mapping(config.get("optimizer"))
        soft_update_tau = dqn_config.get("soft_update_tau", cls.soft_update_tau)
        use_n_step = bool(dqn_config.get("use_n_step", cls.use_n_step))
        n_steps = _positive_int("n_steps", _first_present(dqn_config, "n_step", "n_steps", default=cls.n_steps))
        if not use_n_step:
            n_steps = 1
        lr = _first_present(optimizer_config, "dqn_lr", "learning_rate", default=None)
        if lr is None:
            lr = _first_present(dqn_config, "lr", default=cls.lr)
        return cls(
            replay_size=_positive_int("replay_size", dqn_config.get("replay_size", cls.replay_size)),
            batch_size=_positive_int("batch_size", dqn_config.get("batch_size", cls.batch_size)),
            warmup_steps=_non_negative_int("warmup_steps", dqn_config.get("warmup_steps", cls.warmup_steps)),
            gamma=_unit_interval_float("gamma", dqn_config.get("gamma", cls.gamma)),
            target_update_interval=_positive_int(
                "target_update_interval",
                dqn_config.get("target_update_interval", cls.target_update_interval),
            ),
            soft_update_tau=None if soft_update_tau is None else _unit_interval_float("soft_update_tau", soft_update_tau),
            epsilon_start=_unit_interval_float("epsilon_start", dqn_config.get("epsilon_start", cls.epsilon_start)),
            epsilon_end=_unit_interval_float("epsilon_end", dqn_config.get("epsilon_end", cls.epsilon_end)),
            epsilon_decay_steps=_positive_int(
                "epsilon_decay_steps",
                dqn_config.get("epsilon_decay_steps", cls.epsilon_decay_steps),
            ),
            use_double_dqn=bool(_first_present(dqn_config, "double_dqn", "use_double_dqn", default=cls.use_double_dqn)),
            use_n_step=use_n_step,
            n_steps=n_steps,
            use_prioritized_replay=bool(
                _first_present(dqn_config, "per_enabled", "use_prioritized_replay", default=cls.use_prioritized_replay)
            ),
            per_priority_eps=_positive_float(
                "per_priority_eps",
                dqn_config.get("per_priority_eps", cls.per_priority_eps),
            ),
            max_grad_norm=_positive_float("max_grad_norm", dqn_config.get("max_grad_norm", cls.max_grad_norm)),
            lr=_positive_float("lr", lr),
            detach_encoder=bool(dqn_config.get("detach_encoder", cls.detach_encoder)),
        )


class DQNAgent:
    def __init__(
        self,
        online_network: nn.Module,
        target_network: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        replay_buffer: ReplayBuffer | PrioritizedReplayBuffer | None = None,
        config: Mapping[str, Any] | DQNAgentConfig | None = None,
        device: torch.device | str | None = None,
        encoder: nn.Module | None = None,
        target_encoder: nn.Module | None = None,
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.config = config if isinstance(config, DQNAgentConfig) else DQNAgentConfig.from_mapping(config)
        self.device = torch.device("cpu" if device is None else device)
        self.online_network.to(self.device)
        self.target_network.to(self.device)
        if self.encoder is not None:
            self.encoder.to(self.device)
        if self.target_encoder is not None:
            self.target_encoder.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        self.replay_buffer = replay_buffer or self._default_replay_buffer()
        self.global_step = 0

    def parameters(self) -> list[nn.Parameter]:
        params = list(self.online_network.parameters())
        if self.encoder is not None and not self.config.detach_encoder:
            params.extend(list(self.encoder.parameters()))
        return params

    def update(self, batch: Mapping[str, Any] | None = None) -> dict[str, float]:
        sample = batch if batch is not None else self._sample_replay()
        training_batch = _mapping(sample.get("batch")) if isinstance(sample, Mapping) and "batch" in sample else sample
        if not isinstance(training_batch, Mapping):
            raise TypeError("ERR_DQN_AGENT_BATCH_TYPE")

        q_values = self._q_values(training_batch, next_state=False, target=False)
        actions = _long_column(training_batch["gate_action_t"], self.device)
        selected_q = q_values.gather(1, actions)
        with torch.no_grad():
            next_q_online = self._q_values(training_batch, next_state=True, target=False)
            next_q_target = self._q_values(training_batch, next_state=True, target=True)
            rewards = _float_column(training_batch["reward_t"], self.device)
            terminated = _bool_column(training_batch["terminated_t"], self.device)
            truncated = _bool_column(training_batch["truncated_t"], self.device)
            if self.config.use_double_dqn:
                target = self.compute_double_dqn_target(
                    rewards=rewards,
                    next_q_online=next_q_online,
                    next_q_target=next_q_target,
                    terminated=terminated,
                    truncated=truncated,
                    gamma=self.config.gamma,
                    n_steps=training_batch.get("n_steps", self.config.n_steps),
                    discount=training_batch.get("discount"),
                )
            else:
                target = self.compute_dqn_target(
                    rewards=rewards,
                    next_q_target=next_q_target,
                    terminated=terminated,
                    truncated=truncated,
                    gamma=self.config.gamma,
                    n_steps=training_batch.get("n_steps", self.config.n_steps),
                    discount=training_batch.get("discount"),
                )
            if "bootstrap_mask_t" in training_batch:
                bootstrap_mask = _float_column(training_batch["bootstrap_mask_t"], self.device)
                target = rewards + bootstrap_mask * (target - rewards)

        weights = self._sample_weights(sample, selected_q.shape[0])
        loss_per_item = F.smooth_l1_loss(selected_q, target, reduction="none")
        loss = (loss_per_item * weights).mean()
        assert_finite_tensor(loss, "dqn.loss", "ERR_TRAINING_NON_FINITE_LOSS")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_checked(self.parameters(), self.config.max_grad_norm, "dqn")
        self.optimizer.step()

        td_errors = (selected_q.detach() - target).view(-1)
        self.update_replay_priorities(sample, td_errors)
        self.global_step += 1
        self.update_target_network(self.global_step)
        return {
            "loss": float(loss.detach().cpu()),
            "td_error_abs_mean": float(td_errors.abs().mean().detach().cpu()),
            "q_mean": float(selected_q.detach().mean().cpu()),
            "target_mean": float(target.detach().mean().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }

    @staticmethod
    def compute_double_dqn_target(
        rewards: torch.Tensor,
        next_q_online: torch.Tensor,
        next_q_target: torch.Tensor,
        terminated: torch.Tensor | None = None,
        truncated: torch.Tensor | None = None,
        gamma: float = 0.99,
        n_steps: int | torch.Tensor | Sequence[int] = 1,
        discount: torch.Tensor | Sequence[float] | float | None = None,
        done: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if terminated is None and done is None:
            raise ValueError("ERR_DQN_AGENT_TARGET_DONE_REQUIRED")
        if terminated is None:
            terminated = done
        if done is not None:
            terminated = done if terminated is None else (terminated.to(dtype=torch.bool) | done.to(dtype=torch.bool))
        _assert_column(rewards, "rewards")
        _assert_column(terminated, "terminated")
        if truncated is None:
            truncated = torch.zeros_like(terminated, dtype=torch.bool)
        _assert_column(truncated, "truncated")
        if next_q_online.ndim != 2 or next_q_target.shape != next_q_online.shape:
            raise ValueError("ERR_DQN_AGENT_SHAPE_MISMATCH: next_q_online/next_q_target must share [batch,actions]")
        if rewards.shape[0] != next_q_online.shape[0] or terminated.shape[0] != next_q_online.shape[0]:
            raise ValueError("ERR_DQN_AGENT_SHAPE_MISMATCH: batch sizes must match")
        if not _all_finite(rewards, next_q_online, next_q_target):
            raise ValueError("ERR_DQN_AGENT_NON_FINITE: target inputs contain NaN or Inf")

        next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
        next_values = next_q_target.gather(1, next_actions)
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        discount_tensor = _discount_column(discount, n_steps, gamma, rewards)
        return rewards + (~done).to(dtype=rewards.dtype) * discount_tensor * next_values

    compute_target = compute_double_dqn_target

    @staticmethod
    def compute_dqn_target(
        rewards: torch.Tensor,
        next_q_target: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor | None = None,
        gamma: float = 0.99,
        n_steps: int | torch.Tensor | Sequence[int] = 1,
        discount: torch.Tensor | Sequence[float] | float | None = None,
    ) -> torch.Tensor:
        _assert_column(rewards, "rewards")
        _assert_column(terminated, "terminated")
        if truncated is None:
            truncated = torch.zeros_like(terminated, dtype=torch.bool)
        _assert_column(truncated, "truncated")
        if next_q_target.ndim != 2 or rewards.shape[0] != next_q_target.shape[0]:
            raise ValueError("ERR_DQN_AGENT_SHAPE_MISMATCH: target batch sizes must match")
        if not _all_finite(rewards, next_q_target):
            raise ValueError("ERR_DQN_AGENT_NON_FINITE: target inputs contain NaN or Inf")
        next_values = next_q_target.max(dim=1, keepdim=True).values
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        discount_tensor = _discount_column(discount, n_steps, gamma, rewards)
        return rewards + (~done).to(dtype=rewards.dtype) * discount_tensor * next_values

    def update_target_network(self, global_step: int | None = None) -> bool:
        if self.config.soft_update_tau is not None:
            self.soft_update_target_network(self.online_network, self.target_network, self.config.soft_update_tau)
            if self.encoder is not None and self.target_encoder is not None:
                self.soft_update_target_network(self.encoder, self.target_encoder, self.config.soft_update_tau)
            return True
        step = self.global_step if global_step is None else int(global_step)
        if step % self.config.target_update_interval != 0:
            return False
        self.hard_update_target_network(self.online_network, self.target_network)
        if self.encoder is not None and self.target_encoder is not None:
            self.hard_update_target_network(self.encoder, self.target_encoder)
        return True

    sync_target_network = update_target_network

    @staticmethod
    def hard_update_target_network(source: nn.Module, target: nn.Module) -> None:
        target.load_state_dict(source.state_dict())

    hard_update = hard_update_target_network

    @staticmethod
    def soft_update_target_network(source: nn.Module, target: nn.Module, tau: float) -> None:
        tau_value = _unit_interval_float("soft_update_tau", tau)
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
                target_param.data.mul_(1.0 - tau_value).add_(source_param.data, alpha=tau_value)

    soft_update = soft_update_target_network

    def epsilon_at_step(self, step: int) -> float:
        step_value = max(int(step), 0)
        fraction = min(step_value / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    epsilon_schedule = epsilon_at_step

    def select_action(
        self,
        q_values: torch.Tensor,
        step: int,
        rng: np.random.Generator | None = None,
    ) -> torch.Tensor:
        if q_values.ndim != 2:
            raise ValueError("ERR_DQN_AGENT_SHAPE_MISMATCH: q_values must be [batch,actions]")
        epsilon = self.epsilon_at_step(step)
        greedy = torch.argmax(q_values, dim=1)
        if rng is None:
            random_values = np.random.randint(0, q_values.shape[1], size=q_values.shape[0])
            explore_values = np.random.random(q_values.shape[0]) < epsilon
        else:
            random_values = rng.integers(0, q_values.shape[1], size=q_values.shape[0])
            explore_values = rng.random(q_values.shape[0]) < epsilon
        random_actions = torch.as_tensor(
            random_values,
            dtype=torch.long,
            device=q_values.device,
        )
        explore = torch.as_tensor(
            explore_values,
            dtype=torch.bool,
            device=q_values.device,
        )
        return torch.where(explore, random_actions, greedy)

    def update_replay_priorities(self, sample: Mapping[str, Any], td_errors: torch.Tensor) -> None:
        indices = sample.get("indices") if isinstance(sample, Mapping) else None
        if indices is None or not hasattr(self.replay_buffer, "update_priorities"):
            return
        td_error_values = td_errors.detach().abs().view(-1).cpu().numpy()
        self.replay_buffer.update_priorities(np.asarray(indices, dtype=np.int64).reshape(-1), td_errors=td_error_values)

    update_priorities = update_replay_priorities

    def _default_replay_buffer(self) -> ReplayBuffer | PrioritizedReplayBuffer:
        if self.config.use_prioritized_replay:
            return PrioritizedReplayBuffer(
                capacity=self.config.replay_size,
                gamma=self.config.gamma,
                n_steps=self.config.n_steps,
                per_priority_eps=self.config.per_priority_eps,
            )
        return ReplayBuffer(capacity=self.config.replay_size, gamma=self.config.gamma, n_steps=self.config.n_steps)

    def _sample_replay(self) -> Mapping[str, Any]:
        if len(self.replay_buffer) < self.config.batch_size:
            raise ValueError("ERR_DQN_AGENT_REPLAY_WARMUP")
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            return self.replay_buffer.sample(self.config.batch_size, device=self.device)
        items = self.replay_buffer.sample(self.config.batch_size)
        return self.replay_buffer.as_batch(items, device=self.device)

    def _q_values(self, batch: Mapping[str, Any], next_state: bool, target: bool) -> torch.Tensor:
        network = self.target_network if target else self.online_network
        latent = self._latent(batch, next_state=next_state, target=target)
        candidate = _tensor_from_batch(batch, "candidate_weights_t", self.device)
        current = _current_weights(batch, next_state=next_state, device=self.device)
        turnover = _float_column(batch["estimated_turnover_t"], self.device)
        cost = _float_column(batch["estimated_cost_t"], self.device)
        return network(latent, candidate, current, turnover, cost)

    def _latent(self, batch: Mapping[str, Any], next_state: bool, target: bool) -> torch.Tensor:
        key = "state_tp1" if next_state else "state_t"
        states = batch[key]
        encoder = self.target_encoder if target and self.target_encoder is not None else self.encoder
        if encoder is None:
            return _latent_from_states(states, self.device)
        market_image = torch.as_tensor(
            np.stack([np.asarray(state["market_image"], dtype=np.float32) for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        if self.config.detach_encoder:
            with torch.no_grad():
                return encoder(market_image).detach()
        return encoder(market_image)

    def _sample_weights(self, sample: Mapping[str, Any], batch_size: int) -> torch.Tensor:
        if isinstance(sample, Mapping):
            if "is_weight" in sample:
                return _float_column(sample["is_weight"], self.device)
            if "is_weights" in sample:
                return _float_column(sample["is_weights"], self.device)
            if "batch" in sample and isinstance(sample["batch"], Mapping) and "is_weight" in sample["batch"]:
                return _float_column(sample["batch"]["is_weight"], self.device)
        return torch.ones(batch_size, 1, dtype=torch.float32, device=self.device)


def _latent_from_states(states: Sequence[Any], device: torch.device) -> torch.Tensor:
    values: list[np.ndarray] = []
    for state in states:
        if isinstance(state, Mapping):
            if "latent" in state:
                values.append(np.asarray(state["latent"], dtype=np.float32))
            elif "dqn_latent" in state:
                values.append(np.asarray(state["dqn_latent"], dtype=np.float32))
            elif "market_image" in state:
                values.append(np.asarray(state["market_image"], dtype=np.float32).reshape(-1))
            else:
                raise KeyError("ERR_DQN_AGENT_STATE_KEY: latent")
        else:
            values.append(np.asarray(state, dtype=np.float32))
    return torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)


def _current_weights(batch: Mapping[str, Any], next_state: bool, device: torch.device) -> torch.Tensor:
    if next_state and "current_weights_tp1" in batch:
        return _tensor_from_batch(batch, "current_weights_tp1", device)
    if not next_state and "current_weights_t" in batch:
        return _tensor_from_batch(batch, "current_weights_t", device)
    states = batch["state_tp1"] if next_state else batch["state_t"]
    if isinstance(states, Sequence) and states and isinstance(states[0], Mapping) and "current_weights" in states[0]:
        return torch.as_tensor(
            np.stack([np.asarray(state["current_weights"], dtype=np.float32) for state in states]),
            dtype=torch.float32,
            device=device,
        )
    return _tensor_from_batch(batch, "executed_weights_t", device)


def _tensor_from_batch(batch: Mapping[str, Any], key: str, device: torch.device) -> torch.Tensor:
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32, device=device)


def _float_column(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.float32)
    else:
        result = torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32, device=device)
    return result.view(-1, 1)


def _long_column(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.long)
    else:
        result = torch.as_tensor(np.asarray(value, dtype=np.int64), dtype=torch.long, device=device)
    return result.view(-1, 1)


def _bool_column(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.bool)
    else:
        result = torch.as_tensor(np.asarray(value, dtype=bool), dtype=torch.bool, device=device)
    return result.view(-1, 1)


def _discount_column(
    discount: torch.Tensor | Sequence[float] | float | None,
    n_steps: int | torch.Tensor | Sequence[int],
    gamma: float,
    rewards: torch.Tensor,
) -> torch.Tensor:
    if discount is not None:
        return _float_column(discount, rewards.device).to(dtype=rewards.dtype)
    if isinstance(n_steps, torch.Tensor):
        steps = n_steps.to(device=rewards.device, dtype=rewards.dtype).view(-1, 1)
        return float(gamma) ** steps
    if isinstance(n_steps, Sequence) and not isinstance(n_steps, (str, bytes)):
        steps = torch.as_tensor(np.asarray(n_steps, dtype=np.float32), dtype=rewards.dtype, device=rewards.device).view(-1, 1)
        return float(gamma) ** steps
    return torch.full_like(rewards, float(gamma) ** int(n_steps))


def _assert_column(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"ERR_DQN_AGENT_SHAPE_MISMATCH: {name} must be [batch,1]")


def _all_finite(*values: torch.Tensor) -> bool:
    return all(torch.isfinite(value).all().item() for value in values)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _finite_float(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"ERR_DQN_AGENT_CONFIG_INVALID: {name}")
    return result


def _positive_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"ERR_DQN_AGENT_CONFIG_INVALID: {name}")
    return result


def _unit_interval_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"ERR_DQN_AGENT_CONFIG_INVALID: {name}")
    return result


def _positive_int(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"ERR_DQN_AGENT_CONFIG_INVALID: {name}")
    return result


def _non_negative_int(name: str, value: Any) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"ERR_DQN_AGENT_CONFIG_INVALID: {name}")
    return result


__all__ = ["DQNAgent", "DQNAgentConfig"]
