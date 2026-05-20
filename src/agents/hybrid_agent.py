from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.training_guard import assert_finite_losses, clip_grad_norm_checked
from src.buffers.rollout_buffer import RolloutBuffer, RolloutItem
from src.utils.logger import mark_run_failed
from src.utils.seed import collect_rng_states


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridAgentConfig:
    epochs: int = 1
    validation_episodes: int = 1
    max_train_steps: int | None = None
    max_validation_steps: int | None = None
    auxiliary_lr: float = 3.0e-4
    max_grad_norm: float = 0.50
    run_registry_path: str | None = None
    run_id: str = "default"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None = None) -> HybridAgentConfig:
        if not config:
            return cls()
        training_config = _mapping(config.get("training"))
        optimizer_config = _mapping(config.get("optimizer"))
        evaluation_config = _mapping(config.get("evaluation"))
        registry_config = _mapping(config.get("registry"))
        output_config = _mapping(config.get("output"))
        registry_path = None
        if bool(registry_config.get("enabled", False)) and registry_config.get("path") is not None:
            registry_path = str(registry_config["path"])
        return cls(
            epochs=_positive_int("epochs", training_config.get("epochs", training_config.get("num_epochs", cls.epochs))),
            validation_episodes=_positive_int(
                "validation_episodes",
                evaluation_config.get("validation_episodes", cls.validation_episodes),
            ),
            max_train_steps=_optional_positive_int("max_train_steps", training_config.get("max_train_steps")),
            max_validation_steps=_optional_positive_int(
                "max_validation_steps",
                training_config.get("max_validation_steps"),
            ),
            auxiliary_lr=_positive_float(
                "auxiliary_lr",
                optimizer_config.get("auxiliary_lr") or optimizer_config.get("learning_rate") or cls.auxiliary_lr,
            ),
            max_grad_norm=_positive_float(
                "max_grad_norm",
                optimizer_config.get("max_grad_norm") or training_config.get("max_grad_norm", cls.max_grad_norm),
            ),
            run_registry_path=registry_path,
            run_id=str(output_config.get("run_name", registry_config.get("run_id", cls.run_id))),
        )


class HybridAgent:
    def __init__(
        self,
        ppo_agent: PPOAgent,
        dqn_agent: DQNAgent | None = None,
        auxiliary_heads: nn.Module | None = None,
        auxiliary_optimizer: torch.optim.Optimizer | None = None,
        config: Mapping[str, Any] | HybridAgentConfig | None = None,
        checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ):
        self.ppo_agent = ppo_agent
        self.dqn_agent = dqn_agent
        self.auxiliary_heads = auxiliary_heads
        self.config = config if isinstance(config, HybridAgentConfig) else HybridAgentConfig.from_mapping(config)
        self.checkpoint_callback = checkpoint_callback
        self.device = self.ppo_agent.device
        if self.auxiliary_heads is not None:
            self.auxiliary_heads.to(self.device)
        self.auxiliary_optimizer = auxiliary_optimizer or self._default_auxiliary_optimizer()
        self.status = "initialized"
        self.failure_state: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.best_checkpoint_payload: dict[str, Any] | None = None
        self.best_validation_metric: float | None = None
        self._best_checkpoint_score: tuple[float, float, float, float] | None = None
        self.start_epoch = 0
        self.last_epoch = -1
        self.global_step = 0
        self.gate_step = 0

    def train(
        self,
        env: Any,
        validation_env: Any | None = None,
        epochs: int | None = None,
        num_epochs: int | None = None,
    ) -> dict[str, Any]:
        requested_epochs = self.config.epochs if epochs is None and num_epochs is None else epochs if epochs is not None else num_epochs
        total_epochs = _positive_int("epochs", requested_epochs)
        self.status = "running"
        self.failure_state = None
        try:
            for epoch in range(self.start_epoch, self.start_epoch + total_epochs):
                rollout = self.ppo_agent.collect_rollout(
                    env,
                    gate_action_selector=self._training_gate_action_selector if self.dqn_agent is not None else None,
                    max_steps=self.config.max_train_steps,
                )
                self._sync_dqn_replay(rollout)
                ppo_stats = self.ppo_agent.update(rollout)
                dqn_stats = self._update_dqn()
                auxiliary_stats = self._update_auxiliary(rollout)
                validation_stats = self.evaluate(validation_env, deterministic=True) if validation_env is not None else {}
                record = {
                    "epoch": epoch,
                    "ppo": ppo_stats,
                    "dqn": dqn_stats,
                    "auxiliary": auxiliary_stats,
                    "validation": validation_stats,
                }
                self.history.append(record)
                self.last_epoch = epoch
                self.global_step += 1
                self._maybe_checkpoint(epoch, record)
            self.status = "completed"
            return {
                "status": self.status,
                "history": list(self.history),
                "best_validation_metric": self.best_validation_metric,
            }
        except Exception as exc:
            self.status = "failed"
            self.failure_state = {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "epoch": len(self.history),
            }
            LOGGER.exception("ERR_HYBRID_AGENT_TRAINING_FAILED")
            try:
                mark_run_failed(self.failure_state, self.config.run_registry_path, self.config.run_id)
            except Exception:
                LOGGER.exception("ERR_RUN_REGISTRY_FAILED_UPDATE")
            raise

    def evaluate(self, env: Any, deterministic: bool = True) -> dict[str, float]:
        rewards: list[float] = []
        turnovers: list[float] = []
        for _ in range(self.config.validation_episodes):
            observation, _ = env.reset()
            terminated = False
            truncated = False
            step_count = 0
            while not (terminated or truncated):
                if self.config.max_validation_steps is not None and step_count >= int(self.config.max_validation_steps):
                    break
                action_info = self.ppo_agent.select_action(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(
                    self.ppo_agent.action_for_env(observation, action_info)
                )
                rewards.append(float(reward))
                if "turnover" in info and info["turnover"] is not None:
                    turnovers.append(float(info["turnover"]))
                step_count += 1
        if not rewards:
            return {
                "validation_reward": 0.0,
                "validation_sharpe": 0.0,
                "validation_max_drawdown": 0.0,
                "validation_turnover": 0.0,
            }
        reward_array = np.asarray(rewards, dtype=float)
        std = float(reward_array.std(ddof=0))
        sharpe = 0.0 if std <= 1.0e-12 else float(reward_array.mean() / std)
        nav = np.cumprod(1.0 + reward_array)
        running_max = np.maximum.accumulate(nav)
        drawdown = np.where(running_max > 0.0, 1.0 - nav / running_max, 0.0)
        return {
            "validation_reward": float(reward_array.sum()),
            "validation_sharpe": sharpe,
            "validation_max_drawdown": float(drawdown.max(initial=0.0)),
            "validation_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        }

    def _sync_dqn_replay(self, rollout: RolloutBuffer) -> None:
        if self.dqn_agent is None:
            return
        items = rollout.items
        if not items:
            return
        last_observation = getattr(rollout, "last_observation", None)
        rollout_boundary_split = bool(
            getattr(rollout, "rollout_boundary_split", getattr(rollout, "rollout_boundary_truncated", False))
        )
        for index, item in enumerate(items):
            next_fields = _dqn_next_fields(items, index, last_observation, rollout_boundary_split)
            self.dqn_agent.replay_buffer.add_transition(
                state_t=item.state,
                state_tp1=next_fields["state"],
                decision_date_t=item.decision_date,
                execution_date_t=item.execution_date,
                next_valuation_date_t=item.next_valuation_date,
                decision_date_next=next_fields["decision_date"],
                execution_date_next=next_fields["execution_date"],
                next_valuation_date_next=next_fields["next_valuation_date"],
                execution_price_t=item.execution_price,
                delayed_action_execution_t=item.delayed_action_execution,
                execution_price_next=next_fields["execution_price"],
                delayed_action_execution_next=next_fields["delayed_action_execution"],
                candidate_weights_t=item.candidate_weights,
                executed_weights_t=item.executed_weights,
                gate_action_t=_dqn_gate_action(item, self.dqn_agent),
                rebalance_action_t=_rebalance_action(item),
                rebalance_intensity_t=item.rebalance_intensity,
                estimated_turnover_t=_field_float(item, "estimated_turnover", 0.0),
                realized_turnover_t=_field_float(item, "realized_turnover", 0.0),
                estimated_cost_t=_field_float(item, "estimated_cost", 0.0),
                realized_cost_t=_field_float(item, "realized_cost", 0.0),
                reward_t=item.reward,
                terminated_t=item.terminated,
                truncated_t=item.truncated,
                split_boundary_t=next_fields["split_boundary"],
                q_hold_t=_field_float(item, "q_hold", 0.0),
                q_rebalance_t=_field_float(item, "q_rebalance", 0.0),
                q_gap_t=_field_float(item, "q_gap", 0.0),
            )

    def _update_dqn(self) -> dict[str, float | str]:
        if self.dqn_agent is None:
            return {"status": "not_configured"}
        required_replay = max(int(self.dqn_agent.config.batch_size), int(self.dqn_agent.config.warmup_steps))
        replay_size = len(self.dqn_agent.replay_buffer)
        if replay_size < required_replay:
            return {"status": "warmup", "replay_size": replay_size, "required_replay": required_replay}
        return self.dqn_agent.update()

    def _training_gate_action_selector(self, q_values: torch.Tensor) -> torch.Tensor:
        if self.dqn_agent is None:
            return torch.argmax(q_values, dim=1)
        actions = self.dqn_agent.select_action(q_values, step=self.gate_step)
        self.gate_step += int(q_values.shape[0])
        return actions

    def _update_auxiliary(self, rollout: RolloutBuffer) -> dict[str, float | str]:
        if self.auxiliary_heads is None or self.auxiliary_optimizer is None:
            return {"status": "not_configured"}
        targets = _auxiliary_targets(rollout.items, self.device)
        if not targets:
            return {"status": "no_targets"}
        states = [item.state for item in rollout.items]
        market_image = torch.as_tensor(
            np.stack([np.asarray(state["market_image"], dtype=np.float32) for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        availability_mask = torch.as_tensor(
            np.stack([np.asarray(state["availability_mask"], dtype=bool) for state in states]),
            dtype=torch.bool,
            device=self.device,
        )
        self.ppo_agent.encoder.train()
        self.auxiliary_heads.train()
        latent = self.ppo_agent.encoder(market_image)
        outputs = self.auxiliary_heads(latent)
        losses = self.auxiliary_heads.compute_loss(outputs, targets, latent, availability_mask)
        assert_finite_losses(losses, "auxiliary")
        self.auxiliary_optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_norm = clip_grad_norm_checked(
            list(self.auxiliary_heads.parameters()) + list(self.ppo_agent.encoder.parameters()),
            self.config.max_grad_norm,
            "auxiliary",
        )
        self.auxiliary_optimizer.step()
        stats = {key: float(value.detach().cpu()) for key, value in losses.items()}
        stats["grad_norm"] = float(grad_norm.detach().cpu())
        return stats

    def _maybe_checkpoint(self, epoch: int, record: Mapping[str, Any]) -> None:
        validation = _mapping(record.get("validation"))
        if not validation:
            return
        metric = float(validation.get("validation_sharpe", 0.0))
        score = _checkpoint_score(validation, len(self.history))
        if self._best_checkpoint_score is not None and score <= self._best_checkpoint_score:
            return
        self.best_validation_metric = metric
        self._best_checkpoint_score = score
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "gate_step": self.gate_step,
            "best_validation_metric": self.best_validation_metric,
            "rng_states": collect_rng_states(),
            "record": dict(record),
            "status": self.status,
        }
        self.best_checkpoint_payload = payload
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(payload)

    def _default_auxiliary_optimizer(self) -> torch.optim.Optimizer | None:
        if self.auxiliary_heads is None:
            return None
        params = list(self.auxiliary_heads.parameters()) + list(self.ppo_agent.encoder.parameters())
        return torch.optim.AdamW(params, lr=self.config.auxiliary_lr)


def _field_float(item: RolloutItem, key: str, default: float) -> float:
    for source in (item.auxiliary_labels, item.uncertainty_features, item.distributional_features):
        if key in source and source[key] is not None:
            return float(source[key])
    return float(default)


def _gate_action(item: RolloutItem) -> int:
    if item.gate_action is not None:
        return int(item.gate_action)
    return _rebalance_action(item)


def _dqn_gate_action(item: RolloutItem, dqn_agent: DQNAgent | None) -> int:
    if dqn_agent is None:
        return _gate_action(item)
    output_dim = int(getattr(dqn_agent.online_network, "output_dim", 2))
    if output_dim <= 2:
        return _gate_action(item)
    action_index = _field_float(item, "gate_action_index", float("nan"))
    if np.isfinite(action_index):
        return int(np.clip(round(action_index), 0, output_dim - 1))
    rho_values = getattr(dqn_agent.online_network, "rho_values", None)
    if rho_values is not None:
        values = np.asarray(rho_values.detach().cpu() if isinstance(rho_values, torch.Tensor) else rho_values, dtype=float)
        if values.shape[0] == output_dim:
            return int(np.argmin(np.abs(values - float(item.rebalance_intensity))))
    return int(np.clip(round(float(item.rebalance_intensity) * (output_dim - 1)), 0, output_dim - 1))


def _rebalance_action(item: RolloutItem) -> int:
    return 1 if item.rebalance_action is None else int(item.rebalance_action)


def _dqn_next_fields(
    items: Sequence[RolloutItem],
    index: int,
    last_observation: Any,
    rollout_boundary_split: bool,
) -> dict[str, Any]:
    item = items[index]
    if index + 1 < len(items):
        next_item = items[index + 1]
        return {
            "state": next_item.state,
            "decision_date": next_item.decision_date,
            "execution_date": next_item.execution_date,
            "next_valuation_date": next_item.next_valuation_date,
            "execution_price": next_item.execution_price,
            "delayed_action_execution": next_item.delayed_action_execution,
            "split_boundary": False,
        }

    state = last_observation if last_observation is not None else item.state
    split_boundary = bool(rollout_boundary_split and not (item.terminated or item.truncated))
    return {
        "state": state,
        "decision_date": _state_field(state, "decision_date", item.next_valuation_date),
        "execution_date": _state_field(state, "execution_date", item.next_valuation_date),
        "next_valuation_date": _state_field(state, "next_valuation_date", item.next_valuation_date),
        "execution_price": _state_field(state, "execution_price", item.execution_price),
        "delayed_action_execution": _state_field(
            state,
            "delayed_action_execution",
            item.delayed_action_execution,
        ),
        "split_boundary": split_boundary,
    }


def _state_field(state: Any, key: str, default: Any) -> Any:
    if isinstance(state, Mapping) and state.get(key) is not None:
        return state[key]
    return default


def _auxiliary_targets(items: Sequence[RolloutItem], device: torch.device) -> dict[str, torch.Tensor]:
    keys: set[str] = set()
    for item in items:
        keys.update(item.auxiliary_labels.keys())
    targets: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item.auxiliary_labels.get(key) for item in items]
        if any(value is None for value in values):
            continue
        targets[key] = torch.as_tensor(np.stack([np.asarray(value, dtype=np.float32) for value in values]), device=device)
    return targets


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _checkpoint_score(validation: Mapping[str, Any], global_step: int) -> tuple[float, float, float, float]:
    return (
        float(validation.get("validation_sharpe", 0.0)),
        -float(validation.get("validation_max_drawdown", 0.0)),
        -float(validation.get("validation_turnover", 0.0)),
        -float(global_step),
    )


def _positive_int(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"ERR_HYBRID_AGENT_CONFIG_INVALID: {name}")
    return result


def _optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _positive_int(name, value)


def _positive_float(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"ERR_HYBRID_AGENT_CONFIG_INVALID: {name}")
    return result


__all__ = ["HybridAgent", "HybridAgentConfig"]
