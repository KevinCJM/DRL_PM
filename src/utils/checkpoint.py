from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.agents.dqn_agent import DQNAgent
from src.agents.hybrid_agent import HybridAgent
from src.agents.ppo_agent import PPOAgent
from src.buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from src.buffers.replay_buffer import ReplayBuffer, ReplayItem
from src.utils.seed import collect_rng_states, restore_rng_states


CHECKPOINT_SCHEMA_VERSION = "1.0"
REQUIRED_CHECKPOINT_KEYS = (
    "schema_version",
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


def build_checkpoint_payload(
    agent: HybridAgent | PPOAgent,
    *,
    epoch: int,
    global_step: int | None = None,
    best_validation_metric: float | None = None,
    resolved_config: Mapping[str, Any] | None = None,
    schedulers: Mapping[str, Any] | Sequence[Any] | None = None,
    grad_scaler: Any | None = None,
    env: Any | None = None,
    include_replay_buffer: bool = True,
) -> dict[str, Any]:
    hybrid_agent = agent if isinstance(agent, HybridAgent) else None
    ppo_agent = hybrid_agent.ppo_agent if hybrid_agent is not None else agent
    dqn_agent = hybrid_agent.dqn_agent if hybrid_agent is not None else None
    auxiliary_heads = hybrid_agent.auxiliary_heads if hybrid_agent is not None else None
    policy_model = _policy_model(hybrid_agent, ppo_agent)
    target_policy_model = getattr(hybrid_agent, "target_policy_model", None) if hybrid_agent is not None else None

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "policy_model_state": _module_state(policy_model),
        "target_policy_model_state": _module_state(target_policy_model),
        "encoder_state": _module_state(ppo_agent.encoder),
        "ppo_actor_state": _module_state(ppo_agent.actor),
        "ppo_critic_state": _module_state(ppo_agent.critic),
        "dqn_gate_state": _module_state(dqn_agent.online_network) if dqn_agent is not None else None,
        "dqn_target_network_state": _module_state(dqn_agent.target_network) if dqn_agent is not None else None,
        "auxiliary_head_states": _module_state(auxiliary_heads),
        "optimizer_states": _optimizer_states(hybrid_agent, ppo_agent, dqn_agent),
        "scheduler_states": _scheduler_states(schedulers),
        "amp_grad_scaler_state": _grad_scaler_state(grad_scaler, ppo_agent.device),
        "replay_buffer_state": _replay_buffer_state(dqn_agent.replay_buffer)
        if dqn_agent is not None and include_replay_buffer
        else None,
        "epoch": int(epoch),
        "global_step": _global_step(global_step, hybrid_agent, dqn_agent),
        "gate_step": 0 if hybrid_agent is None else int(getattr(hybrid_agent, "gate_step", 0)),
        "best_validation_metric": _best_metric(best_validation_metric, hybrid_agent),
        "best_checkpoint_score": None
        if hybrid_agent is None or getattr(hybrid_agent, "_best_checkpoint_score", None) is None
        else list(hybrid_agent._best_checkpoint_score),
        "training_history": [] if hybrid_agent is None else _safe_value(list(hybrid_agent.history)),
        "hybrid_status": None if hybrid_agent is None else str(hybrid_agent.status),
        "rng_states": collect_rng_states(env=env),
        "resolved_config": _safe_value(dict(resolved_config or {})),
    }
    return validate_checkpoint_payload(payload)


def save_checkpoint(
    checkpoint: Mapping[str, Any] | HybridAgent | PPOAgent,
    path: str | Path,
    **payload_kwargs: Any,
) -> Path:
    payload = (
        build_checkpoint_payload(checkpoint, **payload_kwargs)
        if isinstance(checkpoint, (HybridAgent, PPOAgent))
        else validate_checkpoint_payload(dict(checkpoint))
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            torch.save(_safe_value(payload), fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)
        return target
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def load_checkpoint(
    path: str | Path,
    device: torch.device | str | None = "cpu",
    *,
    agent: HybridAgent | PPOAgent | None = None,
    schedulers: Mapping[str, Any] | Sequence[Any] | None = None,
    grad_scaler: Any | None = None,
    env: Any | None = None,
    restore_rng_state: bool = True,
    safe_torch_load: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    map_location = None if device is None else torch.device(device)
    payload = _torch_load(Path(path), map_location=map_location, safe_torch_load=safe_torch_load)
    payload = validate_checkpoint_payload(dict(payload))
    if agent is not None:
        load_checkpoint_payload(
            agent,
            payload,
            schedulers=schedulers,
            grad_scaler=grad_scaler,
            env=env,
            restore_rng_state=restore_rng_state,
            strict=strict,
        )
    elif restore_rng_state:
        restore_rng_states(payload["rng_states"], env=env)
    return payload


def load_checkpoint_payload(
    agent: HybridAgent | PPOAgent,
    payload: Mapping[str, Any],
    *,
    schedulers: Mapping[str, Any] | Sequence[Any] | None = None,
    grad_scaler: Any | None = None,
    env: Any | None = None,
    restore_rng_state: bool = True,
    strict: bool = True,
) -> None:
    payload = validate_checkpoint_payload(dict(payload))
    hybrid_agent = agent if isinstance(agent, HybridAgent) else None
    ppo_agent = hybrid_agent.ppo_agent if hybrid_agent is not None else agent
    dqn_agent = hybrid_agent.dqn_agent if hybrid_agent is not None else None
    policy_model = _policy_model(hybrid_agent, ppo_agent)
    target_policy_model = getattr(hybrid_agent, "target_policy_model", None) if hybrid_agent is not None else None

    if policy_model is not None and payload.get("policy_model_state") is not None:
        policy_model.load_state_dict(payload["policy_model_state"], strict=strict)
    if target_policy_model is not None and payload.get("target_policy_model_state") is not None:
        target_policy_model.load_state_dict(payload["target_policy_model_state"], strict=strict)
    ppo_agent.encoder.load_state_dict(payload["encoder_state"], strict=strict)
    ppo_agent.actor.load_state_dict(payload["ppo_actor_state"], strict=strict)
    ppo_agent.critic.load_state_dict(payload["ppo_critic_state"], strict=strict)
    optimizer_states = payload["optimizer_states"]
    if optimizer_states.get("ppo") is not None:
        ppo_agent.optimizer.load_state_dict(optimizer_states["ppo"])

    if dqn_agent is not None:
        if payload["dqn_gate_state"] is not None:
            dqn_agent.online_network.load_state_dict(payload["dqn_gate_state"], strict=strict)
        if payload["dqn_target_network_state"] is not None:
            dqn_agent.target_network.load_state_dict(payload["dqn_target_network_state"], strict=strict)
        if optimizer_states.get("dqn") is not None:
            dqn_agent.optimizer.load_state_dict(optimizer_states["dqn"])
        if payload["replay_buffer_state"] is not None:
            restore_replay_buffer_state(dqn_agent.replay_buffer, payload["replay_buffer_state"])
        dqn_agent.global_step = int(payload["global_step"])

    if hybrid_agent is not None:
        if hybrid_agent.auxiliary_heads is not None and payload["auxiliary_head_states"] is not None:
            hybrid_agent.auxiliary_heads.load_state_dict(payload["auxiliary_head_states"], strict=strict)
        if hybrid_agent.auxiliary_optimizer is not None and optimizer_states.get("auxiliary") is not None:
            hybrid_agent.auxiliary_optimizer.load_state_dict(optimizer_states["auxiliary"])
        if payload["best_validation_metric"] is not None:
            hybrid_agent.best_validation_metric = float(payload["best_validation_metric"])
        hybrid_agent.start_epoch = int(payload["epoch"]) + 1
        hybrid_agent.last_epoch = int(payload["epoch"])
        hybrid_agent.global_step = int(payload["global_step"])
        hybrid_agent.gate_step = int(payload.get("gate_step", 0))
        history = payload.get("training_history")
        if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
            hybrid_agent.history = [dict(item) if isinstance(item, Mapping) else {"value": item} for item in history]
        best_score = payload.get("best_checkpoint_score")
        if isinstance(best_score, Sequence) and not isinstance(best_score, (str, bytes)) and len(best_score) == 4:
            hybrid_agent._best_checkpoint_score = tuple(float(value) for value in best_score)
        hybrid_agent.status = str(payload.get("hybrid_status") or hybrid_agent.status)
    _load_scheduler_states(schedulers, payload["scheduler_states"])
    if grad_scaler is not None and payload["amp_grad_scaler_state"] is not None:
        grad_scaler.load_state_dict(payload["amp_grad_scaler_state"])
    if restore_rng_state:
        restore_rng_states(payload["rng_states"], env=env)


def validate_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"ERR_CHECKPOINT_SCHEMA_MISSING: {','.join(missing)}")
    result = dict(payload)
    result["schema_version"] = str(result["schema_version"])
    result["epoch"] = int(result["epoch"])
    result["global_step"] = int(result["global_step"])
    if result["best_validation_metric"] is not None:
        result["best_validation_metric"] = float(result["best_validation_metric"])
    return result


def restore_replay_buffer_state(buffer: ReplayBuffer | PrioritizedReplayBuffer, state: Mapping[str, Any]) -> None:
    buffer.clear()
    buffer.capacity = int(state["capacity"])
    buffer.gamma = float(state["gamma"])
    buffer.n_steps = int(state["n_steps"])
    for item in state.get("items", []):
        buffer.add(ReplayItem(**_unsafe_replay_item(item)))
    for item in state.get("pending_items", []):
        buffer._pending.append(ReplayItem(**_unsafe_replay_item(item)))
    if isinstance(buffer, PrioritizedReplayBuffer) and state.get("buffer_type") == "PrioritizedReplayBuffer":
        buffer._priorities = [float(value) for value in state.get("priorities", [])]
        buffer._pending_priorities.clear()
        for value in state.get("pending_priorities", []):
            buffer._pending_priorities.append(None if value is None else float(value))
        buffer._sample_step = int(state.get("sample_step", 0))


def _torch_load(path: Path, map_location: torch.device | None, safe_torch_load: bool) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=bool(safe_torch_load))
    except TypeError:
        if safe_torch_load:
            raise RuntimeError("ERR_CHECKPOINT_SAFE_LOAD_UNSUPPORTED")
        return torch.load(path, map_location=map_location)


def _module_state(module: torch.nn.Module | None) -> dict[str, torch.Tensor] | None:
    if module is None:
        return None
    return {str(key): value.detach().cpu() for key, value in module.state_dict().items()}


def _policy_model(hybrid_agent: HybridAgent | None, ppo_agent: PPOAgent) -> torch.nn.Module | None:
    if hybrid_agent is not None and isinstance(getattr(hybrid_agent, "policy_model", None), torch.nn.Module):
        return hybrid_agent.policy_model
    policy_model = getattr(ppo_agent, "policy_model", None)
    return policy_model if isinstance(policy_model, torch.nn.Module) else None


def _optimizer_states(
    hybrid_agent: HybridAgent | None,
    ppo_agent: PPOAgent,
    dqn_agent: DQNAgent | None,
) -> dict[str, Any]:
    return {
        "ppo": _safe_value(ppo_agent.optimizer.state_dict()),
        "dqn": None if dqn_agent is None else _safe_value(dqn_agent.optimizer.state_dict()),
        "auxiliary": None
        if hybrid_agent is None or hybrid_agent.auxiliary_optimizer is None
        else _safe_value(hybrid_agent.auxiliary_optimizer.state_dict()),
    }


def _scheduler_states(schedulers: Mapping[str, Any] | Sequence[Any] | None) -> dict[str, Any]:
    if schedulers is None:
        return {}
    if isinstance(schedulers, Mapping):
        return {
            str(name): _safe_value(scheduler.state_dict() if hasattr(scheduler, "state_dict") else scheduler)
            for name, scheduler in schedulers.items()
        }
    return {
        str(index): _safe_value(scheduler.state_dict() if hasattr(scheduler, "state_dict") else scheduler)
        for index, scheduler in enumerate(schedulers)
    }


def _load_scheduler_states(schedulers: Mapping[str, Any] | Sequence[Any] | None, states: Mapping[str, Any]) -> None:
    if not schedulers or not states:
        return
    if isinstance(schedulers, Mapping):
        for name, scheduler in schedulers.items():
            if str(name) in states and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(states[str(name)])
        return
    for index, scheduler in enumerate(schedulers):
        key = str(index)
        if key in states and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(states[key])


def _grad_scaler_state(grad_scaler: Any | None, device: torch.device) -> dict[str, Any] | None:
    if grad_scaler is None or device.type != "cuda":
        return None
    if not hasattr(grad_scaler, "state_dict"):
        raise TypeError("ERR_CHECKPOINT_GRAD_SCALER_INVALID")
    return _safe_value(grad_scaler.state_dict())


def _replay_buffer_state(buffer: ReplayBuffer | PrioritizedReplayBuffer) -> dict[str, Any]:
    state = {
        "buffer_type": type(buffer).__name__,
        "capacity": buffer.capacity,
        "gamma": buffer.gamma,
        "n_steps": buffer.n_steps,
        "items": [_safe_value(item.to_dict()) for item in buffer.items],
        "pending_items": [_safe_value(item.to_dict()) for item in buffer.pending_items],
    }
    if isinstance(buffer, PrioritizedReplayBuffer):
        state.update(
            {
                "per_alpha": buffer.per_alpha,
                "per_beta_start": buffer.per_beta_start,
                "per_beta_end": buffer.per_beta_end,
                "beta_anneal_steps": buffer.beta_anneal_steps,
                "per_priority_eps": buffer.per_priority_eps,
                "priorities": buffer.priorities.tolist(),
                "pending_priorities": list(buffer._pending_priorities),
                "sample_step": buffer._sample_step,
            }
        )
    return _safe_value(state)


def _unsafe_replay_item(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    for key in ("candidate_weights_t", "executed_weights_t"):
        if isinstance(payload.get(key), torch.Tensor):
            payload[key] = payload[key].detach().cpu().numpy()
    return payload


def _global_step(value: int | None, hybrid_agent: HybridAgent | None, dqn_agent: DQNAgent | None) -> int:
    if value is not None:
        return int(value)
    if hybrid_agent is not None:
        return int(getattr(hybrid_agent, "global_step", len(hybrid_agent.history)))
    if dqn_agent is not None:
        return int(dqn_agent.global_step)
    return 0


def _best_metric(value: float | None, hybrid_agent: HybridAgent | None) -> float | None:
    if value is not None:
        return float(value)
    if hybrid_agent is not None and hybrid_agent.best_validation_metric is not None:
        return float(hybrid_agent.best_validation_metric)
    return None


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_):
            return torch.as_tensor(value.copy())
        return [_safe_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _safe_value(value.item())
    if hasattr(value, "isoformat") and type(value).__module__.startswith(("pandas", "datetime")):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    return str(value)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "REQUIRED_CHECKPOINT_KEYS",
    "build_checkpoint_payload",
    "collect_rng_states",
    "load_checkpoint",
    "load_checkpoint_payload",
    "restore_replay_buffer_state",
    "save_checkpoint",
    "validate_checkpoint_payload",
]
