from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from src.buffers.replay_buffer import ReplayBuffer, ReplayItem


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        capacity: int = 100000,
        gamma: float = 0.99,
        n_steps: int = 3,
        per_alpha: float = 0.60,
        per_beta_start: float = 0.40,
        per_beta_end: float = 1.00,
        beta_anneal_steps: int = 100000,
        per_priority_eps: float = 1.0e-6,
        per_beta_steps: int | None = None,
        per_beta_anneal_steps: int | None = None,
    ):
        super().__init__(capacity=capacity, gamma=gamma, n_steps=n_steps)
        self.per_alpha = _positive_float("per_alpha", per_alpha)
        self.per_beta_start = _unit_interval_float("per_beta_start", per_beta_start)
        self.per_beta_end = _unit_interval_float("per_beta_end", per_beta_end)
        if self.per_beta_end < self.per_beta_start:
            raise ValueError("ERR_PER_CONFIG_INVALID: per_beta_end must be >= per_beta_start")
        anneal_steps = beta_anneal_steps
        if per_beta_steps is not None:
            anneal_steps = per_beta_steps
        if per_beta_anneal_steps is not None:
            anneal_steps = per_beta_anneal_steps
        self.beta_anneal_steps = _positive_int("beta_anneal_steps", anneal_steps)
        self.per_priority_eps = _positive_float("per_priority_eps", per_priority_eps)
        self._priorities: list[float] = []
        self._pending_priorities: deque[float | None] = deque()
        self._sample_step = 0

    @property
    def priorities(self) -> np.ndarray:
        return np.asarray(self._priorities, dtype=np.float64).copy()

    def clear(self) -> None:
        super().clear()
        self._priorities.clear()
        self._pending_priorities.clear()
        self._sample_step = 0

    def add(
        self,
        item: ReplayItem | Mapping[str, Any] | None = None,
        td_error: float | None = None,
        priority: float | None = None,
        **kwargs: Any,
    ) -> ReplayItem:
        replay_item = _replay_item(item, kwargs)
        self._append(replay_item, priority=self._resolve_priority(td_error=td_error, priority=priority))
        return replay_item

    def add_transition(
        self,
        item: ReplayItem | Mapping[str, Any] | None = None,
        td_error: float | None = None,
        priority: float | None = None,
        **kwargs: Any,
    ) -> None:
        transition = _replay_item(item, kwargs)
        self._pending.append(transition)
        self._pending_priorities.append(self._resolve_priority(td_error=td_error, priority=priority))
        if transition.terminated_t or transition.truncated_t or transition.split_boundary_t:
            self.flush_pending()
            return
        while len(self._pending) >= self.n_steps:
            self._commit_next_n_step()

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
        replace: bool = True,
        beta: float | None = None,
        step: int | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("ERR_PER_SAMPLE_SIZE")
        if not self._items:
            raise ValueError("ERR_PER_EMPTY")
        if not replace and batch_size > len(self._items):
            raise ValueError("ERR_PER_SAMPLE_SIZE")

        probabilities = self.sampling_probabilities()
        generator = rng or np.random.default_rng()
        indices = generator.choice(
            len(self._items),
            size=int(batch_size),
            replace=bool(replace),
            p=probabilities,
        )
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        beta_value = self._sample_beta(beta=beta, step=step)
        weights = self.importance_sampling_weights(indices, beta=beta_value)
        items = tuple(self._items[int(index)] for index in indices)
        batch = super().as_batch(items, device=device)
        target_device = torch.device("cpu" if device is None else device)
        batch["indices"] = torch.as_tensor(indices, dtype=torch.long, device=target_device).view(-1, 1)
        batch["is_weight"] = torch.as_tensor(weights, dtype=torch.float32, device=target_device).view(-1, 1)
        selected_probabilities = probabilities[indices]
        selected_priorities = self.priorities[indices]
        return {
            "items": items,
            "indices": indices,
            "priorities": selected_priorities,
            "sampling_probability": selected_probabilities,
            "sampling_probabilities": selected_probabilities,
            "probabilities": selected_probabilities,
            "is_weight": weights,
            "is_weights": weights,
            "weights": weights,
            "beta": beta_value,
            "batch": batch,
        }

    def sampling_probabilities(self) -> np.ndarray:
        priorities = self.priorities
        if priorities.size == 0:
            raise ValueError("ERR_PER_EMPTY")
        total = float(priorities.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("ERR_PER_PRIORITY_INVALID")
        return priorities / total

    def sample_probabilities(self) -> np.ndarray:
        return self.sampling_probabilities()

    def importance_sampling_weights(self, indices: Sequence[int] | np.ndarray, beta: float | None = None) -> np.ndarray:
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        if index_array.size == 0:
            raise ValueError("ERR_PER_SAMPLE_SIZE")
        if np.any(index_array < 0) or np.any(index_array >= len(self._items)):
            raise IndexError("ERR_PER_INDEX_OUT_OF_RANGE")
        beta_value = self.per_beta_start if beta is None else _unit_interval_float("beta", beta)
        probabilities = self.sampling_probabilities()[index_array]
        weights = (len(self._items) * probabilities) ** (-beta_value)
        max_weight = float(weights.max())
        if not np.isfinite(max_weight) or max_weight <= 0.0:
            raise ValueError("ERR_PER_WEIGHT_INVALID")
        return (weights / max_weight).astype(np.float32)

    def update_priorities(
        self,
        indices: Sequence[int] | np.ndarray,
        td_errors: Sequence[float] | np.ndarray | None = None,
        priorities: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        if index_array.size == 0:
            raise ValueError("ERR_PER_UPDATE_EMPTY")
        if np.any(index_array < 0) or np.any(index_array >= len(self._priorities)):
            raise IndexError("ERR_PER_INDEX_OUT_OF_RANGE")
        if td_errors is not None and priorities is not None:
            raise ValueError("ERR_PER_PRIORITY_SOURCE_AMBIGUOUS")
        if priorities is None:
            if td_errors is None:
                raise ValueError("ERR_PER_PRIORITY_SOURCE_REQUIRED")
            values = np.asarray(
                self.compute_priority(td_errors, self.per_alpha, self.per_priority_eps),
                dtype=np.float64,
            ).reshape(-1)
        else:
            values = _priority_array(priorities)
        if values.shape != index_array.shape:
            raise ValueError("ERR_PER_PRIORITY_SHAPE")
        for index, value in zip(index_array, values, strict=True):
            self._priorities[int(index)] = float(value)

    def beta_at_step(self, step: int) -> float:
        step_value = max(int(step), 0)
        fraction = min(step_value / self.beta_anneal_steps, 1.0)
        return self.per_beta_start + fraction * (self.per_beta_end - self.per_beta_start)

    @staticmethod
    def compute_priority(
        td_error: float | Sequence[float] | np.ndarray,
        per_alpha: float = 0.60,
        per_priority_eps: float = 1.0e-6,
    ) -> float | np.ndarray:
        alpha = _positive_float("per_alpha", per_alpha)
        eps = _positive_float("per_priority_eps", per_priority_eps)
        raw_errors = np.asarray(td_error, dtype=np.float64)
        scalar_input = raw_errors.ndim == 0
        errors = raw_errors.reshape(-1)
        if not np.isfinite(errors).all():
            raise ValueError("ERR_PER_TD_ERROR_NON_FINITE")
        priorities = (np.abs(errors) + eps) ** alpha
        if not np.isfinite(priorities).all() or np.any(priorities <= 0.0):
            raise ValueError("ERR_PER_PRIORITY_INVALID")
        priorities = np.asarray(priorities, dtype=np.float64)
        if scalar_input:
            return float(priorities[0])
        return priorities

    compute_priorities = compute_priority

    def _append(self, item: ReplayItem, priority: float | None = None) -> None:
        if len(self._items) >= self.capacity:
            self._items.pop(0)
            self._priorities.pop(0)
        self._items.append(item)
        self._priorities.append(self._default_priority() if priority is None else _priority_value(priority))

    def _commit_next_n_step(self) -> None:
        window = self._n_step_window()
        base = window[0]
        last = window[-1]
        reward = 0.0
        for step, item in enumerate(window):
            reward += (self.gamma**step) * item.reward_t
            if item.terminated_t or item.truncated_t or item.split_boundary_t:
                break
        payload = base.to_dict()
        boundary = last.terminated_t or last.truncated_t or last.split_boundary_t
        payload.update(
            {
                "state_tp1": last.state_tp1,
                "decision_date_next": last.decision_date_next,
                "execution_date_next": last.execution_date_next,
                "next_valuation_date_next": last.next_valuation_date_next,
                "execution_price_next": last.execution_price_next,
                "delayed_action_execution_next": last.delayed_action_execution_next,
                "reward_t": reward,
                "terminated_t": last.terminated_t,
                "truncated_t": last.truncated_t or last.split_boundary_t,
                "split_boundary_t": last.split_boundary_t,
                "n_steps": len(window),
                "discount": 0.0 if boundary else self.gamma ** len(window),
            }
        )
        priority = self._pending_priorities[0] if self._pending_priorities else None
        self._append(ReplayItem(**payload), priority=priority)
        self._pending.popleft()
        if self._pending_priorities:
            self._pending_priorities.popleft()

    def _resolve_priority(self, td_error: float | None, priority: float | None) -> float | None:
        if td_error is not None and priority is not None:
            raise ValueError("ERR_PER_PRIORITY_SOURCE_AMBIGUOUS")
        if priority is not None:
            return _priority_value(priority)
        if td_error is None:
            return None
        return float(np.asarray(self.compute_priority(td_error, self.per_alpha, self.per_priority_eps)).reshape(-1)[0])

    def _default_priority(self) -> float:
        if self._priorities:
            return float(max(self._priorities))
        return float(np.asarray(self.compute_priority(1.0, self.per_alpha, self.per_priority_eps)).reshape(-1)[0])

    def _sample_beta(self, beta: float | None, step: int | None) -> float:
        if beta is not None:
            return _unit_interval_float("beta", beta)
        if step is None:
            beta_value = self.beta_at_step(self._sample_step)
            self._sample_step += 1
            return beta_value
        return self.beta_at_step(step)


def _replay_item(item: ReplayItem | Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> ReplayItem:
    if isinstance(item, ReplayItem) and not kwargs:
        return item
    if item is None:
        payload: dict[str, Any] = {}
    elif isinstance(item, ReplayItem):
        payload = item.to_dict()
    elif isinstance(item, Mapping):
        payload = dict(item)
    else:
        raise TypeError("ERR_REPLAY_ITEM_TYPE")
    payload.update(kwargs)
    return ReplayItem(**payload)


def _priority_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError("ERR_PER_PRIORITY_INVALID")
    return result


def _priority_value(value: float) -> float:
    return float(_priority_array([value])[0])


def _positive_float(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"ERR_PER_CONFIG_INVALID: {name}")
    return result


def _unit_interval_float(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"ERR_PER_CONFIG_INVALID: {name}")
    return result


def _positive_int(name: str, value: int) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"ERR_PER_CONFIG_INVALID: {name}")
    return result


__all__ = ["PrioritizedReplayBuffer"]
