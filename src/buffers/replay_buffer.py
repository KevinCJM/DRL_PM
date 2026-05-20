from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass
class ReplayItem:
    state_t: Any
    state_tp1: Any
    decision_date_t: pd.Timestamp
    execution_date_t: pd.Timestamp
    next_valuation_date_t: pd.Timestamp
    decision_date_next: pd.Timestamp
    execution_date_next: pd.Timestamp
    next_valuation_date_next: pd.Timestamp
    execution_price_t: str
    delayed_action_execution_t: bool
    candidate_weights_t: np.ndarray
    executed_weights_t: np.ndarray
    gate_action_t: int
    rebalance_action_t: int
    rebalance_intensity_t: float
    estimated_turnover_t: float
    realized_turnover_t: float
    estimated_cost_t: float
    realized_cost_t: float
    reward_t: float
    terminated_t: bool
    truncated_t: bool
    q_hold_t: float
    q_rebalance_t: float
    q_gap_t: float | None = None
    q_reference_t: float | None = None
    q_selected_t: float | None = None
    q_selected_minus_reference_t: float | None = None
    invalid_action_t: bool = False
    bootstrap_mask_t: float = 1.0
    next_state_source_t: str = "env"
    execution_price_next: str | None = None
    delayed_action_execution_next: bool | None = None
    split_boundary_t: bool = False
    n_steps: int = 1
    discount: float = 1.0

    def __post_init__(self) -> None:
        self.decision_date_t = _timestamp("decision_date_t", self.decision_date_t)
        self.execution_date_t = _timestamp("execution_date_t", self.execution_date_t)
        self.next_valuation_date_t = _timestamp("next_valuation_date_t", self.next_valuation_date_t)
        self.decision_date_next = _timestamp("decision_date_next", self.decision_date_next)
        self.execution_date_next = _timestamp("execution_date_next", self.execution_date_next)
        self.next_valuation_date_next = _timestamp("next_valuation_date_next", self.next_valuation_date_next)
        if self.execution_date_t < self.decision_date_t or self.next_valuation_date_t < self.execution_date_t:
            raise ValueError("ERR_REPLAY_ITEM_DATE_ORDER: t")
        if self.execution_date_next < self.decision_date_next or self.next_valuation_date_next < self.execution_date_next:
            raise ValueError("ERR_REPLAY_ITEM_DATE_ORDER: next")

        self.execution_price_t = _non_empty_string("execution_price_t", self.execution_price_t)
        self.execution_price_next = (
            self.execution_price_t
            if self.execution_price_next is None
            else _non_empty_string("execution_price_next", self.execution_price_next)
        )
        self.delayed_action_execution_t = bool(self.delayed_action_execution_t)
        self.delayed_action_execution_next = (
            self.delayed_action_execution_t
            if self.delayed_action_execution_next is None
            else bool(self.delayed_action_execution_next)
        )
        self.candidate_weights_t = _weights("candidate_weights_t", self.candidate_weights_t)
        self.executed_weights_t = _weights("executed_weights_t", self.executed_weights_t, self.candidate_weights_t.shape)
        self.gate_action_t = _non_negative_int("gate_action_t", self.gate_action_t)
        self.rebalance_action_t = _non_negative_int("rebalance_action_t", self.rebalance_action_t)
        self.rebalance_intensity_t = _bounded_float("rebalance_intensity_t", self.rebalance_intensity_t, 0.0, 1.0)
        self.estimated_turnover_t = _non_negative_float("estimated_turnover_t", self.estimated_turnover_t)
        self.realized_turnover_t = _non_negative_float("realized_turnover_t", self.realized_turnover_t)
        self.estimated_cost_t = _non_negative_float("estimated_cost_t", self.estimated_cost_t)
        self.realized_cost_t = _non_negative_float("realized_cost_t", self.realized_cost_t)
        self.reward_t = _finite_float("reward_t", self.reward_t)
        self.terminated_t = bool(self.terminated_t)
        self.truncated_t = bool(self.truncated_t)
        self.split_boundary_t = bool(self.split_boundary_t)
        if self.split_boundary_t:
            self.truncated_t = True
        self.q_hold_t = _finite_float("q_hold_t", self.q_hold_t)
        self.q_rebalance_t = _finite_float("q_rebalance_t", self.q_rebalance_t)
        self.q_gap_t = (
            self.q_rebalance_t - self.q_hold_t
            if self.q_gap_t is None
            else _finite_float("q_gap_t", self.q_gap_t)
        )
        self.q_reference_t = (
            self.q_hold_t
            if self.q_reference_t is None
            else _finite_float("q_reference_t", self.q_reference_t)
        )
        self.q_selected_t = (
            self.q_rebalance_t
            if self.q_selected_t is None
            else _finite_float("q_selected_t", self.q_selected_t)
        )
        self.q_selected_minus_reference_t = (
            self.q_selected_t - self.q_reference_t
            if self.q_selected_minus_reference_t is None
            else _finite_float("q_selected_minus_reference_t", self.q_selected_minus_reference_t)
        )
        self.invalid_action_t = bool(self.invalid_action_t)
        self.bootstrap_mask_t = _bounded_float("bootstrap_mask_t", self.bootstrap_mask_t, 0.0, 1.0)
        self.next_state_source_t = _non_empty_string("next_state_source_t", self.next_state_source_t)
        self.n_steps = _positive_int("n_steps", self.n_steps)
        self.discount = _non_negative_float("discount", self.discount)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_t": self.state_t,
            "state_tp1": self.state_tp1,
            "decision_date_t": self.decision_date_t,
            "execution_date_t": self.execution_date_t,
            "next_valuation_date_t": self.next_valuation_date_t,
            "decision_date_next": self.decision_date_next,
            "execution_date_next": self.execution_date_next,
            "next_valuation_date_next": self.next_valuation_date_next,
            "execution_price_t": self.execution_price_t,
            "delayed_action_execution_t": self.delayed_action_execution_t,
            "candidate_weights_t": self.candidate_weights_t,
            "executed_weights_t": self.executed_weights_t,
            "gate_action_t": self.gate_action_t,
            "rebalance_action_t": self.rebalance_action_t,
            "rebalance_intensity_t": self.rebalance_intensity_t,
            "estimated_turnover_t": self.estimated_turnover_t,
            "realized_turnover_t": self.realized_turnover_t,
            "estimated_cost_t": self.estimated_cost_t,
            "realized_cost_t": self.realized_cost_t,
            "reward_t": self.reward_t,
            "terminated_t": self.terminated_t,
            "truncated_t": self.truncated_t,
            "q_hold_t": self.q_hold_t,
            "q_rebalance_t": self.q_rebalance_t,
            "q_gap_t": self.q_gap_t,
            "q_reference_t": self.q_reference_t,
            "q_selected_t": self.q_selected_t,
            "q_selected_minus_reference_t": self.q_selected_minus_reference_t,
            "invalid_action_t": self.invalid_action_t,
            "bootstrap_mask_t": self.bootstrap_mask_t,
            "next_state_source_t": self.next_state_source_t,
            "execution_price_next": self.execution_price_next,
            "delayed_action_execution_next": self.delayed_action_execution_next,
            "split_boundary_t": self.split_boundary_t,
            "n_steps": self.n_steps,
            "discount": self.discount,
        }


class ReplayBuffer:
    def __init__(self, capacity: int = 100000, gamma: float = 0.99, n_steps: int = 3):
        self.capacity = _positive_int("capacity", capacity)
        self.gamma = _unit_interval_float("gamma", gamma)
        self.n_steps = _positive_int("n_steps", n_steps)
        self._items: list[ReplayItem] = []
        self._pending: deque[ReplayItem] = deque()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[ReplayItem, ...]:
        return tuple(self._items)

    @property
    def pending_items(self) -> tuple[ReplayItem, ...]:
        return tuple(self._pending)

    def clear(self) -> None:
        self._items.clear()
        self._pending.clear()

    def add(self, item: ReplayItem | Mapping[str, Any] | None = None, **kwargs: Any) -> ReplayItem:
        replay_item = _replay_item(item, kwargs)
        self._append(replay_item)
        return replay_item

    def add_transition(self, item: ReplayItem | Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        transition = _replay_item(item, kwargs)
        self._pending.append(transition)
        if transition.terminated_t or transition.truncated_t or transition.split_boundary_t:
            self.flush_pending()
            return
        while len(self._pending) >= self.n_steps:
            self._commit_next_n_step()

    def flush_pending(self) -> None:
        while self._pending:
            self._commit_next_n_step()

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
        replace: bool = False,
    ) -> tuple[ReplayItem, ...]:
        if batch_size <= 0:
            raise ValueError("ERR_REPLAY_BUFFER_SAMPLE_SIZE")
        if not self._items:
            raise ValueError("ERR_REPLAY_BUFFER_EMPTY")
        if not replace and batch_size > len(self._items):
            raise ValueError("ERR_REPLAY_BUFFER_SAMPLE_SIZE")
        generator = rng or np.random.default_rng()
        indices = generator.choice(len(self._items), size=int(batch_size), replace=bool(replace))
        return tuple(self._items[int(index)] for index in np.asarray(indices).reshape(-1))

    def as_batch(
        self,
        items: Sequence[ReplayItem] | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        batch_items = list(self._items if items is None else items)
        if not batch_items:
            raise ValueError("ERR_REPLAY_BUFFER_EMPTY")
        target_device = torch.device("cpu" if device is None else device)
        return {
            "state_t": [item.state_t for item in batch_items],
            "state_tp1": [item.state_tp1 for item in batch_items],
            "candidate_weights_t": _tensor_stack([item.candidate_weights_t for item in batch_items], target_device),
            "executed_weights_t": _tensor_stack([item.executed_weights_t for item in batch_items], target_device),
            "gate_action_t": _long_column([item.gate_action_t for item in batch_items], target_device),
            "rebalance_action_t": _long_column([item.rebalance_action_t for item in batch_items], target_device),
            "rebalance_intensity_t": _tensor_column([item.rebalance_intensity_t for item in batch_items], target_device),
            "estimated_turnover_t": _tensor_column([item.estimated_turnover_t for item in batch_items], target_device),
            "realized_turnover_t": _tensor_column([item.realized_turnover_t for item in batch_items], target_device),
            "estimated_cost_t": _tensor_column([item.estimated_cost_t for item in batch_items], target_device),
            "realized_cost_t": _tensor_column([item.realized_cost_t for item in batch_items], target_device),
            "reward_t": _tensor_column([item.reward_t for item in batch_items], target_device),
            "terminated_t": _bool_column([item.terminated_t for item in batch_items], target_device),
            "truncated_t": _bool_column([item.truncated_t for item in batch_items], target_device),
            "q_hold_t": _tensor_column([item.q_hold_t for item in batch_items], target_device),
            "q_rebalance_t": _tensor_column([item.q_rebalance_t for item in batch_items], target_device),
            "q_gap_t": _tensor_column([item.q_gap_t for item in batch_items], target_device),
            "q_reference_t": _tensor_column([item.q_reference_t for item in batch_items], target_device),
            "q_selected_t": _tensor_column([item.q_selected_t for item in batch_items], target_device),
            "q_selected_minus_reference_t": _tensor_column(
                [item.q_selected_minus_reference_t for item in batch_items],
                target_device,
            ),
            "invalid_action_t": _bool_column([item.invalid_action_t for item in batch_items], target_device),
            "bootstrap_mask_t": _tensor_column([item.bootstrap_mask_t for item in batch_items], target_device),
            "next_state_source_t": [item.next_state_source_t for item in batch_items],
            "n_steps": _long_column([item.n_steps for item in batch_items], target_device),
            "discount": _tensor_column([item.discount for item in batch_items], target_device),
            "decision_date_t": [item.decision_date_t for item in batch_items],
            "execution_date_t": [item.execution_date_t for item in batch_items],
            "next_valuation_date_t": [item.next_valuation_date_t for item in batch_items],
            "decision_date_next": [item.decision_date_next for item in batch_items],
            "execution_date_next": [item.execution_date_next for item in batch_items],
            "next_valuation_date_next": [item.next_valuation_date_next for item in batch_items],
            "execution_price_t": [item.execution_price_t for item in batch_items],
            "delayed_action_execution_t": [item.delayed_action_execution_t for item in batch_items],
            "execution_price_next": [item.execution_price_next for item in batch_items],
            "delayed_action_execution_next": [item.delayed_action_execution_next for item in batch_items],
            "split_boundary_t": [item.split_boundary_t for item in batch_items],
        }

    def _append(self, item: ReplayItem) -> None:
        if len(self._items) >= self.capacity:
            self._items.pop(0)
        self._items.append(item)

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
        self._append(ReplayItem(**payload))
        self._pending.popleft()

    def _n_step_window(self) -> list[ReplayItem]:
        window: list[ReplayItem] = []
        for item in self._pending:
            window.append(item)
            if len(window) >= self.n_steps or item.terminated_t or item.truncated_t or item.split_boundary_t:
                break
        return window


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


def _timestamp(name: str, value: Any) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"ERR_REPLAY_ITEM_DATE_INVALID: {name}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"ERR_REPLAY_ITEM_DATE_INVALID: {name}")
    return timestamp


def _non_empty_string(name: str, value: Any) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"ERR_REPLAY_ITEM_FIELD_INVALID: {name}")
    return result


def _weights(name: str, value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"ERR_REPLAY_ITEM_SHAPE: {name}")
    if shape is not None and tuple(result.shape) != tuple(shape):
        raise ValueError(f"ERR_REPLAY_ITEM_SHAPE: {name}")
    if not np.isfinite(result).all():
        raise ValueError(f"ERR_REPLAY_ITEM_NON_FINITE: {name}")
    return result.copy()


def _finite_float(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"ERR_REPLAY_ITEM_NON_FINITE: {name}")
    return result


def _non_negative_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"ERR_REPLAY_ITEM_FIELD_INVALID: {name}")
    return result


def _bounded_float(name: str, value: Any, lower: float, upper: float) -> float:
    result = _finite_float(name, value)
    if result < lower or result > upper:
        raise ValueError(f"ERR_REPLAY_ITEM_FIELD_INVALID: {name}")
    return result


def _non_negative_int(name: str, value: Any) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"ERR_REPLAY_ITEM_FIELD_INVALID: {name}")
    return result


def _positive_int(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"ERR_REPLAY_BUFFER_CONFIG_INVALID: {name}")
    return result


def _unit_interval_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"ERR_REPLAY_BUFFER_CONFIG_INVALID: {name}")
    return result


def _tensor_stack(values: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)


def _tensor_column(values: Sequence[float | int | None], device: torch.device) -> torch.Tensor:
    array = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float32)
    return torch.as_tensor(array, dtype=torch.float32, device=device).view(-1, 1)


def _long_column(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.int64), dtype=torch.long, device=device).view(-1, 1)


def _bool_column(values: Sequence[bool], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=bool), dtype=torch.bool, device=device).view(-1, 1)


__all__ = ["ReplayBuffer", "ReplayItem"]
