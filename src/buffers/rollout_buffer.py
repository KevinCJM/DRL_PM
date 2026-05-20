from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass
class RolloutItem:
    decision_date: pd.Timestamp
    execution_date: pd.Timestamp
    next_valuation_date: pd.Timestamp
    execution_price: str
    delayed_action_execution: bool
    state: Any
    candidate_weights: np.ndarray
    log_prob: float
    value: float
    reward: float
    terminated: bool
    truncated: bool
    executed_weights: np.ndarray | None = None
    gate_action: int | None = None
    rebalance_action: int | None = None
    rebalance_intensity: float = 1.0
    decision_value: float | None = None
    advantage: float | None = None
    return_value: float | None = None
    auxiliary_labels: Mapping[str, Any] = field(default_factory=dict)
    preference_vector: np.ndarray | None = None
    uncertainty_features: Mapping[str, Any] = field(default_factory=dict)
    distributional_features: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.decision_date = _timestamp("decision_date", self.decision_date)
        self.execution_date = _timestamp("execution_date", self.execution_date)
        self.next_valuation_date = _timestamp("next_valuation_date", self.next_valuation_date)
        if self.execution_date < self.decision_date or self.next_valuation_date < self.execution_date:
            raise ValueError("ERR_ROLLOUT_ITEM_DATE_ORDER")
        self.execution_price = _non_empty_string("execution_price", self.execution_price)
        self.delayed_action_execution = bool(self.delayed_action_execution)
        self.candidate_weights = _weights("candidate_weights", self.candidate_weights)
        if self.executed_weights is None:
            self.executed_weights = self.candidate_weights.copy()
        self.executed_weights = _weights("executed_weights", self.executed_weights, self.candidate_weights.shape)
        self.log_prob = _finite_float("log_prob", self.log_prob)
        self.value = _finite_float("value", self.value)
        self.decision_value = _optional_finite_float("decision_value", self.decision_value, default=self.value)
        self.reward = _finite_float("reward", self.reward)
        self.terminated = bool(self.terminated)
        self.truncated = bool(self.truncated)
        self.gate_action = _optional_int("gate_action", self.gate_action)
        self.rebalance_action = _optional_int("rebalance_action", self.rebalance_action)
        self.rebalance_intensity = _bounded_float("rebalance_intensity", self.rebalance_intensity, 0.0, 1.0)
        self.advantage = _optional_finite_float("advantage", self.advantage)
        self.return_value = _optional_finite_float("return_value", self.return_value)
        self.auxiliary_labels = dict(self.auxiliary_labels or {})
        self.preference_vector = _optional_vector("preference_vector", self.preference_vector)
        self.uncertainty_features = _feature_mapping("uncertainty_features", self.uncertainty_features)
        self.distributional_features = _feature_mapping("distributional_features", self.distributional_features)
        setattr(self, "return", self.return_value)

    @property
    def return_(self) -> float | None:
        return self.return_value

    @return_.setter
    def return_(self, value: float | None) -> None:
        self.return_value = _optional_finite_float("return_value", value)
        setattr(self, "return", self.return_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_date": self.decision_date,
            "execution_date": self.execution_date,
            "next_valuation_date": self.next_valuation_date,
            "execution_price": self.execution_price,
            "delayed_action_execution": self.delayed_action_execution,
            "state": self.state,
            "candidate_weights": self.candidate_weights,
            "executed_weights": self.executed_weights,
            "log_prob": self.log_prob,
            "value": self.value,
            "decision_value": self.decision_value,
            "gate_action": self.gate_action,
            "rebalance_action": self.rebalance_action,
            "rebalance_intensity": self.rebalance_intensity,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "advantage": self.advantage,
            "return": self.return_value,
            "auxiliary_labels": self.auxiliary_labels,
            "preference_vector": self.preference_vector,
            "uncertainty_features": self.uncertainty_features,
            "distributional_features": self.distributional_features,
        }


class RolloutBuffer:
    def __init__(
        self,
        rollout_steps: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        advantage_normalization: bool = True,
    ):
        self.rollout_steps = _positive_int("rollout_steps", rollout_steps)
        self.gamma = _unit_interval_float("gamma", gamma)
        self.gae_lambda = _unit_interval_float("gae_lambda", gae_lambda)
        self.advantage_normalization = bool(advantage_normalization)
        self._items: list[RolloutItem] = []
        self.last_observation: Any | None = None
        self.rollout_boundary_split = False

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[RolloutItem, ...]:
        return tuple(self._items)

    @property
    def is_full(self) -> bool:
        return len(self._items) >= self.rollout_steps

    def clear(self) -> None:
        self._items.clear()
        self.last_observation = None
        self.rollout_boundary_split = False

    def add(self, item: RolloutItem | Mapping[str, Any] | None = None, **kwargs: Any) -> RolloutItem:
        if self.is_full:
            raise ValueError("ERR_ROLLOUT_BUFFER_FULL")
        payload = _item_payload(item, kwargs)
        rollout_item = item if isinstance(item, RolloutItem) and not kwargs else RolloutItem(**payload)
        self._items.append(rollout_item)
        return rollout_item

    def extend(self, items: Sequence[RolloutItem | Mapping[str, Any]]) -> None:
        for item in items:
            self.add(item)

    def compute_gae(self, last_value: float = 0.0, last_terminated: bool = False) -> None:
        if not self._items:
            return
        next_value = 0.0 if bool(last_terminated) else _finite_float("last_value", last_value)
        last_gae = 0.0
        for item in reversed(self._items):
            non_terminal = 0.0 if item.terminated or item.truncated else 1.0
            delta = item.reward + self.gamma * next_value * non_terminal - item.value
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            item.advantage = float(last_gae)
            item.return_ = float(last_gae + item.value)
            next_value = item.value
        if self.advantage_normalization:
            self._normalize_advantages()

    def as_batch(self, device: torch.device | str | None = None) -> dict[str, Any]:
        if not self._items:
            raise ValueError("ERR_ROLLOUT_BUFFER_EMPTY")
        target_device = torch.device("cpu" if device is None else device)
        return {
            "state": [item.state for item in self._items],
            "candidate_weights": _tensor_stack([item.candidate_weights for item in self._items], target_device),
            "executed_weights": _tensor_stack([item.executed_weights for item in self._items], target_device),
            "log_prob": _tensor_column([item.log_prob for item in self._items], target_device),
            "value": _tensor_column([item.value for item in self._items], target_device),
            "decision_value": _tensor_column([item.decision_value for item in self._items], target_device),
            "reward": _tensor_column([item.reward for item in self._items], target_device),
            "advantage": _tensor_column([item.advantage for item in self._items], target_device),
            "return": _tensor_column([item.return_value for item in self._items], target_device),
            "gate_action": _long_column([item.gate_action for item in self._items], target_device),
            "rebalance_action": _long_column([item.rebalance_action for item in self._items], target_device),
            "rebalance_intensity": _tensor_column([item.rebalance_intensity for item in self._items], target_device),
            "terminated": _bool_column([item.terminated for item in self._items], target_device),
            "truncated": _bool_column([item.truncated for item in self._items], target_device),
            "decision_date": [item.decision_date for item in self._items],
            "execution_date": [item.execution_date for item in self._items],
            "next_valuation_date": [item.next_valuation_date for item in self._items],
            "execution_price": [item.execution_price for item in self._items],
            "delayed_action_execution": [item.delayed_action_execution for item in self._items],
            "auxiliary_labels": [item.auxiliary_labels for item in self._items],
            "preference_vector": [item.preference_vector for item in self._items],
            "uncertainty_features": [item.uncertainty_features for item in self._items],
            "distributional_features": [item.distributional_features for item in self._items],
        }

    def _normalize_advantages(self) -> None:
        advantages = np.asarray([item.advantage for item in self._items], dtype=float)
        if advantages.size <= 1:
            return
        std = float(advantages.std(ddof=0))
        if std <= 1.0e-8:
            return
        normalized = (advantages - float(advantages.mean())) / std
        for item, advantage in zip(self._items, normalized, strict=True):
            item.advantage = float(advantage)


def _item_payload(item: RolloutItem | Mapping[str, Any] | None, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if item is None:
        payload: dict[str, Any] = {}
    elif isinstance(item, RolloutItem):
        payload = item.to_dict()
    elif isinstance(item, Mapping):
        payload = dict(item)
    else:
        raise TypeError("ERR_ROLLOUT_ITEM_TYPE")
    payload.update(kwargs)
    if "return" in payload and payload.get("return_value") is None:
        payload["return_value"] = payload.pop("return")
    if "return_" in payload and payload.get("return_value") is None:
        payload["return_value"] = payload.pop("return_")
    payload.pop("return", None)
    payload.pop("return_", None)
    return payload


def _timestamp(name: str, value: Any) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"ERR_ROLLOUT_ITEM_DATE_INVALID: {name}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"ERR_ROLLOUT_ITEM_DATE_INVALID: {name}")
    return timestamp


def _non_empty_string(name: str, value: Any) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"ERR_ROLLOUT_ITEM_FIELD_INVALID: {name}")
    return result


def _weights(name: str, value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"ERR_ROLLOUT_ITEM_SHAPE: {name}")
    if shape is not None and tuple(result.shape) != tuple(shape):
        raise ValueError(f"ERR_ROLLOUT_ITEM_SHAPE: {name}")
    if not np.isfinite(result).all():
        raise ValueError(f"ERR_ROLLOUT_ITEM_NON_FINITE: {name}")
    return result.copy()


def _optional_vector(name: str, value: Any) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"ERR_ROLLOUT_ITEM_SHAPE: {name}")
    return result.copy()


def _feature_mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.size == 0:
        return {}
    return {f"{name}_vector": array}


def _finite_float(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"ERR_ROLLOUT_ITEM_NON_FINITE: {name}")
    return result


def _optional_finite_float(name: str, value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    return _finite_float(name, value)


def _bounded_float(name: str, value: Any, lower: float, upper: float) -> float:
    result = _finite_float(name, value)
    if result < lower or result > upper:
        raise ValueError(f"ERR_ROLLOUT_ITEM_FIELD_INVALID: {name}")
    return result


def _optional_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result < 0:
        raise ValueError(f"ERR_ROLLOUT_ITEM_FIELD_INVALID: {name}")
    return result


def _positive_int(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"ERR_ROLLOUT_BUFFER_CONFIG_INVALID: {name}")
    return result


def _unit_interval_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"ERR_ROLLOUT_BUFFER_CONFIG_INVALID: {name}")
    return result


def _tensor_stack(values: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)


def _tensor_column(values: Sequence[float | None], device: torch.device) -> torch.Tensor:
    array = np.asarray([np.nan if value is None else float(value) for value in values], dtype=np.float32)
    return torch.as_tensor(array, dtype=torch.float32, device=device).view(-1, 1)


def _long_column(values: Sequence[int | None], device: torch.device) -> torch.Tensor:
    array = np.asarray([-1 if value is None else int(value) for value in values], dtype=np.int64)
    return torch.as_tensor(array, dtype=torch.long, device=device).view(-1, 1)


def _bool_column(values: Sequence[bool], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=bool), dtype=torch.bool, device=device).view(-1, 1)


__all__ = ["RolloutBuffer", "RolloutItem"]
