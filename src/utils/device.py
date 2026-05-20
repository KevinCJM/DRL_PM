from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from src.config import ConfigError


class _CpuDevice:
    type = "cpu"

    def __str__(self) -> str:
        return self.type


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise ConfigError("ERR_DEVICE_UNAVAILABLE", "device.mode", "ERR_DEVICE_UNAVAILABLE: torch") from exc


def _read(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _cuda_available(torch_module: Any) -> bool:
    return bool(getattr(torch_module, "cuda", None) and torch_module.cuda.is_available())


def _mps_available(torch_module: Any) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps and mps.is_available())


def get_device(config: Mapping[str, Any] | object, torch_module: Any | None = None) -> Any:
    mode = str(_read(config, "mode", "auto")).lower()
    if torch_module is None:
        try:
            torch = _load_torch()
        except ConfigError:
            if mode in {"auto", "cpu"}:
                return _CpuDevice()
            raise
    else:
        torch = torch_module

    if mode == "auto":
        if _cuda_available(torch):
            return torch.device("cuda")
        if _mps_available(torch):
            return torch.device("mps")
        return torch.device("cpu")

    if mode == "cuda" and not _cuda_available(torch):
        raise ConfigError("ERR_DEVICE_UNAVAILABLE", "device.mode", "ERR_DEVICE_UNAVAILABLE: device.mode")
    if mode == "mps" and not _mps_available(torch):
        raise ConfigError("ERR_DEVICE_UNAVAILABLE", "device.mode", "ERR_DEVICE_UNAVAILABLE: device.mode")
    if mode not in {"cpu", "cuda", "mps"}:
        raise ConfigError("ERR_DEVICE_UNAVAILABLE", "device.mode", "ERR_DEVICE_UNAVAILABLE: device.mode")
    return torch.device(mode)


def _device_type(device: Any) -> str:
    return str(getattr(device, "type", str(device))).split(":", 1)[0]


def enable_amp_if_available(
    config: Mapping[str, Any] | object,
    device: Any | None = None,
    torch_module: Any | None = None,
) -> bool:
    selected_device = device if device is not None else get_device(config, torch_module=torch_module)
    return bool(_read(config, "amp", True) and _device_type(selected_device) == "cuda")


def move_batch_to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to") and callable(batch.to):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return type(batch)((key, move_batch_to_device(value, device)) for key, value in batch.items())
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch
