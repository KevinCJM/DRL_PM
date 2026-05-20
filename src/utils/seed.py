from __future__ import annotations

import importlib
import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_DETERMINISM_CONFIG = {
    "deterministic_torch": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
}


def _read(config: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    return config.get(key, default)


def set_global_seed(seed: int, config: Mapping[str, Any] | None = None, env: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch = _torch_module()
    if torch is None:
        seed_environment(env, seed)
        return

    torch.manual_seed(seed)
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    configure_torch_determinism(config)
    seed_environment(env, seed)


def configure_torch_determinism(config: Mapping[str, Any] | None = None) -> None:
    workspace_config = _read(config, "cublas_workspace_config")
    if workspace_config:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(workspace_config)

    torch = _torch_module()
    if torch is None:
        return

    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        cudnn.benchmark = bool(_read(config, "cudnn_benchmark", DEFAULT_DETERMINISM_CONFIG["cudnn_benchmark"]))
        cudnn.deterministic = bool(
            _read(config, "cudnn_deterministic", DEFAULT_DETERMINISM_CONFIG["cudnn_deterministic"])
        )

    if bool(_read(config, "deterministic_torch", DEFAULT_DETERMINISM_CONFIG["deterministic_torch"])) and hasattr(
        torch,
        "use_deterministic_algorithms",
    ):
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_environment(env: Any | None, seed: int) -> None:
    if env is None:
        return
    if hasattr(env, "reset"):
        try:
            env.reset(seed=int(seed))
            return
        except TypeError:
            pass
    if hasattr(env, "seed"):
        env.seed(int(seed))


def collect_rng_states(env: Any | None = None) -> dict[str, Any]:
    torch = _torch_module()
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": _numpy_rng_state(),
        "torch_cpu_rng_state": None if torch is None else torch.random.get_rng_state().detach().cpu(),
        "torch_cuda_rng_state": _torch_cuda_rng_states(torch),
        "environment_random_state": get_environment_rng_state(env),
    }


def restore_rng_states(states: Mapping[str, Any] | None, env: Any | None = None) -> None:
    if not states:
        return
    python_state = states.get("python_random_state")
    if python_state is not None:
        random.setstate(_tuple_state(python_state))

    numpy_state = states.get("numpy_random_state")
    if numpy_state is not None:
        restore_numpy_rng_state(numpy_state)

    torch = _torch_module()
    if torch is not None:
        cpu_state = states.get("torch_cpu_rng_state")
        if cpu_state is not None:
            torch.random.set_rng_state(_torch_byte_tensor(cpu_state, torch))
        cuda_states = states.get("torch_cuda_rng_state")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([_torch_byte_tensor(state, torch) for state in cuda_states])

    restore_environment_rng_state(env, states.get("environment_random_state"))


def get_environment_rng_state(env: Any | None) -> Any:
    if env is None:
        return None
    if hasattr(env, "get_rng_state"):
        return env.get_rng_state()
    rng = getattr(env, "np_random", None)
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is not None:
        return bit_generator.state
    return None


def restore_environment_rng_state(env: Any | None, state: Any) -> None:
    if env is None or state is None:
        return
    restored_state = _restore_nested_state(state)
    if hasattr(env, "set_rng_state"):
        env.set_rng_state(restored_state)
        return
    rng = getattr(env, "np_random", None)
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is not None:
        bit_generator.state = restored_state


def restore_numpy_rng_state(state: Mapping[str, Any] | tuple[Any, ...] | list[Any]) -> None:
    if isinstance(state, Mapping):
        np.random.set_state(
            (
                str(state["bit_generator"]),
                np.asarray(_tensor_to_array(state["keys"]), dtype=np.uint32),
                int(state["pos"]),
                int(state["has_gauss"]),
                float(state["cached_gaussian"]),
            )
        )
        return
    restored = _tuple_state(state)
    if len(restored) != 5:
        raise ValueError("ERR_RNG_STATE_INVALID: numpy_random_state")
    np.random.set_state(
        (
            str(restored[0]),
            np.asarray(_tensor_to_array(restored[1]), dtype=np.uint32),
            int(restored[2]),
            int(restored[3]),
            float(restored[4]),
        )
    )


def _torch_module() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError:
        return None


def _torch_cuda_rng_states(torch: Any | None) -> list[Any] | None:
    if torch is None or not torch.cuda.is_available():
        return None
    return [state.detach().cpu() for state in torch.cuda.get_rng_state_all()]


def _torch_byte_tensor(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.uint8)
    return torch.as_tensor(np.asarray(value, dtype=np.uint8), dtype=torch.uint8)


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": name,
        "keys": keys.astype(np.uint32).tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _tuple_state(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return tuple(_tuple_state(item) if isinstance(item, (list, tuple)) else _restore_nested_state(item) for item in value)
    if isinstance(value, list):
        return tuple(_tuple_state(item) if isinstance(item, (list, tuple)) else _restore_nested_state(item) for item in value)
    raise ValueError("ERR_RNG_STATE_INVALID")


def _restore_nested_state(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _restore_nested_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_nested_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_restore_nested_state(item) for item in value)
    return _tensor_to_array(value)


def _tensor_to_array(value: Any) -> Any:
    torch = _torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


__all__ = [
    "collect_rng_states",
    "configure_torch_determinism",
    "get_environment_rng_state",
    "restore_environment_rng_state",
    "restore_numpy_rng_state",
    "restore_rng_states",
    "seed_environment",
    "set_global_seed",
]
