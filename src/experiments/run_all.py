from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.experiments.registry import ExperimentRegistry
from src.utils.logger import save_json_atomic, write_run_outputs


FULL_REPRODUCTION_SEQUENCE: tuple[str, ...] = (
    "main_model",
    "baseline_comparison",
    "ablation",
    "input_matrix_ablation",
    "pca_ablation",
    "kernel_size_ablation",
    "reward_ablation",
    "transaction_cost_sensitivity",
    "asset_universe_sensitivity",
    "auxiliary_task_sensitivity",
    "rebalance_frequency_analysis",
    "seed_stability",
    "hyperparameter_sweep",
    "market_regime",
    "preference_conditioned_analysis",
    "uncertainty_analysis",
    "distributional_cvar_analysis",
    "partial_rebalance_analysis",
    "walk_forward",
)


def run_experiment_matrix(
    config: Mapping[str, Any],
    registry: ExperimentRegistry | None = None,
    device: Any | None = None,
    run_dir: str | Path | None = None,
    experiment_sequence: Sequence[str] | None = None,
) -> dict[str, Any]:
    base_config = _config_copy(config)
    parent_run_id = str(_mapping(base_config.get("output")).get("run_name", "full_reproduction"))
    selected_sequence = tuple(experiment_sequence or FULL_REPRODUCTION_SEQUENCE)
    selected_registry = registry or ExperimentRegistry()
    parent_run_dir = None if run_dir is None else Path(run_dir)
    resume_completed_children = bool(_mapping(base_config.get("full_reproduction")).get("resume_completed_children"))

    lineage: list[dict[str, Any]] = []
    for index, experiment_type in enumerate(selected_sequence, start=1):
        child_run_id = f"{parent_run_id}.{index:02d}_{experiment_type}"
        child_config = _child_config(base_config, experiment_type, child_run_id)
        child_run_dir = None if parent_run_dir is None else parent_run_dir / child_run_id
        experiment = selected_registry.create_experiment(child_config, device=device, run_dir=child_run_dir)
        cached_result = _completed_child_result(child_run_dir) if resume_completed_children else None
        result = cached_result if cached_result is not None else _result_mapping(experiment.run())
        status = str(result.get("status", "unknown"))
        if status != "completed":
            raise RuntimeError(f"ERR_EXPERIMENT_MATRIX_CHILD_NOT_COMPLETED: {experiment_type}: status={status}")
        if child_run_dir is not None and cached_result is None:
            save_json_atomic(_result_summary(result), child_run_dir / "logs" / "experiment_result.json")
            write_run_outputs(
                result,
                child_run_dir,
                config=child_config,
                manifest_overrides={
                    "status": "success",
                    "parent_run_id": parent_run_id,
                    "experiment_type": experiment_type,
                    "output_name": experiment.output_name,
                },
            )
        lineage.append(
            {
                "order": index,
                "parent_run_id": parent_run_id,
                "child_run_id": child_run_id,
                "experiment_type": experiment_type,
                "output_name": experiment.output_name,
                "relation_type": "full_reproduction_child",
                "status": str(result.get("status", "unknown")),
                "resumed_from_completed_child": cached_result is not None,
                "result": result,
            }
        )

    payload = {
        "status": "completed",
        "parent_run_id": parent_run_id,
        "run_sequence": list(selected_sequence),
        "lineage": lineage,
    }
    if parent_run_dir is not None:
        save_json_atomic(payload, parent_run_dir / "logs" / "lineage.json")
        save_json_atomic(payload, parent_run_dir / "metrics" / "full_reproduction_summary.json")
    return payload


def _child_config(config: Mapping[str, Any], experiment_type: str, child_run_id: str) -> dict[str, Any]:
    child_config = deepcopy(dict(config))
    child_config.setdefault("experiment", {})
    child_config["experiment"]["type"] = experiment_type
    child_config.setdefault("output", {})
    child_config["output"]["run_name"] = child_run_id
    return child_config


def _config_copy(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("ERR_EXPERIMENT_MATRIX_CONFIG_TYPE")
    return deepcopy(dict(config))


def _result_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    return {"status": "completed", "result": result}


def _completed_child_result(child_run_dir: Path | None) -> dict[str, Any] | None:
    if child_run_dir is None:
        return None
    result_path = child_run_dir / "logs" / "experiment_result.json"
    if not result_path.exists():
        return None
    try:
        with result_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    result = dict(payload)
    return result if str(result.get("status", "unknown")) == "completed" else None


def _result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in result.items():
        if hasattr(value, "shape") and hasattr(value, "columns"):
            summary[key] = {
                "rows": int(value.shape[0]),
                "columns": [str(column) for column in value.columns],
            }
        else:
            summary[key] = value
    return summary


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["FULL_REPRODUCTION_SEQUENCE", "run_experiment_matrix"]
