from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import config_hash
from src.agents.dqn_agent import DQNAgent
from src.agents.hybrid_agent import HybridAgent
from src.agents.ppo_agent import PPOAgent
from src.baselines.base_strategy import BaseStrategy
from src.baselines.cage_common import MODEL_EXTENSION_ID, mapping
from src.baselines.equal_weight import EqualWeightStrategy
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.models.cost_estimator import CostEstimator
from src.models.distributional_cvar_gated_ppo import DistributionalCVaRGatedPPO
from src.models.dqn_gated_multitask_cnn_ppo import FullGatedModel
from src.models.otar_cqr_gate import OTarCQRGate
from src.models.partial_rebalance_gated_ppo import PartialRebalanceGatedPPO
from src.models.preference_conditioned_gated_ppo import PreferenceConditionedGatedPPO
from src.models.risk_aware_graph_transformer import (
    RA_GT_RCPO_ALGORITHM,
    RA_GT_RCPO_MODEL_EXTENSION_ID,
    RA_GT_RCPO_MODEL_NAMES,
)
from src.models.uncertainty_aware_gated_ppo import UncertaintyAwareGatedPPO
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.experiments.aggregate_results import aggregate_walk_forward
from src.data.feature_matrix import (
    FEATURE_GROUP_SUMMARY_COLUMNS,
    FEATURE_PROVENANCE_COLUMNS,
    METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS,
    FeatureMatrix,
    FeatureMatrixBuilder,
    MarketImageDataset,
)
from src.data.feature_reduction import FeatureReductionPipeline
from src.data.loader import DataContractError, MarketDatasetBundle, load_market_dataset
from src.data.splits import SplitSpec, create_split, split_to_dict
from src.envs.backtest_engine import BacktestEngine, BacktestResult
from src.experiments.external_baselines import run_external_pgportfolio_baseline
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv


VALIDATION_METRIC_ALIASES = {
    "validation_penalized_sharpe": "validation_sharpe_minus_drawdown_turnover_penalty",
}

VALIDATION_ONLY_HPO_PARAMS = {"eta_v"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def filter_validation_only_hpo_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k not in VALIDATION_ONLY_HPO_PARAMS}


def run_trained_model_experiment(
    config: Mapping[str, Any],
    *,
    model_name: str,
    train_split: str = "train",
    validation_split: str = "validation",
    test_split: str = "test",
    run_dir: str | None = None,
    split_override: SplitSpec | None = None,
) -> dict[str, Any]:
    _assert_checkpoint_file_exists(config)
    artifacts = build_pipeline_artifacts(config, split_override=split_override)
    active_split = artifacts["split"]
    runtime_artifacts = dict(artifacts)
    runtime_artifacts["split"] = active_split
    runtime_config = strategy_runtime_config(config, artifacts["dataset"], artifacts["market_image_dataset"])
    runtime_config["model_name"] = model_name
    agent = build_hybrid_agent(runtime_config)
    train_env = _portfolio_env(artifacts, runtime_config, train_split, split=active_split)
    validation_env = _portfolio_env(artifacts, runtime_config, validation_split, split=active_split)
    checkpoint_paths = _checkpoint_paths(run_dir)
    _load_training_checkpoint(agent, runtime_config, train_env)
    _attach_checkpoint_callback(agent, checkpoint_paths, runtime_config, train_env)
    training_result = agent.train(train_env, validation_env=validation_env)
    last_checkpoint = _save_last_checkpoint(agent, checkpoint_paths, runtime_config, train_env)
    strategy = TrainedFullGatedStrategy(runtime_config, agent.ppo_agent)
    engine = BacktestEngine(runtime_config, market_image_dataset=artifacts["market_image_dataset"])
    backtest_result = engine.run(
        artifacts["dataset"],
        active_split,
        strategy,
        segment=test_split,
    )
    benchmark_result = engine.run(
        artifacts["dataset"],
        active_split,
        EqualWeightStrategy(runtime_config),
        segment=test_split,
    )
    payload = result_mapping(
        backtest_result,
        config=runtime_config,
        artifacts=runtime_artifacts,
        status="completed",
        model_name=model_name,
    )
    payload["model_returns"] = {model_name: backtest_result.daily_returns}
    payload["benchmark_returns"] = {"equal_weight": benchmark_result.daily_returns}
    payload["main_comparison"] = _comparison_rows(
        {
            model_name: dict(backtest_result.metrics),
            "equal_weight": dict(benchmark_result.metrics),
        },
        primary_model=model_name,
        config=runtime_config,
    )
    payload["training_status"] = training_result["status"]
    payload["training_history"] = training_result.get("history", [])
    payload["best_validation_metric"] = training_result.get("best_validation_metric")
    payload["best_checkpoint_path"] = None if checkpoint_paths["best"] is None else str(checkpoint_paths["best"])
    payload["last_checkpoint_path"] = None if last_checkpoint is None else str(last_checkpoint)
    payload["checkpoint_count"] = int(sum(path is not None and path.exists() for path in checkpoint_paths.values()))
    return payload


def run_trained_walk_forward_experiment(
    config: Mapping[str, Any],
    *,
    model_name: str,
    run_dir: str | None = None,
) -> dict[str, Any]:
    _assert_checkpoint_file_exists(config)
    splits = _splits_for_config(config)
    fold_results: list[BacktestResult] = []
    fold_payloads: list[dict[str, Any]] = []
    for index, split in enumerate(splits):
        fold_id = getattr(split, "fold_id", f"fold_{index + 1}")
        fold_run_dir = None if run_dir is None else str(_path_join(run_dir, str(fold_id)))
        fold_artifacts = build_pipeline_artifacts(config, split_override=split)
        payload, result = _train_and_backtest(
            config,
            fold_artifacts,
            split=split,
            model_name=model_name,
            test_split="test",
            run_dir=fold_run_dir,
        )
        fold_payloads.append(payload)
        fold_results.append(result)

    aggregation = aggregate_walk_forward(fold_results, run_dir=run_dir)
    result = dict(fold_payloads[-1])
    result["daily_returns"] = _concat([fold.daily_returns for fold in fold_results])
    result["daily_weights"] = _concat([fold.daily_weights for fold in fold_results])
    result["daily_turnover"] = _concat([fold.daily_turnover for fold in fold_results])
    result["daily_rebalance"] = _concat([fold.daily_rebalance for fold in fold_results])
    result["daily_costs"] = _concat([fold.daily_costs for fold in fold_results])
    result["daily_asset_returns"] = _concat([getattr(fold, "daily_asset_returns", pd.DataFrame()) for fold in fold_results])
    result["baseline_daily_diagnostics"] = _concat([_baseline_daily_diagnostics(fold) for fold in fold_results])
    result["metrics"] = _metrics_from_walk_forward(aggregation, fold_results)
    result["walk_forward_results"] = aggregation["walk_forward_results"]
    result["all_oos_daily_returns"] = aggregation["all_oos_daily_returns"]
    result["fold_count"] = aggregation["fold_count"]
    result["duplicate_oos_date_count"] = aggregation["duplicate_oos_date_count"]
    result["fold_training"] = [
        {
            "fold_id": getattr(split, "fold_id", f"fold_{index + 1}"),
            "training_status": payload.get("training_status"),
            "best_validation_metric": payload.get("best_validation_metric"),
            "best_checkpoint_path": payload.get("best_checkpoint_path"),
            "last_checkpoint_path": payload.get("last_checkpoint_path"),
        }
        for index, (split, payload) in enumerate(zip(splits, fold_payloads, strict=True))
    ]
    return result


def run_seed_stability_training(
    config: Mapping[str, Any],
    *,
    model_name: str,
    run_dir: str | None = None,
) -> dict[str, Any]:
    seeds = _seed_values(config)
    frames: dict[str, list[pd.DataFrame]] = {
        "daily_returns": [],
        "daily_weights": [],
        "daily_turnover": [],
        "daily_rebalance": [],
        "daily_costs": [],
        "daily_asset_returns": [],
        "baseline_daily_diagnostics": [],
    }
    metrics_by_seed: dict[int, dict[str, float]] = {}
    last_payload: dict[str, Any] | None = None
    for seed in seeds:
        seeded_config = deepcopy(dict(config))
        seeded_config.setdefault("reproducibility", {})
        seeded_config["reproducibility"]["seed"] = int(seed)
        seed_run_dir = None if run_dir is None else str(_path_join(run_dir, f"seed_{int(seed)}"))
        payload = run_trained_model_experiment(
            seeded_config,
            model_name=model_name,
            run_dir=seed_run_dir,
        )
        last_payload = payload
        metrics_by_seed[int(seed)] = dict(payload["metrics"])
        for name in frames:
            frame = payload[name].copy()
            frame["seed"] = int(seed)
            frames[name].append(frame)
    if last_payload is None:
        raise RuntimeError("ERR_SEED_STABILITY_EMPTY")
    result = dict(last_payload)
    result["seed_count"] = len(seeds)
    result["seed_metrics"] = metrics_by_seed
    result["seed_aggregate_summary"] = _seed_aggregate_summary(metrics_by_seed, model_name)
    for name, frame_list in frames.items():
        result[name] = _concat(frame_list)
    result["metrics"] = _mean_metrics(metrics_by_seed)
    _refresh_ppo_dqn_aggregate_comparison(result, model_name)
    return result


def run_trained_variant_matrix(
    config: Mapping[str, Any],
    *,
    model_name: str,
    matrix_name: str,
    variants: Sequence[Mapping[str, Any]],
    run_dir: str | None = None,
) -> dict[str, Any]:
    if not variants:
        raise RuntimeError("ERR_EXPERIMENT_MATRIX_EMPTY")
    frames: dict[str, list[pd.DataFrame]] = {
        "daily_returns": [],
        "daily_weights": [],
        "daily_turnover": [],
        "daily_rebalance": [],
        "daily_costs": [],
        "daily_asset_returns": [],
        "baseline_daily_diagnostics": [],
    }
    rows: list[dict[str, Any]] = []
    last_payload: dict[str, Any] | None = None
    for index, variant in enumerate(variants, start=1):
        variant_id = str(variant.get("variant_id") or f"variant_{index}")
        variant_config = deepcopy(dict(variant.get("config", config)))
        variant_config.setdefault("experiment", {})
        variant_config["experiment"]["type"] = "main_model"
        variant_config.setdefault("output", {})
        variant_config["output"]["run_name"] = f"{config.get('output', {}).get('run_name', 'matrix')}.{variant_id}"
        variant_run_dir = None if run_dir is None else str(_path_join(run_dir, variant_id))
        payload = run_trained_model_experiment(
            variant_config,
            model_name=str(variant.get("model_name") or model_name),
            run_dir=variant_run_dir,
        )
        if str(payload.get("status", "unknown")) != "completed":
            raise RuntimeError(f"ERR_EXPERIMENT_MATRIX_CHILD_NOT_COMPLETED: {variant_id}")
        last_payload = payload
        metric_values = dict(payload.get("metrics", {})) if isinstance(payload.get("metrics"), Mapping) else {}
        rows.append(
            {
                "variant_id": variant_id,
                "changed_key_path": variant.get("changed_key_path", ""),
                "variant_value": variant.get("variant_value", ""),
                "status": payload.get("status", "unknown"),
                **metric_values,
            }
        )
        for name in frames:
            frame = payload.get(name)
            if isinstance(frame, pd.DataFrame):
                variant_frame = frame.copy()
                variant_frame["variant_id"] = variant_id
                frames[name].append(variant_frame)
    if last_payload is None:
        raise RuntimeError("ERR_EXPERIMENT_MATRIX_EMPTY")
    matrix = pd.DataFrame(rows)
    result = dict(last_payload)
    result["status"] = "completed"
    result["model_name"] = model_name
    result["child_run_count"] = len(rows)
    result[matrix_name] = matrix
    result["matrix_results"] = matrix
    for name, frame_list in frames.items():
        result[name] = _concat(frame_list)
    return result


def _set_nested_config_value(config: dict[str, Any], dotpath: str, value: Any) -> None:
    keys = dotpath.split(".")
    cursor = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[keys[-1]] = value


def expand_otar_formal_matrix(
    matrix_path: str | Path,
    base_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    path = Path(matrix_path)
    if not path.exists():
        raise FileNotFoundError(f"ERR_OTAR_FORMAL_MATRIX_MISSING: {path}")
    with path.open("r", encoding="utf-8") as f:
        matrix = json.load(f) if path.suffix == ".json" else __import__("yaml").safe_load(f)
    if not isinstance(matrix, Mapping):
        raise ValueError("ERR_OTAR_FORMAL_MATRIX_INVALID: not a mapping")

    ablation_models = matrix.get("ablation_models")
    if not isinstance(ablation_models, Mapping) or not ablation_models:
        raise ValueError("ERR_OTAR_FORMAL_MATRIX_MISSING_ABLATION_MODELS")
    field_map = matrix.get("ablation_field_to_config_path")
    if not isinstance(field_map, Mapping):
        raise ValueError("ERR_OTAR_FORMAL_MATRIX_MISSING_FIELD_MAP")
    universes = matrix.get("universes", [])
    seeds = matrix.get("seeds", [])
    tier1 = matrix.get("tier1", {})
    split = matrix.get("split", {})
    if not universes:
        raise ValueError("ERR_OTAR_FORMAL_MATRIX_EMPTY_UNIVERSES")
    if not seeds:
        raise ValueError("ERR_OTAR_FORMAL_MATRIX_EMPTY_SEEDS")

    base = deepcopy(dict(base_config)) if base_config else {}
    child_experiment_type = _otar_child_experiment_type(base)
    runs: list[dict[str, Any]] = []

    for ablation_id, ablation_spec in ablation_models.items():
        if not isinstance(ablation_spec, Mapping):
            continue
        for universe in universes:
            for seed in seeds:
                run_config = deepcopy(base)
                run_config.setdefault("experiment", {})
                run_config["experiment"]["type"] = child_experiment_type
                run_config["experiment"]["ablation_id"] = ablation_id
                run_config.setdefault("model", {})
                model_name = _canonical_otar_formal_model_name(
                    ablation_spec.get("model", "dqn_gated_multitask_cnn_ppo")
                )
                run_config["model"]["name"] = model_name
                _apply_otar_child_model_scope(run_config, str(model_name))
                for field_key, config_path in field_map.items():
                    if field_key in ablation_spec:
                        _set_nested_config_value(run_config, config_path, ablation_spec[field_key])
                if tier1:
                    cost_val = tier1.get("cost")
                    if cost_val is not None:
                        _set_nested_config_value(run_config, "cost_model.proportional_cost", cost_val)
                    cq_val = tier1.get("confidence_q")
                    if cq_val is not None:
                        _set_nested_config_value(run_config, "reward.confidence_q", cq_val)
                _set_nested_config_value(run_config, "training.seed", seed)
                _set_nested_config_value(run_config, "reproducibility.seed", seed)
                _set_nested_config_value(run_config, "protocol.asset_universe_id", str(universe))
                _apply_otar_universe(run_config, str(universe))
                if split:
                    _apply_otar_split(run_config, split)
                run_config.setdefault("output", {})
                run_config["output"]["run_name"] = f"OTAR_{ablation_id}_{universe}_seed{seed}"
                runs.append(run_config)

    tier2 = matrix.get("tier2")
    if isinstance(tier2, Mapping):
        t2_models = tier2.get("models", [])
        t2_costs = tier2.get("costs", [])
        t2_seeds = tier2.get("seeds", [])
        t2_universe = tier2.get("universe", "Small-8")
        for t2_model in t2_models:
            for t2_cost in t2_costs:
                for t2_seed in t2_seeds:
                    run_config = deepcopy(base)
                    run_config.setdefault("experiment", {})
                    run_config["experiment"]["type"] = child_experiment_type
                    run_config["experiment"]["ablation_id"] = ""
                    run_config.setdefault("model", {})
                    model_name = _canonical_otar_formal_model_name(t2_model)
                    run_config["model"]["name"] = model_name
                    _apply_otar_child_model_scope(run_config, str(model_name))
                    _set_nested_config_value(run_config, "cost_model.proportional_cost", t2_cost)
                    _set_nested_config_value(run_config, "training.seed", t2_seed)
                    _set_nested_config_value(run_config, "reproducibility.seed", t2_seed)
                    _set_nested_config_value(run_config, "protocol.asset_universe_id", str(t2_universe))
                    _apply_otar_universe(run_config, str(t2_universe))
                    if split:
                        _apply_otar_split(run_config, split)
                    run_config.setdefault("output", {})
                    run_config["output"]["run_name"] = f"OTAR_tier2_{model_name}_cost{t2_cost}_seed{t2_seed}"
                    runs.append(run_config)

    return runs


def run_otar_formal_matrix(
    base_config: Mapping[str, Any],
    *,
    matrix_path: str | Path = "configs/paper/otar_formal_matrix.yaml",
    run_dir: str | Path | None = None,
    max_runs: int | None = None,
    device: Any | None = None,
    resume_completed: bool = False,
) -> dict[str, Any]:
    from src.experiments.registry import ExperimentRegistry, _merge_otar_hpo_grid_sources
    from src.utils.logger import save_json_atomic, write_run_outputs

    runs = expand_otar_formal_matrix(matrix_path, base_config)
    if max_runs is not None:
        runs = runs[: max(0, int(max_runs))]
    registry = ExperimentRegistry()
    parent_dir = None if run_dir is None else Path(run_dir)
    lineage: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for index, child_config in enumerate(runs, start=1):
        child = deepcopy(dict(child_config))
        child.setdefault("output", {})
        run_name = str(child["output"].get("run_name") or f"OTAR_formal_{index:03d}")
        child["output"]["run_name"] = run_name
        child = _merge_otar_hpo_grid_sources(child)
        child["config_hash"] = config_hash(child)
        child_dir = None if parent_dir is None else parent_dir / f"{index:03d}_{run_name}"
        if resume_completed and child_dir is not None:
            resumed = _completed_otar_formal_child_summary(child_dir, child)
            if resumed is not None:
                metrics = resumed.get("metrics", {})
                result_rows.append(
                    {
                        "run_name": run_name,
                        "ablation_id": child.get("experiment", {}).get("ablation_id", ""),
                        "universe": child.get("protocol", {}).get("asset_universe_id", ""),
                        "seed": child.get("training", {}).get("seed", child.get("reproducibility", {}).get("seed")),
                        "model_name": child.get("model", {}).get("name", ""),
                        "status": "completed",
                        **(dict(metrics) if isinstance(metrics, Mapping) else {}),
                    }
                )
                lineage.append(
                    {
                        "order": index,
                        "child_run_id": run_name,
                        "ablation_id": child.get("experiment", {}).get("ablation_id", ""),
                        "universe": child.get("protocol", {}).get("asset_universe_id", ""),
                        "status": "completed",
                        "run_dir": str(child_dir),
                        "resumed_from_completed_child": True,
                        "child_config_hash": child["config_hash"],
                    }
                )
                continue
        experiment = registry.create_experiment(child, device=device, run_dir=child_dir)
        result = experiment.run()
        if not isinstance(result, Mapping):
            result = {"status": "completed", "result": result}
        status = str(result.get("status", "unknown"))
        if status != "completed":
            raise RuntimeError(f"ERR_OTAR_FORMAL_CHILD_NOT_COMPLETED: {run_name}: status={status}")
        if child_dir is not None:
            save_json_atomic(_result_summary_for_matrix(result), child_dir / "logs" / "experiment_result.json")
            write_run_outputs(
                result,
                child_dir,
                config=child,
                manifest_overrides={
                    "status": "success",
                    "parent_run_id": str(_mapping(base_config.get("output")).get("run_name", "otar_formal_matrix")),
                    "experiment_type": getattr(experiment, "experiment_type", child.get("experiment", {}).get("type")),
                    "output_name": getattr(experiment, "output_name", None),
                    "ablation_id": child.get("experiment", {}).get("ablation_id", ""),
                    "child_config_hash": child["config_hash"],
                },
            )
        metrics = result.get("metrics", {})
        result_rows.append(
            {
                "run_name": run_name,
                "ablation_id": child.get("experiment", {}).get("ablation_id", ""),
                "universe": child.get("protocol", {}).get("asset_universe_id", ""),
                "seed": child.get("training", {}).get("seed", child.get("reproducibility", {}).get("seed")),
                "model_name": child.get("model", {}).get("name", ""),
                "status": status,
                **(dict(metrics) if isinstance(metrics, Mapping) else {}),
            }
        )
        lineage.append(
            {
                "order": index,
                "child_run_id": run_name,
                "ablation_id": child.get("experiment", {}).get("ablation_id", ""),
                "universe": child.get("protocol", {}).get("asset_universe_id", ""),
                "status": status,
                "run_dir": None if child_dir is None else str(child_dir),
                "resumed_from_completed_child": False,
                "child_config_hash": child["config_hash"],
            }
        )
    summary = pd.DataFrame(result_rows)
    payload = {
        "status": "completed",
        "run_count": len(runs),
        "lineage": lineage,
        "otar_formal_matrix_summary": summary,
        "output_name": "otar_formal_matrix_summary",
    }
    if parent_dir is not None:
        save_json_atomic({"status": "completed", "lineage": lineage}, parent_dir / "logs" / "otar_formal_lineage.json")
    return payload


def _completed_otar_formal_child_summary(child_dir: Path, child_config: Mapping[str, Any]) -> dict[str, Any] | None:
    result_path = child_dir / "logs" / "experiment_result.json"
    manifest_path = child_dir / "logs" / "run_manifest.json"
    if not result_path.exists() or not manifest_path.exists():
        return None
    result = _read_json_mapping(result_path)
    manifest = _read_json_mapping(manifest_path)
    result_status = str(result.get("status", ""))
    manifest_status = str(manifest.get("status", ""))
    if result_status != "completed" or manifest_status != "success":
        return None
    expected_hash = str(child_config.get("config_hash", ""))
    actual_hash = str(manifest.get("config_hash") or manifest.get("child_config_hash") or "")
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"ERR_OTAR_FORMAL_RESUME_STALE_CHILD: {child_dir}: config_hash={actual_hash}, expected={expected_hash}"
        )
    expected_protocol = _mapping(child_config.get("protocol")).get("protocol_id")
    expected_cutoff = _mapping(child_config.get("protocol")).get("data_cutoff_date")
    expected_ablation = _mapping(child_config.get("experiment")).get("ablation_id", "")
    expected_universe = _mapping(child_config.get("protocol")).get("asset_universe_id", "")
    mismatches: list[str] = []
    if manifest.get("protocol_id") != expected_protocol:
        mismatches.append("protocol_id")
    if manifest.get("data_cutoff_date") != expected_cutoff:
        mismatches.append("data_cutoff_date")
    if manifest.get("ablation_id", "") != expected_ablation:
        mismatches.append("ablation_id")
    if manifest.get("asset_universe_id", "") != expected_universe:
        mismatches.append("asset_universe_id")
    if mismatches:
        raise RuntimeError(f"ERR_OTAR_FORMAL_RESUME_STALE_CHILD: {child_dir}: {','.join(mismatches)}")
    return result


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ERR_OTAR_FORMAL_RESUME_INVALID_JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"ERR_OTAR_FORMAL_RESUME_INVALID_JSON: {path}")
    return payload


def _result_summary_for_matrix(result: Mapping[str, Any]) -> dict[str, Any]:
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


def _apply_otar_split(run_config: dict[str, Any], split: Mapping[str, Any]) -> None:
    _set_nested_config_value(run_config, "split.mode", "fixed")
    for split_name, split_spec in split.items():
        if not isinstance(split_spec, Mapping):
            continue
        for key, value in split_spec.items():
            _set_nested_config_value(run_config, f"split.{split_name}_{key}", value)
    test_spec = split.get("test")
    if isinstance(test_spec, Mapping) and test_spec.get("end") not in (None, ""):
        _set_nested_config_value(run_config, "data.end_date", test_spec["end"])
    train_spec = split.get("train")
    if isinstance(train_spec, Mapping):
        start = train_spec.get("start")
        if start not in (None, "", "earliest"):
            _set_nested_config_value(run_config, "data.start_date", start)


def _apply_otar_universe(run_config: dict[str, Any], universe: str) -> None:
    key = universe.lower().replace("_", "-")
    run_config.setdefault("data", {})
    if key in {"small-8", "small8", "small_8"}:
        run_config["data"]["asset_universe_assets"] = _small8_asset_codes()
        return
    if key in {"core-13", "core13", "core_13"}:
        run_config["data"]["asset_universe_assets"] = []
        run_config["data"]["asset_universe_pools"] = []


def _canonical_otar_formal_model_name(model_name: Any) -> str:
    name = str(model_name)
    if name == "dqn_gated_multitask_cnn_ppo":
        return "full_dqn_gated_multitask_cnn_ppo"
    return name


def _apply_otar_child_model_scope(run_config: dict[str, Any], model_name: str) -> None:
    if isinstance(run_config.get("hpo"), dict):
        run_config["hpo"]["trainable_models"] = [str(model_name)]
        run_config["hpo"]["equal_budget_across_models"] = False
    native_rl = run_config.get("baselines", {}).get("native_rl") if isinstance(run_config.get("baselines"), Mapping) else None
    if isinstance(native_rl, dict):
        native_rl["enabled_models"] = [str(model_name)]


def _otar_child_experiment_type(base_config: Mapping[str, Any]) -> str:
    experiment_config = base_config.get("experiment")
    experiment_type = str(experiment_config.get("type", "")) if isinstance(experiment_config, Mapping) else ""
    if experiment_type and experiment_type != "otar_formal_matrix":
        return experiment_type
    hpo_config = base_config.get("hpo")
    return "hyperparameter_sweep" if isinstance(hpo_config, Mapping) and hpo_config.get("enabled") is True else "main_model"


def _small8_asset_codes() -> list[str]:
    path = Path("configs/data/small8_universe.yaml")
    if not path.exists():
        return [
            "510300.SH",
            "510500.SH",
            "510050.SH",
            "159915.SZ",
            "159920.SZ",
            "513100.SH",
            "518880.SH",
            "511010.SH",
        ]
    with path.open("r", encoding="utf-8") as fh:
        payload = __import__("yaml").safe_load(fh) or {}
    assets = payload.get("assets") if isinstance(payload, Mapping) else None
    if not isinstance(assets, Sequence):
        return []
    return [str(item.get("ts_code")) for item in assets if isinstance(item, Mapping) and item.get("ts_code")]


def run_strategy_backtest(
    config: Mapping[str, Any],
    strategy_factory: Any,
    *,
    model_name: str,
    segment: str = "test",
    run_dir: str | None = None,
    split_override: SplitSpec | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if artifacts is None:
        artifacts = build_pipeline_artifacts(config, split_override=split_override)
    backtest_output = _run_backtest_with_artifacts(
        config,
        artifacts,
        strategy_factory,
        model_name=model_name,
        split=artifacts["split"],
        segment=segment,
        run_dir=run_dir,
    )
    strategy_config, result, strategy = _unpack_backtest_output(backtest_output)
    payload = result_mapping(
        result,
        config=strategy_config,
        artifacts=artifacts,
        status="completed",
        model_name=model_name,
    )
    summary_row, history_frame = _baseline_training_artifacts(model_name, strategy, result.baseline_daily_diagnostics)
    _block_rankable_without_paper_model_id(model_name, result.baseline_daily_diagnostics, summary_row)
    payload["baseline_training_summary"] = pd.DataFrame([summary_row])
    payload["baseline_training_history"] = (
        history_frame if history_frame is not None else pd.DataFrame()
    )
    payload["baseline_comparison"] = _comparison_rows(
        {model_name: dict(result.metrics)},
        training_summary_rows=[summary_row],
        config=strategy_config,
    )
    payload["best_checkpoint_path"] = summary_row.get("checkpoint_best_path")
    payload["last_checkpoint_path"] = summary_row.get("checkpoint_last_path")
    payload["evaluated_checkpoint_path"] = summary_row.get("evaluated_checkpoint_path")
    payload["best_validation_metric"] = summary_row.get("best_validation_metric")
    payload.update(_new_model_artifacts(payload, config=strategy_config))
    return payload


def run_walk_forward_backtest(
    config: Mapping[str, Any],
    strategy_factory: Any,
    *,
    model_name: str,
    segment: str = "test",
    run_dir: str | None = None,
) -> dict[str, Any]:
    splits = _splits_for_config(config)
    fold_results: list[BacktestResult] = []
    strategy_config: dict[str, Any] | None = None
    last_artifacts: dict[str, Any] | None = None
    for split in splits:
        artifacts = build_pipeline_artifacts(config, split_override=split)
        last_artifacts = artifacts
        backtest_output = _run_backtest_with_artifacts(
            config,
            artifacts,
            strategy_factory,
            model_name=model_name,
            split=split,
            segment=segment,
            run_dir=None if run_dir is None else str(_path_join(run_dir, str(getattr(split, "fold_id", "fold")))),
        )
        strategy_config, result, _strategy = _unpack_backtest_output(backtest_output)
        fold_results.append(result)
    aggregation = aggregate_walk_forward(fold_results, run_dir=run_dir)
    daily_returns = _concat([result.daily_returns for result in fold_results])
    daily_weights = _concat([result.daily_weights for result in fold_results])
    daily_turnover = _concat([result.daily_turnover for result in fold_results])
    daily_rebalance = _concat([result.daily_rebalance for result in fold_results])
    daily_costs = _concat([result.daily_costs for result in fold_results])
    daily_asset_returns = _concat([getattr(result, "daily_asset_returns", pd.DataFrame()) for result in fold_results])
    baseline_daily_diagnostics = _concat([_baseline_daily_diagnostics(result) for result in fold_results])
    metrics = _metrics_from_walk_forward(aggregation, fold_results)
    return {
        "status": "completed",
        "model_name": model_name,
        "metrics": metrics,
        "daily_returns": daily_returns,
        "daily_weights": daily_weights,
        "daily_turnover": daily_turnover,
        "daily_rebalance": daily_rebalance,
        "daily_costs": daily_costs,
        "daily_asset_returns": daily_asset_returns,
        "baseline_daily_diagnostics": baseline_daily_diagnostics,
        "walk_forward_results": aggregation["walk_forward_results"],
        "all_oos_daily_returns": aggregation["all_oos_daily_returns"],
        "fold_count": aggregation["fold_count"],
        "duplicate_oos_date_count": aggregation["duplicate_oos_date_count"],
        "training_status": "not_applicable_for_backtest_pipeline",
        "device": None if strategy_config is None else strategy_config.get("device"),
        **artifact_payload(last_artifacts),
    }


def run_seed_stability_backtests(
    config: Mapping[str, Any],
    strategy_factory: Any,
    *,
    model_name: str,
    segment: str = "test",
) -> dict[str, Any]:
    seeds = _seed_values(config)
    frames: dict[str, list[pd.DataFrame]] = {
        "daily_returns": [],
        "daily_weights": [],
        "daily_turnover": [],
        "daily_rebalance": [],
        "daily_costs": [],
        "daily_asset_returns": [],
        "baseline_daily_diagnostics": [],
    }
    metrics_by_seed: dict[int, dict[str, float]] = {}
    last_payload: dict[str, Any] | None = None
    for seed in seeds:
        seeded_config = deepcopy(dict(config))
        seeded_config.setdefault("reproducibility", {})
        seeded_config["reproducibility"]["seed"] = int(seed)
        payload = run_strategy_backtest(seeded_config, strategy_factory, model_name=model_name, segment=segment)
        last_payload = payload
        metrics_by_seed[int(seed)] = dict(payload["metrics"])
        for name in frames:
            frame = payload[name].copy()
            frame["seed"] = int(seed)
            frames[name].append(frame)

    if last_payload is None:
        raise RuntimeError("ERR_SEED_STABILITY_EMPTY")

    result = dict(last_payload)
    result["status"] = "completed"
    result["seed_count"] = len(seeds)
    result["seed_metrics"] = metrics_by_seed
    result["seed_aggregate_summary"] = _seed_aggregate_summary(metrics_by_seed, model_name)
    for name, frame_list in frames.items():
        result[name] = _concat(frame_list)
    result["metrics"] = _mean_metrics(metrics_by_seed)
    _refresh_ppo_dqn_aggregate_comparison(result, model_name)
    return result


def run_strategy_comparison(
    config: Mapping[str, Any],
    strategy_factories: Mapping[str, Any],
    *,
    segment: str = "test",
    run_dir: str | None = None,
) -> dict[str, Any]:
    artifacts = build_pipeline_artifacts(config)
    frames: dict[str, list[pd.DataFrame]] = {
        "daily_returns": [],
        "daily_weights": [],
        "daily_turnover": [],
        "daily_rebalance": [],
        "daily_costs": [],
        "daily_asset_returns": [],
        "baseline_daily_diagnostics": [],
    }
    metrics: dict[str, Any] = {}
    returns_by_model: dict[str, pd.DataFrame] = {}
    training_summary_rows: list[dict[str, Any]] = []
    training_history_frames: list[pd.DataFrame] = []

    for model_name, factory in strategy_factories.items():
        strategy_config = strategy_runtime_config(config, artifacts["dataset"], artifacts["market_image_dataset"])
        strategy_config["model_name"] = model_name
        if run_dir is not None:
            strategy_config["baseline_run_dir"] = str(_path_join(run_dir, str(model_name)))
        if model_name in EXTERNAL_BASELINES:
            external_result = run_external_pgportfolio_baseline(
                strategy_config,
                artifacts,
                segment=segment,
                run_dir=run_dir,
            )
            summary = _frame_or_none(external_result.get("baseline_training_summary"))
            if summary is not None:
                training_summary_rows.extend(summary.to_dict("records"))
            history = _frame_or_none(external_result.get("baseline_training_history"))
            if history is not None and (not history.empty or len(history.columns) > 0):
                training_history_frames.append(history)
            if external_result.get("status") == "completed":
                metrics[model_name] = dict(external_result["metrics"])
                returns_by_model[model_name] = external_result["daily_returns"].copy()
                for name in frames:
                    frames[name].append(external_result.get(name, pd.DataFrame()))
            continue
        strategy = factory(strategy_config)
        _assign_strategy_model_name(strategy, model_name)
        result = BacktestEngine(strategy_config, market_image_dataset=artifacts["market_image_dataset"]).run(
            artifacts["dataset"],
            artifacts["split"],
            strategy,
            segment=segment,
        )
        metrics[model_name] = dict(result.metrics)
        returns_by_model[model_name] = result.daily_returns.copy()
        frames["daily_returns"].append(result.daily_returns)
        frames["daily_weights"].append(result.daily_weights)
        frames["daily_turnover"].append(result.daily_turnover)
        frames["daily_rebalance"].append(result.daily_rebalance)
        frames["daily_costs"].append(result.daily_costs)
        frames["daily_asset_returns"].append(getattr(result, "daily_asset_returns", pd.DataFrame()))
        frames["baseline_daily_diagnostics"].append(result.baseline_daily_diagnostics)
        summary_row, history_frame = _baseline_training_artifacts(model_name, strategy, result.baseline_daily_diagnostics)
        _block_rankable_without_paper_model_id(model_name, result.baseline_daily_diagnostics, summary_row)
        training_summary_rows.append(summary_row)
        if history_frame is not None and (not history_frame.empty or len(history_frame.columns) > 0):
            training_history_frames.append(history_frame)

    if not frames["daily_returns"]:
        raise RuntimeError("ERR_EXPERIMENT_NO_COMPLETED_STRATEGY")

    paired = _paired_return_payload(returns_by_model, config, training_summary_rows=training_summary_rows)
    daily_returns = _concat(frames["daily_returns"])
    daily_weights = _concat(frames["daily_weights"])
    payload = {
        "status": "completed",
        "metrics": metrics,
        "daily_returns": daily_returns,
        "daily_weights": daily_weights,
        "daily_turnover": _concat(frames["daily_turnover"]),
        "daily_rebalance": _concat(frames["daily_rebalance"]),
        "daily_costs": _concat(frames["daily_costs"]),
        "daily_asset_returns": _concat(frames["daily_asset_returns"]),
        "baseline_daily_diagnostics": _concat(frames["baseline_daily_diagnostics"]),
        "baseline_comparison": _comparison_rows(metrics, training_summary_rows=training_summary_rows, config=config),
        "baseline_training_summary": pd.DataFrame(training_summary_rows),
        "baseline_training_history": _concat(training_history_frames) if training_history_frames else pd.DataFrame(),
        **paired,
        **artifact_payload(artifacts),
    }
    payload["availability_mask_contract"] = _availability_mask_contract_from_frames(daily_returns, daily_weights, artifacts["dataset"])
    payload.update(_new_model_artifacts(payload, config=config))
    return payload


def _run_backtest_with_artifacts(
    config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    strategy_factory: Any,
    *,
    model_name: str,
    split: SplitSpec,
    segment: str,
    run_dir: str | None = None,
) -> tuple[dict[str, Any], BacktestResult, Any]:
    strategy_config = strategy_runtime_config(config, artifacts["dataset"], artifacts["market_image_dataset"])
    strategy_config["model_name"] = model_name
    if run_dir is not None:
        strategy_config["baseline_run_dir"] = str(run_dir)
    strategy = strategy_factory(strategy_config)
    _assign_strategy_model_name(strategy, model_name)
    result = BacktestEngine(strategy_config, market_image_dataset=artifacts["market_image_dataset"]).run(
        artifacts["dataset"],
        split,
        strategy,
        segment=segment,
    )
    return strategy_config, result, strategy


def _unpack_backtest_output(output: Any) -> tuple[dict[str, Any], BacktestResult, Any | None]:
    if isinstance(output, tuple) and len(output) == 3:
        return output
    if isinstance(output, tuple) and len(output) == 2:
        strategy_config, result = output
        return strategy_config, result, None
    raise ValueError("ERR_BACKTEST_OUTPUT_CONTRACT")


def _train_and_backtest(
    config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    split: SplitSpec,
    model_name: str,
    test_split: str,
    run_dir: str | None,
) -> tuple[dict[str, Any], BacktestResult]:
    _assert_checkpoint_file_exists(config)
    runtime_config = strategy_runtime_config(config, artifacts["dataset"], artifacts["market_image_dataset"])
    runtime_config["model_name"] = model_name
    agent = build_hybrid_agent(runtime_config)
    train_env = _portfolio_env(artifacts, runtime_config, "train", split=split)
    validation_env = _portfolio_env(artifacts, runtime_config, "validation", split=split)
    checkpoint_paths = _checkpoint_paths(run_dir)
    _load_training_checkpoint(agent, runtime_config, train_env)
    _attach_checkpoint_callback(agent, checkpoint_paths, runtime_config, train_env)
    training_result = agent.train(train_env, validation_env=validation_env)
    last_checkpoint = _save_last_checkpoint(agent, checkpoint_paths, runtime_config, train_env)
    strategy = TrainedFullGatedStrategy(runtime_config, agent.ppo_agent)
    engine = BacktestEngine(runtime_config, market_image_dataset=artifacts["market_image_dataset"])
    backtest_result = engine.run(
        artifacts["dataset"],
        split,
        strategy,
        segment=test_split,
    )
    benchmark_result = engine.run(
        artifacts["dataset"],
        split,
        EqualWeightStrategy(runtime_config),
        segment=test_split,
    )
    runtime_artifacts = dict(artifacts)
    runtime_artifacts["split"] = split
    payload = result_mapping(
        backtest_result,
        config=runtime_config,
        artifacts=runtime_artifacts,
        status="completed",
        model_name=model_name,
    )
    payload["training_status"] = training_result["status"]
    payload["training_history"] = training_result.get("history", [])
    payload["best_validation_metric"] = training_result.get("best_validation_metric")
    payload["best_checkpoint_path"] = None if checkpoint_paths["best"] is None else str(checkpoint_paths["best"])
    payload["last_checkpoint_path"] = None if last_checkpoint is None else str(last_checkpoint)
    payload["checkpoint_count"] = int(sum(path is not None and path.exists() for path in checkpoint_paths.values()))
    payload["model_returns"] = {model_name: backtest_result.daily_returns}
    payload["benchmark_returns"] = {"equal_weight": benchmark_result.daily_returns}
    payload["main_comparison"] = _comparison_rows(
        {
            model_name: dict(backtest_result.metrics),
            "equal_weight": dict(benchmark_result.metrics),
        },
        primary_model=model_name,
        config=runtime_config,
    )
    return payload, backtest_result


def build_hybrid_agent(config: Mapping[str, Any]) -> HybridAgent:
    device = _device_from_config(config)
    model_class = _model_class(config)
    model = model_class(config).to(device)
    target_model = model_class(config).to(device)
    target_model.load_state_dict(model.state_dict())
    dqn_enabled = _dqn_enabled(config)
    auxiliary_enabled = _auxiliary_enabled(config)
    ppo_agent = PPOAgent(
        model.encoder,
        model.actor,
        model.critic,
        config=config,
        device=device,
        gate_network=model.gate if dqn_enabled else None,
        q_gap_threshold=float(getattr(model, "q_gap_threshold", 0.0)),
        policy_model=model,
    )
    online_gate, target_gate = _dqn_gate_modules(model, target_model)
    dqn_agent = None
    if dqn_enabled and online_gate is not None and target_gate is not None:
        model_config = _mapping(config.get("model"))
        use_risk_state = bool(model_config.get("use_risk_state", False))
        risk_state_dim = int(model_config.get("risk_state_dim", 8))
        dqn_agent = DQNAgent(
            online_gate,
            target_gate,
            config=config,
            device=device,
            encoder=model.encoder,
            target_encoder=target_model.encoder,
            use_risk_state=use_risk_state,
            risk_state_dim=risk_state_dim,
        )
        dqn_agent.hard_update_target_network(online_gate, target_gate)
    agent = HybridAgent(
        ppo_agent,
        dqn_agent=dqn_agent,
        auxiliary_heads=model.aux_heads if auxiliary_enabled else None,
        config=config,
    )
    agent.policy_model = model
    agent.target_policy_model = target_model
    return agent


def _dqn_enabled(config: Mapping[str, Any]) -> bool:
    dqn_config = config.get("dqn")
    if isinstance(dqn_config, Mapping):
        return bool(dqn_config.get("enabled", True))
    model_config = config.get("model")
    model_dqn = model_config.get("dqn") if isinstance(model_config, Mapping) else None
    if isinstance(model_dqn, Mapping):
        return bool(model_dqn.get("enabled", True))
    return True


def _auxiliary_enabled(config: Mapping[str, Any]) -> bool:
    auxiliary_config = config.get("auxiliary")
    if isinstance(auxiliary_config, Mapping):
        return bool(auxiliary_config.get("enabled", True))
    model_config = config.get("model")
    model_auxiliary = model_config.get("auxiliary") if isinstance(model_config, Mapping) else None
    if isinstance(model_auxiliary, Mapping):
        return bool(model_auxiliary.get("enabled", True))
    return True


def _model_class(config: Mapping[str, Any] | str) -> type[FullGatedModel]:
    if isinstance(config, str):
        model_config: Mapping[str, Any] = {"name": config}
        experiment_config: Mapping[str, Any] = {}
    else:
        model_config = config.get("model")
        experiment_config = config.get("experiment")
    configured_name = model_config.get("name") if isinstance(model_config, Mapping) else None
    experiment_type = experiment_config.get("type") if isinstance(experiment_config, Mapping) else None
    key = str(configured_name or experiment_type or "full_dqn_gated_multitask_cnn_ppo").lower()
    key = key.replace("-", "_")
    if "preference" in key:
        return PreferenceConditionedGatedPPO
    if "uncertainty" in key:
        return UncertaintyAwareGatedPPO
    if "distributional" in key or "cvar" in key:
        return DistributionalCVaRGatedPPO
    if "otar_cqr" in key or "cqr_gate" in key:
        return OTarCQRGate
    if "partial" in key:
        return PartialRebalanceGatedPPO
    return FullGatedModel


def _dqn_gate_modules(model: FullGatedModel, target_model: FullGatedModel) -> tuple[nn.Module | None, nn.Module | None]:
    if isinstance(model, PartialRebalanceGatedPPO) and model.mode == "continuous_beta":
        return None, None
    if isinstance(model, PartialRebalanceGatedPPO) and model.mode == "discrete_dqn":
        return (
            _DiscretePartialGateDQNAdapter(model.discrete_gate),
            _DiscretePartialGateDQNAdapter(target_model.discrete_gate),
        )
    if isinstance(model, UncertaintyAwareGatedPPO) and model.method in {"multi_head", "multihead"}:
        return (
            _UncertaintyGateDQNAdapter(model.uncertainty_heads),
            _UncertaintyGateDQNAdapter(target_model.uncertainty_heads),
        )
    if isinstance(model, PreferenceConditionedGatedPPO):
        return (
            _PreferenceGateDQNAdapter(model.conditioner, model.default_preference_omega, model.gate),
            _PreferenceGateDQNAdapter(target_model.conditioner, target_model.default_preference_omega, target_model.gate),
        )
    if isinstance(model, DistributionalCVaRGatedPPO):
        return (
            _DistributionalGateDQNAdapter(model),
            _DistributionalGateDQNAdapter(target_model),
        )
    if isinstance(model, OTarCQRGate):
        return (model.cqr_critic, target_model.cqr_critic)
    return model.gate, target_model.gate


class _DiscretePartialGateDQNAdapter(nn.Module):
    def __init__(self, discrete_gate: nn.Module):
        super().__init__()
        self.discrete_gate = discrete_gate
        self.output_dim = int(discrete_gate.n_rho)
        self.register_buffer("rho_values", discrete_gate.rho_values.detach().clone(), persistent=False)

    def forward(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
    ) -> torch.Tensor:
        _, q_values = self.discrete_gate(
            latent,
            candidate_weights,
            current_weights,
            estimated_turnover,
            estimated_cost,
        )
        return q_values


class _UncertaintyGateDQNAdapter(nn.Module):
    def __init__(self, uncertainty_heads: nn.Module):
        super().__init__()
        if uncertainty_heads is None:
            raise ValueError("ERR_UNCERTAINTY_CONFIG_INVALID: uncertainty_heads is required")
        self.uncertainty_heads = uncertainty_heads
        self.output_dim = int(uncertainty_heads.gate_output_dim)

    def forward(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
    ) -> torch.Tensor:
        q_values = [
            gate(latent, candidate_weights, current_weights, estimated_turnover, estimated_cost)
            for gate in self.uncertainty_heads.gate_heads
        ]
        return torch.stack(q_values, dim=0).mean(dim=0)


class _PreferenceGateDQNAdapter(nn.Module):
    def __init__(self, conditioner: nn.Module, default_omega: torch.Tensor, gate: nn.Module):
        super().__init__()
        self.conditioner = conditioner
        self.gate = gate
        self.output_dim = int(gate.output_dim)
        self.register_buffer("default_omega", default_omega.detach().clone(), persistent=False)

    def forward(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
    ) -> torch.Tensor:
        omega = self.default_omega.to(device=latent.device, dtype=latent.dtype).view(1, -1).expand(latent.shape[0], -1)
        conditioned = self.conditioner(latent, omega)
        return self.gate(conditioned, candidate_weights, current_weights, estimated_turnover, estimated_cost)


class _DistributionalGateDQNAdapter(nn.Module):
    def __init__(self, model: DistributionalCVaRGatedPPO):
        super().__init__()
        self.model = model
        self.output_dim = int(model.gate.output_dim)

    def forward(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
    ) -> torch.Tensor:
        base_q = self.model.gate(latent, candidate_weights, current_weights, estimated_turnover, estimated_cost)
        candidate_quantiles = self.model.candidate_dist_critic(latent)
        hold_quantiles = self.model.hold_dist_critic(latent)
        candidate_expected_value = self.model.candidate_dist_critic.expected_value(candidate_quantiles)
        hold_expected_value = self.model.hold_dist_critic.expected_value(hold_quantiles)
        candidate_cvar = self.model.candidate_dist_critic.get_cvar(candidate_quantiles, self.model.cvar_alpha)
        hold_cvar = self.model.hold_dist_critic.get_cvar(hold_quantiles, self.model.cvar_alpha)
        candidate_tail_loss = self.model.candidate_dist_critic.get_tail_loss(candidate_quantiles, self.model.cvar_alpha)
        hold_tail_loss = self.model.hold_dist_critic.get_tail_loss(hold_quantiles, self.model.cvar_alpha)
        delta_u = (candidate_expected_value - candidate_tail_loss) - (hold_expected_value - hold_tail_loss)
        risk_features = torch.cat(
            [
                candidate_expected_value,
                hold_expected_value,
                candidate_cvar,
                hold_cvar,
                candidate_tail_loss,
                hold_tail_loss,
                delta_u,
            ],
            dim=1,
        )
        return base_q + self.model.gate_risk_head(risk_features)


class TrainedFullGatedStrategy(BaseStrategy):
    strategy_name = "full_dqn_gated_multitask_cnn_ppo"
    requires_daily_diagnostics = True

    def __init__(self, config: Mapping[str, Any], ppo_agent: PPOAgent):
        super().__init__(config)
        self.ppo_agent = ppo_agent

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        portfolio = self.validate_portfolio_state(portfolio_state)
        action_info = self._select_action(state, portfolio)
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=np.asarray(action_info["candidate_weights"], dtype=float),
                rebalance_action=int(action_info["gate_action"]),
                rebalance_intensity=float(action_info["rebalance_intensity"]),
                action_info={
                    "strategy": self.strategy_name,
                    "paper_model_id": self.strategy_name,
                    "child_model_name": self.strategy_name,
                    "baseline_family": "platform_native_rl",
                    "platform_native_rl_training": True,
                    "gate_action": int(action_info["gate_action"]),
                    "q_hold": action_info["q_hold"],
                    "q_rebalance": action_info["q_rebalance"],
                    "q_gap": action_info["q_gap"],
                    "estimated_turnover": action_info["estimated_turnover"],
                    "estimated_cost": action_info["estimated_cost"],
                    "candidate_log_prob": action_info["log_prob"],
                    "decision_value": action_info["value"],
                },
            )
        )

    @torch.no_grad()
    def _select_action(self, state: DecisionMarketState, portfolio: PortfolioState) -> dict[str, Any]:
        observation = {
            "market_image": np.asarray(state.market_image, dtype=np.float32),
            "availability_mask": np.asarray(state.available_mask_at_decision, dtype=bool),
            "current_weights": np.asarray(portfolio.current_weights, dtype=np.float32),
            "adv20_at_decision": np.asarray(state.adv20_at_decision, dtype=np.float32),
            "volatility_20d_at_decision": np.asarray(state.volatility_20d_at_decision, dtype=np.float32),
            "amount_at_decision": np.asarray(state.amount_at_decision, dtype=np.float32),
            "turnover_rate_at_decision": np.asarray(state.turnover_rate_at_decision, dtype=np.float32),
            "portfolio_value": np.asarray(portfolio.portfolio_value, dtype=np.float32),
        }
        model_config = mapping(self.config.get("model"))
        if bool(model_config.get("use_risk_state", False)):
            if portfolio.risk_state_vector is None:
                raise DataContractError(
                    "ERR_OBSERVATION_RISK_STATE_MISSING",
                    "ERR_OBSERVATION_RISK_STATE_MISSING: trained strategy portfolio_state.risk_state_vector",
                )
            observation["risk_state"] = np.asarray(portfolio.risk_state_vector, dtype=np.float32)
        return self.ppo_agent.select_action(observation, deterministic=True)


def build_pipeline_artifacts(config: Mapping[str, Any], split_override: SplitSpec | None = None) -> dict[str, Any]:
    dataset = load_market_dataset(config)
    date_index = pd.DatetimeIndex(dataset.wide["close"].index)
    split = create_split(date_index, config)
    primary_split = split_override if split_override is not None else split[0] if isinstance(split, list) else split
    feature_config = feature_pipeline_config(config)
    fallback_reason = None
    fallback_provenance = None
    fallback_audit_sample = None
    try:
        feature_matrix = FeatureMatrixBuilder(feature_config).build(dataset, primary_split, feature_config)
    except DataContractError as exc:
        if exc.code != "ERR_METRICS_FACTORY_AUDIT_FAILED":
            raise
        fallback_reason = str(exc)
        fallback_provenance = _fallback_provenance_from_error(exc)
        fallback_audit_sample = _fallback_audit_sample_from_error(exc)
        feature_config = fallback_feature_pipeline_config(config)
        feature_matrix = FeatureMatrixBuilder(feature_config).build(dataset, primary_split, feature_config)
        feature_matrix = _merge_fallback_metrics_audit(
            feature_matrix,
            fallback_provenance,
            fallback_audit_sample,
        )
    train_panel = _panel_for_dates(feature_matrix.feature_panel, primary_split.train_dates)
    reduction = FeatureReductionPipeline(feature_config)
    reduced_train = reduction.fit_transform(
        train_panel,
        feature_matrix.feature_cols,
        split=primary_split,
        validation_dates=primary_split.validation_dates,
        test_dates=primary_split.test_dates,
        auxiliary_target_cols=dataset.auxiliary_target_cols,
        wide_log_return=dataset.wide["log_return"],
    )
    reduced_panel = reduction.transform(feature_matrix.feature_panel)
    reduced_feature_matrix = FeatureMatrix(
        feature_panel=reduced_panel,
        feature_cols=list(reduction.feature_cols_),
        provenance=feature_matrix.provenance,
        feature_group_summary=feature_matrix.feature_group_summary,
        metrics_factory_audit_sample=feature_matrix.metrics_factory_audit_sample,
    )
    market_image_dataset = MarketImageDataset(
        reduced_feature_matrix,
        window_size=int(feature_config["feature_matrix"]["window_size"]),
        asset_order=asset_order(dataset),
        date_index=date_index,
        materialize_market_images=False,
    )
    return {
        "dataset": dataset,
        "split": primary_split,
        "all_splits": split,
        "feature_matrix": feature_matrix,
        "feature_reduction": reduction,
        "reduced_train_rows": len(reduced_train),
        "market_image_dataset": market_image_dataset,
        "feature_config": feature_config,
        "requested_feature_config": feature_pipeline_config(config),
        "feature_fallback_reason": fallback_reason,
    }


def feature_pipeline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    feature_matrix = dict(resolved.get("feature_matrix", {}))
    feature_matrix.setdefault("window_size", resolved.get("env", {}).get("window_size", 60))
    resolved["feature_matrix"] = feature_matrix
    return resolved


def fallback_feature_pipeline_config(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = feature_pipeline_config(config)
    feature_matrix = dict(resolved.get("feature_matrix", {}))
    feature_matrix["input_matrix_id"] = "M0"
    resolved["feature_matrix"] = feature_matrix
    feature_reduction = dict(resolved.get("feature_reduction", {}))
    pca = dict(feature_reduction.get("pca", {}))
    pca["enabled"] = False
    feature_reduction["pca"] = pca
    resolved["feature_reduction"] = feature_reduction
    return resolved


def _fallback_provenance_from_error(exc: DataContractError) -> pd.DataFrame:
    provenance = getattr(exc, "provenance", None)
    if not isinstance(provenance, pd.DataFrame) or provenance.empty:
        return pd.DataFrame(columns=FEATURE_PROVENANCE_COLUMNS)
    result = provenance.copy()
    for column in FEATURE_PROVENANCE_COLUMNS:
        if column not in result:
            result[column] = None
    audit_sample = _fallback_audit_sample_from_error(exc)
    failed_features = set()
    if not audit_sample.empty and "status" in audit_sample and "feature_name" in audit_sample:
        failed_features = set(audit_sample.loc[audit_sample["status"] == "fail", "feature_name"].astype(str))
    metrics_mask = result["source_family"].astype(str).eq("metrics_factory")
    result.loc[metrics_mask, "is_model_feature"] = False
    result.loc[metrics_mask & result["drop_reason"].astype(str).eq(""), "drop_reason"] = (
        "runtime_fallback_metrics_factory_audit_failed"
    )
    fail_mask = result["feature_name"].astype(str).isin(failed_features)
    result.loc[fail_mask, "leakage_check_status"] = "fail"
    result.loc[fail_mask, "drop_reason"] = "metrics_factory_audit_failed"
    return result.loc[:, FEATURE_PROVENANCE_COLUMNS]


def _fallback_audit_sample_from_error(exc: DataContractError) -> pd.DataFrame:
    sample = getattr(exc, "metrics_factory_audit_sample", None)
    if not isinstance(sample, pd.DataFrame) or sample.empty:
        return pd.DataFrame(columns=METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS)
    result = sample.copy()
    for column in METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS:
        if column not in result:
            result[column] = None
    return result.loc[:, METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS]


def _merge_fallback_metrics_audit(
    fallback_matrix: FeatureMatrix,
    primary_provenance: pd.DataFrame | None,
    primary_audit_sample: pd.DataFrame | None,
) -> FeatureMatrix:
    provenance_frames = [
        frame
        for frame in (primary_provenance, fallback_matrix.provenance)
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    audit_frames = [
        frame
        for frame in (primary_audit_sample, fallback_matrix.metrics_factory_audit_sample)
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    provenance = _concat(provenance_frames) if provenance_frames else fallback_matrix.provenance
    audit_sample = _concat(audit_frames) if audit_frames else fallback_matrix.metrics_factory_audit_sample
    return FeatureMatrix(
        feature_panel=fallback_matrix.feature_panel,
        feature_cols=fallback_matrix.feature_cols,
        provenance=provenance.loc[:, FEATURE_PROVENANCE_COLUMNS]
        if not provenance.empty
        else pd.DataFrame(columns=FEATURE_PROVENANCE_COLUMNS),
        feature_group_summary=_feature_group_summary_from_provenance(provenance),
        metrics_factory_audit_sample=audit_sample.loc[:, METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS]
        if not audit_sample.empty
        else pd.DataFrame(columns=METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS),
    )


def _feature_group_summary_from_provenance(provenance: pd.DataFrame) -> pd.DataFrame:
    if provenance.empty:
        return pd.DataFrame(columns=FEATURE_GROUP_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for (feature_group, source_family), group in provenance.groupby(["feature_group", "source_family"], sort=True):
        rows.append(
            {
                "feature_group": feature_group,
                "source_family": source_family,
                "n_total": int(len(group)),
                "n_used": int(group["is_model_feature"].astype(bool).sum()),
                "n_dropped": int((~group["is_model_feature"].astype(bool)).sum()),
                "n_shifted": int(group["requires_shift"].astype(bool).sum()),
                "n_train_only_fit": int((group["fit_scope"] == "train_only").sum()),
                "n_warning": int((group["leakage_check_status"] == "warning").sum()),
                "n_fail": int((group["leakage_check_status"] == "fail").sum()),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_GROUP_SUMMARY_COLUMNS)


def strategy_runtime_config(
    config: Mapping[str, Any],
    dataset: MarketDatasetBundle,
    market_image_dataset: Any | None = None,
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    window_size = int(resolved.get("env", {}).get("window_size", resolved.get("feature_matrix", {}).get("window_size", 60)))
    resolved["n_assets"] = len(asset_order(dataset))
    resolved["n_features"] = len(getattr(market_image_dataset, "feature_cols", []) or ["log_return"])
    resolved["window_size"] = window_size
    env_config = dict(resolved.get("env", {}))
    env_config["n_assets"] = resolved["n_assets"]
    env_config["n_features"] = resolved["n_features"]
    env_config["window_size"] = window_size
    resolved["env"] = env_config
    feature_matrix = dict(resolved.get("feature_matrix", {}))
    feature_matrix["window_size"] = window_size
    resolved["feature_matrix"] = feature_matrix
    return resolved


def result_mapping(
    result: BacktestResult,
    *,
    config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    status: str,
    model_name: str,
) -> dict[str, Any]:
    metrics = dict(result.metrics)
    payload = {
        "status": status,
        "model_name": model_name,
        "metrics": metrics,
        "daily_returns": _with_model_name(result.daily_returns, model_name),
        "daily_weights": _with_model_name(result.daily_weights, model_name),
        "daily_turnover": _with_model_name(result.daily_turnover, model_name),
        "daily_rebalance": _with_model_name(result.daily_rebalance, model_name),
        "daily_costs": _with_model_name(result.daily_costs, model_name),
        "daily_asset_returns": _with_model_name(getattr(result, "daily_asset_returns", pd.DataFrame()), model_name),
        "baseline_daily_diagnostics": _baseline_daily_diagnostics(result),
        "run_manifest": result.run_manifest,
        "v_init_resolved_value": result.run_manifest.get("v_init_resolved_value"),
        **_availability_mask_contract_payload(result, artifacts),
        "training_status": "not_applicable_for_backtest_pipeline",
        "device": config.get("device"),
        **artifact_payload(artifacts),
    }
    payload.update(_new_model_artifacts(payload, config=config))
    return payload


def _with_model_name(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    result = frame.copy()
    if "model_name" in result.columns:
        result["model_name"] = str(model_name)
    return result


def _availability_mask_contract_payload(
    result: BacktestResult,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = artifacts.get("dataset")
    if dataset is None:
        return {}
    return {"availability_mask_contract": _availability_mask_contract(result, dataset)}


def _availability_mask_contract(result: BacktestResult, dataset: MarketDatasetBundle) -> dict[str, Any]:
    return _availability_mask_contract_from_frames(result.daily_returns, result.daily_weights, dataset)


def _availability_mask_contract_from_frames(
    daily_returns: pd.DataFrame,
    daily_weights: pd.DataFrame,
    dataset: MarketDatasetBundle,
) -> dict[str, Any]:
    availability = dataset.availability_mask.astype(bool)
    min_available = int(availability.sum(axis=1).min()) if not availability.empty else 0
    reason_counts: dict[str, int] = {}
    if dataset.availability_reason is not None and not dataset.availability_reason.empty:
        reasons = pd.Series(dataset.availability_reason.to_numpy(dtype=object).ravel())
        reason_counts = {
            str(key): int(value)
            for key, value in reasons.value_counts(dropna=False).sort_index().items()
        }

    daily_returns_finite = _finite_columns(daily_returns, ("net_return", "portfolio_log_return"))
    daily_nav_finite = _finite_columns(daily_returns, ("nav",))
    unavailable_weight_abs_max = _unavailable_weight_abs_max(daily_weights, availability)
    frozen_or_imputed_valuation_count = 0
    passed = (
        min_available >= 2
        and unavailable_weight_abs_max <= 1.0e-12
        and daily_returns_finite
        and daily_nav_finite
        and frozen_or_imputed_valuation_count == 0
    )
    return {
        "availability_mask_contract_passed": bool(passed),
        "min_available_assets_per_date": min_available,
        "unavailable_asset_weight_abs_max": float(unavailable_weight_abs_max),
        "daily_returns_finite": bool(daily_returns_finite),
        "daily_nav_finite": bool(daily_nav_finite),
        "frozen_or_imputed_valuation_count": frozen_or_imputed_valuation_count,
        "availability_reason_counts": reason_counts,
    }


def _finite_columns(frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    for column in columns:
        if column not in frame.columns:
            return False
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.empty or not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            return False
    return True


def _unavailable_weight_abs_max(daily_weights: pd.DataFrame, availability: pd.DataFrame) -> float:
    if daily_weights.empty or availability.empty:
        return float("inf")
    max_weight = 0.0
    for row in daily_weights.itertuples(index=False):
        date = pd.Timestamp(getattr(row, "date"))
        asset_id = str(getattr(row, "asset_id"))
        if date not in availability.index or asset_id not in availability.columns:
            continue
        if not bool(availability.loc[date, asset_id]):
            max_weight = max(max_weight, abs(float(getattr(row, "weight"))))
    return float(max_weight)


def artifact_payload(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    dataset = artifacts["dataset"]
    feature_matrix = artifacts["feature_matrix"]
    reduction = artifacts["feature_reduction"]
    split = artifacts["split"]
    market_image_dataset = artifacts["market_image_dataset"]
    return {
        "canonical_asset_order": asset_order(dataset),
        "asset_list": asset_order(dataset),
        "data_split": split_to_dict(split),
        "feature_provenance_report": feature_matrix.provenance,
        "feature_group_summary": feature_matrix.feature_group_summary,
        "metrics_factory_audit_sample": feature_matrix.metrics_factory_audit_sample,
        "selected_input_matrix": {
            "input_matrix_id": artifacts["feature_config"]["feature_matrix"]["input_matrix_id"],
            "requested_input_matrix_id": artifacts["requested_feature_config"]["feature_matrix"].get("input_matrix_id"),
            "fallback_reason": artifacts.get("feature_fallback_reason"),
            "runtime_market_image_features": market_image_dataset.feature_cols,
            "market_image_date_count": len(market_image_dataset),
            "reduced_train_rows": artifacts["reduced_train_rows"],
        },
        "input_matrix_feature_groups": {
            "feature_count": len(feature_matrix.feature_cols),
            "runtime_feature_count": len(market_image_dataset.feature_cols),
        },
        "pca_report": {
            "enabled": getattr(reduction, "pca_", None) is not None,
            "n_components": None if getattr(reduction, "pca_", None) is None else int(reduction.pca_.n_components_),
            "status": "completed" if getattr(reduction, "is_fitted", False) else "not_fitted",
        },
    }


def _new_model_artifacts(payload: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, Any]:
    model_extension_id = _configured_model_extension_id(config)
    diagnostics = _new_model_diagnostics(payload.get("baseline_daily_diagnostics"), model_extension_id=model_extension_id)
    if diagnostics.empty:
        return {}
    asset_ids = [str(item) for item in payload.get("canonical_asset_order", [])]
    gate_actions = _gate_actions_frame(diagnostics)
    result = {
        "gate_actions": gate_actions,
        "gate_action_summary": _gate_action_summary(gate_actions, model_extension_id=model_extension_id),
        "cage_eiie_candidate_weights": _weights_sidecar(diagnostics, "candidate_weights_json", asset_ids, "candidate_weight"),
        "cage_final_weights": _weights_sidecar(diagnostics, "executed_weights_json", asset_ids, "executed_weight"),
        "turnover_cost_breakdown": _turnover_cost_breakdown(diagnostics),
        "risk_metrics": _risk_metrics(payload.get("daily_returns"), diagnostics, model_extension_id=model_extension_id),
        "validation_selection_report": _validation_selection_report(payload, config, diagnostics, model_extension_id=model_extension_id),
        "new_model_sidecar_manifest": _new_model_sidecar_manifest(config, gate_actions, model_extension_id=model_extension_id),
    }
    result.update(_ra_gt_rcpo_artifacts(payload, diagnostics))
    return {key: value for key, value in result.items() if not (_is_empty_frame(value))}


def _configured_model_extension_id(config: Mapping[str, Any]) -> str:
    new_model = mapping(config.get("new_model_protocol"))
    return str(new_model.get("model_extension_id") or MODEL_EXTENSION_ID)


def _new_model_diagnostics(value: Any, *, model_extension_id: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        return pd.DataFrame()
    frame = value.copy()
    if "model_extension_id" in frame.columns:
        mask = frame["model_extension_id"].fillna("").astype(str).eq(str(model_extension_id))
    elif "paper_model_id" in frame.columns:
        mask = frame["paper_model_id"].fillna("").astype(str).isin(PAPER_NEW_MODEL_NAMES)
    else:
        mask = pd.Series(False, index=frame.index)
    if str(model_extension_id) == MODEL_EXTENSION_ID and "rho" in frame.columns and "paper_model_id" in frame.columns:
        mask = mask | frame["paper_model_id"].fillna("").astype(str).isin(PAPER_NEW_MODEL_NAMES)
    return frame.loc[mask].copy()


def _ra_gt_rcpo_artifacts(payload: Mapping[str, Any], diagnostics: pd.DataFrame) -> dict[str, Any]:
    if diagnostics.empty or "paper_model_id" not in diagnostics.columns:
        return {}
    mask = diagnostics["paper_model_id"].fillna("").astype(str).isin(P16_RA_GT_RCPO_MODEL_NAMES)
    frame = diagnostics.loc[mask].copy()
    if frame.empty:
        return {}
    return {
        "ra_gt_rcpo_daily_diagnostics": _ra_gt_rcpo_daily_diagnostics(frame),
        "ra_gt_rcpo_constraint_multipliers": _ra_gt_rcpo_constraint_multipliers(frame),
        "ra_gt_rcpo_graph_diagnostics": _ra_gt_rcpo_graph_diagnostics(frame),
        "ra_gt_rcpo_actor_critic_training_history": _ra_gt_rcpo_training_history(payload, frame),
        "ra_gt_rcpo_risk_decomposition": _ra_gt_rcpo_risk_decomposition(frame),
    }


def _ra_gt_rcpo_daily_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "model_name",
        "seed",
        "fold_id",
        "rho",
        "rebalance_intensity",
        "scheduler_allowed_rebalance",
        "estimated_turnover",
        "realized_turnover",
        "estimated_cost",
        "realized_cost",
        "CVaR_loss_5",
        "max_drawdown_loss",
        "lambda_turnover",
        "lambda_cost",
        "lambda_cvar",
        "lambda_drawdown",
        "graph_feature_mode",
        "constraint_violation_count",
        "model_extension_id",
    ]
    frame = diagnostics.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns]


def _ra_gt_rcpo_constraint_multipliers(diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "model_name",
        "seed",
        "fold_id",
        "lambda_turnover",
        "lambda_cost",
        "lambda_cvar",
        "lambda_drawdown",
        "average_turnover_per_step_budget",
        "average_cost_per_step_budget",
        "cvar_loss_budget",
        "drawdown_budget",
        "constraint_violation_count",
        "model_extension_id",
    ]
    frame = diagnostics.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns]


def _ra_gt_rcpo_graph_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "model_name",
        "seed",
        "fold_id",
        "graph_feature_mode",
        "graph_edge_threshold",
        "graph_density",
        "mean_abs_correlation",
        "model_extension_id",
    ]
    frame = diagnostics.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns]


def _ra_gt_rcpo_training_history(payload: Mapping[str, Any], diagnostics: pd.DataFrame) -> pd.DataFrame:
    history = _frame_or_none(payload.get("baseline_training_history"))
    if history is None or history.empty:
        return pd.DataFrame()
    model_ids = set(diagnostics["model_name"].dropna().astype(str)) if "model_name" in diagnostics.columns else set()
    result = history.copy()
    if model_ids and "model_name" in result.columns:
        result = result.loc[result["model_name"].astype(str).isin(model_ids)].copy()
    return result


def _ra_gt_rcpo_risk_decomposition(diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "model_name",
        "seed",
        "fold_id",
        "value_return",
        "value_cost",
        "value_drawdown",
        "value_cvar_loss",
        "CVaR_loss_5",
        "max_drawdown_loss",
        "net_return",
        "portfolio_log_return",
        "nav",
        "model_extension_id",
    ]
    frame = diagnostics.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns]


def _gate_actions_frame(diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "decision_date",
        "execution_date",
        "model_name",
        "paper_model_id",
        "seed",
        "fold_id",
        "gate_action",
        "gate_action_index",
        "rho",
        "rebalance_intensity",
        "rebalance_values",
        "scheduler_allowed_rebalance",
        "forced_hold_reason",
        "estimated_turnover",
        "realized_turnover",
        "estimated_cost",
        "realized_cost",
        "CVaR_loss_5",
        "drawdown",
        "model_extension_id",
    ]
    frame = diagnostics.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns]


def _weights_sidecar(diagnostics: pd.DataFrame, json_column: str, asset_ids: Sequence[str], value_column: str) -> pd.DataFrame:
    if json_column not in diagnostics.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for record in diagnostics.to_dict("records"):
        weights = _parse_weights_json(record.get(json_column))
        if not weights:
            continue
        names = list(asset_ids)
        if len(names) != len(weights):
            names = [f"asset_{index}" for index in range(len(weights))]
        for asset_id, weight in zip(names, weights, strict=False):
            rows.append(
                {
                    "date": record.get("date"),
                    "decision_date": record.get("decision_date"),
                    "execution_date": record.get("execution_date"),
                    "model_name": record.get("model_name"),
                    "paper_model_id": record.get("paper_model_id"),
                    "seed": record.get("seed"),
                    "fold_id": record.get("fold_id"),
                    "asset_id": asset_id,
                    value_column: float(weight),
                    "rho": record.get("rho"),
                    "model_extension_id": record.get("model_extension_id"),
                }
            )
    return pd.DataFrame(rows)


def _parse_weights_json(value: Any) -> list[float]:
    if value is None or pd.isna(value):
        return []
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return []
    result: list[float] = []
    for item in payload:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return []
    return result


def _turnover_cost_breakdown(diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "decision_date",
        "execution_date",
        "model_name",
        "paper_model_id",
        "seed",
        "fold_id",
        "rho",
        "candidate_turnover",
        "estimated_turnover",
        "realized_turnover",
        "turnover",
        "estimated_cost",
        "realized_cost",
        "total_transaction_cost",
        "model_extension_id",
    ]
    frame = diagnostics.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, columns]


def _risk_metrics(daily_returns_value: Any, diagnostics: pd.DataFrame, *, model_extension_id: str) -> pd.DataFrame:
    if not isinstance(daily_returns_value, pd.DataFrame) or daily_returns_value.empty:
        return pd.DataFrame()
    daily_returns = daily_returns_value.copy()
    if "model_name" not in daily_returns.columns or "net_return" not in daily_returns.columns:
        return pd.DataFrame()
    model_names = set(diagnostics["model_name"].dropna().astype(str)) if "model_name" in diagnostics.columns else set()
    if model_names:
        daily_returns = daily_returns.loc[daily_returns["model_name"].astype(str).isin(model_names)].copy()
    rows: list[dict[str, Any]] = []
    group_cols = [column for column in ("model_name", "seed", "fold_id") if column in daily_returns.columns]
    for keys, group in daily_returns.groupby(group_cols, dropna=False, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        key_map = dict(zip(group_cols, key_values, strict=False))
        returns = pd.to_numeric(group["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        nav = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(nav) if nav.size else np.array([], dtype=float)
        drawdown = np.divide(running_max - nav, running_max, out=np.zeros_like(nav), where=running_max > 0.0)
        tail_n = max(1, int(np.ceil(0.05 * returns.size))) if returns.size else 1
        cvar_loss_5 = float(max(0.0, -np.mean(np.sort(returns)[:tail_n]))) if returns.size else np.nan
        rows.append(
            {
                **key_map,
                "paper_model_id": _paper_id_for_model(diagnostics, str(key_map.get("model_name", ""))),
                "n_steps": int(returns.size),
                "cumulative_return": float(nav[-1] - 1.0) if nav.size else np.nan,
                "max_drawdown_loss": float(np.max(drawdown)) if drawdown.size else np.nan,
                "CVaR_loss_5": cvar_loss_5,
                "mean_net_return": float(np.mean(returns)) if returns.size else np.nan,
                "volatility": float(np.std(returns, ddof=0)) if returns.size else np.nan,
                "model_extension_id": model_extension_id,
            }
        )
    return pd.DataFrame(rows)


def _paper_id_for_model(diagnostics: pd.DataFrame, model_name: str) -> Any:
    if "model_name" not in diagnostics.columns or "paper_model_id" not in diagnostics.columns:
        return model_name
    selected = diagnostics.loc[diagnostics["model_name"].astype(str).eq(model_name), "paper_model_id"].dropna()
    return model_name if selected.empty else selected.iloc[0]


def _gate_action_summary(gate_actions: pd.DataFrame, *, model_extension_id: str) -> pd.DataFrame:
    if gate_actions.empty or "model_name" not in gate_actions.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [column for column in ("model_name", "paper_model_id", "seed", "fold_id") if column in gate_actions.columns]
    for keys, group in gate_actions.groupby(group_cols, dropna=False, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        rho = pd.to_numeric(group.get("rho"), errors="coerce")
        gate_action = pd.to_numeric(group.get("gate_action"), errors="coerce")
        forced = group.get("forced_hold_reason", pd.Series([pd.NA] * len(group))).fillna("").astype(str)
        rows.append(
            {
                **dict(zip(group_cols, key_values, strict=False)),
                "n_decisions": int(len(group)),
                "mean_rho": float(rho.mean()) if not rho.dropna().empty else np.nan,
                "rebalance_decision_rate": float((gate_action > 0).mean()) if not gate_action.dropna().empty else np.nan,
                "scheduler_forced_hold_count": int(forced.eq("scheduler_blocked").sum()),
                "model_chosen_hold_count": int(forced.eq("model_chosen_hold").sum()),
                "model_extension_id": model_extension_id,
            }
        )
    return pd.DataFrame(rows)


def _validation_selection_report(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    diagnostics: pd.DataFrame,
    *,
    model_extension_id: str,
) -> pd.DataFrame:
    hpo_cfg = mapping(config.get("hpo"))
    summary = payload.get("baseline_training_summary")
    allowed_ids = set()
    for column in ("model_name", "paper_model_id"):
        if column in diagnostics.columns:
            allowed_ids.update(diagnostics[column].dropna().astype(str).tolist())
    rows: list[dict[str, Any]] = []
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        source = summary
    else:
        source = diagnostics.drop_duplicates(["model_name"]) if "model_name" in diagnostics.columns else pd.DataFrame()
    for record in source.to_dict("records"):
        model_name = str(record.get("model_name") or record.get("paper_model_id") or "")
        if not model_name:
            continue
        paper_model_id = str(record.get("paper_model_id", model_name))
        if allowed_ids and model_name not in allowed_ids and paper_model_id not in allowed_ids:
            continue
        rows.append(
            {
                "model_name": model_name,
                "paper_model_id": record.get("paper_model_id", model_name),
                "selection_split": hpo_cfg.get("selection_split", "validation"),
                "final_report_split": hpo_cfg.get("final_report_split", "test"),
                "test_used_for_model_selection": False,
                "validation_only_promotion_gate": True,
                "best_validation_metric": record.get("best_validation_metric", payload.get("best_validation_metric")),
                "best_trial_number": payload.get("best_trial_number"),
                "best_value": payload.get("best_value"),
                "model_extension_id": model_extension_id,
            }
        )
    return pd.DataFrame(rows)


def _new_model_sidecar_manifest(config: Mapping[str, Any], gate_actions: pd.DataFrame, *, model_extension_id: str) -> dict[str, Any]:
    protocol = mapping(config.get("protocol"))
    new_model = mapping(config.get("new_model_protocol"))
    artifact_names = [
        "gate_actions",
        "gate_action_summary",
        "cage_eiie_candidate_weights",
        "cage_final_weights",
        "turnover_cost_breakdown",
        "risk_metrics",
        "validation_selection_report",
    ]
    if model_extension_id == RA_GT_RCPO_MODEL_EXTENSION_ID:
        artifact_names.extend(
            [
                "ra_gt_rcpo_daily_diagnostics",
                "ra_gt_rcpo_constraint_multipliers",
                "ra_gt_rcpo_graph_diagnostics",
                "ra_gt_rcpo_actor_critic_training_history",
                "ra_gt_rcpo_risk_decomposition",
            ]
        )
    return {
        "base_protocol_id": new_model.get("base_protocol_id", protocol.get("protocol_id")),
        "protocol_id": protocol.get("protocol_id"),
        "model_extension_id": model_extension_id,
        "post_hoc_development_disclosure": True,
        "test_used_for_model_selection": False,
        "selection_split": mapping(config.get("hpo")).get("selection_split", new_model.get("selection_split", "validation")),
        "artifact_names": artifact_names,
        "gate_action_rows": int(len(gate_actions)),
    }


def _is_empty_frame(value: Any) -> bool:
    return isinstance(value, pd.DataFrame) and value.empty and len(value.columns) == 0


def objective_metric(result: Mapping[str, Any], metric: str, config: Mapping[str, Any] | None = None) -> float:
    metric_key = VALIDATION_METRIC_ALIASES.get(str(metric), str(metric))
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    if metric_key == "validation_return_risk_cost_constrained":
        return _validation_return_risk_cost_constrained(result, config=config)
    if metric_key in result:
        return float(result[metric_key])
    if metric_key in metrics:
        return float(metrics[metric_key])
    if metric_key in {"validation_sharpe_minus_drawdown_turnover_penalty", "net_sharpe"}:
        daily_returns = result.get("daily_returns")
        if isinstance(daily_returns, pd.DataFrame):
            returns = pd.to_numeric(daily_returns["net_return"], errors="coerce").dropna()
            if not returns.empty:
                std = float(returns.std(ddof=0))
                sharpe = 0.0 if std <= 0.0 else float(returns.mean() / std * np.sqrt(252.0))
                nav = pd.to_numeric(daily_returns.get("nav"), errors="coerce").dropna()
                max_drawdown = _max_drawdown(nav)
                average_turnover = float(metrics.get("average_turnover", 0.0) or 0.0)
                return sharpe - max_drawdown - average_turnover
    if "cvar95_loss" in metric_key or "cvar_loss" in metric_key:
        daily_returns = result.get("daily_returns")
        if isinstance(daily_returns, pd.DataFrame):
            returns = pd.to_numeric(daily_returns.get("net_return", daily_returns.get("return")), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not returns.empty:
                return _cvar_loss(returns)
    if "max_drawdown" in metric_key:
        daily_returns = result.get("daily_returns")
        if isinstance(daily_returns, pd.DataFrame):
            nav = pd.to_numeric(daily_returns.get("nav"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not nav.empty:
                return _max_drawdown(nav)
    if "sortino" in metric_key:
        daily_returns = result.get("daily_returns")
        if isinstance(daily_returns, pd.DataFrame):
            returns = pd.to_numeric(daily_returns.get("net_return", daily_returns.get("return")), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not returns.empty:
                downside = returns[returns < 0.0]
                downside_std = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
                ann_return = float(returns.mean() * 252.0)
                ann_vol = float(downside_std * np.sqrt(252.0))
                return 0.0 if ann_vol <= 0.0 else ann_return / ann_vol
    raise ValueError(f"ERR_EXPERIMENT_METRIC_MISSING: {metric_key}")


def _validation_return_risk_cost_constrained(
    result: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> float:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    daily_returns = result.get("daily_returns")
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty or "net_return" not in daily_returns.columns:
        raise ValueError("failed_metric_unavailable")
    returns = pd.to_numeric(daily_returns["net_return"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        raise ValueError("failed_metric_unavailable")
    std = float(returns.std(ddof=0))
    sharpe = 0.0 if std <= 0.0 else float(returns.mean() / std * np.sqrt(252.0))
    nav = pd.to_numeric(daily_returns.get("nav"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    max_drawdown_loss = _max_drawdown(nav)
    cvar_loss_5 = _cvar_loss(returns)
    constraints = _activity_constraints(config)
    model_rebalance_hit_rate = float(metrics.get("model_rebalance_hit_rate", 0.0) or 0.0)
    non_initial_turnover = float(metrics.get("non_initial_turnover_per_opportunity", 0.0) or 0.0)
    avg_turnover = float(metrics.get("average_turnover", 0.0) or 0.0)
    total_cost = float(metrics.get("total_transaction_cost", 0.0) or 0.0)
    base = sharpe - max_drawdown_loss - cvar_loss_5
    activity_underuse_penalty = (
        float(constraints.get("hit_rate_underuse_penalty", 5.0))
        * max(0.0, float(constraints.get("min_model_rebalance_hit_rate", 0.05)) - model_rebalance_hit_rate)
        + float(constraints.get("turnover_underuse_penalty", 5.0))
        * max(0.0, float(constraints.get("min_non_initial_turnover_per_opportunity", 0.002)) - non_initial_turnover)
    )
    max_hit_rate = constraints.get("max_model_rebalance_hit_rate")
    hit_rate_overuse = 0.0
    if max_hit_rate is not None:
        hit_rate_overuse = float(
            constraints.get("hit_rate_overuse_penalty", constraints.get("turnover_overuse_penalty", 2.0))
        ) * max(0.0, model_rebalance_hit_rate - float(max_hit_rate))
    turnover_overuse = float(constraints.get("turnover_overuse_penalty", 2.0)) * max(
        0.0,
        avg_turnover - float(constraints.get("max_average_turnover", 0.030)),
    )
    cost_penalty = float(constraints.get("cost_over_budget_penalty", 10.0)) * max(
        0.0,
        total_cost - float(constraints.get("cost_budget", 0.010)),
    )
    return float(base - activity_underuse_penalty - hit_rate_overuse - turnover_overuse - cost_penalty)


def _activity_constraints(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    constraints = dict(mapping(mapping(config.get("hpo")).get("activity_constraints")))
    activity = mapping(config.get("execution_activity"))
    if constraints.get("enabled") is True:
        return constraints
    if activity.get("activity_gate_enforced") is True:
        fallback = dict(constraints)
        for key in (
            "min_model_rebalance_hit_rate",
            "max_model_rebalance_hit_rate",
            "min_non_initial_turnover_per_opportunity",
            "max_average_turnover",
        ):
            if key in activity:
                fallback[key] = activity[key]
        fallback["enabled"] = True
        fallback.setdefault("scope_activity_protocols", [activity.get("protocol", "daily_gate_with_cost_constraint")])
        fallback.setdefault(
            "scope_baseline_families",
            [
                "model",
                "new_model_extension",
                "native_rl",
                "native_rl_reimplementation",
                "platform_native_rl",
            ],
        )
        return fallback
    return constraints


def _cvar_loss(returns: pd.Series) -> float:
    values = returns.to_numpy(dtype=float)
    if values.size == 0:
        return np.nan
    tail_n = max(1, int(np.ceil(0.05 * values.size)))
    return float(max(0.0, -float(np.mean(np.sort(values)[:tail_n]))))


def _portfolio_env(
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
    segment: str,
    *,
    split: SplitSpec | None = None,
) -> PortfolioRebalanceEnv:
    return PortfolioRebalanceEnv(
        artifacts["dataset"],
        artifacts["split"] if split is None else split,
        config=config,
        segment=segment,
        market_image_dataset=artifacts["market_image_dataset"],
    )


def _checkpoint_paths(run_dir: str | None) -> dict[str, Path | None]:
    if run_dir is None:
        return {"best": None, "last": None}
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return {
        "best": checkpoint_dir / "best.pt",
        "last": checkpoint_dir / "last.pt",
    }


def _path_join(root: str, child: str) -> Path:
    path = Path(root) / child
    path.mkdir(parents=True, exist_ok=True)
    return path


def _splits_for_config(config: Mapping[str, Any]) -> list[SplitSpec]:
    dataset = load_market_dataset(config)
    split = create_split(pd.DatetimeIndex(dataset.wide["close"].index), config)
    return list(split) if isinstance(split, list) else [split]


def _load_training_checkpoint(agent: HybridAgent, config: Mapping[str, Any], env: Any) -> Mapping[str, Any] | None:
    training = config.get("training")
    checkpoint_path = training.get("checkpoint_load_path") if isinstance(training, Mapping) else None
    if checkpoint_path is None:
        return None
    return load_checkpoint(checkpoint_path, device=agent.device, agent=agent, env=env)


def _assert_checkpoint_file_exists(config: Mapping[str, Any]) -> None:
    training = config.get("training")
    checkpoint_path = training.get("checkpoint_load_path") if isinstance(training, Mapping) else None
    if checkpoint_path is not None and not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"ERR_CHECKPOINT_NOT_FOUND: {checkpoint_path}")


def _attach_checkpoint_callback(
    agent: HybridAgent,
    checkpoint_paths: Mapping[str, Path | None],
    config: Mapping[str, Any],
    env: Any,
) -> None:
    best_path = checkpoint_paths.get("best")
    if best_path is None:
        return

    def callback(payload: Mapping[str, Any]) -> None:
        save_checkpoint(
            agent,
            best_path,
            epoch=int(payload.get("epoch", 0)),
            global_step=int(payload.get("global_step", getattr(agent, "global_step", len(agent.history)))),
            best_validation_metric=payload.get("best_validation_metric"),
            resolved_config=config,
            env=env,
            include_replay_buffer=_checkpoint_include_replay_buffer(config),
        )

    agent.checkpoint_callback = callback


def _save_last_checkpoint(
    agent: HybridAgent,
    checkpoint_paths: Mapping[str, Path | None],
    config: Mapping[str, Any],
    env: Any,
) -> Path | None:
    last_path = checkpoint_paths.get("last")
    if last_path is None:
        return None
    epoch = max(int(getattr(agent, "last_epoch", len(agent.history) - 1)), 0)
    return save_checkpoint(
        agent,
        last_path,
        epoch=epoch,
        global_step=int(getattr(agent, "global_step", len(agent.history))),
        best_validation_metric=agent.best_validation_metric,
        resolved_config=config,
        env=env,
        include_replay_buffer=_checkpoint_include_replay_buffer(config),
    )


def _checkpoint_include_replay_buffer(config: Mapping[str, Any]) -> bool:
    training = config.get("training")
    if isinstance(training, Mapping) and "checkpoint_include_replay_buffer" in training:
        return bool(training.get("checkpoint_include_replay_buffer"))
    checkpoint = config.get("checkpoint")
    if isinstance(checkpoint, Mapping) and "include_replay_buffer" in checkpoint:
        return bool(checkpoint.get("include_replay_buffer"))
    return True


def _device_from_config(config: Mapping[str, Any]) -> torch.device:
    device = config.get("device")
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, Mapping):
        mode = str(device.get("mode", "cpu")).lower()
        if mode == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if mode == "auto" and torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def asset_order(dataset: MarketDatasetBundle) -> list[str]:
    order = dataset.data_manifest.get("canonical_asset_order")
    if isinstance(order, list):
        return [str(item) for item in order]
    return [str(item) for item in dataset.availability_mask.columns]


def _panel_for_dates(panel: pd.DataFrame, dates: Sequence[Any]) -> pd.DataFrame:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    selected_dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    return frame.loc[frame["date"].isin(selected_dates)].copy()


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _baseline_daily_diagnostics(result: Any) -> pd.DataFrame:
    frame = getattr(result, "baseline_daily_diagnostics", None)
    if isinstance(frame, pd.DataFrame):
        return frame
    return pd.DataFrame()


def _metrics_from_walk_forward(aggregation: Mapping[str, Any], fold_results: Sequence[BacktestResult]) -> dict[str, float]:
    frame = aggregation["walk_forward_results"]
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        all_oos = frame.loc[frame["fold_id"] == "all_oos"]
        if not all_oos.empty:
            return {
                key: float(all_oos.iloc[0][key])
                for key in ("n_steps", "cumulative_return", "turnover", "total_transaction_cost")
                if key in all_oos
            }
    if not fold_results:
        return {}
    return dict(fold_results[-1].metrics)


def _seed_values(config: Mapping[str, Any]) -> list[int]:
    seed_config = config.get("seed_stability")
    values = None
    if isinstance(seed_config, Mapping):
        values = seed_config.get("seeds")
    if values is None:
        reproducibility = config.get("reproducibility")
        if isinstance(reproducibility, Mapping):
            values = reproducibility.get("seeds")
    if values is None:
        reproducibility = config.get("reproducibility")
        seed = 42
        if isinstance(reproducibility, Mapping) and reproducibility.get("seed") is not None:
            seed = int(reproducibility["seed"])
        return [seed]
    return [int(value) for value in values]


def _seed_aggregate_summary(metrics_by_seed: Mapping[int, Mapping[str, Any]], model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = sorted({metric for metrics in metrics_by_seed.values() for metric in metrics})
    for metric_name in metric_names:
        raw_values = [metrics[metric_name] for metrics in metrics_by_seed.values() if metric_name in metrics]
        values = pd.to_numeric(pd.Series(raw_values), errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        rows.append(
            {
                "model_name": model_name,
                "metric_name": metric_name,
                "n_seeds": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }
        )
    return pd.DataFrame(rows)


PROXY_BASELINES = {
    "ppo_proxy",
    "ppo_baseline",
    "cnn_ppo_proxy",
    "cnn_ppo_baseline",
    "bernoulli_gated_ppo_proxy",
    "bernoulli_gated_ppo",
    "dqn_template_proxy",
    "dqn_only",
    "eiie_proxy",
    "eiie",
}
NATIVE_BASELINES = {
    "ppo_native",
    "cnn_ppo_native",
    "bernoulli_gated_ppo_native",
    "dqn_template_native",
    "eiie_native",
    "pgportfolio_eiie_native",
}
EXTERNAL_BASELINES = {"pgportfolio_original_external"}
RELATED_WORK_REIMPLEMENTATION_MODEL_NAMES = {
    "ppo_dqn_hierarchical_reimplementation",
    "hybrid_dqn_optimizer_equal_weight",
    "hybrid_dqn_optimizer_markowitz_mean_variance",
    "hybrid_dqn_optimizer_minimum_variance",
    "hybrid_dqn_optimizer_sharpe_maximization",
    "hybrid_dqn_optimizer_risk_parity",
}
PPO_DQN_HIERARCHICAL_REIMPLEMENTATION = "ppo_dqn_hierarchical_reimplementation"
PPO_DQN_HIGH_LEVEL_ACTION_SELECTOR = "high_level_action_selector"
PPO_DQN_HIERARCHY_ACTIONS = (0, 1, 2, 3, 4, 5)
HYBRID_DQN_OPTIMIZER_TRAINING_ALGORITHM = "factorized_dqn_signal_plus_portfolio_optimizer"
HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES = (
    "hybrid_dqn_optimizer_equal_weight",
    "hybrid_dqn_optimizer_markowitz_mean_variance",
    "hybrid_dqn_optimizer_minimum_variance",
    "hybrid_dqn_optimizer_sharpe_maximization",
    "hybrid_dqn_optimizer_risk_parity",
)
HYBRID_DQN_OPTIMIZER_BY_MODEL = {
    "hybrid_dqn_optimizer_equal_weight": "equal_weight",
    "hybrid_dqn_optimizer_markowitz_mean_variance": "markowitz_mean_variance",
    "hybrid_dqn_optimizer_minimum_variance": "minimum_variance",
    "hybrid_dqn_optimizer_sharpe_maximization": "sharpe_maximization",
    "hybrid_dqn_optimizer_risk_parity": "risk_parity",
}
NATIVE_REIMPLEMENTATION_BASELINES = frozenset(RELATED_WORK_REIMPLEMENTATION_MODEL_NAMES)
P12_CAGE_EIIE_MODEL_NAMES = {
    "cage_eiie_frozen_gate",
    "cage_eiie_multilevel_gate",
    "cage_eiie_distributional",
    "cage_eiie_no_cvar",
    "cage_eiie_distributional_no_cvar",
    "cage_eiie_joint_light",
    "cage_eiie_fixed_rho_25",
    "cage_eiie_fixed_rho_50",
    "cage_eiie_fixed_rho_75",
}
P13_GT_RCPO_MODEL_NAMES = {
    "graph_transformer_risk_constrained_actor_critic_lite",
    "gt_rcpo_lite",
}
P16_RA_GT_RCPO_MODEL_NAMES = set(RA_GT_RCPO_MODEL_NAMES)
P12_P13_NEW_MODEL_NAMES = P12_CAGE_EIIE_MODEL_NAMES | P13_GT_RCPO_MODEL_NAMES
PAPER_NEW_MODEL_NAMES = P12_P13_NEW_MODEL_NAMES | P16_RA_GT_RCPO_MODEL_NAMES
TRADITIONAL_BASELINE_NAMES = {
    "fixed_ratio",
    "equal_weight",
    "buy_and_hold",
    "markowitz",
    "traditional_markowitz_mean_variance",
    "markowitz_min_variance",
    "markowitz_max_sharpe",
    "risk_parity",
    "inverse_volatility",
    "minimum_drawdown",
    "risk_evaluation",
    "hrp",
    "momentum",
}


def _comparison_rows(
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    *,
    primary_model: str | None = None,
    training_summary_rows: Any | None = None,
    config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    summary_by_model = _training_summary_by_model(training_summary_rows)
    rows: list[dict[str, Any]] = []
    for model_name, metric_values in metrics_by_model.items():
        row = {
            "model_name": str(model_name),
            "role": "model" if primary_model is not None and str(model_name) == str(primary_model) else "benchmark",
            "status": "completed",
            **_baseline_metadata(str(model_name)),
        }
        row.update(summary_by_model.get(str(model_name), {}))
        _ensure_ppo_dqn_comparison_metadata(row)
        _ensure_hybrid_dqn_comparison_metadata(row)
        row.update({str(key): value for key, value in dict(metric_values).items()})
        _block_diagnostic_shared_dqn(row)
        _apply_activity_failure_to_comparison_row(row, config)
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_activity_failure_to_comparison_row(row: dict[str, Any], config: Mapping[str, Any] | None) -> None:
    reason = _comparison_activity_failure_reason(row, config)
    if not reason:
        return
    row["activity_failure_reason"] = reason
    row["final_activity_failure_reason"] = reason
    row["final_activity_passed"] = False
    row["rankable_in_unified_table"] = False
    row["paper_included"] = False
    if _missing_scalar(row.get("reason")):
        row["reason"] = reason


def _comparison_activity_failure_reason(row: Mapping[str, Any], config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(config, Mapping):
        return None
    activity = mapping(config.get("execution_activity"))
    if activity.get("activity_gate_enforced") is not True:
        return None
    constraints = _activity_constraints(config)
    if constraints.get("enabled") is not True:
        return None
    protocol = str(activity.get("protocol", "monthly_gate"))
    scope_protocols = {str(item) for item in constraints.get("scope_activity_protocols", [])}
    if scope_protocols and protocol not in scope_protocols:
        return None
    if not _comparison_row_in_activity_scope(row, constraints):
        return None
    hit_rate = _finite_float(row.get("model_rebalance_hit_rate"), 0.0)
    turnover_per_opportunity = _finite_float(row.get("non_initial_turnover_per_opportunity"), 0.0)
    avg_turnover = _finite_float(row.get("average_turnover"), 0.0)
    if hit_rate < float(constraints.get("min_model_rebalance_hit_rate", 0.05)):
        return "failed_low_trade_activity"
    if turnover_per_opportunity < float(constraints.get("min_non_initial_turnover_per_opportunity", 0.002)):
        return "failed_low_trade_activity"
    max_hit_rate = constraints.get("max_model_rebalance_hit_rate")
    if max_hit_rate is not None and hit_rate > float(max_hit_rate):
        return "failed_high_trade_activity"
    max_average_turnover = constraints.get("max_average_turnover")
    if max_average_turnover is not None and avg_turnover > float(max_average_turnover):
        return "failed_high_trade_activity"
    return None


def _comparison_row_in_activity_scope(row: Mapping[str, Any], constraints: Mapping[str, Any]) -> bool:
    if str(row.get("role", "")).strip().lower() == "model":
        return True
    scope_families = {str(item) for item in constraints.get("scope_baseline_families", [])}
    family = str(row.get("baseline_family", "")).strip()
    if family and family in scope_families:
        return True
    return _truthy_scalar(row.get("platform_native_rl_training"))


def _training_summary_by_model(training_summary_rows: Any | None) -> dict[str, dict[str, Any]]:
    if training_summary_rows is None:
        return {}
    if isinstance(training_summary_rows, pd.DataFrame):
        records = training_summary_rows.to_dict("records")
    else:
        records = list(training_summary_rows)
    selected = {
        "checkpoint_best_path",
        "checkpoint_last_path",
        "evaluated_checkpoint_path",
        "best_validation_metric",
        "status",
        "env_steps",
        "gradient_updates",
        "max_train_steps",
        "max_validation_steps",
        "max_gradient_updates_per_epoch",
        "paper_model_id",
        "child_model_name",
        "baseline_family",
        "training_algorithm",
        "rl_training",
        "platform_native_rl_training",
        "proxy_training",
        "external_original_implementation",
        "clean_room_reimplementation",
        "algorithm_fidelity",
        "dqn_role",
        "platform_adapted_surrogate",
        "hierarchy_action_distribution",
        "hierarchy_action_0_count",
        "hierarchy_action_1_count",
        "hierarchy_action_2_count",
        "hierarchy_action_3_count",
        "hierarchy_action_4_count",
        "hierarchy_action_5_count",
        "optimizer_name",
        "factorized_q",
        "portfolio_level_reward_shared",
        "counterfactual_asset_reward",
        "platform_adapted_approximation",
        "dqn_signal_include_rate",
        "optimizer_allocation_method",
        "optimizer_fallback_rate",
        "rankable_in_unified_table",
        "model_extension_id",
        "post_hoc_development_disclosure",
        "test_used_for_model_selection",
        "cost_availability",
        "cost_model_shared",
        "constraint_protocol_shared",
        "data_protocol",
        "execution_protocol",
        "evaluation_protocol",
        "diagnostic_status",
        "reason",
    }
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        model_name = record.get("model_name")
        if model_name is None:
            continue
        selected_record = {key: record.get(key) for key in selected if key in record}
        result[str(model_name)] = selected_record
        paper_model_id = record.get("paper_model_id")
        if paper_model_id is not None and str(paper_model_id) in NATIVE_REIMPLEMENTATION_BASELINES:
            result[str(paper_model_id)] = selected_record
    return result


def _baseline_metadata(model_name: str) -> dict[str, Any]:
    if model_name == PPO_DQN_HIERARCHICAL_REIMPLEMENTATION:
        return {
            "paper_model_id": model_name,
            "child_model_name": model_name,
            "baseline_family": "native_rl_reimplementation",
            "training_algorithm": model_name,
            "rl_training": True,
            "platform_native_rl_training": True,
            "proxy_training": False,
            "external_original_implementation": False,
            "source_code_vendored": False,
            "license": pd.NA,
            "data_protocol": "platform",
            "execution_protocol": "platform_backtest_engine",
            "evaluation_protocol": "unified_platform",
            "cost_model_shared": True,
            "cost_availability": "available",
            "constraint_protocol_shared": True,
            "clean_room_reimplementation": True,
            "algorithm_fidelity": "platform_adapted",
            "dqn_role": PPO_DQN_HIGH_LEVEL_ACTION_SELECTOR,
            "platform_adapted_surrogate": False,
            "hierarchy_action_distribution": "{}",
            "hierarchy_action_0_count": 0,
            "hierarchy_action_1_count": 0,
            "hierarchy_action_2_count": 0,
            "hierarchy_action_3_count": 0,
            "hierarchy_action_4_count": 0,
            "hierarchy_action_5_count": 0,
            "rankable_in_unified_table": True,
        }
    if model_name in HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES:
        optimizer_name = HYBRID_DQN_OPTIMIZER_BY_MODEL[model_name]
        return {
            "paper_model_id": model_name,
            "child_model_name": model_name,
            "baseline_family": "native_rl_reimplementation",
            "training_algorithm": HYBRID_DQN_OPTIMIZER_TRAINING_ALGORITHM,
            "rl_training": True,
            "platform_native_rl_training": True,
            "proxy_training": False,
            "external_original_implementation": False,
            "source_code_vendored": False,
            "license": pd.NA,
            "data_protocol": "platform",
            "execution_protocol": "platform_backtest_engine",
            "evaluation_protocol": "unified_platform",
            "cost_model_shared": True,
            "cost_availability": "available",
            "constraint_protocol_shared": True,
            "clean_room_reimplementation": True,
            "algorithm_fidelity": "platform_adapted",
            "dqn_role": pd.NA,
            "optimizer_name": optimizer_name,
            "platform_adapted_surrogate": pd.NA,
            "factorized_q": True,
            "portfolio_level_reward_shared": True,
            "counterfactual_asset_reward": False,
            "platform_adapted_approximation": True,
            "dqn_signal_include_rate": pd.NA,
            "optimizer_allocation_method": optimizer_name,
            "optimizer_fallback_rate": pd.NA,
            "rankable_in_unified_table": True,
        }
    if model_name in P12_P13_NEW_MODEL_NAMES:
        algorithm = (
            "graph_transformer_risk_constrained_actor_critic_lite"
            if model_name in P13_GT_RCPO_MODEL_NAMES
            else f"cage_eiie_{model_name.removeprefix('cage_eiie_')}"
        )
        return {
            "paper_model_id": model_name,
            "child_model_name": model_name,
            "baseline_family": "new_model_extension",
            "training_algorithm": algorithm,
            "rl_training": True,
            "platform_native_rl_training": True,
            "proxy_training": False,
            "external_original_implementation": False,
            "source_code_vendored": False,
            "license": pd.NA,
            "data_protocol": "platform",
            "execution_protocol": "platform_backtest_engine",
            "evaluation_protocol": "unified_platform",
            "cost_model_shared": True,
            "cost_availability": "available",
            "constraint_protocol_shared": True,
            "clean_room_reimplementation": True,
            "algorithm_fidelity": "platform_adapted",
            "model_extension_id": MODEL_EXTENSION_ID,
            "post_hoc_development_disclosure": True,
            "test_used_for_model_selection": False,
            "rankable_in_unified_table": True,
        }
    if model_name in P16_RA_GT_RCPO_MODEL_NAMES:
        return {
            "paper_model_id": model_name,
            "child_model_name": model_name,
            "baseline_family": "new_model_extension",
            "training_algorithm": RA_GT_RCPO_ALGORITHM,
            "rl_training": True,
            "platform_native_rl_training": True,
            "proxy_training": False,
            "external_original_implementation": False,
            "source_code_vendored": False,
            "license": pd.NA,
            "data_protocol": "platform",
            "execution_protocol": "platform_backtest_engine",
            "evaluation_protocol": "unified_platform",
            "cost_model_shared": True,
            "cost_availability": "available",
            "constraint_protocol_shared": True,
            "clean_room_reimplementation": True,
            "algorithm_fidelity": "platform_native",
            "model_extension_id": RA_GT_RCPO_MODEL_EXTENSION_ID,
            "post_hoc_development_disclosure": True,
            "test_used_for_model_selection": False,
            "rankable_in_unified_table": True,
        }
    if model_name in PROXY_BASELINES:
        return {
            "baseline_family": "neural_proxy",
            "training_algorithm": "supervised_execution_aligned_proxy",
            "rl_training": False,
            "platform_native_rl_training": False,
            "proxy_training": True,
            "external_original_implementation": False,
            "source_code_vendored": False,
            "license": pd.NA,
            "data_protocol": "platform",
            "execution_protocol": "platform_backtest_engine",
            "evaluation_protocol": "proxy_diagnostics",
            "cost_model_shared": True,
            "cost_availability": "available",
            "constraint_protocol_shared": True,
            "rankable_in_unified_table": False,
        }
    if model_name in NATIVE_BASELINES:
        algorithm = {
            "ppo_native": "ppo_clipped_gae",
            "cnn_ppo_native": "ppo_clipped_gae",
            "dqn_template_native": "double_dqn_template_selector",
            "bernoulli_gated_ppo_native": "bernoulli_gated_ppo_on_policy",
            "eiie_native": "eiie_policy_gradient_pvm",
            "pgportfolio_eiie_native": "pgportfolio_eiie_osbl",
        }.get(model_name, "native_rl")
        return {
            "baseline_family": "native_rl",
            "training_algorithm": algorithm,
            "rl_training": True,
            "platform_native_rl_training": True,
            "proxy_training": False,
            "external_original_implementation": False,
            "source_code_vendored": False,
            "license": pd.NA,
            "data_protocol": "platform",
            "execution_protocol": "platform_backtest_engine",
            "evaluation_protocol": "unified_platform",
            "cost_model_shared": True,
            "cost_availability": "available",
            "constraint_protocol_shared": True,
            "online_stochastic_batch_learning": model_name == "pgportfolio_eiie_native",
            "clean_room_reimplementation": model_name == "pgportfolio_eiie_native",
            "rankable_in_unified_table": True,
        }
    if model_name in EXTERNAL_BASELINES:
        return {
            "baseline_family": "external_original",
            "training_algorithm": "pgportfolio_original",
            "rl_training": True,
            "platform_native_rl_training": False,
            "proxy_training": False,
            "external_original_implementation": True,
            "source_code_vendored": False,
            "license": "GPL-3.0",
            "data_protocol": "external_export_import",
            "execution_protocol": "pgportfolio_original_external",
            "evaluation_protocol": "pgportfolio_original_external",
            "cost_model_shared": False,
            "cost_availability": "not_available",
            "constraint_protocol_shared": False,
            "rankable_in_unified_table": False,
        }
    if model_name in TRADITIONAL_BASELINE_NAMES:
        return {
            "baseline_family": "traditional",
            "deterministic_baseline": True,
            "n_independent_seeds": 1,
            "training_algorithm": "deterministic_strategy",
            "rl_training": False,
            "platform_native_rl_training": False,
            "proxy_training": False,
            "external_original_implementation": False,
            "source_code_vendored": False,
            "license": pd.NA,
            "data_protocol": "platform",
            "execution_protocol": "platform_backtest_engine",
            "evaluation_protocol": "unified_platform",
            "cost_model_shared": True,
            "cost_availability": "available",
            "constraint_protocol_shared": True,
            "rankable_in_unified_table": True,
        }
    return {
        "baseline_family": "model",
        "training_algorithm": pd.NA,
        "rl_training": pd.NA,
        "platform_native_rl_training": pd.NA,
        "proxy_training": False,
        "external_original_implementation": False,
        "source_code_vendored": False,
        "license": pd.NA,
        "data_protocol": pd.NA,
        "execution_protocol": pd.NA,
        "evaluation_protocol": pd.NA,
        "cost_model_shared": pd.NA,
        "cost_availability": pd.NA,
        "constraint_protocol_shared": pd.NA,
        "rankable_in_unified_table": True,
    }


def _ensure_ppo_dqn_comparison_metadata(row: dict[str, Any]) -> None:
    if str(row.get("model_name")) != PPO_DQN_HIERARCHICAL_REIMPLEMENTATION:
        return
    _set_missing(row, "paper_model_id", PPO_DQN_HIERARCHICAL_REIMPLEMENTATION)
    _set_missing(row, "child_model_name", PPO_DQN_HIERARCHICAL_REIMPLEMENTATION)
    _set_missing(row, "baseline_family", "native_rl_reimplementation")
    _set_missing(row, "training_algorithm", PPO_DQN_HIERARCHICAL_REIMPLEMENTATION)
    _set_missing(row, "clean_room_reimplementation", True)
    _set_missing(row, "algorithm_fidelity", "platform_adapted")
    _set_missing(row, "dqn_role", PPO_DQN_HIGH_LEVEL_ACTION_SELECTOR)
    row["platform_adapted_surrogate"] = _truthy_scalar(row.get("platform_adapted_surrogate"))
    counts = {}
    for action in PPO_DQN_HIERARCHY_ACTIONS:
        column = f"hierarchy_action_{action}_count"
        counts[str(action)] = _nonnegative_int(row.get(column, 0))
        row[column] = counts[str(action)]
    if _missing_scalar(row.get("hierarchy_action_distribution")):
        row["hierarchy_action_distribution"] = json.dumps(counts, sort_keys=True, separators=(",", ":"))
    if str(row.get("dqn_role")) != PPO_DQN_HIGH_LEVEL_ACTION_SELECTOR:
        row["rankable_in_unified_table"] = False
        if str(row.get("status")) == "completed":
            row["status"] = "deferred_variant"
        if _missing_scalar(row.get("reason")):
            row["reason"] = "unsupported_dqn_role"


def _ensure_hybrid_dqn_comparison_metadata(row: dict[str, Any]) -> None:
    child_id = _hybrid_child_id(row)
    if child_id is None:
        return
    optimizer_name = HYBRID_DQN_OPTIMIZER_BY_MODEL[child_id]
    _set_missing(row, "paper_model_id", child_id)
    _set_missing(row, "child_model_name", child_id)
    _set_missing(row, "baseline_family", "native_rl_reimplementation")
    _set_missing(row, "training_algorithm", HYBRID_DQN_OPTIMIZER_TRAINING_ALGORITHM)
    _set_missing(row, "clean_room_reimplementation", True)
    _set_missing(row, "algorithm_fidelity", "platform_adapted")
    _set_missing(row, "optimizer_name", optimizer_name)
    _set_missing(row, "optimizer_allocation_method", row.get("optimizer_name", optimizer_name))
    _set_missing(row, "factorized_q", True)
    _set_missing(row, "portfolio_level_reward_shared", True)
    _set_missing(row, "counterfactual_asset_reward", False)
    _set_missing(row, "platform_adapted_approximation", True)
    _set_missing(row, "dqn_role", pd.NA)
    _set_missing(row, "platform_adapted_surrogate", pd.NA)
    _set_missing(row, "rankable_in_unified_table", True)
    _block_diagnostic_shared_dqn(row)


def _block_diagnostic_shared_dqn(row: dict[str, Any]) -> None:
    if str(row.get("diagnostic_status")) != "diagnostic_shared_dqn":
        return
    if _hybrid_child_id(row) is None:
        return
    row["rankable_in_unified_table"] = False
    if _missing_scalar(row.get("reason")):
        row["reason"] = "diagnostic_shared_dqn"


def _hybrid_child_id(row: Mapping[str, Any]) -> str | None:
    for key in ("paper_model_id", "child_model_name", "model_name"):
        value = row.get(key)
        if value is not None and str(value) in HYBRID_DQN_OPTIMIZER_CHILD_MODEL_NAMES:
            return str(value)
    return None


def _apply_ppo_dqn_diagnostics_metadata(
    model_name: str,
    baseline_daily_diagnostics: pd.DataFrame | None,
    summary_row: dict[str, Any],
) -> None:
    if str(model_name) != PPO_DQN_HIERARCHICAL_REIMPLEMENTATION:
        return
    counts = {str(action): 0 for action in PPO_DQN_HIERARCHY_ACTIONS}
    if isinstance(baseline_daily_diagnostics, pd.DataFrame) and "hierarchy_action" in baseline_daily_diagnostics.columns:
        actions = pd.to_numeric(baseline_daily_diagnostics["hierarchy_action"], errors="coerce").dropna().astype(int)
        for action in PPO_DQN_HIERARCHY_ACTIONS:
            counts[str(action)] = int((actions == action).sum())
    for action in PPO_DQN_HIERARCHY_ACTIONS:
        summary_row[f"hierarchy_action_{action}_count"] = counts[str(action)]
    summary_row["hierarchy_action_distribution"] = json.dumps(counts, sort_keys=True, separators=(",", ":"))
    if isinstance(baseline_daily_diagnostics, pd.DataFrame) and "platform_adapted_surrogate" in baseline_daily_diagnostics.columns:
        summary_row["platform_adapted_surrogate"] = _truthy_scalar(
            summary_row.get("platform_adapted_surrogate")
        ) or bool(baseline_daily_diagnostics["platform_adapted_surrogate"].map(_truthy_scalar).any())
    else:
        summary_row["platform_adapted_surrogate"] = _truthy_scalar(summary_row.get("platform_adapted_surrogate"))


def _apply_hybrid_dqn_diagnostics_metadata(
    model_name: str,
    baseline_daily_diagnostics: pd.DataFrame | None,
    summary_row: dict[str, Any],
) -> None:
    child_id = _hybrid_child_id({"model_name": model_name, **summary_row})
    if child_id is None:
        return
    _ensure_hybrid_dqn_comparison_metadata(summary_row)
    frame = _hybrid_diagnostics_frame(child_id, baseline_daily_diagnostics)
    if frame is None or frame.empty:
        return
    include_count = _numeric_sum(frame, "include_count")
    neutral_count = _numeric_sum(frame, "neutral_count")
    exclude_count = _numeric_sum(frame, "exclude_count")
    denominator = include_count + neutral_count + exclude_count
    summary_row["dqn_signal_include_rate"] = pd.NA if denominator <= 0.0 else float(include_count / denominator)
    optimizer_name = _first_non_missing(frame, "optimizer_name")
    if optimizer_name is not None:
        summary_row["optimizer_name"] = str(optimizer_name)
        summary_row["optimizer_allocation_method"] = str(optimizer_name)
    status = (
        frame["optimizer_status"]
        if "optimizer_status" in frame.columns
        else pd.Series(["success"] * len(frame), index=frame.index)
    )
    fallback_reason = (
        frame["fallback_reason"]
        if "fallback_reason" in frame.columns
        else pd.Series([pd.NA] * len(frame), index=frame.index)
    )
    fallback_mask = status.fillna("success").astype(str).ne("success") | fallback_reason.map(
        lambda value: not _missing_scalar(value)
    )
    summary_row["optimizer_fallback_rate"] = float(fallback_mask.mean()) if len(fallback_mask) else pd.NA
    _ensure_hybrid_dqn_comparison_metadata(summary_row)


def _hybrid_diagnostics_frame(child_id: str, baseline_daily_diagnostics: pd.DataFrame | None) -> pd.DataFrame | None:
    if not isinstance(baseline_daily_diagnostics, pd.DataFrame) or baseline_daily_diagnostics.empty:
        return None
    frame = baseline_daily_diagnostics
    if "paper_model_id" not in frame.columns or "child_model_name" not in frame.columns:
        return frame.iloc[0:0].copy()
    paper_mask = frame["paper_model_id"].fillna("").astype(str).str.strip().eq(child_id)
    child_mask = frame["child_model_name"].fillna("").astype(str).str.strip().eq(child_id)
    return frame.loc[paper_mask & child_mask].copy()


def _numeric_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _first_non_missing(frame: pd.DataFrame, column: str) -> Any | None:
    if column not in frame.columns:
        return None
    values = frame[column]
    for value in values:
        if not _missing_scalar(value):
            return value
    return None


def _baseline_training_artifacts(
    model_name: str,
    strategy: Any,
    baseline_daily_diagnostics: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    metadata = _baseline_metadata(str(model_name))
    result = getattr(strategy, "training_result", None)
    row: dict[str, Any] = {
        "model_name": str(model_name),
        "status": "not_applicable",
        **metadata,
        "checkpoint_best_path": None,
        "checkpoint_last_path": None,
        "evaluated_checkpoint_path": None,
        "best_validation_metric": None,
        "env_steps": 0,
        "gradient_updates": 0,
    }
    history_frame: pd.DataFrame | None = None
    if isinstance(result, Mapping):
        for key, value in result.items():
            if key == "training_history":
                continue
            if key == "model_name":
                continue
            if _is_scalar_for_summary(value):
                row[str(key)] = value
        history = result.get("training_history")
        history_frame = _frame_or_none(history)
    elif getattr(strategy, "fit_required", False):
        row["status"] = "missing_training_result"

    if history_frame is None:
        history = getattr(strategy, "training_history", None)
        history_frame = _frame_or_none(history)
    if history_frame is not None:
        history_frame = history_frame.copy()
        history_frame.insert(0, "model_name", str(model_name))
    _apply_ppo_dqn_diagnostics_metadata(str(model_name), baseline_daily_diagnostics, row)
    _apply_hybrid_dqn_diagnostics_metadata(str(model_name), baseline_daily_diagnostics, row)
    return row, history_frame


def _refresh_ppo_dqn_aggregate_comparison(result: dict[str, Any], model_name: str) -> None:
    if str(model_name) not in NATIVE_REIMPLEMENTATION_BASELINES:
        return
    summary_frame = _frame_or_none(result.get("baseline_training_summary"))
    if summary_frame is None or summary_frame.empty:
        return
    summary_rows = summary_frame.to_dict("records")
    for row in summary_rows:
        if str(row.get("model_name")) == PPO_DQN_HIERARCHICAL_REIMPLEMENTATION:
            _apply_ppo_dqn_diagnostics_metadata(
                PPO_DQN_HIERARCHICAL_REIMPLEMENTATION,
                _frame_or_none(result.get("baseline_daily_diagnostics")),
                row,
            )
        _apply_hybrid_dqn_diagnostics_metadata(
            str(row.get("model_name", model_name)),
            _frame_or_none(result.get("baseline_daily_diagnostics")),
            row,
        )
    result["baseline_training_summary"] = pd.DataFrame(summary_rows)
    comparison_frame = _frame_or_none(result.get("baseline_comparison"))
    if comparison_frame is None or comparison_frame.empty:
        return
    metrics = dict(result.get("metrics")) if isinstance(result.get("metrics"), Mapping) else {}
    if not metrics:
        metrics = dict(comparison_frame.iloc[0].to_dict())
    result["baseline_comparison"] = _comparison_rows(
        {str(model_name): metrics},
        training_summary_rows=summary_rows,
    )


def _block_rankable_without_paper_model_id(
    model_name: str,
    baseline_daily_diagnostics: pd.DataFrame,
    summary_row: dict[str, Any],
) -> None:
    if str(model_name) not in RELATED_WORK_REIMPLEMENTATION_MODEL_NAMES:
        return
    if _diagnostics_has_paper_model_id(baseline_daily_diagnostics, str(model_name)):
        return
    summary_row["rankable_in_unified_table"] = False
    summary_row["diagnostic_status"] = "missing_paper_model_id"
    summary_row["reason"] = "missing_paper_model_id"


def _diagnostics_has_paper_model_id(frame: pd.DataFrame, expected_model_name: str | None = None) -> bool:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "paper_model_id" not in frame.columns:
        return False
    values = frame["paper_model_id"].fillna("").astype(str).str.strip()
    if not bool(values.ne("").all()):
        return False
    if expected_model_name is not None and str(expected_model_name) in RELATED_WORK_REIMPLEMENTATION_MODEL_NAMES:
        return bool(values.eq(str(expected_model_name)).all())
    return True


def _assign_strategy_model_name(strategy: Any, model_name: str) -> None:
    if hasattr(strategy, "strategy_name"):
        try:
            setattr(strategy, "strategy_name", str(model_name))
        except Exception:
            pass


def _frame_or_none(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.DataFrame(value)
    except Exception:
        return None


def _is_scalar_for_summary(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_))


def _missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _set_missing(row: dict[str, Any], key: str, value: Any) -> None:
    if key not in row or _missing_scalar(row.get(key)):
        row[key] = value


def _truthy_scalar(value: Any) -> bool:
    if _missing_scalar(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _nonnegative_int(value: Any) -> int:
    if _missing_scalar(value):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


def _paired_return_payload(
    returns_by_model: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    *,
    training_summary_rows: Any | None = None,
) -> dict[str, Any]:
    summary_by_model = _training_summary_by_model(training_summary_rows)

    def rankable(model_name: str) -> bool:
        metadata = dict(_baseline_metadata(model_name))
        metadata.update(summary_by_model.get(model_name, {}))
        return bool(metadata.get("rankable_in_unified_table", True))

    rankable_returns = {
        name: frame
        for name, frame in returns_by_model.items()
        if rankable(str(name))
    }
    if len(rankable_returns) < 2:
        return {}
    benchmark_name = _select_statistics_benchmark(rankable_returns, config)
    if benchmark_name is None:
        return {}
    model_returns = {
        name: frame
        for name, frame in rankable_returns.items()
        if name != benchmark_name
    }
    if not model_returns:
        return {}
    return {
        "model_returns": model_returns,
        "benchmark_returns": {benchmark_name: rankable_returns[benchmark_name]},
    }


def _select_statistics_benchmark(returns_by_model: Mapping[str, pd.DataFrame], config: Mapping[str, Any]) -> str | None:
    names = [str(name) for name in returns_by_model]
    normalized = {name.lower().replace("-", "_"): name for name in names}
    stats_config = config.get("statistics")
    requested = None
    if isinstance(stats_config, Mapping):
        requested = stats_config.get("primary_benchmark")
    aliases = []
    if requested is not None:
        aliases.append(str(requested))
    aliases.extend(["cnn_ppo_baseline", "ppo_baseline", "equal_weight"])
    alias_map = {
        "cnn_ppo": "cnn_ppo_baseline",
        "cnn_ppo_baseline": "cnn_ppo_baseline",
        "ppo": "ppo_baseline",
        "ppo_baseline": "ppo_baseline",
        "equal weight": "equal_weight",
        "equal_weight": "equal_weight",
    }
    for alias in aliases:
        key = str(alias).lower().replace("-", "_")
        key = alias_map.get(key, key)
        if key in normalized:
            return normalized[key]
    return names[0] if names else None


def _mean_metrics(metrics_by_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, float]:
    summary = _seed_aggregate_summary(metrics_by_seed, "model")
    if summary.empty:
        return {}
    return {str(row.metric_name): float(row.mean) for row in summary.itertuples(index=False)}


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    running_max = nav.cummax()
    drawdown = 1.0 - nav / running_max.replace(0.0, np.nan)
    value = float(drawdown.max(skipna=True))
    return 0.0 if not np.isfinite(value) else value


__all__ = [
    "VALIDATION_METRIC_ALIASES",
    "build_pipeline_artifacts",
    "objective_metric",
    "run_strategy_backtest",
    "run_strategy_comparison",
    "run_walk_forward_backtest",
    "run_seed_stability_backtests",
    "run_trained_variant_matrix",
    "expand_otar_formal_matrix",
    "run_otar_formal_matrix",
    "strategy_runtime_config",
]
