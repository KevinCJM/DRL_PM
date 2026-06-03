from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.config import ConfigLoader, PROJECT_ROOT
from src.utils.logger import save_json_atomic


PROTOCOL_ID = "core13_v2_full_reset_20260522"
DATA_CUTOFF_DATE = "2026-05-20"
ACTIVE_FORMAL_PHASES = {"p7", "p9", "p12", "p13", "p16"}
ACTIVE_ACTIVITY_PROTOCOL = "daily_gate_with_cost_constraint"
ACTIVE_HPO_OBJECTIVE = "validation_return_risk_cost_constrained"
ACTIVE_RHO_POLICIES = {"learned", "straight_through_gumbel_softmax_v1"}
ACTIVE_CONFIG_EXEMPTIONS = {"p16_p1_fixed_deterministic_formal_export.yaml"}
P16_FORMAL_MIN_EPOCHS = 8
P16_FORMAL_MIN_VALIDATION_STEPS = 512
P16_FORMAL_MIN_UPDATES_PER_EPOCH = 64
REQUIRED_CONFIGS = (
    "configs/paper/p0_main_native_baseline_smoke.yaml",
    "configs/paper/hpo_equal_budget_main_native_pilot.yaml",
    "configs/paper/baseline_comparison_main_native_fixed.yaml",
    "configs/paper/baseline_comparison_main_native_from_hpo.yaml",
    "configs/paper/baseline_comparison_related_work.yaml",
    "configs/paper/p9_related_work_smoke.yaml",
    "configs/paper/hpo_equal_budget_related_work_pilot.yaml",
    "configs/paper/hpo_equal_budget_main_native_seed_runner.yaml",
    "configs/paper/hpo_equal_budget_related_work_seed_runner.yaml",
    "configs/paper/p12_cage_eiie_smoke.yaml",
    "configs/paper/p12_cage_eiie_pilot.yaml",
    "configs/paper/p12_cage_eiie_ablation.yaml",
    "configs/paper/p12_cage_eiie_formal_seed_runner.yaml",
    "configs/paper/p12_cage_eiie_formal_comparison.yaml",
    "configs/paper/p12_cage_eiie_joint_light_pilot.yaml",
    "configs/paper/p12_cage_eiie_distributional_pilot.yaml",
    "configs/paper/p12_cage_eiie_fixed_rho_ablation.yaml",
    "configs/paper/p13_gt_rcpo_lite_smoke.yaml",
    "configs/paper/p13_gt_rcpo_lite_pilot.yaml",
    "configs/paper/p13_gt_rcpo_lite_formal_seed_runner.yaml",
    "configs/paper/p13_gt_rcpo_lite_formal_comparison.yaml",
    "configs/paper/p16_ra_gt_rcpo_smoke.yaml",
    "configs/paper/p16_ra_gt_rcpo_pilot.yaml",
    "configs/paper/p16_ra_gt_rcpo_ablation.yaml",
    "configs/paper/p16_p1_fixed_deterministic_formal_export.yaml",
    "configs/paper/p16_ra_gt_rcpo_formal_seed_runner.yaml",
    "configs/paper/p16_ra_gt_rcpo_formal_comparison.yaml",
    "configs/paper/full_reproduction_paper.yaml",
)
FORMAL_PAPER_TABLE_GROUPS = (
    "main_hpo_5seed",
    "main_hpo_plus_p9",
    "p9_related_work_hpo",
    "p12_cage_eiie_formal",
    "p14_new_model_final",
)
OPTIONAL_P13_FORMAL_PAPER_TABLE_GROUPS = ("p13_gt_rcpo_lite_formal",)
DIAGNOSTIC_PAPER_TABLE_GROUPS = (
    "p2_input_pca",
    "p3_components",
    "p4_reward",
    "p5_cost_rebalance",
    "p6_robustness",
    "p8_modules",
)
FORMAL_TABLE_GROUPS = (*FORMAL_PAPER_TABLE_GROUPS, *DIAGNOSTIC_PAPER_TABLE_GROUPS)
FORMAL_MAIN_RUN_PREFIXES = (
    "EXP05_P7_formal_hpo_main_native",
    "EXP09_P9_formal_hpo_related_work",
    "EXP30_P12_formal_cage_eiie",
)
OPTIONAL_P13_FORMAL_RUN_PREFIXES = ("EXP33_P13_formal_gt_rcpo_lite",)
P16_MODEL_EXTENSION_ID = "core13_v2_p16_ra_gt_rcpo_20260525"
P16_FINAL_RUN_PREFIXES = ("EXP35_P16_formal_ra_gt_rcpo",)
P16_PRIMARY_MODEL_ID = "risk_aware_graph_transformer_constrained_actor_critic"
P16_DETERMINISTIC_BASELINES = ("risk_parity", "buy_and_hold", "equal_weight")
P1_MAIN_HPO_RUN_PREFIX = "EXP11_P1_hpo_final_main_native_from_hpo"
SEEDS = (42, 123, 2024, 3407, 9999)


def audit_formal_readiness(
    *,
    root: str | Path = PROJECT_ROOT,
    output_dir: str | Path = "results/full_reproduction/core13_v2_full_reset_20260522",
    protocol_id: str = PROTOCOL_ID,
    data_cutoff_date: str = DATA_CUTOFF_DATE,
) -> dict[str, Path]:
    project = _scoped_root(root)
    rows: list[dict[str, Any]] = []
    rows.extend(_audit_protocol_reset(project, protocol_id))
    rows.extend(_audit_data_freeze(project, data_cutoff_date))
    rows.extend(_audit_configs(project, protocol_id, data_cutoff_date))
    rows.extend(_audit_p12_p13_validation_gate(project))
    rows.extend(_audit_p16_readiness(project))
    rows.extend(_audit_formal_seed_runs(project, protocol_id))
    rows.extend(_audit_paper_tables(project, protocol_id, data_cutoff_date))
    rows.extend(_audit_artifact_bundle(project, protocol_id, data_cutoff_date))

    passed = all(row["status"] == "pass" for row in rows)
    target = _scoped_path(project, output_dir)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / "formal_readiness_audit.csv"
    json_path = target / "formal_readiness_audit.json"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    save_json_atomic(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_id": protocol_id,
            "data_cutoff_date": data_cutoff_date,
            "status": "go" if passed else "no_go",
            "passed": passed,
            "fail_count": int((frame["status"] != "pass").sum()) if not frame.empty else 0,
            "checks": rows,
        },
        json_path,
    )
    return {"csv": csv_path, "json": json_path}


def _audit_protocol_reset(root: Path, protocol_id: str) -> list[dict[str, Any]]:
    path = root / "results/protocol_reset" / protocol_id / "protocol_reset_manifest.json"
    payload = _read_json(path)
    return [
        _check(
            "p-2",
            "protocol_reset_manifest",
            path,
            bool(payload)
            and payload.get("new_protocol_id") == protocol_id
            and payload.get("discard_previous_results") is True
            and payload.get("forbid_checkpoint_reuse") is True
            and payload.get("forbid_hpo_reuse") is True,
            detail=json.dumps(
                {
                    "new_protocol_id": payload.get("new_protocol_id"),
                    "discard_previous_results": payload.get("discard_previous_results"),
                    "forbid_checkpoint_reuse": payload.get("forbid_checkpoint_reuse"),
                    "forbid_hpo_reuse": payload.get("forbid_hpo_reuse"),
                },
                ensure_ascii=False,
            ),
        )
    ]


def _audit_data_freeze(root: Path, data_cutoff_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nav_manifest = root / "data/reports/core13_etf_lof_fund_nav_tushare_manifest.json"
    token_present = bool(os.environ.get("TUSHARE_TOKEN"))
    rows.append(
        _check(
            "p-1",
            "tushare_token_present_for_from_zero_nav_download",
            "env:TUSHARE_TOKEN",
            token_present or nav_manifest.exists(),
            detail=f"token={'present' if token_present else 'missing'}; frozen_nav_manifest={nav_manifest.exists()}",
        )
    )
    rows.append(_cutoff_manifest_check(root, "p-1", "nav_source_cutoff", root / "data/reports/core13_etf_lof_fund_nav_tushare_manifest.json", "end_date_requested", data_cutoff_date))
    rows.append(_cutoff_manifest_check(root, "p-1", "ohlcv_source_cutoff", root / "data/reports/core13_ohlcv_download_manifest.json", "end_date_requested", data_cutoff_date))
    rows.append(_manifest_value_check(root / "data/reports/core13_data_download_manifest.json", "p-1", "standard_schema_return_source", "return_source", "adj_nav"))
    rows.append(_manifest_value_check(root / "data/reports/core13_data_download_manifest.json", "p-1", "standard_schema_execution_source", "execution_price_source", "ohlcv"))
    rows.append(_manifest_value_check(root / "data/reports/core13_data_download_manifest.json", "p-1", "standard_schema_split", "valuation_execution_split", True))
    rows.append(_calendar_loss_check(root))
    rows.append(_asset_count_check(root))
    rows.append(_metrics_factory_check(root))
    return rows


def _audit_configs(root: Path, protocol_id: str, data_cutoff_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in REQUIRED_CONFIGS:
        path = root / relative
        exists = path.exists()
        load_ok = False
        detail = "missing"
        if exists:
            try:
                config = ConfigLoader.load(path)
                load_ok = (
                    _get(config, "protocol", "protocol_id") == protocol_id
                    and str(_get(config, "protocol", "data_cutoff_date")) == data_cutoff_date
                    and _get(config, "data_governance", "return_source") == "adj_nav"
                    and _get(config, "data_governance", "execution_price_source") == "ohlcv"
                )
                detail = "loaded" if load_ok else "loaded_but_protocol_or_governance_mismatch"
            except Exception as exc:  # noqa: BLE001
                detail = str(exc)
        rows.append(_check("p-1", f"config_ready:{Path(relative).name}", path, exists and load_ok, detail=detail))
    rows.extend(_audit_all_paper_configs_active(root))
    rows.extend(_audit_hpo_config(root / "configs/paper/hpo_equal_budget_main_native_seed_runner.yaml", "p7"))
    rows.extend(_audit_hpo_config(root / "configs/paper/hpo_equal_budget_related_work_seed_runner.yaml", "p9"))
    rows.extend(_audit_hpo_config(root / "configs/paper/p12_cage_eiie_formal_seed_runner.yaml", "p12"))
    rows.extend(_audit_hpo_config(root / "configs/paper/p13_gt_rcpo_lite_formal_seed_runner.yaml", "p13"))
    rows.extend(_audit_hpo_config(root / "configs/paper/p16_ra_gt_rcpo_formal_seed_runner.yaml", "p16"))
    return rows


def _audit_all_paper_configs_active(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "configs/paper").glob("*.yaml")):
        if path.name.startswith("._") or path.name in ACTIVE_CONFIG_EXEMPTIONS:
            continue
        try:
            config = ConfigLoader.load(path)
        except Exception as exc:  # noqa: BLE001
            rows.append(_check("p-1", f"active_config_load:{path.name}", path, False, detail=str(exc)))
            continue
        rows.extend(_audit_active_protocol_config(config, path, phase=_phase_from_config_name(path.name)))
    return rows


def _phase_from_config_name(name: str) -> str:
    lowered = name.lower()
    for phase in ("p16", "p13", "p12", "p9", "p8", "p7", "p6", "p5", "p4", "p3", "p2", "p1", "p0"):
        if lowered.startswith(phase) or f"_{phase}_" in lowered:
            return phase
    if "related_work" in lowered:
        return "p9"
    if "hpo_equal_budget_main_native" in lowered:
        return "p7"
    return "p-1"


def _audit_active_protocol_config(config: Mapping[str, Any], path: Path, *, phase: str) -> list[dict[str, Any]]:
    activity = _get(config, "execution_activity") or {}
    hpo = _get(config, "hpo") or {}
    experiment_type = _get(config, "experiment", "type")
    rows = [
        _check(
            phase,
            f"active_config_rebalance_daily:{path.name}",
            path,
            _get(config, "rebalance", "mode") == "daily",
            detail=f"rebalance.mode={_get(config, 'rebalance', 'mode')}",
        ),
        _check(
            phase,
            f"active_config_execution_activity:{path.name}",
            path,
            activity.get("protocol") == ACTIVE_ACTIVITY_PROTOCOL
            and activity.get("scheduler_blocks_model_actions") is False
            and activity.get("activity_gate_enforced") is True
            and activity.get("turnover_optimization_protocol_id") == "turnover_active_v1",
            detail=json.dumps(
                {
                    "protocol": activity.get("protocol"),
                    "scheduler_blocks_model_actions": activity.get("scheduler_blocks_model_actions"),
                    "activity_gate_enforced": activity.get("activity_gate_enforced"),
                    "turnover_optimization_protocol_id": activity.get("turnover_optimization_protocol_id"),
                },
                ensure_ascii=False,
            ),
        ),
    ]
    if hpo.get("enabled") is True or experiment_type == "hyperparameter_sweep":
        rows.extend(_audit_active_hpo_objective_and_constraints(config, path, phase))
    if "p12" in path.name.lower() or bool(_get(config, "cage_eiie", "enabled")):
        rows.append(
            _check(
                phase,
                f"active_config_p12_gate_scoring_normalized:{path.name}",
                path,
                _get(config, "cage_eiie", "gate_scoring", "mode") == "normalized",
                detail=f"mode={_get(config, 'cage_eiie', 'gate_scoring', 'mode')}",
            )
        )
    if "p16" in path.name.lower() or bool(_get(config, "ra_gt_rcpo", "enabled")):
        rows.append(
            _check(
                phase,
                f"active_config_p16_rho_policy:{path.name}",
                path,
                str(_get(config, "ra_gt_rcpo", "rho_policy")) in ACTIVE_RHO_POLICIES,
                detail=f"rho_policy={_get(config, 'ra_gt_rcpo', 'rho_policy')}",
            )
        )
    return rows


def _audit_active_hpo_objective_and_constraints(config: Mapping[str, Any], path: Path, phase: str) -> list[dict[str, Any]]:
    hpo = _get(config, "hpo") or {}
    constraints = _get(config, "hpo", "activity_constraints") or {}
    scope_protocols = {str(item) for item in constraints.get("scope_activity_protocols", [])}
    return [
        _check(
            phase,
            f"active_config_hpo_objective:{path.name}",
            path,
            hpo.get("metric") == ACTIVE_HPO_OBJECTIVE and hpo.get("objective") == ACTIVE_HPO_OBJECTIVE,
            detail=f"metric={hpo.get('metric')}, objective={hpo.get('objective')}",
        ),
        _check(
            phase,
            f"active_config_hpo_activity_constraints:{path.name}",
            path,
            constraints.get("enabled") is True
            and scope_protocols == {ACTIVE_ACTIVITY_PROTOCOL}
            and float(constraints.get("min_model_rebalance_hit_rate") or 0.0) >= 0.05
            and float(constraints.get("min_non_initial_turnover_per_opportunity") or 0.0) >= 0.002,
            detail=json.dumps(
                {
                    "enabled": constraints.get("enabled"),
                    "scope_activity_protocols": sorted(scope_protocols),
                    "min_model_rebalance_hit_rate": constraints.get("min_model_rebalance_hit_rate"),
                    "min_non_initial_turnover_per_opportunity": constraints.get(
                        "min_non_initial_turnover_per_opportunity"
                    ),
                },
                ensure_ascii=False,
            ),
        ),
    ]


def _audit_hpo_config(path: Path, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        config = ConfigLoader.load(path)
    except Exception as exc:  # noqa: BLE001
        return [_check(phase, f"hpo_config_load:{path.name}", path, False, detail=str(exc))]
    rows.append(
        _check(
            phase,
            f"hpo_m6_pca:{path.name}",
            path,
            _get(config, "feature_matrix", "input_matrix_id") == "M6"
            and _get(config, "feature_reduction", "pca", "enabled") is True
            and float(_get(config, "feature_reduction", "pca", "explained_variance") or 0.0) == 0.95,
            detail=f"input={_get(config, 'feature_matrix', 'input_matrix_id')}, pca={_get(config, 'feature_reduction', 'pca')}",
        )
    )
    rows.append(
        _check(
            phase,
            f"hpo_seed_runner_external_seed_grid:{path.name}",
            path,
            bool(config.get("long_running")) is False and list(_get(config, "reproducibility", "seeds") or []) == [42],
            detail=f"long_running={config.get('long_running')}, seeds={_get(config, 'reproducibility', 'seeds')}",
        )
    )
    rows.append(
        _check(
            phase,
            f"hpo_equal_budget_50:{path.name}",
            path,
            _get(config, "hpo", "equal_budget_across_models") is True and int(_get(config, "hpo", "n_trials_per_model") or 0) == 50,
            detail=f"n_trials_per_model={_get(config, 'hpo', 'n_trials_per_model')}",
        )
    )
    if phase in ACTIVE_FORMAL_PHASES:
        rows.extend(_audit_active_formal_hpo_config(config, path, phase))
    return rows


def _audit_active_formal_hpo_config(config: Mapping[str, Any], path: Path, phase: str) -> list[dict[str, Any]]:
    activity = _get(config, "execution_activity") or {}
    hpo = _get(config, "hpo") or {}
    constraints = _get(config, "hpo", "activity_constraints") or {}
    scope_protocols = {str(item) for item in constraints.get("scope_activity_protocols", [])}
    rows = [
        _check(
            phase,
            f"active_rebalance_daily:{path.name}",
            path,
            _get(config, "rebalance", "mode") == "daily",
            detail=f"rebalance.mode={_get(config, 'rebalance', 'mode')}",
        ),
        _check(
            phase,
            f"active_execution_activity:{path.name}",
            path,
            activity.get("protocol") == ACTIVE_ACTIVITY_PROTOCOL
            and activity.get("scheduler_blocks_model_actions") is False
            and activity.get("activity_gate_enforced") is True
            and activity.get("turnover_optimization_protocol_id") == "turnover_active_v1",
            detail=json.dumps(
                {
                    "protocol": activity.get("protocol"),
                    "scheduler_blocks_model_actions": activity.get("scheduler_blocks_model_actions"),
                    "activity_gate_enforced": activity.get("activity_gate_enforced"),
                    "turnover_optimization_protocol_id": activity.get("turnover_optimization_protocol_id"),
                },
                ensure_ascii=False,
            ),
        ),
        _check(
            phase,
            f"active_hpo_objective:{path.name}",
            path,
            hpo.get("metric") == ACTIVE_HPO_OBJECTIVE and hpo.get("objective") == ACTIVE_HPO_OBJECTIVE,
            detail=f"metric={hpo.get('metric')}, objective={hpo.get('objective')}",
        ),
        _check(
            phase,
            f"active_activity_constraints:{path.name}",
            path,
            constraints.get("enabled") is True
            and scope_protocols == {ACTIVE_ACTIVITY_PROTOCOL}
            and float(constraints.get("min_model_rebalance_hit_rate") or 0.0) >= 0.05
            and float(constraints.get("min_non_initial_turnover_per_opportunity") or 0.0) >= 0.002,
            detail=json.dumps(
                {
                    "enabled": constraints.get("enabled"),
                    "scope_activity_protocols": sorted(scope_protocols),
                    "min_model_rebalance_hit_rate": constraints.get("min_model_rebalance_hit_rate"),
                    "min_non_initial_turnover_per_opportunity": constraints.get(
                        "min_non_initial_turnover_per_opportunity"
                    ),
                },
                ensure_ascii=False,
            ),
        ),
    ]
    if phase == "p12":
        rows.append(
            _check(
                phase,
                f"p12_gate_scoring_normalized:{path.name}",
                path,
                _get(config, "cage_eiie", "gate_scoring", "mode") == "normalized",
                detail=f"mode={_get(config, 'cage_eiie', 'gate_scoring', 'mode')}",
            )
        )
    if phase == "p16":
        rows.append(
            _check(
                phase,
                f"p16_learned_rho_policy:{path.name}",
                path,
                str(_get(config, "ra_gt_rcpo", "rho_policy")) in ACTIVE_RHO_POLICIES,
                detail=f"rho_policy={_get(config, 'ra_gt_rcpo', 'rho_policy')}",
            )
        )
        budget = _p16_effective_training_budget(config)
        rows.append(
            _check(
                phase,
                f"p16_effective_training_budget:{path.name}",
                path,
                budget["epochs"] >= P16_FORMAL_MIN_EPOCHS
                and budget["train_steps"] >= budget["min_env_steps"]
                and budget["validation_steps"] >= P16_FORMAL_MIN_VALIDATION_STEPS
                and budget["effective_updates_per_epoch"] >= P16_FORMAL_MIN_UPDATES_PER_EPOCH
                and budget["estimated_gradient_updates"] >= budget["min_gradient_updates"]
                and budget["estimated_env_steps"] >= budget["min_env_steps"],
                detail=json.dumps(budget, ensure_ascii=False, sort_keys=True),
            )
        )
    return rows


def _p16_effective_training_budget(config: Mapping[str, Any]) -> dict[str, int | None]:
    native = _get(config, "baselines", "native_rl") or {}
    training = _get(config, "training") or {}
    section = _get(config, "ra_gt_rcpo") or {}
    epochs = max(1, _int_or_default(native.get("epochs", training.get("epochs")), 1))
    train_steps = _int_or_default(native.get("max_train_steps") or training.get("max_train_steps"), 0)
    validation_steps = _int_or_default(native.get("max_validation_steps") or training.get("max_validation_steps"), 0)
    batch_size = max(1, _int_or_default(section.get("batch_size", training.get("batch_size")), 32))
    raw_updates_per_epoch = native.get("max_gradient_updates_per_epoch", training.get("max_gradient_updates_per_epoch"))
    configured_updates_per_epoch = None if raw_updates_per_epoch is None else max(0, _int_or_default(raw_updates_per_epoch, 0))
    batches_per_epoch = 0 if train_steps <= 0 else int((train_steps + batch_size - 1) // batch_size)
    effective_updates_per_epoch = (
        batches_per_epoch
        if configured_updates_per_epoch is None
        else int(min(configured_updates_per_epoch, batches_per_epoch))
    )
    estimated_gradient_updates = int(epochs * effective_updates_per_epoch)
    estimated_env_steps_per_epoch = int(min(train_steps, effective_updates_per_epoch * batch_size))
    estimated_env_steps = int(epochs * estimated_env_steps_per_epoch)
    return {
        "epochs": int(epochs),
        "train_steps": int(train_steps),
        "validation_steps": int(validation_steps),
        "batch_size": int(batch_size),
        "configured_updates_per_epoch": configured_updates_per_epoch,
        "batches_per_epoch": int(batches_per_epoch),
        "effective_updates_per_epoch": int(effective_updates_per_epoch),
        "estimated_gradient_updates": int(estimated_gradient_updates),
        "estimated_env_steps": int(estimated_env_steps),
        "min_gradient_updates": _int_or_default(section.get("min_gradient_updates_for_formal"), 128),
        "min_env_steps": _int_or_default(section.get("min_env_steps_for_formal"), 2048),
    }


def _int_or_default(value: Any, default: int) -> int:
    try:
        if value is None or pd.isna(value):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _audit_p12_p13_validation_gate(root: Path) -> list[dict[str, Any]]:
    reference_dir = root / "results/paper_tables/p12_p13_validation_references"
    promotion_dir = root / "results/paper_tables/p12_p13_promotion_gate"
    reference_manifest = _read_json(reference_dir / "validation_reference_manifest.json")
    reference_comparison = reference_dir / "validation_reference_comparison.csv"
    reference_returns = reference_dir / "validation_reference_daily_returns.csv"
    reference_selection = reference_dir / "validation_selection_report.csv"
    promotion_report = promotion_dir / "promotion_gate_report.csv"
    promotion_manifest = _read_json(promotion_dir / "promotion_gate_manifest.json")
    reference_models = {
        "eiie_native",
        "full_dqn_gated_multitask_cnn_ppo",
        "ppo_dqn_hierarchical_reimplementation",
        "cnn_ppo_native",
        "pgportfolio_eiie_native",
    }
    comparison_models = _model_set(reference_comparison)
    returns_models = _model_set(reference_returns)
    promotion = _read_csv(promotion_report)
    p12_rows = promotion.loc[promotion.get("phase", pd.Series(dtype=str)).astype(str).eq("P12")] if not promotion.empty else pd.DataFrame()
    p13_rows = promotion.loc[promotion.get("phase", pd.Series(dtype=str)).astype(str).eq("P13")] if not promotion.empty else pd.DataFrame()
    p12_promoted = (
        not p12_rows.empty
        and "promotion_gate_passed" in p12_rows.columns
        and p12_rows["promotion_gate_passed"].map(_truthy).any()
    )
    p13_decided = not p13_rows.empty and "promotion_gate_passed" in p13_rows.columns
    p13_declared_skipped = promotion_manifest.get("p13_evaluated") is False and p13_rows.empty
    return [
        _check(
            "p12_p13",
            "validation_reference_manifest",
            reference_dir / "validation_reference_manifest.json",
            reference_manifest.get("selection_split") == "validation"
            and reference_manifest.get("test_used_for_model_selection") is False,
            detail=json.dumps(
                {
                    "selection_split": reference_manifest.get("selection_split"),
                    "test_used_for_model_selection": reference_manifest.get("test_used_for_model_selection"),
                },
                ensure_ascii=False,
            ),
        ),
        _check(
            "p12_p13",
            "validation_reference_required_files",
            reference_dir,
            reference_comparison.exists() and reference_returns.exists() and reference_selection.exists(),
            detail=json.dumps(
                {
                    "comparison": reference_comparison.exists(),
                    "daily_returns": reference_returns.exists(),
                    "selection_report": reference_selection.exists(),
                },
                ensure_ascii=False,
            ),
        ),
        _check(
            "p12_p13",
            "validation_reference_model_coverage",
            reference_comparison,
            reference_models.issubset(comparison_models) and reference_models.issubset(returns_models),
            detail=json.dumps(
                {
                    "comparison_missing": sorted(reference_models - comparison_models),
                    "returns_missing": sorted(reference_models - returns_models),
                },
                ensure_ascii=False,
            ),
        ),
        _check(
            "p12",
            "p12_promotion_gate_passed",
            promotion_report,
            p12_promoted,
            detail=f"p12_promoted={p12_promoted}",
        ),
        _check(
            "p13",
            "p13_promotion_gate_decided_or_skipped",
            promotion_report,
            (p13_decided or p13_declared_skipped)
            and promotion_manifest.get("selection_split") == "validation"
            and promotion_manifest.get("test_used_for_model_selection") is False,
            detail=json.dumps(
                {
                    "p13_decided": p13_decided,
                    "p13_evaluated": promotion_manifest.get("p13_evaluated"),
                    "p13_declared_skipped": p13_declared_skipped,
                    "p13_formal_required": _p13_formal_required(root),
                    "selection_split": promotion_manifest.get("selection_split"),
                    "test_used_for_model_selection": promotion_manifest.get("test_used_for_model_selection"),
                },
                ensure_ascii=False,
            ),
        ),
    ]


def _p13_formal_required(root: Path) -> bool:
    report = _read_csv(root / "results/paper_tables/p12_p13_promotion_gate/promotion_gate_report.csv")
    if report.empty or "phase" not in report.columns or "promotion_gate_passed" not in report.columns:
        return False
    p13_rows = report.loc[report["phase"].astype(str).eq("P13")]
    return bool(not p13_rows.empty and p13_rows["promotion_gate_passed"].map(_truthy).any())


def _audit_p16_readiness(root: Path) -> list[dict[str, Any]]:
    reference_dir = root / "results/paper_tables/p16_validation_references"
    promotion_dir = root / "results/paper_tables/p16_promotion_gate"
    final_dir = root / "results/paper_tables/p16_ra_gt_rcpo_final"
    reference_comparison = reference_dir / "validation_reference_comparison.csv"
    reference_returns = reference_dir / "validation_reference_daily_returns.csv"
    reference_risk = reference_dir / "validation_reference_risk_metrics.csv"
    reference_selection = reference_dir / "validation_selection_report.csv"
    promotion_report = promotion_dir / "promotion_gate_report.csv"
    promotion_comparison = promotion_dir / "validation_reference_comparison.csv"
    final_manifest = final_dir / "paper_aggregate_manifest.json"
    promoted = _p16_promoted(root)
    rows = [
        _check(
            "p16",
            "p16_validation_reference_required_files",
            reference_dir,
            reference_comparison.exists()
            and reference_returns.exists()
            and reference_risk.exists()
            and reference_selection.exists(),
            detail=json.dumps(
                {
                    "comparison": reference_comparison.exists(),
                    "daily_returns": reference_returns.exists(),
                    "risk_metrics": reference_risk.exists(),
                    "selection_report": reference_selection.exists(),
                },
                ensure_ascii=False,
            ),
        ),
        _check(
            "p16",
            "p16_promotion_gate_required_files",
            promotion_dir,
            promotion_report.exists() and promotion_comparison.exists(),
            detail=json.dumps(
                {
                    "promotion_gate_report": promotion_report.exists(),
                    "validation_reference_comparison": promotion_comparison.exists(),
                },
                ensure_ascii=False,
            ),
        ),
    ]
    if promoted:
        rows.extend(_audit_p16_formal_runs(root))
        rows.extend(_audit_p16_final_table(final_dir, PROTOCOL_ID, DATA_CUTOFF_DATE, promoted=promoted))
    else:
        rows.append(
            _check(
                "p16",
                "p16_formal_not_required_until_promotion",
                promotion_report,
                True,
                detail="P16 formal/final aggregation is conditional on validation promotion gate.",
            )
        )
    return rows


def _audit_p16_final_table(
    final_dir: Path,
    protocol_id: str,
    data_cutoff_date: str,
    *,
    promoted: bool,
) -> list[dict[str, Any]]:
    manifest = _read_json(final_dir / "paper_aggregate_manifest.json")
    main_path = final_dir / "paper_main_comparison.csv"
    seed_summary = final_dir / "paper_seed_summary.csv"
    paired = final_dir / "paper_paired_statistics.csv"
    main = _read_csv(main_path)
    formal_filter = manifest.get("formal_filter") if isinstance(manifest.get("formal_filter"), Mapping) else {}
    duplicate_count = _source_model_seed_duplicate_count(main)
    p16_seed_set = _seed_set(_rows_for_model(main, P16_PRIMARY_MODEL_ID))
    deterministic = _rows_for_models(main, P16_DETERMINISTIC_BASELINES)
    p1_rows = main.loc[main.get("source_run", pd.Series(dtype="object")).astype(str).eq(
        "EXP36_P1_fixed_deterministic_formal_export"
    )].copy() if "source_run" in main.columns else pd.DataFrame()
    rows = [
        _check(
            "p16",
            "p16_final_table_required_if_promoted",
            final_dir,
            bool(manifest) and main_path.exists() and seed_summary.exists() and paired.exists(),
            detail=json.dumps(
                {
                    "promoted": promoted,
                    "manifest": bool(manifest),
                    "main": main_path.exists(),
                    "seed_summary": seed_summary.exists(),
                    "paired": paired.exists(),
                },
                ensure_ascii=False,
            ),
        ),
        _check(
            "p16",
            "p16_final_table_formal_filter",
            final_dir / "paper_aggregate_manifest.json",
            _formal_filter_matches(formal_filter, protocol_id, data_cutoff_date),
            detail=json.dumps(
                {
                    "protocol_id": formal_filter.get("required_protocol_id")
                    if isinstance(formal_filter, Mapping)
                    else None,
                    "data_cutoff_date": formal_filter.get("required_data_cutoff_date")
                    if isinstance(formal_filter, Mapping)
                    else None,
                    "require_formal_manifest": formal_filter.get("require_formal_manifest")
                    if isinstance(formal_filter, Mapping)
                    else None,
                    "require_availability_mask_contract": formal_filter.get("require_availability_mask_contract")
                    if isinstance(formal_filter, Mapping)
                    else None,
                },
                ensure_ascii=False,
            ),
        ),
        _check(
            "p16",
            "p16_final_table_formal_rows",
            main_path,
            _group_has_formal_rows(main_path),
            detail=f"rows={len(main)}",
        ),
        _check(
            "p16",
            "p16_final_table_no_source_model_seed_duplicates",
            main_path,
            duplicate_count == 0,
            detail=f"duplicate_count={duplicate_count}",
        ),
        _check(
            "p16",
            "p16_final_table_p16_primary_5seed_hpo",
            main_path,
            p16_seed_set == set(SEEDS)
            and _all_source_file(_rows_for_model(main, P16_PRIMARY_MODEL_ID), "hpo_model_final_comparison.csv"),
            detail=json.dumps(
                {
                    "seeds": sorted(p16_seed_set),
                    "expected_seeds": list(SEEDS),
                },
                ensure_ascii=False,
            ),
        ),
        _check(
            "p16",
            "p16_final_table_deterministic_baseline_1seed",
            main_path,
            _p16_deterministic_baselines_valid(deterministic),
            detail=json.dumps(_p16_deterministic_baseline_detail(deterministic), ensure_ascii=False),
        ),
        _check(
            "p16",
            "p16_final_table_p1_export_traditional_only",
            main_path,
            not p1_rows.empty
            and set(p1_rows["paper_model_id"].astype(str)) == set(P16_DETERMINISTIC_BASELINES)
            and _all_source_file(p1_rows, "baseline_comparison.csv"),
            detail=json.dumps(
                {
                    "p1_models": sorted(p1_rows["paper_model_id"].astype(str).unique())
                    if "paper_model_id" in p1_rows.columns
                    else [],
                    "rows": len(p1_rows),
                },
                ensure_ascii=False,
            ),
        ),
    ]
    return rows


def _rows_for_model(frame: pd.DataFrame, model_id: str) -> pd.DataFrame:
    return _rows_for_models(frame, (model_id,))


def _rows_for_models(frame: pd.DataFrame, model_ids: Sequence[str]) -> pd.DataFrame:
    if frame.empty or "paper_model_id" not in frame.columns:
        return pd.DataFrame()
    wanted = {str(model_id) for model_id in model_ids}
    return frame.loc[frame["paper_model_id"].astype(str).isin(wanted)].copy()


def _seed_set(frame: pd.DataFrame) -> set[int]:
    if frame.empty or "seed" not in frame.columns:
        return set()
    seeds: set[int] = set()
    for value in frame["seed"].dropna():
        try:
            seeds.add(int(value))
        except (TypeError, ValueError):
            continue
    return seeds


def _all_source_file(frame: pd.DataFrame, source_file: str) -> bool:
    return not frame.empty and "source_file" in frame.columns and frame["source_file"].astype(str).eq(source_file).all()


def _source_model_seed_duplicate_count(frame: pd.DataFrame) -> int:
    columns = ["source_run", "paper_model_id", "seed"]
    if frame.empty or any(column not in frame.columns for column in columns):
        return -1
    return int(frame.duplicated(columns).sum())


def _p16_deterministic_baselines_valid(frame: pd.DataFrame) -> bool:
    if frame.empty or set(frame.get("paper_model_id", pd.Series(dtype="object")).astype(str)) != set(P16_DETERMINISTIC_BASELINES):
        return False
    return (
        len(frame) == len(P16_DETERMINISTIC_BASELINES)
        and frame.get("baseline_family", pd.Series(dtype="object")).astype(str).eq("traditional").all()
        and frame.get("deterministic_baseline", pd.Series(dtype="object")).map(_truthy).all()
        and _numeric_series_equals(frame.get("n_independent_seeds"), 1.0)
        and _seed_set(frame) == {42}
    )


def _p16_deterministic_baseline_detail(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "models": [], "seeds": []}
    return {
        "rows": int(len(frame)),
        "models": sorted(frame["paper_model_id"].dropna().astype(str).unique())
        if "paper_model_id" in frame.columns
        else [],
        "seeds": sorted(_seed_set(frame)),
        "baseline_family": sorted(frame["baseline_family"].dropna().astype(str).unique())
        if "baseline_family" in frame.columns
        else [],
        "deterministic_baseline_all_true": bool(frame["deterministic_baseline"].map(_truthy).all())
        if "deterministic_baseline" in frame.columns
        else False,
        "n_independent_seeds": sorted(frame["n_independent_seeds"].dropna().astype(float).unique())
        if "n_independent_seeds" in frame.columns
        else [],
    }


def _numeric_series_equals(series: pd.Series | None, expected: float) -> bool:
    if series is None:
        return False
    values = pd.to_numeric(series, errors="coerce")
    return bool(not values.isna().any() and values.eq(expected).all())


def _audit_p16_formal_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefix in P16_FINAL_RUN_PREFIXES:
        for seed in SEEDS:
            run_dir = root / "results" / f"{prefix}_s{seed}"
            manifest = _read_run_manifest(run_dir)
            rows.append(
                _check(
                    "p16",
                    f"p16_formal_seed_run:{prefix}_s{seed}",
                    run_dir,
                    bool(manifest)
                    and manifest.get("model_extension_id") == P16_MODEL_EXTENSION_ID
                    and manifest.get("diagnostic_status") == "formal"
                    and manifest.get("rankable_in_unified_table") is True
                    and _manifest_uses_active_protocol(manifest)
                    and (run_dir / "metrics/hpo_model_final_comparison.csv").exists()
                    and (run_dir / "logs/hpo_search_space_manifest.csv").exists(),
                detail=json.dumps(
                        {
                            "manifest": bool(manifest),
                            "model_extension_id": manifest.get("model_extension_id"),
                            "diagnostic_status": manifest.get("diagnostic_status"),
                            "rankable": manifest.get("rankable_in_unified_table"),
                            "rebalance_mode": manifest.get("rebalance_mode"),
                            "execution_activity_protocol": manifest.get("execution_activity_protocol"),
                            "scheduler_blocks_model_actions": manifest.get("scheduler_blocks_model_actions"),
                            "activity_gate_enforced": manifest.get("activity_gate_enforced"),
                            "hpo_metric": manifest.get("hpo_metric"),
                            "hpo_objective": manifest.get("hpo_objective"),
                        },
                        ensure_ascii=False,
                    ),
                )
            )
    return rows


def _p16_promoted(root: Path) -> bool:
    report = _read_csv(root / "results/paper_tables/p16_promotion_gate/promotion_gate_report.csv")
    if report.empty or "promotion_gate_passed" not in report.columns:
        return False
    return bool(report["promotion_gate_passed"].map(_truthy).any())


def _model_set(path: Path) -> set[str]:
    frame = _read_csv(path)
    if frame.empty or "model_name" not in frame.columns:
        return set()
    return set(frame["model_name"].dropna().astype(str))


def _audit_formal_seed_runs(root: Path, protocol_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefixes = [*FORMAL_MAIN_RUN_PREFIXES]
    if _p13_formal_required(root):
        prefixes.extend(OPTIONAL_P13_FORMAL_RUN_PREFIXES)
    else:
        gate_report = root / "results/paper_tables/p12_p13_promotion_gate/promotion_gate_report.csv"
        rows.append(
            _check(
                "p13",
                "formal_seed_run:p13_not_promoted_not_required",
                gate_report,
                True,
                detail="P13 formal is conditional on validation promotion gate.",
            )
        )
    for prefix in prefixes:
        for seed in SEEDS:
            run_dir = root / "results" / f"{prefix}_s{seed}"
            manifest = _read_run_manifest(run_dir)
            comparison = run_dir / "metrics/hpo_model_final_comparison.csv"
            returns = run_dir / "metrics/hpo_model_final_daily_returns.csv"
            search_space = run_dir / "logs/hpo_search_space_manifest.csv"
            rows.append(
                _check(
                    _formal_prefix_phase(prefix),
                    f"formal_seed_run:{prefix}_s{seed}",
                    run_dir,
                    bool(manifest)
                    and manifest.get("protocol_id") == protocol_id
                    and manifest.get("diagnostic_status") == "formal"
                    and manifest.get("rankable_in_unified_table") is True
                    and _manifest_uses_active_protocol(manifest)
                    and comparison.exists()
                    and returns.exists()
                    and search_space.exists(),
                    detail=json.dumps(
                        {
                            "manifest": bool(manifest),
                            "diagnostic_status": manifest.get("diagnostic_status"),
                            "rankable": manifest.get("rankable_in_unified_table"),
                            "rebalance_mode": manifest.get("rebalance_mode"),
                            "execution_activity_protocol": manifest.get("execution_activity_protocol"),
                            "scheduler_blocks_model_actions": manifest.get("scheduler_blocks_model_actions"),
                            "activity_gate_enforced": manifest.get("activity_gate_enforced"),
                            "hpo_metric": manifest.get("hpo_metric"),
                            "hpo_objective": manifest.get("hpo_objective"),
                            "comparison": comparison.exists(),
                            "returns": returns.exists(),
                            "search_space": search_space.exists(),
                        },
                        ensure_ascii=False,
                    ),
                )
            )
    return rows


def _formal_prefix_phase(prefix: str) -> str:
    if "P7" in prefix:
        return "p7"
    if "P9" in prefix:
        return "p9"
    if "P12" in prefix:
        return "p12"
    if "P13" in prefix:
        return "p13"
    return "formal"


def _manifest_uses_active_protocol(manifest: Mapping[str, Any]) -> bool:
    return (
        manifest.get("rebalance_mode") == "daily"
        and manifest.get("execution_activity_protocol") == ACTIVE_ACTIVITY_PROTOCOL
        and manifest.get("turnover_optimization_protocol_id") == "turnover_active_v1"
        and manifest.get("scheduler_blocks_model_actions") is False
        and manifest.get("activity_gate_enforced") is True
        and manifest.get("hpo_metric") == ACTIVE_HPO_OBJECTIVE
        and manifest.get("hpo_objective") == ACTIVE_HPO_OBJECTIVE
        and manifest.get("hpo_activity_constraints_enabled") is True
    )


def _audit_paper_tables(root: Path, protocol_id: str, data_cutoff_date: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [*FORMAL_PAPER_TABLE_GROUPS]
    if _p13_formal_required(root):
        groups.extend(OPTIONAL_P13_FORMAL_PAPER_TABLE_GROUPS)
    else:
        gate_report = root / "results/paper_tables/p12_p13_promotion_gate/promotion_gate_report.csv"
        rows.append(
            _check(
                "p13",
                "formal_paper_table_group:p13_not_promoted_not_required",
                gate_report,
                True,
                detail="P13 formal table is conditional on validation promotion gate.",
            )
        )
    for group in groups:
        group_dir = root / "results/paper_tables" / group
        manifest = _read_json(group_dir / "paper_aggregate_manifest.json")
        main = group_dir / "paper_main_comparison.csv"
        seed_summary = group_dir / "paper_seed_summary.csv"
        paired = group_dir / "paper_paired_statistics.csv"
        formal_filter = manifest.get("formal_filter") if isinstance(manifest.get("formal_filter"), Mapping) else {}
        rows.append(
            _check(
                "aggregation",
                f"formal_paper_table_group:{group}",
                group_dir,
                bool(manifest)
                and main.exists()
                and seed_summary.exists()
                and paired.exists()
                and _group_has_formal_rows(main)
                and _formal_filter_matches(formal_filter, protocol_id, data_cutoff_date),
                detail=json.dumps(
                    {
                        "manifest": bool(manifest),
                        "protocol_id": formal_filter.get("required_protocol_id") if isinstance(formal_filter, Mapping) else None,
                        "data_cutoff_date": formal_filter.get("required_data_cutoff_date") if isinstance(formal_filter, Mapping) else None,
                        "require_formal_manifest": formal_filter.get("require_formal_manifest") if isinstance(formal_filter, Mapping) else None,
                        "require_availability_mask_contract": formal_filter.get("require_availability_mask_contract") if isinstance(formal_filter, Mapping) else None,
                        "main": main.exists(),
                        "seed_summary": seed_summary.exists(),
                        "paired": paired.exists(),
                        "has_formal_rows": _group_has_formal_rows(main),
                        "expected_protocol_id": protocol_id,
                        "expected_data_cutoff_date": data_cutoff_date,
                    },
                    ensure_ascii=False,
                ),
            )
        )
        if group == "main_hpo_5seed":
            frame = _read_csv(main)
            rows.append(
                _check(
                    "aggregation",
                    "formal_paper_table_group:main_hpo_5seed_requires_p1_from_hpo_sources",
                    main,
                    _main_hpo_5seed_sources_valid(frame),
                    detail=json.dumps(_main_hpo_5seed_source_detail(frame), ensure_ascii=False),
                )
            )
        if group == "main_hpo_plus_p9":
            frame = _read_csv(main)
            rows.append(
                _check(
                    "aggregation",
                    "formal_paper_table_group:main_hpo_plus_p9_requires_p1_and_p9_sources",
                    main,
                    _main_hpo_plus_p9_sources_valid(frame),
                    detail=json.dumps(_main_hpo_plus_p9_source_detail(frame), ensure_ascii=False),
                )
            )
    for group in DIAGNOSTIC_PAPER_TABLE_GROUPS:
        group_dir = root / "results/paper_tables" / group
        manifest = _read_json(group_dir / "paper_aggregate_manifest.json")
        diagnostic = _read_json(group_dir / "diagnostic_status.json")
        main = group_dir / "paper_main_comparison.csv"
        seed_summary = group_dir / "paper_seed_summary.csv"
        paired = group_dir / "paper_paired_statistics.csv"
        not_applicable = group_dir / "not_applicable_reason.txt"
        rows.append(
            _check(
                "aggregation",
                f"diagnostic_paper_table_group:{group}",
                group_dir,
                bool(manifest)
                and diagnostic.get("status") == "diagnostic_complete"
                and main.exists()
                and seed_summary.exists()
                and (paired.exists() or not_applicable.exists()),
                detail=json.dumps(
                    {
                        "manifest": bool(manifest),
                        "diagnostic_status": diagnostic.get("status"),
                        "diagnostic_reason": diagnostic.get("reason"),
                        "source_run_dir_count": diagnostic.get("source_run_dir_count"),
                        "source_run_manifest_count": diagnostic.get("source_run_manifest_count"),
                        "main": main.exists(),
                        "seed_summary": seed_summary.exists(),
                        "paired": paired.exists(),
                        "not_applicable": not_applicable.exists(),
                    },
                    ensure_ascii=False,
                ),
            )
        )
    return rows


def _formal_filter_matches(formal_filter: Mapping[str, Any], protocol_id: str, data_cutoff_date: str) -> bool:
    return (
        _clean(formal_filter.get("required_protocol_id")) == protocol_id
        and _normalize_date_token(formal_filter.get("required_data_cutoff_date")) == _normalize_date_token(data_cutoff_date)
        and formal_filter.get("require_formal_manifest") is True
        and formal_filter.get("require_availability_mask_contract") is True
    )


def _main_hpo_5seed_sources_valid(frame: pd.DataFrame) -> bool:
    detail = _main_hpo_5seed_source_detail(frame)
    return bool(
        detail["expected_source_runs_matched"]
        and detail["source_files"] == ["baseline_comparison.csv"]
    )


def _main_hpo_5seed_source_detail(frame: pd.DataFrame) -> dict[str, Any]:
    expected = {f"{P1_MAIN_HPO_RUN_PREFIX}_s{seed}" for seed in SEEDS}
    if frame.empty:
        return {
            "rows": 0,
            "expected_source_runs": sorted(expected),
            "actual_source_runs": [],
            "source_files": [],
            "expected_source_runs_matched": False,
        }
    actual = (
        set(frame["source_run"].dropna().astype(str).unique())
        if "source_run" in frame.columns
        else set()
    )
    source_files = (
        sorted(frame["source_file"].dropna().astype(str).unique())
        if "source_file" in frame.columns
        else []
    )
    return {
        "rows": int(len(frame)),
        "expected_source_runs": sorted(expected),
        "actual_source_runs": sorted(actual),
        "source_files": source_files,
        "expected_source_runs_matched": actual == expected,
    }


def _main_hpo_plus_p9_sources_valid(frame: pd.DataFrame) -> bool:
    detail = _main_hpo_plus_p9_source_detail(frame)
    return bool(
        detail["expected_source_runs_matched"]
        and detail["source_files"] == ["baseline_comparison.csv", "hpo_model_final_comparison.csv"]
    )


def _main_hpo_plus_p9_source_detail(frame: pd.DataFrame) -> dict[str, Any]:
    expected_p1 = {f"{P1_MAIN_HPO_RUN_PREFIX}_s{seed}" for seed in SEEDS}
    expected_p9 = {f"EXP09_P9_formal_hpo_related_work_s{seed}" for seed in SEEDS}
    expected = expected_p1 | expected_p9
    if frame.empty:
        return {
            "rows": 0,
            "expected_p1_source_runs": sorted(expected_p1),
            "expected_p9_source_runs": sorted(expected_p9),
            "actual_source_runs": [],
            "source_files": [],
            "expected_source_runs_matched": False,
        }
    actual = (
        set(frame["source_run"].dropna().astype(str).unique())
        if "source_run" in frame.columns
        else set()
    )
    source_files = (
        sorted(frame["source_file"].dropna().astype(str).unique())
        if "source_file" in frame.columns
        else []
    )
    return {
        "rows": int(len(frame)),
        "expected_p1_source_runs": sorted(expected_p1),
        "expected_p9_source_runs": sorted(expected_p9),
        "actual_source_runs": sorted(actual),
        "source_files": source_files,
        "expected_source_runs_matched": actual == expected,
    }


def _audit_artifact_bundle(root: Path, protocol_id: str, data_cutoff_date: str) -> list[dict[str, Any]]:
    manifest_path = root / "paper/artifact_bundle/MANIFEST.json"
    manifest = _read_json(manifest_path)
    rows = [
        _check(
            "p11",
            "artifact_bundle_formal_manifest",
            manifest_path,
            manifest.get("protocol_id") == protocol_id
            and manifest.get("data_cutoff_date") == data_cutoff_date
            and manifest.get("bundle_status") == "formal",
            detail=json.dumps(
                {
                    "protocol_id": manifest.get("protocol_id"),
                    "data_cutoff_date": manifest.get("data_cutoff_date"),
                    "bundle_status": manifest.get("bundle_status"),
                },
                ensure_ascii=False,
            ),
        ),
        _check(
            "p11",
            "experiment_run_ledger_exists",
            root / "results/full_reproduction/core13_v2_full_reset_20260522/experiment_run_ledger.csv",
            (root / "results/full_reproduction/core13_v2_full_reset_20260522/experiment_run_ledger.csv").exists(),
        ),
        _check(
            "p10",
            "table_figure_manifest_core13",
            root / "paper/table_figure_manifest.md",
            protocol_id in (root / "paper/table_figure_manifest.md").read_text(encoding="utf-8")
            if (root / "paper/table_figure_manifest.md").exists()
            else False,
        ),
    ]
    return rows


def _cutoff_manifest_check(root: Path, phase: str, requirement: str, path: Path, key: str, expected: str) -> dict[str, Any]:
    payload = _read_json(path)
    actual = _normalize_date_token(payload.get(key))
    return _check(phase, requirement, path, actual == _normalize_date_token(expected), detail=f"{key}={actual}; expected={_normalize_date_token(expected)}")


def _manifest_value_check(path: Path, phase: str, requirement: str, key: str, expected: Any) -> dict[str, Any]:
    payload = _read_json(path)
    return _check(phase, requirement, path, payload.get(key) == expected, detail=f"{key}={payload.get(key)!r}; expected={expected!r}")


def _calendar_loss_check(root: Path) -> dict[str, Any]:
    path = root / "data/reports/core13_calendar_loss_summary.json"
    payload = _read_json(path)
    return _check(
        "p-1",
        "availability_mask_calendar_loss_passed",
        path,
        payload.get("data_mode") == "availability_mask" and payload.get("passed") is True,
        detail=json.dumps(
            {
                "data_mode": payload.get("data_mode"),
                "passed": payload.get("passed"),
                "strict_common_history_passed": payload.get("strict_common_history_passed"),
                "retention": payload.get("calendar_loss_retention_ratio"),
            },
            ensure_ascii=False,
        ),
    )


def _asset_count_check(root: Path) -> dict[str, Any]:
    path = root / "data/processed/core13_asset_universe.csv"
    try:
        frame = pd.read_csv(path)
        count = int(frame.shape[0])
    except Exception:  # noqa: BLE001
        count = -1
    return _check("p-1", "core13_asset_count_13", path, count == 13, detail=f"asset_count={count}")


def _metrics_factory_check(root: Path) -> dict[str, Any]:
    path = root / "data/reports/core13_metrics_factory_manifest.json"
    payload = _read_json(path)
    return _check(
        "p-1",
        "metrics_factory_core13_adj_nav",
        path,
        payload.get("return_source") == "adj_nav" and int(payload.get("feature_count") or 0) >= 40,
        detail=f"return_source={payload.get('return_source')}, feature_count={payload.get('feature_count')}",
    )


def _group_has_formal_rows(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return False
    if frame.empty:
        return False
    if "diagnostic_status" in frame.columns and not frame["diagnostic_status"].astype(str).eq("formal").all():
        return False
    if "rankable_in_unified_table" in frame.columns and not frame["rankable_in_unified_table"].map(_truthy).all():
        return False
    return True


def _check(phase: str, requirement: str, ref: str | Path, passed: bool, *, detail: str = "") -> dict[str, Any]:
    return {
        "phase": phase,
        "requirement": requirement,
        "status": "pass" if passed else "fail",
        "kind": "file" if isinstance(ref, Path) else "external",
        "ref": str(ref),
        "detail": detail,
    }


def _get(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _read_run_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "logs/run_manifest.json")
    snapshot = _read_yaml(run_dir / "logs/config_snapshot.yaml")
    if not snapshot:
        return manifest
    merged = dict(manifest)
    execution_activity = _get(snapshot, "execution_activity") or {}
    hpo = _get(snapshot, "hpo") or {}
    rankability = _get(snapshot, "rankability") or {}
    fallbacks = {
        "protocol_id": _get(snapshot, "protocol", "protocol_id"),
        "data_cutoff_date": _get(snapshot, "protocol", "data_cutoff_date"),
        "model_extension_id": _get(snapshot, "new_model_protocol", "model_extension_id"),
        "diagnostic_status": rankability.get("diagnostic_status"),
        "rankable_in_unified_table": rankability.get("rankable_in_unified_table"),
        "rebalance_mode": _get(snapshot, "rebalance", "mode"),
        "execution_activity_protocol": execution_activity.get("protocol"),
        "turnover_optimization_protocol_id": execution_activity.get("turnover_optimization_protocol_id"),
        "scheduler_blocks_model_actions": execution_activity.get("scheduler_blocks_model_actions"),
        "activity_gate_enforced": execution_activity.get("activity_gate_enforced"),
        "hpo_metric": hpo.get("metric"),
        "hpo_objective": hpo.get("objective"),
        "hpo_activity_constraints_enabled": _get(snapshot, "hpo", "activity_constraints", "enabled"),
    }
    if not merged.get("execution_activity") and execution_activity:
        merged["execution_activity"] = execution_activity
    for key, value in fallbacks.items():
        if _manifest_value_missing(merged.get(key)) and value is not None:
            merged[key] = value
    return merged


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _normalize_date_token(value: Any) -> str:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _manifest_value_missing(value: Any) -> bool:
    return value is None or _clean(value) == ""


def _scoped_root(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    return resolved


def _scoped_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"ERR_FORMAL_READINESS_OUT_OF_SCOPE: {resolved}") from exc
    return resolved


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Core-13 v2 formal experiment readiness.")
    parser.add_argument("--output-dir", default="results/full_reproduction/core13_v2_full_reset_20260522")
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    parser.add_argument("--data-cutoff-date", default=DATA_CUTOFF_DATE)
    parser.add_argument("--fail-on-no-go", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    outputs = audit_formal_readiness(
        output_dir=args.output_dir,
        protocol_id=args.protocol_id,
        data_cutoff_date=args.data_cutoff_date,
    )
    payload = _read_json(outputs["json"])
    print(json.dumps({"status": payload.get("status"), "fail_count": payload.get("fail_count"), "json": str(outputs["json"]), "csv": str(outputs["csv"])}, ensure_ascii=False))
    if args.fail_on_no_go and payload.get("status") != "go":
        raise SystemExit(1)
    return outputs


if __name__ == "__main__":
    main()


__all__ = ["audit_formal_readiness"]
