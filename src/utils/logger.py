from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


DAILY_OUTPUT_SCHEMAS: dict[str, tuple[str, ...]] = {
    "daily_returns": (
        "date",
        "decision_date",
        "execution_date",
        "execution_price_type",
        "next_valuation_date",
        "split",
        "seed",
        "fold_id",
        "model_name",
        "variant_id",
        "pre_execution_return",
        "post_execution_return",
        "gross_return",
        "transaction_cost",
        "transaction_cost_on_initial_nav",
        "net_return",
        "portfolio_log_return",
        "nav",
    ),
    "daily_weights": ("date", "split", "seed", "fold_id", "model_name", "variant_id", "asset_id", "weight"),
    "daily_turnover": (
        "date",
        "decision_date",
        "execution_date",
        "execution_price_type",
        "next_valuation_date",
        "split",
        "seed",
        "fold_id",
        "model_name",
        "variant_id",
        "turnover",
        "rebalance_action",
        "rebalance_intensity",
        "average_holding_period",
    ),
    "daily_rebalance": (
        "date",
        "decision_date",
        "execution_date",
        "execution_price_type",
        "next_valuation_date",
        "split",
        "seed",
        "fold_id",
        "model_name",
        "variant_id",
        "rebalance_action",
        "rebalance_intensity",
        "estimated_turnover",
        "realized_turnover",
        "turnover",
        "estimated_cost",
        "realized_cost",
        "q_hold",
        "q_rebalance",
        "q_gap",
        "fallback_reason",
    ),
    "daily_costs": (
        "date",
        "decision_date",
        "execution_date",
        "execution_price_type",
        "next_valuation_date",
        "split",
        "seed",
        "fold_id",
        "model_name",
        "variant_id",
        "proportional_cost",
        "fixed_cost",
        "slippage_cost",
        "market_impact_cost",
        "total_transaction_cost",
        "estimated_cost",
        "realized_cost",
        "turnover",
    ),
}
SPLIT_METRIC_FILES = {
    "train": "train_metrics.csv",
    "validation": "validation_metrics.csv",
    "test": "test_metrics.csv",
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_RESULT_WRITE_DIRS = (
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "metrics_factory",
    PROJECT_ROOT / "data" / "reports",
)
ERR_SECURITY_PATH_DENIED = "ERR_SECURITY_PATH_DENIED"


def assert_output_path_under_run_dir(path: str | Path, run_dir: str | Path) -> Path:
    target = Path(path)
    root = Path(run_dir)
    target_resolved = target.expanduser().resolve()
    root_resolved = root.expanduser().resolve()
    _assert_not_forbidden_result_path(root_resolved)
    _assert_not_forbidden_result_path(target_resolved)
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"{ERR_SECURITY_PATH_DENIED}: output path outside run_dir: {target_resolved}") from exc
    return target


def _resolve_safe_run_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    _assert_not_forbidden_result_path(root)
    return root


def _assert_not_forbidden_result_path(path: str | Path) -> None:
    resolved = Path(path).expanduser().resolve()
    for forbidden in FORBIDDEN_RESULT_WRITE_DIRS:
        forbidden_resolved = forbidden.resolve()
        if resolved == forbidden_resolved or resolved.is_relative_to(forbidden_resolved):
            raise ValueError(f"{ERR_SECURITY_PATH_DENIED}: result output path denied: {resolved}")


def save_yaml_atomic(payload: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)
        return target
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def save_json_atomic(payload: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)
        return target
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def write_run_outputs(
    result: Any,
    run_dir: str | Path,
    *,
    metrics_by_split: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    command: str | None = None,
    asset_list: Any = None,
    data_split: Any = None,
    manifest_overrides: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Path]:
    output_root = _resolve_safe_run_dir(run_dir)
    metrics_dir = output_root / "metrics"
    logs_dir = output_root / "logs"
    assert_output_path_under_run_dir(metrics_dir, output_root)
    assert_output_path_under_run_dir(logs_dir, output_root)
    assert_output_path_under_run_dir(output_root / "figures", output_root)
    registry_target = assert_output_path_under_run_dir(registry_path or logs_dir / "run_registry.sqlite", output_root)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: dict[str, Path] = {}
    daily_frames: dict[str, pd.DataFrame] = {}

    for name, columns in DAILY_OUTPUT_SCHEMAS.items():
        frame = _normalized_daily_frame(_value(result, name), columns)
        daily_frames[name] = frame
        artifact_paths[name] = _write_csv_atomic(frame, metrics_dir / f"{name}.csv")

    metrics_payload = _value(result, "metrics")
    if metrics_payload is None:
        metrics_payload = _calculated_metrics(daily_frames)
    artifact_paths["metrics"] = _write_metric_csv(metrics_payload, metrics_dir / "metrics.csv")

    split_payloads = metrics_by_split if metrics_by_split is not None else _split_metric_payloads(metrics_payload)
    for split, filename in SPLIT_METRIC_FILES.items():
        artifact_paths[f"{split}_metrics"] = _write_metric_csv(
            _split_metric_row(split, _mapping(split_payloads).get(split)),
            metrics_dir / filename,
        )
    artifact_paths.update(_write_matrix_metric_outputs(result, metrics_dir))
    artifact_paths["statistics_summary"] = _write_statistics_summary(result, metrics_dir / "statistics_summary.csv", config=config)
    artifact_paths.update(
        _write_figure_outputs(
            result,
            output_root,
            daily_frames=daily_frames,
            metrics_payload=metrics_payload,
            config=config,
        )
    )
    artifact_paths.update(
        _write_log_outputs(
            result,
            logs_dir,
            config=config,
            config_path=config_path,
            command=command,
            asset_list=asset_list,
            data_split=data_split,
            daily_frames=daily_frames,
            manifest_overrides=manifest_overrides,
        )
    )
    manifest = _read_json_file(artifact_paths["run_manifest"])
    run_id = str(_first_not_none(manifest.get("run_id"), _value(result, "run_id"), "default"))
    artifact_paths["run_registry"] = write_run_registry(
        registry_target,
        run_id=run_id,
        run_dir=output_root,
        artifact_paths=artifact_paths,
        manifest=manifest,
        lineage=_value(result, "lineage"),
    )
    for artifact_path in artifact_paths.values():
        assert_output_path_under_run_dir(artifact_path, output_root)
    return artifact_paths


def _write_matrix_metric_outputs(result: Any, metrics_dir: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    matrix_name = _value(result, "output_name")
    if isinstance(matrix_name, str) and matrix_name:
        matrix_frame = _frame(_value(result, matrix_name))
        if not matrix_frame.empty or len(matrix_frame.columns) > 0:
            artifacts[matrix_name] = _write_csv_atomic(matrix_frame, metrics_dir / f"{matrix_name}.csv")
    for name in EXTRA_METRIC_FRAME_OUTPUTS:
        if name in artifacts:
            continue
        frame = _frame(_value(result, name))
        if frame.empty and len(frame.columns) == 0:
            continue
        frame = _normalized_extra_metric_frame(name, frame)
        artifacts[name] = _write_csv_atomic(frame, metrics_dir / f"{name}.csv")
    return artifacts


def _write_figure_outputs(
    result: Any,
    output_root: Path,
    *,
    daily_frames: Mapping[str, pd.DataFrame],
    metrics_payload: Any,
    config: Mapping[str, Any] | None,
) -> dict[str, Path]:
    from src.utils.plotting import generate_figures

    plot_payload: dict[str, Any] = dict(result) if isinstance(result, Mapping) else {}
    plot_payload.update(daily_frames)
    plot_payload["metrics"] = metrics_payload
    figure_paths = generate_figures(plot_payload, output_root, config=config)
    return {f"figure_{Path(name).stem}": path for name, path in figure_paths.items()}


def _write_statistics_summary(result: Any, path: Path, *, config: Mapping[str, Any] | None) -> Path:
    from src.utils.stats import STATISTICS_SUMMARY_COLUMNS, run_statistical_tests

    explicit = _value(result, "statistics_summary")
    if explicit is not None:
        frame = _frame(explicit)
    elif _value(result, "model_returns") is not None and _value(result, "benchmark_returns") is not None:
        frame = run_statistical_tests(
            _value(result, "model_returns"),
            _value(result, "benchmark_returns"),
            config=config,
            auxiliary_forecast_errors=_value(result, "auxiliary_forecast_errors"),
        )
    else:
        frame = pd.DataFrame([{"test_name": "all", "status": "not_applicable"}])
    for column in STATISTICS_SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return _write_csv_atomic(frame.loc[:, list(STATISTICS_SUMMARY_COLUMNS)], path)


FEATURE_PROVENANCE_COLUMNS = (
    "feature_name",
    "feature_group",
    "source_file",
    "source_family",
    "window",
    "uses_price",
    "uses_volume",
    "uses_return",
    "uses_cross_asset_data",
    "is_metrics_factory_feature",
    "is_auxiliary_target",
    "is_model_feature",
    "requires_shift",
    "shift_steps",
    "fit_scope",
    "leakage_risk_level",
    "leakage_check_status",
    "drop_reason",
)
FEATURE_GROUP_SUMMARY_COLUMNS = (
    "feature_group",
    "source_family",
    "n_total",
    "n_used",
    "n_dropped",
    "n_shifted",
    "n_train_only_fit",
    "n_warning",
    "n_fail",
)
METRICS_FACTORY_AUDIT_COLUMNS = (
    "feature_name",
    "ts_code",
    "date",
    "stored_value",
    "recomputed_value",
    "abs_error",
    "status",
)
COST_CALIBRATION_COLUMNS = (
    "amount_bucket",
    "turnover_rate_bucket",
    "sigma20_bucket",
    "sample_count",
    "realized_bps_mean",
    "realized_bps_median",
    "fallback_used",
    "fallback_reason",
    "status",
)
HPO_TRIAL_COLUMNS = (
    "model_name",
    "fold_id",
    "study_name",
    "trial_number",
    "seed",
    "params_json",
    "state",
    "objective_value",
    "validation_metric",
    "train_start",
    "train_end",
    "duration_sec",
    "pruned_step",
    "fail_reason",
)
SEED_AGGREGATE_COLUMNS = ("model_name", "metric_name", "n_seeds", "mean", "std", "min", "max", "median")
FEATURE_SELECTION_COLUMNS = ("feature_name", "selected", "score", "rank", "drop_reason", "skip_reason", "status")
BASELINE_DAILY_DIAGNOSTICS_COLUMNS = (
    "date",
    "decision_date",
    "execution_date",
    "model_name",
    "paper_model_id",
    "seed",
    "fold_id",
    "model_extension_id",
    "post_hoc_development_disclosure",
    "test_used_for_model_selection",
    "gate_action",
    "gate_action_index",
    "rho",
    "rebalance_intensity",
    "rebalance_values",
    "candidate_turnover",
    "estimated_turnover",
    "realized_turnover",
    "estimated_cost",
    "realized_cost",
    "scheduler_allowed_rebalance",
    "forced_hold_reason",
    "CVaR_loss_5",
    "drawdown",
    "candidate_weights_json",
    "executed_weights_json",
    "hierarchy_action",
    "hierarchy_action_name",
    "ppo_actor_update_mask",
    "ppo_attribution_weight",
    "platform_adapted_surrogate",
    "child_model_name",
    "baseline_family",
    "optimizer_name",
    "include_count",
    "exclude_count",
    "neutral_count",
    "selected_asset_count",
    "optimizer_asset_count",
    "optimizer_status",
    "fallback_reason",
)
HPO_MODEL_FINAL_DAILY_DIAGNOSTICS = "hpo_model_final_daily_diagnostics"
HPO_MODEL_FINAL_DAILY_DIAGNOSTICS_COLUMNS = (
    "hpo_model_name",
    *BASELINE_DAILY_DIAGNOSTICS_COLUMNS,
    "best_trial_number",
    "best_value",
)
BASELINE_TRAINING_SUMMARY_COLUMNS = (
    "model_name",
    "paper_model_id",
    "child_model_name",
    "baseline_family",
    "status",
    "training_algorithm",
    "rl_training",
    "platform_native_rl_training",
    "proxy_training",
    "external_original_implementation",
    "rankable_in_unified_table",
    "model_extension_id",
    "post_hoc_development_disclosure",
    "test_used_for_model_selection",
    "clean_room_reimplementation",
    "algorithm_fidelity",
    "dqn_role",
    "optimizer_name",
    "platform_adapted_surrogate",
    "hierarchy_action_distribution",
    "hierarchy_action_0_count",
    "hierarchy_action_1_count",
    "hierarchy_action_2_count",
    "hierarchy_action_3_count",
    "hierarchy_action_4_count",
    "factorized_q",
    "portfolio_level_reward_shared",
    "counterfactual_asset_reward",
    "platform_adapted_approximation",
    "dqn_signal_include_rate",
    "optimizer_allocation_method",
    "optimizer_fallback_rate",
    "source_code_vendored",
    "license",
    "data_protocol",
    "execution_protocol",
    "evaluation_protocol",
    "cost_model_shared",
    "cost_availability",
    "constraint_protocol_shared",
    "external_repo",
    "external_license",
    "external_dependency_stack",
    "repo_path",
    "external_work_dir",
    "import_results_csv",
    "command",
    "fail_reason",
    "returncode",
    "out_of_scope_edge",
    "checkpoint_best_path",
    "checkpoint_last_path",
    "evaluated_checkpoint_path",
    "best_validation_metric",
    "env_steps",
    "gradient_updates",
    "max_train_steps",
    "max_validation_steps",
    "max_gradient_updates_per_epoch",
    "gate_training",
    "gate_entropy",
    "p_rebalance_mean",
    "rebalance_frequency",
    "portfolio_vector_memory",
    "pre_execution_return_in_actor_loss",
    "online_stochastic_batch_learning",
    "osbl_sample_count",
    "osbl_batch_count",
)
BASELINE_TRAINING_HISTORY_COLUMNS = (
    "model_name",
    "epoch",
    "step",
    "env_steps",
    "gradient_updates",
    "max_train_steps",
    "max_validation_steps",
    "max_gradient_updates_per_epoch",
    "train_reward",
    "validation_metric",
    "loss",
    "gate_loss",
    "gate_entropy",
    "gate_grad_norm",
    "p_rebalance_mean",
    "rebalance_frequency",
    "portfolio_vector_memory",
    "pre_execution_return_in_actor_loss",
    "ppo_actor_update_mask_rate",
    "ppo_attribution_weight_mean",
    "platform_adapted_surrogate",
    "training_algorithm",
    "online_stochastic_batch_learning",
    "clean_room_reimplementation",
    "source_code_vendored",
    "osbl_sample_count",
    "osbl_batch_count",
    "include_count",
    "neutral_count",
    "exclude_count",
    "selected_asset_count",
    "optimizer_asset_count",
    "optimizer_fallback_count",
    "status",
)
EXTRA_METRIC_FRAME_OUTPUTS = (
    "main_comparison",
    "baseline_comparison",
    "baseline_daily_diagnostics",
    "hpo_final_reports_table",
    "hpo_model_final_reports",
    "hpo_model_final_comparison",
    "hpo_model_final_daily_returns",
    "hpo_model_final_daily_weights",
    "hpo_model_final_daily_turnover",
    "hpo_model_final_daily_rebalance",
    "hpo_model_final_daily_costs",
    HPO_MODEL_FINAL_DAILY_DIAGNOSTICS,
    "gate_actions",
    "gate_action_summary",
    "cage_eiie_candidate_weights",
    "cage_final_weights",
    "turnover_cost_breakdown",
    "risk_metrics",
    "validation_selection_report",
    "hpo_model_final_gate_actions",
    "hpo_model_final_gate_action_summary",
    "hpo_model_final_cage_eiie_candidate_weights",
    "hpo_model_final_cage_final_weights",
    "hpo_model_final_turnover_cost_breakdown",
    "hpo_model_final_risk_metrics",
    "hpo_model_final_validation_selection_report",
    "ra_gt_rcpo_daily_diagnostics",
    "ra_gt_rcpo_constraint_multipliers",
    "ra_gt_rcpo_graph_diagnostics",
    "ra_gt_rcpo_actor_critic_training_history",
    "ra_gt_rcpo_risk_decomposition",
    "hpo_model_final_ra_gt_rcpo_daily_diagnostics",
    "hpo_model_final_ra_gt_rcpo_constraint_multipliers",
    "hpo_model_final_ra_gt_rcpo_graph_diagnostics",
    "hpo_model_final_ra_gt_rcpo_actor_critic_training_history",
    "hpo_model_final_ra_gt_rcpo_risk_decomposition",
    "paper_main_comparison",
    "paper_diagnostic_comparison",
    "paper_paired_statistics",
    "paper_seed_summary",
    "closest_hybrid_figure_source",
)
EXTRA_METRIC_FRAME_SCHEMAS = {
    "baseline_daily_diagnostics": BASELINE_DAILY_DIAGNOSTICS_COLUMNS,
    HPO_MODEL_FINAL_DAILY_DIAGNOSTICS: HPO_MODEL_FINAL_DAILY_DIAGNOSTICS_COLUMNS,
}


def _write_log_outputs(
    result: Any,
    logs_dir: Path,
    *,
    config: Mapping[str, Any] | None,
    config_path: str | Path | None,
    command: str | None,
    asset_list: Any,
    data_split: Any,
    daily_frames: Mapping[str, pd.DataFrame],
    manifest_overrides: Mapping[str, Any] | None,
) -> dict[str, Path]:
    config_payload = dict(config) if isinstance(config, Mapping) else {"status": "not_applicable"}
    output_asset_list = _coerce_asset_list(asset_list) if asset_list is not None else _asset_list(result, daily_frames)
    output_data_split = data_split if data_split is not None else _data_split(result, config)
    manifest = _run_manifest(
        result,
        config,
        daily_frames,
        config_path=config_path,
        command=command,
        asset_list=output_asset_list,
        data_split=output_data_split,
        overrides=manifest_overrides,
    )

    artifacts: dict[str, Path] = {}
    artifacts["train_log"] = _write_text_atomic(_value(result, "train_log") or "", logs_dir / "train.log")
    artifacts["config_snapshot"] = save_yaml_atomic(config_payload, logs_dir / "config_snapshot.yaml")
    artifacts["selected_input_matrix"] = save_yaml_atomic(
        _value(result, "selected_input_matrix") or {"status": "not_applicable"},
        logs_dir / "selected_input_matrix.yaml",
    )
    artifacts["input_matrix_feature_groups"] = save_json_atomic(
        _value(result, "input_matrix_feature_groups") or {"status": "not_applicable"},
        logs_dir / "input_matrix_feature_groups.json",
    )
    artifacts["pca_report"] = save_json_atomic(
        _value(result, "pca_report") or {"status": "not_applicable"},
        logs_dir / "pca_report.json",
    )
    artifacts["feature_provenance_report"] = _write_log_csv(
        _value(result, "feature_provenance_report"),
        logs_dir / "feature_provenance_report.csv",
        FEATURE_PROVENANCE_COLUMNS,
    )
    artifacts["feature_group_summary"] = _write_log_csv(
        _value(result, "feature_group_summary"),
        logs_dir / "feature_group_summary.csv",
        FEATURE_GROUP_SUMMARY_COLUMNS,
    )
    artifacts["feature_selection_report"] = _write_log_csv(
        _value(result, "feature_selection_report"),
        logs_dir / "feature_selection_report.csv",
        FEATURE_SELECTION_COLUMNS,
    )
    artifacts["metrics_factory_audit_sample"] = _write_log_csv(
        _value(result, "metrics_factory_audit_sample"),
        logs_dir / "metrics_factory_audit_sample.csv",
        METRICS_FACTORY_AUDIT_COLUMNS,
    )
    artifacts["cost_calibration_report"] = _write_log_csv(
        _value(result, "cost_calibration_report"),
        logs_dir / "cost_calibration_report.csv",
        COST_CALIBRATION_COLUMNS,
    )
    artifacts["hpo_trials"] = _write_log_csv(
        _value(result, "hpo_trials"),
        logs_dir / "hpo_trials.csv",
        HPO_TRIAL_COLUMNS,
        status_column="state",
    )
    artifacts["seed_aggregate_summary"] = _write_log_csv(
        _value(result, "seed_aggregate_summary"),
        logs_dir / "seed_aggregate_summary.csv",
        SEED_AGGREGATE_COLUMNS,
    )
    artifacts["baseline_training_summary"] = _write_log_csv(
        _value(result, "baseline_training_summary"),
        logs_dir / "baseline_training_summary.csv",
        BASELINE_TRAINING_SUMMARY_COLUMNS,
    )
    artifacts["baseline_training_history"] = _write_log_csv(
        _value(result, "baseline_training_history"),
        logs_dir / "baseline_training_history.csv",
        BASELINE_TRAINING_HISTORY_COLUMNS,
    )
    matrix_name = _value(result, "output_name")
    if isinstance(matrix_name, str) and matrix_name:
        matrix_payload = _value(result, matrix_name)
        matrix_frame = _frame(matrix_payload)
        if not matrix_frame.empty or len(matrix_frame.columns) > 0:
            artifacts[f"log_{matrix_name}"] = _write_csv_atomic(matrix_frame, logs_dir / f"{matrix_name}.csv")
    artifacts["data_split"] = save_json_atomic(output_data_split, logs_dir / "data_split.json")
    artifacts["asset_list"] = _write_text_atomic("\n".join(output_asset_list) + ("\n" if output_asset_list else ""), logs_dir / "asset_list.txt")
    artifacts["fixed_ratio_weights"] = save_json_atomic(
        _value(result, "fixed_ratio_weights") or {"status": "not_applicable"},
        logs_dir / "fixed_ratio_weights.json",
    )
    artifacts["new_model_sidecar_manifest"] = save_json_atomic(
        _value(result, "new_model_sidecar_manifest") or {"status": "not_applicable"},
        logs_dir / "new_model_sidecar_manifest.json",
    )
    artifacts["run_manifest"] = save_json_atomic(manifest, logs_dir / "run_manifest.json")
    return artifacts


def _run_manifest(
    result: Any,
    config: Mapping[str, Any] | None,
    daily_frames: Mapping[str, pd.DataFrame],
    *,
    config_path: str | Path | None,
    command: str | None,
    asset_list: list[str],
    data_split: Any,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    existing = dict(_mapping(_value(result, "run_manifest")))
    cfg = _mapping(config)
    output_cfg = _mapping(cfg.get("output"))
    data_cfg = _mapping(cfg.get("data"))
    data_governance = dict(_mapping(cfg.get("data_governance")))
    execution_model = dict(_mapping(cfg.get("execution_model")))
    protocol = _mapping(cfg.get("protocol"))
    new_model_protocol = _mapping(cfg.get("new_model_protocol"))
    rankability = dict(_mapping(cfg.get("rankability")))
    portfolio = _mapping(cfg.get("portfolio"))
    metrics_factory = _mapping(data_cfg.get("metrics_factory"))
    reproducibility = _mapping(cfg.get("reproducibility"))
    device = cfg.get("device")

    execution_price = _first_not_none(
        execution_model.get("execution_price"),
        existing.get("execution_price"),
        _first_frame_value(daily_frames.get("daily_returns"), "execution_price_type"),
    )
    execution_price_type = _first_not_none(
        existing.get("execution_price_type"),
        _first_frame_value(daily_frames.get("daily_returns"), "execution_price_type"),
        _execution_price_type(execution_price),
    )
    run_name = _first_not_none(output_cfg.get("run_name"), existing.get("run_name"), "default")
    fold_id = _first_not_none(
        existing.get("fold_id"),
        _value(result, "fold_id"),
        _first_frame_value(daily_frames.get("daily_returns"), "fold_id"),
        _mapping(data_split).get("fold_id"),
    )
    same_close_idealized_execution_enabled = bool(
        execution_model.get("same_close_idealized_execution_enabled")
        or data_governance.get("same_close_idealized_execution_enabled")
        or existing.get("same_close_idealized_execution_enabled")
    )
    idealized_execution = bool(
        execution_model.get("idealized_execution")
        or existing.get("idealized_execution")
        or same_close_idealized_execution_enabled
    )
    data_mode = _first_not_none(
        data_cfg.get("data_mode"),
        existing.get("data_mode"),
        "strict_common_history" if data_cfg.get("strict_common_history_mode") is True else "availability_mask",
    )
    valuation_source = _first_not_none(
        existing.get("valuation_source"),
        data_governance.get("valuation_source"),
        data_governance.get("return_source"),
    )
    return_source = _first_not_none(existing.get("return_source"), data_governance.get("return_source"), valuation_source)
    reward_return_source = _first_not_none(data_governance.get("reward_return_source"), existing.get("reward_return_source"))
    metrics_return_source = _first_not_none(data_governance.get("metrics_return_source"), existing.get("metrics_return_source"))
    execution_price_source = _first_not_none(
        data_governance.get("execution_price_source"),
        existing.get("execution_price_source"),
    )
    valuation_execution_split = bool(
        _first_not_none(
            existing.get("valuation_execution_split"),
            data_governance.get("valuation_execution_split"),
            False,
        )
    )
    reward_valuation_split = bool(
        _first_not_none(
            existing.get("reward_valuation_split"),
            data_governance.get("reward_valuation_split"),
            False,
        )
    )
    rankable_in_unified_table = bool(
        _first_not_none(
            _value(result, "rankable_in_unified_table"),
            rankability.get("rankable_in_unified_table"),
            existing.get("rankable_in_unified_table"),
            False,
        )
    )
    diagnostic_status = _first_not_none(
        _value(result, "diagnostic_status"),
        rankability.get("diagnostic_status"),
        existing.get("diagnostic_status"),
        "diagnostic",
    )
    availability_mask_contract = dict(_mapping(_value(result, "availability_mask_contract")))
    manifest = {
        "run_id": _first_not_none(existing.get("run_id"), _value(result, "run_id"), run_name),
        "timestamp": existing.get("timestamp") or now,
        "command": _first_not_none(command, existing.get("command")),
        "config_path": str(config_path) if config_path is not None else existing.get("config_path"),
        "run_name": run_name,
        "config_hash": _first_not_none(cfg.get("config_hash"), existing.get("config_hash")),
        "data_path": _first_not_none(data_cfg.get("panel_path"), existing.get("data_path")),
        "data_hash": _first_not_none(data_cfg.get("data_hash"), existing.get("data_hash"), _value(result, "data_hash")),
        "split_id": _first_not_none(existing.get("split_id"), _value(result, "split_id"), _mapping(data_split).get("mode")),
        "asset_universe_hash": _first_not_none(
            existing.get("asset_universe_hash"),
            _value(result, "asset_universe_hash"),
            _stable_hash(asset_list),
        ),
        "device": _first_not_none(existing.get("device"), _value(result, "device"), device),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "package_versions": _first_not_none(existing.get("package_versions"), _value(result, "package_versions"), {}),
        "git_commit_if_available": _first_not_none(existing.get("git_commit_if_available"), _value(result, "git_commit_if_available")),
        "code_version": _first_not_none(existing.get("code_version"), _value(result, "code_version")),
        "created_at": existing.get("created_at") or now,
        "execution_model": execution_model,
        "data_governance": data_governance,
        "portfolio_initial_nav": _first_not_none(portfolio.get("initial_nav"), existing.get("portfolio_initial_nav")),
        "portfolio_initial_capital_currency": _first_not_none(
            portfolio.get("initial_capital_currency"),
            existing.get("portfolio_initial_capital_currency"),
        ),
        "portfolio_currency": _first_not_none(portfolio.get("currency"), existing.get("portfolio_currency")),
        "execution_price": execution_price,
        "execution_price_type": execution_price_type,
        "delayed_action_execution": bool(
            _first_not_none(execution_model.get("delayed_action_execution"), existing.get("delayed_action_execution"), False)
        ),
        "same_close_idealized_execution_enabled": same_close_idealized_execution_enabled,
        "idealized_execution": idealized_execution,
        "strict_no_lookahead_execution": bool(
            _first_not_none(execution_model.get("strict_no_lookahead_execution"), existing.get("strict_no_lookahead_execution"), False)
        ),
        "t_plus_one": bool(_first_not_none(execution_model.get("t_plus_one"), existing.get("t_plus_one"), False)),
        "initial_build_cost": bool(_first_not_none(execution_model.get("initial_build_cost"), existing.get("initial_build_cost"), False)),
        "amount_is_proxy": bool(_first_not_none(data_governance.get("amount_is_proxy"), existing.get("amount_is_proxy"), False)),
        "metrics_factory_enabled": bool(_first_not_none(metrics_factory.get("enabled"), existing.get("metrics_factory_enabled"), False)),
        "turnover_rate_all_missing": bool(_first_not_none(data_governance.get("turnover_rate_all_missing"), existing.get("turnover_rate_all_missing"), False)),
        "long_running": bool(_first_not_none(cfg.get("long_running"), existing.get("long_running"), _value(result, "long_running"), False)),
        "best_trial_number": _first_not_none(existing.get("best_trial_number"), _value(result, "best_trial_number")),
        "seed": _first_not_none(reproducibility.get("seed"), existing.get("seed"), _value(result, "seed")),
        "fold_id": fold_id,
        "protocol_id": _first_not_none(protocol.get("protocol_id"), existing.get("protocol_id")),
        "asset_universe_id": _first_not_none(protocol.get("asset_universe_id"), existing.get("asset_universe_id")),
        "data_cutoff_date": _first_not_none(protocol.get("data_cutoff_date"), existing.get("data_cutoff_date")),
        "base_protocol_id": _first_not_none(new_model_protocol.get("base_protocol_id"), existing.get("base_protocol_id")),
        "model_extension_id": _first_not_none(
            new_model_protocol.get("model_extension_id"),
            existing.get("model_extension_id"),
            _value(result, "model_extension_id"),
        ),
        "post_hoc_development_disclosure": bool(
            _first_not_none(
                new_model_protocol.get("post_hoc_development_disclosure"),
                existing.get("post_hoc_development_disclosure"),
                False,
            )
        ),
        "selection_split": _first_not_none(
            _mapping(cfg.get("hpo")).get("selection_split"),
            new_model_protocol.get("selection_split"),
            existing.get("selection_split"),
        ),
        "test_used_for_model_selection": bool(
            _first_not_none(
                new_model_protocol.get("test_used_for_model_selection"),
                existing.get("test_used_for_model_selection"),
                False,
            )
        ),
        "data_mode": data_mode,
        "data_contract": {
            "data_mode": data_mode,
            "strict_common_history_mode": bool(data_cfg.get("strict_common_history_mode", False)),
            "return_source": return_source,
            "valuation_source": valuation_source,
            "reward_return_source": reward_return_source,
            "metrics_return_source": metrics_return_source,
            "execution_price_source": execution_price_source,
        },
        "valuation_source": valuation_source,
        "return_source": return_source,
        "reward_return_source": reward_return_source,
        "metrics_return_source": metrics_return_source,
        "execution_price_source": execution_price_source,
        "valuation_execution_split": valuation_execution_split,
        "reward_valuation_split": reward_valuation_split,
        "rankability": {
            **rankability,
            "rankable_in_unified_table": rankable_in_unified_table,
            "diagnostic_status": diagnostic_status,
        },
        "rankable_in_unified_table": rankable_in_unified_table,
        "diagnostic_status": diagnostic_status,
        "availability_mask_contract": availability_mask_contract,
        "availability_mask_contract_passed": availability_mask_contract.get("availability_mask_contract_passed"),
        "min_available_assets_per_date": availability_mask_contract.get("min_available_assets_per_date"),
        "unavailable_asset_weight_abs_max": availability_mask_contract.get("unavailable_asset_weight_abs_max"),
        "daily_returns_finite": availability_mask_contract.get("daily_returns_finite"),
        "daily_nav_finite": availability_mask_contract.get("daily_nav_finite"),
        "frozen_or_imputed_valuation_count": availability_mask_contract.get("frozen_or_imputed_valuation_count"),
    }
    if overrides:
        manifest.update(dict(overrides))
    return manifest


def _asset_list(result: Any, daily_frames: Mapping[str, pd.DataFrame]) -> list[str]:
    explicit = _value(result, "asset_list")
    if explicit is None:
        explicit = _value(result, "canonical_asset_order")
    if explicit is not None:
        return _coerce_asset_list(explicit)
    weights = daily_frames.get("daily_weights")
    if weights is None or weights.empty or "asset_id" not in weights.columns:
        return []
    return [str(item) for item in weights["asset_id"].dropna().drop_duplicates().tolist()]


def _coerce_asset_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        items = [value]
    else:
        items = list(value)
    return [str(item) for item in items if pd.notna(item)]


def _data_split(result: Any, config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    explicit = _value(result, "data_split")
    if isinstance(explicit, Mapping):
        return explicit
    split = _mapping(_mapping(config).get("split"))
    if split:
        return split
    return {"status": "not_applicable"}


def _write_log_csv(
    payload: Any,
    path: Path,
    columns: tuple[str, ...],
    *,
    status_column: str = "status",
) -> Path:
    frame = _frame(payload)
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame([{status_column: "not_applicable"}])
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    if status_column in frame.columns:
        frame[status_column] = frame[status_column].fillna("not_applicable")
    return _write_csv_atomic(frame.loc[:, list(columns)], path)


def _write_text_atomic(payload: str, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)
        return target
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_frame_value(frame: pd.DataFrame | None, column: str) -> Any:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _execution_price_type(execution_price: Any) -> str | None:
    if execution_price is None:
        return None
    value = str(execution_price).lower()
    if "open" in value:
        return "open"
    if "close" in value:
        return "close"
    return str(execution_price)


def _stable_hash(value: Any) -> str | None:
    if value in (None, [], {}, ()):
        return None
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _normalized_daily_frame(value: Any, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = _frame(value)
    if frame.empty and len(frame.columns) == 0:
        return pd.DataFrame(columns=columns)
    if "date" in columns and "next_valuation_date" in columns:
        if "next_valuation_date" not in frame.columns and "date" in frame.columns:
            frame["next_valuation_date"] = frame["date"]
        if "next_valuation_date" in frame.columns:
            frame["date"] = frame["next_valuation_date"]
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, list(columns)]


def _normalized_extra_metric_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = EXTRA_METRIC_FRAME_SCHEMAS.get(name)
    if columns is None:
        return frame
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = pd.NA
    ordered = list(columns) + [column for column in result.columns if column not in columns]
    return result.loc[:, ordered]


def _calculated_metrics(daily_frames: Mapping[str, pd.DataFrame]) -> Mapping[str, Any]:
    daily_returns = daily_frames.get("daily_returns")
    if daily_returns is None or daily_returns.empty or "net_return" not in daily_returns.columns:
        return {"status": "not_applicable"}
    from src.utils.metrics import calculate_performance_metrics

    return calculate_performance_metrics(
        daily_returns,
        daily_frames.get("daily_turnover"),
        daily_frames.get("daily_costs"),
    )


def _split_metric_payloads(metrics_payload: Any) -> Mapping[str, Any]:
    metrics = _mapping(metrics_payload)
    if all(isinstance(metrics.get(split), Mapping) for split in SPLIT_METRIC_FILES):
        return metrics
    return {}


def _split_metric_row(split: str, payload: Any) -> Mapping[str, Any]:
    if payload is None:
        return {"split": split, "status": "not_applicable"}
    row = dict(_mapping(payload))
    row.setdefault("split", split)
    row.setdefault("status", "completed")
    return row


def _write_metric_csv(payload: Any, path: Path) -> Path:
    frame = _frame(payload)
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame([{"status": "not_applicable"}])
    return _write_csv_atomic(frame, path)


def _write_csv_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            frame.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)
        return target
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        return pd.read_csv(value)
    if isinstance(value, Mapping):
        try:
            return pd.DataFrame(dict(value))
        except ValueError:
            return pd.DataFrame([dict(value)])
    return pd.DataFrame(value)


def _value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def write_run_registry(
    registry_path: str | Path,
    *,
    run_id: str,
    run_dir: str | Path | None = None,
    artifact_paths: Mapping[str, Path] | None = None,
    manifest: Mapping[str, Any] | None = None,
    lineage: Any = None,
) -> Path:
    target = Path(registry_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(target) as connection:
        _ensure_registry_schema(connection)
        _upsert_run(
            connection,
            run_id=str(run_id),
            status="running",
            updated_at=now,
            run_dir=str(run_dir) if run_dir is not None else None,
            manifest_json=json.dumps(manifest, ensure_ascii=False, sort_keys=True, default=str) if manifest else None,
        )
        for artifact_name, artifact_path in (artifact_paths or {}).items():
            _upsert_artifact(connection, str(run_id), str(artifact_name), Path(artifact_path), now)
        _insert_metric_rows(connection, str(run_id), artifact_paths or {}, now)
        _insert_lineage_rows(connection, str(run_id), lineage, now)
        _upsert_run(
            connection,
            run_id=str(run_id),
            status="success",
            updated_at=now,
            completed_at=now,
            run_dir=str(run_dir) if run_dir is not None else None,
            manifest_json=json.dumps(manifest, ensure_ascii=False, sort_keys=True, default=str) if manifest else None,
        )
        connection.commit()
    return target


def mark_run_status(
    status: str,
    registry_path: str | Path | None,
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Path | None:
    if registry_path is None:
        return None
    target = Path(registry_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    failed_run_id = "default" if run_id is None else str(run_id)
    state_payload = dict(payload or {})
    fail_reason = str(state_payload.get("message", state_payload.get("error_type", ""))) if status == "failed" else None
    updated_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(target) as connection:
        _ensure_registry_schema(connection)
        _upsert_run(
            connection,
            run_id=failed_run_id,
            status=str(status),
            updated_at=updated_at,
            completed_at=updated_at if status in {"success", "failed"} else None,
            fail_reason=fail_reason,
            failure_state_json=json.dumps(state_payload, ensure_ascii=False, sort_keys=True, default=str) if state_payload else None,
        )
        connection.commit()
    return target


def mark_run_failed(
    failure_state: dict[str, Any],
    registry_path: str | Path | None,
    run_id: str | None = None,
) -> Path | None:
    return mark_run_status("failed", registry_path, run_id, failure_state)


def _upsert_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    status: str,
    updated_at: str,
    run_dir: str | None = None,
    completed_at: str | None = None,
    fail_reason: str | None = None,
    failure_state_json: str | None = None,
    manifest_json: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO runs(run_id, status, started_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, status, updated_at, updated_at),
    )
    connection.execute(
        """
        UPDATE runs
        SET status = ?,
            run_dir = COALESCE(?, run_dir),
            updated_at = ?,
            completed_at = COALESCE(?, completed_at),
            fail_reason = ?,
            failure_state_json = COALESCE(?, failure_state_json),
            manifest_json = COALESCE(?, manifest_json)
        WHERE run_id = ?
        """,
        (status, run_dir, updated_at, completed_at, fail_reason, failure_state_json, manifest_json, run_id),
    )


def _upsert_artifact(connection: sqlite3.Connection, run_id: str, artifact_name: str, artifact_path: Path, created_at: str) -> None:
    size_bytes = artifact_path.stat().st_size if artifact_path.is_file() else None
    connection.execute(
        """
        INSERT INTO artifacts(run_id, artifact_name, path, sha256, size_bytes, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, artifact_name) DO UPDATE SET
            path = excluded.path,
            sha256 = excluded.sha256,
            size_bytes = excluded.size_bytes,
            created_at = excluded.created_at
        """,
        (run_id, artifact_name, str(artifact_path), _file_sha256(artifact_path), size_bytes, created_at),
    )


def _insert_metric_rows(
    connection: sqlite3.Connection,
    run_id: str,
    artifact_paths: Mapping[str, Path],
    created_at: str,
) -> None:
    for artifact_name, artifact_path in artifact_paths.items():
        if artifact_name != "metrics" and not artifact_name.endswith("_metrics"):
            continue
        path = Path(artifact_path)
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        default_split = artifact_name.removesuffix("_metrics") if artifact_name.endswith("_metrics") else "all"
        for row in frame.to_dict(orient="records"):
            split = str(row.get("split") or default_split)
            for metric_name, value in row.items():
                if metric_name in {"split", "status"}:
                    continue
                try:
                    metric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if pd.isna(metric_value):
                    continue
                connection.execute(
                    """
                    INSERT INTO metrics(run_id, metric_name, metric_value, split, source_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, metric_name, split) DO UPDATE SET
                        metric_value = excluded.metric_value,
                        source_path = excluded.source_path,
                        created_at = excluded.created_at
                    """,
                    (run_id, str(metric_name), metric_value, split, str(path), created_at),
                )


def _insert_lineage_rows(connection: sqlite3.Connection, run_id: str, lineage: Any, created_at: str) -> None:
    rows: list[Mapping[str, Any]]
    if lineage is None:
        rows = []
    elif isinstance(lineage, Mapping):
        rows = [lineage]
    else:
        rows = [item for item in lineage if isinstance(item, Mapping)]
    for row in rows:
        parent_run_id = row.get("parent_run_id") or row.get("source_run_id") or row.get("run_id")
        if parent_run_id is None:
            continue
        relation = str(row.get("relation") or "parent")
        connection.execute(
            """
            INSERT INTO lineage(run_id, parent_run_id, relation, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, parent_run_id, relation) DO UPDATE SET
                created_at = excluded.created_at
            """,
            (run_id, str(parent_run_id), relation, created_at),
        )


def _ensure_runs_table(connection: sqlite3.Connection) -> None:
    _ensure_registry_schema(connection)


def _ensure_registry_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            status TEXT,
            run_dir TEXT,
            started_at TEXT,
            updated_at TEXT,
            completed_at TEXT,
            fail_reason TEXT,
            failure_state_json TEXT,
            manifest_json TEXT
        )
        """
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
    for name, definition in {
        "status": "TEXT",
        "run_dir": "TEXT",
        "started_at": "TEXT",
        "fail_reason": "TEXT",
        "failure_state_json": "TEXT",
        "updated_at": "TEXT",
        "completed_at": "TEXT",
        "manifest_json": "TEXT",
    }.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE runs ADD COLUMN {name} {definition}")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            run_id TEXT NOT NULL,
            artifact_name TEXT NOT NULL,
            path TEXT NOT NULL,
            sha256 TEXT,
            size_bytes INTEGER,
            created_at TEXT NOT NULL,
            PRIMARY KEY(run_id, artifact_name)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            run_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            split TEXT NOT NULL,
            source_path TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY(run_id, metric_name, split)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS lineage (
            run_id TEXT NOT NULL,
            parent_run_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(run_id, parent_run_id, relation)
        )
        """
    )
