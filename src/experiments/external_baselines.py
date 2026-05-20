from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.external_pgportfolio import external_pgportfolio_summary
from src.config import ConfigError, PROJECT_ROOT, assert_path_allowed
from src.utils.metrics import calculate_performance_metrics


PGPORTFOLIO_REQUIRED_IMPORT_COLUMNS = ("date", "nav", "net_return")
PGPORTFOLIO_OPTIONAL_IMPORT_COLUMNS = (
    "gross_return",
    "transaction_cost",
    "total_transaction_cost",
    "proportional_cost",
    "fixed_cost",
    "slippage_cost",
    "market_impact_cost",
)
CASH_WEIGHT_COLUMNS = {"cash", "cash_weight"}
EXTERNAL_PGPORTFOLIO_REPO_ROOT = PROJECT_ROOT / "external" / "PGPortfolio"


def run_external_pgportfolio_baseline(
    config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    segment: str = "test",
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    del segment
    external_cfg = external_pgportfolio_config(config)
    if not bool(external_cfg.get("enabled", False)):
        return _payload("skipped_external_disabled")

    command = external_cfg.get("command")
    import_path = external_cfg.get("import_results_csv") or external_cfg.get("results_csv_path")
    resolved_import_path: Path | None = None
    if import_path not in (None, ""):
        try:
            resolved_import_path = assert_path_allowed(
                import_path,
                [PROJECT_ROOT],
                "baselines.external_pgportfolio.import_results_csv",
            )
        except ConfigError:
            return _payload(
                "skipped_out_of_scope",
                out_of_scope_edge='{"kind":"out_of_scope","ref":"external_pgportfolio_import_results"}',
                import_results_csv=str(import_path),
            )
        if not command:
            return _import_payload(resolved_import_path, artifacts, config)

    repo_path = external_cfg.get("repo_path")
    if repo_path in (None, ""):
        return _payload("skipped_external_dependency_missing", fail_reason="missing_repo_path")

    try:
        resolved_repo = assert_path_allowed(
            repo_path,
            [EXTERNAL_PGPORTFOLIO_REPO_ROOT],
            "baselines.external_pgportfolio.repo_path",
        )
    except ConfigError:
        return _payload(
            "skipped_out_of_scope",
            out_of_scope_edge='{"kind":"out_of_scope","ref":"external_pgportfolio_repo"}',
            repo_path=str(repo_path),
        )

    if not resolved_repo.exists():
        return _payload(
            "skipped_external_dependency_missing",
            fail_reason="repo_path_not_found",
            repo_path=str(resolved_repo),
        )

    try:
        work_dir = assert_path_allowed(
            Path(run_dir) / "external_pgportfolio" if run_dir is not None else PROJECT_ROOT / "results" / "external_pgportfolio",
            [PROJECT_ROOT],
            "baselines.external_pgportfolio.work_dir",
        )
    except ConfigError:
        return _payload(
            "skipped_out_of_scope",
            out_of_scope_edge='{"kind":"out_of_scope","ref":"external_pgportfolio_work_dir"}',
            run_dir=str(run_dir),
        )
    if command:
        work_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            _command_args(command),
            cwd=resolved_repo,
            capture_output=True,
            text=True,
            timeout=int(external_cfg.get("timeout_seconds", 86400)),
            check=False,
        )
        (work_dir / "stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
        (work_dir / "stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            return _payload(
                "failed",
                fail_reason="external_process_nonzero",
                returncode=int(completed.returncode),
                repo_path=str(resolved_repo),
                external_work_dir=str(work_dir),
                command=" ".join(_command_args(command)),
            )
        if resolved_import_path is not None and resolved_import_path.exists():
            return _import_payload(
                resolved_import_path,
                artifacts,
                config,
                repo_path=str(resolved_repo),
                external_work_dir=str(work_dir),
                command=" ".join(_command_args(command)),
            )
        return _payload(
            "skipped_external_result_missing",
            fail_reason="external_process_completed_without_import_results_csv",
            repo_path=str(resolved_repo),
            external_work_dir=str(work_dir),
        )
    return _payload(
        "skipped_external_runner_not_implemented",
        fail_reason="external_subprocess_and_import_not_implemented",
        repo_path=str(resolved_repo),
        external_work_dir=str(work_dir),
    )


def validate_external_pgportfolio_import(
    frame: pd.DataFrame,
    test_asset_universe: list[str] | tuple[str, ...],
    *,
    test_dates: Any | None = None,
    availability_mask: pd.DataFrame | None = None,
    weight_sum_tolerance: float = 1.0e-6,
    allow_extra_asset_weights: bool = False,
    allow_cash_weight: bool = False,
    availability_weight_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    prepared = _prepare_external_pgportfolio_import_frame(
        frame,
        test_asset_universe,
        test_dates=test_dates,
        availability_mask=availability_mask,
        weight_sum_tolerance=weight_sum_tolerance,
        allow_extra_asset_weights=allow_extra_asset_weights,
        allow_cash_weight=allow_cash_weight,
        availability_weight_tolerance=availability_weight_tolerance,
    )
    if prepared["status"] != "completed":
        return {key: value for key, value in prepared.items() if key != "frame"}
    normalized = prepared["frame"]
    return {
        "status": "completed",
        "asset_count": len([str(asset) for asset in test_asset_universe]),
        "row_count": int(len(normalized)),
        "weight_sum_tolerance": float(weight_sum_tolerance),
    }


def _prepare_external_pgportfolio_import_frame(
    frame: pd.DataFrame,
    test_asset_universe: list[str] | tuple[str, ...],
    *,
    test_dates: Any | None = None,
    availability_mask: pd.DataFrame | None = None,
    weight_sum_tolerance: float = 1.0e-6,
    allow_extra_asset_weights: bool = False,
    allow_cash_weight: bool = False,
    availability_weight_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    if frame.empty:
        return {"status": "failed", "fail_reason": "empty_result_csv"}
    missing_required = [column for column in PGPORTFOLIO_REQUIRED_IMPORT_COLUMNS if column not in frame.columns]
    if missing_required:
        return {"status": "failed_missing_required_columns", "missing_columns": missing_required}
    normalized = frame.copy()
    try:
        normalized["date"] = pd.to_datetime(normalized["date"])
    except Exception:
        return {"status": "failed_date_parse"}
    if normalized["date"].isna().any():
        return {"status": "failed_date_parse"}
    if normalized["date"].duplicated().any():
        duplicates = normalized.loc[normalized["date"].duplicated(), "date"].astype(str).tolist()
        return {"status": "failed_duplicate_dates", "duplicate_dates": duplicates[:10]}
    nav_payload = _numeric_required_series(
        normalized,
        "nav",
        "failed_nav_numeric",
        "failed_nav_nan",
        "failed_nav_non_finite",
    )
    if nav_payload["status"] != "completed":
        return nav_payload
    net_payload = _numeric_required_series(
        normalized,
        "net_return",
        "failed_net_return_numeric",
        "failed_net_return_nan",
        "failed_net_return_non_finite",
    )
    if net_payload["status"] != "completed":
        return net_payload
    normalized["nav"] = nav_payload["series"]
    normalized["net_return"] = net_payload["series"]
    if test_dates is not None:
        allowed_dates = set(pd.DatetimeIndex(pd.to_datetime(list(test_dates))))
        outside = [str(item) for item in normalized["date"] if pd.Timestamp(item) not in allowed_dates]
        if outside:
            return {"status": "failed_date_outside_test_split", "outside_dates": outside[:10]}
        imported_dates = set(pd.DatetimeIndex(pd.to_datetime(normalized["date"])))
        missing_dates = sorted(str(item) for item in allowed_dates - imported_dates)
        if missing_dates:
            return {"status": "failed_missing_test_dates", "missing_dates": missing_dates[:10]}

    assets = [str(asset) for asset in test_asset_universe]
    if not assets:
        return {"status": "failed_missing_test_asset_universe"}
    weights_payload = _external_weight_frame(normalized, assets)
    if weights_payload["status"] != "completed":
        return weights_payload
    weights = weights_payload["weights"]
    for asset in assets:
        normalized[asset] = weights[asset].to_numpy(dtype=float, copy=True)

    cash_columns = [str(column) for column in normalized.columns if str(column).lower() in CASH_WEIGHT_COLUMNS]
    if cash_columns and not allow_cash_weight:
        return {"status": "failed_cash_weight_column", "cash_weight_columns": cash_columns}

    metadata_columns = set(PGPORTFOLIO_REQUIRED_IMPORT_COLUMNS) | set(PGPORTFOLIO_OPTIONAL_IMPORT_COLUMNS)
    metadata_columns |= {"weights", "cost_availability"}
    extra_columns = [
        str(column)
        for column in normalized.columns
        if str(column) not in metadata_columns
        and str(column) not in assets
        and str(column).lower() not in CASH_WEIGHT_COLUMNS
    ]
    if extra_columns and not allow_extra_asset_weights:
        return {"status": "failed_extra_asset_weights", "extra_asset_weights": extra_columns}

    if weights.isna().any().any():
        return {"status": "failed_weight_nan"}
    if (weights < 0.0).any().any():
        return {"status": "failed_negative_weight"}
    weight_sums = weights.sum(axis=1)
    weight_sum_error = np.abs(weight_sums.to_numpy(dtype=float) - 1.0)
    if (weight_sum_error > float(weight_sum_tolerance)).any():
        return {"status": "failed_weight_sum_tolerance", "max_abs_sum_error": float(np.max(weight_sum_error))}

    try:
        availability = _availability_for_external_dates(availability_mask, normalized["date"], assets)
    except ValueError as exc:
        return {"status": "failed_availability_schema", "fail_reason": str(exc)}
    if availability is not None:
        unavailable_nonzero = (~availability) & (weights.to_numpy(dtype=float) > float(availability_weight_tolerance))
        if unavailable_nonzero.any():
            row_index, col_index = np.argwhere(unavailable_nonzero)[0]
            return {
                "status": "failed_unavailable_asset_nonzero_weight",
                "date": str(normalized["date"].iloc[int(row_index)]),
                "asset": assets[int(col_index)],
                "weight": float(weights.iloc[int(row_index), int(col_index)]),
            }

    return {
        "status": "completed",
        "asset_count": len(assets),
        "row_count": int(len(normalized)),
        "weight_sum_tolerance": float(weight_sum_tolerance),
        "frame": normalized,
    }


def import_external_pgportfolio_outputs(
    frame: pd.DataFrame,
    test_asset_universe: list[str] | tuple[str, ...],
    *,
    config: Mapping[str, Any] | None = None,
    test_dates: Any | None = None,
    availability_mask: pd.DataFrame | None = None,
) -> dict[str, Any]:
    prepared = _prepare_external_pgportfolio_import_frame(
        frame,
        test_asset_universe,
        test_dates=test_dates,
        availability_mask=availability_mask,
    )
    validation = {key: value for key, value in prepared.items() if key != "frame"}
    if validation["status"] != "completed":
        return {"status": validation["status"], "validation": validation}
    frame = prepared["frame"]

    assets = [str(asset) for asset in test_asset_universe]
    cfg = config if isinstance(config, Mapping) else {}
    seed = int(_mapping(_mapping(cfg).get("reproducibility")).get("seed", 0))
    fold_id = str(cfg.get("fold_id", "external"))
    dates = pd.to_datetime(frame["date"])
    net_return = pd.to_numeric(frame["net_return"], errors="coerce")
    nav = pd.to_numeric(frame["nav"], errors="coerce")
    model_name = "pgportfolio_original_external"
    base = {
        "date": dates,
        "decision_date": dates,
        "execution_date": dates,
        "execution_price_type": "external",
        "next_valuation_date": dates,
        "split": "test",
        "seed": seed,
        "fold_id": fold_id,
        "model_name": model_name,
    }
    daily_returns = pd.DataFrame(
        {
            **base,
            "pre_execution_return": np.nan,
            "post_execution_return": np.nan,
            "gross_return": net_return,
            "transaction_cost": np.nan,
            "transaction_cost_on_initial_nav": np.nan,
            "net_return": net_return,
            "portfolio_log_return": np.log1p(net_return),
            "nav": nav,
        }
    )
    daily_weights = frame.loc[:, ["date", *assets]].copy()
    daily_weights["date"] = dates
    daily_weights = daily_weights.melt(id_vars="date", var_name="asset_id", value_name="weight")
    daily_weights["split"] = "test"
    daily_weights["seed"] = seed
    daily_weights["fold_id"] = fold_id
    daily_weights["model_name"] = model_name
    daily_weights = daily_weights.loc[:, ["date", "split", "seed", "fold_id", "model_name", "asset_id", "weight"]]

    daily_turnover = pd.DataFrame(
        {
            **base,
            "turnover": np.nan,
            "rebalance_action": pd.NA,
            "rebalance_intensity": np.nan,
            "average_holding_period": np.nan,
        }
    )
    daily_rebalance = pd.DataFrame(
        {
            **base,
            "rebalance_action": pd.NA,
            "rebalance_intensity": np.nan,
            "estimated_turnover": np.nan,
            "realized_turnover": np.nan,
            "turnover": np.nan,
            "estimated_cost": np.nan,
            "realized_cost": np.nan,
            "q_hold": np.nan,
            "q_rebalance": np.nan,
            "q_gap": np.nan,
        }
    )
    daily_costs = pd.DataFrame(
        {
            **base,
            "proportional_cost": np.nan,
            "fixed_cost": np.nan,
            "slippage_cost": np.nan,
            "market_impact_cost": np.nan,
            "total_transaction_cost": np.nan,
            "estimated_cost": np.nan,
            "realized_cost": np.nan,
            "turnover": np.nan,
        }
    )
    evaluation = _mapping(cfg.get("evaluation"))
    metrics = calculate_performance_metrics(
        daily_returns,
        daily_turnover,
        daily_costs,
        annualization_factor=int(evaluation.get("annualization_factor", 252)),
        risk_free_rate_annual=float(evaluation.get("risk_free_rate_annual", 0.0)),
    )
    return {
        "status": "completed",
        "metrics": metrics,
        "daily_returns": daily_returns,
        "daily_weights": daily_weights,
        "daily_turnover": daily_turnover,
        "daily_rebalance": daily_rebalance,
        "daily_costs": daily_costs,
        "validation": validation,
    }


def external_pgportfolio_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    baselines = config.get("baselines")
    if not isinstance(baselines, Mapping):
        return {}
    value = baselines.get("external_pgportfolio")
    return value if isinstance(value, Mapping) else {}


def _artifact_asset_order(artifacts: Mapping[str, Any]) -> list[str]:
    dataset = artifacts.get("dataset") if isinstance(artifacts, Mapping) else None
    manifest = getattr(dataset, "data_manifest", {})
    order = manifest.get("canonical_asset_order") if isinstance(manifest, Mapping) else None
    if isinstance(order, list) and order:
        return [str(item) for item in order]
    asset_universe = getattr(dataset, "asset_universe", None)
    if isinstance(asset_universe, pd.DataFrame):
        for column in ("ts_code", "asset_id", "asset"):
            if column in asset_universe.columns:
                return [str(item) for item in asset_universe[column].dropna().tolist()]
    availability = getattr(dataset, "availability_mask", None)
    if isinstance(availability, pd.DataFrame):
        return [str(item) for item in availability.columns.tolist()]
    return []


def _artifact_test_dates(artifacts: Mapping[str, Any]) -> pd.DatetimeIndex | None:
    split = artifacts.get("split") if isinstance(artifacts, Mapping) else None
    dates = getattr(split, "test_dates", None)
    if dates is None:
        return None
    return pd.DatetimeIndex(pd.to_datetime(list(dates)))


def _artifact_availability_mask(artifacts: Mapping[str, Any]) -> pd.DataFrame | None:
    dataset = artifacts.get("dataset") if isinstance(artifacts, Mapping) else None
    availability = getattr(dataset, "availability_mask", None)
    return availability if isinstance(availability, pd.DataFrame) else None


def _external_weight_frame(frame: pd.DataFrame, assets: list[str]) -> dict[str, Any]:
    missing_assets = [asset for asset in assets if asset not in frame.columns]
    if not missing_assets:
        return {"status": "completed", "weights": frame.loc[:, assets].apply(pd.to_numeric, errors="coerce")}
    if "weights" not in frame.columns:
        return {"status": "failed_missing_asset_weights", "missing_asset_weights": missing_assets}
    rows = []
    for value in frame["weights"]:
        parsed = _parse_weights_cell(value)
        if parsed is None:
            return {"status": "failed_weights_column_parse"}
        if isinstance(parsed, Mapping):
            missing = [asset for asset in assets if asset not in parsed]
            extra = [str(asset) for asset in parsed if str(asset) not in assets]
            if missing:
                return {"status": "failed_missing_asset_weights", "missing_asset_weights": missing}
            if extra:
                return {"status": "failed_extra_asset_weights", "extra_asset_weights": extra}
            rows.append([parsed[asset] for asset in assets])
        else:
            values = list(parsed)
            if len(values) != len(assets):
                return {"status": "failed_weights_column_length", "expected": len(assets), "actual": len(values)}
            rows.append(values)
    weights = pd.DataFrame(rows, columns=assets, index=frame.index).apply(pd.to_numeric, errors="coerce")
    return {"status": "completed", "weights": weights}


def _numeric_required_series(
    frame: pd.DataFrame,
    column: str,
    numeric_status: str,
    nan_status: str,
    non_finite_status: str,
) -> dict[str, Any]:
    raw = frame[column]
    converted = pd.to_numeric(raw, errors="coerce")
    raw_missing = raw.isna()
    coerced_missing = converted.isna()
    if (coerced_missing & ~raw_missing).any():
        return {"status": numeric_status, "column": column}
    if coerced_missing.any():
        return {"status": nan_status, "column": column}
    converted = converted.astype(float)
    if not np.isfinite(converted.to_numpy(dtype=float)).all():
        return {"status": non_finite_status, "column": column}
    return {"status": "completed", "series": converted}


def _parse_weights_cell(value: Any) -> Mapping[str, Any] | list[Any] | None:
    if isinstance(value, Mapping):
        return {str(key): weight for key, weight in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, Mapping):
        return {str(key): weight for key, weight in parsed.items()}
    if isinstance(parsed, list):
        return parsed
    return None


def _availability_for_external_dates(
    availability_mask: pd.DataFrame | None,
    dates: Any,
    assets: list[str],
) -> np.ndarray | None:
    if availability_mask is None:
        return None
    availability = availability_mask.copy()
    availability.index = pd.DatetimeIndex(pd.to_datetime(availability.index))
    missing_assets = [asset for asset in assets if asset not in availability.columns]
    if missing_assets:
        raise ValueError(f"ERR_EXTERNAL_AVAILABILITY_SCHEMA: missing_assets={missing_assets}")
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    missing_dates = [str(date) for date in date_index if pd.Timestamp(date) not in availability.index]
    if missing_dates:
        raise ValueError(f"ERR_EXTERNAL_AVAILABILITY_SCHEMA: missing_dates={missing_dates[:10]}")
    return availability.reindex(index=date_index, columns=assets).to_numpy(dtype=bool, copy=True)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _command_args(command: Any) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    if isinstance(command, (list, tuple)):
        return [str(item) for item in command]
    raise TypeError("ERR_EXTERNAL_COMMAND_TYPE")


def _payload(status: str, **summary_overrides: Any) -> dict[str, Any]:
    summary = external_pgportfolio_summary(status, **summary_overrides)
    return {
        "status": status,
        "baseline_training_summary": pd.DataFrame([summary]),
        "baseline_training_history": pd.DataFrame(),
    }


def _import_payload(
    path: Path,
    artifacts: Mapping[str, Any],
    config: Mapping[str, Any],
    **summary_overrides: Any,
) -> dict[str, Any]:
    frame = pd.read_csv(path)
    assets = _artifact_asset_order(artifacts)
    try:
        import_payload = import_external_pgportfolio_outputs(
            frame,
            assets,
            config=config,
            test_dates=_artifact_test_dates(artifacts),
            availability_mask=_artifact_availability_mask(artifacts),
        )
    except ValueError as exc:
        return _payload("failed_availability_schema", fail_reason=str(exc))
    if import_payload["status"] != "completed":
        return _payload(import_payload["status"], **import_payload.get("validation", {}))
    import_payload["baseline_training_summary"] = pd.DataFrame(
        [external_pgportfolio_summary("completed", import_results_csv=str(path), **summary_overrides)]
    )
    import_payload["baseline_training_history"] = pd.DataFrame()
    return import_payload


__all__ = [
    "external_pgportfolio_config",
    "import_external_pgportfolio_outputs",
    "run_external_pgportfolio_baseline",
    "validate_external_pgportfolio_import",
]
