from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.validate_otar_s0_protocol import RUN_CONFIGS, validate_protocol
from src.config import ConfigLoader, PROJECT_ROOT, config_hash
from src.experiments.paper_aggregate import aggregate_paper_results
from src.experiments.pipeline import expand_otar_formal_matrix, run_otar_formal_matrix
from src.experiments.registry import _merge_otar_hpo_grid_sources
from src.experiments.run_experiment import _create_run_dir
from src.utils.device import get_device
from src.utils.logger import save_json_atomic, save_yaml_atomic


PROTOCOL_ID = "otar_v2_s0_20260605"
DATA_CUTOFF_DATE = "2026-05-20"
DEFAULT_CONFIG_PATH = Path("configs/paper/otar_small8_pilot.yaml")
CANONICAL_FORMAL_MATRIX = Path("configs/paper/otar_formal_matrix.yaml")
COMPARISON_FILES = (
    "hpo_model_final_comparison.csv",
    "baseline_comparison.csv",
    "main_comparison.csv",
)
DAILY_RETURN_FILES = (
    "hpo_model_final_daily_returns.csv",
    "daily_returns.csv",
)


def run_otar_v2_full_pipeline(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    matrix_path: str | Path = CANONICAL_FORMAL_MATRIX,
    output_root: str | Path | None = None,
    run_name: str = "OTAR_v2_full",
    aggregate_output_dir: str | Path | None = None,
    formal_max_runs: int | None = None,
    dry_run: bool = False,
    skip_s0_validation: bool = False,
    skip_formal_runs: bool = False,
    skip_aggregate: bool = False,
    resume_completed_children: bool = False,
    protocol_id: str = PROTOCOL_ID,
    data_cutoff_date: str = DATA_CUTOFF_DATE,
    require_formal_manifest: bool = True,
    require_availability_mask_contract: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    started = _utc_now()
    resolved_config_path = _resolve_project_path(config_path)
    resolved_matrix_path = _resolve_project_path(matrix_path)
    config = _load_formal_base_config(
        resolved_config_path,
        output_root=output_root,
        run_name=run_name,
        device=device,
    )
    run_dir = _create_run_dir(config)
    logs_dir = run_dir / "logs"
    manifest_path = logs_dir / "otar_v2_full_manifest.json"

    _repair_stale_manifest(manifest_path, run_dir)
    if resume_completed_children:
        _cleanup_incomplete_children(run_dir)

    manifest: dict[str, Any] = {
        "status": "running",
        "config_path": str(resolved_config_path),
        "matrix_path": str(resolved_matrix_path),
        "parent_run_dir": str(run_dir),
        "aggregate_output_dir": str(_aggregate_dir(run_dir, aggregate_output_dir)),
        "protocol_id": protocol_id,
        "data_cutoff_date": data_cutoff_date,
        "s0_validation_status": "pending",
        "formal_status": "pending",
        "formal_run_count": 0,
        "formal_scope": _formal_scope(resolved_matrix_path, formal_max_runs, skip_s0_validation),
        "child_run_dirs": [],
        "aggregate_status": "pending",
        "aggregate_outputs": None,
        "resume_completed_children": bool(resume_completed_children),
        "dry_run": bool(dry_run),
        "started_at_utc": started,
        "finished_at_utc": None,
        "pid": os.getpid(),
    }
    save_json_atomic(manifest, manifest_path)
    _write_pid_file(run_dir)

    try:
        save_yaml_atomic(config, logs_dir / "config_snapshot.yaml")
        if not skip_s0_validation:
            _assert_canonical_matrix(resolved_matrix_path)
            _assert_frozen_config(resolved_config_path)
            s0_payload = validate_protocol(post_freeze=True)
            manifest["s0_validation_status"] = "success"
        else:
            s0_payload = {"status": "skipped"}
            manifest["s0_validation_status"] = "skipped"
        save_json_atomic(s0_payload, logs_dir / "s0_validation.json")

        planned_runs = _planned_child_runs(resolved_matrix_path, config, run_dir, formal_max_runs=formal_max_runs)
        save_json_atomic(planned_runs, logs_dir / "formal_matrix_plan.json")
        manifest["formal_run_count"] = int(planned_runs["selected_run_count"])

        if dry_run:
            manifest["status"] = "dry_run"
            manifest["formal_status"] = "dry_run"
            manifest["aggregate_status"] = "skipped"
            manifest["finished_at_utc"] = _utc_now()
            save_json_atomic(manifest, manifest_path)
            return manifest

        if skip_formal_runs:
            lineage = _load_existing_lineage(run_dir)
            manifest["formal_status"] = "skipped_existing_lineage"
        else:
            formal_result = run_otar_formal_matrix(
                config,
                matrix_path=resolved_matrix_path,
                run_dir=run_dir,
                max_runs=formal_max_runs,
                device=get_device(config["device"]),
                resume_completed=resume_completed_children,
            )
            lineage = list(formal_result.get("lineage", []))
            manifest["formal_status"] = str(formal_result.get("status", "unknown"))

        child_run_dirs = [] if skip_aggregate and not lineage else _child_run_dirs_from_lineage(lineage)
        manifest["child_run_dirs"] = [str(path) for path in child_run_dirs]
        if not skip_aggregate:
            _validate_child_artifacts(child_run_dirs, planned_runs)
            aggregate_dir = _aggregate_dir(run_dir, aggregate_output_dir)
            outputs = aggregate_paper_results(
                child_run_dirs,
                aggregate_dir,
                required_protocol_id=protocol_id,
                required_data_cutoff_date=data_cutoff_date,
                require_formal_manifest=require_formal_manifest,
                require_availability_mask_contract=require_availability_mask_contract,
            )
            _assert_nonempty_main_table(outputs)
            manifest["aggregate_status"] = "completed"
            manifest["aggregate_outputs"] = {name: str(path) for name, path in outputs.items()}
        else:
            manifest["aggregate_status"] = "skipped"
            manifest["aggregate_outputs"] = None

        manifest["status"] = "completed"
        manifest["finished_at_utc"] = _utc_now()
        save_json_atomic(manifest, manifest_path)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_stage"] = _failure_stage(manifest)
        manifest["failure_reason"] = str(exc)
        manifest["finished_at_utc"] = _utc_now()
        save_json_atomic(manifest, manifest_path)
        raise
    finally:
        _remove_pid_file(run_dir)


def _load_formal_base_config(
    config_path: Path,
    *,
    output_root: str | Path | None,
    run_name: str,
    device: str | None,
) -> dict[str, Any]:
    overrides = SimpleNamespace(output=None if output_root is None else str(output_root), run_name=run_name, device=device, seed=None)
    config = ConfigLoader.load(config_path, cli_overrides=overrides)
    config.setdefault("rankability", {})
    config["rankability"]["rankable_in_unified_table"] = True
    config["rankability"]["diagnostic_status"] = "formal"
    config.setdefault("experiment", {})
    config["experiment"]["type"] = "otar_formal_matrix"
    config["config_hash"] = config_hash(config)
    return config


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _assert_canonical_matrix(matrix_path: Path) -> None:
    canonical = _resolve_project_path(CANONICAL_FORMAL_MATRIX)
    if matrix_path != canonical:
        raise RuntimeError(f"ERR_OTAR_V2_FULL_UNFROZEN_MATRIX: {matrix_path}")


def _assert_frozen_config(config_path: Path) -> None:
    frozen = {_resolve_project_path(path) for path in RUN_CONFIGS}
    if config_path not in frozen:
        raise RuntimeError(f"ERR_OTAR_V2_FULL_UNFROZEN_CONFIG: {config_path}")


def _aggregate_dir(run_dir: Path, aggregate_output_dir: str | Path | None) -> Path:
    if aggregate_output_dir is None:
        return run_dir / "paper_tables"
    path = Path(aggregate_output_dir)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _formal_scope(matrix_path: Path, formal_max_runs: int | None, skip_s0_validation: bool) -> str:
    if formal_max_runs is not None:
        return "truncated"
    if skip_s0_validation and matrix_path != _resolve_project_path(CANONICAL_FORMAL_MATRIX):
        return "unfrozen_matrix_debug"
    if skip_s0_validation:
        return "s0_skipped_debug"
    return "full"


def _planned_child_runs(
    matrix_path: Path,
    base_config: Mapping[str, Any],
    run_dir: Path,
    *,
    formal_max_runs: int | None,
) -> dict[str, Any]:
    all_runs = expand_otar_formal_matrix(matrix_path, base_config)
    selected = all_runs if formal_max_runs is None else all_runs[: max(0, int(formal_max_runs))]
    child_rows: list[dict[str, Any]] = []
    for index, child_config in enumerate(selected, start=1):
        child = deepcopy(dict(child_config))
        run_name = str(child.get("output", {}).get("run_name") or f"OTAR_formal_{index:03d}")
        child.setdefault("output", {})
        child["output"]["run_name"] = run_name
        child = _merge_otar_hpo_grid_sources(child)
        child["config_hash"] = config_hash(child)
        child_rows.append(
            {
                "order": index,
                "child_run_name": run_name,
                "child_run_dir": str(run_dir / f"{index:03d}_{run_name}"),
                "child_config_hash": child["config_hash"],
                "ablation_id": _mapping(child.get("experiment")).get("ablation_id", ""),
                "universe": _mapping(child.get("protocol")).get("asset_universe_id", ""),
                "seed": _mapping(child.get("training")).get("seed", _mapping(child.get("reproducibility")).get("seed")),
                "model_name": _mapping(child.get("model")).get("name", ""),
            }
        )
    return {
        "status": "planned",
        "full_run_count": len(all_runs),
        "selected_run_count": len(selected),
        "formal_scope": "full" if formal_max_runs is None else "truncated",
        "ablation_counts": dict(Counter(str(row["ablation_id"]) for row in child_rows)),
        "universe_counts": dict(Counter(str(row["universe"]) for row in child_rows)),
        "model_counts": dict(Counter(str(row["model_name"]) for row in child_rows)),
        "child_runs": child_rows,
    }


def _load_existing_lineage(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "logs" / "otar_formal_lineage.json"
    if not path.exists():
        raise RuntimeError(f"ERR_OTAR_V2_FULL_LINEAGE_MISSING: {path}")
    payload = _read_json_mapping(path)
    lineage = payload.get("lineage")
    if not isinstance(lineage, list) or not lineage:
        raise RuntimeError(f"ERR_OTAR_V2_FULL_LINEAGE_EMPTY: {path}")
    return [dict(item) for item in lineage if isinstance(item, Mapping)]


def _child_run_dirs_from_lineage(lineage: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for item in lineage:
        if str(item.get("status", "")) != "completed":
            raise RuntimeError(f"ERR_OTAR_V2_FULL_CHILD_NOT_COMPLETED: {item.get('child_run_id')}")
        raw_path = item.get("run_dir")
        if not raw_path:
            raise RuntimeError(f"ERR_OTAR_V2_FULL_CHILD_RUN_DIR_MISSING: {item.get('child_run_id')}")
        paths.append(Path(str(raw_path)).expanduser().resolve())
    if not paths:
        raise RuntimeError("ERR_OTAR_V2_FULL_NO_CHILD_RUN_DIRS")
    return paths


def _validate_child_artifacts(child_run_dirs: Sequence[Path], planned_runs: Mapping[str, Any]) -> None:
    if not child_run_dirs:
        raise RuntimeError("ERR_OTAR_V2_FULL_NO_CHILD_RUN_DIRS")
    expected_rows = planned_runs.get("child_runs") if isinstance(planned_runs.get("child_runs"), list) else []
    expected_by_dir = {
        Path(str(row.get("child_run_dir"))).expanduser().resolve(): dict(row)
        for row in expected_rows
        if isinstance(row, Mapping) and row.get("child_run_dir")
    }
    actual_dirs = [Path(path).expanduser().resolve() for path in child_run_dirs]
    if set(actual_dirs) != set(expected_by_dir) or len(actual_dirs) != int(planned_runs.get("selected_run_count", len(actual_dirs))):
        raise RuntimeError("ERR_OTAR_V2_FULL_CHILD_LINEAGE_PLAN_MISMATCH")
    for run_dir in child_run_dirs:
        run_dir = Path(run_dir).expanduser().resolve()
        expected = expected_by_dir.get(run_dir)
        if expected is None:
            raise RuntimeError(f"ERR_OTAR_V2_FULL_CHILD_LINEAGE_PLAN_MISMATCH: {run_dir}")
        result_path = run_dir / "logs" / "experiment_result.json"
        manifest_path = run_dir / "logs" / "run_manifest.json"
        if not result_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"ERR_OTAR_V2_FULL_CHILD_ARTIFACT_INCOMPLETE: {run_dir}")
        result = _read_json_mapping(result_path)
        manifest = _read_json_mapping(manifest_path)
        if str(result.get("status", "")) != "completed" or str(manifest.get("status", "")) != "success":
            raise RuntimeError(f"ERR_OTAR_V2_FULL_CHILD_ARTIFACT_INCOMPLETE: {run_dir}")
        if manifest.get("rankable_in_unified_table") is not True or manifest.get("diagnostic_status") != "formal":
            raise RuntimeError(f"ERR_OTAR_V2_FULL_CHILD_NOT_FORMAL: {run_dir}")
        expected_hash = str(expected.get("child_config_hash", ""))
        actual_hash = str(manifest.get("config_hash") or manifest.get("child_config_hash") or "")
        if actual_hash != expected_hash:
            raise RuntimeError(f"ERR_OTAR_V2_FULL_CHILD_LINEAGE_PLAN_MISMATCH: {run_dir}")
        metrics_dir = run_dir / "metrics"
        if not any((metrics_dir / filename).exists() for filename in COMPARISON_FILES):
            raise RuntimeError(f"ERR_OTAR_V2_FULL_CHILD_ARTIFACT_INCOMPLETE: {run_dir}")
        if not any((metrics_dir / filename).exists() for filename in DAILY_RETURN_FILES):
            raise RuntimeError(f"ERR_OTAR_V2_FULL_CHILD_ARTIFACT_INCOMPLETE: {run_dir}")


def _assert_nonempty_main_table(outputs: Mapping[str, Path]) -> None:
    manifest_path = outputs.get("paper_aggregate_manifest")
    if manifest_path is None or not Path(manifest_path).exists():
        raise RuntimeError("ERR_OTAR_V2_FULL_AGGREGATE_MANIFEST_MISSING")
    manifest = _read_json_mapping(Path(manifest_path))
    row_counts = manifest.get("row_counts") if isinstance(manifest.get("row_counts"), Mapping) else {}
    if int(row_counts.get("paper_main_comparison", 0)) <= 0:
        raise RuntimeError("ERR_OTAR_V2_FULL_AGGREGATE_EMPTY_MAIN_TABLE")


def _failure_stage(manifest: Mapping[str, Any]) -> str:
    if manifest.get("s0_validation_status") in {"pending"}:
        return "s0_validation"
    if manifest.get("formal_status") in {"pending"}:
        return "formal_runs"
    if manifest.get("aggregate_status") in {"pending"}:
        return "aggregate"
    return "unknown"


def _pid_file_path(run_dir: Path) -> Path:
    return run_dir / "logs" / "otar_v2_full.pid"


def _write_pid_file(run_dir: Path) -> None:
    pid_path = _pid_file_path(run_dir)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")


def _remove_pid_file(run_dir: Path) -> None:
    pid_path = _pid_file_path(run_dir)
    try:
        pid_path.unlink(missing_ok=True)
    except OSError:
        pass


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid_from_file(run_dir: Path) -> int | None:
    pid_path = _pid_file_path(run_dir)
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _repair_stale_manifest(manifest_path: Path, run_dir: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = _read_json_mapping(manifest_path)
    except RuntimeError:
        return None
    if manifest.get("status") != "running":
        return None
    pid = _read_pid_from_file(run_dir)
    if pid is not None and _is_pid_alive(pid):
        return None
    manifest["status"] = "interrupted"
    manifest["interrupted_at_utc"] = _utc_now()
    manifest["interrupted_pid"] = pid
    manifest["failure_stage"] = _failure_stage(manifest)
    save_json_atomic(manifest, manifest_path)
    return manifest


def _cleanup_incomplete_child_run(child_dir: Path) -> bool:
    result_path = child_dir / "logs" / "experiment_result.json"
    manifest_path = child_dir / "logs" / "run_manifest.json"
    if result_path.exists() and manifest_path.exists():
        try:
            result = _read_json_mapping(result_path)
            manifest = _read_json_mapping(manifest_path)
            if str(result.get("status", "")) == "completed" and str(manifest.get("status", "")) == "success":
                return False
        except RuntimeError:
            pass
    trial_dirs = sorted(child_dir.glob("trial_*"))
    for trial_dir in trial_dirs:
        if trial_dir.is_dir():
            shutil.rmtree(trial_dir, ignore_errors=True)
    for subdir_name in ("figures", "metrics", "final_test_best", "final_test_median", "final_test_worst"):
        subdir = child_dir / subdir_name
        if subdir.exists() and subdir.is_dir():
            shutil.rmtree(subdir, ignore_errors=True)
    return True


def _cleanup_incomplete_children(run_dir: Path) -> int:
    cleaned = 0
    for child_dir in sorted(run_dir.iterdir()):
        if not child_dir.is_dir():
            continue
        if not child_dir.name[:3].isdigit():
            continue
        if _cleanup_incomplete_child_run(child_dir):
            cleaned += 1
    return cleaned


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ERR_OTAR_V2_FULL_INVALID_JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"ERR_OTAR_V2_FULL_INVALID_JSON: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full OTAR V2 formal experiment pipeline.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--formal-matrix", default=str(CANONICAL_FORMAL_MATRIX))
    parser.add_argument("--output")
    parser.add_argument("--run-name", default="OTAR_v2_full")
    parser.add_argument("--aggregate-output-dir")
    parser.add_argument("--formal-max-runs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-s0-validation", action="store_true")
    parser.add_argument("--skip-formal-runs", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--resume-completed-children", action="store_true")
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    parser.add_argument("--data-cutoff-date", default=DATA_CUTOFF_DATE)
    parser.add_argument("--no-require-formal-manifest", action="store_true")
    parser.add_argument("--no-require-availability-mask-contract", action="store_true")
    parser.add_argument("--device")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return run_otar_v2_full_pipeline(
        config_path=args.config,
        matrix_path=args.formal_matrix,
        output_root=args.output,
        run_name=args.run_name,
        aggregate_output_dir=args.aggregate_output_dir,
        formal_max_runs=args.formal_max_runs,
        dry_run=bool(args.dry_run),
        skip_s0_validation=bool(args.skip_s0_validation),
        skip_formal_runs=bool(args.skip_formal_runs),
        skip_aggregate=bool(args.skip_aggregate),
        resume_completed_children=bool(args.resume_completed_children),
        protocol_id=args.protocol_id,
        data_cutoff_date=args.data_cutoff_date,
        require_formal_manifest=not bool(args.no_require_formal_manifest),
        require_availability_mask_contract=not bool(args.no_require_availability_mask_contract),
        device=args.device,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CANONICAL_FORMAL_MATRIX",
    "DATA_CUTOFF_DATE",
    "DEFAULT_CONFIG_PATH",
    "PROTOCOL_ID",
    "run_otar_v2_full_pipeline",
]
