from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GENERATED_ROOTS = ("outputs", "results", "logs")
SMALL_CLEAN_NAMES = {".DS_Store", ".pytest_cache", "__pycache__"}
SMALL_CLEAN_PREFIXES = ("._", ".__")
SMALL_CLEAN_SUFFIXES = (".tmp",)


def main() -> int:
    args = _parse_args()
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_path = log_dir / "progress.jsonl"
    cleanup_path = log_dir / "cleanup.jsonl"
    pid = int(Path(args.pid_file).read_text(encoding="utf-8").strip())

    while True:
        alive = _process_alive(pid)
        progress = _progress_snapshot(project_root, run_dir, pid, alive)
        _append_json(progress_path, progress)
        if progress["disk_free_gb"] < float(args.min_free_gb):
            cleanup = _cleanup_generated_files(
                project_root=project_root,
                current_run_dir=run_dir,
                min_age_hours=float(args.min_age_hours),
                target_free_gb=float(args.target_free_gb),
            )
            _append_json(cleanup_path, cleanup)
        if not alive:
            break
        time.sleep(float(args.interval_seconds))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor OTAR V2 full experiment progress and disk space.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--interval-seconds", type=float, default=1800)
    parser.add_argument("--min-free-gb", type=float, default=30)
    parser.add_argument("--target-free-gb", type=float, default=40)
    parser.add_argument("--min-age-hours", type=float, default=24)
    return parser.parse_args()


def _progress_snapshot(project_root: Path, run_dir: Path, pid: int, alive: bool) -> dict[str, Any]:
    manifest = _read_json(run_dir / "logs" / "otar_v2_full_manifest.json")
    plan = _read_json(run_dir / "logs" / "formal_matrix_plan.json")
    lineage = _read_json(run_dir / "logs" / "otar_formal_lineage.json")
    lineage_items = lineage.get("lineage") if isinstance(lineage.get("lineage"), list) else []
    completed = [item for item in lineage_items if isinstance(item, dict) and item.get("status") == "completed"]
    artifact_completed = _completed_child_artifacts(run_dir)
    last_child = completed[-1] if completed else None
    last_artifact_child = artifact_completed[-1].name if artifact_completed else None
    usage = shutil.disk_usage(project_root)
    return {
        "timestamp_utc": _utc_now(),
        "pid": pid,
        "process_alive": alive,
        "run_dir": str(run_dir),
        "status": manifest.get("status", "unknown"),
        "formal_status": manifest.get("formal_status", "unknown"),
        "aggregate_status": manifest.get("aggregate_status", "unknown"),
        "planned_run_count": plan.get("selected_run_count"),
        "completed_child_count": max(len(completed), len(artifact_completed)),
        "lineage_child_count": len(lineage_items),
        "last_completed_child": (
            None if last_child is None and last_artifact_child is None
            else last_child.get("child_run_id") if last_child is not None
            else last_artifact_child
        ),
        "disk_free_gb": round(usage.free / 1024**3, 3),
        "disk_total_gb": round(usage.total / 1024**3, 3),
    }


def _completed_child_artifacts(run_dir: Path) -> list[Path]:
    completed: list[Path] = []
    for child in sorted(run_dir.glob("[0-9][0-9][0-9]_*")):
        result = _read_json(child / "logs" / "experiment_result.json")
        manifest = _read_json(child / "logs" / "run_manifest.json")
        if result.get("status") == "completed" and manifest.get("status") == "success":
            completed.append(child)
    return completed


def _cleanup_generated_files(
    *,
    project_root: Path,
    current_run_dir: Path,
    min_age_hours: float,
    target_free_gb: float,
) -> dict[str, Any]:
    before = shutil.disk_usage(project_root).free / 1024**3
    deleted: list[dict[str, Any]] = []
    cutoff = time.time() - min_age_hours * 3600

    for path in _small_cleanup_paths(project_root):
        if _overlaps(path, current_run_dir):
            continue
        if _mtime(path) > cutoff:
            continue
        deleted.append(_delete_path(path, reason="small_generated"))

    for path in _large_generated_candidates(project_root, current_run_dir, cutoff):
        if shutil.disk_usage(project_root).free / 1024**3 >= target_free_gb:
            break
        deleted.append(_delete_path(path, reason="large_generated"))

    after = shutil.disk_usage(project_root).free / 1024**3
    return {
        "timestamp_utc": _utc_now(),
        "free_gb_before": round(before, 3),
        "free_gb_after": round(after, 3),
        "deleted": deleted,
    }


def _small_cleanup_paths(project_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root_name in GENERATED_ROOTS:
        root = project_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            name = path.name
            if (
                name in SMALL_CLEAN_NAMES
                or any(name.startswith(prefix) for prefix in SMALL_CLEAN_PREFIXES)
                or any(name.endswith(suffix) for suffix in SMALL_CLEAN_SUFFIXES)
            ):
                paths.append(path)
    return paths


def _large_generated_candidates(project_root: Path, current_run_dir: Path, cutoff: float) -> list[Path]:
    candidates: list[tuple[int, Path]] = []
    for root_name in GENERATED_ROOTS:
        root = project_root / root_name
        if not root.exists():
            continue
        for path in root.iterdir():
            if _overlaps(path.resolve(), current_run_dir):
                continue
            if _mtime(path) > cutoff:
                continue
            candidates.append((_path_size_bytes(path), path))
    return [path for _, path in sorted(candidates, reverse=True)]


def _delete_path(path: Path, *, reason: str) -> dict[str, Any]:
    size = _path_size_bytes(path)
    payload = {"path": str(path), "size_gb": round(size / 1024**3, 3), "reason": reason}
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        payload["status"] = "deleted"
    except OSError as exc:
        payload["status"] = "failed"
        payload["error"] = str(exc)
    return payload


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _overlaps(path: Path, current_run_dir: Path) -> bool:
    path = path.resolve()
    current = current_run_dir.resolve()
    return path == current or path in current.parents or current in path.parents


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return time.time()


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _append_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
