from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT
from src.utils.logger import save_json_atomic


LEDGER_COLUMNS = (
    "phase",
    "run_name",
    "config_path",
    "run_dir",
    "seed",
    "model_scope",
    "data_mode",
    "protocol_id",
    "diagnostic_status",
    "rankable_in_unified_table",
    "started_at",
    "finished_at",
    "status",
    "artifact_paths",
    "blocking_reason",
)


def build_experiment_run_ledger(
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    protocol_id: str = "core13_v2_full_reset_20260522",
    output_name_stem: str = "experiment_run_ledger",
) -> dict[str, Path]:
    runs = [_scoped_path(path) for path in run_dirs]
    if not runs:
        raise ValueError("ERR_RUN_LEDGER_NO_RUN_DIRS")
    target = _scoped_output_dir(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = [_ledger_row(run_dir) for run_dir in runs]
    frame = pd.DataFrame(rows)
    for column in LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.loc[:, LEDGER_COLUMNS]
    csv_path = target / f"{output_name_stem}.csv"
    json_path = target / f"{output_name_stem}.json"
    frame.to_csv(csv_path, index=False)
    save_json_atomic(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_id": protocol_id,
            "row_count": int(frame.shape[0]),
            "columns": list(LEDGER_COLUMNS),
            "rows": frame.where(pd.notna(frame), None).to_dict(orient="records"),
        },
        json_path,
    )
    return {"csv": csv_path, "json": json_path}


def discover_run_dirs(results_root: str | Path) -> list[Path]:
    root = _scoped_path(results_root)
    if not root.exists():
        return []
    runs = {path.parent.parent for path in root.rglob("logs/run_manifest.json")}
    return sorted(runs)


def _ledger_row(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "logs" / "run_manifest.json")
    experiment_result = _read_json(run_dir / "logs" / "experiment_result.json")
    run_name = _clean(manifest.get("run_name")) or _clean(manifest.get("run_id")) or run_dir.name
    status = _clean(manifest.get("status")) or _clean(experiment_result.get("status")) or "missing_manifest"
    artifacts = _artifact_paths(run_dir)
    return {
        "phase": _phase_label(run_name, manifest),
        "run_name": run_name,
        "config_path": _clean(manifest.get("config_path")),
        "run_dir": str(run_dir),
        "seed": manifest.get("seed"),
        "model_scope": json.dumps(_model_scope(run_dir, manifest, experiment_result), ensure_ascii=False),
        "data_mode": _clean(manifest.get("data_mode")),
        "protocol_id": _clean(manifest.get("protocol_id")),
        "diagnostic_status": _clean(manifest.get("diagnostic_status")),
        "rankable_in_unified_table": manifest.get("rankable_in_unified_table"),
        "started_at": _clean(manifest.get("created_at")) or _clean(manifest.get("timestamp")),
        "finished_at": _clean(manifest.get("finished_at")) or _clean(manifest.get("timestamp")),
        "status": status,
        "artifact_paths": json.dumps([str(path) for path in artifacts], ensure_ascii=False),
        "blocking_reason": _blocking_reason(run_dir, manifest, experiment_result, artifacts),
    }


def _phase_label(run_name: str, manifest: Mapping[str, Any]) -> str:
    text = " ".join(
        _clean(value).lower()
        for value in (run_name, manifest.get("experiment_type"), manifest.get("config_path"))
    )
    ordered = (
        ("p-2", ("protocol_reset", "p-2")),
        ("p-1", ("data_freeze", "p-1")),
        ("p0", ("p0",)),
        ("p9", ("p9", "related_work")),
        ("p1", ("p1", "baseline_comparison", "hpo_final")),
        ("p2", ("p2", "input_matrix", "pca")),
        ("p3", ("p3", "ablation", "without_dqn", "without_auxiliary", "mlp_encoder", "kernel_size")),
        ("p4", ("p4", "reward")),
        ("p5", ("p5", "cost", "rebalance")),
        ("p6", ("p6", "seed_stability", "market_regime", "asset_universe", "walk_forward")),
        ("p7", ("p7", "hpo", "hyperparameter_sweep")),
        ("p8", ("p8", "preference", "uncertainty", "distributional", "partial_rebalance")),
        ("p10", ("p10", "paper_tables", "paper_figures")),
        ("p11", ("p11", "artifact_bundle", "ledger")),
    )
    for phase, markers in ordered:
        if any(_phase_marker_present(text, marker) for marker in markers):
            return phase
    return "unknown"


def _phase_marker_present(text: str, marker: str) -> bool:
    if re.fullmatch(r"p\d+", marker):
        return re.search(rf"(^|[^a-z0-9]){re.escape(marker)}([^a-z0-9]|$)", text) is not None
    return marker in text


def _model_scope(run_dir: Path, manifest: Mapping[str, Any], experiment_result: Mapping[str, Any]) -> list[str]:
    metrics_dir = run_dir / "metrics"
    for filename in (
        "hpo_model_final_comparison.csv",
        "baseline_comparison.csv",
        "main_comparison.csv",
        "ablation_results.csv",
    ):
        path = metrics_dir / filename
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for column in ("paper_model_id", "hpo_model_name", "model_name", "variant_id"):
            if column in frame.columns:
                values = [str(item) for item in frame[column].dropna().drop_duplicates().tolist()]
                if values:
                    return values
    result_model = _clean(experiment_result.get("model_name"))
    manifest_model = _clean(manifest.get("model_name"))
    return [value for value in (result_model, manifest_model) if value]


def _artifact_paths(run_dir: Path) -> list[Path]:
    candidates = (
        run_dir / "logs" / "run_manifest.json",
        run_dir / "logs" / "config_snapshot.yaml",
        run_dir / "logs" / "experiment_result.json",
        run_dir / "logs" / "hpo_search_space_manifest.csv",
        run_dir / "metrics" / "hpo_model_final_comparison.csv",
        run_dir / "metrics" / "hpo_model_final_daily_returns.csv",
        run_dir / "metrics" / "main_comparison.csv",
        run_dir / "metrics" / "baseline_comparison.csv",
        run_dir / "metrics" / "daily_returns.csv",
    )
    return [path for path in candidates if path.exists()]


def _blocking_reason(
    run_dir: Path,
    manifest: Mapping[str, Any],
    experiment_result: Mapping[str, Any],
    artifacts: Sequence[Path],
) -> str:
    if not manifest:
        return "missing_run_manifest"
    status = _clean(manifest.get("status")) or _clean(experiment_result.get("status"))
    if status.startswith("failed"):
        failure_state = manifest.get("failure_state") if isinstance(manifest.get("failure_state"), Mapping) else {}
        return _clean(failure_state.get("error_code")) or _clean(failure_state.get("error")) or status
    if not artifacts:
        return "missing_artifacts"
    if not (run_dir / "metrics" / "daily_returns.csv").exists() and not (
        run_dir / "metrics" / "hpo_model_final_daily_returns.csv"
    ).exists():
        return "missing_final_daily_returns"
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, Mapping) else {}


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


def _scoped_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"ERR_RUN_LEDGER_OUT_OF_SCOPE: {resolved}") from exc
    return resolved


def _scoped_output_dir(path: str | Path) -> Path:
    return _scoped_path(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Core-13 experiment run ledger CSV/JSON.")
    parser.add_argument("--run-dir", action="append", help="Experiment run directory. Repeatable.")
    parser.add_argument("--results-root", default="results", help="Root to discover run_manifest.json files.")
    parser.add_argument(
        "--output-dir",
        default="results/full_reproduction/core13_v2_full_reset_20260522",
    )
    parser.add_argument("--protocol-id", default="core13_v2_full_reset_20260522")
    parser.add_argument("--output-name-stem", default="experiment_run_ledger")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    run_dirs = args.run_dir if args.run_dir else discover_run_dirs(args.results_root)
    return build_experiment_run_ledger(
        run_dirs,
        args.output_dir,
        protocol_id=args.protocol_id,
        output_name_stem=args.output_name_stem,
    )


if __name__ == "__main__":
    main()


__all__ = ["LEDGER_COLUMNS", "build_experiment_run_ledger", "discover_run_dirs"]
