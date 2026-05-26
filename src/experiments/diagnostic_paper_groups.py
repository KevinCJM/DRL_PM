from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT
from src.utils.logger import save_json_atomic


PROTOCOL_ID = "core13_v2_full_reset_20260522"
DATA_CUTOFF_DATE = "2026-05-20"
DIAGNOSTIC_GROUPS = (
    "p2_input_pca",
    "p3_components",
    "p4_reward",
    "p5_cost_rebalance",
    "p6_robustness",
    "p8_modules",
)
REQUIRED_TABLE_FILES = (
    "paper_main_comparison.csv",
    "paper_seed_summary.csv",
)


def audit_diagnostic_paper_groups(
    *,
    root: str | Path = PROJECT_ROOT,
    groups: Sequence[str] = DIAGNOSTIC_GROUPS,
    protocol_id: str = PROTOCOL_ID,
    data_cutoff_date: str = DATA_CUTOFF_DATE,
    write: bool = True,
) -> dict[str, Path]:
    project = _scoped_root(root)
    rows = [
        _audit_group(
            project,
            group,
            protocol_id=protocol_id,
            data_cutoff_date=data_cutoff_date,
            write=write,
        )
        for group in groups
    ]
    output_dir = project / "results/full_reproduction" / protocol_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "diagnostic_paper_group_audit.csv"
    json_path = output_dir / "diagnostic_paper_group_audit.json"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    save_json_atomic(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_id": protocol_id,
            "data_cutoff_date": data_cutoff_date,
            "group_count": len(rows),
            "status_counts": frame["status"].value_counts(dropna=False).to_dict() if not frame.empty else {},
            "groups": rows,
        },
        json_path,
    )
    return {"csv": csv_path, "json": json_path}


def _audit_group(
    root: Path,
    group: str,
    *,
    protocol_id: str,
    data_cutoff_date: str,
    write: bool,
) -> dict[str, Any]:
    group_dir = root / "results/paper_tables" / group
    group_dir.mkdir(parents=True, exist_ok=True)
    source_path = group_dir / "source_run_dirs.txt"
    if write and not source_path.exists():
        source_path.write_text("", encoding="utf-8")

    source_run_dirs = _source_run_dirs(source_path, root)
    source_manifests = [path / "logs/run_manifest.json" for path in source_run_dirs]
    files = _file_status(group_dir)
    missing_required = [name for name in REQUIRED_TABLE_FILES if not files[name]]
    has_paired_or_reason = files["paper_paired_statistics.csv"] or files["not_applicable_reason.txt"]
    source_manifest_count = sum(path.exists() for path in source_manifests)

    if not source_run_dirs:
        status = "aggregation_pending"
        reason = "missing_source_run_dirs"
    elif source_manifest_count != len(source_run_dirs):
        status = "aggregation_pending"
        reason = "source_run_manifest_missing"
    elif missing_required:
        status = "aggregation_pending"
        reason = "missing_required_outputs:" + ",".join(missing_required)
    elif not has_paired_or_reason:
        status = "aggregation_pending"
        reason = "missing_paired_statistics_or_not_applicable_reason"
    else:
        status = "diagnostic_complete"
        reason = ""

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "group_id": group,
        "protocol_id": protocol_id,
        "data_cutoff_date": data_cutoff_date,
        "status": status,
        "reason": reason,
        "rankable_in_unified_table": False,
        "group_dir": str(group_dir),
        "source_run_dirs": [str(path) for path in source_run_dirs],
        "source_run_dir_count": len(source_run_dirs),
        "source_run_manifest_count": source_manifest_count,
        "files": files,
    }
    if write:
        save_json_atomic(payload, group_dir / "diagnostic_status.json")

    return {
        "group_id": group,
        "status": status,
        "reason": reason,
        "group_dir": str(group_dir),
        "source_run_dir_count": len(source_run_dirs),
        "source_run_manifest_count": source_manifest_count,
        "paper_main_comparison": files["paper_main_comparison.csv"],
        "paper_seed_summary": files["paper_seed_summary.csv"],
        "paper_paired_statistics": files["paper_paired_statistics.csv"],
        "not_applicable_reason": files["not_applicable_reason.txt"],
        "diagnostic_status": (group_dir / "diagnostic_status.json").exists() if write else files["diagnostic_status.json"],
    }


def _source_run_dirs(path: Path, root: Path) -> list[Path]:
    if not path.exists():
        return []
    dirs: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        dirs.append(resolved)
    return dirs


def _file_status(group_dir: Path) -> dict[str, bool]:
    names = (
        "paper_main_comparison.csv",
        "paper_seed_summary.csv",
        "paper_paired_statistics.csv",
        "not_applicable_reason.txt",
        "source_run_dirs.txt",
        "diagnostic_status.json",
        "paper_aggregate_manifest.json",
    )
    return {name: (group_dir / name).exists() for name in names}


def _scoped_root(root: str | Path) -> Path:
    resolved = Path(root).expanduser().resolve()
    return resolved


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit diagnostic paper-table group provenance.")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--group", action="append", dest="groups", help="Group id to audit. Repeatable.")
    parser.add_argument("--protocol-id", default=PROTOCOL_ID)
    parser.add_argument("--data-cutoff-date", default=DATA_CUTOFF_DATE)
    parser.add_argument("--no-write", action="store_true", help="Do not write per-group diagnostic_status.json files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    outputs = audit_diagnostic_paper_groups(
        root=args.root,
        groups=args.groups or DIAGNOSTIC_GROUPS,
        protocol_id=args.protocol_id,
        data_cutoff_date=args.data_cutoff_date,
        write=not bool(args.no_write),
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False))
    return outputs


if __name__ == "__main__":
    main()


__all__ = ["DIAGNOSTIC_GROUPS", "audit_diagnostic_paper_groups"]
