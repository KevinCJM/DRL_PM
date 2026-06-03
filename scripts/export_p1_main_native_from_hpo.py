from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigLoader
from src.utils.logger import save_json_atomic, write_run_outputs


PROTOCOL_ID = "core13_v2_full_reset_20260522"
DATA_CUTOFF_DATE = "2026-05-20"
ACTIVE_ACTIVITY_PROTOCOL = "daily_gate_with_cost_constraint"
SOURCE_COMPARISON = "hpo_model_final_comparison.csv"
SOURCE_METRIC_FILES = {
    "baseline_comparison": "hpo_model_final_comparison.csv",
    "daily_returns": "hpo_model_final_daily_returns.csv",
    "daily_weights": "hpo_model_final_daily_weights.csv",
    "daily_turnover": "hpo_model_final_daily_turnover.csv",
    "daily_rebalance": "hpo_model_final_daily_rebalance.csv",
    "daily_costs": "hpo_model_final_daily_costs.csv",
}


def export_p1_from_hpo(
    *,
    config_path: str | Path,
    source_run_dir: str | Path,
    output_run_dir: str | Path,
) -> dict[str, Path]:
    config = ConfigLoader.load(config_path)
    source_dir = Path(source_run_dir)
    output_dir = Path(output_run_dir)
    manifest = _read_json(source_dir / "logs" / "run_manifest.json")
    _assert_source_ready(source_dir, manifest)

    seed = int(manifest.get("seed"))
    run_name = output_dir.name
    resolved = deepcopy(dict(config))
    resolved.setdefault("output", {})
    resolved["output"]["run_name"] = run_name
    resolved.setdefault("reproducibility", {})
    resolved["reproducibility"]["seed"] = seed
    resolved.setdefault("rankability", {})
    resolved["rankability"]["rankable_in_unified_table"] = True
    resolved["rankability"]["diagnostic_status"] = "formal"
    resolved["rankability"]["discard_reason"] = None

    frames = {name: _read_csv(source_dir / "metrics" / filename) for name, filename in SOURCE_METRIC_FILES.items()}
    comparison = frames["baseline_comparison"]
    comparison["seed"] = seed
    frames["baseline_comparison"] = comparison
    for name in ("daily_returns", "daily_weights", "daily_turnover", "daily_rebalance", "daily_costs"):
        frame = frames[name]
        if "seed" not in frame.columns or frame["seed"].isna().all():
            frame["seed"] = seed
        frames[name] = frame

    result = {
        "status": "completed",
        "baseline_comparison": frames["baseline_comparison"],
        "daily_returns": frames["daily_returns"],
        "daily_weights": frames["daily_weights"],
        "daily_turnover": frames["daily_turnover"],
        "daily_rebalance": frames["daily_rebalance"],
        "daily_costs": frames["daily_costs"],
        "run_manifest": manifest,
        "rankable_in_unified_table": True,
        "diagnostic_status": "formal",
        "availability_mask_contract": manifest.get("availability_mask_contract") or {},
        "lineage": {
            "source_run_dir": str(source_dir.resolve()),
            "source_run_name": str(manifest.get("run_name") or source_dir.name),
            "source_comparison_file": SOURCE_COMPARISON,
        },
    }

    result_path = output_dir / "logs" / "experiment_result.json"
    save_json_atomic(_result_summary(result), result_path)
    artifacts = write_run_outputs(
        result,
        output_dir,
        config=resolved,
        config_path=config_path,
        command=_command_string(),
        asset_list=_read_asset_list(source_dir / "logs" / "asset_list.txt"),
        data_split=_read_json(source_dir / "logs" / "data_split.json"),
        manifest_overrides={
            "status": "success",
            "experiment_type": "baseline_comparison",
            "output_name": "baseline_comparison",
            "result_path": str(result_path),
            "run_id": run_name,
            "run_name": run_name,
            "seed": seed,
            "source_hpo_run": str(manifest.get("run_name") or source_dir.name),
            "source_hpo_run_dir": str(source_dir.resolve()),
            "source_hpo_comparison_file": SOURCE_COMPARISON,
            "protocol_id": PROTOCOL_ID,
            "data_cutoff_date": DATA_CUTOFF_DATE,
        },
    )
    save_json_atomic(
        {
            "source_run_dir": str(source_dir.resolve()),
            "source_run_name": str(manifest.get("run_name") or source_dir.name),
            "source_manifest_path": str((source_dir / "logs" / "run_manifest.json").resolve()),
            "source_metric_files": {name: filename for name, filename in SOURCE_METRIC_FILES.items()},
            "output_run_dir": str(output_dir.resolve()),
            "output_run_name": run_name,
            "seed": seed,
        },
        output_dir / "logs" / "p1_from_hpo_source.json",
    )
    return artifacts


def _assert_source_ready(source_dir: Path, manifest: dict[str, Any]) -> None:
    if not manifest:
        raise FileNotFoundError(f"ERR_P1_HPO_SOURCE_MANIFEST_MISSING: {source_dir}")
    required = (
        manifest.get("status") == "success"
        and manifest.get("diagnostic_status") == "formal"
        and manifest.get("rankable_in_unified_table") is True
        and manifest.get("protocol_id") == PROTOCOL_ID
        and str(manifest.get("data_cutoff_date")) == DATA_CUTOFF_DATE
        and manifest.get("execution_activity_protocol") == ACTIVE_ACTIVITY_PROTOCOL
        and manifest.get("turnover_optimization_protocol_id") == "turnover_active_v1"
        and manifest.get("scheduler_blocks_model_actions") is False
        and manifest.get("activity_gate_enforced") is True
    )
    if not required:
        raise RuntimeError(f"ERR_P1_HPO_SOURCE_NOT_READY: {source_dir}")
    if "seed" not in manifest or manifest.get("seed") is None:
        raise RuntimeError(f"ERR_P1_HPO_SOURCE_SEED_MISSING: {source_dir}")
    for filename in SOURCE_METRIC_FILES.values():
        path = source_dir / "metrics" / filename
        if not path.exists():
            raise FileNotFoundError(f"ERR_P1_HPO_SOURCE_METRIC_MISSING: {path}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ERR_P1_HPO_SOURCE_METRIC_MISSING: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload) if isinstance(payload, dict) else {}


def _read_asset_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
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


def _command_string() -> str:
    return " ".join(["python", "scripts/export_p1_main_native_from_hpo.py", *map(str, sys.argv[1:])])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export P1 main-native formal run dirs from completed P7 HPO outputs.")
    parser.add_argument("--config", default="configs/paper/baseline_comparison_main_native_from_hpo.yaml")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_p1_from_hpo(
        config_path=args.config,
        source_run_dir=args.source_run_dir,
        output_run_dir=args.output_run_dir,
    )


if __name__ == "__main__":
    main()
