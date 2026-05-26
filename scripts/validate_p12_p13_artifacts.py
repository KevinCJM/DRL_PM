from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_METRIC_FILES = {
    "gate_actions.csv": {"date", "model_name", "paper_model_id", "gate_action_index", "rho", "scheduler_allowed_rebalance"},
    "cage_eiie_candidate_weights.csv": {"date", "model_name", "asset_id", "candidate_weight"},
    "cage_final_weights.csv": {"date", "model_name", "asset_id", "executed_weight"},
    "turnover_cost_breakdown.csv": {"date", "model_name", "estimated_turnover", "realized_turnover", "estimated_cost", "realized_cost"},
    "risk_metrics.csv": {"model_name", "cumulative_return", "max_drawdown_loss", "CVaR_loss_5"},
    "validation_selection_report.csv": {"model_name", "selection_split", "test_used_for_model_selection", "model_extension_id"},
}
MODEL_EXTENSION_ID = "core13_v2_p12_p13_20260524"


def validate_run(run_dir: str | Path) -> list[dict[str, str]]:
    root = Path(run_dir)
    rows: list[dict[str, str]] = []
    metrics_dir = root / "metrics"
    logs_dir = root / "logs"
    manifest = _read_json(logs_dir / "new_model_sidecar_manifest.json")
    rows.append(
        _row(
            "new_model_sidecar_manifest",
            logs_dir / "new_model_sidecar_manifest.json",
            manifest.get("model_extension_id") == MODEL_EXTENSION_ID,
            f"model_extension_id={manifest.get('model_extension_id')}",
        )
    )
    for filename, required_columns in REQUIRED_METRIC_FILES.items():
        path = metrics_dir / filename
        frame = _read_csv(path)
        columns_ok = required_columns.issubset(set(frame.columns))
        non_empty = not frame.empty
        extension_ok = "model_extension_id" not in frame.columns or frame["model_extension_id"].fillna("").astype(str).eq(MODEL_EXTENSION_ID).all()
        rows.append(
            _row(
                filename,
                path,
                path.exists() and columns_ok and non_empty and extension_ok,
                f"exists={path.exists()}, rows={len(frame)}, columns_ok={columns_ok}, extension_ok={extension_ok}",
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate P12/P13 new-model artifacts under a run directory.")
    parser.add_argument("run_dir")
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    rows = validate_run(args.run_dir)
    frame = pd.DataFrame(rows)
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    if not bool(frame["passed"].all()):
        raise SystemExit(1)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _row(name: str, path: Path, passed: bool, detail: str) -> dict[str, str]:
    return {"check": name, "path": str(path), "passed": bool(passed), "detail": detail}


if __name__ == "__main__":
    main()
