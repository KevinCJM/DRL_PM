from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_REFERENCE_MODELS = {
    "eiie_native",
    "full_dqn_gated_multitask_cnn_ppo",
    "ppo_dqn_hierarchical_reimplementation",
    "cnn_ppo_native",
    "pgportfolio_eiie_native",
}
REQUIRED_REFERENCE_FILES = {
    "validation_reference_comparison.csv",
    "validation_reference_daily_returns.csv",
    "validation_selection_report.csv",
    "validation_reference_manifest.json",
}


def validate_reference(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if target.is_dir():
        return _validate_reference_dir(target)
    payload = _read_payload(target)
    if target.name == "config_snapshot.yaml":
        hpo = payload.get("hpo", {}) if isinstance(payload, dict) else {}
        new_model = payload.get("new_model_protocol", {}) if isinstance(payload, dict) else {}
    else:
        hpo = payload if isinstance(payload, dict) else {}
        new_model = payload if isinstance(payload, dict) else {}
    selection_split = hpo.get("selection_split") or new_model.get("selection_split")
    test_used = new_model.get("test_used_for_model_selection", payload.get("test_used_for_model_selection") if isinstance(payload, dict) else None)
    return [
        _row("selection_split_validation", target, selection_split == "validation", f"selection_split={selection_split}"),
        _row("test_not_used_for_selection", target, test_used is False, f"test_used_for_model_selection={test_used}"),
    ]


def _validate_reference_dir(target: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filename in REQUIRED_REFERENCE_FILES:
        path = target / filename
        rows.append(_row(f"required_file:{filename}", path, path.exists(), f"exists={path.exists()}"))

    comparison = _read_csv(target / "validation_reference_comparison.csv")
    returns = _read_csv(target / "validation_reference_daily_returns.csv")
    selection = _read_csv(target / "validation_selection_report.csv")
    manifest = _read_payload(target / "validation_reference_manifest.json")
    rows.extend(
        [
            _model_coverage_row("comparison_models", target / "validation_reference_comparison.csv", comparison),
            _model_coverage_row("daily_return_models", target / "validation_reference_daily_returns.csv", returns),
            _model_coverage_row("selection_report_models", target / "validation_selection_report.csv", selection),
            _row(
                "selection_split_validation",
                target / "validation_reference_manifest.json",
                manifest.get("selection_split") == "validation",
                f"selection_split={manifest.get('selection_split')}",
            ),
            _row(
                "test_not_used_for_selection",
                target / "validation_reference_manifest.json",
                manifest.get("test_used_for_model_selection") is False,
                f"test_used_for_model_selection={manifest.get('test_used_for_model_selection')}",
            ),
        ]
    )
    if not comparison.empty:
        finite_columns = [
            column
            for column in (
                "cumulative_return",
                "turnover_mean",
                "transaction_cost_total",
                "max_drawdown_loss",
                "CVaR_loss_5",
                "validation_return_cost_risk_utility",
            )
            if column in comparison.columns
        ]
        finite_ok = bool(finite_columns) and all(
            pd.to_numeric(comparison[column], errors="coerce").notna().all()
            for column in finite_columns
        )
        rows.append(
            _row(
                "comparison_metrics_finite",
                target / "validation_reference_comparison.csv",
                finite_ok,
                f"finite_columns={finite_columns}",
            )
        )
    return rows


def _model_coverage_row(name: str, path: Path, frame: pd.DataFrame) -> dict[str, str]:
    models = set()
    if not frame.empty and "model_name" in frame.columns:
        models = set(frame["model_name"].dropna().astype(str))
    missing = sorted(REQUIRED_REFERENCE_MODELS - models)
    return _row(name, path, not missing, f"missing={missing}, observed={sorted(models)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate P12/P13 validation-only model selection metadata.")
    parser.add_argument("path", help="config_snapshot.yaml or run_manifest.json")
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    rows = validate_reference(args.path)
    frame = pd.DataFrame(rows)
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    if not bool(frame["passed"].all()):
        raise SystemExit(1)


def _read_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    else:
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
