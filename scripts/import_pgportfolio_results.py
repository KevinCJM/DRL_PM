from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import ConfigLoader, PROJECT_ROOT, assert_path_allowed
from src.data.loader import load_market_dataset
from src.data.splits import create_split
from src.experiments.external_baselines import import_external_pgportfolio_outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import PGPortfolio external results into platform CSV shape.")
    parser.add_argument("--results-csv", required=True, help="External result CSV with date/nav/net_return and weights.")
    parser.add_argument("--config", help="Platform config YAML used to derive test split and availability validation.")
    parser.add_argument("--asset-universe-csv", help="CSV containing ts_code/asset_id columns for test assets.")
    parser.add_argument("--output-dir", required=True, help="Output directory under the project subtree.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = assert_path_allowed(args.output_dir, [PROJECT_ROOT], "external_pgportfolio.import_output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(assert_path_allowed(args.results_csv, [PROJECT_ROOT], "external_pgportfolio.results_csv"))
    test_dates = None
    availability_mask = None
    if args.config:
        config = ConfigLoader.load(args.config)
        dataset = load_market_dataset(config)
        split = create_split(pd.DatetimeIndex(dataset.wide["close"].index), config)
        primary_split = split[0] if isinstance(split, list) else split
        assets = [str(item) for item in dataset.data_manifest.get("canonical_asset_order", dataset.wide["close"].columns)]
        test_dates = primary_split.test_dates
        availability_mask = dataset.availability_mask
    else:
        if not args.asset_universe_csv:
            raise SystemExit("--asset-universe-csv is required when --config is not provided")
        asset_universe = pd.read_csv(
            assert_path_allowed(args.asset_universe_csv, [PROJECT_ROOT], "external_pgportfolio.asset_universe_csv")
        )
        assets = _asset_order(asset_universe)
    payload = import_external_pgportfolio_outputs(
        results,
        assets,
        config=config if args.config else None,
        test_dates=test_dates,
        availability_mask=availability_mask,
    )
    validation = payload.get("validation", {"status": payload["status"]})
    manifest = {
        "status": validation["status"],
        "validation": validation,
        "cost_model_shared": False,
        "cost_availability": "not_available",
        "rankable_in_unified_table": False,
    }
    (output_dir / "import_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload["status"] != "completed":
        return 1

    for name in ("daily_returns", "daily_weights", "daily_turnover", "daily_rebalance", "daily_costs"):
        payload[name].to_csv(output_dir / f"{name}.csv", index=False)
    return 0


def _asset_order(asset_universe: pd.DataFrame) -> list[str]:
    for column in ("ts_code", "asset_id", "asset"):
        if column in asset_universe.columns:
            return [str(item) for item in asset_universe[column].dropna().tolist()]
    if asset_universe.empty:
        return []
    return [str(item) for item in asset_universe.iloc[:, 0].dropna().tolist()]


if __name__ == "__main__":
    raise SystemExit(main())
