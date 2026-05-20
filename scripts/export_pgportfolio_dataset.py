from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import ConfigLoader, assert_path_allowed
from src.data.loader import load_market_dataset
from src.data.splits import create_split, split_to_dict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export platform market data for PGPortfolio external runs.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--output-dir", required=True, help="Output directory under the configured whitelist.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ConfigLoader.load(args.config)
    output_dir = assert_path_allowed(args.output_dir, config["security"]["path_whitelist"], "external_pgportfolio.output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_market_dataset(config)
    date_index = pd.DatetimeIndex(dataset.wide["close"].index)
    split = create_split(date_index, config)
    asset_order = list(dataset.data_manifest.get("canonical_asset_order", dataset.wide["close"].columns))

    dataset.asset_universe.to_csv(output_dir / "asset_universe.csv", index=False)
    for field in ("open", "close", "amount", "vol", "log_return"):
        frame = dataset.wide.get(field)
        if frame is not None:
            frame.reindex(columns=asset_order).to_csv(output_dir / f"wide_{field}.csv", index_label="date")

    manifest = {
        "status": "completed",
        "data_protocol": "platform_to_pgportfolio_export",
        "asset_order": asset_order,
        "split": split_to_dict(split),
        "files": {
            "asset_universe": "asset_universe.csv",
            "wide_open": "wide_open.csv",
            "wide_close": "wide_close.csv",
            "wide_amount": "wide_amount.csv",
            "wide_vol": "wide_vol.csv",
            "wide_log_return": "wide_log_return.csv",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
