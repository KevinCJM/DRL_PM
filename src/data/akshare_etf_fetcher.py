from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.config import PROJECT_ROOT
from src.data.preprocess import WIDE_FIELDS, detect_amount_proxy, panel_to_wide_tables


DEFAULT_PANEL_PATH = PROJECT_ROOT / "data/processed/etf_lof_daily_panel.parquet"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data/reports/akshare_etf_lof_download_manifest.json"


def _read_panel(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
        return path
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp.parquet",
            delete=False,
        ) as fh:
            temp_path = Path(fh.name)
        frame.to_parquet(temp_path)
        with temp_path.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
        return path
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _source_range(panel: pd.DataFrame) -> dict[str, str | None]:
    trade_dates = pd.to_datetime(panel["trade_date"]).dropna()
    if trade_dates.empty:
        return {"start": None, "end": None}
    return {
        "start": trade_dates.min().date().isoformat(),
        "end": trade_dates.max().date().isoformat(),
    }


def regenerate_from_panel(
    panel: pd.DataFrame,
    *,
    processed_dir: Path,
    manifest_path: Path,
    source_panel_path: Path | None = None,
    asset_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    panel_with_amount, amount_is_proxy = detect_amount_proxy(panel)
    wide = panel_to_wide_tables(panel_with_amount, asset_order=asset_order)

    panel_output_path = processed_dir / "etf_lof_daily_panel.parquet"
    _write_parquet_atomic(panel_with_amount, panel_output_path)
    for field in WIDE_FIELDS:
        _write_parquet_atomic(wide[field], processed_dir / f"wide_{field}.parquet")

    turnover_rate_all_missing = bool(wide["turnover_rate"].isna().all().all())
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": "src.data.akshare_etf_fetcher",
        "source_panel_path": str(source_panel_path) if source_panel_path is not None else None,
        "source_range": _source_range(panel_with_amount),
        "panel_rows": int(len(panel_with_amount)),
        "asset_count": int(panel_with_amount["ts_code"].nunique()),
        "amount_is_proxy": bool(amount_is_proxy),
        "proxy_flags": {"amount": bool(amount_is_proxy)},
        "turnover_rate_all_missing": turnover_rate_all_missing,
        "wide_fields": list(WIDE_FIELDS),
        "processed_dir": str(processed_dir),
    }
    _write_json_atomic(manifest, manifest_path)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit ETF/LOF processed data regeneration entrypoint.")
    parser.add_argument("--panel", default=str(DEFAULT_PANEL_PATH))
    parser.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    panel_path = Path(args.panel).expanduser().resolve()
    processed_dir = Path(args.processed_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    panel = _read_panel(panel_path)
    return regenerate_from_panel(
        panel,
        processed_dir=processed_dir,
        manifest_path=manifest_path,
        source_panel_path=panel_path,
    )


if __name__ == "__main__":
    main()
