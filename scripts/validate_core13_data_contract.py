from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ConfigLoader
from src.data.loader import load_market_dataset


CORE13_TOKENS = ("core13_", "core13")
LEGACY_FORBIDDEN = (
    "data/processed/asset_universe.csv",
    "data/processed/etf_lof_daily_panel.parquet",
    "data/metrics_factory/all_metrics_features.parquet",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise RuntimeError(f"{code}: {detail}")


def _path_value(config: Mapping[str, Any], key: str) -> str:
    data = _mapping(config.get("data"))
    value = data.get(key)
    return "" if value is None else str(value)


def validate_core13_paths(config: Mapping[str, Any]) -> None:
    data = _mapping(config.get("data"))
    required_keys = [
        "asset_universe_path",
        "panel_path",
        "wide_open_path",
        "wide_high_path",
        "wide_low_path",
        "wide_close_path",
        "wide_pre_close_path",
        "wide_pct_chg_path",
        "wide_log_return_path",
        "wide_amount_path",
        "wide_vol_path",
        "wide_turnover_rate_path",
        "download_manifest_path",
        "metrics_manifest_path",
    ]
    if bool(_mapping(data.get("metrics_factory")).get("enabled", True)):
        required_keys.append("all_metrics_features_path")
    for key in required_keys:
        value = _path_value(config, key)
        _require("core13" in value, "ERR_CORE13_PATH_REQUIRED", f"data.{key}={value}")
        for forbidden in LEGACY_FORBIDDEN:
            _require(value != forbidden, "ERR_LEGACY_17_PATH_FORBIDDEN", f"data.{key}={value}")


def validate_manifest(config: Mapping[str, Any], *, require_valuation_execution_split: bool) -> None:
    manifest_path = ROOT / _path_value(config, "download_manifest_path")
    _require(manifest_path.exists(), "ERR_CORE13_MANIFEST_MISSING", str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calendar_loss = _mapping(manifest.get("calendar_loss"))
    _require(bool(calendar_loss), "ERR_CORE13_CALENDAR_LOSS_MISSING", str(manifest_path))
    data_mode = str(calendar_loss.get("data_mode", ""))
    if data_mode == "strict_common_history":
        _require(
            float(calendar_loss.get("calendar_loss_retention_ratio", 0.0)) >= 0.90,
            "ERR_CORE13_CALENDAR_LOSS_THRESHOLD_FAILED",
            str(calendar_loss.get("calendar_loss_retention_ratio")),
        )
    else:
        _require(data_mode == "availability_mask", "ERR_CORE13_DATA_MODE_INVALID", data_mode)
    _require(calendar_loss.get("passed") is True, "ERR_CORE13_CALENDAR_LOSS_NOT_PASSED", str(calendar_loss))
    if require_valuation_execution_split:
        _require(
            manifest.get("valuation_execution_split") is True,
            "ERR_CORE13_VALUATION_EXECUTION_SPLIT_REQUIRED",
            str(manifest.get("valuation_execution_split")),
        )
        _require(manifest.get("return_source") == "adj_nav", "ERR_CORE13_RETURN_SOURCE_INVALID", str(manifest.get("return_source")))
        _require(
            manifest.get("execution_price_source") == "ohlcv",
            "ERR_CORE13_EXECUTION_SOURCE_INVALID",
            str(manifest.get("execution_price_source")),
        )


def _normalize_date_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
    return text


def validate_source_cutoff_manifests(expected_data_cutoff_date: str) -> None:
    expected = _normalize_date_token(expected_data_cutoff_date)
    checks = (
        (
            ROOT / "data/reports/core13_etf_lof_fund_nav_tushare_manifest.json",
            "end_date_requested",
            "ERR_CORE13_NAV_CUTOFF_MISMATCH",
        ),
        (
            ROOT / "data/reports/core13_ohlcv_download_manifest.json",
            "end_date_requested",
            "ERR_CORE13_OHLCV_CUTOFF_MISMATCH",
        ),
    )
    for path, key, code in checks:
        _require(path.exists(), "ERR_CORE13_SOURCE_MANIFEST_MISSING", str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = _normalize_date_token(payload.get(key))
        _require(actual == expected, code, f"{path.relative_to(ROOT)}.{key}={actual}; expected={expected}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a config against the Core-13 formal data contract.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-asset-count", type=int, default=13)
    parser.add_argument("--require-valuation-execution-split", action="store_true")
    parser.add_argument(
        "--require-data-cutoff-date",
        help="Require raw NAV/OHLCV source download manifests to request exactly this cutoff date.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = ConfigLoader.load(args.config)
    validate_core13_paths(config)
    validate_manifest(config, require_valuation_execution_split=args.require_valuation_execution_split)
    if args.require_data_cutoff_date:
        validate_source_cutoff_manifests(args.require_data_cutoff_date)
    bundle = load_market_dataset(config)
    _require(
        len(bundle.asset_universe) == args.expected_asset_count,
        "ERR_CORE13_ASSET_COUNT_INVALID",
        str(len(bundle.asset_universe)),
    )
    manifest_path = ROOT / _path_value(config, "download_manifest_path")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_mode = str(_mapping(manifest.get("calendar_loss")).get("data_mode", ""))
    if data_mode == "strict_common_history":
        _require(
            bool(bundle.data_manifest.get("all_assets_available_each_date")),
            "ERR_CORE13_COMMON_HISTORY_INVALID",
            str(bundle.data_manifest.get("all_assets_available_each_date")),
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "config": args.config,
                "asset_count": len(bundle.asset_universe),
                "date_start": bundle.data_manifest.get("date_start"),
                "date_end": bundle.data_manifest.get("date_end"),
                "date_count": bundle.data_manifest.get("date_count"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
