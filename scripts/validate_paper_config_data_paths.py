from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATHS = {
    "data/processed/asset_universe.csv",
    "data/processed/etf_lof_daily_panel.parquet",
    "data/metrics_factory/all_metrics_features.parquet",
}
DATA_PATH_KEYS = (
    "asset_universe_path",
    "panel_path",
    "wide_open_path",
    "wide_high_path",
    "wide_low_path",
    "wide_close_path",
    "wide_adj_nav_path",
    "wide_pre_close_path",
    "wide_pct_chg_path",
    "wide_log_return_path",
    "wide_amount_path",
    "wide_vol_path",
    "wide_turnover_rate_path",
    "all_metrics_features_path",
    "download_manifest_path",
    "metrics_manifest_path",
)
REQUIRED_PROTOCOL_VALUES = {
    "protocol_id": "core13_v2_full_reset_20260522",
    "asset_universe_id": "core13_v2",
    "data_cutoff_date": "2026-05-20",
}
REQUIRED_DATA_GOVERNANCE_VALUES = {
    "return_source": "adj_nav",
    "valuation_source": "adj_nav",
    "reward_return_source": "adj_nav",
    "metrics_return_source": "adj_nav",
    "execution_price_source": "ohlcv",
    "valuation_table": "core13_adj_nav",
    "execution_price_table": "core13_ohlcv",
    "valuation_execution_split": True,
    "reward_valuation_split": True,
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def load_yaml(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"ERR_CONFIG_INVALID_TYPE: {path}")
    return payload


def validate_path_value(path: Path, key: str, value: Any, *, require_core13: bool) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    text = "" if value is None else str(value)
    if text in LEGACY_PATHS:
        errors.append({"config": str(path), "key": key, "code": "ERR_LEGACY_17_PATH_FORBIDDEN", "value": text})
    if require_core13 and "core13" not in text:
        errors.append({"config": str(path), "key": key, "code": "ERR_CORE13_PATH_REQUIRED", "value": text})
    return errors


def validate_config(path: Path, *, require_core13: bool) -> list[dict[str, str]]:
    payload = load_yaml(path)
    data = _mapping(payload.get("data"))
    errors: list[dict[str, str]] = []
    if require_core13 and not data:
        return [{"config": str(path), "key": "data", "code": "ERR_CORE13_DATA_OVERRIDE_MISSING", "value": ""}]
    for key in DATA_PATH_KEYS:
        if key in data:
            errors.extend(validate_path_value(path, f"data.{key}", data[key], require_core13=require_core13))
        elif require_core13:
            errors.append({"config": str(path), "key": f"data.{key}", "code": "ERR_CORE13_PATH_KEY_MISSING", "value": ""})
    metrics_factory = _mapping(data.get("metrics_factory"))
    if metrics_factory:
        value = metrics_factory.get("all_metrics_features_path")
        errors.extend(
            validate_path_value(
                path,
                "data.metrics_factory.all_metrics_features_path",
                value,
                require_core13=require_core13,
            )
        )
    elif require_core13:
        errors.append(
            {
                "config": str(path),
                "key": "data.metrics_factory.all_metrics_features_path",
                "code": "ERR_CORE13_PATH_KEY_MISSING",
                "value": "",
            }
        )
    if require_core13:
        errors.extend(validate_protocol(path, payload))
        errors.extend(validate_data_governance(path, payload))
    return errors


def validate_protocol(path: Path, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    protocol = _mapping(payload.get("protocol"))
    errors: list[dict[str, str]] = []
    for key, expected in REQUIRED_PROTOCOL_VALUES.items():
        actual = protocol.get(key)
        if str(actual) != str(expected):
            errors.append(
                {
                    "config": str(path),
                    "key": f"protocol.{key}",
                    "code": "ERR_CORE13_PROTOCOL_VALUE_REQUIRED",
                    "value": "" if actual is None else str(actual),
                }
            )
    return errors


def validate_data_governance(path: Path, payload: Mapping[str, Any]) -> list[dict[str, str]]:
    governance = _mapping(payload.get("data_governance"))
    data = _mapping(payload.get("data"))
    errors: list[dict[str, str]] = []
    if data.get("data_mode") != "availability_mask":
        errors.append(
            {
                "config": str(path),
                "key": "data.data_mode",
                "code": "ERR_CORE13_DATA_MODE_REQUIRED",
                "value": "" if data.get("data_mode") is None else str(data.get("data_mode")),
            }
        )
    for key, expected in REQUIRED_DATA_GOVERNANCE_VALUES.items():
        actual = governance.get(key)
        if actual != expected:
            errors.append(
                {
                    "config": str(path),
                    "key": f"data_governance.{key}",
                    "code": "ERR_CORE13_DATA_GOVERNANCE_REQUIRED",
                    "value": "" if actual is None else str(actual),
                }
            )
    return errors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate paper YAML configs use Core-13 data paths.")
    parser.add_argument("--config-dir", default=str(ROOT / "configs" / "paper"))
    parser.add_argument("--require-core13", action="store_true")
    parser.add_argument("--json-output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir = Path(args.config_dir)
    config_paths = sorted(config_dir.glob("*.yaml"))
    errors: list[dict[str, str]] = []
    for path in config_paths:
        errors.extend(validate_config(path, require_core13=bool(args.require_core13)))
    payload = {"status": "failed" if errors else "passed", "config_count": len(config_paths), "errors": errors}
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
