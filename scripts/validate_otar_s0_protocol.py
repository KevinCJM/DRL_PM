from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ConfigLoader, PROJECT_ROOT


PROTOCOL_ID = "otar_v2_s0_20260605"
OUTPUT_DIR = Path("results/paper_tables/otar_protocol_manifest")
RUN_CONFIGS = (
    Path("configs/paper/otar_small8_smoke.yaml"),
    Path("configs/paper/otar_small8_pilot.yaml"),
    Path("configs/paper/otar_core13_robustness.yaml"),
)
CQR_GRID = Path("configs/paper/otar_cqr_hpo_grid.yaml")
ACTOR_GRID = Path("configs/paper/otar_actor_policy_hpo_grid.yaml")
SCHEMA_FIXTURE = Path("configs/paper/otar_s0_schemas/schema_freeze.yaml")
SMALL8_UNIVERSE = Path("configs/data/small8_universe.yaml")
FORMAL_MATRIX = Path("configs/paper/otar_formal_matrix.yaml")

REQUIRED_CQR_KEYS = {
    "gate_gamma",
    "n_quantiles",
    "quantile_huber_kappa",
    "gate_margin",
    "gate_lr",
    "target_update_interval",
    "replay_min_size",
    "gate_batch_size",
    "calibration_horizon_H",
    "epsilon_start",
    "epsilon_end",
    "min_rebalance_ratio_in_buffer",
    "min_hold_ratio_in_buffer",
}
REQUIRED_ACTOR_KEYS = {
    "concentration_alpha_min",
    "concentration_alpha_max",
    "entropy_coef",
    "simplex_parameterization",
}
FORBIDDEN_FIELDS = {
    "pred_candidate_cvar_loss",
    "pred_hold_cvar_loss",
    "realized_gate_action_cvar_proxy",
}
REQUIRED_SCHEMA_ATTRS = {
    "field_name",
    "dtype",
    "shape",
    "unit",
    "level",
    "required",
    "missing_value_policy",
    "forbidden_aliases",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ERR_OTAR_S0_INVALID_YAML: {path} must contain a mapping")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"ERR_OTAR_S0_MISSING_FILE: {path}")


def _validate_grid(path: Path, required: set[str]) -> dict[str, Any]:
    _require_path(path)
    payload = _read_yaml(path)
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"ERR_OTAR_S0_PROTOCOL_ID: {path}")
    candidate_values = payload.get("candidate_values")
    if not isinstance(candidate_values, dict):
        raise ValueError(f"ERR_OTAR_S0_GRID_MISSING_CANDIDATES: {path}")
    missing = sorted(required - set(candidate_values))
    if missing:
        raise ValueError(f"ERR_OTAR_S0_GRID_MISSING_KEYS: {path}: {missing}")
    empty = sorted(key for key in required if not candidate_values.get(key))
    if empty:
        raise ValueError(f"ERR_OTAR_S0_GRID_EMPTY_VALUES: {path}: {empty}")
    if payload.get("no_test_peeking") is not True:
        raise ValueError(f"ERR_OTAR_S0_GRID_TEST_PEEKING_NOT_BLOCKED: {path}")
    return payload


def _validate_schema_fixture(path: Path) -> dict[str, Any]:
    _require_path(path)
    payload = _read_yaml(path)
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"ERR_OTAR_S0_SCHEMA_PROTOCOL_ID: {path}")
    declared_forbidden = set(payload.get("forbidden_new_output_fields") or [])
    if not FORBIDDEN_FIELDS.issubset(declared_forbidden):
        raise ValueError("ERR_OTAR_S0_SCHEMA_FORBIDDEN_FIELDS_INCOMPLETE")
    schemas = payload.get("schemas")
    if not isinstance(schemas, dict):
        raise ValueError("ERR_OTAR_S0_SCHEMA_MISSING_SCHEMAS")
    for required_schema in ("selected_config", "daily_diagnostics", "metrics", "run_manifest", "daily_asset_returns"):
        if required_schema not in schemas:
            raise ValueError(f"ERR_OTAR_S0_SCHEMA_MISSING_SECTION: {required_schema}")
        fields = schemas[required_schema].get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"ERR_OTAR_S0_SCHEMA_EMPTY_FIELDS: {required_schema}")
        for field in fields:
            if not isinstance(field, dict):
                raise ValueError(f"ERR_OTAR_S0_SCHEMA_FIELD_NOT_MAPPING: {required_schema}")
            missing = REQUIRED_SCHEMA_ATTRS - set(field)
            if missing:
                raise ValueError(
                    f"ERR_OTAR_S0_SCHEMA_FIELD_MISSING_ATTRS: {required_schema}.{field.get('field_name')}: {sorted(missing)}"
                )
            if field["field_name"] in FORBIDDEN_FIELDS:
                raise ValueError(f"ERR_OTAR_S0_SCHEMA_USES_FORBIDDEN_FIELD: {field['field_name']}")
    return payload


def _validate_small8(path: Path) -> dict[str, Any]:
    _require_path(path)
    payload = _read_yaml(path)
    assets = payload.get("assets")
    if not isinstance(assets, list) or len(assets) != 8:
        raise ValueError("ERR_OTAR_S0_SMALL8_ASSET_COUNT")
    codes = [asset.get("ts_code") for asset in assets if isinstance(asset, dict)]
    if len(set(codes)) != 8 or any(not code for code in codes):
        raise ValueError("ERR_OTAR_S0_SMALL8_ASSET_CODES")
    protocol = payload.get("selection_protocol") or {}
    if protocol.get("future_performance_screening_used") is not False:
        raise ValueError("ERR_OTAR_S0_SMALL8_FUTURE_SCREENING_NOT_BLOCKED")
    return payload


_ABLATION_MODE_MAP = {
    "A0": "A2_net_log_return_after_cost",
    "A1": "A2_net_log_return_after_cost",
    "A2": "A8_cvar_sensitive",
    "A3": "A13_otar_soft_ru_cvar_fixed",
    "A4_lite": "A13_otar_soft_ru_cvar_fixed",
    "A4": "A13_otar_soft_ru_cvar_fixed",
}
_A2_PENALTY_LAMBDAS = (
    "lambda_turnover",
    "lambda_downside",
    "lambda_drawdown",
    "lambda_dd",
    "lambda_volatility",
    "lambda_concentration",
)
_BASELINE_CONTRACTS_PATH = Path("configs/paper/otar_baseline_contracts.yaml")
_REQUIRED_BASELINE_CONTRACT_KEYS = {
    "rebalance_frequency",
    "transaction_cost_model",
    "input_feature_set",
    "hpo_budget",
}


def _validate_run_config(path: Path) -> dict[str, Any]:
    _require_path(path)
    config = ConfigLoader.load(PROJECT_ROOT / path)
    if config["protocol"]["protocol_id"] != PROTOCOL_ID:
        raise ValueError(f"ERR_OTAR_S0_PROTOCOL_ID_MISMATCH: {path}")
    if config["rebalance"]["mode"] != "daily":
        raise ValueError(f"ERR_OTAR_S0_RUN_NOT_DAILY: {path}")
    activity = config["execution_activity"]
    if activity["protocol"] != "daily_gate_with_cost_constraint":
        raise ValueError(f"ERR_OTAR_S0_RUN_ACTIVITY_PROTOCOL: {path}")
    if activity["scheduler_blocks_model_actions"] is not False:
        raise ValueError(f"ERR_OTAR_S0_RUN_SCHEDULER_BLOCKS_MODEL: {path}")
    if activity["activity_gate_enforced"] is not True:
        raise ValueError(f"ERR_OTAR_S0_RUN_ACTIVITY_GATE_DISABLED: {path}")
    if config["training"]["checkpoint_include_replay_buffer"] is not False:
        raise ValueError(f"ERR_OTAR_S0_RUN_REPLAY_CHECKPOINT_ENABLED: {path}")
    if config["hpo"]["selection_split"] != "validation" or config["hpo"]["final_report_split"] != "test":
        raise ValueError(f"ERR_OTAR_S0_RUN_SPLIT_POLICY: {path}")

    reward = config.get("reward") or {}
    reward_mode = reward.get("mode")
    if not reward_mode:
        raise ValueError(f"ERR_OTAR_S0_RUN_REWARD_MODE_MISSING: {path}")

    ablation_id = (config.get("experiment") or {}).get("ablation_id", "")
    if ablation_id:
        expected_mode = _ABLATION_MODE_MAP.get(ablation_id)
        if expected_mode is None:
            raise ValueError(f"ERR_OTAR_S0_RUN_ABLATION_ID_UNKNOWN: {path}: {ablation_id}")
        if reward_mode != expected_mode:
            raise ValueError(
                f"ERR_OTAR_S0_RUN_REWARD_MODE_MISMATCH: {path}: "
                f"ablation_id={ablation_id} requires reward.mode={expected_mode}, got {reward_mode}"
            )
        if ablation_id == "A2":
            violations = [
                key for key in _A2_PENALTY_LAMBDAS
                if float(reward.get(key, 0.0)) != 0.0
            ]
            if violations:
                raise ValueError(
                    f"ERR_OTAR_A2_PENALTY_LOCK_VIOLATION: {path}: "
                    f"non-zero lambdas: {violations}"
                )
    else:
        if reward_mode not in ("A13_otar_soft_ru_cvar_fixed", "A2_net_log_return_after_cost", "A8_cvar_sensitive"):
            raise ValueError(f"ERR_OTAR_S0_RUN_ABLATION_ID_MISSING: {path}")
        if reward_mode != "A13_otar_soft_ru_cvar_fixed":
            raise ValueError(
                f"ERR_OTAR_S0_RUN_REWARD_MODE_MISMATCH: {path}: "
                f"base OTAR config requires reward.mode=A13_otar_soft_ru_cvar_fixed, got {reward_mode}"
            )

    return config


def _validate_baseline_contracts() -> dict[str, Any]:
    if not _BASELINE_CONTRACTS_PATH.exists():
        raise FileNotFoundError(
            f"ERR_OTAR_S0_BASELINE_CONTRACTS_INVALID: {_BASELINE_CONTRACTS_PATH} does not exist"
        )
    payload = _read_yaml(_BASELINE_CONTRACTS_PATH)
    contracts = payload.get("contracts")
    if not isinstance(contracts, dict) or not contracts:
        raise ValueError(
            f"ERR_OTAR_S0_BASELINE_CONTRACTS_INVALID: {_BASELINE_CONTRACTS_PATH} missing 'contracts' section"
        )
    for name, contract in contracts.items():
        if not isinstance(contract, dict):
            raise ValueError(
                f"ERR_OTAR_S0_BASELINE_CONTRACTS_INVALID: contract '{name}' is not a mapping"
            )
        missing = sorted(_REQUIRED_BASELINE_CONTRACT_KEYS - set(contract))
        if missing:
            raise ValueError(
                f"ERR_OTAR_S0_BASELINE_CONTRACTS_INVALID: contract '{name}' missing keys: {missing}"
            )
    return payload


def _detect_hash_drift(current_files: dict[str, str], post_freeze: bool) -> list[str]:
    manifest_path = OUTPUT_DIR / "protocol_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    stored_files = stored.get("files") or {}
    changed = sorted(
        path for path, digest in current_files.items()
        if stored_files.get(path) != digest
    )
    if not changed:
        return []
    if post_freeze:
        raise ValueError(
            f"ERR_OTAR_S0_PROTOCOL_HASH_DRIFT_POST_FREEZE: "
            f"managed files changed: {changed}. Bump protocol_id."
        )
    print(f"WARNING: MANAGED_FILE_HASH_CHANGED: {changed}", file=sys.stderr)
    return changed


def validate_protocol(post_freeze: bool = False) -> dict[str, Any]:
    cqr_grid = _validate_grid(CQR_GRID, REQUIRED_CQR_KEYS)
    actor_grid = _validate_grid(ACTOR_GRID, REQUIRED_ACTOR_KEYS)
    schema = _validate_schema_fixture(SCHEMA_FIXTURE)
    universe = _validate_small8(SMALL8_UNIVERSE)
    run_configs = {str(path): _validate_run_config(path) for path in RUN_CONFIGS}
    baseline_contracts = _validate_baseline_contracts()
    files = {
        str(CQR_GRID): _sha256(CQR_GRID),
        str(ACTOR_GRID): _sha256(ACTOR_GRID),
        str(SCHEMA_FIXTURE): _sha256(SCHEMA_FIXTURE),
        str(SMALL8_UNIVERSE): _sha256(SMALL8_UNIVERSE),
        str(_BASELINE_CONTRACTS_PATH): _sha256(_BASELINE_CONTRACTS_PATH),
        str(FORMAL_MATRIX): _sha256(FORMAL_MATRIX),
        **{str(path): _sha256(path) for path in RUN_CONFIGS},
    }
    _detect_hash_drift(files, post_freeze)
    return {
        "status": "success",
        "protocol_id": PROTOCOL_ID,
        "files": files,
        "cqr_grid_keys": sorted((cqr_grid.get("candidate_values") or {}).keys()),
        "actor_grid_keys": sorted((actor_grid.get("candidate_values") or {}).keys()),
        "schema_sections": sorted((schema.get("schemas") or {}).keys()),
        "small8_assets": [asset["ts_code"] for asset in universe["assets"]],
        "run_configs": {
            path: {
                "run_name": config["output"]["run_name"],
                "asset_universe_id": config["protocol"]["asset_universe_id"],
                "rankable": config["rankability"]["rankable_in_unified_table"],
                "diagnostic_status": config["rankability"]["diagnostic_status"],
            }
            for path, config in run_configs.items()
        },
    }


def write_manifest(payload: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "protocol_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (OUTPUT_DIR / "protocol_manifest_hashes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256"])
        writer.writeheader()
        for path, digest in sorted(payload["files"].items()):
            writer.writerow({"path": path, "sha256": digest})
    universe = _read_yaml(SMALL8_UNIVERSE)
    (OUTPUT_DIR / "small8_universe_selection_manifest.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "selection_protocol": universe["selection_protocol"],
                "selected_assets": universe["assets"],
                "universe_manifest_hash": payload["files"][str(SMALL8_UNIVERSE)],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OTAR V2 S0 protocol freeze artifacts.")
    parser.add_argument("--write-manifest", action="store_true", help="Write protocol manifest under results/paper_tables.")
    parser.add_argument("--post-freeze", action="store_true", help="Treat hash drift as error (post-freeze mode).")
    args = parser.parse_args()
    payload = validate_protocol(post_freeze=args.post_freeze)
    if args.write_manifest:
        write_manifest(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
