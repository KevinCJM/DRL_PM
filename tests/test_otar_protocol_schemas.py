from __future__ import annotations

from pathlib import Path

import yaml

from scripts.validate_otar_s0_protocol import (
    ACTOR_GRID,
    CQR_GRID,
    FORBIDDEN_FIELDS,
    REQUIRED_ACTOR_KEYS,
    REQUIRED_CQR_KEYS,
    SCHEMA_FIXTURE,
    SMALL8_UNIVERSE,
    validate_protocol,
)
from src.config import ConfigLoader, PROJECT_ROOT


def _read_yaml(path: Path) -> dict:
    with (PROJECT_ROOT / path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def test_otar_s0_protocol_validator_passes():
    payload = validate_protocol()

    assert payload["status"] == "success"
    assert payload["protocol_id"] == "otar_v2_s0_20260605"
    assert set(payload["small8_assets"]) == {
        "510300.SH",
        "510500.SH",
        "510050.SH",
        "159915.SZ",
        "159920.SZ",
        "513100.SH",
        "518880.SH",
        "511010.SH",
    }


def test_otar_cqr_and_actor_grids_freeze_required_candidates():
    cqr = _read_yaml(CQR_GRID)
    actor = _read_yaml(ACTOR_GRID)

    assert REQUIRED_CQR_KEYS.issubset(cqr["candidate_values"])
    assert REQUIRED_ACTOR_KEYS.issubset(actor["candidate_values"])
    assert cqr["no_test_peeking"] is True
    assert actor["no_test_peeking"] is True
    assert cqr["allowed_target"] == "gross simple-return return-to-go distribution"
    assert "PPO shaped-reward distribution" in cqr["forbidden_targets"]


def test_otar_schema_fixture_declares_required_attrs_and_forbidden_aliases():
    fixture = _read_yaml(SCHEMA_FIXTURE)
    assert FORBIDDEN_FIELDS.issubset(set(fixture["forbidden_new_output_fields"]))

    seen_fields = set()
    for schema in fixture["schemas"].values():
        for field in schema["fields"]:
            seen_fields.add(field["field_name"])
            assert {
                "field_name",
                "dtype",
                "shape",
                "unit",
                "level",
                "required",
                "missing_value_policy",
                "forbidden_aliases",
            }.issubset(field)

    assert not (FORBIDDEN_FIELDS & seen_fields)
    assert "pred_candidate_lower_tail_loss" in seen_fields
    assert "pred_hold_lower_tail_loss" in seen_fields
    assert "realized_gate_action_soft_tail_proxy" in seen_fields


def test_otar_small8_universe_blocks_future_performance_selection():
    universe = _read_yaml(SMALL8_UNIVERSE)

    assert len(universe["assets"]) == 8
    assert universe["selection_protocol"]["future_performance_screening_used"] is False
    assert universe["selection_protocol"]["selection_date"] == "2021-12-31"


def test_otar_run_configs_use_daily_gate_and_validation_selection():
    for relative in (
        "configs/paper/otar_small8_smoke.yaml",
        "configs/paper/otar_small8_pilot.yaml",
        "configs/paper/otar_core13_robustness.yaml",
    ):
        config = ConfigLoader.load(PROJECT_ROOT / relative)
        assert config["protocol"]["protocol_id"] == "otar_v2_s0_20260605"
        assert config["rebalance"]["mode"] == "daily"
        assert config["execution_activity"]["protocol"] == "daily_gate_with_cost_constraint"
        assert config["execution_activity"]["scheduler_blocks_model_actions"] is False
        assert config["execution_activity"]["activity_gate_enforced"] is True
        assert config["hpo"]["selection_split"] == "validation"
        assert config["hpo"]["final_report_split"] == "test"
        assert config["training"]["checkpoint_include_replay_buffer"] is False
