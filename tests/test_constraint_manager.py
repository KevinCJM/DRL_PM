from copy import deepcopy

import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError
from src.envs.constraint_manager import ConstraintManager


def test_simplex_and_availability_projection():
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"]["max_weight"] = 0.70
    config["constraints"]["min_weight"] = 0.05
    raw_weights = np.array([0.80, -0.20, 0.50, 0.40])
    available_mask = np.array([True, True, False, True])

    result = ConstraintManager(config).project(raw_weights, available_mask)

    assert result.projected_weights[2] == 0.0
    assert np.all(result.projected_weights[available_mask] >= 0.05 - 1.0e-12)
    assert np.all(result.projected_weights[available_mask] <= 0.70 + 1.0e-12)
    assert result.projected_weights.sum() == pytest.approx(1.0)
    assert result.projection_distance > 0.0
    assert result.active_constraints == ["availability", "long_only", "simplex", "max_weight", "min_weight"]
    assert {record["constraint"] for record in result.constraint_violations} == {"availability", "long_only"}


def test_no_available_asset_cash_disabled_raises():
    config = deepcopy(DEFAULT_CONFIG)
    config["execution_model"]["cash_enabled"] = False

    with pytest.raises(DataContractError) as error:
        ConstraintManager(config).project(np.array([0.40, 0.60]), np.array([False, False]))

    assert error.value.code == "ERR_CONSTRAINT_NO_AVAILABLE_ASSET"


def test_no_available_asset_cash_enabled_returns_zero_weights():
    config = deepcopy(DEFAULT_CONFIG)
    config["execution_model"]["cash_enabled"] = True

    result = ConstraintManager(config).project(np.array([0.40, 0.60]), np.array([False, False]))

    np.testing.assert_array_equal(result.projected_weights, np.zeros(2))
    assert result.active_constraints == ["availability", "cash"]
    assert result.constraint_violations[0]["reason"] == "no_available_asset_cash_enabled"


def test_infeasible_max_min_are_relaxed_and_recorded():
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"]["max_weight"] = 0.20
    config["constraints"]["min_weight"] = 0.50
    raw_weights = np.array([0.20, 0.30, 0.50])
    available_mask = np.array([True, True, True])

    result = ConstraintManager(config).project(raw_weights, available_mask)

    assert result.projected_weights.sum() == pytest.approx(1.0)
    assert np.all(result.projected_weights <= (1.0 / 3.0) + 1.0e-12)
    relaxations = {
        record["constraint"]: record
        for record in result.constraint_violations
        if record["reason"] == "infeasible_relaxed"
    }
    assert relaxations["max_weight"]["effective_value"] == pytest.approx(1.0 / 3.0)
    assert relaxations["min_weight"]["effective_value"] == 0.0


def test_turnover_limit_requires_reference_weights():
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"]["turnover_limit"] = 0.10

    with pytest.raises(DataContractError) as error:
        ConstraintManager(config).project(np.array([0.80, 0.20]), np.array([True, True]))

    assert error.value.code == "ERR_CONSTRAINT_REFERENCE_WEIGHTS_REQUIRED"


def test_turnover_limit_projects_to_reference_ball():
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"]["turnover_limit"] = 0.10
    raw_weights = np.array([1.00, 0.00, 0.00])
    reference_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])

    result = ConstraintManager(config).project(
        raw_weights,
        np.array([True, True, True]),
        reference_weights=reference_weights,
    )

    turnover = 0.5 * np.sum(np.abs(result.projected_weights - reference_weights))
    assert turnover <= 0.10 + 1.0e-12
    assert result.projected_weights.sum() == pytest.approx(1.0)
    assert "turnover" in result.active_constraints
    assert result.constraint_violations[-1]["constraint"] == "turnover"


def test_hhi_limit_relaxes_to_theoretical_lower_bound():
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"]["hhi_limit"] = 0.20

    result = ConstraintManager(config).project(np.array([1.00, 0.00, 0.00]), np.array([True, True, True]))

    np.testing.assert_allclose(result.projected_weights, np.array([1.0 / 3.0] * 3))
    relaxations = [record for record in result.constraint_violations if record["constraint"] == "hhi"]
    assert relaxations[0]["reason"] == "infeasible_relaxed"
    assert relaxations[0]["effective_value"] == pytest.approx(1.0 / 3.0)


def test_asset_class_metadata_missing_skip_or_fail():
    skip_config = deepcopy(DEFAULT_CONFIG)
    skip_config["constraints"]["asset_class_exposure"] = {"equity": {"max_exposure": 0.60}}

    result = ConstraintManager(skip_config).project(np.array([0.70, 0.30]), np.array([True, True]))

    assert result.constraint_violations[-1]["constraint"] == "asset_class_exposure"
    assert result.constraint_violations[-1]["skip_reason"] == "missing_asset_class_metadata"

    fail_config = deepcopy(skip_config)
    fail_config["constraints"]["asset_class_required"] = True
    with pytest.raises(DataContractError) as error:
        ConstraintManager(fail_config).project(np.array([0.70, 0.30]), np.array([True, True]))
    assert error.value.code == "ERR_CONSTRAINT_ASSET_CLASS_METADATA_REQUIRED"


def test_asset_class_mapping_uses_asset_universe_ts_code():
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"]["asset_class_exposure"] = {"equity": {"max_exposure": 0.60}}
    config["constraints"]["asset_class_mapping"] = {
        "510300.SH": "equity",
        "159915.SZ": "equity",
        "511010.SH": "bond",
    }
    asset_universe = {"ts_code": ["510300.SH", "159915.SZ", "511010.SH"]}

    result = ConstraintManager(config).project(
        np.array([0.70, 0.20, 0.10]),
        np.array([True, True, True]),
        asset_universe=asset_universe,
    )

    assert result.projected_weights[:2].sum() <= 0.60 + 1.0e-12
    assert result.projected_weights.sum() == pytest.approx(1.0)
    assert result.constraint_violations[-1]["constraint"] == "asset_class_exposure"


def test_empty_asset_class_mapping_falls_back_to_asset_universe_pool():
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"]["asset_class_exposure"] = {"equity": {"max_exposure": 0.60}}
    asset_universe = {"pool": ["equity", "equity", "bond"]}

    result = ConstraintManager(config).project(
        np.array([0.70, 0.20, 0.10]),
        np.array([True, True, True]),
        asset_universe=asset_universe,
    )

    assert result.projected_weights[:2].sum() <= 0.60 + 1.0e-12
    assert result.projected_weights.sum() == pytest.approx(1.0)


def test_soft_penalty_records_penalty_without_soft_projection():
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"]["hhi_limit"] = 0.50
    config["constraints"]["soft_penalty_enabled"] = True

    result = ConstraintManager(config).project(np.array([1.00, 0.00, 0.00]), np.array([True, True, True]))

    np.testing.assert_allclose(result.projected_weights, np.array([1.00, 0.00, 0.00]), atol=1.0e-12)
    hhi_record = result.constraint_violations[-1]
    assert hhi_record["constraint"] == "hhi"
    assert hhi_record["constraint_method"] == "soft_penalty"
    assert hhi_record["penalty"] == pytest.approx(0.50)
    assert "lagrangian_violation_scalar" not in hhi_record


def test_ppo_lagrangian_records_violation_scalar_without_soft_projection():
    config = deepcopy(DEFAULT_CONFIG)
    config["constraints"]["hhi_limit"] = 0.50
    config["constraints"]["ppo_lagrangian_enabled"] = True

    result = ConstraintManager(config).project(np.array([1.00, 0.00, 0.00]), np.array([True, True, True]))

    np.testing.assert_allclose(result.projected_weights, np.array([1.00, 0.00, 0.00]), atol=1.0e-12)
    hhi_record = result.constraint_violations[-1]
    assert hhi_record["constraint"] == "hhi"
    assert hhi_record["constraint_method"] == "ppo_lagrangian"
    assert hhi_record["lagrangian_violation_scalar"] == pytest.approx(0.50)
