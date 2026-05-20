from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError


WEIGHT_SUM_EPS = 1.0e-12
PROJECTION_ITERATIONS = 100


@dataclass
class ConstraintResult:
    projected_weights: np.ndarray
    projection_distance: float
    constraint_violations: list[dict[str, Any]] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.projected_weights = _array_1d("projected_weights", self.projected_weights)
        self.projection_distance = _finite_float("projection_distance", self.projection_distance)
        self.constraint_violations = [dict(record) for record in self.constraint_violations]
        self.active_constraints = [str(value) for value in self.active_constraints]


class ConstraintManager:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.raw_config = config or DEFAULT_CONFIG
        self.constraint_config = _constraint_config(config)
        self.execution_config = _execution_model_config(config)

    def project(
        self,
        raw_weights: np.ndarray,
        available_mask: np.ndarray,
        reference_weights: np.ndarray | None = None,
        *,
        asset_universe: Any | None = None,
        asset_ids: Any | None = None,
    ) -> ConstraintResult:
        raw = _array_1d("raw_weights", raw_weights)
        available = _bool_array("available_mask", available_mask, raw.shape)
        active_constraints: list[str] = []
        violations: list[dict[str, Any]] = []
        constraint_method = _constraint_method(self.constraint_config)

        if not available.any():
            if bool(self.execution_config.get("cash_enabled", False)):
                projected = np.zeros_like(raw, dtype=float)
                _append_unique(active_constraints, "availability")
                _append_unique(active_constraints, "cash")
                violations.append(
                    {
                        "constraint": "availability",
                        "reason": "no_available_asset_cash_enabled",
                        "available_count": 0,
                    }
                )
                return _result(raw, projected, violations, active_constraints)
            raise DataContractError(
                "ERR_CONSTRAINT_NO_AVAILABLE_ASSET",
                "ERR_CONSTRAINT_NO_AVAILABLE_ASSET: available_mask",
            )

        projected = raw.astype(float, copy=True)
        if np.any(np.abs(projected[~available]) > WEIGHT_SUM_EPS):
            violations.append(
                {
                    "constraint": "availability",
                    "reason": "unavailable_weight_zeroed",
                    "asset_indices": np.flatnonzero(~available).astype(int).tolist(),
                }
            )
        projected[~available] = 0.0
        _append_unique(active_constraints, "availability")

        if bool(self.constraint_config.get("long_only", True)):
            negative_mask = available & (projected < 0.0)
            if negative_mask.any():
                violations.append(
                    {
                        "constraint": "long_only",
                        "reason": "negative_weight_clipped",
                        "asset_indices": np.flatnonzero(negative_mask).astype(int).tolist(),
                    }
                )
                projected[negative_mask] = 0.0
            _append_unique(active_constraints, "long_only")

        if bool(self.constraint_config.get("simplex", True)):
            projected = _simplex_projection(projected, available, violations)
            _append_unique(active_constraints, "simplex")

        n_available = int(available.sum())
        max_weight = _positive_config_float("max_weight", self.constraint_config.get("max_weight", 1.0))
        min_weight = _non_negative_config_float("min_weight", self.constraint_config.get("min_weight", 0.0))
        if max_weight * n_available < 1.0 - WEIGHT_SUM_EPS:
            relaxed_max = 1.0 / n_available
            violations.append(
                {
                    "constraint": "max_weight",
                    "reason": "infeasible_relaxed",
                    "configured_value": max_weight,
                    "effective_value": relaxed_max,
                    "available_count": n_available,
                }
            )
            max_weight = relaxed_max
        if min_weight * n_available > 1.0 + WEIGHT_SUM_EPS:
            violations.append(
                {
                    "constraint": "min_weight",
                    "reason": "infeasible_relaxed",
                    "configured_value": min_weight,
                    "effective_value": 0.0,
                    "available_count": n_available,
                }
            )
            min_weight = 0.0
        if min_weight > max_weight:
            violations.append(
                {
                    "constraint": "min_weight",
                    "reason": "min_exceeds_max_relaxed",
                    "configured_value": min_weight,
                    "effective_value": 0.0,
                    "max_weight": max_weight,
                }
            )
            min_weight = 0.0

        if bool(self.constraint_config.get("simplex", True)):
            bounded = _bounded_simplex_projection(projected[available], min_weight, max_weight)
            if np.any(bounded > max_weight + WEIGHT_SUM_EPS):
                raise DataContractError(
                    "ERR_CONSTRAINT_PROJECTION_FAILED",
                    "ERR_CONSTRAINT_PROJECTION_FAILED: max_weight",
                )
            if np.any(bounded < min_weight - WEIGHT_SUM_EPS):
                raise DataContractError(
                    "ERR_CONSTRAINT_PROJECTION_FAILED",
                    "ERR_CONSTRAINT_PROJECTION_FAILED: min_weight",
                )
            projected = np.zeros_like(projected, dtype=float)
            projected[available] = bounded
            _append_unique(active_constraints, "max_weight")
            _append_unique(active_constraints, "min_weight")
        else:
            above_max = available & (projected > max_weight)
            below_min = available & (projected < min_weight)
            if above_max.any():
                projected[above_max] = max_weight
                _append_unique(active_constraints, "max_weight")
            if below_min.any():
                projected[below_min] = min_weight
                _append_unique(active_constraints, "min_weight")

        projected[~available] = 0.0
        reference = self._reference_weights(reference_weights, raw.shape)
        projected = _apply_turnover_limit(
            projected,
            available,
            reference,
            self.constraint_config,
            constraint_method,
            min_weight,
            max_weight,
            violations,
            active_constraints,
        )
        projected = _apply_hhi_limit(
            projected,
            available,
            self.constraint_config,
            constraint_method,
            violations,
            active_constraints,
        )
        projected = _apply_asset_class_exposure(
            projected,
            available,
            self.constraint_config,
            constraint_method,
            min_weight,
            max_weight,
            violations,
            active_constraints,
            asset_universe=asset_universe,
            asset_ids=asset_ids,
        )
        projected[~available] = 0.0
        return _result(raw, projected, violations, active_constraints)

    def _reference_weights(self, reference_weights: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
        turnover_limit = self.constraint_config.get("turnover_limit")
        if turnover_limit is None:
            return None
        if reference_weights is None:
            raise DataContractError(
                "ERR_CONSTRAINT_REFERENCE_WEIGHTS_REQUIRED",
                "ERR_CONSTRAINT_REFERENCE_WEIGHTS_REQUIRED: reference_weights",
            )
        return _array_1d("reference_weights", reference_weights, shape)


def _constraint_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_CONFIG["constraints"]
    if config is None:
        return dict(source)
    if "constraints" in config:
        return {**source, **dict(config["constraints"])}
    return {**source, **dict(config)}


def _execution_model_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_CONFIG["execution_model"]
    if config is None or "execution_model" not in config:
        return dict(source)
    return {**source, **dict(config["execution_model"])}


def _constraint_method(config: Mapping[str, Any]) -> str:
    if bool(config.get("ppo_lagrangian_enabled", False)):
        return "ppo_lagrangian"
    if bool(config.get("soft_penalty_enabled", False)):
        return "soft_penalty"
    method = str(config.get("constraint_method", "hard_projection"))
    if method not in {"hard_projection", "soft_penalty", "ppo_lagrangian"}:
        raise DataContractError(
            "ERR_CONFIG_INVALID_CONSTRAINT",
            "ERR_CONFIG_INVALID_CONSTRAINT: constraints.constraint_method",
        )
    return method


def _simplex_projection(
    weights: np.ndarray,
    available: np.ndarray,
    violations: list[dict[str, Any]],
) -> np.ndarray:
    projected = weights.astype(float, copy=True)
    total = float(np.sum(projected[available]))
    if total <= WEIGHT_SUM_EPS:
        projected[available] = 1.0 / int(available.sum())
        violations.append(
            {
                "constraint": "simplex",
                "reason": "non_positive_available_sum_equal_weight",
                "available_count": int(available.sum()),
            }
        )
        return projected
    projected[available] = projected[available] / total
    return projected


def _bounded_simplex_projection(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if values.size == 0:
        raise DataContractError("ERR_CONSTRAINT_SHAPE_MISMATCH", "ERR_CONSTRAINT_SHAPE_MISMATCH: available_mask")
    if upper < lower:
        raise DataContractError("ERR_CONFIG_INVALID_CONSTRAINT", "ERR_CONFIG_INVALID_CONSTRAINT: constraints.max_weight")
    if lower * values.size > 1.0 + WEIGHT_SUM_EPS or upper * values.size < 1.0 - WEIGHT_SUM_EPS:
        raise DataContractError(
            "ERR_CONSTRAINT_INFEASIBLE",
            "ERR_CONSTRAINT_INFEASIBLE: constraints.min_weight/max_weight",
        )

    low = float(np.min(values - upper) - 1.0)
    high = float(np.max(values - lower) + 1.0)
    for _ in range(PROJECTION_ITERATIONS):
        theta = (low + high) / 2.0
        total = float(np.sum(np.clip(values - theta, lower, upper)))
        if total > 1.0:
            low = theta
        else:
            high = theta
    projected = np.clip(values - (low + high) / 2.0, lower, upper)
    residual = 1.0 - float(np.sum(projected))
    adjustable = (projected > lower + WEIGHT_SUM_EPS) & (projected < upper - WEIGHT_SUM_EPS)
    if adjustable.any():
        projected[adjustable] += residual / int(adjustable.sum())
    else:
        index = int(np.argmax(upper - projected)) if residual > 0.0 else int(np.argmax(projected - lower))
        projected[index] = float(np.clip(projected[index] + residual, lower, upper))
    return projected


def _apply_turnover_limit(
    weights: np.ndarray,
    available: np.ndarray,
    reference_weights: np.ndarray | None,
    config: Mapping[str, Any],
    constraint_method: str,
    min_weight: float,
    max_weight: float,
    violations: list[dict[str, Any]],
    active_constraints: list[str],
) -> np.ndarray:
    configured_limit = config.get("turnover_limit")
    if configured_limit is None:
        return weights
    _append_unique(active_constraints, "turnover")
    limit = _non_negative_config_float("turnover_limit", configured_limit)
    turnover_before = _turnover(weights, reference_weights)
    violation_scalar = max(0.0, turnover_before - limit)
    if violation_scalar <= WEIGHT_SUM_EPS:
        return weights

    record = {
        "constraint": "turnover",
        "limit": limit,
        "turnover_before": turnover_before,
        "violation_scalar": violation_scalar,
    }
    if constraint_method != "hard_projection":
        record["reason"] = "soft_violation"
        _append_method_fields(record, constraint_method, violation_scalar)
        violations.append(record)
        return weights

    anchor = _reference_anchor(reference_weights, available, min_weight, max_weight, config)
    anchor_turnover = _turnover(anchor, reference_weights)
    if anchor_turnover > limit + WEIGHT_SUM_EPS:
        record.update(
            {
                "reason": "minimum_turnover_exceeds_limit",
                "turnover_after": anchor_turnover,
            }
        )
        violations.append(record)
        return anchor

    if turnover_before <= anchor_turnover + WEIGHT_SUM_EPS:
        projected = anchor
    else:
        fraction = (limit - anchor_turnover) / (turnover_before - anchor_turnover)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        projected = anchor + fraction * (weights - anchor)
    projected[~available] = 0.0
    record.update(
        {
            "reason": "projected_to_turnover_limit",
            "turnover_after": _turnover(projected, reference_weights),
        }
    )
    violations.append(record)
    return projected


def _apply_hhi_limit(
    weights: np.ndarray,
    available: np.ndarray,
    config: Mapping[str, Any],
    constraint_method: str,
    violations: list[dict[str, Any]],
    active_constraints: list[str],
) -> np.ndarray:
    configured_limit = config.get("hhi_limit")
    if configured_limit is None:
        return weights
    _append_unique(active_constraints, "hhi")
    limit = _positive_config_float("hhi_limit", configured_limit)
    n_available = int(available.sum())
    lower_bound = 1.0 / n_available
    if limit < lower_bound - WEIGHT_SUM_EPS:
        violations.append(
            {
                "constraint": "hhi",
                "reason": "infeasible_relaxed",
                "configured_value": limit,
                "effective_value": lower_bound,
                "available_count": n_available,
            }
        )
        limit = lower_bound

    hhi_before = _hhi(weights)
    violation_scalar = max(0.0, hhi_before - limit)
    if violation_scalar <= WEIGHT_SUM_EPS:
        return weights

    record = {
        "constraint": "hhi",
        "limit": limit,
        "hhi_before": hhi_before,
        "violation_scalar": violation_scalar,
    }
    if constraint_method != "hard_projection":
        record["reason"] = "soft_violation"
        _append_method_fields(record, constraint_method, violation_scalar)
        violations.append(record)
        return weights

    equal = np.zeros_like(weights, dtype=float)
    equal[available] = 1.0 / n_available
    if limit <= lower_bound + WEIGHT_SUM_EPS:
        projected = equal
    else:
        low = 0.0
        high = 1.0
        for _ in range(PROJECTION_ITERATIONS):
            fraction = (low + high) / 2.0
            candidate = equal + fraction * (weights - equal)
            if _hhi(candidate) <= limit:
                low = fraction
            else:
                high = fraction
        projected = equal + low * (weights - equal)
    projected[~available] = 0.0
    record.update(
        {
            "reason": "projected_to_hhi_limit",
            "hhi_after": _hhi(projected),
        }
    )
    violations.append(record)
    return projected


def _apply_asset_class_exposure(
    weights: np.ndarray,
    available: np.ndarray,
    config: Mapping[str, Any],
    constraint_method: str,
    min_weight: float,
    max_weight: float,
    violations: list[dict[str, Any]],
    active_constraints: list[str],
    *,
    asset_universe: Any | None,
    asset_ids: Any | None,
) -> np.ndarray:
    exposure_config = config.get("asset_class_exposure") or {}
    if not exposure_config:
        return weights
    _append_unique(active_constraints, "asset_class_exposure")
    classes = _asset_classes(config, weights.shape[0], asset_universe, asset_ids)
    if classes is None or any(value is None or str(value) == "" for value in classes):
        if bool(config.get("asset_class_required", False)):
            raise DataContractError(
                "ERR_CONSTRAINT_ASSET_CLASS_METADATA_REQUIRED",
                "ERR_CONSTRAINT_ASSET_CLASS_METADATA_REQUIRED: constraints.asset_class_exposure",
            )
        violations.append(
            {
                "constraint": "asset_class_exposure",
                "reason": "skipped",
                "skip_reason": "missing_asset_class_metadata",
            }
        )
        return weights

    rules = _asset_class_rules(exposure_config)
    exposure_violations = _asset_class_violation_records(weights, available, classes, rules)
    if not exposure_violations:
        return weights
    total_violation = float(sum(record["violation_scalar"] for record in exposure_violations))
    if constraint_method != "hard_projection":
        for record in exposure_violations:
            record["reason"] = "soft_violation"
            _append_method_fields(record, constraint_method, record["violation_scalar"])
            violations.append(record)
        return weights

    projected, residual_records = _project_asset_class_limits(
        weights,
        available,
        classes,
        rules,
        min_weight,
        max_weight,
    )
    for record in exposure_violations:
        record["reason"] = "projected_to_asset_class_exposure"
        violations.append(record)
    for record in residual_records:
        record["violation_scalar"] = total_violation
        violations.append(record)
    return projected


def _turnover(weights: np.ndarray, reference_weights: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(weights - reference_weights)))


def _hhi(weights: np.ndarray) -> float:
    return float(np.sum(np.square(weights)))


def _reference_anchor(
    reference_weights: np.ndarray,
    available: np.ndarray,
    min_weight: float,
    max_weight: float,
    config: Mapping[str, Any],
) -> np.ndarray:
    anchor = reference_weights.astype(float, copy=True)
    anchor[~available] = 0.0
    if bool(config.get("long_only", True)):
        anchor[available & (anchor < 0.0)] = 0.0
    if bool(config.get("simplex", True)):
        anchor = _simplex_projection(anchor, available, [])
        bounded = _bounded_simplex_projection(anchor[available], min_weight, max_weight)
        result = np.zeros_like(anchor, dtype=float)
        result[available] = bounded
        return result
    anchor[available] = np.clip(anchor[available], min_weight, max_weight)
    return anchor


def _append_method_fields(record: dict[str, Any], constraint_method: str, violation_scalar: float) -> None:
    record["constraint_method"] = constraint_method
    if constraint_method == "soft_penalty":
        record["penalty"] = violation_scalar
    elif constraint_method == "ppo_lagrangian":
        record["lagrangian_violation_scalar"] = violation_scalar


def _asset_classes(
    config: Mapping[str, Any],
    n_assets: int,
    asset_universe: Any | None,
    asset_ids: Any | None,
) -> list[str | None] | None:
    mapping = config.get("asset_class_mapping") or None
    ids = asset_ids if asset_ids is not None else config.get("asset_ids", config.get("ts_codes"))
    if mapping is not None:
        if isinstance(mapping, Mapping):
            if ids is None:
                ids = _asset_ids_from_universe(asset_universe, n_assets)
            if ids is not None:
                asset_id_list = list(ids)
                if len(asset_id_list) != n_assets:
                    return None
                return [_mapping_value(mapping, value) for value in asset_id_list]
            return [_mapping_value(mapping, index) for index in range(n_assets)]
        values = list(mapping)
        if len(values) != n_assets:
            return None
        return [None if value is None else str(value) for value in values]

    if asset_universe is None:
        return None
    for column in ("pool", "type"):
        try:
            values = asset_universe[column]
        except (KeyError, TypeError):
            continue
        values_list = list(values)
        if len(values_list) != n_assets:
            return None
        if any(value is not None and str(value) != "" for value in values_list):
            return [None if value is None else str(value) for value in values_list]
    return None


def _asset_ids_from_universe(asset_universe: Any | None, n_assets: int) -> list[Any] | None:
    if asset_universe is None:
        return None
    try:
        values = asset_universe["ts_code"]
    except (KeyError, TypeError):
        return None
    values_list = list(values)
    if len(values_list) != n_assets:
        return None
    return values_list


def _mapping_value(mapping: Mapping[Any, Any], key: Any) -> str | None:
    value = mapping.get(key)
    if value is None:
        value = mapping.get(str(key))
    return None if value is None else str(value)


def _asset_class_rules(exposure_config: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
    rules: dict[str, dict[str, float | None]] = {}
    for asset_class, raw_rule in exposure_config.items():
        if isinstance(raw_rule, Mapping):
            min_value = raw_rule.get("min_exposure", raw_rule.get("min"))
            max_value = raw_rule.get("max_exposure", raw_rule.get("max"))
        else:
            min_value = None
            max_value = raw_rule
        rules[str(asset_class)] = {
            "min": None if min_value is None else _non_negative_config_float("asset_class_exposure.min", min_value),
            "max": None if max_value is None else _non_negative_config_float("asset_class_exposure.max", max_value),
        }
    return rules


def _asset_class_violation_records(
    weights: np.ndarray,
    available: np.ndarray,
    classes: list[str | None],
    rules: Mapping[str, Mapping[str, float | None]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for asset_class, rule in rules.items():
        class_mask = np.array([value == asset_class for value in classes], dtype=bool) & available
        exposure = float(np.sum(weights[class_mask]))
        min_exposure = rule.get("min")
        max_exposure = rule.get("max")
        if min_exposure is not None and exposure < min_exposure - WEIGHT_SUM_EPS:
            records.append(
                {
                    "constraint": "asset_class_exposure",
                    "asset_class": asset_class,
                    "bound": "min_exposure",
                    "configured_value": min_exposure,
                    "exposure": exposure,
                    "violation_scalar": float(min_exposure - exposure),
                }
            )
        if max_exposure is not None and exposure > max_exposure + WEIGHT_SUM_EPS:
            records.append(
                {
                    "constraint": "asset_class_exposure",
                    "asset_class": asset_class,
                    "bound": "max_exposure",
                    "configured_value": max_exposure,
                    "exposure": exposure,
                    "violation_scalar": float(exposure - max_exposure),
                }
            )
    return records


def _project_asset_class_limits(
    weights: np.ndarray,
    available: np.ndarray,
    classes: list[str | None],
    rules: Mapping[str, Mapping[str, float | None]],
    min_weight: float,
    max_weight: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    projected = weights.astype(float, copy=True)
    residual_records: list[dict[str, Any]] = []
    for asset_class, rule in rules.items():
        class_mask = np.array([value == asset_class for value in classes], dtype=bool) & available
        if not class_mask.any():
            continue
        max_exposure = rule.get("max")
        if max_exposure is not None:
            exposure = float(np.sum(projected[class_mask]))
            if exposure > max_exposure + WEIGHT_SUM_EPS:
                reduction = exposure - max_exposure
                projected[class_mask] *= max_exposure / exposure if exposure > WEIGHT_SUM_EPS else 0.0
                residual = _add_mass(projected, available & ~class_mask, reduction, max_weight)
                if residual > WEIGHT_SUM_EPS:
                    _add_mass(projected, class_mask, residual, max_weight)
                    residual_records.append(
                        {
                            "constraint": "asset_class_exposure",
                            "reason": "max_exposure_residual",
                            "asset_class": asset_class,
                            "residual": residual,
                        }
                    )
        min_exposure = rule.get("min")
        if min_exposure is not None:
            exposure = float(np.sum(projected[class_mask]))
            if exposure < min_exposure - WEIGHT_SUM_EPS:
                deficit = min_exposure - exposure
                class_capacity = _mass_add_capacity(projected, class_mask, max_weight)
                donor_capacity = _mass_remove_capacity(projected, available & ~class_mask, min_weight)
                transfer = min(deficit, class_capacity, donor_capacity)
                removed = _remove_mass(projected, available & ~class_mask, transfer, min_weight)
                residual = transfer - removed
                if removed > WEIGHT_SUM_EPS:
                    residual = max(residual, _add_mass(projected, class_mask, removed, max_weight))
                residual += deficit - transfer
                if residual > WEIGHT_SUM_EPS:
                    residual_records.append(
                        {
                            "constraint": "asset_class_exposure",
                            "reason": "min_exposure_residual",
                            "asset_class": asset_class,
                            "residual": residual,
                        }
                    )
    projected[~available] = 0.0
    return projected, residual_records


def _mass_add_capacity(weights: np.ndarray, mask: np.ndarray, max_weight: float) -> float:
    return float(np.sum(np.where(mask, np.maximum(max_weight - weights, 0.0), 0.0)))


def _mass_remove_capacity(weights: np.ndarray, mask: np.ndarray, min_weight: float) -> float:
    return float(np.sum(np.where(mask, np.maximum(weights - min_weight, 0.0), 0.0)))


def _add_mass(weights: np.ndarray, mask: np.ndarray, amount: float, max_weight: float) -> float:
    if amount <= WEIGHT_SUM_EPS:
        return 0.0
    capacity = np.where(mask, np.maximum(max_weight - weights, 0.0), 0.0)
    total_capacity = float(np.sum(capacity))
    added = min(amount, total_capacity)
    if added > WEIGHT_SUM_EPS and total_capacity > WEIGHT_SUM_EPS:
        weights += capacity / total_capacity * added
    return amount - added


def _remove_mass(weights: np.ndarray, mask: np.ndarray, amount: float, min_weight: float) -> float:
    if amount <= WEIGHT_SUM_EPS:
        return 0.0
    capacity = np.where(mask, np.maximum(weights - min_weight, 0.0), 0.0)
    total_capacity = float(np.sum(capacity))
    removed = min(amount, total_capacity)
    if removed > WEIGHT_SUM_EPS and total_capacity > WEIGHT_SUM_EPS:
        weights -= capacity / total_capacity * removed
    return removed


def _result(
    raw_weights: np.ndarray,
    projected_weights: np.ndarray,
    violations: list[dict[str, Any]],
    active_constraints: list[str],
) -> ConstraintResult:
    return ConstraintResult(
        projected_weights=projected_weights,
        projection_distance=float(np.linalg.norm(projected_weights - raw_weights)),
        constraint_violations=violations,
        active_constraints=active_constraints,
    )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _array_1d(name: str, values: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_CONSTRAINT_SHAPE_MISMATCH", f"ERR_CONSTRAINT_SHAPE_MISMATCH: {name}") from exc
    if array.ndim != 1:
        raise DataContractError("ERR_CONSTRAINT_SHAPE_MISMATCH", f"ERR_CONSTRAINT_SHAPE_MISMATCH: {name}")
    if shape is not None and array.shape != shape:
        raise DataContractError("ERR_CONSTRAINT_SHAPE_MISMATCH", f"ERR_CONSTRAINT_SHAPE_MISMATCH: {name}")
    if not np.isfinite(array).all():
        raise DataContractError("ERR_CONSTRAINT_NON_FINITE", f"ERR_CONSTRAINT_NON_FINITE: {name}")
    return array


def _bool_array(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape != shape:
        raise DataContractError("ERR_CONSTRAINT_SHAPE_MISMATCH", f"ERR_CONSTRAINT_SHAPE_MISMATCH: {name}")
    return array.astype(bool)


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_CONSTRAINT_NON_FINITE", f"ERR_CONSTRAINT_NON_FINITE: {name}") from exc
    if not np.isfinite(result):
        raise DataContractError("ERR_CONSTRAINT_NON_FINITE", f"ERR_CONSTRAINT_NON_FINITE: {name}")
    return result


def _non_negative_config_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise DataContractError("ERR_CONFIG_INVALID_CONSTRAINT", f"ERR_CONFIG_INVALID_CONSTRAINT: constraints.{name}")
    return result


def _positive_config_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise DataContractError("ERR_CONFIG_INVALID_CONSTRAINT", f"ERR_CONFIG_INVALID_CONSTRAINT: constraints.{name}")
    return result


__all__ = [
    "ConstraintManager",
    "ConstraintResult",
    "WEIGHT_SUM_EPS",
]
