from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from src.data.loader import DataContractError


FUTURE_LABEL_TOKENS = ("future", "forward", "label", "target", "next", "lead", "t+1", "t_plus_1")
WARNING_TOKENS = ("warning", "review", "unknown", "ambiguous", "maybe_leak")
EXECUTION_ONLY_FIELDS = {
    "execution_date",
    "next_valuation_date",
    "execution_price_type",
    "execution_price",
    "tradeable_mask_at_execution",
    "availability_reason_at_execution",
    "return_from_decision_to_execution",
    "holding_simple_return",
    "amount_at_execution",
    "volume_at_execution",
    "vol_at_execution",
    "adv20_at_execution",
    "volatility_20d_at_execution",
    "pre_execution_return",
    "post_execution_return",
    "holding_return",
}
EXECUTION_PHASE_DATA_FIELDS = ("amount", "volume", "vol", "availability", "available")
EXECUTION_PHASE_MARKERS = ("_at_execution", "_after_execution", "post_execution_")
EXECUTION_PHASE_FUTURE_MARKERS = ("next", "lead", "t+1", "t_plus_1", "after", "post", "execution")


def assert_no_future_label_in_features(
    feature_cols: Sequence[str],
    auxiliary_target_cols: Sequence[str] | None = None,
) -> None:
    auxiliary_targets = {_normalize_name(name) for name in (auxiliary_target_cols or [])}
    violations = []
    for feature in feature_cols:
        normalized = _normalize_name(feature)
        if normalized in auxiliary_targets or _contains_future_label_token(normalized):
            violations.append(feature)

    if violations:
        raise DataContractError(
            "ERR_LEAKAGE_FUTURE_LABEL",
            f"ERR_LEAKAGE_FUTURE_LABEL: {sorted(violations)}",
        )


def assert_pca_not_fit_on_validation_or_test(
    fit_dates: Any,
    train_dates: Any | None = None,
    validation_dates: Any | None = None,
    test_dates: Any | None = None,
    *,
    fit_scope: str = "train_only",
) -> None:
    if train_dates is not None and hasattr(train_dates, "train_dates"):
        split = train_dates
        train_dates = split.train_dates
        validation_dates = split.validation_dates
        test_dates = split.test_dates

    if fit_scope != "train_only":
        _raise_pca_scope(f"fit_scope={fit_scope}")

    fit_index = _to_datetime_index(fit_dates)
    train_index = _to_datetime_index(train_dates)
    if fit_index.empty or train_index.empty:
        _raise_pca_scope("empty_fit_or_train_dates")

    non_train_dates = fit_index[~fit_index.isin(train_index)]
    if len(non_train_dates) > 0:
        _raise_pca_scope(_format_dates(non_train_dates))

    validation_index = _to_datetime_index(validation_dates)
    test_index = _to_datetime_index(test_dates)
    leaked_dates = fit_index[fit_index.isin(validation_index) | fit_index.isin(test_index)]
    if len(leaked_dates) > 0:
        _raise_pca_scope(_format_dates(leaked_dates))


def assert_rolling_estimator_uses_past_only(provenance: Any) -> None:
    violations = []
    for record in _iter_records(provenance):
        name = str(record.get("feature_name", record.get("name", "unknown")))
        if bool(record.get("uses_future_data", False)) or bool(record.get("center", False)):
            violations.append(name)
            continue

        feature_date = _record_timestamp(record, ("date", "feature_date", "as_of_date"))
        source_end_date = _record_timestamp(record, ("max_source_date", "window_end_date", "source_end_date"))
        if feature_date is not None and source_end_date is not None and source_end_date > feature_date:
            violations.append(name)

    if violations:
        raise DataContractError(
            "ERR_LEAKAGE_ROLLING_FUTURE",
            f"ERR_LEAKAGE_ROLLING_FUTURE: {sorted(set(violations))}",
        )


def assert_no_execution_field_in_observation(observation: Any) -> None:
    field_names = _extract_field_names(observation)
    violations = [name for name in field_names if _is_execution_only_field(name)]
    if violations:
        raise DataContractError(
            "ERR_LEAKAGE_EXECUTION_FIELD",
            f"ERR_LEAKAGE_EXECUTION_FIELD: {sorted(set(violations))}",
        )


def assert_decision_visibility_contract(
    observation: Any | None = None,
    *,
    market_image: Any | None = None,
    gate_input: Any | None = None,
    feature_window: Any | None = None,
    strategy_state: Any | None = None,
) -> None:
    for payload in (
        observation,
        {"market_image": market_image} if market_image is not None else None,
        {"gate_input": gate_input} if gate_input is not None else None,
        {"feature_window": feature_window} if feature_window is not None else None,
        {"strategy_state": strategy_state} if strategy_state is not None else None,
    ):
        if payload is not None:
            assert_no_execution_field_in_observation(payload)


def audit_feature_provenance(
    feature_name: str,
    config: Mapping[str, Any] | None = None,
    auxiliary_target_cols: Sequence[str] | None = None,
) -> dict[str, Any]:
    feature_audit = _feature_audit_config(config)
    normalized = _normalize_name(feature_name)
    auxiliary_targets = {_normalize_name(name) for name in (auxiliary_target_cols or [])}

    if normalized in auxiliary_targets:
        return _audit_result("dropped", "auxiliary_target", "high", False)

    blacklist_patterns = tuple(
        _normalize_name(pattern)
        for pattern in feature_audit.get("blacklist_patterns", FUTURE_LABEL_TOKENS)
    )
    if any(pattern in normalized for pattern in blacklist_patterns):
        return _audit_result("fail", "blacklist_pattern", "high", False)

    warning_patterns = tuple(
        _normalize_name(pattern)
        for pattern in feature_audit.get("warning_patterns", WARNING_TOKENS)
    )
    if any(pattern in normalized for pattern in warning_patterns):
        warning_policy = feature_audit.get("warning_policy", "keep")
        include_warning = bool(feature_audit.get("include_warning_features", True))
        is_model_feature = warning_policy == "keep" or (warning_policy == "report_only" and include_warning)
        drop_reason = "" if is_model_feature else "warning_policy_drop"
        return _audit_result("warning", drop_reason, "medium", is_model_feature)

    return _audit_result("pass", "", "low", True)


def _raise_pca_scope(detail: str) -> None:
    raise DataContractError(
        "ERR_LEAKAGE_PCA_FIT_SCOPE",
        f"ERR_LEAKAGE_PCA_FIT_SCOPE: {detail}",
    )


def _normalize_name(name: Any) -> str:
    return str(name).strip().lower()


def _contains_future_label_token(name: str) -> bool:
    return any(token in name for token in FUTURE_LABEL_TOKENS)


def _to_datetime_index(values: Any) -> pd.DatetimeIndex:
    if values is None:
        return pd.DatetimeIndex([])
    if isinstance(values, pd.DataFrame):
        if "date" in values.columns:
            values = values["date"]
        else:
            values = values.index
    elif isinstance(values, pd.Series):
        values = values.tolist()
    elif isinstance(values, pd.Timestamp):
        values = [values]
    return pd.DatetimeIndex(pd.to_datetime(list(values)))


def _format_dates(dates: pd.DatetimeIndex) -> list[str]:
    return [date.strftime("%Y-%m-%d") for date in dates.unique()]


def _iter_records(provenance: Any) -> list[dict[str, Any]]:
    if isinstance(provenance, pd.DataFrame):
        return provenance.to_dict("records")
    if isinstance(provenance, Mapping):
        return [dict(provenance)]
    if isinstance(provenance, Sequence) and not isinstance(provenance, (str, bytes)):
        return [dict(item) for item in provenance]
    return []


def _record_timestamp(record: Mapping[str, Any], keys: Sequence[str]) -> pd.Timestamp | None:
    for key in keys:
        if key not in record:
            continue
        value = pd.to_datetime(record[key], errors="coerce")
        if pd.notna(value):
            return pd.Timestamp(value)
    return None


def _extract_field_names(payload: Any) -> list[str]:
    if isinstance(payload, pd.DataFrame):
        return [str(column) for column in payload.columns]
    if isinstance(payload, Mapping):
        names = [str(key) for key in payload]
        for value in payload.values():
            names.extend(_extract_field_names(value))
        return names
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        if all(isinstance(item, str) for item in payload):
            return [str(item) for item in payload]
        names: list[str] = []
        for item in payload:
            names.extend(_extract_field_names(item))
        return names
    return []


def _is_execution_only_field(name: str) -> bool:
    normalized = _normalize_name(name)
    if normalized in EXECUTION_ONLY_FIELDS:
        return True
    if normalized.endswith("_at_execution"):
        return True
    tokens = set(normalized.replace("-", "_").split("_"))
    data_field_hit = bool(tokens & set(EXECUTION_PHASE_DATA_FIELDS))
    return data_field_hit and (
        any(marker in normalized for marker in EXECUTION_PHASE_MARKERS)
        or bool(tokens & set(EXECUTION_PHASE_FUTURE_MARKERS))
        or any(marker in normalized for marker in ("t+1", "t_plus_1"))
    )


def _feature_audit_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is None:
        return {}
    if "feature_audit" in config and isinstance(config["feature_audit"], Mapping):
        return config["feature_audit"]
    return config


def _audit_result(
    status: str,
    drop_reason: str,
    leakage_risk_level: str,
    is_model_feature: bool,
) -> dict[str, Any]:
    return {
        "leakage_check_status": status,
        "drop_reason": drop_reason,
        "leakage_risk_level": leakage_risk_level,
        "is_model_feature": is_model_feature,
    }
