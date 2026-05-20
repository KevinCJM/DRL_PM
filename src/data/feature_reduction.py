from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

from src.data.leakage_checks import (
    FUTURE_LABEL_TOKENS,
    assert_no_future_label_in_features,
    assert_pca_not_fit_on_validation_or_test,
)
from src.data.loader import DataContractError


IDENTIFIER_COLUMNS = {"date", "ts_code"}
AVAILABILITY_COLUMNS = {"availability_mask", "available_mask", "is_available"}
PORTFOLIO_STATE_COLUMNS = {
    "cash_weight",
    "current_weight",
    "current_weights",
    "holding_weight",
    "nav",
    "portfolio_state",
    "portfolio_value",
    "prev_weight",
    "target_weight",
}
PORTFOLIO_STATE_PREFIXES = (
    "cash_",
    "current_weight",
    "holding_",
    "portfolio_",
    "prev_weight",
    "target_weight",
)
DEFAULT_FEATURE_REDUCTION_CONFIG: dict[str, Any] = {
    "imputer": {
        "strategy": "median",
        "fit_scope": "train_only",
    },
    "winsorize": {
        "enabled": True,
        "lower_quantile": 0.005,
        "upper_quantile": 0.995,
    },
    "feature_selection": {
        "enabled": False,
        "variance_threshold": 1.0e-8,
        "correlation_threshold": 0.98,
        "max_features": 512,
    },
    "pca": {
        "enabled": True,
        "explained_variance": 0.95,
        "fixed_components": None,
        "fit_scope": "train_only",
    },
}
INPUT_MATRIX_REDUCTION_RULES: dict[str, dict[str, bool]] = {
    "M0": {"pca": False, "feature_selection": False},
    "M1": {"pca": False, "feature_selection": False},
    "M2": {"pca": False, "feature_selection": False},
    "M3": {"pca": False, "feature_selection": False},
    "M4": {"pca": False, "feature_selection": False},
    "M5": {"pca": False, "feature_selection": False},
    "M6": {"pca": True, "feature_selection": False},
    "M7": {"pca": True, "feature_selection": True},
}


class FeatureReductionPipeline:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.raw_config = config or {}
        self.input_matrix_id = _input_matrix_id(config)
        self.config = _feature_reduction_config(config)
        self.auxiliary_config = _auxiliary_config(config)
        self.is_fitted = False

    def fit(
        self,
        train_panel: pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
        split: Any | None = None,
        *,
        train_dates: Any | None = None,
        validation_dates: Any | None = None,
        test_dates: Any | None = None,
        fit_scope: str | None = None,
        auxiliary_target_cols: Sequence[str] | None = None,
        wide_log_return: pd.DataFrame | None = None,
    ) -> "FeatureReductionPipeline":
        if train_panel.empty:
            raise DataContractError("ERR_SPLIT_EMPTY", "ERR_SPLIT_EMPTY: empty train_panel")

        fit_dates = _panel_dates(train_panel)
        scope = fit_scope or self._fit_scope()
        split_or_train_dates = split if split is not None else train_dates
        if split_or_train_dates is None:
            split_or_train_dates = fit_dates
        assert_pca_not_fit_on_validation_or_test(
            fit_dates,
            split_or_train_dates,
            validation_dates,
            test_dates,
            fit_scope=scope,
        )

        self.passthrough_cols_ = _passthrough_cols(train_panel.columns)
        self.input_feature_cols_ = _resolve_feature_cols(train_panel, feature_cols, auxiliary_target_cols)
        assert_no_future_label_in_features(self.input_feature_cols_, auxiliary_target_cols)
        if not self.input_feature_cols_:
            raise DataContractError("ERR_FEATURE_REDUCTION_EMPTY", "ERR_FEATURE_REDUCTION_EMPTY: feature_cols")

        train_features = _numeric_frame(train_panel, self.input_feature_cols_)
        self.medians_ = train_features.median(axis=0).fillna(0.0)
        imputed = train_features.fillna(self.medians_)

        self.winsor_lower_, self.winsor_upper_ = self._fit_winsor_bounds(imputed)
        winsorized = self._apply_winsor_bounds(imputed)

        self.scaler_ = StandardScaler()
        scaled_values = self.scaler_.fit_transform(winsorized.to_numpy(dtype=float, copy=True))
        scaled = pd.DataFrame(scaled_values, columns=self.input_feature_cols_, index=train_features.index)

        train_dates_index = _train_dates_index(split_or_train_dates, fit_dates)
        self.selected_feature_cols_ = self._fit_selector(
            scaled,
            train_features,
            train_panel,
            train_dates_index,
            wide_log_return,
        )
        if not self.selected_feature_cols_:
            raise DataContractError("ERR_FEATURE_REDUCTION_EMPTY", "ERR_FEATURE_REDUCTION_EMPTY: selected_feature_cols")
        selected = scaled.loc[:, self.selected_feature_cols_]

        self.pca_ = None
        self.output_feature_cols_ = list(self.selected_feature_cols_)
        if self._pca_enabled():
            n_components = self._resolve_pca_components(selected)
            self.pca_ = PCA(n_components=n_components)
            self.pca_.fit(selected.to_numpy(dtype=float, copy=True))
            self.output_feature_cols_ = [
                f"pca_component_{index + 1}"
                for index in range(int(self.pca_.n_components_))
            ]

        self.feature_cols_ = list(self.output_feature_cols_)
        self.fit_dates_ = pd.DatetimeIndex(fit_dates)
        self.is_fitted = True
        return self

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise DataContractError("ERR_FEATURE_REDUCTION_NOT_FITTED", "ERR_FEATURE_REDUCTION_NOT_FITTED")
        if panel.empty:
            passthrough = panel.loc[:, [col for col in self.passthrough_cols_ if col in panel.columns]].copy()
            return passthrough.reindex(columns=self.passthrough_cols_ + self.output_feature_cols_)

        features = _numeric_frame(panel, self.input_feature_cols_)
        imputed = features.fillna(self.medians_)
        winsorized = self._apply_winsor_bounds(imputed)
        scaled_values = self.scaler_.transform(winsorized.to_numpy(dtype=float, copy=True))
        scaled = pd.DataFrame(scaled_values, columns=self.input_feature_cols_, index=features.index)
        selected = scaled.loc[:, self.selected_feature_cols_]

        if self.pca_ is not None:
            reduced_values = self.pca_.transform(selected.to_numpy(dtype=float, copy=True))
            reduced = pd.DataFrame(reduced_values, columns=self.output_feature_cols_, index=panel.index)
        else:
            reduced = selected.copy()
            reduced.columns = self.output_feature_cols_

        passthrough_cols = [col for col in self.passthrough_cols_ if col in panel.columns]
        passthrough = panel.loc[:, passthrough_cols].copy()
        return pd.concat([passthrough, reduced], axis=1)

    def fit_transform(
        self,
        train_panel: pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
        split: Any | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        return self.fit(train_panel, feature_cols, split, **kwargs).transform(train_panel)

    def _fit_scope(self) -> str:
        imputer_scope = str(self.config["imputer"].get("fit_scope", "train_only"))
        pca_scope = str(self.config["pca"].get("fit_scope", "train_only"))
        if imputer_scope != "train_only":
            return imputer_scope
        return pca_scope

    def _fit_winsor_bounds(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        winsor_config = self.config["winsorize"]
        if not bool(winsor_config.get("enabled", True)):
            return (
                pd.Series(-np.inf, index=frame.columns, dtype=float),
                pd.Series(np.inf, index=frame.columns, dtype=float),
            )
        lower_quantile = float(winsor_config.get("lower_quantile", 0.005))
        upper_quantile = float(winsor_config.get("upper_quantile", 0.995))
        lower = frame.quantile(lower_quantile).fillna(self.medians_)
        upper = frame.quantile(upper_quantile).fillna(self.medians_)
        return lower, upper

    def _apply_winsor_bounds(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.clip(lower=self.winsor_lower_, upper=self.winsor_upper_, axis=1)

    def _fit_selector(
        self,
        scaled: pd.DataFrame,
        raw_train_features: pd.DataFrame,
        train_panel: pd.DataFrame,
        train_dates: pd.DatetimeIndex,
        wide_log_return: pd.DataFrame | None,
    ) -> list[str]:
        selector_config = self.config["feature_selection"]
        selected = list(scaled.columns)
        if not bool(selector_config.get("enabled", False)):
            self.feature_selection_report_ = _feature_selection_report(selected, selected)
            return selected

        variance_threshold = float(selector_config.get("variance_threshold", 1.0e-8))
        variances = scaled.var(axis=0, ddof=0)
        selected = [column for column in selected if float(variances[column]) >= variance_threshold]
        if not selected:
            self.feature_selection_report_ = _feature_selection_report(list(scaled.columns), [])
            return []

        correlation_threshold = float(selector_config.get("correlation_threshold", 0.98))
        selected = _drop_correlated_features(
            scaled.loc[:, selected],
            raw_train_features.loc[:, selected].isna().mean(axis=0),
            correlation_threshold,
        )

        max_features = int(selector_config.get("max_features", len(selected)))
        selected = self._fit_mutual_information_selector(
            scaled.loc[:, selected],
            train_panel,
            train_dates,
            wide_log_return,
            max_features,
        )
        return selected

    def _fit_mutual_information_selector(
        self,
        selected_frame: pd.DataFrame,
        train_panel: pd.DataFrame,
        train_dates: pd.DatetimeIndex,
        wide_log_return: pd.DataFrame | None,
        max_features: int,
    ) -> list[str]:
        selected = list(selected_frame.columns)
        if not selected:
            self.feature_selection_report_ = _feature_selection_report([], [])
            return []

        horizon = 5
        purge_horizon = int(self.auxiliary_config.get("purge_horizon_days", horizon))
        target = pd.to_numeric(
            _future_log_return_5d(train_panel, wide_log_return, train_dates, horizon),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        valid_mask = _purged_mi_mask(train_panel, train_dates, target, purge_horizon)
        if int(valid_mask.sum()) <= 1:
            self.feature_selection_report_ = _feature_selection_report(
                selected,
                selected,
                skip_reason="empty_purged_mi_label",
            )
            return selected[:max_features] if max_features > 0 else selected

        n_neighbors = min(3, int(valid_mask.sum()) - 1)
        mi_scores = mutual_info_regression(
            selected_frame.loc[valid_mask, selected].to_numpy(dtype=float, copy=True),
            target.loc[valid_mask].to_numpy(dtype=float, copy=True),
            random_state=0,
            n_neighbors=n_neighbors,
        )
        score_by_feature = dict(zip(selected, mi_scores))
        ranked = sorted(selected, key=lambda column: (-float(score_by_feature[column]), selected.index(column)))
        if max_features > 0:
            ranked = ranked[:max_features]
        self.feature_selection_report_ = _feature_selection_report(
            selected,
            ranked,
            score_by_feature=score_by_feature,
        )
        return ranked

    def _pca_enabled(self) -> bool:
        return bool(self.config["pca"].get("enabled", True))

    def _resolve_pca_components(self, selected: pd.DataFrame) -> int | float:
        n_rows, n_features = selected.shape
        max_components = min(n_rows, n_features)
        if max_components <= 0:
            raise DataContractError("ERR_FEATURE_REDUCTION_EMPTY", "ERR_FEATURE_REDUCTION_EMPTY: pca_input")

        pca_config = self.config["pca"]
        fixed_components = pca_config.get("fixed_components")
        if fixed_components is not None:
            return max(1, min(int(fixed_components), max_components))

        explained_variance = float(pca_config.get("explained_variance", 0.95))
        if 0.0 < explained_variance < 1.0:
            return explained_variance
        return max(1, min(int(explained_variance), max_components))


def _feature_reduction_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    import copy

    resolved = copy.deepcopy(DEFAULT_FEATURE_REDUCTION_CONFIG)
    if config is None:
        return resolved
    config_section = config.get("feature_reduction", config)
    if not isinstance(config_section, Mapping):
        return resolved
    _deep_update(resolved, config_section)
    _apply_input_matrix_reduction_rules(resolved, config)
    return resolved


def _apply_input_matrix_reduction_rules(resolved: dict[str, Any], config: Mapping[str, Any] | None) -> None:
    if not isinstance(config, Mapping) or "feature_matrix" not in config:
        return
    input_matrix_id = _input_matrix_id(config)
    if input_matrix_id not in INPUT_MATRIX_REDUCTION_RULES:
        raise DataContractError(
            "ERR_FEATURE_MATRIX_INVALID_INPUT_MATRIX",
            f"ERR_FEATURE_MATRIX_INVALID_INPUT_MATRIX: {input_matrix_id}",
        )

    rule = INPUT_MATRIX_REDUCTION_RULES[input_matrix_id]
    pca_ablation = _experiment_type(config) == "pca_ablation"
    if input_matrix_id in {"M0", "M1", "M2", "M3", "M4", "M5"} and not pca_ablation:
        resolved["pca"]["enabled"] = False
    elif input_matrix_id in {"M6", "M7"} and not pca_ablation:
        resolved["pca"]["enabled"] = bool(rule["pca"])

    resolved["feature_selection"]["enabled"] = bool(rule["feature_selection"])


def _input_matrix_id(config: Mapping[str, Any] | None) -> str:
    if not isinstance(config, Mapping):
        return "M6"
    feature_matrix = config.get("feature_matrix", {})
    if not isinstance(feature_matrix, Mapping):
        return "M6"
    return str(feature_matrix.get("input_matrix_id", "M6"))


def _experiment_type(config: Mapping[str, Any]) -> str:
    experiment = config.get("experiment", {})
    if not isinstance(experiment, Mapping):
        return ""
    return str(experiment.get("type", ""))


def _auxiliary_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {"future_return_horizons": [5], "purge_horizon_days": 5}
    auxiliary = config.get("auxiliary", {}) if isinstance(config, Mapping) else {}
    if not isinstance(auxiliary, Mapping):
        auxiliary = {}
    return {
        "future_return_horizons": list(auxiliary.get("future_return_horizons", [5])),
        "purge_horizon_days": int(auxiliary.get("purge_horizon_days", 5)),
    }


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _panel_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    if "date" in panel.columns:
        return pd.DatetimeIndex(pd.to_datetime(panel["date"].drop_duplicates()))
    return pd.DatetimeIndex(pd.to_datetime(panel.index.drop_duplicates()))


def _train_dates_index(train_dates: Any, fallback_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if train_dates is not None and hasattr(train_dates, "train_dates"):
        train_dates = train_dates.train_dates
    if train_dates is None:
        return pd.DatetimeIndex(fallback_dates)
    return pd.DatetimeIndex(pd.to_datetime(list(train_dates)))


def _numeric_frame(panel: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    missing = [column for column in feature_cols if column not in panel.columns]
    if missing:
        raise DataContractError("ERR_FEATURE_REDUCTION_EMPTY", f"ERR_FEATURE_REDUCTION_EMPTY: missing={missing}")
    return panel.loc[:, list(feature_cols)].apply(pd.to_numeric, errors="coerce")


def _resolve_feature_cols(
    panel: pd.DataFrame,
    feature_cols: Sequence[str] | None,
    auxiliary_target_cols: Sequence[str] | None,
) -> list[str]:
    candidates = list(feature_cols) if feature_cols is not None else list(panel.columns)
    return [
        str(column)
        for column in candidates
        if column in panel.columns and not _is_non_model_column(str(column), auxiliary_target_cols)
    ]


def _passthrough_cols(columns: Sequence[Any]) -> list[str]:
    passthrough = []
    for column in columns:
        normalized = _normalize_name(column)
        if normalized in IDENTIFIER_COLUMNS or normalized in AVAILABILITY_COLUMNS:
            passthrough.append(str(column))
    return passthrough


def _is_non_model_column(column: str, auxiliary_target_cols: Sequence[str] | None) -> bool:
    normalized = _normalize_name(column)
    auxiliary_targets = {_normalize_name(name) for name in (auxiliary_target_cols or [])}
    if normalized in IDENTIFIER_COLUMNS or normalized in AVAILABILITY_COLUMNS:
        return True
    if normalized in auxiliary_targets:
        return True
    if any(token in normalized for token in FUTURE_LABEL_TOKENS):
        return True
    if normalized in PORTFOLIO_STATE_COLUMNS:
        return True
    return any(normalized.startswith(prefix) for prefix in PORTFOLIO_STATE_PREFIXES)


def _drop_correlated_features(
    frame: pd.DataFrame,
    missing_ratio: pd.Series,
    threshold: float,
) -> list[str]:
    selected = list(frame.columns)
    if len(selected) <= 1:
        return selected
    corr = frame.loc[:, selected].corr().abs().fillna(0.0)
    avg_corr = corr.mean(axis=0).fillna(0.0)
    selected_set = set(selected)

    for left_index, left in enumerate(selected):
        if left not in selected_set:
            continue
        for right in selected[left_index + 1 :]:
            if right not in selected_set:
                continue
            if float(corr.loc[left, right]) <= threshold:
                continue
            drop = _choose_correlated_drop(left, right, missing_ratio, avg_corr)
            selected_set.discard(drop)

    return [column for column in selected if column in selected_set]


def _choose_correlated_drop(
    left: str,
    right: str,
    missing_ratio: pd.Series,
    avg_corr: pd.Series,
) -> str:
    left_missing = float(missing_ratio.get(left, 0.0))
    right_missing = float(missing_ratio.get(right, 0.0))
    if left_missing != right_missing:
        return left if left_missing > right_missing else right
    left_avg_corr = float(avg_corr.get(left, 0.0))
    right_avg_corr = float(avg_corr.get(right, 0.0))
    if left_avg_corr != right_avg_corr:
        return left if left_avg_corr > right_avg_corr else right
    return right


def _future_log_return_5d(
    train_panel: pd.DataFrame,
    wide_log_return: pd.DataFrame | None,
    train_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.Series:
    if wide_log_return is not None:
        return _future_log_return_from_wide(train_panel, wide_log_return, train_dates, horizon)
    if "future_log_return_5d" in train_panel.columns:
        return pd.to_numeric(train_panel["future_log_return_5d"], errors="coerce")
    raise DataContractError(
        "ERR_FEATURE_SELECTION_TARGET_MISSING",
        "ERR_FEATURE_SELECTION_TARGET_MISSING: future_log_return_5d",
    )


def _future_log_return_from_wide(
    train_panel: pd.DataFrame,
    wide_log_return: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.Series:
    if "date" not in train_panel.columns or "ts_code" not in train_panel.columns:
        raise DataContractError(
            "ERR_FEATURE_SELECTION_TARGET_MISSING",
            "ERR_FEATURE_SELECTION_TARGET_MISSING: date_ts_code",
        )

    wide = wide_log_return.copy()
    wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index))
    wide.columns = [str(column) for column in wide.columns]
    future = wide.shift(-1)
    for step in range(2, horizon + 1):
        future = future + wide.shift(-step)

    train_date_set = set(pd.DatetimeIndex(pd.to_datetime(train_dates)))
    valid_dates = []
    for position, date in enumerate(wide.index):
        label_window_dates = wide.index[position + 1 : position + horizon + 1]
        if len(label_window_dates) == horizon and all(pd.Timestamp(item) in train_date_set for item in label_window_dates):
            valid_dates.append(pd.Timestamp(date))
    future.loc[~future.index.isin(valid_dates), :] = np.nan

    target_by_key = (
        future.rename_axis(index="date", columns="ts_code")
        .reset_index()
        .melt(id_vars="date", var_name="ts_code", value_name="future_log_return_5d")
        .set_index(["date", "ts_code"])["future_log_return_5d"]
    )
    target_key = pd.MultiIndex.from_frame(
        pd.DataFrame(
            {
                "date": pd.to_datetime(train_panel["date"]),
                "ts_code": train_panel["ts_code"].astype(str),
            },
            index=train_panel.index,
        )
    )
    target = target_by_key.reindex(target_key)
    target.index = train_panel.index
    return pd.to_numeric(target, errors="coerce")


def _purged_mi_mask(
    train_panel: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    target: pd.Series,
    purge_horizon: int,
) -> pd.Series:
    valid_mask = target.notna()
    if "date" not in train_panel.columns or purge_horizon <= 0:
        return valid_mask
    ordered_train_dates = pd.DatetimeIndex(pd.to_datetime(train_dates)).sort_values().unique()
    if ordered_train_dates.empty:
        return valid_mask & False
    purged_dates = set(ordered_train_dates[-min(purge_horizon, len(ordered_train_dates)) :])
    panel_dates = pd.to_datetime(train_panel["date"])
    return valid_mask & ~panel_dates.isin(purged_dates)


def _feature_selection_report(
    candidate_features: Sequence[str],
    selected_features: Sequence[str],
    *,
    score_by_feature: Mapping[str, float] | None = None,
    skip_reason: str = "",
) -> pd.DataFrame:
    selected_set = set(selected_features)
    rank_by_feature = {feature: index + 1 for index, feature in enumerate(selected_features)}
    return pd.DataFrame(
        [
            {
                "feature_name": feature,
                "selected": feature in selected_set,
                "mi_score": np.nan if score_by_feature is None else float(score_by_feature.get(feature, np.nan)),
                "mi_rank": rank_by_feature.get(feature, np.nan),
                "skip_reason": skip_reason,
            }
            for feature in candidate_features
        ],
        columns=["feature_name", "selected", "mi_score", "mi_rank", "skip_reason"],
    )


def _normalize_name(name: Any) -> str:
    return str(name).strip().lower()
