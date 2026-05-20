from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from src.data.leakage_checks import (
    assert_no_execution_field_in_observation,
    audit_feature_provenance,
    assert_no_future_label_in_features,
)
from src.data.loader import DataContractError, MarketDatasetBundle


TECHNICAL_WINDOWS = (5, 10, 20, 60, 120)
RISK_WINDOWS = (20, 60, 120)
METRICS_FACTORY_RETURN_ANN_FACTOR = 365
METRICS_FACTORY_RISK_ANN_FACTOR = 252
FEATURE_PROVENANCE_COLUMNS = [
    "feature_name",
    "feature_group",
    "source_file",
    "source_family",
    "window",
    "uses_price",
    "uses_volume",
    "uses_return",
    "uses_cross_asset_data",
    "is_metrics_factory_feature",
    "is_auxiliary_target",
    "is_model_feature",
    "requires_shift",
    "shift_steps",
    "fit_scope",
    "leakage_risk_level",
    "leakage_check_status",
    "drop_reason",
]
FEATURE_GROUP_SUMMARY_COLUMNS = [
    "feature_group",
    "source_family",
    "n_total",
    "n_used",
    "n_dropped",
    "n_shifted",
    "n_train_only_fit",
    "n_warning",
    "n_fail",
]
METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS = [
    "feature_name",
    "ts_code",
    "date",
    "stored_value",
    "recomputed_value",
    "abs_error",
    "status",
]
INPUT_MATRIX_BASE_GROUPS: dict[str, tuple[str, ...]] = {
    "M0": ("G0",),
    "M1": ("G1",),
    "M2": ("G1", "G2"),
    "M3": ("G1", "G3"),
    "M4": ("G1", "G2", "G4"),
    "M5": ("G1", "G2", "G3", "G4"),
    "M6": ("G1", "G2", "G3", "G4"),
    "M7": ("G1", "G2", "G3", "G4"),
}


@dataclass(frozen=True)
class FeatureMatrix:
    feature_panel: pd.DataFrame
    feature_cols: list[str]
    provenance: pd.DataFrame
    feature_group_summary: pd.DataFrame
    metrics_factory_audit_sample: pd.DataFrame


class MarketImageDataset:
    def __init__(
        self,
        feature_matrix: FeatureMatrix,
        window_size: int,
        asset_order: Sequence[str] | None = None,
        date_index: Sequence[Any] | None = None,
        *,
        materialize_market_images: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        if window_size <= 0:
            raise DataContractError("ERR_SPLIT_EMPTY", f"ERR_SPLIT_EMPTY: window_size={window_size}")
        assert_no_execution_field_in_observation(feature_matrix.feature_cols)
        self.feature_cols = list(feature_matrix.feature_cols)
        if not self.feature_cols:
            raise DataContractError("ERR_SPLIT_EMPTY", "ERR_SPLIT_EMPTY: empty feature_cols")

        self.window_size = int(window_size)
        self.dtype = dtype
        self.feature_panel = feature_matrix.feature_panel.copy()
        if self.feature_panel.empty:
            raise DataContractError("ERR_SPLIT_EMPTY", "ERR_SPLIT_EMPTY: empty feature_panel")
        self.feature_panel["date"] = pd.to_datetime(self.feature_panel["date"])
        self.feature_panel["ts_code"] = self.feature_panel["ts_code"].astype(str)
        self.asset_order = [str(asset) for asset in (asset_order or _panel_asset_order(self.feature_panel))]
        self._dates = pd.DatetimeIndex(pd.to_datetime(self.feature_panel["date"].drop_duplicates())).sort_values()
        self._date_positions = {date: position for position, date in enumerate(self._dates)}
        self.date_index = self._resolve_date_index(date_index)
        if len(self.date_index) == 0:
            raise DataContractError("ERR_SPLIT_EMPTY", "ERR_SPLIT_EMPTY: no decision dates with full window")

        self._feature_frames = {
            feature: self._feature_frame(feature)
            for feature in self.feature_cols
        }
        self.market_images: np.ndarray | None = None
        if materialize_market_images:
            self.materialize()

    def __len__(self) -> int:
        return len(self.date_index)

    def __getitem__(self, index: int | str | pd.Timestamp) -> np.ndarray:
        if self.market_images is not None and isinstance(index, (int, np.integer)):
            return self.market_images[self._resolve_item_index(int(index))]
        item_index = self._resolve_item_index(index)
        return self._build_market_image(self.date_index[item_index])

    def materialize(self) -> np.ndarray:
        self.market_images = np.stack(
            [self._build_market_image(date) for date in self.date_index],
            axis=0,
        ).astype(self.dtype, copy=False)
        return self.market_images

    def _resolve_date_index(self, date_index: Sequence[Any] | None) -> pd.DatetimeIndex:
        candidates = self._dates if date_index is None else pd.DatetimeIndex(pd.to_datetime(list(date_index)))
        dates = []
        for date in candidates:
            timestamp = pd.Timestamp(date)
            position = self._date_positions.get(timestamp)
            if position is not None and position >= self.window_size - 1:
                dates.append(timestamp)
        return pd.DatetimeIndex(dates)

    def _resolve_item_index(self, index: int | str | pd.Timestamp) -> int:
        if isinstance(index, (str, pd.Timestamp)):
            matches = np.where(self.date_index == pd.Timestamp(index))[0]
            if len(matches) == 0:
                raise IndexError(index)
            return int(matches[0])
        item_index = int(index)
        if item_index < 0:
            item_index += len(self)
        if item_index < 0 or item_index >= len(self):
            raise IndexError(index)
        return item_index

    def _feature_frame(self, feature: str) -> pd.DataFrame:
        frame = self.feature_panel.pivot_table(
            index="date",
            columns="ts_code",
            values=feature,
            aggfunc="last",
        )
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        return frame.reindex(index=self._dates, columns=self.asset_order)

    def _build_market_image(self, date: pd.Timestamp) -> np.ndarray:
        end_position = self._date_positions[pd.Timestamp(date)]
        window_dates = self._dates[end_position - self.window_size + 1 : end_position + 1]
        image = np.empty((len(self.feature_cols), self.window_size, len(self.asset_order)), dtype=self.dtype)
        for feature_index, feature in enumerate(self.feature_cols):
            image[feature_index] = self._feature_frames[feature].reindex(
                index=window_dates,
                columns=self.asset_order,
            ).to_numpy(dtype=self.dtype, copy=True)
        return image


@dataclass(frozen=True)
class _FeatureSpec:
    name: str
    frame: pd.DataFrame
    group: str
    source_file: str
    window: int | None = None
    uses_price: bool = False
    uses_volume: bool = False
    uses_return: bool = False
    uses_cross_asset_data: bool = False
    source_family: str = "local_wide"
    is_metrics_factory_feature: bool = False
    leakage_check_status: str = "pass"
    drop_reason: str = ""
    leakage_risk_level: str = "low"
    is_model_feature: bool = True


class FeatureMatrixBuilder:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = config

    def build(
        self,
        bundle: MarketDatasetBundle,
        split: Any | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> FeatureMatrix:
        del split
        resolved_config = config or self.config or {}
        groups = self._resolve_groups(resolved_config)
        feature_specs: list[_FeatureSpec] = []

        if "G0" in groups:
            feature_specs.extend(self.build_g0_features(bundle))
        if "G1" in groups:
            feature_specs.extend(self.build_g1_features(bundle))
        if "G2" in groups:
            feature_specs.extend(self.build_g2_features(bundle))
        if "G3" in groups:
            feature_specs.extend(self.build_g3_features(bundle, resolved_config))
        if "G4" in groups:
            feature_specs.extend(self.build_g4_features(bundle))

        feature_specs = _dedupe_feature_specs(feature_specs)
        feature_specs = _drop_all_missing_turnover_features(feature_specs, bundle)
        feature_specs = _mark_unrecomputed_metrics_features(feature_specs)
        feature_cols = _filter_auxiliary_targets(
            [spec.name for spec in feature_specs if spec.is_model_feature],
            bundle.auxiliary_target_cols,
        )
        assert_no_future_label_in_features(feature_cols, bundle.auxiliary_target_cols)

        feature_panel = _feature_panel_from_specs(feature_specs, _asset_order(bundle))
        provenance = pd.DataFrame(
            [_provenance_record(spec, spec.name in feature_cols) for spec in feature_specs],
            columns=FEATURE_PROVENANCE_COLUMNS,
        )
        try:
            metrics_factory_audit_sample = _metrics_factory_audit_sample(
                bundle,
                feature_specs,
                feature_cols,
                resolved_config,
            )
        except DataContractError as exc:
            if exc.code == "ERR_METRICS_FACTORY_AUDIT_FAILED":
                metrics_factory_audit_sample = getattr(exc, "metrics_factory_audit_sample", pd.DataFrame(columns=METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS))
                failed_features = set()
                if isinstance(metrics_factory_audit_sample, pd.DataFrame) and not metrics_factory_audit_sample.empty:
                    failed_features = set(
                        metrics_factory_audit_sample.loc[
                            metrics_factory_audit_sample["status"].astype(str).eq("fail"),
                            "feature_name",
                        ].astype(str)
                    )
                feature_specs = [
                    replace(
                        spec,
                        is_model_feature=False,
                        leakage_check_status="fail",
                        drop_reason="metrics_factory_audit_failed",
                    )
                    if spec.name in failed_features
                    else spec
                    for spec in feature_specs
                ]
                feature_cols = _filter_auxiliary_targets(
                    [spec.name for spec in feature_specs if spec.is_model_feature],
                    bundle.auxiliary_target_cols,
                )
                assert_no_future_label_in_features(feature_cols, bundle.auxiliary_target_cols)
                provenance = pd.DataFrame(
                    [_provenance_record(spec, spec.name in feature_cols) for spec in feature_specs],
                    columns=FEATURE_PROVENANCE_COLUMNS,
                )
            else:
                raise
        return FeatureMatrix(
            feature_panel=feature_panel,
            feature_cols=feature_cols,
            provenance=provenance,
            feature_group_summary=_feature_group_summary(provenance),
            metrics_factory_audit_sample=metrics_factory_audit_sample,
        )

    def build_g0_features(self, bundle: MarketDatasetBundle) -> list[_FeatureSpec]:
        return [
            _FeatureSpec(
                name="close",
                frame=_aligned_wide(bundle, "close"),
                group="G0",
                source_file="wide_close",
                uses_price=True,
            ),
            _FeatureSpec(
                name="log_return",
                frame=_aligned_wide(bundle, "log_return"),
                group="G0",
                source_file="wide_log_return",
                uses_return=True,
            ),
            _FeatureSpec(
                name="availability_mask",
                frame=_availability_frame(bundle),
                group="G0",
                source_file="availability_mask",
            ),
        ]

    def build_g1_features(self, bundle: MarketDatasetBundle) -> list[_FeatureSpec]:
        field_flags = {
            "open": {"uses_price": True},
            "high": {"uses_price": True},
            "low": {"uses_price": True},
            "close": {"uses_price": True},
            "vol": {"uses_volume": True},
            "amount": {"uses_volume": True},
            "turnover_rate": {"uses_volume": True},
            "log_return": {"uses_return": True},
        }
        specs = [
            _FeatureSpec(
                name=field,
                frame=_aligned_wide(bundle, field),
                group="G1",
                source_file=f"wide_{field}",
                **flags,
            )
            for field, flags in field_flags.items()
        ]
        specs.append(
            _FeatureSpec(
                name="availability_mask",
                frame=_availability_frame(bundle),
                group="G1",
                source_file="availability_mask",
            )
        )
        return specs

    def build_g2_features(self, bundle: MarketDatasetBundle) -> list[_FeatureSpec]:
        close = _aligned_wide(bundle, "close")
        open_ = _aligned_wide(bundle, "open")
        high = _aligned_wide(bundle, "high")
        low = _aligned_wide(bundle, "low")
        log_return = _aligned_wide(bundle, "log_return")
        pct_chg = _aligned_wide(bundle, "pct_chg")
        amount = _aligned_wide(bundle, "amount")
        vol = _aligned_wide(bundle, "vol")
        turnover_rate = _aligned_wide(bundle, "turnover_rate")

        specs: list[_FeatureSpec] = []
        for window in TECHNICAL_WINDOWS:
            rolling_close_mean = close.rolling(window=window, min_periods=window).mean()
            rolling_close_max = close.rolling(window=window, min_periods=window).max()
            drawdown_abs = (1.0 - _safe_divide(close, rolling_close_max)).clip(lower=0.0)
            downside_return = pct_chg.clip(upper=0.0)
            specs.extend(
                [
                    _technical_spec(
                        f"close_ma_ratio_{window}",
                        _safe_divide(close, rolling_close_mean) - 1.0,
                        window,
                        uses_price=True,
                    ),
                    _technical_spec(
                        f"momentum_log_return_{window}",
                        np.log(_safe_divide(close, close.shift(window))),
                        window,
                        uses_price=True,
                        uses_return=True,
                    ),
                    _technical_spec(
                        f"rolling_log_return_sum_{window}",
                        log_return.rolling(window=window, min_periods=window).sum(),
                        window,
                        uses_return=True,
                    ),
                    _technical_spec(
                        f"rolling_volatility_{window}",
                        pct_chg.rolling(window=window, min_periods=window).std(),
                        window,
                        uses_return=True,
                    ),
                    _technical_spec(
                        f"rolling_downside_vol_{window}",
                        downside_return.rolling(window=window, min_periods=window).std(),
                        window,
                        uses_return=True,
                    ),
                    _technical_spec(
                        f"drawdown_signed_{window}",
                        _safe_divide(close, rolling_close_max) - 1.0,
                        window,
                        uses_price=True,
                    ),
                    _technical_spec(
                        f"drawdown_abs_{window}",
                        drawdown_abs,
                        window,
                        uses_price=True,
                    ),
                    _technical_spec(
                        f"max_drawdown_abs_{window}",
                        drawdown_abs.rolling(window=window, min_periods=window).max(),
                        window,
                        uses_price=True,
                    ),
                ]
            )

        close_ma_5 = close.rolling(window=5, min_periods=5).mean()
        close_ma_20 = close.rolling(window=20, min_periods=20).mean()
        close_ma_60 = close.rolling(window=60, min_periods=60).mean()
        amount_ma_20 = amount.rolling(window=20, min_periods=20).mean()
        vol_ma_20 = vol.rolling(window=20, min_periods=20).mean()
        specs.extend(
            [
                _technical_spec("ma_5_over_20_ratio", _safe_divide(close_ma_5, close_ma_20) - 1.0, 20, True),
                _technical_spec("ma_20_over_60_ratio", _safe_divide(close_ma_20, close_ma_60) - 1.0, 60, True),
                _technical_spec("log1p_amount", np.log1p(amount.clip(lower=0.0)), None, uses_volume=True),
                _technical_spec("log1p_vol", np.log1p(vol.clip(lower=0.0)), None, uses_volume=True),
                _technical_spec("amount_ratio_20", _safe_divide(amount, amount_ma_20) - 1.0, 20, uses_volume=True),
                _technical_spec("vol_ratio_20", _safe_divide(vol, vol_ma_20) - 1.0, 20, uses_volume=True),
                _technical_spec("turnover_rate_ma_20", turnover_rate.rolling(window=20, min_periods=20).mean(), 20, uses_volume=True),
                _technical_spec("turnover_rate", turnover_rate, None, uses_volume=True),
                _technical_spec("high_low_over_close", _safe_divide(high - low, close), None, uses_price=True),
                _technical_spec("close_open_over_open", _safe_divide(close - open_, open_), None, uses_price=True),
            ]
        )
        return specs

    def build_g4_features(self, bundle: MarketDatasetBundle) -> list[_FeatureSpec]:
        returns = _aligned_wide(bundle, "pct_chg")
        availability = _availability_bool_frame(bundle)
        available_returns = returns.where(availability)
        benchmark_return = available_returns.mean(axis=1)

        specs: list[_FeatureSpec] = []
        for window in RISK_WINDOWS:
            benchmark_var = benchmark_return.rolling(window=window, min_periods=window).var()
            benchmark_vol = benchmark_return.rolling(window=window, min_periods=window).std()
            cov_to_benchmark = available_returns.rolling(window=window, min_periods=window).cov(benchmark_return)
            asset_vol = available_returns.rolling(window=window, min_periods=window).std()
            specs.extend(
                [
                    _cross_asset_risk_spec(
                        f"rolling_beta_to_benchmark_{window}",
                        cov_to_benchmark.div(benchmark_var, axis=0),
                        window,
                    ),
                    _cross_asset_risk_spec(
                        f"rolling_corr_to_benchmark_{window}",
                        available_returns.rolling(window=window, min_periods=window).corr(benchmark_return),
                        window,
                    ),
                    _cross_asset_risk_spec(
                        f"rolling_cov_to_benchmark_{window}",
                        cov_to_benchmark,
                        window,
                    ),
                    _cross_asset_risk_spec(f"asset_vol_{window}", asset_vol, window),
                    _cross_asset_risk_spec(
                        f"asset_vol_over_benchmark_vol_{window}",
                        asset_vol.div(benchmark_vol, axis=0),
                        window,
                    ),
                ]
            )

        benchmark_return_20 = _rolling_compound_return(benchmark_return, 20)
        benchmark_return_60 = _rolling_compound_return(benchmark_return, 60)
        benchmark_return_120 = _rolling_compound_return(benchmark_return, 120)
        benchmark_vol_20 = benchmark_return.rolling(window=20, min_periods=20).std()
        benchmark_vol_percentile_252 = _rolling_last_percentile(benchmark_vol_20, 252)
        trend_signal = pd.Series(
            np.select(
                [benchmark_return_120 >= 0.10, benchmark_return_120 <= -0.10],
                [1.0, -1.0],
                default=0.0,
            ),
            index=benchmark_return.index,
        ).where(benchmark_return_120.notna())

        specs.extend(
            [
                _cross_asset_risk_spec(
                    "benchmark_return_20",
                    _series_to_asset_frame(benchmark_return_20, availability),
                    20,
                ),
                _cross_asset_risk_spec(
                    "benchmark_return_60",
                    _series_to_asset_frame(benchmark_return_60, availability),
                    60,
                ),
                _cross_asset_risk_spec(
                    "benchmark_return_120",
                    _series_to_asset_frame(benchmark_return_120, availability),
                    120,
                ),
                _cross_asset_risk_spec(
                    "benchmark_vol_20",
                    _series_to_asset_frame(benchmark_vol_20, availability),
                    20,
                ),
                _cross_asset_risk_spec(
                    "benchmark_vol_percentile_252",
                    _series_to_asset_frame(benchmark_vol_percentile_252, availability),
                    252,
                ),
                _cross_asset_risk_spec(
                    "trend_signal",
                    _series_to_asset_frame(trend_signal, availability),
                    120,
                ),
            ]
        )
        for loading_index, frame in enumerate(_eigen_loading_frames(available_returns, availability), start=1):
            specs.append(_cross_asset_risk_spec(f"eigen_loading_{loading_index}", frame, 60))
        return specs

    def build_g3_features(self, bundle: MarketDatasetBundle, config: Mapping[str, Any]) -> list[_FeatureSpec]:
        metrics_features = bundle.metrics_features
        if metrics_features is None:
            return []

        required_keys = {"date", "ts_code"}
        missing = sorted(required_keys - set(metrics_features.columns))
        if missing:
            raise DataContractError(
                "ERR_DATA_SCHEMA_MISMATCH",
                f"ERR_DATA_SCHEMA_MISMATCH: metrics_features missing columns {missing}",
            )

        asset_order = _asset_order(bundle)
        metrics = metrics_features.copy()
        metrics["date"] = pd.to_datetime(metrics["date"])
        metric_columns = [column for column in metrics.columns if column not in required_keys]
        specs: list[_FeatureSpec] = []
        for column in metric_columns:
            audit = audit_feature_provenance(str(column), config, bundle.auxiliary_target_cols)
            frame = metrics.pivot(index="date", columns="ts_code", values=column)
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
            frame = _clean_feature_frame(frame.reindex(columns=asset_order).sort_index())
            specs.append(
                _FeatureSpec(
                    name=str(column),
                    frame=frame,
                    group="G3",
                    source_file="all_metrics_features.parquet",
                    source_family="metrics_factory",
                    is_metrics_factory_feature=True,
                    leakage_check_status=audit["leakage_check_status"],
                    drop_reason=audit["drop_reason"],
                    leakage_risk_level=audit["leakage_risk_level"],
                    is_model_feature=bool(audit["is_model_feature"]),
                )
            )
        return specs

    @staticmethod
    def _resolve_groups(config: Mapping[str, Any]) -> tuple[str, ...]:
        feature_matrix_config = config.get("feature_matrix", {}) if isinstance(config, Mapping) else {}
        input_matrix_id = feature_matrix_config.get("input_matrix_id", "M6")
        if input_matrix_id not in INPUT_MATRIX_BASE_GROUPS:
            raise DataContractError(
                "ERR_FEATURE_MATRIX_INVALID_INPUT_MATRIX",
                f"ERR_FEATURE_MATRIX_INVALID_INPUT_MATRIX: {input_matrix_id}",
            )
        return INPUT_MATRIX_BASE_GROUPS[input_matrix_id]


def _asset_order(bundle: MarketDatasetBundle) -> list[str]:
    manifest_order = bundle.data_manifest.get("canonical_asset_order")
    if isinstance(manifest_order, list) and manifest_order:
        return [str(asset) for asset in manifest_order]
    attr_order = bundle.asset_universe.attrs.get("canonical_asset_order")
    if isinstance(attr_order, list) and attr_order:
        return [str(asset) for asset in attr_order]
    return [str(asset) for asset in bundle.availability_mask.columns]


def _panel_asset_order(feature_panel: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(feature_panel["ts_code"].astype(str).tolist()))


def _aligned_wide(bundle: MarketDatasetBundle, field: str) -> pd.DataFrame:
    if field not in bundle.wide:
        raise DataContractError("ERR_DATA_SCHEMA_MISMATCH", f"ERR_DATA_SCHEMA_MISMATCH: missing wide_{field}")
    frame = bundle.wide[field].copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    return _clean_feature_frame(frame.reindex(columns=_asset_order(bundle)).sort_index())


def _availability_frame(bundle: MarketDatasetBundle) -> pd.DataFrame:
    frame = bundle.availability_mask.reindex(columns=_asset_order(bundle)).sort_index()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    return frame.astype(float)


def _availability_bool_frame(bundle: MarketDatasetBundle) -> pd.DataFrame:
    frame = bundle.availability_mask.reindex(columns=_asset_order(bundle)).sort_index()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    return frame.astype(bool)


def _technical_spec(
    name: str,
    frame: pd.DataFrame,
    window: int | None,
    uses_price: bool = False,
    uses_volume: bool = False,
    uses_return: bool = False,
) -> _FeatureSpec:
    return _FeatureSpec(
        name=name,
        frame=_clean_feature_frame(frame),
        group="G2",
        source_file="wide",
        window=window,
        uses_price=uses_price,
        uses_volume=uses_volume,
        uses_return=uses_return,
    )


def _cross_asset_risk_spec(name: str, frame: pd.DataFrame, window: int | None) -> _FeatureSpec:
    return _FeatureSpec(
        name=name,
        frame=_clean_feature_frame(frame),
        group="G4",
        source_file="wide_pct_chg+availability_mask",
        window=window,
        uses_return=True,
        uses_cross_asset_data=True,
        source_family="cross_asset_risk",
    )


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return _clean_feature_frame(numerator / denominator)


def _rolling_compound_return(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).apply(lambda values: np.prod(1.0 + values) - 1.0, raw=True)


def _rolling_last_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).apply(_last_percentile, raw=True)


def _last_percentile(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    return float((valid <= valid[-1]).sum() / len(valid))


def _series_to_asset_frame(series: pd.Series, availability: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({asset: series for asset in availability.columns}, index=availability.index)
    return frame.where(availability)


def _eigen_loading_frames(returns: pd.DataFrame, availability: pd.DataFrame) -> list[pd.DataFrame]:
    frames = [
        pd.DataFrame(np.nan, index=returns.index, columns=returns.columns),
        pd.DataFrame(np.nan, index=returns.index, columns=returns.columns),
        pd.DataFrame(np.nan, index=returns.index, columns=returns.columns),
    ]
    for row_index in range(59, len(returns.index)):
        date = returns.index[row_index]
        current_assets = availability.columns[availability.iloc[row_index].to_numpy()].tolist()
        window_returns = returns.iloc[row_index - 59 : row_index + 1][current_assets]
        usable_assets = [asset for asset in current_assets if window_returns[asset].notna().all()]
        if not usable_assets:
            continue

        corr = window_returns[usable_assets].corr().reindex(index=usable_assets, columns=usable_assets).fillna(0.0)
        corr_values = corr.to_numpy(dtype=float).copy()
        np.fill_diagonal(corr_values, 1.0)
        eigen_values, eigen_vectors = np.linalg.eigh(corr_values)
        order = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[:, order]
        for component_index in range(min(3, eigen_vectors.shape[1])):
            vector = eigen_vectors[:, component_index]
            if vector.sum() < 0:
                vector = -vector
            frames[component_index].loc[date, usable_assets] = vector
    return frames


def _clean_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def _dedupe_feature_specs(specs: Sequence[_FeatureSpec]) -> list[_FeatureSpec]:
    deduped: dict[str, _FeatureSpec] = {}
    for spec in specs:
        deduped[spec.name] = spec
    return list(deduped.values())


def _drop_all_missing_turnover_features(
    specs: Sequence[_FeatureSpec],
    bundle: MarketDatasetBundle,
) -> list[_FeatureSpec]:
    if not _turnover_rate_all_missing(bundle):
        return list(specs)
    dropped = []
    for spec in specs:
        if _is_turnover_rate_feature(spec.name):
            dropped.append(
                replace(
                    spec,
                    leakage_check_status="dropped",
                    drop_reason="all_missing",
                    is_model_feature=False,
                )
            )
        else:
            dropped.append(spec)
    return dropped


def _turnover_rate_all_missing(bundle: MarketDatasetBundle) -> bool:
    if bool(bundle.data_manifest.get("turnover_rate_all_missing", False)):
        return True
    if "turnover_rate" not in bundle.wide:
        return False
    turnover_rate = bundle.wide["turnover_rate"].reindex(columns=_asset_order(bundle))
    availability = bundle.availability_mask.reindex(columns=_asset_order(bundle)).astype(bool)
    return bool(turnover_rate.where(availability).isna().all().all())


def _is_turnover_rate_feature(feature_name: str) -> bool:
    return feature_name == "turnover_rate" or feature_name.startswith("turnover_rate_")


def _mark_unrecomputed_metrics_features(specs: Sequence[_FeatureSpec]) -> list[_FeatureSpec]:
    marked = []
    for spec in specs:
        if (
            spec.is_metrics_factory_feature
            and spec.is_model_feature
            and not spec.drop_reason
            and not _is_recomputable_metrics_factory_feature(spec.name)
        ):
            marked.append(replace(spec, drop_reason="not_recomputable"))
        else:
            marked.append(spec)
    return marked


def _is_recomputable_metrics_factory_feature(feature_name: str) -> bool:
    metric_name, period = _split_metrics_factory_feature_name(feature_name)
    return period is not None and _period_days(period) is not None and metric_name in {
        "TotalReturn",
        "AnnualizedReturn",
        "AverageDailyReturn",
        "MedianDailyReturn",
        "Volatility",
        "AnnualizedVolatility",
        "MaxGain",
        "MaxLoss",
        "ReturnRange",
        "MeanAbsoluteDeviation",
    }


def _filter_auxiliary_targets(feature_cols: Sequence[str], auxiliary_target_cols: Sequence[str]) -> list[str]:
    auxiliary_targets = {str(column) for column in auxiliary_target_cols}
    return [column for column in feature_cols if column not in auxiliary_targets]


def _feature_group_summary(provenance: pd.DataFrame) -> pd.DataFrame:
    if provenance.empty:
        return pd.DataFrame(columns=FEATURE_GROUP_SUMMARY_COLUMNS)

    rows = []
    for (feature_group, source_family), group in provenance.groupby(["feature_group", "source_family"], sort=True):
        rows.append(
            {
                "feature_group": feature_group,
                "source_family": source_family,
                "n_total": int(len(group)),
                "n_used": int(group["is_model_feature"].astype(bool).sum()),
                "n_dropped": int((~group["is_model_feature"].astype(bool)).sum()),
                "n_shifted": int(group["requires_shift"].astype(bool).sum()),
                "n_train_only_fit": int((group["fit_scope"] == "train_only").sum()),
                "n_warning": int((group["leakage_check_status"] == "warning").sum()),
                "n_fail": int((group["leakage_check_status"] == "fail").sum()),
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_GROUP_SUMMARY_COLUMNS)


def _metrics_factory_audit_sample(
    bundle: MarketDatasetBundle,
    specs: Sequence[_FeatureSpec],
    feature_cols: Sequence[str],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if bundle.metrics_features is None or bundle.metrics_features.empty:
        return pd.DataFrame(columns=METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS)

    tolerance = float(_feature_audit_config(config).get("audit_abs_error_tolerance", 1.0e-8))
    metrics = bundle.metrics_features.copy()
    metrics["date"] = pd.to_datetime(metrics["date"])
    metrics = metrics.set_index(["date", "ts_code"]).sort_index()

    feature_col_set = set(feature_cols)
    rows = []
    for spec in specs:
        if (
            not spec.is_metrics_factory_feature
            or spec.name not in feature_col_set
            or spec.name not in metrics.columns
            or not _is_recomputable_metrics_factory_feature(spec.name)
        ):
            continue
        stored_sample = metrics[spec.name].dropna()
        date = ts_code = None
        stored_value = recomputed_value = None
        for (sample_date, sample_ts_code), sample_stored_value in stored_sample.items():
            sample_recomputed_value = _recompute_metrics_factory_value(
                spec.name,
                pd.Timestamp(sample_date),
                str(sample_ts_code),
                bundle,
            )
            if sample_recomputed_value is None or pd.isna(sample_recomputed_value):
                continue
            date = pd.Timestamp(sample_date)
            ts_code = str(sample_ts_code)
            stored_value = float(sample_stored_value)
            recomputed_value = float(sample_recomputed_value)
            break
        if date is None or ts_code is None or stored_value is None or recomputed_value is None:
            continue
        abs_error = abs(stored_value - recomputed_value)
        status = "pass" if abs_error <= tolerance else "fail"
        rows.append(
            {
                "feature_name": spec.name,
                "ts_code": ts_code,
                "date": pd.Timestamp(date),
                "stored_value": stored_value,
                "recomputed_value": recomputed_value,
                "abs_error": abs_error,
                "status": status,
            }
        )

    sample_report = pd.DataFrame(rows, columns=METRICS_FACTORY_AUDIT_SAMPLE_COLUMNS)
    if not sample_report.empty and (sample_report["status"] == "fail").any():
        failed = sample_report.loc[sample_report["status"] == "fail", "feature_name"].drop_duplicates().tolist()
        error = DataContractError(
            "ERR_METRICS_FACTORY_AUDIT_FAILED",
            f"ERR_METRICS_FACTORY_AUDIT_FAILED: {failed}",
        )
        setattr(error, "metrics_factory_audit_sample", sample_report)
        raise error
    return sample_report


def _recompute_metrics_factory_value(
    feature_name: str,
    date: pd.Timestamp,
    ts_code: str,
    bundle: MarketDatasetBundle,
) -> float | None:
    metric_name, period = _split_metrics_factory_feature_name(feature_name)
    if period is None:
        return None
    days = _period_days(period)
    if days is None:
        return None
    log_return = _aligned_wide(bundle, "log_return")
    if ts_code not in log_return.columns or date not in log_return.index:
        return None

    end_position = log_return.index.get_loc(date)
    if not isinstance(end_position, (int, np.integer)) or end_position < days:
        return None

    start_date = pd.Timestamp(log_return.index[end_position - days])
    nature_days = max((date - start_date).days, 1)
    values = log_return[ts_code].iloc[end_position - days + 1 : end_position + 1].to_numpy(dtype=float)
    valid_count = int(np.sum(~np.isnan(values)))
    if valid_count < 2:
        return np.nan

    total_return = float(np.nansum(values))
    if metric_name == "TotalReturn":
        return total_return
    if metric_name == "AnnualizedReturn":
        return total_return / nature_days * METRICS_FACTORY_RETURN_ANN_FACTOR
    if metric_name == "AverageDailyReturn":
        return float(np.nanmean(values))
    if metric_name == "MedianDailyReturn":
        return float(np.nanmedian(values))
    if metric_name == "Volatility":
        return float(np.nanstd(values, ddof=1))
    if metric_name == "AnnualizedVolatility":
        return float(np.nanstd(values, ddof=1) * np.sqrt(METRICS_FACTORY_RISK_ANN_FACTOR))
    if metric_name == "MaxGain":
        return float(np.nanmax(values))
    if metric_name == "MaxLoss":
        return float(np.nanmin(values))
    if metric_name == "ReturnRange":
        return float(np.nanmax(values) - np.nanmin(values))
    if metric_name == "MeanAbsoluteDeviation":
        return float(np.nanmean(np.abs(values - np.nanmean(values))))
    return None


def _split_metrics_factory_feature_name(feature_name: str) -> tuple[str, str | None]:
    if ":" not in feature_name:
        return feature_name, None
    metric_name, period = feature_name.rsplit(":", 1)
    return metric_name, period


def _period_days(period: str) -> int | None:
    if not period.endswith("d"):
        return None
    days = period[:-1]
    return int(days) if days.isdigit() else None


def _feature_audit_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    feature_audit = config.get("feature_audit", {})
    return feature_audit if isinstance(feature_audit, Mapping) else {}


def _feature_panel_from_specs(specs: Sequence[_FeatureSpec], asset_order: Sequence[str]) -> pd.DataFrame:
    if not specs:
        return pd.DataFrame(columns=["date", "ts_code"])

    series = []
    for spec in specs:
        frame = spec.frame.copy()
        frame.index.name = "date"
        frame.columns.name = "ts_code"
        series.append(frame.stack().rename(spec.name))

    panel = pd.concat(series, axis=1).reset_index()
    panel["_asset_order"] = pd.Categorical(panel["ts_code"], categories=list(asset_order), ordered=True)
    panel = panel.sort_values(["date", "_asset_order"]).drop(columns=["_asset_order"]).reset_index(drop=True)
    return panel


def _provenance_record(spec: _FeatureSpec, is_model_feature: bool) -> dict[str, Any]:
    status = spec.leakage_check_status
    drop_reason = spec.drop_reason
    if not is_model_feature and status == "pass":
        status = "dropped"
        drop_reason = "auxiliary_target"
    is_auxiliary_target = drop_reason == "auxiliary_target"
    return {
        "feature_name": spec.name,
        "feature_group": spec.group,
        "source_file": spec.source_file,
        "source_family": spec.source_family,
        "window": spec.window,
        "uses_price": spec.uses_price,
        "uses_volume": spec.uses_volume,
        "uses_return": spec.uses_return,
        "uses_cross_asset_data": spec.uses_cross_asset_data,
        "is_metrics_factory_feature": spec.is_metrics_factory_feature,
        "is_auxiliary_target": is_auxiliary_target,
        "is_model_feature": is_model_feature,
        "requires_shift": False,
        "shift_steps": 0,
        "fit_scope": "none",
        "leakage_risk_level": spec.leakage_risk_level,
        "leakage_check_status": status,
        "drop_reason": drop_reason,
    }
