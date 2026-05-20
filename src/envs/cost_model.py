from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, VALID_COST_MODES, VALID_FIXED_COST_UNITS
from src.data.loader import DataContractError
from src.envs.state import ExecutionMarketState, PortfolioState


GLOBAL_EPS = 1.0e-8
VOLATILITY_EPS = 1.0e-8
PROBABILITY_EPS = 1.0e-8
WEIGHT_SUM_EPS = 1.0e-12
ADV_EPS = 1000000.0
COST_CALIBRATION_REPORT_COLUMNS = [
    "amount_bucket",
    "turnover_rate_bucket",
    "sigma20_bucket",
    "sample_count",
    "realized_bps_mean",
    "realized_bps_median",
    "fallback_used",
    "fallback_reason",
    "status",
]
CALIBRATION_REQUIRED_COLUMNS = {"amount", "turnover_rate", "sigma20", "realized_bps"}


@dataclass
class CostBreakdown:
    turnover: float
    proportional_cost: float
    fixed_cost: float
    slippage_cost: float
    market_impact_cost: float
    total_transaction_cost: float
    per_asset_trade_weight: np.ndarray
    per_asset_market_impact_cost: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.turnover = _finite_float("turnover", self.turnover)
        self.proportional_cost = _finite_float("proportional_cost", self.proportional_cost)
        self.fixed_cost = _finite_float("fixed_cost", self.fixed_cost)
        self.slippage_cost = _finite_float("slippage_cost", self.slippage_cost)
        self.market_impact_cost = _finite_float("market_impact_cost", self.market_impact_cost)
        self.total_transaction_cost = _finite_float("total_transaction_cost", self.total_transaction_cost)
        self.per_asset_trade_weight = _array_1d("per_asset_trade_weight", self.per_asset_trade_weight)
        self.per_asset_market_impact_cost = _array_1d(
            "per_asset_market_impact_cost",
            self.per_asset_market_impact_cost,
            self.per_asset_trade_weight.shape,
        )
        self.info = dict(self.info)


class CostModel:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.raw_config = config or DEFAULT_CONFIG
        self.cost_config = _cost_model_config(config)
        self.execution_config = _execution_model_config(config)
        self.is_calibrated_ = False
        self.calibration_bins_: dict[str, tuple[float, ...] | None] = {}
        self.calibration_tables_: dict[str, dict[Any, dict[str, Any]]] = {}
        self.calibration_report_: pd.DataFrame | None = None

    def fit_calibration(
        self,
        train_samples: pd.DataFrame | None = None,
        *,
        report_path: str | Path | None = None,
        fit_scope: str = "train_only",
    ) -> pd.DataFrame:
        if fit_scope != "train_only":
            raise DataContractError(
                "ERR_LEAKAGE_COST_CALIBRATION_FIT_SCOPE",
                "ERR_LEAKAGE_COST_CALIBRATION_FIT_SCOPE: CostModel.fit_calibration",
            )

        mode = str(self.cost_config.get("mode", "empirical_default"))
        if mode not in VALID_COST_MODES:
            raise DataContractError("ERR_CONFIG_INVALID_COST_MODE", "ERR_CONFIG_INVALID_COST_MODE: cost_model.mode")

        if mode == "empirical_default":
            report = _not_applicable_report()
            self.calibration_report_ = report
            self._write_calibration_report(report, report_path)
            return report

        samples = _calibration_samples(train_samples)
        if samples.empty:
            report = _empty_calibration_report()
            self.calibration_report_ = report
            self.calibration_tables_ = {"exact": {}, "amount_sigma": {}, "amount": {}}
            self.is_calibrated_ = True
            self._write_calibration_report(report, report_path)
            return report

        bucketed = samples.copy()
        self.calibration_bins_ = {}
        for source_col, bucket_col in (
            ("amount", "amount_bucket"),
            ("turnover_rate", "turnover_rate_bucket"),
            ("sigma20", "sigma20_bucket"),
        ):
            bucketed[bucket_col], self.calibration_bins_[source_col] = _fit_buckets(bucketed[source_col])

        min_bucket_samples = self._min_bucket_samples()
        report = _calibration_report(bucketed, min_bucket_samples)
        self.calibration_report_ = report
        self.calibration_tables_ = {
            "exact": _lookup_table(bucketed, ["amount_bucket", "turnover_rate_bucket", "sigma20_bucket"]),
            "amount_sigma": _lookup_table(bucketed, ["amount_bucket", "sigma20_bucket"]),
            "amount": _lookup_table(bucketed, ["amount_bucket"]),
        }
        self.is_calibrated_ = True
        self._write_calibration_report(report, report_path)
        return report

    def estimate(
        self,
        prev_weights: np.ndarray,
        target_weights: np.ndarray,
        execution_market_state: ExecutionMarketState,
        portfolio_state: PortfolioState,
    ) -> CostBreakdown:
        mode = str(self.cost_config.get("mode", "empirical_default"))
        if mode not in VALID_COST_MODES:
            raise DataContractError("ERR_CONFIG_INVALID_COST_MODE", "ERR_CONFIG_INVALID_COST_MODE: cost_model.mode")

        prev = _array_1d("prev_weights", prev_weights)
        target = _array_1d("target_weights", target_weights, prev.shape)
        trade_weight = np.abs(target - prev)
        turnover = float(0.5 * np.sum(trade_weight))
        info = {
            "mode": mode,
            "market_impact_enabled": bool(self.cost_config.get("market_impact_enabled", True)),
            "fixed_cost_unit": str(self.execution_config.get("fixed_cost_unit", "nav_fraction")),
            "adv_eps": self._adv_eps(),
            "volatility_eps": self._volatility_eps(),
            "cost_observation_date": str(getattr(execution_market_state, "cost_observation_date", "")),
            "cost_observation_timing": str(getattr(execution_market_state, "cost_observation_timing", "")),
        }

        if turnover <= WEIGHT_SUM_EPS:
            return CostBreakdown(
                turnover=0.0,
                proportional_cost=0.0,
                fixed_cost=0.0,
                slippage_cost=0.0,
                market_impact_cost=0.0,
                total_transaction_cost=0.0,
                per_asset_trade_weight=np.zeros_like(trade_weight, dtype=float),
                per_asset_market_impact_cost=np.zeros_like(trade_weight, dtype=float),
                info=info,
            )

        proportional_cost = _non_negative_config_float("proportional_cost", self.cost_config.get("proportional_cost", 0.0)) * turnover
        fixed_cost = self._fixed_cost(portfolio_state)
        slippage_cost = 0.0
        per_asset_market_impact_cost = np.zeros_like(trade_weight, dtype=float)
        market_impact_cost = 0.0

        if mode == "calibrated":
            per_asset_market_impact_cost = self._calibrated_variable_cost(
                trade_weight,
                execution_market_state,
                portfolio_state,
                info,
            )
            market_impact_cost = float(np.sum(per_asset_market_impact_cost))
        elif mode == "empirical_default":
            slippage_cost = _non_negative_config_float("slippage", self.cost_config.get("slippage", 0.0)) * turnover
            if bool(self.cost_config.get("market_impact_enabled", True)):
                per_asset_market_impact_cost = self._empirical_market_impact_per_asset(
                    trade_weight,
                    execution_market_state,
                    portfolio_state,
                    info,
                )
                market_impact_cost = float(np.sum(per_asset_market_impact_cost))

        total_transaction_cost = proportional_cost + fixed_cost + slippage_cost + market_impact_cost
        return CostBreakdown(
            turnover=turnover,
            proportional_cost=proportional_cost,
            fixed_cost=fixed_cost,
            slippage_cost=slippage_cost,
            market_impact_cost=market_impact_cost,
            total_transaction_cost=total_transaction_cost,
            per_asset_trade_weight=trade_weight,
            per_asset_market_impact_cost=per_asset_market_impact_cost,
            info=info,
        )

    def _fixed_cost(self, portfolio_state: PortfolioState) -> float:
        fixed_cost = _non_negative_config_float("fixed_cost", self.cost_config.get("fixed_cost", 0.0))
        if fixed_cost == 0.0:
            return 0.0
        fixed_cost_unit = str(self.execution_config.get("fixed_cost_unit", "nav_fraction"))
        if fixed_cost_unit not in VALID_FIXED_COST_UNITS:
            raise DataContractError(
                "ERR_CONFIG_INVALID_FIXED_COST_UNIT",
                "ERR_CONFIG_INVALID_FIXED_COST_UNIT: execution_model.fixed_cost_unit",
            )
        if fixed_cost_unit == "nav_fraction":
            return fixed_cost
        return fixed_cost / _portfolio_value_required(portfolio_state)

    def _adv_eps(self) -> float:
        return _positive_config_float("adv_eps", self.cost_config.get("adv_eps", ADV_EPS))

    def _volatility_eps(self) -> float:
        return _positive_config_float(
            "volatility_eps",
            self.cost_config.get("volatility_eps", VOLATILITY_EPS),
        )

    def _min_bucket_samples(self) -> int:
        calibration = self.cost_config.get("calibration", {})
        min_samples = int(calibration.get("min_bucket_samples", 30))
        if min_samples <= 0:
            raise DataContractError(
                "ERR_CONFIG_INVALID_COST_MODEL",
                "ERR_CONFIG_INVALID_COST_MODEL: cost_model.calibration.min_bucket_samples",
            )
        return min_samples

    def _calibrated_variable_cost(
        self,
        trade_weight: np.ndarray,
        execution_market_state: ExecutionMarketState,
        portfolio_state: PortfolioState,
        info: dict[str, Any],
    ) -> np.ndarray:
        if not self.is_calibrated_:
            raise DataContractError(
                "ERR_COST_CALIBRATION_NOT_FITTED",
                "ERR_COST_CALIBRATION_NOT_FITTED: CostModel.fit_calibration",
            )

        amount = _cost_observation_array(
            "amount_at_cost_observation",
            "amount_at_execution",
            execution_market_state,
            trade_weight.shape,
        )
        sigma20 = _cost_observation_array(
            "volatility_20d_at_cost_observation",
            "volatility_20d_at_execution",
            execution_market_state,
            trade_weight.shape,
        )
        turnover_rate = self._execution_turnover_rate(
            execution_market_state,
            amount,
            trade_weight.shape,
            info,
        )
        empirical_default: np.ndarray | None = None
        min_bucket_samples = self._min_bucket_samples()
        calibrated_cost = np.zeros_like(trade_weight, dtype=float)
        fallback_reasons: list[str] = []
        fallback_used = False

        for index, trade in enumerate(trade_weight):
            amount_bucket = _assign_bucket(float(amount[index]), self.calibration_bins_.get("amount"))
            turnover_bucket = _assign_bucket(float(turnover_rate[index]), self.calibration_bins_.get("turnover_rate"))
            sigma_bucket = _assign_bucket(float(sigma20[index]), self.calibration_bins_.get("sigma20"))
            lookup = self._calibration_record(
                (amount_bucket, turnover_bucket, sigma_bucket),
                (amount_bucket, sigma_bucket),
                (amount_bucket,),
                min_bucket_samples,
            )
            if lookup is None:
                if empirical_default is None:
                    empirical_default = self._empirical_variable_cost_per_asset(
                        trade_weight,
                        execution_market_state,
                        portfolio_state,
                        info,
                    )
                calibrated_cost[index] = empirical_default[index]
                fallback_reasons.append("empirical_default")
                fallback_used = True
                continue
            record, fallback_reason = lookup
            calibrated_cost[index] = float(record["realized_bps_median"]) * trade / 10000.0
            fallback_reasons.append(fallback_reason)
            fallback_used = fallback_used or bool(fallback_reason)

        info["calibration_fallback_reason"] = fallback_reasons
        info["calibration_fallback_used"] = fallback_used
        return calibrated_cost

    def _execution_turnover_rate(
        self,
        execution_market_state: ExecutionMarketState,
        amount: np.ndarray,
        shape: tuple[int, ...],
        info: dict[str, Any],
    ) -> np.ndarray:
        turnover_rate = getattr(execution_market_state, "turnover_rate_at_cost_observation", None)
        turnover_rate_name = "turnover_rate_at_cost_observation"
        if turnover_rate is None:
            turnover_rate = getattr(execution_market_state, "turnover_rate_at_execution", None)
            turnover_rate_name = "turnover_rate_at_execution"
        if turnover_rate is not None:
            turnover_rate_array = _execution_array(turnover_rate_name, turnover_rate, shape)
            if np.isfinite(turnover_rate_array).any():
                info["turnover_rate_source"] = turnover_rate_name
                return turnover_rate_array

        adv20 = _cost_observation_array(
            "adv20_at_cost_observation",
            "adv20_at_execution",
            execution_market_state,
            shape,
        )
        adv20_safe = np.where(np.isfinite(adv20) & (adv20 > 0.0), np.maximum(adv20, self._adv_eps()), self._adv_eps())
        info["turnover_rate_source"] = "amount_over_adv20_at_cost_observation"
        return amount / adv20_safe

    def _calibration_record(
        self,
        exact_key: tuple[str, str, str],
        amount_sigma_key: tuple[str, str],
        amount_key: tuple[str],
        min_bucket_samples: int,
    ) -> tuple[dict[str, Any], str] | None:
        exact = self.calibration_tables_.get("exact", {}).get(exact_key)
        if exact is not None and int(exact["sample_count"]) >= min_bucket_samples:
            return exact, ""
        amount_sigma = self.calibration_tables_.get("amount_sigma", {}).get(amount_sigma_key)
        if amount_sigma is not None and int(amount_sigma["sample_count"]) >= min_bucket_samples:
            return amount_sigma, "same_amount_sigma_bucket"
        amount = self.calibration_tables_.get("amount", {}).get(amount_key)
        if amount is not None and int(amount["sample_count"]) >= min_bucket_samples:
            return amount, "same_amount_bucket"
        return None

    def _empirical_variable_cost_per_asset(
        self,
        trade_weight: np.ndarray,
        execution_market_state: ExecutionMarketState,
        portfolio_state: PortfolioState,
        info: dict[str, Any],
    ) -> np.ndarray:
        slippage_per_asset = _non_negative_config_float("slippage", self.cost_config.get("slippage", 0.0)) * 0.5 * trade_weight
        if not bool(self.cost_config.get("market_impact_enabled", True)):
            return slippage_per_asset
        return slippage_per_asset + self._empirical_market_impact_per_asset(
            trade_weight,
            execution_market_state,
            portfolio_state,
            info,
        )

    def _empirical_market_impact_per_asset(
        self,
        trade_weight: np.ndarray,
        execution_market_state: ExecutionMarketState,
        portfolio_state: PortfolioState,
        info: dict[str, Any],
    ) -> np.ndarray:
        portfolio_value = _portfolio_value_required(portfolio_state)
        adv20 = _cost_observation_array(
            "adv20_at_cost_observation",
            "adv20_at_execution",
            execution_market_state,
            trade_weight.shape,
        )
        sigma20 = _cost_observation_array(
            "volatility_20d_at_cost_observation",
            "volatility_20d_at_execution",
            execution_market_state,
            trade_weight.shape,
        )
        adv_eps = self._adv_eps()
        volatility_eps = self._volatility_eps()
        valid_adv = np.isfinite(adv20) & (adv20 > 0.0)
        valid_sigma = np.isfinite(sigma20) & (sigma20 > 0.0)
        adv20_safe = np.where(valid_adv, np.maximum(adv20, adv_eps), adv_eps)
        sigma20_safe = np.where(valid_sigma, sigma20, volatility_eps)
        liquidity_ratio = trade_weight * portfolio_value / adv20_safe
        coef = _non_negative_config_float("market_impact_coef", self.cost_config.get("market_impact_coef", 0.0))
        if not valid_adv.all():
            info["adv20_fallback_index"] = np.flatnonzero(~valid_adv).astype(int).tolist()
        if not valid_sigma.all():
            info["volatility_20d_fallback_index"] = np.flatnonzero(~valid_sigma).astype(int).tolist()
        return coef * trade_weight * sigma20_safe * np.sqrt(np.maximum(liquidity_ratio, 0.0))

    def _write_calibration_report(self, report: pd.DataFrame, report_path: str | Path | None) -> None:
        output_path = Path(report_path) if report_path is not None else Path("logs/cost_calibration_report.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)


def _cost_model_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_CONFIG["cost_model"]
    if config is None:
        return dict(source)
    if "cost_model" in config:
        return {**source, **dict(config["cost_model"])}
    return {**source, **dict(config)}


def _execution_model_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_CONFIG["execution_model"]
    if config is None or "execution_model" not in config:
        return dict(source)
    return {**source, **dict(config["execution_model"])}


def _not_applicable_report() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "amount_bucket": "",
                "turnover_rate_bucket": "",
                "sigma20_bucket": "",
                "sample_count": 0,
                "realized_bps_mean": np.nan,
                "realized_bps_median": np.nan,
                "fallback_used": False,
                "fallback_reason": "",
                "status": "not_applicable",
            }
        ],
        columns=COST_CALIBRATION_REPORT_COLUMNS,
    )


def _empty_calibration_report() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "amount_bucket": "",
                "turnover_rate_bucket": "",
                "sigma20_bucket": "",
                "sample_count": 0,
                "realized_bps_mean": np.nan,
                "realized_bps_median": np.nan,
                "fallback_used": True,
                "fallback_reason": "no_train_samples",
                "status": "insufficient_sample",
            }
        ],
        columns=COST_CALIBRATION_REPORT_COLUMNS,
    )


def _calibration_samples(train_samples: pd.DataFrame | None) -> pd.DataFrame:
    if train_samples is None:
        return pd.DataFrame(columns=sorted(CALIBRATION_REQUIRED_COLUMNS))
    missing = sorted(CALIBRATION_REQUIRED_COLUMNS - set(train_samples.columns))
    if missing:
        raise DataContractError(
            "ERR_COST_CALIBRATION_SCHEMA_MISMATCH",
            f"ERR_COST_CALIBRATION_SCHEMA_MISMATCH: {missing}",
        )
    samples = train_samples.loc[:, ["amount", "turnover_rate", "sigma20", "realized_bps"]].copy()
    for column in samples.columns:
        samples[column] = pd.to_numeric(samples[column], errors="coerce")
    finite_mask = np.isfinite(samples.to_numpy(dtype=float)).all(axis=1)
    samples = samples.loc[finite_mask]
    samples = samples.loc[(samples["amount"] >= 0.0) & (samples["turnover_rate"] >= 0.0) & (samples["sigma20"] >= 0.0)]
    return samples.reset_index(drop=True)


def _fit_buckets(values: pd.Series) -> tuple[pd.Series, tuple[float, ...] | None]:
    series = pd.to_numeric(values, errors="coerce").astype(float)
    if series.nunique(dropna=True) <= 1:
        return pd.Series(["all"] * len(series), index=series.index, dtype=object), None
    q = min(3, int(series.nunique(dropna=True)))
    try:
        _, raw_bins = pd.qcut(series, q=q, retbins=True, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(series), index=series.index, dtype=object), None
    if len(raw_bins) <= 2:
        return pd.Series(["all"] * len(series), index=series.index, dtype=object), None
    bins = np.asarray(raw_bins, dtype=float)
    bins[0] = -np.inf
    bins[-1] = np.inf
    labels = [f"q{index}" for index in range(len(bins) - 1)]
    bucketed = pd.cut(series, bins=bins, labels=labels, include_lowest=True).astype(object).fillna("missing")
    return bucketed, tuple(float(value) for value in bins)


def _assign_bucket(value: float, bins: tuple[float, ...] | None) -> str:
    if bins is None:
        return "all"
    if not np.isfinite(value):
        return "missing"
    index = int(np.searchsorted(np.asarray(bins, dtype=float), value, side="right") - 1)
    index = max(0, min(index, len(bins) - 2))
    return f"q{index}"


def _calibration_report(bucketed: pd.DataFrame, min_bucket_samples: int) -> pd.DataFrame:
    grouped = _group_calibration(bucketed, ["amount_bucket", "turnover_rate_bucket", "sigma20_bucket"])
    grouped["fallback_used"] = grouped["sample_count"] < min_bucket_samples
    grouped["fallback_reason"] = np.where(grouped["fallback_used"], "sample_count_below_min_bucket_samples", "")
    grouped["status"] = np.where(grouped["fallback_used"], "insufficient_sample", "fitted")
    return grouped.reindex(columns=COST_CALIBRATION_REPORT_COLUMNS)


def _group_calibration(bucketed: pd.DataFrame, bucket_cols: list[str]) -> pd.DataFrame:
    return (
        bucketed.groupby(bucket_cols, dropna=False, observed=True)["realized_bps"]
        .agg(sample_count="count", realized_bps_mean="mean", realized_bps_median="median")
        .reset_index()
    )


def _lookup_table(bucketed: pd.DataFrame, bucket_cols: list[str]) -> dict[Any, dict[str, Any]]:
    grouped = _group_calibration(bucketed, bucket_cols)
    lookup: dict[Any, dict[str, Any]] = {}
    for record in grouped.to_dict("records"):
        key = tuple(str(record[column]) for column in bucket_cols)
        lookup[key] = record
    return lookup


def _array_1d(name: str, values: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_COST_SHAPE_MISMATCH", f"ERR_COST_SHAPE_MISMATCH: {name}") from exc
    if array.ndim != 1:
        raise DataContractError("ERR_COST_SHAPE_MISMATCH", f"ERR_COST_SHAPE_MISMATCH: {name}")
    if shape is not None and array.shape != shape:
        raise DataContractError("ERR_COST_SHAPE_MISMATCH", f"ERR_COST_SHAPE_MISMATCH: {name}")
    if not np.isfinite(array).all():
        raise DataContractError("ERR_COST_NON_FINITE", f"ERR_COST_NON_FINITE: {name}")
    return array


def _execution_array(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_COST_SHAPE_MISMATCH", f"ERR_COST_SHAPE_MISMATCH: {name}") from exc
    if array.ndim != 1 or array.shape != shape:
        raise DataContractError("ERR_COST_SHAPE_MISMATCH", f"ERR_COST_SHAPE_MISMATCH: {name}")
    return array


def _cost_observation_array(
    name: str,
    legacy_name: str,
    execution_market_state: ExecutionMarketState,
    shape: tuple[int, ...],
) -> np.ndarray:
    values = getattr(execution_market_state, name, None)
    if values is None:
        values = getattr(execution_market_state, legacy_name)
        return _execution_array(legacy_name, values, shape)
    return _execution_array(name, values, shape)


def _portfolio_value_required(portfolio_state: PortfolioState) -> float:
    value = getattr(portfolio_state, "portfolio_value", None)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError(
            "ERR_COST_PORTFOLIO_VALUE_REQUIRED",
            "ERR_COST_PORTFOLIO_VALUE_REQUIRED: portfolio_state.portfolio_value",
        ) from exc
    if not np.isfinite(result) or result <= 0.0:
        raise DataContractError(
            "ERR_COST_PORTFOLIO_VALUE_REQUIRED",
            "ERR_COST_PORTFOLIO_VALUE_REQUIRED: portfolio_state.portfolio_value",
        )
    return result


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_COST_NON_FINITE", f"ERR_COST_NON_FINITE: {name}") from exc
    if not np.isfinite(result):
        raise DataContractError("ERR_COST_NON_FINITE", f"ERR_COST_NON_FINITE: {name}")
    return result


def _non_negative_config_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise DataContractError("ERR_CONFIG_INVALID_COST_MODEL", f"ERR_CONFIG_INVALID_COST_MODEL: cost_model.{name}")
    return result


def _positive_config_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise DataContractError("ERR_CONFIG_INVALID_COST_MODEL", f"ERR_CONFIG_INVALID_COST_MODEL: cost_model.{name}")
    return result


__all__ = [
    "ADV_EPS",
    "GLOBAL_EPS",
    "PROBABILITY_EPS",
    "VOLATILITY_EPS",
    "WEIGHT_SUM_EPS",
    "CostBreakdown",
    "CostModel",
]
