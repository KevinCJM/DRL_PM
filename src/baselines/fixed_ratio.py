from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.baselines.base_strategy import TraditionalStrategyBase
from src.data.loader import DataContractError
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


VOL_EPS = 1.0e-12


class FixedRatioStrategy(TraditionalStrategyBase):
    strategy_name = "fixed_ratio"

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        self._fixed_ratio_report: dict[str, Any] = {}
        action = super().compute_target_weights(decision_market_state, portfolio_state)
        report = dict(self._fixed_ratio_report)
        report["projected_weights"] = action.target_weights.astype(float).tolist()
        action.action_info["fixed_ratio_weights"] = report
        return action

    def _raw_weights(self, decision_market_state: DecisionMarketState) -> np.ndarray:
        config = _fixed_ratio_config(self.config)
        available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
        allocation_mode = _allocation_mode(config)
        if allocation_mode == "asset_weights":
            raw_weights, report = _asset_weights(config, available.shape[0])
        elif allocation_mode == "asset_class_bucket":
            raw_weights, report = _asset_class_weights(config, decision_market_state, self.config)
        elif allocation_mode == "equal_weight":
            raw_weights = np.zeros(available.shape, dtype=float)
            if available.any():
                raw_weights[available] = 1.0 / int(available.sum())
            report = {"allocation_mode": allocation_mode}
        else:
            raise DataContractError(
                "ERR_STRATEGY_CONFIG_INVALID",
                "ERR_STRATEGY_CONFIG_INVALID: fixed_ratio.allocation_mode",
            )
        self._fixed_ratio_report = {**report, "raw_weights": raw_weights.astype(float).tolist()}
        return raw_weights


def _fixed_ratio_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("fixed_ratio"), Mapping):
        result = dict(config["fixed_ratio"])
        for key in ("asset_ids",):
            if key not in result and key in config:
                result[key] = config[key]
        return result
    return dict(config)


def _allocation_mode(config: Mapping[str, Any]) -> str:
    mode = config.get("allocation_mode")
    if mode is not None:
        return str(mode)
    if any(key in config for key in ("asset_weights", "fixed_ratio_weights", "weights")):
        return "asset_weights"
    if "asset_class_weights" in config:
        return "asset_class_bucket"
    return "equal_weight"


def _asset_weights(config: Mapping[str, Any], n_assets: int) -> tuple[np.ndarray, dict[str, Any]]:
    raw_value = _first_present(config, ("asset_weights", "fixed_ratio_weights", "weights"))
    if isinstance(raw_value, Mapping):
        asset_ids = _asset_ids(config, n_assets)
        weights = np.array([float(raw_value.get(asset_id, 0.0)) for asset_id in asset_ids], dtype=float)
        report_asset_ids = asset_ids
    else:
        weights = np.asarray(raw_value, dtype=float)
        if weights.shape != (n_assets,):
            raise DataContractError(
                "ERR_STRATEGY_CONFIG_INVALID",
                "ERR_STRATEGY_CONFIG_INVALID: fixed_ratio.asset_weights shape",
            )
        report_asset_ids = _optional_asset_ids(config, n_assets)
    _validate_non_negative_weights(weights, "fixed_ratio.asset_weights")
    return weights, {
        "allocation_mode": "asset_weights",
        "configured_asset_weights": weights.astype(float).tolist(),
        "asset_ids": report_asset_ids,
    }


def _asset_class_weights(
    config: Mapping[str, Any],
    decision_market_state: DecisionMarketState,
    root_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    available = np.asarray(decision_market_state.available_mask_at_decision, dtype=bool)
    class_weights = _class_weights(config)
    classes = _asset_classes(config, root_config, available.shape[0])
    intra_bucket_mode = str(config.get("intra_bucket_mode", "equal_weight"))
    raw_weights = np.zeros(available.shape, dtype=float)
    active_classes: dict[str, float] = {}
    unavailable_classes: list[str] = []
    for asset_class, configured_weight in class_weights.items():
        class_mask = np.array([value == asset_class for value in classes], dtype=bool)
        if (class_mask & available).any():
            active_classes[asset_class] = configured_weight
        else:
            unavailable_classes.append(asset_class)

    total_active_weight = float(sum(active_classes.values()))
    if total_active_weight > 0.0:
        for asset_class, configured_weight in active_classes.items():
            class_weight = configured_weight / total_active_weight
            class_mask = np.array([value == asset_class for value in classes], dtype=bool) & available
            raw_weights[class_mask] = _intra_bucket_weights(
                class_weight,
                class_mask,
                decision_market_state,
                intra_bucket_mode,
            )[class_mask]

    return raw_weights, {
        "allocation_mode": "asset_class_bucket",
        "intra_bucket_mode": intra_bucket_mode,
        "configured_asset_class_weights": dict(class_weights),
        "effective_asset_class_weights": {
            asset_class: weight / total_active_weight
            for asset_class, weight in active_classes.items()
        }
        if total_active_weight > 0.0
        else {},
        "unavailable_asset_classes": unavailable_classes,
        "asset_classes": classes,
        "asset_ids": _optional_asset_ids(config, available.shape[0]),
    }


def _intra_bucket_weights(
    class_weight: float,
    class_mask: np.ndarray,
    decision_market_state: DecisionMarketState,
    intra_bucket_mode: str,
) -> np.ndarray:
    weights = np.zeros(class_mask.shape, dtype=float)
    if intra_bucket_mode == "equal_weight":
        weights[class_mask] = class_weight / int(class_mask.sum())
        return weights
    if intra_bucket_mode == "inverse_volatility":
        volatility = np.asarray(decision_market_state.volatility_20d_at_decision, dtype=float)
        inverse = np.zeros(class_mask.shape, dtype=float)
        valid = class_mask & np.isfinite(volatility) & (volatility > 0.0)
        inverse[valid] = 1.0 / np.maximum(volatility[valid], VOL_EPS)
        inverse_sum = float(inverse[class_mask].sum())
        if inverse_sum <= 0.0:
            weights[class_mask] = class_weight / int(class_mask.sum())
        else:
            weights[class_mask] = class_weight * inverse[class_mask] / inverse_sum
        return weights
    raise DataContractError(
        "ERR_STRATEGY_CONFIG_INVALID",
        "ERR_STRATEGY_CONFIG_INVALID: fixed_ratio.intra_bucket_mode",
    )


def _class_weights(config: Mapping[str, Any]) -> dict[str, float]:
    raw_weights = config.get("asset_class_weights")
    if not isinstance(raw_weights, Mapping) or not raw_weights:
        raise DataContractError(
            "ERR_STRATEGY_CONFIG_INVALID",
            "ERR_STRATEGY_CONFIG_INVALID: fixed_ratio.asset_class_weights",
        )
    result = {str(asset_class): float(weight) for asset_class, weight in raw_weights.items()}
    _validate_non_negative_weights(np.array(list(result.values()), dtype=float), "fixed_ratio.asset_class_weights")
    return result


def _asset_classes(config: Mapping[str, Any], root_config: Mapping[str, Any], n_assets: int) -> list[str]:
    if isinstance(config.get("asset_classes"), Sequence) and not isinstance(config.get("asset_classes"), str):
        classes = [str(value) for value in config["asset_classes"]]
        if len(classes) != n_assets:
            raise DataContractError(
                "ERR_STRATEGY_CONFIG_INVALID",
                "ERR_STRATEGY_CONFIG_INVALID: fixed_ratio.asset_classes shape",
            )
        return classes

    mapping = config.get("asset_class_mapping")
    if mapping is None and isinstance(root_config.get("constraints"), Mapping):
        mapping = root_config["constraints"].get("asset_class_mapping")
    if not isinstance(mapping, Mapping) or not mapping:
        raise DataContractError(
            "ERR_STRATEGY_CONFIG_INVALID",
            "ERR_STRATEGY_CONFIG_INVALID: fixed_ratio.asset_class_mapping",
        )
    asset_ids = _asset_ids(config, n_assets)
    return [str(mapping.get(asset_id, "")) for asset_id in asset_ids]


def _asset_ids(config: Mapping[str, Any], n_assets: int) -> list[str]:
    asset_ids = _optional_asset_ids(config, n_assets)
    if not asset_ids:
        raise DataContractError(
            "ERR_STRATEGY_CONFIG_INVALID",
            "ERR_STRATEGY_CONFIG_INVALID: fixed_ratio.asset_ids",
        )
    return asset_ids


def _optional_asset_ids(config: Mapping[str, Any], n_assets: int) -> list[str]:
    raw_asset_ids = config.get("asset_ids")
    if raw_asset_ids is None:
        return []
    asset_ids = [str(value) for value in raw_asset_ids]
    if len(asset_ids) != n_assets:
        raise DataContractError(
            "ERR_STRATEGY_CONFIG_INVALID",
            "ERR_STRATEGY_CONFIG_INVALID: fixed_ratio.asset_ids shape",
        )
    return asset_ids


def _first_present(config: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    raise DataContractError(
        "ERR_STRATEGY_CONFIG_INVALID",
        "ERR_STRATEGY_CONFIG_INVALID: fixed_ratio.asset_weights",
    )


def _validate_non_negative_weights(weights: np.ndarray, key_path: str) -> None:
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise DataContractError(
            "ERR_STRATEGY_CONFIG_INVALID",
            f"ERR_STRATEGY_CONFIG_INVALID: {key_path}",
        )


__all__ = ["FixedRatioStrategy"]
