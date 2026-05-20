from __future__ import annotations

from typing import Any, Mapping

import torch


class CostEstimator:
    @staticmethod
    def estimate(
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        adv20: torch.Tensor,
        sigma20: torch.Tensor,
        portfolio_value: float,
        config: Mapping[str, Any],
        amount: torch.Tensor | None = None,
        turnover_rate: torch.Tensor | None = None,
        calibration_table: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate transaction cost based on DecisionMarketState visible fields.
        All inputs are torch tensors [batch, n_assets].
        """
        candidate_weights = _matrix("candidate_weights", candidate_weights, finite=True)
        current_weights = _matrix("current_weights", current_weights, candidate_weights.shape, finite=True)
        adv20 = _matrix("adv20", adv20, candidate_weights.shape, finite=False, like=candidate_weights)
        sigma20 = _matrix("sigma20", sigma20, candidate_weights.shape, finite=False, like=candidate_weights)
        amount = (
            _matrix("amount", amount, candidate_weights.shape, finite=False, like=candidate_weights)
            if amount is not None
            else torch.zeros_like(candidate_weights)
        )
        turnover_rate = (
            _matrix("turnover_rate", turnover_rate, candidate_weights.shape, finite=False, like=candidate_weights)
            if turnover_rate is not None
            else _safe_divide(amount, _safe_adv20(adv20, _cost_config(config)))
        )
        portfolio_value_tensor = _portfolio_value_tensor(
            portfolio_value,
            candidate_weights.shape[0],
            candidate_weights,
        )

        trade_weights = torch.abs(candidate_weights - current_weights)
        turnover = 0.5 * torch.sum(trade_weights, dim=1, keepdim=True)

        cost_config = _cost_config(config)
        mode = str(cost_config.get("mode", "empirical_default"))
        proportional_cost = _non_negative_config_float(cost_config, "proportional_cost", 0.001) * turnover
        fixed_cost = _fixed_cost(config, cost_config, portfolio_value_tensor, turnover)

        if mode == "empirical_default":
            variable_cost = _empirical_variable_cost(trade_weights, adv20, sigma20, portfolio_value_tensor, cost_config)
        elif mode == "calibrated":
            variable_cost = _calibrated_variable_cost(
                trade_weights,
                amount,
                turnover_rate,
                sigma20,
                cost_config,
                calibration_table,
                like=candidate_weights,
            )
        else:
            raise ValueError("ERR_CONFIG_INVALID_COST_MODE: cost_model.mode")

        total_cost = proportional_cost + fixed_cost + variable_cost
        total_cost = torch.where(turnover > 0.0, total_cost, torch.zeros_like(total_cost))
        return turnover, total_cost

    @staticmethod
    def estimate_from_decision_state(
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        decision_market_state: Any,
        portfolio_state: Any,
        config: Mapping[str, Any],
        calibration_table: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_weights = _matrix("candidate_weights", candidate_weights, finite=True)
        current_weights = _matrix("current_weights", current_weights, candidate_weights.shape, finite=True)
        device = candidate_weights.device
        dtype = candidate_weights.dtype
        return CostEstimator.estimate(
            candidate_weights,
            current_weights,
            _batch_decision_array("adv20_at_decision", decision_market_state, candidate_weights.shape, device, dtype),
            _batch_decision_array(
                "volatility_20d_at_decision",
                decision_market_state,
                candidate_weights.shape,
                device,
                dtype,
            ),
            getattr(portfolio_state, "portfolio_value", None),
            config,
            amount=_batch_decision_array("amount_at_decision", decision_market_state, candidate_weights.shape, device, dtype),
            turnover_rate=_batch_decision_array(
                "turnover_rate_at_decision",
                decision_market_state,
                candidate_weights.shape,
                device,
                dtype,
            ),
            calibration_table=calibration_table,
        )


def _cost_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("cost_model", config))


def _execution_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("execution_model", {}))


def _matrix(
    name: str,
    value: Any,
    shape: tuple[int, int] | torch.Size | None = None,
    *,
    finite: bool,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    device = like.device if like is not None else None
    dtype = like.dtype if like is not None else torch.float32
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if shape is not None:
        shape = tuple(int(dim) for dim in shape)
        if tensor.ndim == 0:
            tensor = tensor.expand(shape)
        elif tensor.ndim == 1 and tensor.shape[0] == shape[1]:
            tensor = tensor.unsqueeze(0).expand(shape)
    if tensor.ndim != 2:
        raise ValueError(f"ERR_COST_ESTIMATOR_SHAPE_MISMATCH: {name} must be [batch,n_assets]")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"ERR_COST_ESTIMATOR_SHAPE_MISMATCH: {name}")
    if finite and not torch.isfinite(tensor).all():
        raise ValueError(f"ERR_COST_ESTIMATOR_NON_FINITE: {name}")
    return tensor


def _batch_decision_array(
    name: str,
    decision_market_state: Any,
    shape: tuple[int, int] | torch.Size,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = getattr(decision_market_state, name, None)
    if value is None:
        raise ValueError(f"ERR_COST_ESTIMATOR_DECISION_FIELD_MISSING: {name}")
    return _matrix(name, value, shape, finite=False, like=torch.empty(0, device=device, dtype=dtype))


def _portfolio_value_tensor(portfolio_value: Any, batch_size: int, like: torch.Tensor) -> torch.Tensor:
    try:
        value = torch.as_tensor(portfolio_value, device=like.device, dtype=like.dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("ERR_COST_PORTFOLIO_VALUE_REQUIRED: portfolio_value") from exc
    if value.ndim == 0:
        value = value.view(1, 1).expand(batch_size, 1)
    elif value.ndim == 1 and value.shape[0] == batch_size:
        value = value.view(batch_size, 1)
    elif value.ndim == 2 and tuple(value.shape) == (batch_size, 1):
        pass
    else:
        raise ValueError("ERR_COST_ESTIMATOR_SHAPE_MISMATCH: portfolio_value")
    if not torch.isfinite(value).all() or (value <= 0.0).any():
        raise ValueError("ERR_COST_PORTFOLIO_VALUE_REQUIRED: portfolio_value")
    return value


def _fixed_cost(
    config: Mapping[str, Any],
    cost_config: Mapping[str, Any],
    portfolio_value: torch.Tensor,
    turnover: torch.Tensor,
) -> torch.Tensor:
    fixed_cost_value = _non_negative_config_float(cost_config, "fixed_cost", 0.0)
    if fixed_cost_value == 0.0:
        return torch.zeros_like(turnover)

    fixed_cost_unit = str(_execution_config(config).get("fixed_cost_unit", "nav_fraction"))
    if fixed_cost_unit == "nav_fraction":
        fixed_cost = torch.full_like(turnover, fixed_cost_value)
    elif fixed_cost_unit == "currency":
        fixed_cost = fixed_cost_value / portfolio_value
    else:
        raise ValueError("ERR_CONFIG_INVALID_FIXED_COST_UNIT: execution_model.fixed_cost_unit")
    return torch.where(turnover > 0.0, fixed_cost, torch.zeros_like(fixed_cost))


def _empirical_variable_cost(
    trade_weights: torch.Tensor,
    adv20: torch.Tensor,
    sigma20: torch.Tensor,
    portfolio_value: torch.Tensor,
    cost_config: Mapping[str, Any],
) -> torch.Tensor:
    turnover = 0.5 * torch.sum(trade_weights, dim=1, keepdim=True)
    slippage_cost = _non_negative_config_float(cost_config, "slippage", 0.0005) * turnover
    if not bool(cost_config.get("market_impact_enabled", True)):
        return slippage_cost

    adv20_safe = _safe_adv20(adv20, cost_config)
    sigma20_safe = _safe_sigma20(sigma20, cost_config)
    liquidity_ratio = trade_weights * portfolio_value / adv20_safe
    coef = _non_negative_config_float(cost_config, "market_impact_coef", 0.1)
    market_impact = coef * trade_weights * sigma20_safe * torch.sqrt(torch.clamp(liquidity_ratio, min=0.0))
    return slippage_cost + torch.sum(market_impact, dim=1, keepdim=True)


def _calibrated_variable_cost(
    trade_weights: torch.Tensor,
    amount: torch.Tensor,
    turnover_rate: torch.Tensor,
    sigma20: torch.Tensor,
    cost_config: Mapping[str, Any],
    calibration_table: Any | None,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    table = calibration_table
    if table is None:
        table = cost_config.get("train_fitted_calibration_table")
    if table is None:
        table = cost_config.get("calibration_table")
    if table is None:
        raise ValueError("ERR_COST_CALIBRATION_NOT_FITTED: train_fitted_calibration_table")
    bps = _calibration_bps(table, amount, turnover_rate, sigma20, trade_weights, cost_config, like)
    return torch.sum((bps / 10000.0) * trade_weights, dim=1, keepdim=True)


def _calibration_bps(
    table: Any,
    amount: torch.Tensor,
    turnover_rate: torch.Tensor,
    sigma20: torch.Tensor,
    trade_weights: torch.Tensor,
    cost_config: Mapping[str, Any],
    like: torch.Tensor,
) -> torch.Tensor:
    if isinstance(table, Mapping) and "tables" in table:
        return _bucketed_calibration_bps(table, amount, turnover_rate, sigma20, trade_weights, cost_config, like)
    if isinstance(table, Mapping):
        for key in ("realized_bps_median", "median_bps", "per_asset_bps", "bps", "default_bps"):
            if key in table:
                return _matrix(f"calibration_table.{key}", table[key], trade_weights.shape, finite=True, like=like)
    return _matrix("calibration_table", table, trade_weights.shape, finite=True, like=like)


def _bucketed_calibration_bps(
    table: Mapping[str, Any],
    amount: torch.Tensor,
    turnover_rate: torch.Tensor,
    sigma20: torch.Tensor,
    trade_weights: torch.Tensor,
    cost_config: Mapping[str, Any],
    like: torch.Tensor,
) -> torch.Tensor:
    bins = table.get("bins", {})
    tables = table.get("tables", {})
    min_samples = int(table.get("min_bucket_samples", cost_config.get("calibration", {}).get("min_bucket_samples", 30)))
    default_bps = table.get("default_bps", None)
    bps = torch.empty_like(trade_weights)
    flat_amount = amount.detach().cpu().reshape(-1)
    flat_turnover = turnover_rate.detach().cpu().reshape(-1)
    flat_sigma = sigma20.detach().cpu().reshape(-1)
    flat_bps = bps.reshape(-1)
    for index in range(flat_bps.numel()):
        amount_bucket = _assign_bucket(float(flat_amount[index]), bins.get("amount"))
        turnover_bucket = _assign_bucket(float(flat_turnover[index]), bins.get("turnover_rate"))
        sigma_bucket = _assign_bucket(float(flat_sigma[index]), bins.get("sigma20"))
        record = _lookup_calibration_record(
            tables,
            (amount_bucket, turnover_bucket, sigma_bucket),
            (amount_bucket, sigma_bucket),
            (amount_bucket,),
            min_samples,
        )
        if record is None:
            if default_bps is None:
                raise ValueError("ERR_COST_CALIBRATION_BUCKET_MISSING: train_fitted_calibration_table")
            flat_bps[index] = torch.as_tensor(default_bps, device=like.device, dtype=like.dtype)
            continue
        flat_bps[index] = float(record["realized_bps_median"])
    return bps


def _lookup_calibration_record(
    tables: Mapping[str, Any],
    exact_key: tuple[str, str, str],
    amount_sigma_key: tuple[str, str],
    amount_key: tuple[str],
    min_samples: int,
) -> Mapping[str, Any] | None:
    for table_name, key in (("exact", exact_key), ("amount_sigma", amount_sigma_key), ("amount", amount_key)):
        record = tables.get(table_name, {}).get(key)
        if record is not None and int(record.get("sample_count", min_samples)) >= min_samples:
            return record
    return None


def _assign_bucket(value: float, bins: Any) -> str:
    if bins is None:
        return "all"
    tensor = torch.as_tensor(bins, dtype=torch.float64)
    if tensor.numel() < 2:
        return "all"
    if not torch.isfinite(torch.as_tensor(value, dtype=torch.float64)):
        return "missing"
    index = int(torch.searchsorted(tensor, torch.as_tensor(value, dtype=torch.float64), right=True).item() - 1)
    index = max(0, min(index, int(tensor.numel()) - 2))
    return f"q{index}"


def _safe_adv20(adv20: torch.Tensor, cost_config: Mapping[str, Any]) -> torch.Tensor:
    adv_eps = _positive_config_float(cost_config, "adv_eps", 1000000.0)
    adv_eps_tensor = torch.as_tensor(adv_eps, device=adv20.device, dtype=adv20.dtype)
    return torch.where(torch.isfinite(adv20) & (adv20 > 0.0), torch.maximum(adv20, adv_eps_tensor), adv_eps_tensor)


def _safe_sigma20(sigma20: torch.Tensor, cost_config: Mapping[str, Any]) -> torch.Tensor:
    volatility_eps = _positive_config_float(cost_config, "volatility_eps", 1.0e-8)
    eps_tensor = torch.as_tensor(volatility_eps, device=sigma20.device, dtype=sigma20.dtype)
    return torch.where(torch.isfinite(sigma20) & (sigma20 > 0.0), sigma20, eps_tensor)


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / torch.where(denominator > 0.0, denominator, torch.ones_like(denominator))


def _non_negative_config_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if value < 0.0:
        raise ValueError(f"ERR_COST_CONFIG_INVALID: {key}")
    return value


def _positive_config_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = _non_negative_config_float(config, key, default)
    if value <= 0.0:
        raise ValueError(f"ERR_COST_CONFIG_INVALID: {key}")
    return value


__all__ = ["CostEstimator"]
