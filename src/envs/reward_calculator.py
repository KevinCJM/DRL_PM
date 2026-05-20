from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.config import DEFAULT_CONFIG
from src.data.loader import DataContractError
from src.envs.state import ExecutionResult, PortfolioState


VALID_REWARD_VARIANTS = {
    "A0_raw_simple_return",
    "A1_log_return",
    "A2_net_log_return_after_cost",
    "A3_net_log_return_plus_turnover",
    "A4_net_log_return_plus_turnover_downside",
    "A5_net_log_return_plus_turnover_drawdown",
    "A6_net_log_return_plus_turnover_downside_drawdown",
    "A7_differential_sharpe",
    "A8_cvar_sensitive",
    "A9_benchmark_relative",
    "A10_ppo_lagrangian",
    "A11_regime_aware",
    "A12_multi_objective_preference_conditioned",
}
DEFAULT_REWARD_PARAMS = {
    "lambda_turnover": 0.001,
    "lambda_downside": 0.10,
    "lambda_drawdown": 0.20,
    "lambda_volatility": 0.05,
    "lambda_cvar": 0.10,
    "lambda_concentration": 0.02,
    "target_return_daily": 0.0,
    "drawdown_threshold": 0.10,
    "volatility_window": 20,
    "volatility_threshold_annual": 0.25,
    "cvar_window": 60,
    "cvar_confidence": 0.95,
    "cvar_loss_threshold": 0.0,
    "hhi_threshold": 0.20,
    "eta": 1.0 / 252.0,
    "differential_sharpe_eps": 1.0e-8,
    "differential_sharpe_warmup": 20,
}


class RewardCalculator:
    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.raw_config = config or DEFAULT_CONFIG
        self.reward_config = _reward_config(config)
        self.params = {**DEFAULT_REWARD_PARAMS, **self.reward_config}
        self.A_t = 0.0
        self.B_t = 0.0
        self.step_count = 0

    def reset_episode(self) -> None:
        self.A_t = 0.0
        self.B_t = 0.0
        self.step_count = 0

    def calculate(
        self,
        execution_result: ExecutionResult,
        portfolio_state: PortfolioState,
        *,
        reward_variant: str | None = None,
        benchmark_log_return: float | None = None,
        market_regime: str | None = None,
        omega: np.ndarray | None = None,
        reset_episode: bool = False,
        reward_context: Mapping[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        if reset_episode:
            self.reset_episode()
        context = dict(reward_context or {})
        variant = reward_variant or str(self.reward_config.get("mode", self.raw_config.get("env", {}).get("reward_mode", "A2_net_log_return_after_cost")))
        if variant not in VALID_REWARD_VARIANTS:
            raise DataContractError("ERR_CONFIG_INVALID_REWARD", f"ERR_CONFIG_INVALID_REWARD: reward.mode={variant}")

        params = self._variant_params(variant, market_regime)
        metrics = _reward_metrics(execution_result, portfolio_state, params, benchmark_log_return)
        reward_vector = _reward_vector(metrics)
        if variant == "A12_multi_objective_preference_conditioned" or bool(context.get("preference_conditioned", False)):
            reward = _preference_reward(reward_vector, omega)
            info = {"variant": variant, **metrics, "reward_vector": reward_vector.tolist(), "omega": _omega(omega, reward_vector).tolist()}
            return _finite_reward(reward), info

        if variant == "A0_raw_simple_return":
            reward = metrics["raw_return"]
        elif variant == "A1_log_return":
            reward = metrics["gross_log_return"]
        elif variant == "A2_net_log_return_after_cost":
            reward = metrics["net_log_return"]
        elif variant == "A3_net_log_return_plus_turnover":
            reward = metrics["net_log_return"] - params["lambda_turnover"] * metrics["turnover"]
        elif variant == "A4_net_log_return_plus_turnover_downside":
            reward = metrics["net_log_return"] - params["lambda_turnover"] * metrics["turnover"] - params["lambda_downside"] * metrics["downside_penalty"]
        elif variant == "A5_net_log_return_plus_turnover_drawdown":
            reward = metrics["net_log_return"] - params["lambda_turnover"] * metrics["turnover"] - params["lambda_drawdown"] * metrics["drawdown_penalty"]
        elif variant == "A6_net_log_return_plus_turnover_downside_drawdown":
            reward = (
                metrics["net_log_return"]
                - params["lambda_turnover"] * metrics["turnover"]
                - params["lambda_downside"] * metrics["downside_penalty"]
                - params["lambda_drawdown"] * metrics["drawdown_penalty"]
            )
        elif variant == "A7_differential_sharpe":
            reward = self._differential_sharpe(metrics["net_log_return"], params)
        elif variant == "A9_benchmark_relative":
            reward = self._penalized(metrics["benchmark_relative_log_return"], metrics, params)
        else:
            reward = self._penalized(metrics["net_log_return"], metrics, params)

        info = {"variant": variant, **metrics}
        if variant == "A10_ppo_lagrangian":
            info["lagrangian_violation_scalar"] = _lagrangian_violation_scalar(context)
        if variant == "A7_differential_sharpe":
            info.update({"differential_sharpe_A": self.A_t, "differential_sharpe_B": self.B_t, "differential_sharpe_step": self.step_count})
        return _finite_reward(reward), info

    def _variant_params(self, variant: str, market_regime: str | None) -> dict[str, Any]:
        params = dict(self.params)
        if variant == "A8_cvar_sensitive":
            params["cvar_confidence"] = 0.95
        if variant == "A11_regime_aware" and market_regime is not None:
            regime_params = self.reward_config.get("regime_params", {})
            params.update(dict(regime_params.get("default", {})))
            params.update(dict(regime_params.get(str(market_regime), {})))
        return params

    def _penalized(self, base_reward: float, metrics: Mapping[str, float], params: Mapping[str, Any]) -> float:
        return (
            base_reward
            - float(params["lambda_turnover"]) * metrics["turnover"]
            - float(params["lambda_downside"]) * metrics["downside_penalty"]
            - float(params["lambda_drawdown"]) * metrics["drawdown_penalty"]
            - float(params["lambda_volatility"]) * metrics["volatility_penalty"]
            - float(params["lambda_cvar"]) * metrics["cvar_penalty"]
            - float(params["lambda_concentration"]) * metrics["concentration_penalty"]
        )

    def _differential_sharpe(self, net_log_return: float, params: Mapping[str, Any]) -> float:
        self.step_count += 1
        eta = float(params["eta"])
        delta_a = net_log_return - self.A_t
        delta_b = net_log_return * net_log_return - self.B_t
        variance = max(self.B_t - self.A_t * self.A_t, 0.0)
        denominator = max(variance ** 1.5, float(params["differential_sharpe_eps"]))
        differential = (self.B_t * delta_a - 0.5 * self.A_t * delta_b) / denominator
        self.A_t += eta * delta_a
        self.B_t += eta * delta_b
        if self.step_count <= int(params["differential_sharpe_warmup"]):
            return net_log_return
        return differential


def _reward_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    source = DEFAULT_CONFIG["reward"]
    if config is None or "reward" not in config:
        return dict(source)
    return {**source, **dict(config["reward"])}


def _reward_metrics(
    execution_result: ExecutionResult,
    portfolio_state: PortfolioState,
    params: Mapping[str, Any],
    benchmark_log_return: float | None,
) -> dict[str, float]:
    net_return = _finite_float("net_return", execution_result.net_return)
    net_log_return = _finite_float("portfolio_log_return", execution_result.portfolio_log_return)
    gross_return = _finite_float("gross_return", execution_result.gross_return)
    current_drawdown = _finite_float("current_drawdown_abs", portfolio_state.current_drawdown_abs)
    returns = _rolling_returns(portfolio_state, net_return)
    rolling_volatility = _rolling_volatility_annual(returns, int(params["volatility_window"]))
    cvar_loss = _cvar_loss(returns, int(params["cvar_window"]), float(params["cvar_confidence"]))
    hhi = float(np.sum(np.square(execution_result.executed_weights)))
    return {
        "raw_return": gross_return,
        "gross_log_return": float(np.log1p(gross_return)),
        "net_return": net_return,
        "net_log_return": net_log_return,
        "turnover": _finite_float("turnover", execution_result.turnover),
        "transaction_cost": _finite_float("transaction_cost", execution_result.transaction_cost),
        "downside_penalty": max(0.0, float(params["target_return_daily"]) - net_return) ** 2,
        "drawdown_penalty": max(0.0, current_drawdown - float(params["drawdown_threshold"])),
        "rolling_volatility": rolling_volatility,
        "volatility_penalty": max(0.0, rolling_volatility - float(params["volatility_threshold_annual"])),
        "cvar_confidence": float(params["cvar_confidence"]),
        "cvar_alpha": 1.0 - float(params["cvar_confidence"]),
        "cvar_loss": cvar_loss,
        "cvar_penalty": max(0.0, cvar_loss - float(params["cvar_loss_threshold"])),
        "hhi": hhi,
        "concentration_penalty": max(0.0, hhi - float(params["hhi_threshold"])),
        "benchmark_log_return": 0.0 if benchmark_log_return is None else _finite_float("benchmark_log_return", benchmark_log_return),
        "benchmark_relative_log_return": net_log_return - (0.0 if benchmark_log_return is None else float(benchmark_log_return)),
    }


def _rolling_returns(portfolio_state: PortfolioState, net_return: float) -> np.ndarray:
    values = list(portfolio_state.rolling_returns or [])
    if not values or abs(float(values[-1]) - net_return) > 1.0e-12:
        values.append(net_return)
    return np.asarray(values, dtype=float)


def _rolling_volatility_annual(returns: np.ndarray, window: int) -> float:
    if returns.size <= 1:
        return 0.0
    windowed = returns[-max(1, window) :]
    return float(np.std(windowed, ddof=0) * np.sqrt(252.0))


def _cvar_loss(returns: np.ndarray, window: int, confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise DataContractError("ERR_CONFIG_INVALID_REWARD", "ERR_CONFIG_INVALID_REWARD: reward.cvar_confidence")
    windowed = np.sort(returns[-max(1, window) :])
    tail_count = max(1, int(np.ceil(windowed.size * (1.0 - confidence))))
    return max(0.0, -float(np.mean(windowed[:tail_count])))


def _reward_vector(metrics: Mapping[str, float]) -> np.ndarray:
    return np.array(
        [
            metrics["net_log_return"],
            -metrics["turnover"],
            -metrics["downside_penalty"],
            -metrics["drawdown_penalty"],
            -metrics["volatility_penalty"],
            -metrics["cvar_penalty"],
            -metrics["concentration_penalty"],
        ],
        dtype=float,
    )


def _omega(omega: np.ndarray | None, reward_vector: np.ndarray) -> np.ndarray:
    if omega is None:
        result = np.zeros_like(reward_vector, dtype=float)
        result[0] = 1.0
        return result
    result = np.asarray(omega, dtype=float)
    if result.ndim == 1 and result.shape[0] == 5 and reward_vector.shape[0] == 7:
        expanded = np.zeros_like(reward_vector, dtype=float)
        expanded[0] = result[0]
        expanded[1] = result[1]
        expanded[3] = result[2]
        expanded[5] = result[3]
        expanded[6] = result[4]
        result = expanded
    if result.ndim != 1 or result.shape != reward_vector.shape or not np.isfinite(result).all():
        raise DataContractError("ERR_CONFIG_INVALID_REWARD", "ERR_CONFIG_INVALID_REWARD: preference.omega")
    return result


def _preference_reward(reward_vector: np.ndarray, omega: np.ndarray | None) -> float:
    return float(np.dot(_omega(omega, reward_vector), reward_vector))


def _lagrangian_violation_scalar(context: Mapping[str, Any]) -> float:
    violations = context.get("constraint_violations", [])
    total = 0.0
    for record in violations:
        if isinstance(record, Mapping):
            total += float(record.get("lagrangian_violation_scalar", record.get("violation_scalar", 0.0)))
    return total


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataContractError("ERR_REWARD_NON_FINITE", f"ERR_REWARD_NON_FINITE: {name}") from exc
    if not np.isfinite(result):
        raise DataContractError("ERR_REWARD_NON_FINITE", f"ERR_REWARD_NON_FINITE: {name}")
    return result


def _finite_reward(value: float) -> float:
    reward = _finite_float("reward", value)
    return reward


__all__ = ["RewardCalculator", "VALID_REWARD_VARIANTS"]
