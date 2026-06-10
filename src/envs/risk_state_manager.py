from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.envs.state import ExecutionResult, PortfolioState


class RiskStateManager:
    def __init__(self, config: Mapping[str, Any]) -> None:
        risk_cfg = config.get("risk_state", {})
        self._ewma_span_downside_vol = int(risk_cfg.get("ewma_span_downside_vol", 60))
        self._ewma_span_downside_return = int(risk_cfg.get("ewma_span_downside_return", 60))
        self._cvar_window = int(risk_cfg.get("cvar_window", 60))
        self._alpha_vol = 2.0 / (self._ewma_span_downside_vol + 1.0)
        self._alpha_return = 2.0 / (self._ewma_span_downside_return + 1.0)
        self.reset()

    def reset(self) -> None:
        self._downside_vol_ewma = 0.0
        self._downside_return_ewma = 0.0
        self._days_since_last_rebalance = 0
        self._prev_turnover = 0.0
        self._prev_cost = 0.0
        self._last_net_return: float | None = None
        self._rolling_returns: list[float] = []
        self._drawdown_abs = 0.0
        self._drawdown_increment = 0.0
        self._soft_cvar_loss_state = 0.0
        self._update_count = 0

    @property
    def drawdown_increment(self) -> float:
        return self._drawdown_increment

    def update_pre_reward(
        self,
        execution_result: ExecutionResult,
        portfolio_state: PortfolioState,
        final_action: int,
    ) -> None:
        current_drawdown = float(portfolio_state.current_drawdown_abs)
        self._drawdown_increment = max(0.0, current_drawdown - self._drawdown_abs)
        self._drawdown_abs = current_drawdown

        if final_action == 1:
            self._days_since_last_rebalance = 0
        else:
            self._days_since_last_rebalance += 1

        self._prev_turnover = float(execution_result.turnover)
        self._prev_cost = float(execution_result.transaction_cost)

        net_return = float(execution_result.net_return)
        self._last_net_return = net_return
        self._rolling_returns.append(net_return)
        if len(self._rolling_returns) > self._cvar_window:
            self._rolling_returns = self._rolling_returns[-self._cvar_window:]

        if net_return < 0.0:
            self._downside_vol_ewma = (
                self._alpha_vol * net_return ** 2
                + (1.0 - self._alpha_vol) * self._downside_vol_ewma
            )
            self._downside_return_ewma = (
                self._alpha_return * net_return
                + (1.0 - self._alpha_return) * self._downside_return_ewma
            )

    def update_reward_info(self, reward_info: Mapping[str, Any] | None) -> None:
        reward_info = reward_info or {}
        if "soft_cvar_loss_state" in reward_info:
            self._soft_cvar_loss_state = float(reward_info["soft_cvar_loss_state"])
        else:
            self._soft_cvar_loss_state = _cvar_loss(
                np.asarray(self._rolling_returns, dtype=float),
                window=self._cvar_window,
                confidence=0.95,
            )
        self._update_count += 1

    def get_observation_vector(self) -> np.ndarray:
        return np.array(
            [
                self._downside_vol_ewma,
                self._downside_return_ewma,
                self._soft_cvar_loss_state,
                self._drawdown_abs,
                self._drawdown_increment,
                self._prev_turnover,
                self._prev_cost,
                float(self._days_since_last_rebalance),
            ],
            dtype=np.float64,
        )

    def get_diagnostics_dict(self) -> dict[str, Any]:
        is_warmup = self._update_count == 0
        return {
            "downside_vol_ewma": float("nan") if is_warmup else self._downside_vol_ewma,
            "downside_return_ewma": float("nan") if is_warmup else self._downside_return_ewma,
            "soft_cvar_loss_state": float("nan") if is_warmup else self._soft_cvar_loss_state,
            "drawdown_abs": self._drawdown_abs,
            "drawdown_increment": self._drawdown_increment,
            "turnover_prev": self._prev_turnover,
            "cost_prev": self._prev_cost,
            "days_since_last_rebalance": self._days_since_last_rebalance,
        }


def _cvar_loss(returns: np.ndarray, window: int, confidence: float) -> float:
    if returns.size == 0:
        return 0.0
    windowed = np.sort(returns[-max(1, window):])
    tail_count = max(1, int(np.ceil(windowed.size * (1.0 - confidence))))
    return max(0.0, -float(np.mean(windowed[:tail_count])))


__all__ = ["RiskStateManager"]
