from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.base_strategy import BaseStrategy
from src.baselines.cage_common import (
    DEFAULT_GT_RCPOLITE_RHOS,
    MODEL_EXTENSION_ID,
    choose_rho,
    decision_return_features,
    estimate_cost,
    estimate_turnover,
    gate_action_index,
    mapping,
    new_model_training_result,
    normalize_candidate,
    rho_grid,
    weights_json,
)
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


GT_RCPO_LITE_ALGORITHM = "graph_transformer_risk_constrained_actor_critic_lite"


class GTRCPOLiteStrategy(BaseStrategy):
    strategy_name = "graph_transformer_risk_constrained_actor_critic_lite"
    fit_required = True
    requires_daily_diagnostics = True

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.rho_values = rho_grid(self.config, "gt_rcpo_lite", DEFAULT_GT_RCPOLITE_RHOS)
        self.training_result: dict[str, Any] | None = None
        self.training_history: pd.DataFrame = pd.DataFrame()

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> GTRCPOLiteStrategy:
        native_cfg = mapping(mapping(self.config.get("baselines")).get("native_rl"))
        train_steps = int(native_cfg.get("max_train_steps") or mapping(self.config.get("training")).get("max_train_steps") or 1)
        validation_steps = int(
            native_cfg.get("max_validation_steps") or mapping(self.config.get("training")).get("max_validation_steps") or train_steps
        )
        self.training_history = pd.DataFrame(
            [
                {
                    "epoch": 0,
                    "step": 1,
                    "env_steps": train_steps,
                    "gradient_updates": 0,
                    "max_train_steps": train_steps,
                    "max_validation_steps": validation_steps,
                    "train_reward": 0.0,
                    "validation_metric": np.nan,
                    "loss": 0.0,
                    "training_algorithm": GT_RCPO_LITE_ALGORITHM,
                    "model_extension_id": MODEL_EXTENSION_ID,
                    "status": "completed",
                }
            ]
        )
        self.training_result = new_model_training_result(
            model_name=self.strategy_name,
            status="completed",
            algorithm=GT_RCPO_LITE_ALGORITHM,
            training_history=self.training_history,
            config=self.config,
            env_steps=train_steps,
            gradient_updates=0,
            best_validation_metric=None,
        )
        self.is_fitted = True
        return self

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        portfolio = self.validate_portfolio_state(portfolio_state)
        candidate = self._candidate_weights(state)
        current = np.asarray(portfolio.current_weights, dtype=float)
        turnover = estimate_turnover(candidate, current)
        cost = estimate_cost(self.config, turnover)
        risk = decision_return_features(state.log_return_window, portfolio.rolling_returns)
        context = mapping(getattr(self, "decision_context", {}))
        scheduler_allowed = bool(context.get("scheduler_allowed_rebalance", True))
        first_trade = bool(context.get("first_trade", False))
        rho, scores, forced_hold_reason = self._rho_action(
            scheduler_allowed=scheduler_allowed,
            first_trade=first_trade,
            expected_return=_candidate_expected_return(candidate, state.log_return_window),
            estimated_turnover=turnover,
            estimated_cost=cost,
            cvar_loss_5=risk["cvar_loss_5"],
            drawdown=float(portfolio.current_drawdown_abs),
        )
        action_index = gate_action_index(self.rho_values, rho)
        action_info = {
            "strategy": self.strategy_name,
            "paper_model_id": self.strategy_name,
            "child_model_name": self.strategy_name,
            "baseline_family": "new_model_extension",
            "training_algorithm": GT_RCPO_LITE_ALGORITHM,
            "model_extension_id": MODEL_EXTENSION_ID,
            "post_hoc_development_disclosure": True,
            "test_used_for_model_selection": False,
            "gate_action": int(rho > 0.0),
            "gate_action_index": int(action_index),
            "rho": float(rho),
            "rebalance_intensity": float(rho),
            "rebalance_values": json.dumps(scores, sort_keys=True, separators=(",", ":")),
            "q_hold": float(scores.get("0", scores.get("0.0", 0.0))),
            "q_rebalance": float(max((value for key, value in scores.items() if float(key) > 0.0), default=0.0)),
            "q_gap": float(
                max((value for key, value in scores.items() if float(key) > 0.0), default=0.0)
                - float(scores.get("0", scores.get("0.0", 0.0)))
            ),
            "estimated_turnover": float(turnover),
            "candidate_turnover": float(turnover),
            "candidate_turnover_estimate": float(turnover),
            "estimated_cost": float(cost),
            "candidate_cost_estimate": float(cost),
            "scheduler_allowed_rebalance": bool(scheduler_allowed),
            "scheduler_pre_allowed": bool(context.get("scheduler_pre_allowed", scheduler_allowed)),
            "first_trade": bool(first_trade),
            "forced_hold_reason": forced_hold_reason,
            "execution_weight_mode": "candidate_plus_rho_execution_core",
            "candidate_weights_json": weights_json(candidate),
            "decision_time_current_weights_json": weights_json(current),
            "CVaR_loss_5": float(risk["cvar_loss_5"]),
            "drawdown": float(portfolio.current_drawdown_abs),
            "rolling_return_mean": float(risk["mean"]),
            "rolling_return_volatility": float(risk["volatility"]),
            "graph_feature_mode": "decision_visible_rolling_correlation",
        }
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=candidate,
                rebalance_action=1 if scheduler_allowed and rho > 0.0 else 0,
                rebalance_intensity=float(rho),
                action_info=action_info,
            )
        )

    def _candidate_weights(self, state: DecisionMarketState) -> np.ndarray:
        window = np.asarray(state.log_return_window, dtype=float)
        if window.ndim == 3:
            window = window.reshape(window.shape[-2], window.shape[-1])
        if window.ndim != 2 or window.shape[-1] != np.asarray(state.available_mask_at_decision).size:
            return normalize_candidate(np.ones_like(state.available_mask_at_decision, dtype=float), state.available_mask_at_decision)
        recent = np.nan_to_num(window[-min(60, window.shape[0]) :], nan=0.0, posinf=0.0, neginf=0.0)
        momentum = recent[-min(20, recent.shape[0]) :].mean(axis=0)
        volatility = recent.std(axis=0, ddof=0) + 1.0e-6
        correlation_penalty = _rolling_correlation_penalty(recent)
        score = momentum / volatility - 0.10 * correlation_penalty
        score = score - float(np.nanmax(score))
        raw = np.exp(np.clip(score, -20.0, 20.0))
        return normalize_candidate(raw, state.available_mask_at_decision)

    def _rho_action(
        self,
        *,
        scheduler_allowed: bool,
        first_trade: bool,
        expected_return: float,
        estimated_turnover: float,
        estimated_cost: float,
        cvar_loss_5: float,
        drawdown: float,
    ) -> tuple[float, dict[str, float], str | None]:
        if not scheduler_allowed:
            return 0.0, {"0": 0.0}, "scheduler_blocked"
        if first_trade and bool(mapping(self.config.get("gt_rcpo_lite")).get("initial_build_full_rho", True)):
            return 1.0, {str(value).rstrip("0").rstrip("."): float(value) for value in self.rho_values}, None
        section = mapping(self.config.get("gt_rcpo_lite"))
        rho, scores = choose_rho(
            rho_values=self.rho_values,
            expected_return=float(expected_return),
            estimated_turnover=float(estimated_turnover),
            estimated_cost=float(estimated_cost),
            cvar_loss_5=float(cvar_loss_5),
            drawdown=float(drawdown),
            lambda_turnover=float(section.get("lambda_turnover", 2.0)),
            lambda_cost=float(section.get("lambda_cost", 10.0)),
            lambda_cvar=float(section.get("lambda_cvar", 0.35)),
            lambda_drawdown=float(section.get("lambda_dd", 0.25)),
            cvar_loss_budget=float(section.get("cvar_loss_budget", 0.02)),
            drawdown_budget=float(section.get("drawdown_budget", 0.10)),
        )
        return rho, scores, "model_chosen_hold" if rho == 0.0 else None


def _rolling_correlation_penalty(window: np.ndarray) -> np.ndarray:
    if window.shape[0] < 2:
        return np.zeros(window.shape[1], dtype=float)
    corr = np.corrcoef(window, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return np.mean(np.abs(corr), axis=1)


def _candidate_expected_return(candidate_weights: np.ndarray, log_return_window: Any) -> float:
    window = np.asarray(log_return_window, dtype=float)
    if window.ndim == 3:
        window = window.reshape(window.shape[-2], window.shape[-1])
    if window.ndim != 2 or window.shape[-1] != candidate_weights.shape[0] or window.size == 0:
        return 0.0
    recent = np.nan_to_num(window[-min(20, window.shape[0]) :], nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.dot(candidate_weights, np.mean(recent, axis=0)))


__all__ = ["GTRCPOLiteStrategy"]
