from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.baselines.base_strategy import BaseStrategy
from src.baselines.cage_common import (
    DEFAULT_CAGE_RHOS,
    MODEL_EXTENSION_ID,
    checkpoint_string,
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
from src.baselines.native_eiie import NativeEIIEStrategy
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState


CAGE_EIIE_ALGORITHM_PREFIX = "cage_eiie"


class CageEIIEStrategy(BaseStrategy):
    strategy_name = "cage_eiie_distributional"
    fit_required = True
    requires_daily_diagnostics = True
    cage_variant = "distributional"
    fixed_rho: float | None = None

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.rho_values = rho_grid(self.config, "cage_eiie", DEFAULT_CAGE_RHOS)
        self.expert = NativeEIIEStrategy(config)
        self.expert.strategy_name = f"{self.strategy_name}_eiie_expert"
        self.training_result: dict[str, Any] | None = None
        self.training_history: pd.DataFrame = pd.DataFrame()

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> CageEIIEStrategy:
        self.expert.strategy_name = f"{self.strategy_name}_eiie_expert"
        self.expert.fit(train_data, validation_data)
        expert_result = self.expert.training_result if isinstance(self.expert.training_result, Mapping) else {}
        history = _history_frame(self.expert.training_history, self.strategy_name, self.cage_variant)
        if expert_result.get("status") != "completed" or self.expert.is_fitted is not True:
            self.training_history = history
            self.training_result = new_model_training_result(
                model_name=self.strategy_name,
                status=str(expert_result.get("status") or "failed_eiie_expert"),
                algorithm=_algorithm_name(self.cage_variant),
                training_history=history,
                config=self.config,
                env_steps=int(expert_result.get("env_steps") or 0),
                gradient_updates=int(expert_result.get("gradient_updates") or 0),
                checkpoint_best_path=checkpoint_string(expert_result.get("checkpoint_best_path")),
                checkpoint_last_path=checkpoint_string(expert_result.get("checkpoint_last_path")),
            )
            self.is_fitted = False
            return self

        self.training_history = history
        self.training_result = new_model_training_result(
            model_name=self.strategy_name,
            status="completed",
            algorithm=_algorithm_name(self.cage_variant),
            training_history=history,
            config=self.config,
            env_steps=int(expert_result.get("env_steps") or 0),
            gradient_updates=int(expert_result.get("gradient_updates") or 0),
            checkpoint_best_path=checkpoint_string(expert_result.get("checkpoint_best_path")),
            checkpoint_last_path=checkpoint_string(expert_result.get("checkpoint_last_path")),
            evaluated_checkpoint_path=checkpoint_string(expert_result.get("evaluated_checkpoint_path")),
            best_validation_metric=_optional_float(expert_result.get("best_validation_metric")),
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
        candidate = self._candidate_weights(state, portfolio)
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
            estimated_turnover=turnover,
            estimated_cost=cost,
            expected_return=_candidate_expected_return(candidate, state.log_return_window),
            cvar_loss_5=risk["cvar_loss_5"],
            drawdown=float(portfolio.current_drawdown_abs),
        )
        action_index = gate_action_index(self.rho_values, rho)
        rebalance_action = 1 if scheduler_allowed and rho > 0.0 else 0
        action_info = {
            "strategy": self.strategy_name,
            "paper_model_id": self.strategy_name,
            "child_model_name": self.strategy_name,
            "baseline_family": "new_model_extension",
            "training_algorithm": _algorithm_name(self.cage_variant),
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
        }
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=candidate,
                rebalance_action=rebalance_action,
                rebalance_intensity=float(rho),
                action_info=action_info,
            )
        )

    def _candidate_weights(self, state: DecisionMarketState, portfolio: PortfolioState) -> np.ndarray:
        action = self.expert.compute_target_weights(state, portfolio)
        weights = action.target_weights
        return normalize_candidate(weights, state.available_mask_at_decision)

    def _rho_action(
        self,
        *,
        scheduler_allowed: bool,
        first_trade: bool,
        estimated_turnover: float,
        estimated_cost: float,
        expected_return: float,
        cvar_loss_5: float,
        drawdown: float,
    ) -> tuple[float, dict[str, float], str | None]:
        if not scheduler_allowed:
            return 0.0, {"0": 0.0}, "scheduler_blocked"
        if first_trade and bool(mapping(self.config.get("cage_eiie")).get("initial_build_full_rho", True)):
            return 1.0, {str(value).rstrip("0").rstrip("."): float(value) for value in self.rho_values}, None
        section = mapping(self.config.get("cage_eiie"))
        fixed = self.fixed_rho
        if fixed is None and section.get("variant") == "fixed_rho":
            fixed = float(section.get("fixed_rho", 0.5))
        if fixed is not None:
            rho = max(0.0, min(1.0, float(fixed)))
            return rho, {"0": 0.0, str(rho).rstrip("0").rstrip("."): float(expected_return - estimated_cost)}, None
        rho, scores = choose_rho(
            rho_values=self.rho_values,
            expected_return=float(expected_return),
            estimated_turnover=float(estimated_turnover),
            estimated_cost=float(estimated_cost),
            cvar_loss_5=float(cvar_loss_5),
            drawdown=float(drawdown),
            lambda_turnover=float(section.get("lambda_turnover", 2.0)),
            lambda_cost=float(section.get("lambda_cost", 10.0)),
                lambda_cvar=0.0
                if self.cage_variant == "no_cvar"
                else float(section.get("lambda_cvar", 0.25 if self.cage_variant == "distributional" else 0.0)),
            lambda_drawdown=float(section.get("lambda_dd", 0.25)),
            cvar_loss_budget=float(section.get("cvar_loss_budget", 0.02)),
            drawdown_budget=float(section.get("drawdown_budget", 0.10)),
        )
        return rho, scores, "model_chosen_hold" if rho == 0.0 else None


class CageEIIEFrozenGateStrategy(CageEIIEStrategy):
    strategy_name = "cage_eiie_frozen_gate"
    cage_variant = "frozen_gate"


class CageEIIEMultilevelGateStrategy(CageEIIEStrategy):
    strategy_name = "cage_eiie_multilevel_gate"
    cage_variant = "multilevel_gate"


class CageEIIEDistributionalStrategy(CageEIIEStrategy):
    strategy_name = "cage_eiie_distributional"
    cage_variant = "distributional"


class CageEIIENoCvarStrategy(CageEIIEStrategy):
    strategy_name = "cage_eiie_no_cvar"
    cage_variant = "no_cvar"


class CageEIIEDistributionalNoCvarStrategy(CageEIIEStrategy):
    strategy_name = "cage_eiie_distributional_no_cvar"
    cage_variant = "no_cvar"


class CageEIIEJointLightStrategy(CageEIIEStrategy):
    strategy_name = "cage_eiie_joint_light"
    cage_variant = "joint_light"


class CageEIIEFixedRho25Strategy(CageEIIEStrategy):
    strategy_name = "cage_eiie_fixed_rho_25"
    cage_variant = "fixed_rho_25"
    fixed_rho = 0.25


class CageEIIEFixedRho50Strategy(CageEIIEStrategy):
    strategy_name = "cage_eiie_fixed_rho_50"
    cage_variant = "fixed_rho_50"
    fixed_rho = 0.50


class CageEIIEFixedRho75Strategy(CageEIIEStrategy):
    strategy_name = "cage_eiie_fixed_rho_75"
    cage_variant = "fixed_rho_75"
    fixed_rho = 0.75


def _history_frame(history: Any, model_name: str, variant: str) -> pd.DataFrame:
    frame = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame()
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame([{"epoch": 0, "step": 0, "status": "completed"}])
    frame["training_algorithm"] = _algorithm_name(variant)
    frame["cage_variant"] = variant
    frame["model_extension_id"] = MODEL_EXTENSION_ID
    return frame


def _algorithm_name(variant: str) -> str:
    return f"{CAGE_EIIE_ALGORITHM_PREFIX}_{variant}"


def _candidate_expected_return(candidate_weights: np.ndarray, log_return_window: Any) -> float:
    window = np.asarray(log_return_window, dtype=float)
    if window.ndim == 3:
        window = window.reshape(window.shape[-2], window.shape[-1])
    if window.ndim != 2 or window.shape[-1] != candidate_weights.shape[0] or window.size == 0:
        return 0.0
    recent = np.nan_to_num(window[-min(20, window.shape[0]) :], nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.dot(candidate_weights, np.mean(recent, axis=0)))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


__all__ = [
    "CageEIIEStrategy",
    "CageEIIEFrozenGateStrategy",
    "CageEIIEMultilevelGateStrategy",
    "CageEIIEDistributionalStrategy",
    "CageEIIENoCvarStrategy",
    "CageEIIEDistributionalNoCvarStrategy",
    "CageEIIEJointLightStrategy",
    "CageEIIEFixedRho25Strategy",
    "CageEIIEFixedRho50Strategy",
    "CageEIIEFixedRho75Strategy",
]
