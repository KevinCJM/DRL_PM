from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.agents.constrained_actor_critic_agent import ConstrainedActorCriticAgent, agent_config_from_mapping
from src.baselines.base_strategy import BaseStrategy
from src.baselines.cage_common import (
    choose_rho,
    decision_return_features,
    estimate_cost,
    estimate_turnover,
    gate_action_index,
    mapping,
    normalize_candidate,
    weights_json,
)
from src.baselines.deep_training import collect_training_batch
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.models.risk_aware_graph_transformer import (
    RA_GT_RCPO_ALGORITHM,
    RA_GT_RCPO_MODEL_EXTENSION_ID,
    RA_GT_RCPO_MODEL_NAME,
    build_risk_aware_graph_transformer,
    config_for_model_name,
)


DEFAULT_RA_GT_RCPO_RHOS = (0.0, 0.25, 0.5, 1.0)


class RiskAwareGTRCPOStrategy(BaseStrategy):
    strategy_name = RA_GT_RCPO_MODEL_NAME
    fit_required = True
    requires_daily_diagnostics = True

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.device = _device(self.config)
        self.training_result: dict[str, Any] | None = None
        self.training_history: pd.DataFrame = pd.DataFrame()
        self._built_for_model_name: str | None = None
        self._agent: ConstrainedActorCriticAgent | None = None
        self._build_agent_for_current_name()

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> RiskAwareGTRCPOStrategy:
        self._build_agent_for_current_name()
        native_cfg = mapping(mapping(self.config.get("baselines")).get("native_rl"))
        train_steps = int(native_cfg.get("max_train_steps") or mapping(self.config.get("training")).get("max_train_steps") or 128)
        validation_steps = int(
            native_cfg.get("max_validation_steps") or mapping(self.config.get("training")).get("max_validation_steps") or train_steps
        )
        train_batch = collect_training_batch(
            train_data,
            n_features=int(self.config["n_features"]),
            window_size=int(self.config["window_size"]),
            n_assets=int(self.config["n_assets"]),
            device=self.device,
            max_samples=train_steps,
        )
        validation_batch = collect_training_batch(
            validation_data,
            n_features=int(self.config["n_features"]),
            window_size=int(self.config["window_size"]),
            n_assets=int(self.config["n_assets"]),
            device=self.device,
            max_samples=validation_steps,
        )
        if train_batch is None:
            self.training_history = pd.DataFrame()
            self.training_result = self._training_result(
                status="failed_missing_train_data",
                env_steps=0,
                gradient_updates=0,
                best_validation_metric=None,
            )
            self.is_fitted = False
            return self

        history, stats = self._agent_or_raise().train_offline(train_batch, validation_batch)
        history = history.copy()
        history["max_train_steps"] = train_steps
        history["max_validation_steps"] = validation_steps
        history["max_gradient_updates_per_epoch"] = mapping(mapping(self.config.get("baselines")).get("native_rl")).get(
            "max_gradient_updates_per_epoch"
        )
        history["training_algorithm"] = RA_GT_RCPO_ALGORITHM
        history["model_extension_id"] = RA_GT_RCPO_MODEL_EXTENSION_ID
        history["clean_room_reimplementation"] = True
        self.training_history = history
        gradient_updates = int(stats.get("gradient_updates", 0))
        status = "completed" if gradient_updates > 0 else "failed_no_gradient_updates"
        self.training_result = self._training_result(
            status=status,
            env_steps=int(stats.get("env_steps", 0)),
            gradient_updates=gradient_updates,
            best_validation_metric=stats.get("best_validation_metric"),
        )
        self.is_fitted = status == "completed"
        return self

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        portfolio = self.validate_portfolio_state(portfolio_state)
        self._build_agent_for_current_name()
        current = np.asarray(portfolio.current_weights, dtype=float)
        action = self._agent_or_raise().select_action(
            np.asarray(state.market_image, dtype=np.float32),
            current,
            np.asarray(state.available_mask_at_decision, dtype=bool),
        )
        candidate = normalize_candidate(action["candidate_weights"], state.available_mask_at_decision)
        turnover = estimate_turnover(candidate, current)
        cost = estimate_cost(self.config, turnover)
        risk = decision_return_features(state.log_return_window, portfolio.rolling_returns)
        expected_return = _candidate_expected_return(candidate, state.log_return_window)
        context = mapping(getattr(self, "decision_context", {}))
        scheduler_allowed = bool(context.get("scheduler_allowed_rebalance", True))
        first_trade = bool(context.get("first_trade", False))
        rho, scores, forced_hold_reason = self._rho_action(
            scheduler_allowed=scheduler_allowed,
            first_trade=first_trade,
            expected_return=expected_return,
            estimated_turnover=turnover,
            estimated_cost=cost,
            cvar_loss_5=risk["cvar_loss_5"],
            drawdown=float(portfolio.current_drawdown_abs),
        )
        rho_values = _rho_values(self.config)
        section = self._section()
        action_index = gate_action_index(rho_values, rho)
        violations = _constraint_violation_count(
            turnover=turnover,
            cost=cost,
            cvar_loss_5=risk["cvar_loss_5"],
            drawdown=float(portfolio.current_drawdown_abs),
            section=section,
        )
        action_info = {
            "strategy": self.strategy_name,
            "paper_model_id": self.strategy_name,
            "child_model_name": self.strategy_name,
            "baseline_family": "new_model_extension",
            "training_algorithm": RA_GT_RCPO_ALGORITHM,
            "model_extension_id": RA_GT_RCPO_MODEL_EXTENSION_ID,
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
            "max_drawdown_loss": float(portfolio.current_drawdown_abs),
            "drawdown": float(portfolio.current_drawdown_abs),
            "rolling_return_mean": float(risk["mean"]),
            "rolling_return_volatility": float(risk["volatility"]),
            "graph_feature_mode": "decision_visible_rolling_correlation",
            "graph_edge_threshold": float(section.get("graph_edge_threshold", 0.10)),
            "graph_density": float(action["graph_density"]),
            "mean_abs_correlation": float(action["mean_abs_correlation"]),
            "value_return": float(action["value_return"]),
            "value_cost": float(action["value_cost"]),
            "value_drawdown": float(action["value_drawdown"]),
            "value_cvar_loss": float(action["value_cvar_loss"]),
            "lambda_turnover": float(section.get("lambda_turnover", 2.0)),
            "lambda_cost": float(section.get("lambda_cost", 10.0)),
            "lambda_cvar": float(section.get("lambda_cvar", 0.35)),
            "lambda_drawdown": float(section.get("lambda_drawdown", section.get("lambda_dd", 0.25))),
            "average_turnover_per_step_budget": float(section.get("average_turnover_per_step_budget", 0.20)),
            "average_cost_per_step_budget": float(section.get("average_cost_per_step_budget", 0.001)),
            "cvar_loss_budget": float(section.get("cvar_loss_budget", 0.02)),
            "drawdown_budget": float(section.get("drawdown_budget", 0.10)),
            "constraint_violation_count": int(violations),
        }
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=candidate,
                rebalance_action=1 if scheduler_allowed and rho > 0.0 else 0,
                rebalance_intensity=float(rho),
                action_info=action_info,
            )
        )

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
        rho_values = _rho_values(self.config)
        if not scheduler_allowed:
            return 0.0, {"0": 0.0}, "scheduler_blocked"
        if first_trade and bool(self._section().get("initial_build_full_rho", True)):
            return 1.0, {str(value).rstrip("0").rstrip("."): float(value) for value in rho_values}, None
        section = self._section()
        rho, scores = choose_rho(
            rho_values=rho_values,
            expected_return=float(expected_return),
            estimated_turnover=float(estimated_turnover),
            estimated_cost=float(estimated_cost),
            cvar_loss_5=float(cvar_loss_5),
            drawdown=float(drawdown),
            lambda_turnover=float(section.get("lambda_turnover", 2.0)),
            lambda_cost=float(section.get("lambda_cost", 10.0)),
            lambda_cvar=float(section.get("lambda_cvar", 0.35)),
            lambda_drawdown=float(section.get("lambda_drawdown", section.get("lambda_dd", 0.25))),
            cvar_loss_budget=float(section.get("cvar_loss_budget", 0.02)),
            drawdown_budget=float(section.get("drawdown_budget", 0.10)),
        )
        return rho, scores, "model_chosen_hold" if rho == 0.0 else None

    def _build_agent_for_current_name(self) -> None:
        model_name = str(self.strategy_name)
        if self._built_for_model_name == model_name and self._agent is not None:
            return
        model = build_risk_aware_graph_transformer(self.config, model_name=model_name)
        agent_config = agent_config_from_mapping(self.config, section=self._section_for(model_name))
        self._agent = ConstrainedActorCriticAgent(model, config=agent_config, device=self.device)
        self._built_for_model_name = model_name

    def _section(self) -> dict[str, Any]:
        return self._section_for(str(self.strategy_name))

    def _section_for(self, model_name: str) -> dict[str, Any]:
        return config_for_model_name(model_name, self.config)

    def _agent_or_raise(self) -> ConstrainedActorCriticAgent:
        if self._agent is None:
            raise RuntimeError("ERR_RA_GT_RCPO_AGENT_NOT_BUILT")
        return self._agent

    def _training_result(
        self,
        *,
        status: str,
        env_steps: int,
        gradient_updates: int,
        best_validation_metric: float | None,
    ) -> dict[str, Any]:
        rankability = mapping(self.config.get("rankability"))
        native = mapping(mapping(self.config.get("baselines")).get("native_rl"))
        return {
            "model_name": self.strategy_name,
            "paper_model_id": self.strategy_name,
            "child_model_name": self.strategy_name,
            "baseline_family": "new_model_extension",
            "status": status,
            "training_algorithm": RA_GT_RCPO_ALGORITHM,
            "rl_training": True,
            "platform_native_rl_training": True,
            "proxy_training": False,
            "external_original_implementation": False,
            "clean_room_reimplementation": True,
            "algorithm_fidelity": "platform_native",
            "rankable_in_unified_table": bool(rankability.get("rankable_in_unified_table", False)),
            "model_extension_id": RA_GT_RCPO_MODEL_EXTENSION_ID,
            "post_hoc_development_disclosure": True,
            "test_used_for_model_selection": False,
            "data_protocol": "platform",
            "execution_protocol": "platform_backtest_engine",
            "evaluation_protocol": "unified_platform",
            "cost_model_shared": True,
            "cost_availability": "available",
            "constraint_protocol_shared": True,
            "training_history": self.training_history,
            "best_validation_metric": best_validation_metric,
            "env_steps": int(env_steps),
            "gradient_updates": int(gradient_updates),
            "max_train_steps": native.get("max_train_steps", mapping(self.config.get("training")).get("max_train_steps")),
            "max_validation_steps": native.get("max_validation_steps", mapping(self.config.get("training")).get("max_validation_steps")),
            "max_gradient_updates_per_epoch": native.get(
                "max_gradient_updates_per_epoch",
                mapping(self.config.get("training")).get("max_gradient_updates_per_epoch"),
            ),
        }


def _rho_values(config: Mapping[str, Any]) -> tuple[float, ...]:
    raw = mapping(config.get("ra_gt_rcpo")).get("rho_actions") or DEFAULT_RA_GT_RCPO_RHOS
    values = sorted({max(0.0, min(1.0, float(value))) for value in raw})
    if 0.0 not in values:
        values.insert(0, 0.0)
    return tuple(values)


def _candidate_expected_return(candidate_weights: np.ndarray, log_return_window: Any) -> float:
    window = np.asarray(log_return_window, dtype=float)
    if window.ndim == 3:
        window = window.reshape(window.shape[-2], window.shape[-1])
    if window.ndim != 2 or window.shape[-1] != candidate_weights.shape[0] or window.size == 0:
        return 0.0
    recent = np.nan_to_num(window[-min(20, window.shape[0]) :], nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.dot(candidate_weights, np.mean(recent, axis=0)))


def _constraint_violation_count(
    *,
    turnover: float,
    cost: float,
    cvar_loss_5: float,
    drawdown: float,
    section: Mapping[str, Any],
) -> int:
    checks = (
        float(turnover) > float(section.get("average_turnover_per_step_budget", 0.20)),
        float(cost) > float(section.get("average_cost_per_step_budget", 0.001)),
        float(cvar_loss_5) > float(section.get("cvar_loss_budget", 0.02)),
        float(drawdown) > float(section.get("drawdown_budget", 0.10)),
    )
    return int(sum(bool(value) for value in checks))


def _device(config: Mapping[str, Any]) -> torch.device:
    value = config.get("device")
    if isinstance(value, torch.device):
        return value
    if isinstance(value, str):
        return torch.device(value)
    if isinstance(value, Mapping):
        mode = str(value.get("mode", "cpu")).lower()
        if mode in {"cuda", "auto"} and torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


__all__ = ["RiskAwareGTRCPOStrategy"]
