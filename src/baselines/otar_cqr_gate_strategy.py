from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.baselines.base_strategy import BaseStrategy
from src.buffers.replay_buffer import ReplayBuffer, ReplayItem
from src.data.loader import DataContractError
from src.envs.constraint_manager import ConstraintManager
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.models.cost_estimator import CostEstimator
from src.models.otar_cqr_gate import OTarCQRGate


def estimate_candidate_cost(
    observation: Mapping[str, Any],
    pre_trade_drifted_weights: torch.Tensor,
    candidate_weights: torch.Tensor,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if _has_cost_observation(observation):
        turnover, cost = CostEstimator.estimate(
            candidate_weights,
            pre_trade_drifted_weights,
            _observation_matrix(observation, "adv20_at_decision", candidate_weights),
            _observation_matrix(observation, "volatility_20d_at_decision", candidate_weights),
            float(np.asarray(observation["portfolio_value"], dtype=float)),
            config,
            amount=_observation_matrix(observation, "amount_at_decision", candidate_weights),
            turnover_rate=_observation_matrix(observation, "turnover_rate_at_decision", candidate_weights),
        )
        return turnover, cost
    turnover = 0.5 * torch.sum(torch.abs(candidate_weights - pre_trade_drifted_weights), dim=1, keepdim=True)
    cost = torch.zeros_like(turnover)
    return turnover, cost


def stack_observation_batch(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not states:
        raise ValueError("ERR_STACK_OBSERVATION_EMPTY: states is empty")
    keys = states[0].keys()
    batched: dict[str, Any] = {}
    for key in keys:
        values = [s[key] for s in states]
        first = values[0]
        if isinstance(first, np.ndarray):
            batched[key] = torch.as_tensor(np.stack(values), dtype=torch.float32)
        elif isinstance(first, torch.Tensor):
            batched[key] = torch.stack(values)
        elif isinstance(first, (int, float, bool, np.integer, np.floating, np.bool_)):
            batched[key] = torch.as_tensor(values)
        else:
            batched[key] = list(values)
    return batched


def _has_cost_observation(observation: Mapping[str, Any]) -> bool:
    return all(
        key in observation
        for key in (
            "adv20_at_decision",
            "volatility_20d_at_decision",
            "amount_at_decision",
            "turnover_rate_at_decision",
            "portfolio_value",
        )
    )


def _observation_matrix(observation: Mapping[str, Any], key: str, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(observation[key], dtype=np.float32),
        dtype=like.dtype,
        device=like.device,
    ).unsqueeze(0)


def _weights_json(weights: np.ndarray) -> str:
    return json.dumps([round(float(w), 10) for w in weights], ensure_ascii=False)


class OTarCQRGateStrategy(BaseStrategy):
    """CQR Gate strategy — training loop for OTarCQRGate model."""

    def __init__(
        self,
        config: Mapping[str, Any],
        model: OTarCQRGate | None = None,
    ) -> None:
        super().__init__(config)
        self.model: OTarCQRGate | None = model
        self.device = model._device if model is not None else torch.device("cpu")

        self.cqr_config = dict(_mapping(config.get("cqr")))
        self.training_config = dict(_mapping(config.get("training")))
        self.optimizer_config = dict(_mapping(config.get("optimizer")))

        self.replay_buffer = ReplayBuffer(
            capacity=int(self.cqr_config.get("replay_capacity", self.training_config.get("replay_capacity", 100000))),
            gamma=float(self.cqr_config.get("gate_gamma", 0.99)),
            n_steps=1,
        )

        self.optimizer: torch.optim.Optimizer | None = None
        if self.model is not None:
            self._ensure_optimizer()

        self.epsilon_start = float(self.cqr_config.get("epsilon_start", 1.0))
        self.epsilon_end = float(self.cqr_config.get("epsilon_end", 0.05))
        self.epsilon_decay_steps = int(self.cqr_config.get("epsilon_decay_steps", 10000))

        self.replay_min_size = int(self.cqr_config.get("replay_min_size", 1000))
        self.gate_batch_size = int(self.cqr_config.get("gate_batch_size", 64))
        self.target_update_interval = int(self.cqr_config.get("target_update_interval", 100))

        self.min_rebalance_ratio = float(self.cqr_config.get("min_rebalance_ratio_in_buffer", 0.10))
        self.min_hold_ratio = float(self.cqr_config.get("min_hold_ratio_in_buffer", 0.10))

        self._global_step = 0
        self._actor_update_count = 0
        self._actor_skipped_by_gate_count = 0
        self.constraint_manager = ConstraintManager(self.config)
        self.training_result: dict[str, Any] | None = None
        self.requires_daily_diagnostics = True
        self.strategy_name = "otar_cqr_gate"
        self.fit_required = bool(_mapping(config.get("cqr")).get("fit_required", False))

    def _ensure_model(
        self,
        *,
        n_assets: int | None = None,
        market_image: Any | None = None,
    ) -> OTarCQRGate:
        if self.model is not None:
            return self.model
        resolved = dict(self.config)
        if n_assets is not None:
            resolved["n_assets"] = int(n_assets)
        if market_image is not None:
            image = np.asarray(market_image)
            if image.ndim == 3:
                resolved["n_features"] = int(image.shape[0])
                resolved["window_size"] = int(image.shape[1])
        if "n_assets" not in resolved or "n_features" not in resolved or "window_size" not in resolved:
            raise DataContractError(
                "ERR_STRATEGY_STATE_CONTRACT",
                "ERR_STRATEGY_STATE_CONTRACT: otar_cqr_gate model dimensions unavailable",
            )
        self.model = OTarCQRGate(resolved)
        self.config.update(resolved)
        self.device = self.model._device
        self._ensure_optimizer()
        return self.model

    def _ensure_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is not None:
            return self.optimizer
        if self.model is None:
            raise DataContractError(
                "ERR_STRATEGY_STATE_CONTRACT",
                "ERR_STRATEGY_STATE_CONTRACT: otar_cqr_gate optimizer before model",
            )
        gate_lr = float(
            self.optimizer_config.get(
                "gate_lr",
                self.cqr_config.get("gate_lr", self.training_config.get("lr", 3e-4)),
            )
        )
        self.optimizer = torch.optim.Adam(list(self.model.cqr_critic.parameters()), lr=gate_lr)
        return self.optimizer

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> "OTarCQRGateStrategy":
        if isinstance(train_data, PortfolioRebalanceEnv):
            env = train_data
            self._ensure_model(n_assets=len(env.asset_ids), market_image=env.reset()[0]["market_image"])
        elif isinstance(train_data, Mapping) and train_data.get("dataset") is not None:
            from types import SimpleNamespace

            split = train_data.get("split")
            if split is None:
                split = SimpleNamespace(
                    train_dates=train_data.get("dates"),
                    train_last_decision_date=None,
                )
            env = PortfolioRebalanceEnv(
                train_data["dataset"],
                split,
                config=train_data.get("config", self.config),
                segment=str(train_data.get("segment", "train")),
                market_image_dataset=train_data.get("market_image_dataset"),
            )
            self._ensure_model(n_assets=len(env.asset_ids), market_image=env.reset()[0]["market_image"])
        else:
            self.is_fitted = True
            self.training_result = {"status": "completed", "env_steps": 0, "gradient_updates": 0}
            return self

        training_config = _mapping(self.config.get("training"))
        epochs = int(training_config.get("epochs", 1))
        max_steps = training_config.get("max_train_steps")
        epoch_results: list[dict[str, Any]] = []
        for _ in range(max(1, epochs)):
            epoch_results.append(self._train_epoch(env, max_steps=None if max_steps is None else int(max_steps)))
        last = epoch_results[-1] if epoch_results else {"status": "completed"}
        self.training_result = {
            **last,
            "status": "completed",
            "epochs": len(epoch_results),
            "history": epoch_results,
        }
        self.is_fitted = True
        return self

    @torch.no_grad()
    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        portfolio = self.validate_portfolio_state(portfolio_state)
        observation = _observation_from_decision_state(state, portfolio)
        model = self._ensure_model(
            n_assets=len(np.asarray(portfolio.current_weights).reshape(-1)),
            market_image=observation["market_image"],
        )
        risk_state = None
        if model.use_risk_state:
            if portfolio.risk_state_vector is None:
                raise DataContractError(
                    "ERR_OBSERVATION_RISK_STATE_MISSING",
                    "ERR_OBSERVATION_RISK_STATE_MISSING: backtest portfolio_state.risk_state_vector",
                )
            observation["risk_state"] = np.asarray(portfolio.risk_state_vector, dtype=np.float32)
            risk_state = torch.as_tensor(observation["risk_state"], dtype=torch.float32, device=self.device)

        mask = torch.as_tensor(
            np.asarray(observation["availability_mask"], dtype=bool)[None, :],
            dtype=torch.bool,
            device=self.device,
        )
        pre_trade = torch.as_tensor(
            np.asarray(observation["current_weights"], dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        latent = model.encode_latent_from_observation(observation, risk_state=risk_state)
        proposal = model.propose_candidate(latent, mask, deterministic=True)
        candidate = proposal["candidate_weights"]

        asset_ids = list(getattr(state, "asset_ids", self.config.get("data", {}).get("asset_ids", [])))
        projected = self.constraint_manager.project(
            candidate.squeeze(0).detach().cpu().numpy(),
            mask.squeeze(0).detach().cpu().numpy(),
            reference_weights=pre_trade.squeeze(0).detach().cpu().numpy(),
            asset_ids=asset_ids if asset_ids else None,
        )
        candidate_exec = torch.as_tensor(
            projected.projected_weights[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        estimated_turnover_candidate, estimated_cost_candidate = estimate_candidate_cost(
            observation,
            pre_trade,
            candidate_exec,
            self.config,
        )
        estimated_cost_hold = torch.zeros_like(estimated_cost_candidate)
        gate_output = model.gate_decision(
            latent,
            mask,
            pre_trade,
            candidate_exec,
            estimated_cost_candidate,
            estimated_cost_hold,
            deterministic=True,
        )
        gate_action = int(gate_output["gate_action"].item())
        target_weights = candidate_exec.squeeze(0).detach().cpu().numpy() if gate_action == 1 else pre_trade.squeeze(0).detach().cpu().numpy()
        estimated_turnover = float(estimated_turnover_candidate.item()) if gate_action == 1 else 0.0
        estimated_cost = float(estimated_cost_candidate.item()) if gate_action == 1 else 0.0
        action_info = {
            "strategy": self.strategy_name,
            "paper_model_id": self.strategy_name,
            "child_model_name": self.strategy_name,
            "baseline_family": "platform_native_rl",
            "gate_action": gate_action,
            "raw_gate_action": gate_action,
            "executed_gate_action": gate_action,
            "raw_model_requested_rebalance": bool(gate_action),
            "raw_action": gate_action,
            "candidate_weights_json": _weights_json(candidate_exec.squeeze(0).detach().cpu().numpy()),
            "pre_trade_drifted_weights": pre_trade.squeeze(0).detach().cpu().numpy(),
            "q_hold": float(gate_output["pred_hold_utility"].item()),
            "q_rebalance": float(gate_output["pred_candidate_utility"].item()),
            "q_gap": float(gate_output["pred_delta_utility"].item()),
            "estimated_turnover": estimated_turnover,
            "estimated_cost": estimated_cost,
            "candidate_estimated_cost": float(estimated_cost_candidate.item()),
            "hold_estimated_cost": float(estimated_cost_hold.item()),
            "pred_delta_utility": float(gate_output["pred_delta_utility"].item()),
            "pred_candidate_mean_return": float(gate_output["pred_candidate_mean_return"].item()),
            "pred_hold_mean_return": float(gate_output["pred_hold_mean_return"].item()),
            "pred_candidate_lower_tail_loss": float(gate_output["pred_candidate_lower_tail_loss"].item()),
            "pred_hold_lower_tail_loss": float(gate_output["pred_hold_lower_tail_loss"].item()),
            "pred_candidate_utility": float(gate_output["pred_candidate_utility"].item()),
            "pred_hold_utility": float(gate_output["pred_hold_utility"].item()),
            "gate_margin": float(model.gate_margin),
            "quantile_spread_candidate": float(gate_output["quantile_spread_candidate"].item()),
            "quantile_spread_hold": float(gate_output["quantile_spread_hold"].item()),
            "predicted_5pct_quantile_candidate": float(gate_output["predicted_5pct_quantile_candidate"].item()),
            "predicted_5pct_quantile_hold": float(gate_output["predicted_5pct_quantile_hold"].item()),
            "predicted_5pct_quantile_executed": float(
                (
                    gate_output["predicted_5pct_quantile_candidate"]
                    if gate_action == 1
                    else gate_output["predicted_5pct_quantile_hold"]
                ).item()
            ),
            "log_prob": float(proposal["log_prob"].item()),
            "entropy": float(proposal["distribution"].entropy().item()),
            "alpha_min": float(proposal["distribution"].alpha.min().item()),
            "alpha_max": float(proposal["distribution"].alpha.max().item()),
            "alpha_mean": float(proposal["distribution"].alpha.mean().item()),
            "projection_distance": float(projected.projection_distance),
            "projection_violation_count": len(projected.constraint_violations),
        }
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=np.asarray(target_weights, dtype=float),
                rebalance_action=gate_action,
                rebalance_intensity=1.0 if gate_action == 1 else 0.0,
                action_info=action_info,
            )
        )

    def _epsilon(self) -> float:
        frac = min(1.0, self._global_step / max(1, self.epsilon_decay_steps))
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1.0 - frac)

    def _train_epoch(
        self,
        env: PortfolioRebalanceEnv,
        *,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        observation, _ = env.reset()
        model = self._ensure_model(n_assets=len(env.asset_ids), market_image=observation["market_image"])
        optimizer = self._ensure_optimizer()
        terminated = False
        truncated = False
        reward_total = 0.0
        env_steps = 0
        qr_losses: list[float] = []

        asset_ids = list(getattr(env, "asset_ids", self.config.get("data", {}).get("asset_ids", [])))
        constraint_mgr = ConstraintManager(self.config)

        while not (terminated or truncated):
            if max_steps is not None and env_steps >= int(max_steps):
                break

            risk_state = None
            if "risk_state" in observation:
                risk_state = torch.as_tensor(
                    np.asarray(observation["risk_state"], dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )

            mask = torch.as_tensor(
                np.asarray(observation["availability_mask"], dtype=bool)[None, :],
                dtype=torch.bool,
                device=self.device,
            )
            pre_trade_drifted_weights = torch.as_tensor(
                np.asarray(observation["current_weights"], dtype=np.float32)[None, :],
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():
                latent = model.encode_latent_from_observation(observation, risk_state=risk_state)
                proposal = model.propose_candidate(latent, mask, deterministic=False)

            candidate_weights = proposal["candidate_weights"]
            projected_list: list[np.ndarray] = []
            projection_results = []
            for i in range(candidate_weights.shape[0]):
                result = constraint_mgr.project(
                    candidate_weights[i].detach().cpu().numpy(),
                    mask[i].detach().cpu().numpy(),
                    reference_weights=pre_trade_drifted_weights[i].detach().cpu().numpy(),
                    asset_ids=asset_ids,
                )
                projected_list.append(result.projected_weights)
                projection_results.append(result)
            candidate_weights_for_execution = torch.tensor(
                np.stack(projected_list), device=self.device, dtype=torch.float32,
            )

            with torch.no_grad():
                estimated_turnover_candidate, estimated_cost_candidate = estimate_candidate_cost(
                    observation, pre_trade_drifted_weights, candidate_weights_for_execution, self.config,
                )
            estimated_cost_hold = torch.zeros_like(estimated_cost_candidate)

            with torch.no_grad():
                gate_output = model.gate_decision(
                    latent, mask, pre_trade_drifted_weights, candidate_weights_for_execution,
                    estimated_cost_candidate, estimated_cost_hold, deterministic=False,
                )

            raw_gate_action = int(gate_output["gate_action"].item())
            epsilon = self._epsilon()
            if random.random() < epsilon:
                executed_gate_action = 1 - raw_gate_action
            else:
                executed_gate_action = raw_gate_action

            ppo_actor_update_mask = 1 if executed_gate_action == 1 else 0
            self._actor_update_count += ppo_actor_update_mask
            self._actor_skipped_by_gate_count += (1 - ppo_actor_update_mask)

            raw_gate_action_scalar = raw_gate_action
            executed_gate_action_scalar = executed_gate_action

            predicted_5pct_quantile_executed = (
                gate_output["predicted_5pct_quantile_candidate"]
                if executed_gate_action_scalar == 1
                else gate_output["predicted_5pct_quantile_hold"]
            )

            action_info: dict[str, Any] = {
                "candidate_weights": candidate_weights_for_execution.squeeze(0).detach().cpu().numpy(),
                "candidate_weights_json": _weights_json(
                    candidate_weights_for_execution.squeeze(0).detach().cpu().numpy()
                ),
                "raw_gate_action": raw_gate_action_scalar,
                "executed_gate_action": executed_gate_action_scalar,
                "gate_action": executed_gate_action_scalar,
                "ppo_actor_update_mask": int(ppo_actor_update_mask),
                "value_update_mask": 1,
                "gate_update_mask": 1,
                "pred_delta_utility": float(gate_output["pred_delta_utility"].item()),
                "pred_candidate_mean_return": float(gate_output["pred_candidate_mean_return"].item()),
                "pred_hold_mean_return": float(gate_output["pred_hold_mean_return"].item()),
                "pred_candidate_lower_tail_loss": float(gate_output["pred_candidate_lower_tail_loss"].item()),
                "pred_hold_lower_tail_loss": float(gate_output["pred_hold_lower_tail_loss"].item()),
                "pred_candidate_utility": float(gate_output["pred_candidate_utility"].item()),
                "pred_hold_utility": float(gate_output["pred_hold_utility"].item()),
                "gate_margin": float(model.gate_margin),
                "quantile_spread_candidate": float(gate_output["quantile_spread_candidate"].item()),
                "quantile_spread_hold": float(gate_output["quantile_spread_hold"].item()),
                "predicted_5pct_quantile_candidate": float(
                    gate_output["predicted_5pct_quantile_candidate"].item()
                ),
                "predicted_5pct_quantile_hold": float(gate_output["predicted_5pct_quantile_hold"].item()),
                "predicted_5pct_quantile_executed": float(predicted_5pct_quantile_executed.item()),
                "log_prob": float(proposal["log_prob"].item()),
                "entropy": float(proposal["distribution"].entropy().item()),
                "alpha_min": float(proposal["distribution"].alpha.min().item()),
                "alpha_max": float(proposal["distribution"].alpha.max().item()),
                "alpha_mean": float(proposal["distribution"].alpha.mean().item()),
                "projection_distance": float(np.mean([r.projection_distance for r in projection_results])),
                "projection_violation_count": sum(len(r.constraint_violations) for r in projection_results),
                "estimated_turnover": float(estimated_turnover_candidate.item()) if executed_gate_action_scalar == 1 else 0.0,
                "estimated_cost": float(estimated_cost_candidate.item()) if executed_gate_action_scalar == 1 else 0.0,
                "candidate_estimated_cost": float(estimated_cost_candidate.item()),
                "hold_estimated_cost": float(estimated_cost_hold.item()),
                "pre_trade_drifted_weights": pre_trade_drifted_weights.squeeze(0).detach().cpu().numpy(),
            }

            if executed_gate_action_scalar == 1:
                env_weights = candidate_weights_for_execution.squeeze(0).detach().cpu().numpy()
                env_rebalance = 1
                env_rebalance_intensity = 1.0
            else:
                env_weights = pre_trade_drifted_weights.squeeze(0).detach().cpu().numpy()
                env_rebalance = 0
                env_rebalance_intensity = 0.0

            env_action: dict[str, Any] = dict(action_info)
            env_action["weights"] = env_weights
            env_action["rebalance"] = env_rebalance
            env_action["rebalance_intensity"] = env_rebalance_intensity

            next_observation, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)

            execution_result = info.get("execution_result")
            realized_gross_simple_return_t = (
                float(execution_result.gross_return) if execution_result is not None else 0.0
            )

            pre_trade_drifted_weights_t_plus_1 = None
            if "current_weights" in next_observation:
                pre_trade_drifted_weights_t_plus_1 = np.asarray(
                    next_observation["current_weights"], dtype=np.float32
                )

            replay_item_kwargs: dict[str, Any] = {
                "candidate_weights_t": candidate_weights_for_execution.squeeze(0).detach().cpu().numpy(),
                "executed_weights_t": env_weights,
                "gate_action_t": executed_gate_action_scalar,
                "rebalance_action_t": env_rebalance,
                "rebalance_intensity_t": env_rebalance_intensity,
                "estimated_turnover_t": float(estimated_turnover_candidate.item()) if executed_gate_action_scalar == 1 else 0.0,
                "realized_turnover_t": float(info.get("turnover", 0.0)),
                "estimated_cost_t": float(estimated_cost_candidate.item()) if executed_gate_action_scalar == 1 else 0.0,
                "realized_cost_t": float(info.get("transaction_cost", 0.0)),
                "reward_t": float(reward),
                "terminated_t": done,
                "truncated_t": bool(truncated),
                "q_hold_t": float(gate_output["pred_hold_utility"].item()),
                "q_rebalance_t": float(gate_output["pred_candidate_utility"].item()),
                "q_gap_t": float(gate_output["pred_delta_utility"].item()),
                "pre_trade_drifted_weights_t": pre_trade_drifted_weights.squeeze(0).detach().cpu().numpy(),
                "pre_trade_drifted_weights_t_plus_1": pre_trade_drifted_weights_t_plus_1,
                "estimated_cost_candidate_t": float(estimated_cost_candidate.item()),
                "estimated_cost_hold_t": 0.0,
                "realized_gross_simple_return_t": realized_gross_simple_return_t,
            }

            decision_date = info.get("decision_date")
            execution_date = info.get("execution_date")
            next_valuation_date = info.get("next_valuation_date")

            if decision_date is not None and execution_date is not None and next_valuation_date is not None:
                replay_item_kwargs["decision_date_t"] = decision_date
                replay_item_kwargs["execution_date_t"] = execution_date
                replay_item_kwargs["next_valuation_date_t"] = next_valuation_date
                if not done:
                    next_decision = next_observation.get("decision_date", decision_date)
                    next_execution = next_observation.get("execution_date", execution_date)
                    next_valuation = next_observation.get("next_valuation_date", next_valuation_date)
                    replay_item_kwargs["decision_date_next"] = next_decision
                    replay_item_kwargs["execution_date_next"] = next_execution
                    replay_item_kwargs["next_valuation_date_next"] = next_valuation
                else:
                    replay_item_kwargs["decision_date_next"] = decision_date
                    replay_item_kwargs["execution_date_next"] = execution_date
                    replay_item_kwargs["next_valuation_date_next"] = next_valuation_date

            replay_item_kwargs["execution_price_t"] = str(info.get("execution_price_type", "close"))
            replay_item_kwargs["delayed_action_execution_t"] = bool(
                info.get("delayed_action_execution", False)
            )

            state_t = dict(observation)
            state_tp1 = dict(next_observation) if not done else dict(observation)
            replay_item_kwargs["state_t"] = state_t
            replay_item_kwargs["state_tp1"] = state_tp1

            replay_item = ReplayItem(**replay_item_kwargs)
            self.replay_buffer.validate_cqr_fields(replay_item)
            self.replay_buffer.add(replay_item)

            if len(self.replay_buffer) >= self.replay_min_size:
                batch_items = self._sample_cqr_batch()
                if batch_items:
                    batch = self.replay_buffer.as_batch(batch_items, device=self.device)
                    qr_loss = self._compute_qr_loss(batch)
                    if qr_loss is not None:
                        optimizer.zero_grad()
                        qr_loss.backward()
                        optimizer.step()
                        qr_losses.append(float(qr_loss.detach().cpu()))

            self._global_step += 1
            if self._global_step % self.target_update_interval == 0:
                model.update_targets()

            observation = next_observation
            reward_total += float(reward)
            env_steps += 1

        effective_ratio = self._effective_actor_update_ratio()
        return {
            "status": "completed",
            "env_steps": env_steps,
            "gradient_updates": len(qr_losses),
            "train_reward": reward_total,
            "loss": float(np.mean(qr_losses)) if qr_losses else np.nan,
            "effective_actor_update_ratio": effective_ratio,
            "actor_update_count": self._actor_update_count,
            "actor_skipped_by_gate_count": self._actor_skipped_by_gate_count,
        }

    def _effective_actor_update_ratio(self) -> float:
        total = self._actor_update_count + self._actor_skipped_by_gate_count
        if total == 0:
            return 0.0
        return self._actor_update_count / total

    def _sample_cqr_batch(self) -> list[ReplayItem] | None:
        items = list(self.replay_buffer.items)
        if not items:
            return None
        rebalance_items = [item for item in items if int(item.gate_action_t) == 1]
        hold_items = [item for item in items if int(item.gate_action_t) == 0]
        total = len(items)
        rebalance_ratio = len(rebalance_items) / total if total > 0 else 0.0
        hold_ratio = len(hold_items) / total if total > 0 else 0.0

        rng = np.random.default_rng()
        batch_size = min(self.gate_batch_size, total)

        if rebalance_ratio < self.min_rebalance_ratio and rebalance_items:
            n_rebalance = max(1, int(batch_size * 0.5))
            n_hold = batch_size - n_rebalance
            sampled_rebalance = list(rng.choice(rebalance_items, size=min(n_rebalance, len(rebalance_items)), replace=True))
            sampled_hold = list(rng.choice(hold_items, size=min(n_hold, len(hold_items)), replace=True)) if hold_items else []
            batch = sampled_rebalance + sampled_hold
        elif hold_ratio < self.min_hold_ratio and hold_items:
            n_hold = max(1, int(batch_size * 0.5))
            n_rebalance = batch_size - n_hold
            sampled_hold = list(rng.choice(hold_items, size=min(n_hold, len(hold_items)), replace=True))
            sampled_rebalance = list(rng.choice(rebalance_items, size=min(n_rebalance, len(rebalance_items)), replace=True)) if rebalance_items else []
            batch = sampled_rebalance + sampled_hold
        else:
            batch = list(rng.choice(items, size=batch_size, replace=False))

        rng.shuffle(batch)
        return batch

    def _compute_qr_loss(self, batch: dict[str, Any]) -> torch.Tensor | None:
        gate_action_t = batch["gate_action_t"]
        terminated_t = batch["terminated_t"]
        batch_size = gate_action_t.shape[0]
        if batch_size == 0:
            return None

        candidate_quantiles_t = self._critic_forward_from_batch(batch, "t")
        gate_action_expanded = gate_action_t.float()
        selected_quantiles_t = candidate_quantiles_t * gate_action_expanded + \
            self._hold_quantiles_from_batch(batch, "t") * (1.0 - gate_action_expanded)

        realized_gross = batch["realized_gross_simple_return_t"]
        if realized_gross is None:
            return None
        realized_gross_t = realized_gross.view(-1, 1)

        discount = batch.get("discount")
        if discount is None:
            discount = torch.ones(batch_size, 1, device=self.device, dtype=torch.float32)
        else:
            discount = discount.view(-1, 1)

        with torch.no_grad():
            state_tp1_list = batch["state_tp1"]
            pre_trade_tp1 = batch["pre_trade_drifted_weights_t_plus_1"]

            target_q = torch.zeros(batch_size, self.model.n_quantiles, device=self.device, dtype=torch.float32)
            env_config = _mapping(self.config.get("env"))
            data_config = _mapping(self.config.get("data"))
            asset_ids = list(env_config.get("asset_ids") or data_config.get("asset_ids") or [])
            constraint_mgr = ConstraintManager(self.config)

            for idx in range(batch_size):
                if bool(terminated_t[idx].item()):
                    continue

                if isinstance(state_tp1_list, list):
                    tp1_obs = state_tp1_list[idx]
                else:
                    tp1_obs = state_tp1_list

                tp1_mask = torch.as_tensor(
                    np.asarray(tp1_obs["availability_mask"], dtype=bool)[None, :],
                    dtype=torch.bool, device=self.device,
                )
                tp1_pre_trade = pre_trade_tp1[idx].unsqueeze(0).to(device=self.device, dtype=torch.float32)
                tp1_risk_state = None
                if "risk_state" in tp1_obs:
                    tp1_risk_state = torch.as_tensor(
                        np.asarray(tp1_obs["risk_state"], dtype=np.float32),
                        dtype=torch.float32, device=self.device,
                    )
                tp1_latent = self.model.encode_latent_from_observation(tp1_obs, risk_state=tp1_risk_state)

                tp1_candidate = self.model.target_actor(tp1_latent, tp1_mask, deterministic=True)
                tp1_projected_list = []
                for i in range(tp1_candidate.shape[0]):
                    proj_result = constraint_mgr.project(
                        tp1_candidate[i].detach().cpu().numpy(),
                        tp1_mask[i].detach().cpu().numpy(),
                        reference_weights=tp1_pre_trade[i].detach().cpu().numpy(),
                        asset_ids=asset_ids,
                    )
                    tp1_projected_list.append(proj_result.projected_weights)
                tp1_candidate_proj = torch.tensor(
                    np.stack(tp1_projected_list), device=self.device, dtype=torch.float32,
                )

                _, tp1_est_cost_candidate = estimate_candidate_cost(
                    tp1_obs, tp1_pre_trade, tp1_candidate_proj, self.config,
                )
                tp1_est_cost_hold = torch.zeros_like(tp1_est_cost_candidate)

                tp1_candidate_q = self.model.target_cqr_critic(tp1_latent, tp1_pre_trade, tp1_candidate_proj)
                tp1_hold_q = self.model.target_cqr_critic(tp1_latent, tp1_pre_trade, tp1_pre_trade)

                tp1_candidate_mean = tp1_candidate_q.mean(dim=1, keepdim=True)
                tp1_hold_mean = tp1_hold_q.mean(dim=1, keepdim=True)

                if self.model.quantile_tail_enabled:
                    tp1_candidate_ltl = OTarCQRCritic_static_lower_tail_loss(tp1_candidate_q, self.model.tail_alpha)
                    tp1_hold_ltl = OTarCQRCritic_static_lower_tail_loss(tp1_hold_q, self.model.tail_alpha)
                else:
                    tp1_candidate_ltl = torch.zeros_like(tp1_candidate_mean)
                    tp1_hold_ltl = torch.zeros_like(tp1_hold_mean)

                tp1_candidate_utility = tp1_candidate_mean - self.model.lambda_tail * tp1_candidate_ltl - tp1_est_cost_candidate
                tp1_hold_utility = tp1_hold_mean - self.model.lambda_tail * tp1_hold_ltl - tp1_est_cost_hold

                tp1_gate = (tp1_candidate_utility > tp1_hold_utility + self.model.gate_margin).long()

                tp1_action_q = tp1_candidate_q * tp1_gate + tp1_hold_q * (1.0 - tp1_gate)
                target_q[idx] = tp1_action_q.squeeze(0)

        y_j = realized_gross_t + self.model.gamma * target_q * (1.0 - terminated_t.float().view(-1, 1))

        tau = torch.linspace(0.0, 1.0, self.model.n_quantiles + 1, device=self.device)
        tau = 0.5 * (tau[:-1] + tau[1:])
        tau = tau.unsqueeze(0).expand(batch_size, -1)

        u = y_j - selected_quantiles_t
        huber = torch.where(
            u.abs() <= self.model.quantile_huber_kappa,
            0.5 * u.pow(2),
            self.model.quantile_huber_kappa * (u.abs() - 0.5 * self.model.quantile_huber_kappa),
        )
        rho = torch.abs(tau - (u < 0).float()) * huber
        qr_loss = rho.mean()
        return qr_loss

    def _critic_forward_from_batch(self, batch: dict[str, Any], suffix: str) -> torch.Tensor:
        state_key = "state_t" if suffix == "t" else "state_tp1"
        state_list = batch[state_key]
        candidate_weights = batch["candidate_weights_t"]
        pre_trade = batch["pre_trade_drifted_weights_t"]
        batch_size = candidate_weights.shape[0]

        quantiles_list = []
        for idx in range(batch_size):
            if isinstance(state_list, list):
                obs = state_list[idx]
            else:
                obs = state_list

            if isinstance(obs, dict):
                market_image = obs["market_image"]
                if not isinstance(market_image, torch.Tensor):
                    market_image = torch.as_tensor(np.asarray(market_image), dtype=torch.float32, device=self.device)
                if market_image.ndim == 3:
                    market_image = market_image.unsqueeze(0)
                market_image = market_image.to(device=self.device, dtype=torch.float32)
                latent = self.model.encoder(market_image)
                if self.model.use_risk_state and "risk_state" in obs:
                    rs = torch.as_tensor(np.asarray(obs["risk_state"]), dtype=torch.float32, device=self.device)
                    if rs.ndim == 1:
                        rs = rs.unsqueeze(0)
                    latent = torch.cat([latent, rs], dim=-1)
            else:
                latent = torch.zeros(1, self.model.effective_latent_dim, device=self.device)

            q = self.model.cqr_critic(latent, pre_trade[idx].unsqueeze(0), candidate_weights[idx].unsqueeze(0))
            quantiles_list.append(q)

        return torch.cat(quantiles_list, dim=0)

    def _hold_quantiles_from_batch(self, batch: dict[str, Any], suffix: str) -> torch.Tensor:
        state_key = "state_t" if suffix == "t" else "state_tp1"
        state_list = batch[state_key]
        pre_trade = batch["pre_trade_drifted_weights_t"]
        batch_size = pre_trade.shape[0]

        quantiles_list = []
        for idx in range(batch_size):
            if isinstance(state_list, list):
                obs = state_list[idx]
            else:
                obs = state_list

            if isinstance(obs, dict):
                market_image = obs["market_image"]
                if not isinstance(market_image, torch.Tensor):
                    market_image = torch.as_tensor(np.asarray(market_image), dtype=torch.float32, device=self.device)
                if market_image.ndim == 3:
                    market_image = market_image.unsqueeze(0)
                market_image = market_image.to(device=self.device, dtype=torch.float32)
                latent = self.model.encoder(market_image)
                if self.model.use_risk_state and "risk_state" in obs:
                    rs = torch.as_tensor(np.asarray(obs["risk_state"]), dtype=torch.float32, device=self.device)
                    if rs.ndim == 1:
                        rs = rs.unsqueeze(0)
                    latent = torch.cat([latent, rs], dim=-1)
            else:
                latent = torch.zeros(1, self.model.effective_latent_dim, device=self.device)

            q = self.model.cqr_critic(latent, pre_trade[idx].unsqueeze(0), pre_trade[idx].unsqueeze(0))
            quantiles_list.append(q)

        return torch.cat(quantiles_list, dim=0)


def OTarCQRCritic_static_lower_tail_loss(quantiles: torch.Tensor, tail_alpha: float) -> torch.Tensor:
    import math
    n_quantiles = quantiles.shape[1]
    n_below = max(1, math.ceil(tail_alpha * n_quantiles))
    sorted_q, _ = torch.sort(quantiles, dim=1)
    lower_tail = sorted_q[:, :n_below]
    return torch.clamp(-lower_tail.mean(dim=1, keepdim=True), min=0.0)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _observation_from_decision_state(
    state: DecisionMarketState,
    portfolio: PortfolioState,
) -> dict[str, Any]:
    return {
        "market_image": np.asarray(state.market_image, dtype=np.float32),
        "availability_mask": np.asarray(state.available_mask_at_decision, dtype=bool),
        "current_weights": np.asarray(portfolio.current_weights, dtype=np.float32),
        "adv20_at_decision": np.asarray(state.adv20_at_decision, dtype=np.float32),
        "volatility_20d_at_decision": np.asarray(state.volatility_20d_at_decision, dtype=np.float32),
        "amount_at_decision": np.asarray(state.amount_at_decision, dtype=np.float32),
        "turnover_rate_at_decision": np.asarray(state.turnover_rate_at_decision, dtype=np.float32),
        "portfolio_value": np.asarray(portfolio.portfolio_value, dtype=np.float32),
    }


__all__ = ["estimate_candidate_cost", "stack_observation_batch", "OTarCQRGateStrategy"]
