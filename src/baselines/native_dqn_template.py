from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.agents.dqn_agent import DQNAgent
from src.baselines.base_strategy import BaseStrategy
from src.baselines.eiie import _continuous_weight_rebalance_decision
from src.buffers.replay_buffer import ReplayItem
from src.data.splits import SplitSpec
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.models.encoders import EncoderFactory


DQN_TEMPLATE_ALGORITHM = "double_dqn_template_selector"
DQN_TEMPLATE_ACTIONS = (
    "hold",
    "equal_weight",
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
    "inverse_volatility",
    "defensive",
    "momentum_top_k",
)


class TemplateQNetwork(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )

    def forward(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor | None = None,
        current_weights: torch.Tensor | None = None,
        estimated_turnover: torch.Tensor | None = None,
        estimated_cost: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError("ERR_DQN_TEMPLATE_SHAPE: latent must be [batch,latent_dim]")
        return self.net(latent)


class MaskedTemplateDQNAgent(DQNAgent):
    def _q_values(self, batch: Mapping[str, Any], next_state: bool, target: bool) -> torch.Tensor:
        q_values = super()._q_values(batch, next_state=next_state, target=target)
        states = batch["state_tp1"] if next_state else batch["state_t"]
        masks = _valid_action_masks(states, q_values.device, q_values.shape[1])
        if masks is None:
            return q_values
        masked = q_values.masked_fill(~masks, -1.0e9)
        if next_state or "invalid_action_t" not in batch:
            return masked
        invalid_value = batch["invalid_action_t"]
        if isinstance(invalid_value, torch.Tensor):
            invalid = invalid_value.to(device=q_values.device, dtype=torch.bool).view(-1, 1)
        else:
            invalid = torch.as_tensor(
                np.asarray(invalid_value, dtype=bool),
                dtype=torch.bool,
                device=q_values.device,
            ).view(-1, 1)
        return torch.where(invalid, q_values, masked)


class NativeDQNTemplateStrategy(BaseStrategy):
    strategy_name = "dqn_template_native"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.fit_required = True
        self.training_result: dict[str, Any] | None = None
        self.training_history: pd.DataFrame = pd.DataFrame()
        self.device = _device(self.config)
        self.agent = self._build_agent()
        self._real_transition_count = 0

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> NativeDQNTemplateStrategy:
        if not isinstance(train_data, Mapping):
            self.training_result = _training_result(self.strategy_name, "failed_missing_train_data", pd.DataFrame())
            self.is_fitted = False
            return self

        train_dates = _dates(train_data.get("dates"))
        validation_dates = _dates(_mapping(validation_data).get("dates"))
        if validation_dates.empty:
            validation_dates = train_dates
        split = SplitSpec(
            train_dates=train_dates,
            validation_dates=validation_dates,
            test_dates=validation_dates,
            fold_id=str(_mapping(train_data.get("config")).get("fold_id", "baseline_native")),
        )
        dataset = train_data["dataset"]
        market_image_dataset = train_data.get("market_image_dataset")
        train_env = PortfolioRebalanceEnv(
            dataset,
            split,
            config=self.config,
            segment="train",
            market_image_dataset=market_image_dataset,
        )
        validation_env = PortfolioRebalanceEnv(
            dataset,
            split,
            config=self.config,
            segment="validation",
            market_image_dataset=market_image_dataset,
        )

        native_cfg = _native_rl_config(self.config)
        epochs = max(1, int(native_cfg.get("epochs", _mapping(self.config.get("training")).get("epochs", 1))))
        max_train_steps = _optional_positive_int(
            native_cfg.get("max_train_steps", _mapping(self.config.get("dqn_template")).get("max_train_steps"))
        )
        max_validation_steps = _optional_positive_int(
            native_cfg.get(
                "max_validation_steps",
                _mapping(self.config.get("dqn_template")).get("max_validation_steps"),
            )
        )
        max_gradient_updates_per_epoch = _optional_positive_int(
            native_cfg.get(
                "max_gradient_updates_per_epoch",
                _mapping(self.config.get("dqn_template")).get("max_gradient_updates_per_epoch"),
            )
        )
        update_threshold = max(int(self.agent.config.batch_size), int(self.agent.config.warmup_steps))
        checkpoint_paths = self._checkpoint_paths()
        history_rows: list[dict[str, Any]] = []
        best_metric = -np.inf
        env_steps = 0
        gradient_updates = 0

        for epoch in range(epochs):
            reward_total, step_count, update_stats = self._train_epoch(
                train_env,
                update_threshold,
                max_steps=max_train_steps,
                max_gradient_updates=max_gradient_updates_per_epoch,
            )
            env_steps += step_count
            gradient_updates += len(update_stats)
            validation_metric = _evaluate_template_policy(
                self.agent,
                validation_env,
                max_steps=max_validation_steps,
            )
            mean_loss = float(np.mean([item.get("loss", np.nan) for item in update_stats])) if update_stats else np.nan
            history_rows.append(
                {
                    "epoch": int(epoch),
                    "step": int(epoch + 1),
                    "env_steps": int(env_steps),
                    "gradient_updates": int(gradient_updates),
                    "train_reward": float(reward_total),
                    "validation_metric": float(validation_metric),
                    "loss": mean_loss,
                    "max_train_steps": max_train_steps,
                    "max_validation_steps": max_validation_steps,
                    "max_gradient_updates_per_epoch": max_gradient_updates_per_epoch,
                    "status": "completed",
                }
            )
            if np.isfinite(validation_metric) and validation_metric > best_metric:
                best_metric = float(validation_metric)
                self._save_agent_state(checkpoint_paths["best"], epoch, gradient_updates, best_metric)

        self._save_agent_state(
            checkpoint_paths["last"],
            epochs - 1,
            gradient_updates,
            None if not np.isfinite(best_metric) else best_metric,
        )

        history = pd.DataFrame(history_rows)
        if not _has_finite_validation(history):
            self.training_history = history
            self.training_result = _training_result(
                self.strategy_name,
                "failed_no_finite_validation_metric",
                history,
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            )
            self.is_fitted = False
            return self
        if checkpoint_paths["best"] is None or not checkpoint_paths["best"].exists():
            self.training_history = history
            self.training_result = _training_result(
                self.strategy_name,
                "failed_missing_best_checkpoint",
                history,
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            )
            self.is_fitted = False
            return self

        self._load_agent_state(checkpoint_paths["best"])
        self.training_history = history
        self.training_result = _training_result(
            self.strategy_name,
            "completed",
            history,
            checkpoint_best_path=_path_string(checkpoint_paths["best"]),
            checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            evaluated_checkpoint_path=_path_string(checkpoint_paths["best"]),
            best_validation_metric=float(best_metric),
            env_steps=env_steps,
            gradient_updates=gradient_updates,
            max_train_steps=max_train_steps,
            max_validation_steps=max_validation_steps,
            max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
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
        observation = _observation_from_state(state, portfolio)
        templates = template_weights_from_observation(observation, self.config)
        action_index = _greedy_valid_action(self.agent, observation, templates.valid_mask)
        weights = templates.weights[action_index]
        turnover = _turnover(weights, observation["current_weights"])
        q_values = _q_values_for_observation(self.agent, observation, apply_mask=False).detach().cpu().numpy()[0]
        reference_q = float(q_values[0])
        selected_q = float(q_values[int(action_index)])
        rebalance_decision = _template_rebalance_decision(
            self.config,
            self.strategy_name,
            portfolio,
            weights,
            int(action_index),
            getattr(self, "decision_context", {}),
        )
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=weights,
                rebalance_action=rebalance_decision["rebalance_action"],
                rebalance_intensity=rebalance_decision["rebalance_intensity"],
                action_info={
                    "strategy": self.strategy_name,
                    "training_algorithm": DQN_TEMPLATE_ALGORITHM,
                    "rl_training": True,
                    "platform_native_rl_training": True,
                    "template_chosen": DQN_TEMPLATE_ACTIONS[int(action_index)],
                    "gate_action": int(rebalance_decision["rebalance_action"]),
                    "gate_action_index": int(action_index),
                    "template_action_index": int(action_index),
                    "estimated_turnover": turnover,
                    "estimated_cost": 0.0,
                    "q_reference": reference_q,
                    "q_selected": selected_q,
                    "q_selected_minus_reference": selected_q - reference_q,
                    "q_hold": reference_q,
                    "q_rebalance": selected_q,
                    "q_gap": selected_q - reference_q,
                    **rebalance_decision["action_info"],
                },
            )
        )

    def _train_epoch(
        self,
        env: PortfolioRebalanceEnv,
        update_threshold: int,
        max_steps: int | None = None,
        max_gradient_updates: int | None = None,
    ) -> tuple[float, int, list[dict[str, float]]]:
        observation, _ = env.reset()
        terminated = False
        truncated = False
        reward_total = 0.0
        step_count = 0
        update_stats: list[dict[str, float]] = []
        while not (terminated or truncated):
            if max_steps is not None and step_count >= int(max_steps):
                break
            templates = template_weights_from_observation(observation, self.config)
            action_index = _epsilon_valid_action(self.agent, observation, templates.valid_mask)
            weights = templates.weights[action_index]
            turnover = _turnover(weights, observation["current_weights"])
            action = _template_env_action(self.config, self.strategy_name, observation, weights, int(action_index), turnover)
            next_observation, reward, terminated, truncated, info = env.step(action)
            transition_info = {**action, **dict(info)}
            next_templates = template_weights_from_observation(next_observation, self.config)
            state_t = dict(observation)
            state_tp1 = dict(next_observation)
            state_t["valid_action_mask"] = templates.valid_mask.copy()
            state_tp1["valid_action_mask"] = next_templates.valid_mask.copy()
            self.agent.replay_buffer.add_transition(
                _replay_item(
                    state_t,
                    state_tp1,
                    weights,
                    action_index,
                    reward,
                    terminated,
                    truncated,
                    transition_info,
                    turnover,
                    self.agent,
                )
            )
            self._real_transition_count += 1
            invalid_penalty = _invalid_action_penalty(self.config)
            for invalid_index in np.flatnonzero(~templates.valid_mask):
                self.agent.replay_buffer.add(
                    _replay_item(
                        state_t,
                        state_t,
                        templates.weights[int(invalid_index)],
                        int(invalid_index),
                        float(reward) - invalid_penalty,
                        True,
                        True,
                        info,
                        _turnover(templates.weights[int(invalid_index)], observation["current_weights"]),
                        self.agent,
                        invalid_action=True,
                        bootstrap_mask=0.0,
                        next_state_source="none",
                    )
                )
            can_update = (
                self._real_transition_count >= int(update_threshold)
                and len(self.agent.replay_buffer) >= int(self.agent.config.batch_size)
                and (max_gradient_updates is None or len(update_stats) < int(max_gradient_updates))
            )
            if can_update:
                update_stats.append(self.agent.update())
            observation = next_observation
            reward_total += float(reward)
            step_count += 1
        return reward_total, step_count, update_stats

    def _build_agent(self) -> DQNAgent:
        config = dict(self.config)
        encoder = EncoderFactory.create(config)
        target_encoder = EncoderFactory.create(config)
        target_encoder.load_state_dict(encoder.state_dict())
        latent_dim = int(config.get("latent_dim", _mapping(config.get("model")).get("latent_dim", 256)))
        online = TemplateQNetwork(latent_dim, len(DQN_TEMPLATE_ACTIONS))
        target = TemplateQNetwork(latent_dim, len(DQN_TEMPLATE_ACTIONS))
        target.load_state_dict(online.state_dict())
        return MaskedTemplateDQNAgent(
            online,
            target,
            config=config,
            device=self.device,
            encoder=encoder,
            target_encoder=target_encoder,
        )

    def _checkpoint_paths(self) -> dict[str, Path | None]:
        checkpoint_dir = _mapping(self.config.get("baselines")).get("checkpoint_dir")
        if checkpoint_dir is None:
            checkpoint_dir = self.config.get("baseline_run_dir")
        if checkpoint_dir is None:
            return {"best": None, "last": None}
        root = Path(checkpoint_dir) / "checkpoints" / self.strategy_name
        return {"best": root / "best.pt", "last": root / "last.pt"}

    def _save_agent_state(self, path: Path | None, epoch: int, global_step: int, best_metric: float | None) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_network_state": self.agent.online_network.state_dict(),
                "target_network_state": self.agent.target_network.state_dict(),
                "encoder_state": None if self.agent.encoder is None else self.agent.encoder.state_dict(),
                "target_encoder_state": None if self.agent.target_encoder is None else self.agent.target_encoder.state_dict(),
                "optimizer_state": self.agent.optimizer.state_dict(),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_validation_metric": best_metric,
            },
            path,
        )

    def _load_agent_state(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.online_network.load_state_dict(payload["online_network_state"])
        self.agent.target_network.load_state_dict(payload["target_network_state"])
        if self.agent.encoder is not None and payload.get("encoder_state") is not None:
            self.agent.encoder.load_state_dict(payload["encoder_state"])
        if self.agent.target_encoder is not None and payload.get("target_encoder_state") is not None:
            self.agent.target_encoder.load_state_dict(payload["target_encoder_state"])
        self.agent.optimizer.load_state_dict(payload["optimizer_state"])


class TemplateWeights:
    def __init__(self, weights: np.ndarray, valid_mask: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float32)
        self.valid_mask = np.asarray(valid_mask, dtype=bool)


def template_weights_from_observation(observation: Mapping[str, Any], config: Mapping[str, Any]) -> TemplateWeights:
    mask = np.asarray(observation["availability_mask"], dtype=bool)
    image = np.asarray(observation["market_image"], dtype=np.float32)
    returns = _return_window(image, len(mask))
    volatility = np.asarray(observation.get("volatility_20d_at_decision"), dtype=np.float32)
    candidates = [
        _hold_weight(mask, observation),
        _equal_weight(mask),
        _minimum_variance(mask, returns),
        _maximum_sharpe(mask, returns),
        _risk_parity(mask, returns),
        _inverse_volatility(mask, volatility),
        _defensive(mask, volatility),
        _momentum_top_k(mask, returns, config),
    ]
    weights: list[np.ndarray] = []
    valid: list[bool] = []
    fallback = _equal_weight(mask)
    for candidate in candidates:
        normalized = _normalize(candidate, mask)
        valid_candidate = normalized is not None
        weights.append(fallback if normalized is None else normalized)
        valid.append(valid_candidate)
    valid[0] = True
    return TemplateWeights(np.stack(weights, axis=0), np.asarray(valid, dtype=bool))


def _evaluate_template_policy(
    agent: DQNAgent,
    env: PortfolioRebalanceEnv,
    max_steps: int | None = None,
) -> float:
    observation, _ = env.reset()
    terminated = False
    truncated = False
    rewards: list[float] = []
    while not (terminated or truncated):
        if max_steps is not None and len(rewards) >= int(max_steps):
            break
        templates = template_weights_from_observation(observation, env.config)
        action_index = _greedy_valid_action(agent, observation, templates.valid_mask)
        weights = templates.weights[action_index]
        turnover = _turnover(weights, observation["current_weights"])
        action = _template_env_action(env.config, "dqn_template_native", observation, weights, int(action_index), turnover)
        observation, reward, terminated, truncated, _ = env.step(
            action
        )
        rewards.append(float(reward))
    if not rewards:
        return float("-inf")
    return float(np.sum(rewards))


def _epsilon_valid_action(agent: DQNAgent, observation: Mapping[str, Any], valid_mask: np.ndarray) -> int:
    q_values = _q_values(agent, observation)
    valid_indices = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    if valid_indices.size == 0:
        return 0
    compact_q = q_values[:, valid_indices]
    compact_action = agent.select_action(compact_q, agent.global_step)
    return int(valid_indices[int(compact_action.view(-1)[0].detach().cpu())])


def _greedy_valid_action(agent: DQNAgent, observation: Mapping[str, Any], valid_mask: np.ndarray) -> int:
    q_values = _q_values(agent, observation).detach().cpu().numpy()[0]
    valid_indices = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    if valid_indices.size == 0:
        return 0
    return int(valid_indices[int(np.argmax(q_values[valid_indices]))])


def _q_values(agent: DQNAgent, observation: Mapping[str, Any]) -> torch.Tensor:
    return _q_values_for_observation(agent, observation, apply_mask=True)


def _q_values_for_observation(agent: DQNAgent, observation: Mapping[str, Any], *, apply_mask: bool) -> torch.Tensor:
    state = [dict(observation)]
    batch = {
        "state_t": state,
        "candidate_weights_t": np.asarray([observation["current_weights"]], dtype=np.float32),
        "current_weights_t": np.asarray([observation["current_weights"]], dtype=np.float32),
        "estimated_turnover_t": np.asarray([0.0], dtype=np.float32),
        "estimated_cost_t": np.asarray([0.0], dtype=np.float32),
    }
    with torch.no_grad():
        if apply_mask:
            return agent._q_values(batch, next_state=False, target=False)
        return DQNAgent._q_values(agent, batch, next_state=False, target=False)


def _valid_action_masks(states: Any, device: torch.device, action_dim: int) -> torch.Tensor | None:
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        return None
    masks = []
    for state in states:
        if not isinstance(state, Mapping) or "valid_action_mask" not in state:
            return None
        mask = np.asarray(state["valid_action_mask"], dtype=bool)
        if mask.shape != (int(action_dim),):
            return None
        masks.append(mask)
    return torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)


def _replay_item(
    observation: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    weights: np.ndarray,
    action_index: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    turnover: float,
    agent: DQNAgent,
    *,
    invalid_action: bool = False,
    bootstrap_mask: float = 1.0,
    next_state_source: str = "env",
) -> ReplayItem:
    decision_date = pd.Timestamp(info.get("decision_date"))
    execution_date = pd.Timestamp(info.get("execution_date", decision_date))
    next_valuation_date = pd.Timestamp(info.get("next_valuation_date", execution_date))
    q_values = _q_values_for_observation(agent, observation, apply_mask=not invalid_action).detach().cpu().numpy()[0]
    reference_q = float(q_values[0])
    selected_q = float(q_values[int(action_index)])
    return ReplayItem(
        state_t=dict(observation),
        state_tp1=dict(next_observation),
        decision_date_t=decision_date,
        execution_date_t=execution_date,
        next_valuation_date_t=next_valuation_date,
        decision_date_next=next_valuation_date,
        execution_date_next=next_valuation_date,
        next_valuation_date_next=next_valuation_date,
        execution_price_t=str(info.get("execution_price_type", "open")),
        delayed_action_execution_t=bool(info.get("delayed_action_execution", False)),
        candidate_weights_t=np.asarray(weights, dtype=np.float32),
        executed_weights_t=np.asarray(info.get("executed_weights", weights), dtype=np.float32),
        gate_action_t=int(action_index),
        rebalance_action_t=int(info.get("rebalance_action", 1)),
        rebalance_intensity_t=float(info.get("rebalance_intensity", 1.0)),
        estimated_turnover_t=float(turnover),
        realized_turnover_t=float(info.get("realized_turnover", info.get("turnover", turnover))),
        estimated_cost_t=0.0,
        realized_cost_t=float(info.get("realized_cost", 0.0) or 0.0),
        reward_t=float(reward),
        terminated_t=bool(terminated),
        truncated_t=bool(truncated),
        q_hold_t=reference_q,
        q_rebalance_t=selected_q,
        q_gap_t=selected_q - reference_q,
        q_reference_t=reference_q,
        q_selected_t=selected_q,
        q_selected_minus_reference_t=selected_q - reference_q,
        invalid_action_t=bool(invalid_action),
        bootstrap_mask_t=float(bootstrap_mask),
        next_state_source_t=str(next_state_source),
    )


def _observation_from_state(state: DecisionMarketState, portfolio: PortfolioState) -> dict[str, Any]:
    return {
        "market_image": np.asarray(state.market_image, dtype=np.float32),
        "current_weights": np.asarray(portfolio.current_weights, dtype=np.float32),
        "availability_mask": np.asarray(state.available_mask_at_decision, dtype=np.int8),
        "adv20_at_decision": np.nan_to_num(np.asarray(state.adv20_at_decision, dtype=np.float32)),
        "volatility_20d_at_decision": np.nan_to_num(np.asarray(state.volatility_20d_at_decision, dtype=np.float32)),
        "amount_at_decision": np.nan_to_num(np.asarray(state.amount_at_decision, dtype=np.float32)),
        "turnover_rate_at_decision": np.nan_to_num(np.asarray(state.turnover_rate_at_decision, dtype=np.float32)),
        "portfolio_value": np.asarray(portfolio.portfolio_value, dtype=np.float32),
    }


def _template_env_action(
    config: Mapping[str, Any],
    model_key: str,
    observation: Mapping[str, Any],
    weights: np.ndarray,
    action_index: int,
    turnover: float | None = None,
) -> dict[str, Any]:
    portfolio = _portfolio_state_from_observation(observation)
    rebalance_decision = _template_rebalance_decision(config, model_key, portfolio, weights, action_index, None)
    return {
        "weights": np.asarray(weights, dtype=np.float32),
        "rebalance": rebalance_decision["rebalance_action"],
        "rebalance_action": rebalance_decision["rebalance_action"],
        "rebalance_intensity": rebalance_decision["rebalance_intensity"],
        "template_chosen": DQN_TEMPLATE_ACTIONS[int(action_index)],
        "gate_action": int(rebalance_decision["rebalance_action"]),
        "gate_action_index": int(action_index),
        "template_action_index": int(action_index),
        "estimated_turnover": float(rebalance_decision["action_info"]["estimated_turnover"] if turnover is None else turnover),
        "estimated_cost": 0.0,
        **rebalance_decision["action_info"],
    }


def _template_rebalance_decision(
    config: Mapping[str, Any],
    model_key: str,
    portfolio_state: PortfolioState,
    weights: np.ndarray,
    action_index: int,
    decision_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if int(action_index) == 0 and float(np.asarray(portfolio_state.current_weights, dtype=float).sum()) > 0.0:
        turnover = _turnover(weights, portfolio_state.current_weights)
        return {
            "rebalance_action": 0,
            "rebalance_intensity": 0.0,
            "action_info": {
                "continuous_weight_rebalance_gate": True,
                "template_hold_action": True,
                "estimated_turnover": turnover,
                "candidate_turnover": turnover,
                "candidate_turnover_estimate": turnover,
                "raw_model_requested_rebalance": False,
                "raw_action": 0,
                "raw_rho": 0.0,
                "raw_rebalance_intensity": 0.0,
                "rebalance_intensity": 0.0,
                "forced_hold_reason": "model_chosen_hold",
            },
        }
    decision = _continuous_weight_rebalance_decision(
        config,
        model_key,
        portfolio_state,
        np.asarray(weights, dtype=float),
        decision_context,
    )
    decision["action_info"]["template_hold_action"] = int(action_index) == 0
    return decision


def _portfolio_state_from_observation(observation: Mapping[str, Any]) -> PortfolioState:
    current_weights = np.asarray(observation.get("current_weights"), dtype=float)
    portfolio_value = float(np.asarray(observation.get("portfolio_value", 0.0), dtype=float))
    return PortfolioState(
        date=pd.Timestamp("1970-01-01"),
        nav=1.0,
        portfolio_value=portfolio_value,
        current_weights=current_weights,
        step_index=0 if float(current_weights.sum()) <= 0.0 else 1,
    )


def _hold_weight(mask: np.ndarray, observation: Mapping[str, Any]) -> np.ndarray | None:
    current = np.asarray(observation.get("current_weights"), dtype=np.float32)
    if current.shape != mask.shape or not np.isfinite(current).all():
        return None
    if float(np.maximum(current, 0.0).sum()) <= 0.0:
        return _equal_weight(mask)
    return np.maximum(current, 0.0)


def _equal_weight(mask: np.ndarray) -> np.ndarray:
    weights = np.zeros(mask.shape, dtype=np.float32)
    if mask.any():
        weights[mask] = 1.0 / float(mask.sum())
    return weights


def _minimum_variance(mask: np.ndarray, returns: np.ndarray | None) -> np.ndarray | None:
    if returns is None or returns.shape[0] < 2:
        return None
    active = returns[:, mask]
    covariance = np.cov(active, rowvar=False)
    if covariance.ndim == 0:
        return _equal_weight(mask)
    try:
        inv = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * 1.0e-6)
        raw = inv @ np.ones(active.shape[1], dtype=np.float32)
    except Exception:
        return None
    weights = np.zeros(mask.shape, dtype=np.float32)
    weights[mask] = raw
    return weights


def _maximum_sharpe(mask: np.ndarray, returns: np.ndarray | None) -> np.ndarray | None:
    if returns is None:
        return None
    mean = np.nanmean(returns[:, mask], axis=0)
    volatility = np.nanstd(returns[:, mask], axis=0)
    raw = np.maximum(mean, 0.0) / np.maximum(volatility, 1.0e-6)
    weights = np.zeros(mask.shape, dtype=np.float32)
    weights[mask] = raw
    return weights


def _risk_parity(mask: np.ndarray, returns: np.ndarray | None) -> np.ndarray | None:
    if returns is None:
        return None
    volatility = np.nanstd(returns[:, mask], axis=0)
    weights = np.zeros(mask.shape, dtype=np.float32)
    weights[mask] = 1.0 / np.maximum(volatility, 1.0e-6)
    return weights


def _inverse_volatility(mask: np.ndarray, volatility: np.ndarray) -> np.ndarray | None:
    if volatility.shape[0] != mask.shape[0]:
        return None
    weights = np.zeros(mask.shape, dtype=np.float32)
    weights[mask] = 1.0 / np.maximum(volatility[mask], 1.0e-6)
    return weights


def _defensive(mask: np.ndarray, volatility: np.ndarray) -> np.ndarray | None:
    if volatility.shape[0] != mask.shape[0] or not mask.any():
        return None
    active_indices = np.flatnonzero(mask)
    chosen = active_indices[int(np.argmin(volatility[active_indices]))]
    weights = np.zeros(mask.shape, dtype=np.float32)
    weights[chosen] = 1.0
    return weights


def _momentum_top_k(mask: np.ndarray, returns: np.ndarray | None, config: Mapping[str, Any]) -> np.ndarray | None:
    if returns is None:
        return None
    active_indices = np.flatnonzero(mask)
    if active_indices.size == 0:
        return None
    dqn_config = _mapping(config.get("dqn_template"))
    top_k = max(1, min(int(dqn_config.get("momentum_top_k", min(3, active_indices.size))), active_indices.size))
    scores = np.nanmean(returns[:, active_indices], axis=0)
    selected = active_indices[np.argsort(scores)[-top_k:]]
    weights = np.zeros(mask.shape, dtype=np.float32)
    weights[selected] = 1.0 / float(len(selected))
    return weights


def _return_window(image: np.ndarray, n_assets: int) -> np.ndarray | None:
    if image.ndim != 3 or image.shape[-1] != n_assets:
        return None
    window = np.asarray(image[0], dtype=np.float32)
    if window.ndim != 2 or window.shape[1] != n_assets:
        return None
    return np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize(weights: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if weights is None:
        return None
    result = np.asarray(weights, dtype=np.float32).copy()
    if result.shape != mask.shape or not np.isfinite(result).all():
        return None
    result[~mask] = 0.0
    result = np.maximum(result, 0.0)
    total = float(result.sum())
    if total <= 0.0 or not np.isfinite(total):
        return None
    return (result / total).astype(np.float32, copy=False)


def _turnover(weights: np.ndarray, current_weights: Any) -> float:
    current = np.asarray(current_weights, dtype=np.float32)
    return float(0.5 * np.sum(np.abs(np.asarray(weights, dtype=np.float32) - current)))


def _invalid_action_penalty(config: Mapping[str, Any]) -> float:
    dqn_template = _mapping(config.get("dqn_template"))
    return float(dqn_template.get("invalid_action_penalty", 1.0))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("ERR_DQN_TEMPLATE_CONFIG_INVALID: max step limits must be > 0")
    return result


def _training_result(
    model_name: str,
    status: str,
    training_history: pd.DataFrame,
    *,
    checkpoint_best_path: str | None = None,
    checkpoint_last_path: str | None = None,
    evaluated_checkpoint_path: str | None = None,
    best_validation_metric: float | None = None,
    env_steps: int = 0,
    gradient_updates: int = 0,
    max_train_steps: int | None = None,
    max_validation_steps: int | None = None,
    max_gradient_updates_per_epoch: int | None = None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "baseline_family": "native_rl",
        "status": status,
        "training_algorithm": DQN_TEMPLATE_ALGORITHM,
        "rl_training": True,
        "platform_native_rl_training": True,
        "proxy_training": False,
        "external_original_implementation": False,
        "rankable_in_unified_table": True,
        "training_history": training_history,
        "checkpoint_best_path": checkpoint_best_path,
        "checkpoint_last_path": checkpoint_last_path,
        "evaluated_checkpoint_path": evaluated_checkpoint_path,
        "best_validation_metric": best_validation_metric,
        "env_steps": int(env_steps),
        "gradient_updates": int(gradient_updates),
        "max_train_steps": max_train_steps,
        "max_validation_steps": max_validation_steps,
        "max_gradient_updates_per_epoch": max_gradient_updates_per_epoch,
    }


def _has_finite_validation(history: pd.DataFrame) -> bool:
    if history.empty or "validation_metric" not in history.columns:
        return False
    values = pd.to_numeric(history["validation_metric"], errors="coerce")
    return bool(np.isfinite(values).any())


def _native_rl_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    baselines = _mapping(config.get("baselines"))
    return _mapping(baselines.get("native_rl") or baselines.get("native_training"))


def _dates(value: Any) -> pd.DatetimeIndex:
    if value is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(list(value))).sort_values()


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


def _path_string(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "DQN_TEMPLATE_ACTIONS",
    "MaskedTemplateDQNAgent",
    "NativeDQNTemplateStrategy",
    "TemplateQNetwork",
    "template_weights_from_observation",
]
