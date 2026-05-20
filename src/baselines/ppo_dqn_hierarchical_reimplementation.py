from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.dqn_agent import DQNAgent, DQNAgentConfig
from src.baselines.base_strategy import BaseStrategy
from src.buffers.replay_buffer import ReplayItem
from src.data.leakage_checks import assert_decision_visibility_contract
from src.data.loader import DataContractError
from src.data.splits import SplitSpec
from src.envs.backtest_engine import BacktestEngine
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.models.encoders import EncoderFactory
from src.models.ppo_actor import PPOActor
from src.models.ppo_critic import PPOCritic
from src.utils.checkpoint import CHECKPOINT_SCHEMA_VERSION, load_checkpoint, save_checkpoint


PPO_DQN_HIERARCHICAL_REIMPLEMENTATION = "ppo_dqn_hierarchical_reimplementation"
PPO_DQN_HIGH_LEVEL_ACTION_SELECTOR = "high_level_action_selector"
PPO_DQN_HIERARCHY_ACTION_DIM = 5
PPO_DQN_HIERARCHY_ACTION_NAMES = {
    0: "use_ppo_candidate",
    1: "blend_current_ppo_25",
    2: "blend_current_ppo_50",
    3: "blend_current_ppo_75",
    4: "fallback_equal_weight",
}
PPO_DQN_HIERARCHY_BLEND_RHOS = {1: 0.25, 2: 0.50, 3: 0.75}


class HierarchyQNetwork(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_assets: int,
        action_dim: int = PPO_DQN_HIERARCHY_ACTION_DIM,
        hidden_dims: Sequence[int] | None = None,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.n_assets = _positive_int("n_assets", n_assets)
        self.action_dim = _positive_int("action_dim", action_dim)
        hidden = _hidden_dims(hidden_dims, default=(256, 128))
        self.input_dim = self.latent_dim + 2 * self.n_assets + 2

        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for out_dim in hidden:
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(float(dropout))])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor | None = None,
        current_weights: torch.Tensor | None = None,
        estimated_turnover: torch.Tensor | None = None,
        estimated_cost: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("ERR_PPO_DQN_HIERARCHY_SHAPE: latent must be [batch,latent_dim]")
        batch_size = int(latent.shape[0])
        candidate = _feature_tensor(candidate_weights, batch_size, self.n_assets, latent)
        current = _feature_tensor(current_weights, batch_size, self.n_assets, latent)
        turnover = _feature_tensor(estimated_turnover, batch_size, 1, latent)
        cost = _feature_tensor(estimated_cost, batch_size, 1, latent)
        if not _all_finite(latent, candidate, current, turnover, cost):
            raise ValueError("ERR_PPO_DQN_HIERARCHY_NON_FINITE: hierarchy q inputs contain NaN or Inf")
        return self.net(torch.cat([latent, candidate, current, turnover, cost], dim=1))


class PPODQNHierarchicalReimplementationStrategy(BaseStrategy):
    strategy_name = PPO_DQN_HIERARCHICAL_REIMPLEMENTATION
    fit_required = True

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.device = _device(self.config)
        self.model_config = _resolved_model_config(self.config)
        self.dqn_role = _configured_dqn_role(self.config)
        self.n_assets = int(self.model_config["n_assets"])
        self.n_features = int(self.model_config["n_features"])
        self.window_size = int(self.model_config["window_size"])
        self.latent_dim = int(self.model_config["latent_dim"])
        self.encoder = EncoderFactory.create(self.model_config).to(self.device)
        self.target_encoder = EncoderFactory.create(self.model_config).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        ppo_config = _mapping(self.model_config.get("ppo"))
        dqn_config = _mapping(self.model_config.get("dqn"))
        model_config = _mapping(self.model_config.get("model"))
        self.ppo_actor = PPOActor(
            latent_dim=self.latent_dim,
            n_assets=self.n_assets,
            min_alpha=float(ppo_config.get("min_alpha", ppo_config.get("actor_min_alpha", 1.0e-3))),
            hidden_dims=_hidden_dims(
                ppo_config.get("actor_hidden_dims", ppo_config.get("hidden_dims")),
                default=(256, 128),
            ),
        ).to(self.device)
        self.ppo_critic = PPOCritic(
            latent_dim=self.latent_dim,
            hidden_dims=_hidden_dims(
                ppo_config.get("critic_hidden_dims", ppo_config.get("hidden_dims")),
                default=(256, 128),
            ),
        ).to(self.device)
        self.hierarchy_q_network = HierarchyQNetwork(
            latent_dim=self.latent_dim,
            n_assets=self.n_assets,
            action_dim=PPO_DQN_HIERARCHY_ACTION_DIM,
            hidden_dims=_hidden_dims(dqn_config.get("hidden_dims"), default=(256, 128)),
            dropout=float(dqn_config.get("dropout", model_config.get("dropout", 0.10))),
        ).to(self.device)
        self.target_hierarchy_q_network = HierarchyQNetwork(
            latent_dim=self.latent_dim,
            n_assets=self.n_assets,
            action_dim=PPO_DQN_HIERARCHY_ACTION_DIM,
            hidden_dims=_hidden_dims(dqn_config.get("hidden_dims"), default=(256, 128)),
            dropout=float(dqn_config.get("dropout", model_config.get("dropout", 0.10))),
        ).to(self.device)
        self.target_hierarchy_q_network.load_state_dict(self.hierarchy_q_network.state_dict())
        self.dqn_config = DQNAgentConfig.from_mapping(self.model_config)
        self.ppo_optimizer = torch.optim.AdamW(
            _unique_parameters(self.encoder, self.ppo_actor, self.ppo_critic),
            lr=_ppo_lr(self.model_config),
        )
        self.dqn_optimizer = torch.optim.AdamW(
            _dqn_parameters(self.hierarchy_q_network, self.encoder, self.dqn_config),
            lr=self.dqn_config.lr,
        )
        self.dqn_agent = DQNAgent(
            self.hierarchy_q_network,
            self.target_hierarchy_q_network,
            optimizer=self.dqn_optimizer,
            config=self.dqn_config,
            device=self.device,
            encoder=self.encoder,
            target_encoder=self.target_encoder,
        )
        self.replay_buffer = self.dqn_agent.replay_buffer
        self._real_transition_count = 0
        self._checkpoint_loaded_path: str | None = None
        self.training_history: pd.DataFrame = pd.DataFrame()
        self.training_result: dict[str, Any] = _training_result(
            self.strategy_name,
            self.training_history,
            dqn_role=self.dqn_role,
        )

    def fit(
        self,
        train_data: Any | None = None,
        validation_data: Any | None = None,
    ) -> PPODQNHierarchicalReimplementationStrategy:
        self.training_history = pd.DataFrame()
        self.is_fitted = False
        if self.dqn_role != PPO_DQN_HIGH_LEVEL_ACTION_SELECTOR:
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status="deferred_variant",
                reason="unsupported_dqn_role",
                dqn_role=self.dqn_role,
            )
            return self

        if not isinstance(train_data, Mapping):
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status="failed_missing_train_data",
                reason="missing_train_data",
                dqn_role=self.dqn_role,
            )
            return self

        self._real_transition_count = 0
        self.replay_buffer.clear()

        train_dates = _dates(train_data.get("dates"))
        if train_dates.empty:
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status="failed_missing_train_data",
                reason="empty_train_dates",
                dqn_role=self.dqn_role,
            )
            return self

        validation_dates = _dates(_mapping(validation_data).get("dates"))
        if len(validation_dates) < 2:
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status="failed_no_finite_validation_metric",
                reason="missing_validation_data",
                dqn_role=self.dqn_role,
            )
            return self
        split = SplitSpec(
            train_dates=train_dates,
            validation_dates=validation_dates,
            test_dates=validation_dates,
            fold_id=str(_mapping(train_data.get("config")).get("fold_id", "ppo_dqn_hierarchical")),
        )

        try:
            dataset = train_data["dataset"]
        except KeyError:
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status="failed_missing_train_data",
                reason="missing_dataset",
                dqn_role=self.dqn_role,
            )
            return self

        market_image_dataset = train_data.get("market_image_dataset")
        try:
            train_env = PortfolioRebalanceEnv(
                dataset,
                split,
                config=self.config,
                segment="train",
                market_image_dataset=market_image_dataset,
            )
        except DataContractError as exc:
            if exc.code != "ERR_SPLIT_EMPTY":
                raise
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status="failed_missing_train_data",
                reason="empty_train_env",
                dqn_role=self.dqn_role,
            )
            return self

        native_cfg = _native_rl_config(self.config)
        training_cfg = _mapping(self.config.get("training"))
        epochs = max(1, int(native_cfg.get("epochs", training_cfg.get("epochs", 1))))
        max_train_steps = _optional_positive_int(native_cfg.get("max_train_steps", training_cfg.get("max_train_steps")))
        max_validation_steps = _optional_positive_int(
            native_cfg.get("max_validation_steps", training_cfg.get("max_validation_steps"))
        )
        max_gradient_updates_per_epoch = _optional_positive_int(
            native_cfg.get(
                "max_gradient_updates_per_epoch",
                training_cfg.get("max_gradient_updates_per_epoch"),
            )
        )

        history_rows: list[dict[str, Any]] = []
        env_steps = 0
        gradient_updates = 0
        failure_status: str | None = None
        failure_reason: str | None = None
        checkpoint_paths = self._checkpoint_paths()
        best_validation_metric = -np.inf
        platform_adapted_surrogate = False

        for epoch in range(epochs):
            epoch_result = self._train_epoch(
                train_env,
                max_steps=max_train_steps,
                max_gradient_updates=max_gradient_updates_per_epoch,
            )
            platform_adapted_surrogate = platform_adapted_surrogate or bool(
                epoch_result.get("platform_adapted_surrogate", False)
            )
            env_steps += int(epoch_result["env_steps"])
            gradient_updates += int(epoch_result["gradient_updates"])
            if epoch_result["status"] != "completed":
                failure_status = str(epoch_result["status"])
                failure_reason = str(epoch_result.get("reason") or failure_status)
                history_rows.append(
                    {
                        "epoch": int(epoch),
                        "step": int(epoch + 1),
                        "env_steps": int(env_steps),
                        "gradient_updates": int(gradient_updates),
                        "train_reward": float(epoch_result.get("train_reward", np.nan)),
                        "validation_metric": np.nan,
                        "loss": float(epoch_result.get("loss", np.nan)),
                        "max_train_steps": max_train_steps,
                        "max_validation_steps": max_validation_steps,
                        "max_gradient_updates_per_epoch": max_gradient_updates_per_epoch,
                        "platform_adapted_surrogate": bool(epoch_result.get("platform_adapted_surrogate", False)),
                        "status": failure_status,
                    }
                )
                break

            validation_metric = _validation_metric_from_backtest(
                self,
                dataset,
                split,
                market_image_dataset=market_image_dataset,
                max_steps=max_validation_steps,
            )
            history_rows.append(
                {
                    "epoch": int(epoch),
                    "step": int(epoch + 1),
                    "env_steps": int(env_steps),
                    "gradient_updates": int(gradient_updates),
                    "train_reward": float(epoch_result["train_reward"]),
                    "validation_metric": float(validation_metric),
                    "loss": float(epoch_result["loss"]),
                    "max_train_steps": max_train_steps,
                    "max_validation_steps": max_validation_steps,
                    "max_gradient_updates_per_epoch": max_gradient_updates_per_epoch,
                    "platform_adapted_surrogate": bool(epoch_result.get("platform_adapted_surrogate", False)),
                    "status": "completed",
                }
            )
            if np.isfinite(validation_metric) and validation_metric > best_validation_metric:
                best_validation_metric = float(validation_metric)
                self._save_checkpoint_state(checkpoint_paths["best"], epoch, gradient_updates, best_validation_metric)

        self.training_history = pd.DataFrame(history_rows)
        if failure_status is not None:
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status=failure_status,
                reason=failure_reason,
                env_steps=env_steps,
                gradient_updates=gradient_updates,
                dqn_role=self.dqn_role,
                platform_adapted_surrogate=platform_adapted_surrogate,
            )
            self.is_fitted = False
            return self

        self._save_checkpoint_state(
            checkpoint_paths["last"],
            epochs - 1,
            gradient_updates,
            None if not np.isfinite(best_validation_metric) else best_validation_metric,
        )
        if not _has_finite_validation(self.training_history):
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status="failed_no_finite_validation_metric",
                reason="failed_no_finite_validation_metric",
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
                env_steps=env_steps,
                gradient_updates=gradient_updates,
                max_train_steps=max_train_steps,
                max_validation_steps=max_validation_steps,
                max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
                dqn_role=self.dqn_role,
                platform_adapted_surrogate=platform_adapted_surrogate,
            )
            self.is_fitted = False
            return self
        if checkpoint_paths["best"] is None or not checkpoint_paths["best"].exists():
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status="failed_missing_best_checkpoint",
                reason="failed_missing_best_checkpoint",
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
                best_validation_metric=float(best_validation_metric),
                env_steps=env_steps,
                gradient_updates=gradient_updates,
                max_train_steps=max_train_steps,
                max_validation_steps=max_validation_steps,
                max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
                dqn_role=self.dqn_role,
                platform_adapted_surrogate=platform_adapted_surrogate,
            )
            self.is_fitted = False
            return self
        try:
            self._load_checkpoint_state(checkpoint_paths["best"])
        except Exception:
            self.training_result = _training_result(
                self.strategy_name,
                self.training_history,
                status="failed_checkpoint_load",
                reason="failed_checkpoint_load",
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
                best_validation_metric=float(best_validation_metric),
                env_steps=env_steps,
                gradient_updates=gradient_updates,
                max_train_steps=max_train_steps,
                max_validation_steps=max_validation_steps,
                max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
                dqn_role=self.dqn_role,
                platform_adapted_surrogate=platform_adapted_surrogate,
            )
            self.is_fitted = False
            return self

        self.training_result = _training_result(
            self.strategy_name,
            self.training_history,
            status="completed",
            reason=None,
            checkpoint_best_path=_path_string(checkpoint_paths["best"]),
            checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            evaluated_checkpoint_path=_path_string(checkpoint_paths["best"]),
            best_validation_metric=float(best_validation_metric),
            env_steps=env_steps,
            gradient_updates=gradient_updates,
            rankable_in_unified_table=True,
            max_train_steps=max_train_steps,
            max_validation_steps=max_validation_steps,
            max_gradient_updates_per_epoch=max_gradient_updates_per_epoch,
            dqn_role=self.dqn_role,
            platform_adapted_surrogate=platform_adapted_surrogate,
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
        if self.is_fitted is not True:
            raise DataContractError(
                "ERR_STRATEGY_ACTION_CONTRACT",
                f"ERR_STRATEGY_ACTION_CONTRACT: {self.strategy_name} not fitted",
            )
        return self.validate_portfolio_action(self._policy_action(state, portfolio))

    def _mask_normalize_weights(
        self,
        weights: Any,
        available_mask_at_decision: Any,
    ) -> dict[str, Any]:
        available_mask = _available_mask(available_mask_at_decision, self.n_assets)
        raw = _weight_vector(weights, self.n_assets)
        normalized = np.zeros(self.n_assets, dtype=np.float32)
        if not available_mask.any():
            return {"weights": normalized, "valid": False, "reason": "no_available_asset"}
        masked = np.where(available_mask, raw, 0.0).astype(np.float64, copy=False)
        if not np.isfinite(masked).all():
            return {"weights": normalized, "valid": False, "reason": "non_finite_weights"}
        if (masked[available_mask] < 0.0).any():
            return {"weights": normalized, "valid": False, "reason": "negative_weights"}
        weight_sum = float(masked[available_mask].sum())
        if weight_sum <= 0.0:
            return {"weights": normalized, "valid": False, "reason": "non_positive_available_weight_sum"}
        normalized[available_mask] = (masked[available_mask] / weight_sum).astype(np.float32)
        return {"weights": normalized, "valid": True, "reason": None}

    def _resolve_hierarchy_action(
        self,
        hierarchy_action: int,
        candidate_weights: Any,
        current_weights: Any,
        available_mask_at_decision: Any,
    ) -> dict[str, Any]:
        action = int(hierarchy_action)
        if action not in PPO_DQN_HIERARCHY_ACTION_NAMES:
            raise ValueError("ERR_PPO_DQN_HIERARCHY_ACTION_INVALID: hierarchy_action must be 0..4")

        candidate = self._mask_normalize_weights(candidate_weights, available_mask_at_decision)
        current = self._mask_normalize_weights(current_weights, available_mask_at_decision)
        available_mask = _available_mask(available_mask_at_decision, self.n_assets)
        if not available_mask.any():
            return {
                "status": "failed_no_valid_action",
                "reason": "no_available_asset",
                "target_weights": np.zeros(self.n_assets, dtype=np.float32),
                "action_info": {
                    "hierarchy_action": action,
                    "hierarchy_action_name": PPO_DQN_HIERARCHY_ACTION_NAMES[action],
                    "candidate_weights_valid": False,
                    "candidate_invalid_reason": "no_available_asset",
                    "ppo_actor_update_mask": 0,
                    "ppo_attribution_weight": 0.0,
                    "platform_adapted_surrogate": False,
                    "reason": "no_available_asset",
                },
            }
        if action != 4 and not bool(candidate["valid"]):
            return {
                "status": "failed_no_valid_action",
                "reason": "invalid_candidate_weights",
                "target_weights": np.zeros(self.n_assets, dtype=np.float32),
                "action_info": {
                    "hierarchy_action": action,
                    "hierarchy_action_name": PPO_DQN_HIERARCHY_ACTION_NAMES[action],
                    "candidate_weights_valid": False,
                    "candidate_invalid_reason": candidate["reason"],
                    "ppo_actor_update_mask": 0,
                    "ppo_attribution_weight": 0.0,
                    "platform_adapted_surrogate": False,
                    "reason": "invalid_candidate_weights",
                },
            }
        if action in PPO_DQN_HIERARCHY_BLEND_RHOS and not bool(current["valid"]):
            raise ValueError("ERR_PPO_DQN_CURRENT_WEIGHTS_INVALID")

        if action == 0:
            target_weights = np.asarray(candidate["weights"], dtype=np.float32)
            ppo_actor_update_mask = 1
            ppo_attribution_weight = 1.0
            platform_adapted_surrogate = False
        elif action == 4:
            target_weights = _equal_available_weights(available_mask)
            ppo_actor_update_mask = 0
            ppo_attribution_weight = 0.0
            platform_adapted_surrogate = False
        else:
            rho = PPO_DQN_HIERARCHY_BLEND_RHOS[action]
            target_weights = (1.0 - rho) * np.asarray(current["weights"], dtype=np.float32)
            target_weights = target_weights + rho * np.asarray(candidate["weights"], dtype=np.float32)
            target_weights = np.where(available_mask, target_weights, 0.0).astype(np.float32)
            ppo_actor_update_mask = 1
            ppo_attribution_weight = float(rho)
            platform_adapted_surrogate = True

        action_info = {
            "hierarchy_action": action,
            "hierarchy_action_name": PPO_DQN_HIERARCHY_ACTION_NAMES[action],
            "candidate_weights_valid": bool(candidate["valid"]),
            "candidate_invalid_reason": candidate["reason"],
            "ppo_actor_update_mask": int(ppo_actor_update_mask),
            "ppo_attribution_weight": float(ppo_attribution_weight),
            "platform_adapted_surrogate": bool(platform_adapted_surrogate),
        }
        if action == 4 and not bool(candidate["valid"]):
            action_info["fallback_reason"] = "invalid_candidate_weights_equal_weight"
        return {"target_weights": target_weights, "action_info": action_info}

    def _train_epoch(
        self,
        env: PortfolioRebalanceEnv,
        *,
        max_steps: int | None = None,
        max_gradient_updates: int | None = None,
    ) -> dict[str, Any]:
        observation, _ = env.reset()
        terminated = False
        truncated = False
        reward_total = 0.0
        env_steps = 0
        update_stats: list[dict[str, float]] = []
        ppo_losses: list[float] = []
        platform_adapted_surrogate = False

        while not (terminated or truncated):
            if max_steps is not None and env_steps >= int(max_steps):
                break
            decision = self._select_training_action(observation, deterministic=False)
            resolved = decision["resolved"]
            if resolved.get("status") == "failed_no_valid_action":
                return {
                    "status": "failed_no_valid_action",
                    "reason": resolved.get("reason", "failed_no_valid_action"),
                    "env_steps": env_steps,
                    "gradient_updates": len(update_stats),
                    "train_reward": reward_total,
                    "loss": float(np.mean(ppo_losses)) if ppo_losses else np.nan,
                    "platform_adapted_surrogate": bool(platform_adapted_surrogate),
                }
            target_weights = np.asarray(resolved["target_weights"], dtype=np.float32)
            action_info = dict(resolved["action_info"])
            platform_adapted_surrogate = platform_adapted_surrogate or bool(
                action_info.get("platform_adapted_surrogate", False)
            )
            action_info.update(
                {
                    "paper_model_id": self.strategy_name,
                    "baseline_family": "native_rl_reimplementation",
                    "estimated_turnover": _turnover(target_weights, observation["current_weights"]),
                    "estimated_cost": 0.0,
                }
            )
            next_observation, reward, terminated, truncated, info = env.step(
                {
                    "weights": target_weights,
                    "rebalance": 1,
                    "rebalance_intensity": 1.0,
                    **action_info,
                }
            )
            replay_info = {**info, **_next_replay_timing(env, terminal=bool(terminated or truncated))}
            ppo_update = self._update_ppo(decision, float(reward), action_info)
            ppo_losses.append(float(ppo_update["loss"]))
            self.replay_buffer.add_transition(
                _replay_item(
                    observation,
                    next_observation,
                    candidate_weights=np.asarray(decision["candidate_weights"], dtype=np.float32),
                    target_weights=target_weights,
                    hierarchy_action=int(action_info["hierarchy_action"]),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=replay_info,
                    q_values=np.asarray(decision["q_values"], dtype=np.float32),
                )
            )
            self._real_transition_count += 1
            can_update = (
                self._real_transition_count >= int(self.dqn_config.warmup_steps)
                and len(self.replay_buffer) >= int(self.dqn_config.batch_size)
                and (max_gradient_updates is None or len(update_stats) < int(max_gradient_updates))
            )
            if can_update:
                update_stats.append(self.dqn_agent.update())
            observation = next_observation
            reward_total += float(reward)
            env_steps += 1

        losses = [item.get("loss", np.nan) for item in update_stats]
        losses.extend(ppo_losses)
        return {
            "status": "completed",
            "env_steps": env_steps,
            "gradient_updates": len(update_stats),
            "train_reward": reward_total,
            "loss": float(np.nanmean(losses)) if losses else np.nan,
            "platform_adapted_surrogate": bool(platform_adapted_surrogate),
        }

    def _select_training_action(
        self,
        observation: Mapping[str, Any],
        *,
        deterministic: bool,
    ) -> dict[str, Any]:
        assert_decision_visibility_contract(observation=observation)
        market_image = torch.as_tensor(
            np.asarray(observation["market_image"], dtype=np.float32)[None, ...],
            dtype=torch.float32,
            device=self.device,
        )
        available_mask = torch.as_tensor(
            np.asarray(observation["availability_mask"], dtype=bool)[None, :],
            dtype=torch.bool,
            device=self.device,
        )
        current_weights = torch.as_tensor(
            np.asarray(observation["current_weights"], dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        latent = self.encoder(market_image)
        distribution = self.ppo_actor.get_distribution(latent, available_mask)
        candidate_weights = distribution.mean if deterministic else distribution.sample()
        candidate_np = candidate_weights.detach().cpu().numpy()[0]
        current_np = current_weights.detach().cpu().numpy()[0]
        available_np = available_mask.detach().cpu().numpy()[0]
        candidate_status = self._mask_normalize_weights(candidate_np, available_np)
        current_status = self._mask_normalize_weights(current_np, available_np)
        candidate_for_hierarchy = (
            np.asarray(candidate_status["weights"], dtype=np.float32)
            if bool(candidate_status["valid"])
            else _equal_available_weights(available_np)
        )
        current_for_hierarchy = (
            np.asarray(current_status["weights"], dtype=np.float32)
            if bool(current_status["valid"])
            else _equal_available_weights(available_np)
        )
        candidate_hierarchy_tensor = torch.as_tensor(
            candidate_for_hierarchy[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        current_hierarchy_tensor = torch.as_tensor(
            current_for_hierarchy[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        estimated_turnover = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        estimated_cost = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        q_values = self.hierarchy_q_network(
            latent,
            candidate_hierarchy_tensor,
            current_hierarchy_tensor,
            estimated_turnover,
            estimated_cost,
        )
        if deterministic:
            hierarchy_action = int(torch.argmax(q_values, dim=1).detach().cpu().item())
        else:
            hierarchy_action = int(self.dqn_agent.select_action(q_values.detach(), self.dqn_agent.global_step).view(-1)[0].cpu())
        candidate_log_prob = (
            distribution.log_prob(candidate_weights)
            if bool(candidate_status["valid"])
            else torch.zeros(1, dtype=torch.float32, device=self.device)
        )
        current_for_resolve = (
            current_np
            if bool(current_status["valid"])
            else _equal_available_weights(available_np)
        )
        return {
            "latent": latent,
            "candidate_log_prob": candidate_log_prob,
            "value": self.ppo_critic(latent),
            "candidate_weights": candidate_np,
            "current_weights": current_np,
            "q_values": q_values.detach().cpu().numpy()[0],
            "resolved": self._resolve_hierarchy_action(hierarchy_action, candidate_np, current_for_resolve, available_np),
        }

    def _update_ppo(
        self,
        decision: Mapping[str, Any],
        reward: float,
        action_info: Mapping[str, Any],
    ) -> dict[str, float]:
        reward_tensor = torch.as_tensor([[float(reward)]], dtype=torch.float32, device=self.device)
        attribution = float(action_info.get("ppo_attribution_weight", 0.0) or 0.0)
        actor_mask = int(action_info.get("ppo_actor_update_mask", 0) or 0)
        if actor_mask == 0 or attribution == 0.0:
            actor_loss = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        else:
            actor_loss = -decision["candidate_log_prob"].view(1, 1) * reward_tensor * attribution
        value_loss = F.mse_loss(decision["value"], reward_tensor)
        loss = actor_loss.mean() + value_loss
        if not torch.isfinite(loss):
            raise DataContractError("ERR_TRAINING_NON_FINITE_LOSS", "ERR_TRAINING_NON_FINITE_LOSS: ppo_dqn")
        self.ppo_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        max_grad_norm = _max_grad_norm(self.config)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(_unique_parameters(self.encoder, self.ppo_actor, self.ppo_critic), max_grad_norm)
        self.ppo_optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "actor_loss": float(actor_loss.mean().detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
        }

    def _policy_action(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        observation = _observation_from_state(self, decision_market_state, portfolio_state)
        with torch.no_grad():
            decision = self._select_training_action(observation, deterministic=True)
        resolved = decision["resolved"]
        if resolved.get("status") == "failed_no_valid_action":
            raise DataContractError(
                "ERR_STRATEGY_ACTION_CONTRACT",
                f"ERR_STRATEGY_ACTION_CONTRACT: {resolved.get('reason', 'failed_no_valid_action')}",
            )
        target_weights = np.asarray(resolved["target_weights"], dtype=np.float32)
        action_info = dict(resolved["action_info"])
        action_info.update(
            {
                "paper_model_id": self.strategy_name,
                "baseline_family": "native_rl_reimplementation",
                "estimated_turnover": _turnover(target_weights, portfolio_state.current_weights),
                "estimated_cost": 0.0,
            }
        )
        return PortfolioAction(target_weights, 1, 1.0, action_info)

    def _checkpoint_paths(self) -> dict[str, Path | None]:
        checkpoint_dir = _mapping(self.config.get("baselines")).get("checkpoint_dir")
        if checkpoint_dir is None:
            checkpoint_dir = self.config.get("baseline_run_dir")
        if checkpoint_dir is None:
            return {"best": None, "last": None}
        root = Path(checkpoint_dir) / "checkpoints" / self.strategy_name
        return {"best": root / "best.pt", "last": root / "last.pt"}

    def _save_checkpoint_state(self, path: Path | None, epoch: int, global_step: int, best_metric: float | None) -> None:
        if path is None:
            return
        save_checkpoint(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "policy_model_state": None,
                "target_policy_model_state": None,
                "encoder_state": self.encoder.state_dict(),
                "ppo_actor_state": self.ppo_actor.state_dict(),
                "ppo_critic_state": self.ppo_critic.state_dict(),
                "dqn_gate_state": self.hierarchy_q_network.state_dict(),
                "dqn_target_network_state": self.target_hierarchy_q_network.state_dict(),
                "auxiliary_head_states": None,
                "optimizer_states": {
                    "ppo": self.ppo_optimizer.state_dict(),
                    "dqn": self.dqn_optimizer.state_dict(),
                    "auxiliary": None,
                },
                "scheduler_states": {},
                "amp_grad_scaler_state": None,
                "replay_buffer_state": None,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_validation_metric": best_metric,
                "rng_states": {},
                "resolved_config": dict(self.config),
            },
            path,
        )

    def _load_checkpoint_state(self, path: Path) -> None:
        payload = load_checkpoint(path, device=self.device, restore_rng_state=False)
        self.encoder.load_state_dict(payload["encoder_state"])
        self.ppo_actor.load_state_dict(payload["ppo_actor_state"])
        self.ppo_critic.load_state_dict(payload["ppo_critic_state"])
        self.hierarchy_q_network.load_state_dict(payload["dqn_gate_state"])
        self.target_hierarchy_q_network.load_state_dict(payload["dqn_target_network_state"])
        optimizer_states = payload["optimizer_states"]
        if optimizer_states.get("ppo") is not None:
            self.ppo_optimizer.load_state_dict(optimizer_states["ppo"])
        if optimizer_states.get("dqn") is not None:
            self.dqn_optimizer.load_state_dict(optimizer_states["dqn"])
        self.dqn_agent.global_step = int(payload["global_step"])
        self._checkpoint_loaded_path = str(path)


class _PPODQNValidationStrategy(BaseStrategy):
    strategy_name = PPO_DQN_HIERARCHICAL_REIMPLEMENTATION
    fit_required = False

    def __init__(self, strategy: PPODQNHierarchicalReimplementationStrategy) -> None:
        super().__init__(strategy.config)
        self.strategy = strategy

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        portfolio = self.validate_portfolio_state(portfolio_state)
        return self.validate_portfolio_action(self.strategy._policy_action(state, portfolio))


def _training_result(
    model_name: str,
    training_history: pd.DataFrame,
    *,
    status: str = "not_started",
    reason: str | None = "training_logic_not_implemented",
    checkpoint_best_path: str | None = None,
    checkpoint_last_path: str | None = None,
    evaluated_checkpoint_path: str | None = None,
    best_validation_metric: float | None = None,
    env_steps: int = 0,
    gradient_updates: int = 0,
    rankable_in_unified_table: bool = False,
    max_train_steps: int | None = None,
    max_validation_steps: int | None = None,
    max_gradient_updates_per_epoch: int | None = None,
    dqn_role: str = PPO_DQN_HIGH_LEVEL_ACTION_SELECTOR,
    platform_adapted_surrogate: bool = False,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "paper_model_id": model_name,
        "child_model_name": model_name,
        "baseline_family": "native_rl_reimplementation",
        "status": status,
        "reason": reason,
        "training_algorithm": model_name,
        "rl_training": True,
        "platform_native_rl_training": True,
        "proxy_training": False,
        "external_original_implementation": False,
        "clean_room_reimplementation": True,
        "algorithm_fidelity": "platform_adapted",
        "dqn_role": str(dqn_role),
        "platform_adapted_surrogate": bool(platform_adapted_surrogate),
        "rankable_in_unified_table": bool(rankable_in_unified_table),
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


def _resolved_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    model_config = _mapping(config.get("model"))
    feature_matrix = _mapping(config.get("feature_matrix"))
    env_config = _mapping(config.get("env"))
    result["n_assets"] = _config_positive_int(config, model_config, "n_assets", default=1)
    result["n_features"] = _config_positive_int(config, model_config, "n_features", default=1)
    result["window_size"] = _positive_int(
        "window_size",
        config.get(
            "window_size",
            feature_matrix.get("window_size", env_config.get("window_size", model_config.get("window_size", 1))),
        ),
    )
    result["latent_dim"] = _positive_int("latent_dim", config.get("latent_dim", model_config.get("latent_dim", 256)))
    return result


def _config_positive_int(
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    return _positive_int(key, config.get(key, model_config.get(key, default)))


def _ppo_lr(config: Mapping[str, Any]) -> float:
    optimizer_config = _mapping(config.get("optimizer"))
    ppo_config = _mapping(config.get("ppo"))
    return _positive_float(
        "ppo_lr",
        _first_present(optimizer_config, "ppo_lr", "learning_rate", default=ppo_config.get("lr", 3.0e-4)),
    )


def _dqn_parameters(
    hierarchy_q_network: nn.Module,
    encoder: nn.Module,
    dqn_config: DQNAgentConfig,
) -> list[nn.Parameter]:
    modules = (hierarchy_q_network,) if dqn_config.detach_encoder else (hierarchy_q_network, encoder)
    return _unique_parameters(*modules)


def _unique_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            parameters.append(parameter)
            seen.add(id(parameter))
    return parameters


def _hidden_dims(value: Any, *, default: Sequence[int]) -> tuple[int, ...]:
    if value is None:
        return tuple(int(dim) for dim in default)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("ERR_PPO_DQN_CONFIG_INVALID: hidden_dims must be a sequence")
    dims = tuple(_positive_int("hidden_dims", dim) for dim in value)
    if not dims:
        raise ValueError("ERR_PPO_DQN_CONFIG_INVALID: hidden_dims must be non-empty")
    return dims


def _feature_tensor(
    value: torch.Tensor | None,
    batch_size: int,
    width: int,
    latent: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, width, device=latent.device, dtype=latent.dtype)
    tensor = value.to(device=latent.device, dtype=latent.dtype)
    if tensor.ndim == 1 and width == 1:
        tensor = tensor.view(-1, 1)
    if tensor.shape != (batch_size, width):
        raise ValueError("ERR_PPO_DQN_HIERARCHY_SHAPE: hierarchy q features must match [batch,width]")
    return tensor


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _configured_dqn_role(config: Mapping[str, Any]) -> str:
    native_rl = _native_rl_config(config)
    model_config = _mapping(config.get("model"))
    for mapping in (config, native_rl, model_config):
        value = mapping.get("dqn_role")
        if value is not None:
            role = str(value).strip()
            if role:
                return role
    return PPO_DQN_HIGH_LEVEL_ACTION_SELECTOR


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _positive_int(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"ERR_PPO_DQN_CONFIG_INVALID: {name}")
    return result


def _positive_float(name: str, value: Any) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"ERR_PPO_DQN_CONFIG_INVALID: {name}")
    return result


def _all_finite(*values: torch.Tensor) -> bool:
    return all(torch.isfinite(value).all().item() for value in values)


def _available_mask(value: Any, n_assets: int) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape != (int(n_assets),):
        raise ValueError("ERR_PPO_DQN_WEIGHT_SHAPE: available_mask_at_decision must be [n_assets]")
    return mask


def _weight_vector(value: Any, n_assets: int) -> np.ndarray:
    weights = np.asarray(value, dtype=np.float64)
    if weights.shape != (int(n_assets),):
        raise ValueError("ERR_PPO_DQN_WEIGHT_SHAPE: weights must be [n_assets]")
    return weights


def _equal_available_weights(available_mask: np.ndarray) -> np.ndarray:
    weights = np.zeros(available_mask.shape, dtype=np.float32)
    if available_mask.any():
        weights[available_mask] = 1.0 / float(available_mask.sum())
    return weights


def _replay_item(
    observation: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    *,
    candidate_weights: np.ndarray,
    target_weights: np.ndarray,
    hierarchy_action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    q_values: np.ndarray,
) -> ReplayItem:
    decision_date = pd.Timestamp(info.get("decision_date"))
    execution_date = pd.Timestamp(info.get("execution_date", info.get("pending_execution_date", decision_date)))
    next_valuation_date = pd.Timestamp(
        info.get("next_valuation_date", info.get("pending_next_valuation_date", execution_date))
    )
    decision_date_next = pd.Timestamp(info.get("decision_date_next", info.get("next_decision_date", next_valuation_date)))
    execution_date_next = pd.Timestamp(
        info.get("execution_date_next", info.get("next_execution_date", decision_date_next))
    )
    next_valuation_date_next = pd.Timestamp(
        info.get("next_valuation_date_next", info.get("next_next_valuation_date", execution_date_next))
    )
    selected_q = float(q_values[int(hierarchy_action)])
    reference_q = float(q_values[0])
    return ReplayItem(
        state_t=dict(observation),
        state_tp1=dict(next_observation),
        decision_date_t=decision_date,
        execution_date_t=execution_date,
        next_valuation_date_t=next_valuation_date,
        decision_date_next=decision_date_next,
        execution_date_next=execution_date_next,
        next_valuation_date_next=next_valuation_date_next,
        execution_price_t=str(info.get("execution_price_type", "open")),
        delayed_action_execution_t=bool(info.get("delayed_action_execution", False)),
        candidate_weights_t=np.asarray(candidate_weights, dtype=np.float32),
        executed_weights_t=np.asarray(info.get("executed_weights", target_weights), dtype=np.float32),
        gate_action_t=int(hierarchy_action),
        rebalance_action_t=int(info.get("rebalance_action", 1)),
        rebalance_intensity_t=float(info.get("rebalance_intensity", 1.0)),
        estimated_turnover_t=float(info.get("estimated_turnover", _turnover(target_weights, observation["current_weights"]))),
        realized_turnover_t=float(info.get("realized_turnover", info.get("turnover", 0.0))),
        estimated_cost_t=float(info.get("estimated_cost", 0.0) or 0.0),
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
        invalid_action_t=False,
        bootstrap_mask_t=1.0,
        next_state_source_t="env",
        execution_price_next=str(info.get("execution_price_next", info.get("execution_price_type", "open"))),
        delayed_action_execution_next=bool(
            info.get("delayed_action_execution_next", info.get("delayed_action_execution", False))
        ),
        split_boundary_t=bool(terminated or truncated),
    )


def _next_replay_timing(env: PortfolioRebalanceEnv, *, terminal: bool) -> dict[str, Any]:
    if terminal:
        return {}
    step_pos = int(getattr(env, "_step_pos", -1))
    decision_dates = getattr(env, "decision_dates", ())
    if step_pos < 0 or step_pos >= len(decision_dates):
        return {}
    decision_date = pd.Timestamp(decision_dates[step_pos])
    try:
        execution_state = env.execution_core.build_execution_market_state(env.dataset, decision_date)
    except Exception:
        return {
            "decision_date_next": decision_date,
            "execution_date_next": decision_date,
            "next_valuation_date_next": decision_date,
        }
    return {
        "decision_date_next": execution_state.decision_date,
        "execution_date_next": execution_state.execution_date,
        "next_valuation_date_next": execution_state.next_valuation_date,
        "execution_price_next": execution_state.execution_price_type,
        "delayed_action_execution_next": bool(env.execution_config.get("delayed_action_execution", False)),
    }


def _validation_metric_from_backtest(
    strategy: PPODQNHierarchicalReimplementationStrategy,
    dataset: Any,
    split: SplitSpec,
    *,
    market_image_dataset: Any | None,
    max_steps: int | None,
) -> float:
    validation_split = _validation_split(split, max_steps=max_steps)
    try:
        result = BacktestEngine(strategy.config, market_image_dataset=market_image_dataset).run(
            dataset,
            validation_split,
            _PPODQNValidationStrategy(strategy),
            segment="validation",
        )
        from src.experiments.pipeline import objective_metric

        value = objective_metric(
            {"daily_returns": result.daily_returns, "metrics": result.metrics},
            "validation_sharpe_minus_drawdown_turnover_penalty",
        )
    except (DataContractError, ValueError, KeyError):
        return float("-inf")
    return float(value) if np.isfinite(value) else float("-inf")


def _validation_split(split: SplitSpec, *, max_steps: int | None) -> SplitSpec:
    validation_dates = pd.DatetimeIndex(split.validation_dates)
    validation_last_decision_date = split.validation_last_decision_date
    if max_steps is not None and int(max_steps) > 0:
        limit = int(max_steps) + 1 if len(validation_dates) > 1 else int(max_steps)
        validation_dates = validation_dates[:limit]
        if len(validation_dates) > 0:
            decision_pos = min(int(max_steps) - 1, len(validation_dates) - 1)
            validation_last_decision_date = pd.Timestamp(validation_dates[decision_pos])
    return SplitSpec(
        train_dates=split.train_dates,
        validation_dates=validation_dates,
        test_dates=split.test_dates,
        fold_id=split.fold_id,
        train_last_decision_date=split.train_last_decision_date,
        validation_last_decision_date=validation_last_decision_date,
        test_last_decision_date=split.test_last_decision_date,
    )


def _observation_from_state(
    strategy: PPODQNHierarchicalReimplementationStrategy,
    decision_market_state: DecisionMarketState,
    portfolio_state: PortfolioState,
) -> dict[str, np.ndarray]:
    dtype = np.float32
    return {
        "market_image": _pad_window(np.asarray(decision_market_state.market_image, dtype=dtype), strategy.window_size),
        "current_weights": np.asarray(portfolio_state.current_weights, dtype=dtype),
        "availability_mask": np.asarray(decision_market_state.available_mask_at_decision, dtype=np.int8),
        "adv20_at_decision": _finite_array(decision_market_state.adv20_at_decision, dtype),
        "volatility_20d_at_decision": _finite_array(decision_market_state.volatility_20d_at_decision, dtype),
        "amount_at_decision": _finite_array(decision_market_state.amount_at_decision, dtype),
        "turnover_rate_at_decision": _finite_array(decision_market_state.turnover_rate_at_decision, dtype),
        "portfolio_value": np.asarray(portfolio_state.portfolio_value, dtype=dtype),
    }


def _finite_array(values: Any, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array.astype(dtype, copy=False)


def _pad_window(image: np.ndarray, window_size: int) -> np.ndarray:
    if image.ndim < 2:
        raise DataContractError("ERR_STATE_SCHEMA_MISMATCH", "ERR_STATE_SCHEMA_MISMATCH: market_image")
    current = image.shape[-2]
    if current == int(window_size):
        return image
    if current > int(window_size):
        slicer = [slice(None)] * image.ndim
        slicer[-2] = slice(current - int(window_size), current)
        return image[tuple(slicer)]
    pad_width = [(0, 0)] * image.ndim
    pad_width[-2] = (int(window_size) - current, 0)
    return np.pad(image, pad_width, mode="constant", constant_values=0.0)


def _path_string(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _has_finite_validation(history: pd.DataFrame) -> bool:
    if history.empty or "validation_metric" not in history.columns:
        return False
    values = pd.to_numeric(history["validation_metric"], errors="coerce")
    return bool(np.isfinite(values).any())


def _turnover(weights: Any, current_weights: Any) -> float:
    return float(0.5 * np.sum(np.abs(np.asarray(weights, dtype=np.float32) - np.asarray(current_weights, dtype=np.float32))))


def _dates(value: Any) -> pd.DatetimeIndex:
    if value is None:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(list(value))).sort_values()


def _native_rl_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    baselines = _mapping(config.get("baselines"))
    return _mapping(baselines.get("native_rl") or baselines.get("native_training"))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("ERR_PPO_DQN_CONFIG_INVALID: max step limits must be > 0")
    return result


def _max_grad_norm(config: Mapping[str, Any]) -> float | None:
    optimizer = _mapping(config.get("optimizer"))
    training = _mapping(config.get("training"))
    value = optimizer.get("max_grad_norm", training.get("max_grad_norm"))
    if value is None:
        return None
    result = float(value)
    if result <= 0.0:
        return None
    return result


__all__ = [
    "HierarchyQNetwork",
    "PPO_DQN_HIERARCHY_ACTION_DIM",
    "PPO_DQN_HIERARCHY_ACTION_NAMES",
    "PPO_DQN_HIERARCHICAL_REIMPLEMENTATION",
    "PPODQNHierarchicalReimplementationStrategy",
]
