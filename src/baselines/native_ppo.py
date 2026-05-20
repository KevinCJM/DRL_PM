from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.agents.ppo_agent import PPOAgent
from src.baselines.base_strategy import BaseStrategy
from src.data.splits import SplitSpec
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.models.encoders import CNNEncoder, EncoderFactory
from src.models.ppo_actor import PPOActor
from src.models.ppo_critic import PPOCritic
from src.utils.checkpoint import load_checkpoint, save_checkpoint


NATIVE_PPO_ALGORITHM = "ppo_clipped_gae"


class NativePPOBaselineStrategy(BaseStrategy):
    strategy_name = "ppo_native"
    encoder_type = "mlp"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.fit_required = True
        self.training_result: dict[str, Any] | None = None
        self.training_history: pd.DataFrame = pd.DataFrame()
        self.device = _device(self.config)
        self.agent = self._build_agent()

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> NativePPOBaselineStrategy:
        if not isinstance(train_data, Mapping):
            self.training_result = _training_result(
                self.strategy_name,
                "failed_missing_train_data",
                training_history=pd.DataFrame(),
            )
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
        config = dict(self.config)
        dataset = train_data["dataset"]
        market_image_dataset = train_data.get("market_image_dataset")
        train_env = PortfolioRebalanceEnv(
            dataset,
            split,
            config=config,
            segment="train",
            market_image_dataset=market_image_dataset,
        )
        validation_env = PortfolioRebalanceEnv(
            dataset,
            split,
            config=config,
            segment="validation",
            market_image_dataset=market_image_dataset,
        )

        native_cfg = _native_rl_config(config)
        epochs = max(1, int(native_cfg.get("epochs", _mapping(config.get("training")).get("epochs", 1))))
        max_train_steps = _optional_positive_int(native_cfg.get("max_train_steps"))
        max_validation_steps = _optional_positive_int(native_cfg.get("max_validation_steps"))
        checkpoint_paths = self._checkpoint_paths()
        best_metric = -np.inf
        history_rows: list[dict[str, Any]] = []
        env_steps = 0
        gradient_updates = 0

        for epoch in range(epochs):
            rollout = self.agent.collect_rollout(train_env, max_steps=max_train_steps)
            env_steps += len(rollout)
            update_stats = self.agent.update(rollout)
            gradient_updates += int(self.agent.config.update_epochs)
            validation_metric = _evaluate_agent(self.agent, validation_env, max_steps=max_validation_steps)
            loss_value = update_stats.get("actor_loss", update_stats.get("value_loss", np.nan))
            row = {
                "epoch": int(epoch),
                "step": int(epoch + 1),
                "env_steps": int(env_steps),
                "gradient_updates": int(gradient_updates),
                "train_reward": float(np.mean([item.reward for item in rollout.items])) if len(rollout) else np.nan,
                "validation_metric": float(validation_metric),
                "loss": None if loss_value is None else float(loss_value),
                "max_train_steps": max_train_steps,
                "max_validation_steps": max_validation_steps,
                "status": "completed",
            }
            history_rows.append(row)
            if np.isfinite(validation_metric) and validation_metric > best_metric:
                best_metric = float(validation_metric)
                if checkpoint_paths["best"] is not None:
                    save_checkpoint(
                        self.agent,
                        checkpoint_paths["best"],
                        epoch=epoch,
                        global_step=gradient_updates,
                        best_validation_metric=best_metric,
                        resolved_config=config,
                        env=train_env,
                        include_replay_buffer=_checkpoint_include_replay_buffer(config),
                    )

        if checkpoint_paths["last"] is not None:
            save_checkpoint(
                self.agent,
                checkpoint_paths["last"],
                epoch=epochs - 1,
                global_step=gradient_updates,
                best_validation_metric=None if not np.isfinite(best_metric) else best_metric,
                resolved_config=config,
                env=train_env,
                include_replay_buffer=_checkpoint_include_replay_buffer(config),
            )

        history = pd.DataFrame(history_rows)
        if not _has_finite_validation(history):
            self.training_history = history
            self.training_result = _training_result(
                self.strategy_name,
                "failed_no_finite_validation_metric",
                training_history=history,
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
                training_history=history,
                checkpoint_best_path=_path_string(checkpoint_paths["best"]),
                checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            )
            self.is_fitted = False
            return self

        load_checkpoint(checkpoint_paths["best"], device=self.device, agent=self.agent, env=train_env, restore_rng_state=False)
        self.training_history = history
        self.training_result = _training_result(
            self.strategy_name,
            "completed",
            training_history=history,
            checkpoint_best_path=_path_string(checkpoint_paths["best"]),
            checkpoint_last_path=_path_string(checkpoint_paths["last"]),
            evaluated_checkpoint_path=_path_string(checkpoint_paths["best"]),
            best_validation_metric=float(best_metric),
            env_steps=env_steps,
            gradient_updates=gradient_updates,
            max_train_steps=max_train_steps,
            max_validation_steps=max_validation_steps,
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
        action_info = self.agent.select_action(observation, deterministic=True)
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=np.asarray(action_info["candidate_weights"], dtype=float),
                rebalance_action=1,
                rebalance_intensity=1.0,
                action_info={
                    "strategy": self.strategy_name,
                    "training_algorithm": NATIVE_PPO_ALGORITHM,
                    "rl_training": True,
                    "platform_native_rl_training": True,
                    "estimated_turnover": action_info.get("estimated_turnover"),
                    "estimated_cost": action_info.get("estimated_cost"),
                },
            )
        )

    def _build_agent(self) -> PPOAgent:
        model_config = dict(self.config)
        encoder_config = dict(_mapping(model_config.get("encoder")))
        encoder_config["type"] = self.encoder_type
        model_config["encoder"] = encoder_config
        encoder = EncoderFactory.create(model_config)
        latent_dim = int(model_config.get("latent_dim", _mapping(model_config.get("model")).get("latent_dim", 256)))
        actor = PPOActor(latent_dim=latent_dim, n_assets=int(model_config["n_assets"]))
        critic = PPOCritic(latent_dim=latent_dim)
        return PPOAgent(
            encoder,
            actor,
            critic,
            config=model_config,
            device=self.device,
        )

    def _checkpoint_paths(self) -> dict[str, Path | None]:
        checkpoint_dir = _mapping(self.config.get("baselines")).get("checkpoint_dir")
        if checkpoint_dir is None:
            checkpoint_dir = self.config.get("baseline_run_dir")
        if checkpoint_dir is None:
            return {"best": None, "last": None}
        root = Path(checkpoint_dir) / "checkpoints" / self.strategy_name
        return {"best": root / "best.pt", "last": root / "last.pt"}


class NativeCNNPPOBaselineStrategy(NativePPOBaselineStrategy):
    strategy_name = "cnn_ppo_native"
    encoder_type = "cnn"

    def _build_agent(self) -> PPOAgent:
        agent = super()._build_agent()
        if not isinstance(agent.encoder, CNNEncoder):
            raise ValueError("ERR_NATIVE_PPO_ENCODER: cnn_ppo_native requires CNNEncoder")
        return agent


def _evaluate_agent(agent: PPOAgent, env: PortfolioRebalanceEnv, max_steps: int | None = None) -> float:
    observation, _ = env.reset()
    terminated = False
    truncated = False
    rewards: list[float] = []
    while not (terminated or truncated):
        if max_steps is not None and len(rewards) >= int(max_steps):
            break
        action_info = agent.select_action(observation, deterministic=True)
        action = agent.action_for_env(observation, action_info)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
    if not rewards:
        return float("-inf")
    return float(np.sum(rewards))


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


def _training_result(
    model_name: str,
    status: str,
    *,
    training_history: pd.DataFrame,
    checkpoint_best_path: str | None = None,
    checkpoint_last_path: str | None = None,
    evaluated_checkpoint_path: str | None = None,
    best_validation_metric: float | None = None,
    env_steps: int = 0,
    gradient_updates: int = 0,
    max_train_steps: int | None = None,
    max_validation_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "baseline_family": "native_rl",
        "status": status,
        "training_algorithm": NATIVE_PPO_ALGORITHM,
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
    }


def _has_finite_validation(history: pd.DataFrame) -> bool:
    if history.empty or "validation_metric" not in history.columns:
        return False
    values = pd.to_numeric(history["validation_metric"], errors="coerce")
    return bool(np.isfinite(values).any())


def _native_rl_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    baselines = _mapping(config.get("baselines"))
    return _mapping(baselines.get("native_rl") or baselines.get("native_training"))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("ERR_NATIVE_PPO_CONFIG_INVALID: max step limits must be > 0")
    return result


def _checkpoint_include_replay_buffer(config: Mapping[str, Any]) -> bool:
    training = config.get("training")
    if isinstance(training, Mapping) and "checkpoint_include_replay_buffer" in training:
        return bool(training.get("checkpoint_include_replay_buffer"))
    checkpoint = config.get("checkpoint")
    if isinstance(checkpoint, Mapping) and "include_replay_buffer" in checkpoint:
        return bool(checkpoint.get("include_replay_buffer"))
    return True


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
    "NativeCNNPPOBaselineStrategy",
    "NativePPOBaselineStrategy",
]
