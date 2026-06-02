from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from src.baselines.base_strategy import BaseStrategy
from src.baselines.eiie import _rebalance_turnover_threshold
from src.data.splits import SplitSpec
from src.envs.portfolio_rebalance_env import PortfolioRebalanceEnv
from src.envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from src.models.encoders import CNNEncoder, EncoderFactory
from src.models.ppo_actor import PPOActor
from src.models.ppo_critic import PPOCritic


BERNOULLI_GATED_ALGORITHM = "bernoulli_gated_ppo_on_policy"


class BernoulliGate(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError("ERR_BERNOULLI_GATE_SHAPE: latent must be [batch,latent_dim]")
        return torch.sigmoid(self.net(latent)).clamp(1.0e-6, 1.0 - 1.0e-6)


class NativeBernoulliGatedPPOBaselineStrategy(BaseStrategy):
    strategy_name = "bernoulli_gated_ppo_native"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.fit_required = True
        self.training_result: dict[str, Any] | None = None
        self.training_history: pd.DataFrame = pd.DataFrame()
        self.device = _device(self.config)
        self.encoder, self.actor, self.critic, self.gate, self.optimizer = self._build_modules()

    def fit(
        self,
        train_data: Any | None = None,
        validation_data: Any | None = None,
    ) -> NativeBernoulliGatedPPOBaselineStrategy:
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
        max_train_steps = _optional_positive_int(native_cfg.get("max_train_steps"))
        max_validation_steps = _optional_positive_int(native_cfg.get("max_validation_steps"))
        checkpoint_paths = self._checkpoint_paths()
        history_rows: list[dict[str, Any]] = []
        best_metric = -np.inf
        env_steps = 0
        gradient_updates = 0

        for epoch in range(epochs):
            rollout = self._collect_rollout(train_env, deterministic=False, max_steps=max_train_steps)
            stats = self._update(rollout)
            env_steps += len(rollout["reward"])
            gradient_updates += 1
            validation_metric = self._evaluate(validation_env, max_steps=max_validation_steps)
            history_rows.append(
                {
                    "epoch": int(epoch),
                    "step": int(epoch + 1),
                    "env_steps": int(env_steps),
                    "gradient_updates": int(gradient_updates),
                    "train_reward": float(np.mean(rollout["reward"])) if rollout["reward"] else np.nan,
                    "validation_metric": float(validation_metric),
                    "loss": float(stats["loss"]),
                    "gate_loss": float(stats["gate_loss"]),
                    "gate_entropy": float(stats["gate_entropy"]),
                    "gate_grad_norm": float(stats["gate_grad_norm"]),
                    "p_rebalance_mean": float(np.mean(rollout["p_rebalance"])) if rollout["p_rebalance"] else np.nan,
                    "raw_gate_frequency": float(np.mean(rollout["gate_action"])) if rollout["gate_action"] else np.nan,
                    "rebalance_frequency": float(np.mean(rollout["executed_gate_action"]))
                    if rollout["executed_gate_action"]
                    else np.nan,
                    "max_train_steps": max_train_steps,
                    "max_validation_steps": max_validation_steps,
                    "status": "completed",
                }
            )
            if np.isfinite(validation_metric) and validation_metric > best_metric:
                best_metric = float(validation_metric)
                self._save_state(checkpoint_paths["best"], epoch, gradient_updates, best_metric)

        self._save_state(
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

        self._load_state(checkpoint_paths["best"])
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
            gate_entropy=float(pd.to_numeric(history["gate_entropy"], errors="coerce").mean()),
            p_rebalance_mean=float(pd.to_numeric(history["p_rebalance_mean"], errors="coerce").mean()),
            rebalance_frequency=float(pd.to_numeric(history["rebalance_frequency"], errors="coerce").mean()),
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
        action_info = self._select_action(observation, deterministic=True)
        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=np.asarray(action_info["weights"], dtype=float),
                rebalance_action=int(action_info["rebalance"]),
                rebalance_intensity=float(action_info["rebalance_intensity"]),
                action_info={
                    "strategy": self.strategy_name,
                    "training_algorithm": BERNOULLI_GATED_ALGORITHM,
                    "rl_training": True,
                    "platform_native_rl_training": True,
                    "gate_training": "on_policy_bernoulli",
                    "p_rebalance": action_info["p_rebalance"],
                    "deterministic_gate_threshold": action_info["deterministic_gate_threshold"],
                    "gate_entropy": action_info["gate_entropy"],
                    "gate_action": action_info["gate_action"],
                    "raw_bernoulli_gate_action": action_info["raw_bernoulli_gate_action"],
                    "continuous_weight_rebalance_gate": action_info["continuous_weight_rebalance_gate"],
                    "rebalance_turnover_threshold": action_info["rebalance_turnover_threshold"],
                    "estimated_turnover": action_info["estimated_turnover"],
                    "candidate_turnover": action_info["candidate_turnover"],
                    "candidate_turnover_estimate": action_info["candidate_turnover_estimate"],
                    "estimated_cost": 0.0,
                    "raw_model_requested_rebalance": action_info["raw_model_requested_rebalance"],
                    "raw_action": action_info["raw_action"],
                    "raw_rho": action_info["raw_rho"],
                    "raw_rebalance_intensity": action_info["raw_rebalance_intensity"],
                    "rebalance_intensity": action_info["rebalance_intensity"],
                    "forced_hold_reason": action_info["forced_hold_reason"],
                },
            )
        )

    def _collect_rollout(
        self,
        env: PortfolioRebalanceEnv,
        *,
        deterministic: bool,
        max_steps: int | None = None,
    ) -> dict[str, list[Any]]:
        observation, _ = env.reset()
        terminated = False
        truncated = False
        rollout: dict[str, list[Any]] = {
            "state": [],
            "candidate_weights": [],
            "candidate_log_prob": [],
            "gate_action": [],
            "executed_gate_action": [],
            "gate_log_prob": [],
            "gate_entropy": [],
            "p_rebalance": [],
            "value": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
        }
        while not (terminated or truncated):
            if max_steps is not None and len(rollout["reward"]) >= int(max_steps):
                break
            action_info = self._select_action(observation, deterministic=deterministic)
            next_observation, reward, terminated, truncated, _ = env.step(action_info)
            rollout["state"].append(dict(observation))
            rollout["candidate_weights"].append(np.asarray(action_info["candidate_weights"], dtype=np.float32))
            rollout["candidate_log_prob"].append(float(action_info["candidate_log_prob"]))
            rollout["gate_action"].append(int(action_info["raw_bernoulli_gate_action"]))
            rollout["executed_gate_action"].append(int(action_info["gate_action"]))
            rollout["gate_log_prob"].append(float(action_info["gate_log_prob"]))
            rollout["gate_entropy"].append(float(action_info["gate_entropy"]))
            rollout["p_rebalance"].append(float(action_info["p_rebalance"]))
            rollout["value"].append(float(action_info["value"]))
            rollout["reward"].append(float(reward))
            rollout["terminated"].append(bool(terminated))
            rollout["truncated"].append(bool(truncated))
            observation = next_observation
        advantages, returns = _gae(
            rollout["reward"],
            rollout["value"],
            rollout["terminated"],
            rollout["truncated"],
            gamma=_ppo_float(self.config, "gamma", 0.99),
            gae_lambda=_ppo_float(self.config, "gae_lambda", 0.95),
        )
        rollout["advantage"] = [float(item) for item in advantages]
        rollout["return"] = [float(item) for item in returns]
        return rollout

    def _select_action(self, observation: Mapping[str, Any], *, deterministic: bool) -> dict[str, Any]:
        market_image, availability_mask, current_weights = self._observation_tensors([observation])
        latent = self.encoder(market_image)
        distribution = self.actor.get_distribution(latent, availability_mask)
        candidate_weights = distribution.mean if deterministic else distribution.sample()
        candidate_log_prob = distribution.log_prob(candidate_weights).view(-1, 1)
        value = self.critic(latent).view(-1, 1)
        p_rebalance = self.gate(latent).view(-1, 1)
        gate_dist = Bernoulli(probs=p_rebalance)
        if deterministic:
            gate_threshold = _deterministic_gate_threshold(self.config)
            gate_action = (p_rebalance >= gate_threshold).to(dtype=torch.float32)
        else:
            gate_threshold = 0.5
            gate_action = gate_dist.sample()
        gate_log_prob = gate_dist.log_prob(gate_action).view(-1, 1)
        gate_entropy = gate_dist.entropy().view(-1, 1)
        turnover = 0.5 * torch.sum(torch.abs(candidate_weights - current_weights), dim=1, keepdim=True)
        raw_gate_action = int(gate_action.view(-1)[0].detach().cpu())
        turnover_value = float(turnover.view(-1)[0].detach().cpu())
        threshold = _rebalance_turnover_threshold(self.config, self.strategy_name)
        first_trade = bool(float(current_weights.sum().detach().cpu()) <= 0.0)
        requested = bool(first_trade or (raw_gate_action and turnover_value > threshold + 1.0e-12))
        executed_weights = candidate_weights if requested else current_weights
        forced_hold_reason = None
        if not requested:
            forced_hold_reason = "model_chosen_hold" if raw_gate_action == 0 else "below_rebalance_turnover_threshold"
        rebalance_intensity = 1.0 if requested else 0.0
        return {
            "weights": executed_weights.squeeze(0).detach().cpu().numpy(),
            "candidate_weights": candidate_weights.squeeze(0).detach().cpu().numpy(),
            "rebalance": int(requested),
            "rebalance_intensity": rebalance_intensity,
            "gate_action": int(requested),
            "raw_bernoulli_gate_action": raw_gate_action,
            "candidate_log_prob": float(candidate_log_prob.view(-1)[0].detach().cpu()),
            "gate_log_prob": float(gate_log_prob.view(-1)[0].detach().cpu()),
            "gate_entropy": float(gate_entropy.view(-1)[0].detach().cpu()),
            "p_rebalance": float(p_rebalance.view(-1)[0].detach().cpu()),
            "deterministic_gate_threshold": float(gate_threshold),
            "value": float(value.view(-1)[0].detach().cpu()),
            "estimated_turnover": turnover_value,
            "candidate_turnover": turnover_value,
            "candidate_turnover_estimate": turnover_value,
            "estimated_cost": 0.0,
            "continuous_weight_rebalance_gate": True,
            "rebalance_turnover_threshold": threshold,
            "raw_model_requested_rebalance": bool(raw_gate_action),
            "raw_action": raw_gate_action,
            "raw_rho": float(raw_gate_action),
            "raw_rebalance_intensity": float(raw_gate_action),
            "first_trade": first_trade,
            "forced_hold_reason": forced_hold_reason,
        }

    def _update(self, rollout: Mapping[str, list[Any]]) -> dict[str, float]:
        if not rollout["reward"]:
            raise ValueError("ERR_BERNOULLI_ROLLOUT_EMPTY")
        self.encoder.train()
        self.actor.train()
        self.critic.train()
        self.gate.train()
        market_image, availability_mask, _ = self._observation_tensors(rollout["state"])
        latent = self.encoder(market_image)
        candidate_weights = torch.as_tensor(np.stack(rollout["candidate_weights"]), dtype=torch.float32, device=self.device)
        old_candidate_log_prob = _column_tensor(rollout["candidate_log_prob"], self.device)
        old_gate_log_prob = _column_tensor(rollout["gate_log_prob"], self.device)
        gate_action = _column_tensor(rollout["gate_action"], self.device)
        executed_gate_action = _column_tensor(rollout.get("executed_gate_action", rollout["gate_action"]), self.device)
        returns = _column_tensor(rollout["return"], self.device)
        advantages = _column_tensor(rollout["advantage"], self.device)

        distribution = self.actor.get_distribution(latent, availability_mask)
        candidate_log_prob = distribution.log_prob(candidate_weights).view(-1, 1)
        values = self.critic(latent).view(-1, 1)
        p_rebalance = self.gate(latent).view(-1, 1)
        gate_dist = Bernoulli(probs=p_rebalance)
        gate_log_prob = gate_dist.log_prob(gate_action).view(-1, 1)
        gate_entropy = gate_dist.entropy().view(-1, 1).mean()
        actor_entropy = _dirichlet_entropy(distribution).view(-1, 1).mean()

        clip_range = _ppo_float(self.config, "clip_ratio", _ppo_float(self.config, "clip_range", 0.20))
        actor_loss = _clipped_policy_loss(
            candidate_log_prob,
            old_candidate_log_prob,
            advantages,
            clip_range=clip_range,
            loss_weight=executed_gate_action,
        )
        gate_loss = _clipped_policy_loss(
            gate_log_prob,
            old_gate_log_prob,
            advantages,
            clip_range=clip_range,
            loss_weight=None,
        )
        value_loss = F.mse_loss(values, returns)
        entropy_coef = _ppo_float(self.config, "entropy_coef", 0.01)
        value_coef = _ppo_float(self.config, "value_coef", 0.50)
        loss = actor_loss + gate_loss + value_coef * value_loss - entropy_coef * (actor_entropy + gate_entropy)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gate_grad_norm = _grad_norm(self.gate.parameters())
        torch.nn.utils.clip_grad_norm_(self.parameters(), _ppo_float(self.config, "max_grad_norm", 0.50))
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "gate_loss": float(gate_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "gate_entropy": float(gate_entropy.detach().cpu()),
            "gate_grad_norm": float(gate_grad_norm),
        }

    def _evaluate(self, env: PortfolioRebalanceEnv, max_steps: int | None = None) -> float:
        rollout = self._collect_rollout(env, deterministic=True, max_steps=max_steps)
        if not rollout["reward"]:
            return float("-inf")
        return float(np.sum(rollout["reward"]))

    def _observation_tensors(self, states: list[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        market_image = torch.as_tensor(
            np.stack([np.asarray(state["market_image"], dtype=np.float32) for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        availability_mask = torch.as_tensor(
            np.stack([np.asarray(state["availability_mask"], dtype=bool) for state in states]),
            dtype=torch.bool,
            device=self.device,
        )
        current_weights = torch.as_tensor(
            np.stack([np.asarray(state["current_weights"], dtype=np.float32) for state in states]),
            dtype=torch.float32,
            device=self.device,
        )
        return market_image, availability_mask, current_weights

    def _build_modules(self) -> tuple[nn.Module, PPOActor, PPOCritic, BernoulliGate, torch.optim.Optimizer]:
        config = dict(self.config)
        encoder_config = dict(_mapping(config.get("encoder")))
        encoder_config.setdefault("type", "cnn")
        config["encoder"] = encoder_config
        encoder = EncoderFactory.create(config).to(self.device)
        if str(encoder_config.get("type", "cnn")).lower() == "cnn" and not isinstance(encoder, CNNEncoder):
            raise ValueError("ERR_BERNOULLI_ENCODER: expected CNN encoder")
        latent_dim = int(config.get("latent_dim", _mapping(config.get("model")).get("latent_dim", 256)))
        actor = PPOActor(latent_dim=latent_dim, n_assets=int(config["n_assets"])).to(self.device)
        critic = PPOCritic(latent_dim=latent_dim).to(self.device)
        gate = BernoulliGate(latent_dim=latent_dim).to(self.device)
        params = list(encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()) + list(gate.parameters())
        optimizer = torch.optim.AdamW(params, lr=_optimizer_lr(config))
        return encoder, actor, critic, gate, optimizer

    def parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in (self.encoder, self.actor, self.critic, self.gate):
            params.extend(list(module.parameters()))
        return params

    def _checkpoint_paths(self) -> dict[str, Path | None]:
        checkpoint_dir = _mapping(self.config.get("baselines")).get("checkpoint_dir")
        if checkpoint_dir is None:
            checkpoint_dir = self.config.get("baseline_run_dir")
        if checkpoint_dir is None:
            return {"best": None, "last": None}
        root = Path(checkpoint_dir) / "checkpoints" / self.strategy_name
        return {"best": root / "best.pt", "last": root / "last.pt"}

    def _save_state(self, path: Path | None, epoch: int, global_step: int, best_metric: float | None) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state": self.encoder.state_dict(),
                "actor_state": self.actor.state_dict(),
                "critic_state": self.critic.state_dict(),
                "gate_state": self.gate.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": int(epoch),
                "global_step": int(global_step),
                "best_validation_metric": best_metric,
            },
            path,
        )

    def _load_state(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(payload["encoder_state"])
        self.actor.load_state_dict(payload["actor_state"])
        self.critic.load_state_dict(payload["critic_state"])
        self.gate.load_state_dict(payload["gate_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])


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
    training_history: pd.DataFrame,
    *,
    checkpoint_best_path: str | None = None,
    checkpoint_last_path: str | None = None,
    evaluated_checkpoint_path: str | None = None,
    best_validation_metric: float | None = None,
    env_steps: int = 0,
    gradient_updates: int = 0,
    gate_entropy: float | None = None,
    p_rebalance_mean: float | None = None,
    rebalance_frequency: float | None = None,
    max_train_steps: int | None = None,
    max_validation_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "baseline_family": "native_rl",
        "status": status,
        "training_algorithm": BERNOULLI_GATED_ALGORITHM,
        "rl_training": True,
        "platform_native_rl_training": True,
        "proxy_training": False,
        "external_original_implementation": False,
        "rankable_in_unified_table": True,
        "gate_training": "on_policy_bernoulli",
        "training_history": training_history,
        "checkpoint_best_path": checkpoint_best_path,
        "checkpoint_last_path": checkpoint_last_path,
        "evaluated_checkpoint_path": evaluated_checkpoint_path,
        "best_validation_metric": best_validation_metric,
        "env_steps": int(env_steps),
        "gradient_updates": int(gradient_updates),
        "gate_entropy": gate_entropy,
        "p_rebalance_mean": p_rebalance_mean,
        "rebalance_frequency": rebalance_frequency,
        "max_train_steps": max_train_steps,
        "max_validation_steps": max_validation_steps,
    }


def _clipped_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float,
    loss_weight: torch.Tensor | None,
) -> torch.Tensor:
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)) * advantages
    loss = -torch.min(unclipped, clipped)
    if loss_weight is None:
        return loss.mean()
    denominator = loss_weight.sum().clamp_min(1.0)
    return (loss * loss_weight).sum() / denominator


def _gae(
    rewards: list[float],
    values: list[float],
    terminated: list[bool],
    truncated: list[bool],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0
    for index in reversed(range(len(rewards))):
        non_terminal = 0.0 if terminated[index] or truncated[index] else 1.0
        delta = float(rewards[index]) + float(gamma) * next_value * non_terminal - float(values[index])
        last_gae = delta + float(gamma) * float(gae_lambda) * non_terminal * last_gae
        advantages[index] = float(last_gae)
        next_value = float(values[index])
    returns = advantages + np.asarray(values, dtype=np.float32)
    if len(advantages) > 1 and float(advantages.std()) > 1.0e-8:
        advantages = (advantages - float(advantages.mean())) / float(advantages.std())
    return advantages, returns.astype(np.float32)


def _dirichlet_entropy(distribution: Any) -> torch.Tensor:
    entropies = torch.zeros(distribution.batch_size, device=distribution.device)
    for index, dist in enumerate(distribution.dists):
        if dist is not None:
            entropies[index] = dist.entropy()
    return entropies


def _column_tensor(values: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32, device=device).view(-1, 1)


def _grad_norm(parameters: Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().cpu())
    return float(np.sqrt(total))


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
        raise ValueError("ERR_BERNOULLI_NATIVE_CONFIG_INVALID: max step limits must be > 0")
    return result


def _ppo_float(config: Mapping[str, Any], key: str, default: float) -> float:
    ppo = _mapping(config.get("ppo"))
    value = ppo.get(key)
    if value is None and key == "clip_ratio":
        value = ppo.get("clip_range")
    return float(default if value is None else value)


def _optimizer_lr(config: Mapping[str, Any]) -> float:
    optimizer = _mapping(config.get("optimizer"))
    ppo = _mapping(config.get("ppo"))
    for value in (optimizer.get("ppo_lr"), optimizer.get("learning_rate"), ppo.get("lr"), 3.0e-4):
        if value is not None:
            return float(value)
    return 3.0e-4


def _deterministic_gate_threshold(config: Mapping[str, Any]) -> float:
    model_config = _mapping(config.get("bernoulli_gated_ppo_native"))
    for key in ("deterministic_gate_threshold", "rebalance_probability_threshold", "gate_threshold"):
        if model_config.get(key) is not None:
            return _unit_interval_float(key, model_config[key])
    activity = _mapping(config.get("execution_activity"))
    for key in ("deterministic_gate_threshold", "rebalance_probability_threshold", "gate_threshold"):
        if activity.get(key) is not None:
            return _unit_interval_float(key, activity[key])
    return 0.5


def _unit_interval_float(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"ERR_BERNOULLI_NATIVE_CONFIG_INVALID: {name} must be in [0,1]")
    return result


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
    "BERNOULLI_GATED_ALGORITHM",
    "BernoulliGate",
    "NativeBernoulliGatedPPOBaselineStrategy",
]
