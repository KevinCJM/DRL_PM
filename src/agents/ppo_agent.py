from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.training_guard import assert_finite_losses, clip_grad_norm_checked
from src.buffers.rollout_buffer import RolloutBuffer
from src.models.cost_estimator import CostEstimator
from src.models.ppo_actor import MaskedDirichlet
from src.models.ppo_critic import PPOCritic


@dataclass(frozen=True)
class PPOAgentConfig:
    rollout_steps: int = 256
    minibatch_size: int = 64
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    value_clip_range: float = 0.20
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    advantage_normalization: bool = True
    lr: float = 3.0e-4

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None = None) -> PPOAgentConfig:
        if not config:
            return cls()
        ppo_config = _mapping(config.get("ppo"))
        optimizer_config = _mapping(config.get("optimizer"))
        clip_range = _first_present(ppo_config, "clip_ratio", "clip_range", default=cls.clip_range)
        lr = _first_present(optimizer_config, "ppo_lr", "learning_rate", default=None)
        if lr is None:
            lr = _first_present(ppo_config, "lr", default=cls.lr)
        return cls(
            rollout_steps=_positive_int("rollout_steps", ppo_config.get("rollout_steps", cls.rollout_steps)),
            minibatch_size=_positive_int("minibatch_size", ppo_config.get("minibatch_size", cls.minibatch_size)),
            update_epochs=_positive_int("update_epochs", ppo_config.get("update_epochs", cls.update_epochs)),
            gamma=_unit_interval_float("gamma", ppo_config.get("gamma", cls.gamma)),
            gae_lambda=_unit_interval_float("gae_lambda", ppo_config.get("gae_lambda", cls.gae_lambda)),
            clip_range=_non_negative_float("clip_range", clip_range),
            value_clip_range=_non_negative_float(
                "value_clip_range",
                ppo_config.get("value_clip_range", cls.value_clip_range),
            ),
            entropy_coef=_non_negative_float("entropy_coef", ppo_config.get("entropy_coef", cls.entropy_coef)),
            value_coef=_non_negative_float("value_coef", ppo_config.get("value_coef", cls.value_coef)),
            max_grad_norm=_positive_float("max_grad_norm", ppo_config.get("max_grad_norm", cls.max_grad_norm)),
            advantage_normalization=bool(
                ppo_config.get("advantage_normalization", cls.advantage_normalization)
            ),
            lr=_positive_float("lr", lr),
        )


class PPOAgent:
    def __init__(
        self,
        encoder: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        config: Mapping[str, Any] | PPOAgentConfig | None = None,
        device: torch.device | str | None = None,
        gate_network: nn.Module | None = None,
        q_gap_threshold: float = 0.0,
        policy_model: nn.Module | None = None,
    ):
        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.gate_network = gate_network
        self.policy_model = policy_model
        self.q_gap_threshold = float(q_gap_threshold)
        self.raw_config = dict(config) if isinstance(config, Mapping) else {}
        self.config = config if isinstance(config, PPOAgentConfig) else PPOAgentConfig.from_mapping(config)
        self.device = torch.device("cpu" if device is None else device)
        self.encoder.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        if self.gate_network is not None:
            self.gate_network.to(self.device)
        if self.policy_model is not None:
            self.policy_model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        self.rollout_buffer = RolloutBuffer(
            rollout_steps=self.config.rollout_steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            advantage_normalization=self.config.advantage_normalization,
        )

    def parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        seen: set[int] = set()
        for module in (self.encoder, self.actor, self.critic):
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    params.append(parameter)
                    seen.add(id(parameter))
        for module_name in (
            "conditioner",
            "uncertainty_heads",
            "candidate_dist_critic",
            "hold_dist_critic",
            "intensity_actor",
            "discrete_gate",
        ):
            module = getattr(self.policy_model, module_name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    if id(parameter) not in seen:
                        params.append(parameter)
                        seen.add(id(parameter))
        return params

    def collect_rollout(
        self,
        env: Any,
        initial_observation: Mapping[str, Any] | None = None,
        gate_action_selector: Callable[[torch.Tensor], torch.Tensor] | None = None,
        max_steps: int | None = None,
    ) -> RolloutBuffer:
        self.rollout_buffer.clear()
        if initial_observation is None:
            observation, _ = env.reset()
        else:
            observation = dict(initial_observation)
        terminated = False
        truncated = False
        last_value = 0.0

        step_count = 0
        while not self.rollout_buffer.is_full and not (terminated or truncated):
            if max_steps is not None and step_count >= int(max_steps):
                break
            if gate_action_selector is None:
                action_info = self.select_action(observation, deterministic=False)
            else:
                action_info = self.select_action(
                    observation,
                    deterministic=False,
                    gate_action_selector=gate_action_selector,
                )
            gate_action = _gate_action(action_info)
            rebalance_intensity = _rebalance_intensity(action_info, gate_action)
            action = self.action_for_env(observation, action_info)
            next_observation, reward, terminated, truncated, info = env.step(action)
            rollout_features = _rollout_features(action_info, info)
            self.rollout_buffer.add(
                decision_date=info.get("decision_date", _observation_date(observation)),
                execution_date=info.get("execution_date", info.get("decision_date", _observation_date(observation))),
                next_valuation_date=info.get(
                    "next_valuation_date",
                    info.get("execution_date", info.get("decision_date", _observation_date(observation))),
                ),
                execution_price=_execution_price(env, info),
                delayed_action_execution=_delayed_action_execution(env),
                state=observation,
                candidate_weights=action_info["candidate_weights"],
                executed_weights=info.get("executed_weights", action_info["candidate_weights"]),
                log_prob=action_info["log_prob"],
                value=action_info["value"],
                decision_value=action_info["value"],
                gate_action=info.get("gate_action", gate_action),
                rebalance_action=info.get("rebalance_action", info.get("rebalance", gate_action)),
                rebalance_intensity=info.get("rebalance_intensity", rebalance_intensity),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                auxiliary_labels=info.get("auxiliary_labels", {}),
                preference_vector=info.get("preference_vector"),
                uncertainty_features=rollout_features,
                distributional_features=info.get("distributional_features", action_info.get("distributional_features", {})),
            )
            observation = next_observation
            step_count += 1

        self.rollout_buffer.last_observation = observation
        self.rollout_buffer.rollout_boundary_split = bool(self.rollout_buffer.is_full and not (terminated or truncated))
        if not (terminated or truncated):
            last_value = self.value(observation)
        self.rollout_buffer.compute_gae(last_value=last_value, last_terminated=bool(terminated or truncated))
        return self.rollout_buffer

    @torch.no_grad()
    def select_action(
        self,
        observation: Mapping[str, Any],
        deterministic: bool = False,
        gate_action_selector: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if self.policy_model is not None:
            return self._select_action_with_policy_model(
                observation,
                deterministic=deterministic,
                gate_action_selector=gate_action_selector,
            )
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        if self.gate_network is not None:
            self.gate_network.eval()
        market_image, availability_mask = self._observation_tensors([observation])
        current_weights = _current_weights_from_observation(observation, market_image.device)
        latent = self.encoder(market_image)
        distribution = self.actor.get_distribution(latent, availability_mask)
        candidate_weights = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(candidate_weights)
        value = self.critic(latent)
        estimated_turnover, estimated_cost = _estimate_action_cost(
            candidate_weights,
            current_weights,
            self.raw_config,
            observation,
        )
        gate_payload = _gate_payload(
            self.gate_network,
            latent,
            candidate_weights,
            current_weights,
            estimated_turnover,
            estimated_cost,
            self.q_gap_threshold,
            gate_action_selector,
        )
        return {
            "candidate_weights": candidate_weights.squeeze(0).detach().cpu().numpy(),
            "log_prob": float(log_prob.squeeze(0).detach().cpu()),
            "value": float(value.squeeze(0).detach().cpu()),
            "estimated_turnover": float(estimated_turnover.squeeze(0).detach().cpu()),
            "estimated_cost": float(estimated_cost.squeeze(0).detach().cpu()),
            **gate_payload,
        }

    def action_for_env(self, observation: Mapping[str, Any], action_info: Mapping[str, Any]) -> dict[str, Any]:
        gate_action = _gate_action(action_info)
        rebalance_intensity = _rebalance_intensity(action_info, gate_action)
        action = {
            "weights": _env_action_weights(observation, action_info, gate_action),
            "rebalance": gate_action,
            "rebalance_intensity": rebalance_intensity,
        }
        for key in (
            "gate_action",
            "gate_action_index",
            "q_hold",
            "q_rebalance",
            "q_gap",
            "estimated_turnover",
            "estimated_cost",
            "candidate_turnover",
            "candidate_turnover_estimate",
            "candidate_cost_estimate",
            "rho",
            "raw_rho",
            "raw_rebalance_intensity",
            "raw_model_requested_rebalance",
            "raw_action",
            "final_rho",
            "final_rebalance_intensity",
            "final_action",
            "forced_hold_reason",
            "rebalance_values",
            "continuous_weight_rebalance_gate",
        ):
            if key in action_info and action_info[key] is not None:
                action[key] = action_info[key]
        return action

    @torch.no_grad()
    def value(self, observation: Mapping[str, Any]) -> float:
        if self.policy_model is not None:
            self.policy_model.eval()
            market_image, availability_mask = self._observation_tensors([observation])
            current_weights = _current_weights_from_observation(observation, market_image.device)
            estimate = torch.zeros((market_image.shape[0], 1), dtype=market_image.dtype, device=market_image.device)
            outputs = self._policy_forward(
                market_image,
                availability_mask,
                current_weights,
                estimate,
                estimate,
                deterministic=True,
                omega=_preference_omega_from_states([observation], self.device),
            )
            return float(outputs["value"].view(-1)[0].detach().cpu())
        self.encoder.eval()
        self.critic.eval()
        market_image, _ = self._observation_tensors([observation])
        return float(self.critic(self.encoder(market_image)).squeeze(0).detach().cpu())

    def update(self, rollout_buffer: RolloutBuffer | None = None) -> dict[str, float]:
        buffer = rollout_buffer or self.rollout_buffer
        if len(buffer) == 0:
            raise ValueError("ERR_PPO_ROLLOUT_EMPTY")
        if any(item.advantage is None or item.return_ is None for item in buffer.items):
            buffer.compute_gae(last_value=0.0)

        self.encoder.train()
        self.actor.train()
        self.critic.train()
        if self.policy_model is not None:
            self.policy_model.train()
        batch = buffer.as_batch(device=self.device)
        batch_size = int(batch["candidate_weights"].shape[0])
        minibatch_size = min(self.config.minibatch_size, batch_size)
        stats: list[dict[str, float]] = []
        for _ in range(self.config.update_epochs):
            for indices in torch.randperm(batch_size, device=self.device).split(minibatch_size):
                mini_batch = _index_batch(batch, indices)
                losses = self.compute_losses(mini_batch)
                assert_finite_losses(losses, "ppo")
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                grad_norm = clip_grad_norm_checked(self.parameters(), self.config.max_grad_norm, "ppo")
                self.optimizer.step()
                stats.append({**_loss_stats(losses), "grad_norm": float(grad_norm.detach().cpu())})
        return _mean_stats(stats)

    def compute_losses(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if self.policy_model is not None:
            return self._compute_policy_model_losses(batch)
        market_image, availability_mask = self._observation_tensors(batch["state"])
        latent = self.encoder(market_image)
        candidate_weights = batch["candidate_weights"].to(self.device)
        old_log_prob = batch["log_prob"].to(self.device)
        old_values = batch["value"].to(self.device)
        returns = batch["return"].to(self.device)
        advantages = batch["advantage"].to(self.device)
        distribution = self.actor.get_distribution(latent, availability_mask)
        new_log_prob = distribution.log_prob(candidate_weights).view(-1, 1)
        values = self.critic(latent)

        ratio = torch.exp(new_log_prob - old_log_prob)
        actor_loss_weight = self.actor_loss_weights(batch).to(self.device)
        actor_loss = self.clipped_actor_loss(
            new_log_prob,
            old_log_prob,
            advantages,
            actor_loss_weight=actor_loss_weight,
            clip_range=self.config.clip_range,
        )
        value_loss = PPOCritic.clipped_value_loss(
            values,
            old_values,
            returns,
            clip_range=self.config.value_clip_range,
        )
        entropy = self._entropy(distribution).view(-1, 1)
        entropy_loss = entropy.mean()
        loss = actor_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

        with torch.no_grad():
            approx_kl = (old_log_prob - new_log_prob).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_range).to(dtype=torch.float32).mean()
        return {
            "loss": loss,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "actor_loss_weight_mean": actor_loss_weight.mean(),
        }

    def _compute_policy_model_losses(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        market_image, availability_mask = self._observation_tensors(batch["state"])
        candidate_weights = batch["candidate_weights"].to(self.device)
        old_log_prob = batch["log_prob"].to(self.device)
        old_values = batch["value"].to(self.device)
        returns = batch["return"].to(self.device)
        advantages = batch["advantage"].to(self.device)
        current_weights = _current_weights_from_states(batch["state"], self.device)
        fallback_turnover = 0.5 * torch.sum(torch.abs(candidate_weights - current_weights), dim=1, keepdim=True)
        estimated_turnover = _feature_column(batch, "estimated_turnover", self.device, fallback_turnover)
        estimated_cost = _feature_column(batch, "estimated_cost", self.device, torch.zeros_like(fallback_turnover))
        rebalance_intensity = _feature_column(
            batch,
            "raw_rebalance_intensity",
            self.device,
            batch["rebalance_intensity"].to(self.device),
        )
        outputs = self._policy_forward(
            market_image,
            availability_mask,
            current_weights,
            estimated_turnover,
            estimated_cost,
            deterministic=True,
            candidate_weights_override=candidate_weights,
            rebalance_intensity_override=rebalance_intensity,
            omega=_preference_omega_from_states(batch["state"], self.device),
        )
        log_prob = outputs.get("joint_log_prob", outputs.get("log_prob"))
        new_log_prob = _column_tensor(log_prob, self.device, "new_log_prob")
        values = _column_tensor(outputs["value"], self.device, "value")

        ratio = torch.exp(new_log_prob - old_log_prob)
        actor_loss_weight = self.actor_loss_weights(batch).to(self.device)
        actor_loss = self.clipped_actor_loss(
            new_log_prob,
            old_log_prob,
            advantages,
            actor_loss_weight=actor_loss_weight,
            clip_range=self.config.clip_range,
        )
        value_loss = PPOCritic.clipped_value_loss(
            values,
            old_values,
            returns,
            clip_range=self.config.value_clip_range,
        )
        latent = outputs.get("latent")
        if isinstance(latent, torch.Tensor):
            distribution = self.actor.get_distribution(latent, availability_mask)
            entropy_loss = self._entropy(distribution).view(-1, 1).mean()
        else:
            entropy_loss = torch.zeros((), dtype=market_image.dtype, device=self.device)
        loss = actor_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss
        extra_losses = self._specialized_policy_losses(outputs, returns)
        for name, item in extra_losses.items():
            if name.endswith("_coef"):
                continue
            loss = loss + extra_losses.get(f"{name}_coef", torch.tensor(1.0, device=self.device)) * item

        with torch.no_grad():
            approx_kl = (old_log_prob - new_log_prob).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_range).to(dtype=torch.float32).mean()
        stats = {
            "loss": loss,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "actor_loss_weight_mean": actor_loss_weight.mean(),
        }
        stats.update({name: item for name, item in extra_losses.items() if not name.endswith("_coef")})
        return stats

    def actor_loss_weights(self, batch: Mapping[str, Any]) -> torch.Tensor:
        gate_action = batch["gate_action"].to(self.device)
        rebalance_intensity = batch["rebalance_intensity"].to(self.device)
        weights = torch.clamp(rebalance_intensity, min=0.0, max=1.0)
        return torch.where(gate_action == 0, torch.zeros_like(weights), weights)

    def _policy_forward(
        self,
        market_image: torch.Tensor,
        availability_mask: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
        *,
        deterministic: bool,
        candidate_weights_override: torch.Tensor | None = None,
        rebalance_intensity_override: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
        reward_vector: torch.Tensor | None = None,
    ) -> Mapping[str, Any]:
        if self.policy_model is None:
            raise ValueError("ERR_PPO_POLICY_MODEL_MISSING")
        kwargs: dict[str, Any] = {
            "deterministic": deterministic,
            "candidate_weights_override": candidate_weights_override,
            "rebalance_intensity_override": rebalance_intensity_override,
        }
        if omega is not None and hasattr(self.policy_model, "conditioner"):
            kwargs["omega"] = omega
        if reward_vector is not None and hasattr(self.policy_model, "conditioner"):
            kwargs["reward_vector"] = reward_vector
        return self.policy_model(
            market_image,
            availability_mask,
            current_weights,
            estimated_turnover,
            estimated_cost,
            **kwargs,
        )

    def _specialized_policy_losses(
        self,
        outputs: Mapping[str, Any],
        returns: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.policy_model is None or "candidate_quantiles" not in outputs:
            return {}
        quantile_loss = self.policy_model.quantile_huber_loss(outputs["candidate_quantiles"], returns.detach())
        distributional_config = _mapping(self.raw_config.get("distributional_cvar"))
        coef = _non_negative_float("distributional_cvar.loss_coef", distributional_config.get("loss_coef", 0.10))
        return {
            "distributional_loss": quantile_loss,
            "distributional_loss_coef": torch.tensor(coef, dtype=returns.dtype, device=returns.device),
        }

    @staticmethod
    def clipped_actor_loss(
        new_log_prob: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        actor_loss_weight: torch.Tensor | None = None,
        clip_range: float = 0.20,
    ) -> torch.Tensor:
        _assert_column(new_log_prob, "new_log_prob")
        if old_log_prob.shape != new_log_prob.shape or advantages.shape != new_log_prob.shape:
            raise ValueError("ERR_PPO_AGENT_SHAPE_MISMATCH: log_prob and advantage must share [batch,1]")
        if actor_loss_weight is not None and actor_loss_weight.shape != new_log_prob.shape:
            raise ValueError("ERR_PPO_AGENT_SHAPE_MISMATCH: actor_loss_weight must be [batch,1]")
        if not _all_finite(new_log_prob, old_log_prob, advantages):
            raise ValueError("ERR_PPO_AGENT_NON_FINITE: actor loss inputs contain NaN or Inf")
        if actor_loss_weight is not None and not torch.isfinite(actor_loss_weight).all():
            raise ValueError("ERR_PPO_AGENT_NON_FINITE: actor_loss_weight contains NaN or Inf")
        clip_value = _non_negative_float("clip_range", clip_range)
        ratio = torch.exp(new_log_prob - old_log_prob)
        unclipped_policy_loss = ratio * advantages
        clipped_policy_loss = torch.clamp(ratio, 1.0 - clip_value, 1.0 + clip_value) * advantages
        actor_loss_per_item = -torch.min(unclipped_policy_loss, clipped_policy_loss)
        if actor_loss_weight is None:
            return actor_loss_per_item.mean()
        return _weighted_mean(actor_loss_per_item, actor_loss_weight)

    def _observation_tensors(self, states: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
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
        if market_image.ndim != 4:
            raise ValueError("ERR_PPO_AGENT_OBSERVATION_SHAPE: market_image must be [batch,n_features,window,n_assets]")
        if availability_mask.ndim != 2 or availability_mask.shape[0] != market_image.shape[0]:
            raise ValueError("ERR_PPO_AGENT_OBSERVATION_SHAPE: availability_mask must be [batch,n_assets]")
        return market_image, availability_mask

    @staticmethod
    def _entropy(distribution: MaskedDirichlet) -> torch.Tensor:
        entropies = torch.zeros(distribution.batch_size, device=distribution.device)
        for index, dist in enumerate(distribution.dists):
            if dist is not None:
                entropies[index] = dist.entropy()
        return entropies

    def _select_action_with_policy_model(
        self,
        observation: Mapping[str, Any],
        *,
        deterministic: bool,
        gate_action_selector: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> dict[str, Any]:
        self.policy_model.eval()
        market_image, availability_mask = self._observation_tensors([observation])
        current_weights = _current_weights_from_observation(observation, market_image.device)
        zero_estimate = torch.zeros((market_image.shape[0], 1), dtype=market_image.dtype, device=market_image.device)
        outputs = self._policy_forward(
            market_image,
            availability_mask,
            current_weights,
            zero_estimate,
            zero_estimate,
            deterministic=deterministic,
            omega=_preference_omega_from_states([observation], self.device),
        )
        candidate_weights = _output_tensor(outputs, "candidate_weights", self.device)
        model_intensity_tensor = _optional_output_tensor(_raw_rebalance_intensity_output(outputs), self.device)
        estimated_turnover, estimated_cost = _estimate_action_cost(
            candidate_weights,
            current_weights,
            self.raw_config,
            observation,
        )
        outputs = self._policy_forward(
            market_image,
            availability_mask,
            current_weights,
            estimated_turnover,
            estimated_cost,
            deterministic=True,
            candidate_weights_override=candidate_weights,
            rebalance_intensity_override=model_intensity_tensor,
            omega=_preference_omega_from_states([observation], self.device),
        )
        candidate_weights = _output_tensor(outputs, "candidate_weights", self.device)
        estimated_turnover, estimated_cost = _estimate_action_cost(
            candidate_weights,
            current_weights,
            self.raw_config,
            observation,
        )
        gate_q = _output_tensor(outputs, "gate_q", self.device)
        model_intensity = _optional_output_scalar(_raw_rebalance_intensity_output(outputs))
        gate_payload = _gate_payload_from_q(
            gate_q,
            self.q_gap_threshold,
            gate_action_selector,
            rebalance_intensity=model_intensity,
            model_gate_action=_optional_output_action(outputs.get("gate_action"), self.device),
            rebalance_values=outputs.get("rebalance_values"),
        )
        log_prob = outputs.get("joint_log_prob", outputs.get("log_prob"))
        value = outputs["value"]
        payload: dict[str, Any] = {
            "candidate_weights": candidate_weights.squeeze(0).detach().cpu().numpy(),
            "log_prob": float(torch.as_tensor(log_prob).view(-1)[0].detach().cpu()),
            "value": float(torch.as_tensor(value).view(-1)[0].detach().cpu()),
            "estimated_turnover": float(estimated_turnover.squeeze(0).detach().cpu()),
            "estimated_cost": float(estimated_cost.squeeze(0).detach().cpu()),
            **gate_payload,
        }
        for key in ("uncertainty_features", "distributional_features"):
            if key in outputs:
                payload[key] = _detach_value(outputs[key])
        if "omega" in outputs:
            payload["preference_vector"] = _detach_value(outputs["omega"])
        if "reward_vector" in outputs:
            payload["reward_vector"] = _detach_value(outputs["reward_vector"])
        if "raw_rebalance_intensity" in outputs:
            payload["raw_rebalance_intensity"] = _optional_output_scalar(outputs["raw_rebalance_intensity"])
        return payload


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


def _assert_column(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or value.shape[1] != 1:
        raise ValueError(f"ERR_PPO_AGENT_SHAPE_MISMATCH: {name} must be [batch,1]")


def _all_finite(*values: torch.Tensor) -> bool:
    return all(torch.isfinite(value).all().item() for value in values)


def _index_batch(batch: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    cpu_indices = indices.detach().cpu().tolist()
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.index_select(0, indices)
        elif isinstance(value, list):
            result[key] = [value[int(index)] for index in cpu_indices]
        else:
            result[key] = value
    return result


def _loss_stats(losses: Mapping[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items() if key != "loss"}


def _mean_stats(stats: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not stats:
        return {}
    keys = stats[0].keys()
    return {key: float(np.mean([item[key] for item in stats])) for key in keys}


def _current_weights_from_observation(observation: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    weights = np.asarray(observation.get("current_weights"), dtype=np.float32)
    if weights.ndim != 1:
        raise ValueError("ERR_PPO_AGENT_OBSERVATION_SHAPE: current_weights must be [n_assets]")
    return torch.as_tensor(weights, dtype=torch.float32, device=device).unsqueeze(0)


def _current_weights_from_states(states: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor:
    weights = np.stack([np.asarray(state.get("current_weights"), dtype=np.float32) for state in states])
    if weights.ndim != 2:
        raise ValueError("ERR_PPO_AGENT_OBSERVATION_SHAPE: current_weights must be [batch,n_assets]")
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _preference_omega_from_states(states: Sequence[Mapping[str, Any]], device: torch.device) -> torch.Tensor | None:
    if not states or not all(isinstance(state, Mapping) and "preference_omega" in state for state in states):
        return None
    values = [np.asarray(state["preference_omega"], dtype=np.float32) for state in states]
    if any(value.ndim != 1 for value in values):
        return None
    return torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)


def _feature_column(
    batch: Mapping[str, Any],
    key: str,
    device: torch.device,
    fallback: torch.Tensor,
) -> torch.Tensor:
    features = batch.get("uncertainty_features")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        return fallback.to(device=device, dtype=torch.float32)
    values: list[float] = []
    for item in features:
        if not isinstance(item, Mapping) or item.get(key) is None:
            return fallback.to(device=device, dtype=torch.float32)
        values.append(float(np.asarray(item[key], dtype=float).reshape(-1)[0]))
    return torch.as_tensor(values, dtype=torch.float32, device=device).view(-1, 1)


def _estimate_action_cost(
    candidate_weights: torch.Tensor,
    current_weights: torch.Tensor,
    config: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    if _has_cost_observation(observation):
        turnover, cost = CostEstimator.estimate(
            candidate_weights,
            current_weights,
            _observation_matrix(observation, "adv20_at_decision", candidate_weights),
            _observation_matrix(observation, "volatility_20d_at_decision", candidate_weights),
            float(np.asarray(observation["portfolio_value"], dtype=float)),
            config,
            amount=_observation_matrix(observation, "amount_at_decision", candidate_weights),
            turnover_rate=_observation_matrix(observation, "turnover_rate_at_decision", candidate_weights),
        )
        return turnover, cost
    turnover = 0.5 * torch.sum(torch.abs(candidate_weights - current_weights), dim=1, keepdim=True)
    cost = turnover * 0.0
    return turnover, cost


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


def _output_tensor(outputs: Mapping[str, Any], key: str, device: torch.device) -> torch.Tensor:
    value = outputs[key]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _column_tensor(value: Any, device: torch.device, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.view(-1, 1)
    _assert_column(tensor, name)
    return tensor


def _optional_output_tensor(value: Any, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        return None
    return tensor


def _optional_output_scalar(value: Any) -> float | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        return None
    return float(tensor.view(-1)[0].detach().cpu())


def _raw_rebalance_intensity_output(outputs: Mapping[str, Any]) -> Any:
    return outputs.get("raw_rebalance_intensity", outputs.get("rebalance_intensity"))


def _optional_output_action(value: Any, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.numel() == 0:
        return None
    tensor = tensor.to(device=device, dtype=torch.long)
    if tensor.ndim == 0:
        tensor = tensor.view(1)
    if tensor.ndim != 1:
        tensor = tensor.view(-1)
    return tensor


def _detach_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, Mapping):
        return {str(key): _detach_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detach_value(item) for item in value]
    return value


def _gate_payload(
    gate_network: nn.Module | None,
    latent: torch.Tensor,
    candidate_weights: torch.Tensor,
    current_weights: torch.Tensor,
    estimated_turnover: torch.Tensor,
    estimated_cost: torch.Tensor,
    q_gap_threshold: float,
    gate_action_selector: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    if gate_network is None:
        return {}
    gate_q = gate_network(latent, candidate_weights, current_weights, estimated_turnover, estimated_cost)
    return _gate_payload_from_q(gate_q, q_gap_threshold, gate_action_selector)


def _gate_payload_from_q(
    gate_q: torch.Tensor,
    q_gap_threshold: float,
    gate_action_selector: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    rebalance_intensity: float | None = None,
    model_gate_action: torch.Tensor | None = None,
    rebalance_values: Any = None,
) -> dict[str, Any]:
    if gate_q.ndim != 2 or gate_q.shape[0] != 1:
        raise ValueError("ERR_PPO_AGENT_GATE_SHAPE: gate_q must be [1,n_actions]")
    selector_action = None
    if gate_action_selector is not None:
        selector_action = gate_action_selector(gate_q.detach())
        if selector_action.ndim == 0:
            selector_action = selector_action.view(1)
        if selector_action.ndim != 1 or selector_action.shape[0] != gate_q.shape[0]:
            raise ValueError("ERR_PPO_AGENT_GATE_SHAPE: selected gate action must be [batch]")
        selector_action = selector_action.to(device=gate_q.device, dtype=torch.long)
    model_action = None
    if model_gate_action is not None:
        if model_gate_action.ndim != 1 or model_gate_action.shape[0] != gate_q.shape[0]:
            raise ValueError("ERR_PPO_AGENT_GATE_SHAPE: model gate action must be [batch]")
        model_action = model_gate_action.to(device=gate_q.device, dtype=torch.long)
    if gate_q.shape[1] == 2:
        q_gap = gate_q[:, 1] - gate_q[:, 0]
        selected_action = selector_action if selector_action is not None else model_action
        if selected_action is None:
            gate_action = torch.where(
                q_gap >= float(q_gap_threshold),
                torch.ones_like(q_gap, dtype=torch.long),
                torch.zeros_like(q_gap, dtype=torch.long),
            )
        else:
            gate_action = torch.where(
                selected_action > 0,
                torch.ones_like(selected_action, dtype=torch.long),
                torch.zeros_like(selected_action, dtype=torch.long),
            )
        return {
            "gate_action": int(gate_action.squeeze(0).detach().cpu()),
            "rebalance_intensity": _selected_intensity(gate_action, rebalance_intensity),
            "q_hold": float(gate_q[0, 0].detach().cpu()),
            "q_rebalance": float(gate_q[0, 1].detach().cpu()),
            "q_gap": float(q_gap.squeeze(0).detach().cpu()),
        }
    values = _rebalance_value_tensor(gate_q, rebalance_values)
    if selector_action is not None:
        action = selector_action
        action_index = int(action.squeeze(0).detach().cpu())
        intensity = float(values[action_index].detach().cpu())
    elif rebalance_intensity is not None:
        intensity = _unit_interval_float("rebalance_intensity", rebalance_intensity)
        action_index = _nearest_action_index(values, intensity)
    else:
        action = torch.argmax(gate_q, dim=1)
        action_index = int(action.squeeze(0).detach().cpu())
        intensity = float(values[action_index].detach().cpu())
    return {
        "gate_action": 0 if intensity <= 0.0 else 1,
        "gate_action_index": action_index,
        "rebalance_intensity": intensity,
        "q_hold": float(gate_q[0, 0].detach().cpu()),
        "q_rebalance": float(gate_q[0, -1].detach().cpu()),
        "q_gap": float((gate_q[0, -1] - gate_q[0, 0]).detach().cpu()),
    }


def _selected_intensity(gate_action: torch.Tensor, rebalance_intensity: float | None) -> float:
    action = int(gate_action.squeeze(0).detach().cpu())
    if action == 0:
        return 0.0
    if rebalance_intensity is None:
        return float(action)
    return _unit_interval_float("rebalance_intensity", rebalance_intensity)


def _rebalance_value_tensor(gate_q: torch.Tensor, rebalance_values: Any) -> torch.Tensor:
    if rebalance_values is None:
        return torch.linspace(0.0, 1.0, steps=gate_q.shape[1], dtype=gate_q.dtype, device=gate_q.device)
    if isinstance(rebalance_values, torch.Tensor):
        values = rebalance_values.to(device=gate_q.device, dtype=gate_q.dtype).view(-1)
    else:
        values = torch.as_tensor(rebalance_values, dtype=gate_q.dtype, device=gate_q.device).view(-1)
    if values.shape[0] != gate_q.shape[1]:
        raise ValueError("ERR_PPO_AGENT_GATE_SHAPE: rebalance_values must match gate_q action count")
    if not torch.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("ERR_PPO_AGENT_GATE_SHAPE: rebalance_values must be finite in [0,1]")
    return values


def _nearest_action_index(values: torch.Tensor, intensity: float) -> int:
    target = torch.tensor(float(intensity), dtype=values.dtype, device=values.device)
    return int(torch.argmin(torch.abs(values - target)).detach().cpu())


def _gate_action(action_info: Mapping[str, Any]) -> int:
    value = action_info.get("gate_action", action_info.get("rebalance_action", action_info.get("rebalance", 1)))
    return 1 if int(value) != 0 else 0


def _rebalance_intensity(action_info: Mapping[str, Any], gate_action: int) -> float:
    default = 1.0 if gate_action else 0.0
    value = action_info.get("rebalance_intensity", action_info.get("rho", default))
    return _unit_interval_float("rebalance_intensity", value)


def _env_action_weights(
    observation: Mapping[str, Any],
    action_info: Mapping[str, Any],
    gate_action: int,
) -> np.ndarray:
    candidate = np.asarray(action_info["candidate_weights"], dtype=np.float32)
    if gate_action != 0:
        return candidate
    current = observation.get("current_weights")
    if current is None:
        return candidate
    current_weights = np.asarray(current, dtype=np.float32)
    if current_weights.shape != candidate.shape:
        return candidate
    return current_weights


def _rollout_features(action_info: Mapping[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
    features = _feature_payload(info.get("uncertainty_features"), "env_uncertainty_features")
    features.update(_feature_payload(action_info.get("uncertainty_features"), "policy_uncertainty_features"))
    for key in (
        "q_hold",
        "q_rebalance",
        "q_gap",
        "gate_action_index",
        "raw_rebalance_intensity",
        "estimated_turnover",
        "estimated_cost",
        "realized_turnover",
        "realized_cost",
    ):
        if key in action_info and action_info[key] is not None:
            features[key] = action_info[key]
        elif key in info and info[key] is not None:
            features[key] = info[key]
    if "realized_turnover" not in features:
        for key in ("turnover", "actual_turnover"):
            if key in info and info[key] is not None:
                features["realized_turnover"] = info[key]
                break
    if "realized_cost" not in features:
        for key in ("transaction_cost", "total_transaction_cost", "cost"):
            if key in info and info[key] is not None:
                features["realized_cost"] = info[key]
                break
    return features


def _feature_payload(value: Any, vector_key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.size == 0:
        return {}
    return {vector_key: array}


def _execution_price(env: Any, info: Mapping[str, Any]) -> str:
    if "execution_price" in info:
        return str(info["execution_price"])
    config = getattr(env, "execution_config", None)
    if isinstance(config, Mapping):
        return str(config.get("execution_price", "next_open"))
    return "next_open"


def _delayed_action_execution(env: Any) -> bool:
    config = getattr(env, "execution_config", None)
    return bool(config.get("delayed_action_execution", False)) if isinstance(config, Mapping) else False


def _observation_date(observation: Mapping[str, Any]) -> pd.Timestamp:
    value = observation.get("decision_date", pd.Timestamp("1970-01-01"))
    return pd.Timestamp(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _finite_float(name: str, value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"ERR_PPO_AGENT_CONFIG_INVALID: {name}")
    return result


def _positive_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"ERR_PPO_AGENT_CONFIG_INVALID: {name}")
    return result


def _non_negative_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"ERR_PPO_AGENT_CONFIG_INVALID: {name}")
    return result


def _unit_interval_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"ERR_PPO_AGENT_CONFIG_INVALID: {name}")
    return result


def _positive_int(name: str, value: Any) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"ERR_PPO_AGENT_CONFIG_INVALID: {name}")
    return result


__all__ = ["PPOAgent", "PPOAgentConfig"]
