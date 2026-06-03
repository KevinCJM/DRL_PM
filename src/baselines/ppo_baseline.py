from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .base_strategy import BaseStrategy
from .deep_training import (
    collect_training_batch,
    deep_baseline_training_config,
    execution_aligned_future_return_frame,
    iter_minibatches,
    training_summary,
)
from ..envs.state import DecisionMarketState, PortfolioAction, PortfolioState
from ..models.encoders import EncoderFactory
from ..models.ppo_actor import PPOActor


class PPOBaselineModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = dict(config)
        model_config = _mapping(config.get("model"))
        ppo_config = _mapping(config.get("ppo") or model_config.get("ppo"))
        self.n_assets = _resolve_positive_int(config, "n_assets")
        self.n_features = _resolve_positive_int(config, "n_features")
        self.window_size = _resolve_positive_int(config, "window_size")
        self.latent_dim = int(config.get("latent_dim", model_config.get("latent_dim", 256)))
        if self.latent_dim <= 0:
            raise ValueError("ERR_PPO_BASELINE_CONFIG_INVALID: latent_dim must be > 0")

        self.uses_fixed_mlp_state = _is_ppo_baseline_mlp(config)
        if self.uses_fixed_mlp_state:
            self.encoder = PPOBaselineMLPEncoder(
                self.n_features,
                self.window_size,
                self.n_assets,
                self.latent_dim,
                dropout=float(_encoder_config(config).get("dropout", model_config.get("dropout", 0.10))),
            )
        else:
            resolved_config = dict(config)
            resolved_config.update(
                {
                    "n_assets": self.n_assets,
                    "n_features": self.n_features,
                    "window_size": self.window_size,
                    "latent_dim": self.latent_dim,
                }
            )
            self.encoder = EncoderFactory.create(resolved_config)
        ppo_hidden_dims = ppo_config.get("hidden_dims")
        self.actor = PPOActor(
            self.latent_dim,
            self.n_assets,
            min_alpha=float(ppo_config.get("min_alpha", ppo_config.get("actor_min_alpha", 1.0e-3))),
            hidden_dims=ppo_config.get("actor_hidden_dims", ppo_hidden_dims),
        )

    def forward(
        self,
        market_image: torch.Tensor,
        available_mask: torch.Tensor,
        current_weights: torch.Tensor | None = None,
        estimated_turnover: torch.Tensor | None = None,
        estimated_cost: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        if current_weights is None:
            current_weights = torch.zeros(
                market_image.shape[0],
                self.n_assets,
                device=market_image.device,
                dtype=market_image.dtype,
            )
        latent = (
            self.encoder(market_image, current_weights, available_mask)
            if self.uses_fixed_mlp_state
            else self.encoder(market_image)
        )
        dist = self.actor.get_distribution(latent, available_mask)
        candidate_weights = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(candidate_weights)
        estimated_turnover = (
            estimated_turnover
            if estimated_turnover is not None
            else 0.5 * torch.sum(torch.abs(candidate_weights - current_weights), dim=1, keepdim=True)
        )
        gate_action = (estimated_turnover > _proxy_rebalance_turnover_threshold_tensor(self.config, market_image)).long().view(-1)
        return {
            "candidate_weights": candidate_weights,
            "log_prob": log_prob,
            "gate_action": gate_action,
            "estimated_turnover": estimated_turnover,
            "estimated_cost": (
                estimated_cost
                if estimated_cost is not None
                else torch.zeros(market_image.shape[0], 1, device=market_image.device, dtype=market_image.dtype)
            ),
            "latent": latent,
        }


class PPOBaselineStrategy(BaseStrategy):
    strategy_name = "ppo_baseline"
    default_encoder_type = "ppo_mlp"
    fit_required = True

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.device = _resolve_device(self.config.get("device", "cpu"))
        resolved_config = self._resolve_model_config(config)
        self.model = PPOBaselineModel(resolved_config)
        self.model.to(self.device)
        self.model.eval()
        self.fitted_prior_weights: np.ndarray | None = None
        training_config = deep_baseline_training_config(self.config)
        self.prior_blend_weight = training_config.prior_blend_weight
        self.training_result: dict[str, Any] = training_summary(
            "not_started",
            current_weight_mode=training_config.current_weight_mode,
        )

    def fit(self, train_data: Any | None = None, validation_data: Any | None = None) -> PPOBaselineStrategy:
        super().fit(train_data, validation_data)
        self.fitted_prior_weights = _fit_return_prior_weights(train_data, self.model.n_assets)
        self.training_result = _train_ppo_baseline_model(self.model, train_data, self.config, self.device)
        return self

    def compute_target_weights(
        self,
        decision_market_state: DecisionMarketState,
        portfolio_state: PortfolioState,
    ) -> PortfolioAction:
        state = self.validate_decision_market_state(decision_market_state)
        portfolio = self.validate_portfolio_state(portfolio_state)

        market_image = torch.as_tensor(state.market_image, dtype=torch.float32, device=self.device).unsqueeze(0)
        available_mask = torch.as_tensor(
            state.available_mask_at_decision,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        current_weights = torch.as_tensor(
            portfolio_state.current_weights,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(market_image, available_mask, current_weights, deterministic=True)

        target_weights = outputs["candidate_weights"].squeeze(0).detach().cpu().numpy()
        target_weights = _blend_with_fitted_prior(
            target_weights,
            self.fitted_prior_weights,
            state.available_mask_at_decision,
            self.prior_blend_weight,
        )
        target_weights = _enforce_masked_simplex(target_weights, state.available_mask_at_decision)
        log_prob = float(outputs["log_prob"].squeeze(0).detach().cpu().item())
        from .eiie import _continuous_weight_rebalance_decision

        rebalance_decision = _continuous_weight_rebalance_decision(
            self.config,
            self.strategy_name,
            portfolio,
            target_weights,
            getattr(self, "decision_context", {}),
        )

        return self.validate_portfolio_action(
            PortfolioAction(
                target_weights=target_weights,
                rebalance_action=rebalance_decision["rebalance_action"],
                rebalance_intensity=rebalance_decision["rebalance_intensity"],
                action_info={
                    "strategy": self.strategy_name,
                    "log_prob": log_prob,
                    "scheduler_controlled": True,
                    "constraint_controlled": True,
                    "prior_blend_weight": self.prior_blend_weight,
                    **rebalance_decision["action_info"],
                },
            )
        )

    def _resolve_model_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(config)
        model_config = dict(_mapping(resolved.get("model")))
        encoder_config = dict(_mapping(resolved.get("encoder") or model_config.get("encoder")))
        encoder_config["type"] = self.default_encoder_type
        resolved["encoder"] = encoder_config
        return resolved


class PPOBaselineMLPEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        n_assets: int,
        latent_dim: int,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.n_assets = int(n_assets)
        self.latent_dim = int(latent_dim)
        self.input_dim = self.n_features * self.window_size * self.n_assets + 2 * self.n_assets
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
        )

    def forward(
        self,
        market_image: torch.Tensor,
        current_weights: torch.Tensor,
        available_mask: torch.Tensor,
    ) -> torch.Tensor:
        if market_image.ndim != 4:
            raise ValueError("ERR_MODEL_INPUT_SHAPE: expected [batch,n_features,window_size,n_assets]")
        batch_size, n_features, window_size, n_assets = market_image.shape
        if n_features != self.n_features or window_size != self.window_size or n_assets != self.n_assets:
            raise ValueError("ERR_MODEL_INPUT_SHAPE: PPO baseline market_image")
        if current_weights.shape != (batch_size, self.n_assets):
            raise ValueError("ERR_MODEL_INPUT_SHAPE: PPO baseline current_weights")
        if available_mask.shape != (batch_size, self.n_assets):
            raise ValueError("ERR_MODEL_INPUT_SHAPE: PPO baseline availability_mask")
        state_input = torch.cat(
            [
                market_image.reshape(batch_size, -1),
                current_weights.to(dtype=market_image.dtype),
                available_mask.to(dtype=market_image.dtype),
            ],
            dim=1,
        )
        return self.fc(state_input)


def _enforce_masked_simplex(weights: np.ndarray, available_mask: np.ndarray) -> np.ndarray:
    available = np.asarray(available_mask, dtype=bool)
    result = np.asarray(weights, dtype=float).copy()
    if result.ndim != 1 or result.shape != available.shape:
        raise ValueError("ERR_STRATEGY_ACTION_CONTRACT: target_weights shape")
    result[~available] = 0.0
    total = float(result.sum())
    if not np.isfinite(result).all() or total <= 0.0:
        raise ValueError("ERR_STRATEGY_ACTION_CONTRACT: target_weights simplex")
    result = result / total
    result[~available] = 0.0
    return result


def _fit_return_prior_weights(train_data: Any | None, n_assets: int) -> np.ndarray | None:
    if not isinstance(train_data, Mapping):
        return None
    frame = execution_aligned_future_return_frame(train_data, n_assets)
    if frame is None:
        return None
    if frame.empty:
        return None
    scores = frame.replace([np.inf, -np.inf], np.nan).mean(axis=0, skipna=True).to_numpy(dtype=float)
    if scores.shape[0] != int(n_assets) or not np.isfinite(scores).any():
        return None
    finite_scores = scores[np.isfinite(scores)]
    fill_value = float(np.min(finite_scores)) if finite_scores.size else 0.0
    scores = np.nan_to_num(scores, nan=fill_value, posinf=fill_value, neginf=fill_value)
    scores = scores - float(np.max(scores))
    weights = np.exp(scores)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return None
    return weights / total


def _blend_with_fitted_prior(
    weights: np.ndarray,
    fitted_prior_weights: np.ndarray | None,
    available_mask: np.ndarray,
    prior_blend_weight: float = 0.5,
) -> np.ndarray:
    blend = float(np.clip(prior_blend_weight, 0.0, 1.0))
    if fitted_prior_weights is None or blend <= 0.0:
        return _enforce_masked_simplex(weights, available_mask)
    prior = _enforce_masked_simplex(fitted_prior_weights, available_mask)
    current = _enforce_masked_simplex(weights, available_mask)
    return (1.0 - blend) * current + blend * prior


def _resolve_positive_int(config: Mapping[str, Any], key: str) -> int:
    model_config = _mapping(config.get("model"))
    env_config = _mapping(config.get("env"))
    feature_matrix_config = _mapping(config.get("feature_matrix"))
    if key in config:
        value = config[key]
    elif key in model_config:
        value = model_config[key]
    elif key in env_config:
        value = env_config[key]
    elif key in feature_matrix_config:
        value = feature_matrix_config[key]
    else:
        raise KeyError(key)
    result = int(value)
    if result <= 0:
        raise ValueError(f"ERR_PPO_BASELINE_CONFIG_INVALID: {key} must be > 0")
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _encoder_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model_config = _mapping(config.get("model"))
    return _mapping(config.get("encoder") or model_config.get("encoder"))


def _is_ppo_baseline_mlp(config: Mapping[str, Any]) -> bool:
    model_config = _mapping(config.get("model"))
    encoder_config = _encoder_config(config)
    encoder_type = str(encoder_config.get("type", model_config.get("default_encoder", "ppo_mlp"))).lower()
    return encoder_type in {"ppo_mlp", "baseline_mlp", "mlp"}


def _resolve_device(value: Any) -> torch.device:
    if isinstance(value, Mapping):
        value = value.get("mode", "cpu")
    mode = str(value)
    if mode == "auto":
        mode = "cpu"
    return torch.device(mode)


def _proxy_rebalance_turnover_threshold_tensor(config: Mapping[str, Any], reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(
        _proxy_rebalance_turnover_threshold(config),
        dtype=reference.dtype,
        device=reference.device,
    )


def _proxy_rebalance_turnover_threshold(config: Mapping[str, Any]) -> float:
    for section_name in ("ppo_baseline", "cnn_ppo_baseline", "ppo_proxy", "cnn_ppo_proxy"):
        section = _mapping(config.get(section_name))
        for key in ("rebalance_turnover_threshold", "turnover_gate_threshold", "min_rebalance_turnover"):
            if section.get(key) is not None:
                return _non_negative_float(section[key])
    activity = _mapping(config.get("execution_activity"))
    for key in ("model_rebalance_turnover_threshold", "rebalance_turnover_threshold", "turnover_gate_threshold"):
        if activity.get(key) is not None:
            return _non_negative_float(activity[key])
    rebalance = _mapping(config.get("rebalance"))
    if str(rebalance.get("mode", "")) == "threshold_turnover":
        return _non_negative_float(rebalance.get("threshold_turnover", 0.0))
    return 0.0


def _non_negative_float(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("ERR_PPO_BASELINE_REBALANCE_THRESHOLD_INVALID")
    return result


def _train_ppo_baseline_model(
    model: PPOBaselineModel,
    train_data: Any | None,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    training_config = deep_baseline_training_config(config)
    if not training_config.enabled or training_config.epochs <= 0:
        return training_summary("disabled", current_weight_mode=training_config.current_weight_mode)
    batch = collect_training_batch(
        train_data,
        n_features=model.n_features,
        window_size=model.window_size,
        n_assets=model.n_assets,
        device=device,
        max_samples=training_config.max_samples,
    )
    if batch is None:
        return training_summary("skipped_no_samples", current_weight_mode=training_config.current_weight_mode)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    last_loss: torch.Tensor | None = None
    gradient_updates = 0
    skipped_no_gradient_minibatches = 0
    for _ in range(training_config.epochs):
        for indices in iter_minibatches(batch, training_config.batch_size):
            outputs = model(
                batch.market_image[indices],
                batch.availability_mask[indices],
                batch.current_weights[indices],
                deterministic=True,
            )
            weights = outputs["candidate_weights"]
            gross_return = (weights * batch.future_returns[indices]).sum(dim=1, keepdim=True)
            turnover = 0.5 * torch.sum(torch.abs(weights - batch.current_weights[indices]), dim=1, keepdim=True)
            objective = gross_return - float(training_config.turnover_penalty) * turnover
            loss = -objective.mean()
            if not loss.requires_grad:
                skipped_no_gradient_minibatches += 1
                last_loss = loss.detach()
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            last_loss = loss.detach()
            gradient_updates += 1
    model.eval()
    summary = training_summary(
        "completed",
        samples=batch.size,
        loss=None if last_loss is None else float(last_loss.cpu()),
        current_weight_mode=training_config.current_weight_mode,
    )
    summary["gradient_updates"] = int(gradient_updates)
    summary["skipped_no_gradient_minibatches"] = int(skipped_no_gradient_minibatches)
    return summary


__all__ = [
    "PPOBaselineMLPEncoder",
    "PPOBaselineModel",
    "PPOBaselineStrategy",
]
