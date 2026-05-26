from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.risk_aware_graph_transformer import RiskAwareGraphTransformer


@dataclass(frozen=True)
class ConstraintBudget:
    average_turnover_per_step: float = 0.20
    average_cost_per_step: float = 0.001
    cvar_loss: float = 0.02
    drawdown: float = 0.10


@dataclass(frozen=True)
class ConstraintMultipliers:
    turnover: float = 2.0
    cost: float = 10.0
    cvar: float = 0.35
    drawdown: float = 0.25


@dataclass(frozen=True)
class ConstrainedActorCriticConfig:
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 32
    epochs: int = 1
    max_gradient_updates_per_epoch: int | None = None
    entropy_coef: float = 1.0e-3
    critic_coef: float = 0.20
    cost_rate: float = 0.001
    budget: ConstraintBudget = ConstraintBudget()
    multipliers: ConstraintMultipliers = ConstraintMultipliers()


class ConstrainedActorCriticAgent:
    def __init__(
        self,
        model: RiskAwareGraphTransformer,
        *,
        config: ConstrainedActorCriticConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )

    def train_offline(
        self,
        batch: Any,
        validation_batch: Any | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        env_steps = 0
        gradient_updates = 0
        max_updates = self.config.max_gradient_updates_per_epoch
        for epoch in range(max(1, int(self.config.epochs))):
            epoch_losses: list[float] = []
            epoch_rewards: list[float] = []
            updates_this_epoch = 0
            for mini_slice in _iter_minibatches(batch, self.config.batch_size):
                if max_updates is not None and updates_this_epoch >= int(max_updates):
                    break
                stats = self._update_batch(batch, mini_slice)
                gradient_updates += 1
                updates_this_epoch += 1
                env_steps += int(stats["sample_count"])
                epoch_losses.append(float(stats["loss"]))
                epoch_rewards.append(float(stats["objective"]))
            validation_metric = self.evaluate(validation_batch or batch)
            rows.append(
                {
                    "epoch": int(epoch),
                    "step": int(epoch + 1),
                    "env_steps": int(env_steps),
                    "gradient_updates": int(gradient_updates),
                    "train_reward": float(np.mean(epoch_rewards)) if epoch_rewards else np.nan,
                    "validation_metric": float(validation_metric),
                    "loss": float(np.mean(epoch_losses)) if epoch_losses else np.nan,
                    "status": "completed" if updates_this_epoch > 0 else "skipped_no_batch",
                }
            )
        history = pd.DataFrame(rows)
        return history, {
            "env_steps": int(env_steps),
            "gradient_updates": int(gradient_updates),
            "best_validation_metric": _best_validation_metric(history),
        }

    @torch.no_grad()
    def select_action(
        self,
        market_image: np.ndarray,
        current_weights: np.ndarray,
        availability_mask: np.ndarray,
    ) -> dict[str, Any]:
        self.model.eval()
        image = torch.as_tensor(market_image, dtype=torch.float32, device=self.device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        current = torch.as_tensor(current_weights, dtype=torch.float32, device=self.device).reshape(1, -1)
        mask = torch.as_tensor(availability_mask, dtype=torch.bool, device=self.device).reshape(1, -1)
        output = self.model(image, current, mask)
        candidate = output.candidate_weights.squeeze(0).detach().cpu().numpy()
        return {
            "candidate_weights": candidate,
            "value_return": float(output.value_return.squeeze(0).detach().cpu()),
            "value_cost": float(output.value_cost.squeeze(0).detach().cpu()),
            "value_drawdown": float(output.value_drawdown.squeeze(0).detach().cpu()),
            "value_cvar_loss": float(output.value_cvar_loss.squeeze(0).detach().cpu()),
            "graph_density": float(output.graph_density.squeeze(0).detach().cpu()),
            "mean_abs_correlation": float(output.mean_abs_correlation.squeeze(0).detach().cpu()),
        }

    @torch.no_grad()
    def evaluate(self, batch: Any) -> float:
        self.model.eval()
        output = self.model(batch.market_image, batch.current_weights, batch.availability_mask)
        metrics = self._portfolio_metrics(output.candidate_weights, batch.current_weights, batch.future_returns)
        return float(metrics["objective"].mean().detach().cpu())

    def _update_batch(self, batch: Any, mini_slice: slice) -> dict[str, float]:
        self.model.train()
        market_image = batch.market_image[mini_slice]
        current = batch.current_weights[mini_slice]
        mask = batch.availability_mask[mini_slice]
        future_returns = batch.future_returns[mini_slice]
        output = self.model(market_image, current, mask)
        metrics = self._portfolio_metrics(output.candidate_weights, current, future_returns)
        entropy = _portfolio_entropy(output.candidate_weights, mask).mean()
        critic_loss = (
            F.mse_loss(output.value_return, metrics["portfolio_return"].detach())
            + F.mse_loss(output.value_cost, metrics["estimated_cost"].detach())
            + F.mse_loss(output.value_drawdown, metrics["drawdown_loss"].detach())
            + F.mse_loss(output.value_cvar_loss, metrics["cvar_loss"].detach())
        )
        actor_loss = -metrics["objective"].mean()
        loss = actor_loss + float(self.config.critic_coef) * critic_loss - float(self.config.entropy_coef) * entropy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "objective": float(metrics["objective"].mean().detach().cpu()),
            "sample_count": float(market_image.shape[0]),
        }

    def _portfolio_metrics(
        self,
        candidate: torch.Tensor,
        current: torch.Tensor,
        future_returns: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        portfolio_return = (candidate * future_returns).sum(dim=-1)
        turnover = 0.5 * torch.abs(candidate - current).sum(dim=-1)
        estimated_cost = turnover * float(self.config.cost_rate)
        cvar_loss = torch.relu(-portfolio_return)
        drawdown_loss = torch.relu(-portfolio_return)
        budget = self.config.budget
        multipliers = self.config.multipliers
        turnover_violation = torch.relu(turnover - float(budget.average_turnover_per_step))
        cost_violation = torch.relu(estimated_cost - float(budget.average_cost_per_step))
        cvar_violation = torch.relu(cvar_loss - float(budget.cvar_loss))
        drawdown_violation = torch.relu(drawdown_loss - float(budget.drawdown))
        objective = (
            portfolio_return
            - float(multipliers.turnover) * turnover_violation
            - float(multipliers.cost) * cost_violation
            - float(multipliers.cvar) * cvar_violation
            - float(multipliers.drawdown) * drawdown_violation
        )
        return {
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "estimated_cost": estimated_cost,
            "cvar_loss": cvar_loss,
            "drawdown_loss": drawdown_loss,
            "objective": objective,
        }


def agent_config_from_mapping(config: Mapping[str, Any], *, section: Mapping[str, Any]) -> ConstrainedActorCriticConfig:
    native = _mapping(_mapping(config.get("baselines")).get("native_rl"))
    training = _mapping(config.get("training"))
    optimizer = _mapping(config.get("optimizer"))
    cost_model = _mapping(config.get("cost_model"))
    cost_rate = float(cost_model.get("proportional_cost", 0.0) or 0.0) + float(cost_model.get("slippage", 0.0) or 0.0)
    return ConstrainedActorCriticConfig(
        learning_rate=float(section.get("learning_rate", optimizer.get("learning_rate", 3.0e-4))),
        weight_decay=float(section.get("weight_decay", optimizer.get("weight_decay", 1.0e-4))),
        batch_size=max(1, int(section.get("batch_size", training.get("batch_size", 32)))),
        epochs=max(1, int(native.get("epochs", training.get("epochs", 1)))),
        max_gradient_updates_per_epoch=_optional_int(native.get("max_gradient_updates_per_epoch", training.get("max_gradient_updates_per_epoch"))),
        entropy_coef=float(section.get("entropy_coef", 1.0e-3)),
        critic_coef=float(section.get("critic_coef", 0.20)),
        cost_rate=cost_rate,
        budget=ConstraintBudget(
            average_turnover_per_step=float(section.get("average_turnover_per_step_budget", 0.20)),
            average_cost_per_step=float(section.get("average_cost_per_step_budget", 0.001)),
            cvar_loss=float(section.get("cvar_loss_budget", 0.02)),
            drawdown=float(section.get("drawdown_budget", 0.10)),
        ),
        multipliers=ConstraintMultipliers(
            turnover=float(section.get("lambda_turnover", 2.0)),
            cost=float(section.get("lambda_cost", 10.0)),
            cvar=float(section.get("lambda_cvar", 0.35)),
            drawdown=float(section.get("lambda_drawdown", section.get("lambda_dd", 0.25))),
        ),
    )


def _portfolio_entropy(weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1.0e-12)
    entropy = -(safe * safe.log()).sum(dim=-1)
    denom = mask.sum(dim=-1).clamp_min(1).to(dtype=weights.dtype).log().clamp_min(1.0)
    return entropy / denom


def _iter_minibatches(batch: Any, batch_size: int):
    size = int(getattr(batch, "size"))
    for start in range(0, size, int(batch_size)):
        stop = min(start + int(batch_size), size)
        yield slice(start, stop)


def _best_validation_metric(history: pd.DataFrame) -> float | None:
    if history.empty or "validation_metric" not in history.columns:
        return None
    values = pd.to_numeric(history["validation_metric"], errors="coerce")
    finite = values[np.isfinite(values)]
    return None if finite.empty else float(finite.max())


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    result = int(value)
    return None if result <= 0 else result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "ConstrainedActorCriticAgent",
    "ConstrainedActorCriticConfig",
    "ConstraintBudget",
    "ConstraintMultipliers",
    "agent_config_from_mapping",
]
