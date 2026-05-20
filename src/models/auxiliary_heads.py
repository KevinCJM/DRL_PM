from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TASKS = (
    "return",
    "volatility",
    "trend",
    "rank",
    "downside",
    "max_drawdown",
    "cvar",
    "covariance",
    "reconstruction",
)

LOSS_WEIGHTS = {
    "return": 0.10,
    "trend": 0.05,
    "volatility": 0.05,
    "rank": 0.10,
    "downside_volatility": 0.05,
    "max_drawdown": 0.05,
    "cvar": 0.05,
    "covariance": 0.03,
    "reconstruction": 0.05,
    "representation_regularization": 0.005,
}

TARGET_ALIASES = {
    "rank": "future_cross_sectional_rank",
    "downside_volatility": "future_downside_volatility",
    "max_drawdown": "future_max_drawdown",
    "cvar": "future_CVaR",
    "covariance": "future_correlation_or_covariance",
    "reconstruction": "masked_feature_reconstruction",
}


class AuxiliaryHeads(nn.Module):
    def __init__(self, latent_dim: int, n_assets: int, n_features: int, window_size: int, config: Mapping[str, Any]):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_assets = int(n_assets)
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.config = dict(config)
        if min(self.latent_dim, self.n_assets, self.n_features, self.window_size) <= 0:
            raise ValueError("ERR_AUXILIARY_HEAD_CONFIG_INVALID: dimensions must be > 0")

        self.tasks = _resolve_tasks(self.config.get("tasks", []))
        self.return_horizons = _int_sequence(self.config.get("future_return_horizons", [5, 20]))
        self.volatility_horizons = _int_sequence(self.config.get("future_volatility_horizons", [20]))
        self.trend_horizons = _int_sequence(self.config.get("future_trend_horizons", [10]))
        self.volatility_eps = float(self.config.get("volatility_eps", 1.0e-8))
        hidden_dims = _int_sequence(self.config.get("hidden_dims", [128]))

        self.heads = nn.ModuleDict()
        if "return" in self.tasks:
            for horizon in self.return_horizons:
                self.heads[f"return_{horizon}"] = _AssetScalarHead(self.latent_dim, self.n_assets, hidden_dims)
        if "volatility" in self.tasks:
            for horizon in self.volatility_horizons:
                self.heads[f"volatility_{horizon}"] = _AssetScalarHead(
                    self.latent_dim,
                    self.n_assets,
                    hidden_dims,
                    positive=True,
                )
        if "trend" in self.tasks:
            for horizon in self.trend_horizons:
                self.heads[f"trend_{horizon}"] = _AssetScalarHead(self.latent_dim, self.n_assets, hidden_dims)
        if "rank" in self.tasks:
            self.heads["rank"] = _AssetScalarHead(self.latent_dim, self.n_assets, hidden_dims)
        if "downside" in self.tasks:
            self.heads["downside_volatility"] = _AssetScalarHead(
                self.latent_dim,
                self.n_assets,
                hidden_dims,
                positive=True,
            )
        if "max_drawdown" in self.tasks:
            self.heads["max_drawdown"] = _AssetScalarHead(
                self.latent_dim,
                self.n_assets,
                hidden_dims,
                positive=True,
            )
        if "cvar" in self.tasks:
            self.heads["cvar"] = _AssetScalarHead(self.latent_dim, self.n_assets, hidden_dims, positive=True)
        if "covariance" in self.tasks:
            self.heads["covariance"] = _AssetScalarHead(self.latent_dim, self.n_assets, hidden_dims)
        if "reconstruction" in self.tasks:
            self.heads["reconstruction"] = _ReconstructionHead(
                self.latent_dim,
                self.n_assets,
                self.n_features,
                self.window_size,
                hidden_dims,
            )

    def forward(self, latent: torch.Tensor) -> Mapping[str, torch.Tensor]:
        _validate_latent(latent, self.latent_dim, self.n_assets)
        return {name: head(latent) for name, head in self.heads.items()}

    def compute_loss(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        latent: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        total = latent.sum() * 0.0
        for name, prediction in outputs.items():
            target = _target_for(name, targets)
            if target is None:
                continue
            target_tensor = target.to(device=prediction.device, dtype=prediction.dtype)
            task_loss = self._task_loss(name, prediction, target_tensor, availability_mask)
            losses[name] = task_loss
            total = total + _loss_weight(name) * task_loss

        representation_loss = self.get_representation_loss(latent, availability_mask)
        losses["representation_regularization"] = representation_loss
        losses["total"] = total + LOSS_WEIGHTS["representation_regularization"] * representation_loss
        return losses

    def _task_loss(
        self,
        name: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        availability_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"ERR_AUXILIARY_TARGET_SHAPE_MISMATCH: {name}")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError(f"ERR_AUXILIARY_TARGET_NON_FINITE: {name}")
        if name.startswith("trend_"):
            raw_loss = F.binary_cross_entropy_with_logits(prediction, target.to(dtype=prediction.dtype), reduction="none")
        elif name.startswith("volatility_"):
            raw_loss = F.huber_loss(
                torch.log(prediction + self.volatility_eps),
                torch.log(target.clamp_min(0.0) + self.volatility_eps),
                reduction="none",
            )
        elif name in {"rank", "covariance", "reconstruction"}:
            raw_loss = F.mse_loss(prediction, target, reduction="none")
        else:
            raw_loss = F.huber_loss(prediction, target, reduction="none")
        return _masked_mean(raw_loss, availability_mask)

    def get_representation_loss(
        self,
        latent: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_latent(latent, self.latent_dim, self.n_assets)
        squared = latent ** 2
        if latent.ndim == 3 and availability_mask is not None:
            if availability_mask.shape != latent.shape[:2]:
                raise ValueError("ERR_AUXILIARY_MASK_SHAPE_MISMATCH: availability_mask must be [batch,n_assets]")
            mask = availability_mask.to(device=latent.device, dtype=torch.bool).unsqueeze(-1)
            if not mask.any():
                return squared.sum() * 0.0
            return squared.masked_select(mask.expand_as(squared)).mean()
        return squared.mean()


class _AssetScalarHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_assets: int,
        hidden_dims: Sequence[int],
        positive: bool = False,
    ):
        super().__init__()
        self.n_assets = int(n_assets)
        self.positive = bool(positive)
        self.global_net = _mlp(int(latent_dim), hidden_dims, self.n_assets)
        self.asset_net = _mlp(int(latent_dim), hidden_dims, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 2:
            output = self.global_net(latent)
        elif latent.ndim == 3:
            if latent.shape[1] != self.n_assets:
                raise ValueError("ERR_AUXILIARY_HEAD_SHAPE_MISMATCH: latent asset dimension mismatch")
            output = self.asset_net(latent).squeeze(-1)
        else:
            raise ValueError("ERR_AUXILIARY_HEAD_SHAPE_MISMATCH: latent must be [batch,latent_dim] or [batch,n_assets,latent_dim]")
        return F.softplus(output) if self.positive else output


class _ReconstructionHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_assets: int,
        n_features: int,
        window_size: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.n_assets = int(n_assets)
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.global_net = _mlp(int(latent_dim), hidden_dims, self.n_features * self.window_size * self.n_assets)
        self.asset_net = _mlp(int(latent_dim), hidden_dims, self.n_features * self.window_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 2:
            return self.global_net(latent).reshape(-1, self.n_features, self.window_size, self.n_assets)
        if latent.ndim == 3:
            if latent.shape[1] != self.n_assets:
                raise ValueError("ERR_AUXILIARY_HEAD_SHAPE_MISMATCH: latent asset dimension mismatch")
            output = self.asset_net(latent).reshape(-1, self.n_assets, self.n_features, self.window_size)
            return output.permute(0, 2, 3, 1).contiguous()
        raise ValueError("ERR_AUXILIARY_HEAD_SHAPE_MISMATCH: latent must be [batch,latent_dim] or [batch,n_assets,latent_dim]")


def _mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    dims = [int(input_dim), *[int(dim) for dim in hidden_dims]]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
        layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
    layers.append(nn.Linear(dims[-1], int(output_dim)))
    return nn.Sequential(*layers)


def _resolve_tasks(raw_tasks: Any) -> tuple[str, ...]:
    if raw_tasks == "all":
        return DEFAULT_TASKS
    tasks = tuple(str(task) for task in (raw_tasks or ()))
    if "all" in tasks:
        return DEFAULT_TASKS
    normalized = tuple("reconstruction" if task == "masked_reconstruction" else task for task in tasks)
    unknown = sorted(set(normalized) - set(DEFAULT_TASKS))
    if unknown:
        raise ValueError(f"ERR_AUXILIARY_HEAD_CONFIG_INVALID: unknown tasks {unknown}")
    return normalized


def _int_sequence(value: Any) -> tuple[int, ...]:
    values = tuple(int(item) for item in value)
    if not values or any(item <= 0 for item in values):
        raise ValueError("ERR_AUXILIARY_HEAD_CONFIG_INVALID: sequence values must be positive")
    return values


def _validate_latent(latent: torch.Tensor, latent_dim: int, n_assets: int) -> None:
    if latent.ndim == 2 and latent.shape[1] == latent_dim:
        pass
    elif latent.ndim == 3 and latent.shape[1:] == (n_assets, latent_dim):
        pass
    else:
        raise ValueError("ERR_AUXILIARY_HEAD_SHAPE_MISMATCH: latent shape invalid")
    if not torch.isfinite(latent).all():
        raise ValueError("ERR_AUXILIARY_HEAD_NON_FINITE: latent contains NaN or Inf")


def _target_for(name: str, targets: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    aliases = [name]
    if name.startswith("return_"):
        aliases.append(f"future_log_return_{name.removeprefix('return_')}d")
    elif name.startswith("volatility_"):
        aliases.append(f"future_volatility_{name.removeprefix('volatility_')}d")
    elif name.startswith("trend_"):
        aliases.append(f"future_trend_{name.removeprefix('trend_')}d")
    elif name in TARGET_ALIASES:
        aliases.append(TARGET_ALIASES[name])
    for alias in aliases:
        if alias in targets:
            return targets[alias]
    return None


def _loss_weight(name: str) -> float:
    if name.startswith("return_"):
        return LOSS_WEIGHTS["return"]
    if name.startswith("volatility_"):
        return LOSS_WEIGHTS["volatility"]
    if name.startswith("trend_"):
        return LOSS_WEIGHTS["trend"]
    return LOSS_WEIGHTS.get(name, 1.0)


def _masked_mean(raw_loss: torch.Tensor, availability_mask: torch.Tensor | None) -> torch.Tensor:
    if availability_mask is None:
        return raw_loss.mean()
    mask = availability_mask.to(device=raw_loss.device, dtype=torch.bool)
    if raw_loss.ndim == 2:
        if mask.shape != raw_loss.shape:
            raise ValueError("ERR_AUXILIARY_MASK_SHAPE_MISMATCH: availability_mask must match [batch,n_assets]")
        if not mask.any():
            return raw_loss.sum() * 0.0
        return raw_loss.masked_select(mask).mean()
    if raw_loss.ndim == 4:
        if mask.shape != (raw_loss.shape[0], raw_loss.shape[3]):
            raise ValueError("ERR_AUXILIARY_MASK_SHAPE_MISMATCH: availability_mask must be [batch,n_assets]")
        expanded_mask = mask[:, None, None, :].expand_as(raw_loss)
        if not expanded_mask.any():
            return raw_loss.sum() * 0.0
        return raw_loss.masked_select(expanded_mask).mean()
    return raw_loss.mean()


__all__ = ["AuxiliaryHeads"]
