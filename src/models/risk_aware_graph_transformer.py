from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


RA_GT_RCPO_MODEL_NAME = "risk_aware_graph_transformer_constrained_actor_critic"
RA_GT_RCPO_ALGORITHM = RA_GT_RCPO_MODEL_NAME
RA_GT_RCPO_MODEL_EXTENSION_ID = "core13_v2_p16_ra_gt_rcpo_20260525"
RA_GT_RCPO_ABLATION_MODEL_NAMES = (
    "ra_gt_rcpo_no_graph",
    "ra_gt_rcpo_no_transformer",
    "ra_gt_rcpo_no_cvar_constraint",
    "ra_gt_rcpo_no_cost_constraint",
    "ra_gt_rcpo_no_turnover_constraint",
    "ra_gt_rcpo_mlp_actor_critic",
)
RA_GT_RCPO_MODEL_NAMES = (RA_GT_RCPO_MODEL_NAME, *RA_GT_RCPO_ABLATION_MODEL_NAMES)


@dataclass(frozen=True)
class RiskAwareGraphTransformerOutput:
    candidate_weights: torch.Tensor
    actor_logits: torch.Tensor
    rho_logits: torch.Tensor
    rho_probs: torch.Tensor
    rho_action_index: torch.Tensor
    rho: torch.Tensor
    rho_entropy: torch.Tensor
    rho_expected: torch.Tensor
    value_return: torch.Tensor
    value_cost: torch.Tensor
    value_drawdown: torch.Tensor
    value_cvar_loss: torch.Tensor
    graph_density: torch.Tensor
    mean_abs_correlation: torch.Tensor


class RiskAwareGraphTransformer(nn.Module):
    """Decision-visible graph/temporal encoder with actor and risk critic heads."""

    def __init__(
        self,
        *,
        n_features: int,
        window_size: int,
        n_assets: int,
        model_dim: int = 64,
        transformer_layers: int = 1,
        attention_heads: int = 2,
        dropout: float = 0.05,
        graph_edge_threshold: float = 0.10,
        use_graph: bool = True,
        use_transformer: bool = True,
        mlp_actor_critic: bool = False,
        rho_values: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.n_assets = int(n_assets)
        self.model_dim = int(model_dim)
        self.graph_edge_threshold = float(graph_edge_threshold)
        self.use_graph = bool(use_graph)
        self.use_transformer = bool(use_transformer)
        self.mlp_actor_critic = bool(mlp_actor_critic)
        resolved_rhos = _rho_values(rho_values)
        self.register_buffer("rho_values", torch.tensor(resolved_rhos, dtype=torch.float32), persistent=False)

        self.input_projection = nn.Linear(self.n_features, self.model_dim)
        self.time_position = nn.Parameter(torch.zeros(1, self.window_size, self.model_dim))
        if self.use_transformer:
            heads = _valid_attention_heads(self.model_dim, int(attention_heads))
            layer = nn.TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=heads,
                dim_feedforward=self.model_dim * 2,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(transformer_layers)))
        else:
            self.temporal_encoder = nn.Identity()

        self.graph_projection = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
        )
        self.mlp_projection = nn.Sequential(
            nn.Linear(self.n_features * self.window_size, self.model_dim),
            nn.LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.model_dim, 1),
        )
        self.rho_head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.model_dim, len(resolved_rhos)),
        )
        self.return_critic = _critic_head(self.model_dim)
        self.cost_critic = _critic_head(self.model_dim)
        self.drawdown_critic = _critic_head(self.model_dim)
        self.cvar_critic = _critic_head(self.model_dim)

    def forward(
        self,
        market_image: torch.Tensor,
        current_weights: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> RiskAwareGraphTransformerOutput:
        x = _market_image_4d(market_image, self.n_features, self.window_size, self.n_assets)
        mask = availability_mask.to(dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[-1] != self.n_assets:
            raise ValueError("ERR_RA_GT_RCPO_MASK_SHAPE")
        current_weights = current_weights.to(dtype=x.dtype)
        if current_weights.ndim != 2 or current_weights.shape[-1] != self.n_assets:
            raise ValueError("ERR_RA_GT_RCPO_CURRENT_WEIGHT_SHAPE")

        asset_tokens = self._asset_tokens(x)
        correlation = rolling_correlation_adjacency(x[:, 0, :, :], mask, threshold=self.graph_edge_threshold)
        adjacency = correlation.adjacency
        if self.use_graph and not self.mlp_actor_critic:
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            graph_message = torch.bmm(adjacency, asset_tokens) / degree
            asset_tokens = self.graph_projection(torch.cat([asset_tokens, graph_message], dim=-1))

        masked_tokens = asset_tokens * mask.unsqueeze(-1).to(asset_tokens.dtype)
        context = masked_tokens.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(asset_tokens.dtype)
        actor_input = torch.cat([asset_tokens, context.unsqueeze(1).expand(-1, self.n_assets, -1)], dim=-1)
        logits = self.actor_head(actor_input).squeeze(-1)
        candidate = masked_softmax(logits, mask)
        critic_context = context + torch.bmm(candidate.unsqueeze(1), asset_tokens).squeeze(1)
        rho_logits = self.rho_head(critic_context)
        rho_probs = torch.softmax(rho_logits, dim=-1)
        rho_action_index = torch.argmax(rho_probs, dim=-1)
        rho = self.rho_values.to(device=x.device, dtype=x.dtype)[rho_action_index]
        rho_entropy = -(rho_probs.clamp_min(1.0e-12) * rho_probs.clamp_min(1.0e-12).log()).sum(dim=-1)
        rho_expected = (rho_probs * self.rho_values.to(device=x.device, dtype=x.dtype).unsqueeze(0)).sum(dim=-1)
        return RiskAwareGraphTransformerOutput(
            candidate_weights=candidate,
            actor_logits=logits,
            rho_logits=rho_logits,
            rho_probs=rho_probs,
            rho_action_index=rho_action_index,
            rho=rho,
            rho_entropy=rho_entropy,
            rho_expected=rho_expected,
            value_return=self.return_critic(critic_context).squeeze(-1),
            value_cost=F.softplus(self.cost_critic(critic_context).squeeze(-1)),
            value_drawdown=F.softplus(self.drawdown_critic(critic_context).squeeze(-1)),
            value_cvar_loss=F.softplus(self.cvar_critic(critic_context).squeeze(-1)),
            graph_density=correlation.graph_density,
            mean_abs_correlation=correlation.mean_abs_correlation,
        )

    def _asset_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp_actor_critic:
            flat = x.permute(0, 3, 1, 2).reshape(x.shape[0], self.n_assets, self.n_features * self.window_size)
            return self.mlp_projection(flat)
        sequence = x.permute(0, 3, 2, 1).reshape(x.shape[0] * self.n_assets, self.window_size, self.n_features)
        embedded = self.input_projection(sequence) + self.time_position[:, : self.window_size, :]
        encoded = self.temporal_encoder(embedded)
        return encoded.mean(dim=1).reshape(x.shape[0], self.n_assets, self.model_dim)


@dataclass(frozen=True)
class RollingCorrelationGraph:
    adjacency: torch.Tensor
    graph_density: torch.Tensor
    mean_abs_correlation: torch.Tensor


def masked_softmax(logits: torch.Tensor, availability_mask: torch.Tensor) -> torch.Tensor:
    mask = availability_mask.to(dtype=torch.bool)
    masked = logits.masked_fill(~mask, -1.0e9)
    weights = torch.softmax(masked, dim=-1)
    weights = weights * mask.to(dtype=weights.dtype)
    totals = weights.sum(dim=-1, keepdim=True)
    fallback = mask.to(dtype=weights.dtype) / mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=weights.dtype)
    return torch.where(totals > 0.0, weights / totals.clamp_min(1.0e-12), fallback)


def rolling_correlation_adjacency(
    returns_window: torch.Tensor,
    availability_mask: torch.Tensor,
    *,
    threshold: float,
) -> RollingCorrelationGraph:
    if returns_window.ndim != 3:
        raise ValueError("ERR_RA_GT_RCPO_CORRELATION_SHAPE")
    mask = availability_mask.to(dtype=torch.bool)
    centered = returns_window - returns_window.mean(dim=1, keepdim=True)
    std = centered.std(dim=1, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    normalized = centered / std
    corr = torch.bmm(normalized.transpose(1, 2), normalized) / max(1, int(returns_window.shape[1]))
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    eye = torch.eye(corr.shape[-1], dtype=torch.bool, device=corr.device).unsqueeze(0)
    valid_edges = mask.unsqueeze(1) & mask.unsqueeze(2) & ~eye
    abs_corr = corr.abs()
    adjacency = (abs_corr >= float(threshold)).to(dtype=returns_window.dtype) * valid_edges.to(dtype=returns_window.dtype)
    adjacency = adjacency + torch.eye(corr.shape[-1], dtype=returns_window.dtype, device=corr.device).unsqueeze(0) * mask.unsqueeze(1).to(dtype=returns_window.dtype)
    edge_count = valid_edges.to(dtype=returns_window.dtype).sum(dim=(1, 2)).clamp_min(1.0)
    graph_density = ((adjacency > 0.0) & valid_edges).to(dtype=returns_window.dtype).sum(dim=(1, 2)) / edge_count
    mean_abs = (abs_corr * valid_edges.to(dtype=returns_window.dtype)).sum(dim=(1, 2)) / edge_count
    return RollingCorrelationGraph(adjacency=adjacency, graph_density=graph_density, mean_abs_correlation=mean_abs)


def config_for_model_name(model_name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    section = dict(_mapping(config.get("ra_gt_rcpo")))
    name = str(model_name)
    if name == "ra_gt_rcpo_no_graph":
        section["use_graph"] = False
    elif name == "ra_gt_rcpo_no_transformer":
        section["use_transformer"] = False
    elif name == "ra_gt_rcpo_no_cvar_constraint":
        section["lambda_cvar"] = 0.0
    elif name == "ra_gt_rcpo_no_cost_constraint":
        section["lambda_cost"] = 0.0
    elif name == "ra_gt_rcpo_no_turnover_constraint":
        section["lambda_turnover"] = 0.0
    elif name == "ra_gt_rcpo_mlp_actor_critic":
        section["use_graph"] = False
        section["use_transformer"] = False
        section["mlp_actor_critic"] = True
    return section


def build_risk_aware_graph_transformer(config: Mapping[str, Any], *, model_name: str) -> RiskAwareGraphTransformer:
    section = config_for_model_name(model_name, config)
    return RiskAwareGraphTransformer(
        n_features=int(config["n_features"]),
        window_size=int(config.get("window_size")),
        n_assets=int(config["n_assets"]),
        model_dim=int(section.get("model_dim", 64)),
        transformer_layers=int(section.get("transformer_layers", 1)),
        attention_heads=int(section.get("attention_heads", 2)),
        dropout=float(section.get("dropout", 0.05)),
        graph_edge_threshold=float(section.get("graph_edge_threshold", 0.10)),
        use_graph=bool(section.get("use_graph", True)),
        use_transformer=bool(section.get("use_transformer", True)),
        mlp_actor_critic=bool(section.get("mlp_actor_critic", False)),
        rho_values=section.get("rho_actions"),
    )


def _critic_head(model_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 1))


def _market_image_4d(x: torch.Tensor, n_features: int, window_size: int, n_assets: int) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError("ERR_RA_GT_RCPO_INPUT_SHAPE")
    if x.shape[1] != n_features or x.shape[2] != window_size or x.shape[3] != n_assets:
        raise ValueError("ERR_RA_GT_RCPO_INPUT_SHAPE")
    return torch.nan_to_num(x.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _valid_attention_heads(model_dim: int, requested: int) -> int:
    requested = max(1, int(requested))
    if model_dim % requested == 0:
        return requested
    for candidate in range(min(requested, model_dim), 0, -1):
        if model_dim % candidate == 0:
            return candidate
    return 1


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rho_values(values: Sequence[float] | None) -> tuple[float, ...]:
    raw = [0.0, 0.25, 0.5, 0.75, 1.0] if values is None else list(values)
    resolved = sorted({max(0.0, min(1.0, float(value))) for value in raw})
    if 0.0 not in resolved:
        resolved.insert(0, 0.0)
    return tuple(resolved)


__all__ = [
    "RA_GT_RCPO_ABLATION_MODEL_NAMES",
    "RA_GT_RCPO_ALGORITHM",
    "RA_GT_RCPO_MODEL_EXTENSION_ID",
    "RA_GT_RCPO_MODEL_NAME",
    "RA_GT_RCPO_MODEL_NAMES",
    "RiskAwareGraphTransformer",
    "RiskAwareGraphTransformerOutput",
    "build_risk_aware_graph_transformer",
    "config_for_model_name",
    "masked_softmax",
    "rolling_correlation_adjacency",
]
