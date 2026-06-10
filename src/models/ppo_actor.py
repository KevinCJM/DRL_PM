from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from src.data.loader import DataContractError


MIN_ALPHA = 1.0e-3

class MaskedDirichlet:
    def __init__(self, alpha: torch.Tensor, mask: torch.Tensor):
        """Dirichlet policy over available assets only, scattered to full asset shape."""
        if alpha.ndim != 2 or mask.ndim != 2 or alpha.shape != mask.shape:
            raise ValueError("ERR_POLICY_SHAPE_MISMATCH: alpha and mask must be [batch,n_assets]")
        self.alpha = alpha
        self.mask = mask.to(device=alpha.device, dtype=torch.bool)
        self.batch_size, self.n_assets = mask.shape
        self.device = alpha.device

        if not torch.isfinite(alpha).all():
            raise ValueError("ERR_ALPHA_NON_FINITE: Dirichlet concentration parameters contain NaN or Inf")
        if (alpha <= 0.0).any():
            raise ValueError("ERR_ALPHA_NON_POSITIVE: Dirichlet concentration parameters must be > 0")

        self.dists: list[Dirichlet | None] = []
        self.indices: list[torch.Tensor] = []
        for i in range(self.batch_size):
            m = self.mask[i]
            if int(m.sum().item()) < 1:
                raise ValueError("ERR_CONSTRAINT_NO_AVAILABLE_ASSET: PPOActor available asset mask is empty")
            a = alpha[i][m]
            self.dists.append(Dirichlet(a) if a.numel() > 1 else None)
            self.indices.append(torch.where(m)[0])

    def sample(self) -> torch.Tensor:
        samples = torch.zeros(self.batch_size, self.n_assets, device=self.device)
        for i, dist in enumerate(self.dists):
            if dist is not None:
                samples[i, self.indices[i]] = dist.sample()
            else:
                samples[i, self.indices[i]] = 1.0
        return samples

    @property
    def mean(self) -> torch.Tensor:
        means = torch.zeros(self.batch_size, self.n_assets, device=self.device)
        for i, indices in enumerate(self.indices):
            if indices.numel() == 1:
                means[i, indices] = 1.0
                continue
            active_alpha = self.alpha[i, indices]
            means[i, indices] = active_alpha / active_alpha.sum()
        return means

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape != self.alpha.shape:
            raise ValueError("ERR_POLICY_SHAPE_MISMATCH: value must be [batch,n_assets]")
        if not torch.isfinite(value).all():
            raise ValueError("ERR_ACTION_NON_FINITE: policy action contains NaN or Inf")
        log_probs = torch.zeros(self.batch_size, device=self.device)
        for i, dist in enumerate(self.dists):
            if dist is not None:
                v = value[i, self.indices[i]]
                if (v <= 0.0).any() or float(v.sum().detach().cpu()) <= 0.0:
                    raise ValueError("ERR_ACTION_INVALID_SIMPLEX: available action weights must be positive")
                v = v / v.sum()
                log_probs[i] = dist.log_prob(v)
            else:
                log_probs[i] = 0.0
        if not torch.isfinite(log_probs).all():
            raise DataContractError("ERR_POLICY_NON_FINITE_LOG_PROB", "ERR_POLICY_NON_FINITE_LOG_PROB")
        return log_probs

    def entropy(self) -> torch.Tensor:
        entropies = torch.zeros(self.batch_size, device=self.device)
        for i, dist in enumerate(self.dists):
            if dist is not None:
                entropies[i] = dist.entropy()
        if not torch.isfinite(entropies).all():
            raise DataContractError("ERR_POLICY_NON_FINITE_ENTROPY", "ERR_POLICY_NON_FINITE_ENTROPY")
        return entropies

class PPOActor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_assets: int,
        min_alpha: float = MIN_ALPHA,
        max_alpha: float | None = None,
        hidden_dims: Sequence[int] | None = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_assets = int(n_assets)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha) if max_alpha is not None else None
        if self.max_alpha is not None and self.max_alpha <= self.min_alpha:
            raise ValueError(f"ERR_CONFIG_ALPHA_RANGE: max_alpha({self.max_alpha}) <= min_alpha({self.min_alpha})")

        dims = [self.latent_dim, *(hidden_dims or (256, 128))]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.extend([nn.Linear(int(in_dim), int(out_dim)), nn.GELU()])
        layers.append(nn.Linear(int(dims[-1]), self.n_assets))
        self.alpha_net = nn.Sequential(*layers)

    def alpha(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("ERR_POLICY_SHAPE_MISMATCH: latent must be [batch,latent_dim]")
        alpha = F.softplus(self.alpha_net(latent)) + self.min_alpha
        if self.max_alpha is not None:
            alpha = torch.clamp(alpha, max=self.max_alpha)
        if not torch.isfinite(alpha).all():
            raise ValueError("ERR_ALPHA_NON_FINITE: Dirichlet concentration parameters contain NaN or Inf")
        return alpha

    def get_distribution(self, latent: torch.Tensor, mask: torch.Tensor) -> MaskedDirichlet:
        if mask.ndim != 2 or mask.shape[1] != self.n_assets or mask.shape[0] != latent.shape[0]:
            raise ValueError("ERR_POLICY_SHAPE_MISMATCH: mask must be [batch,n_assets]")
        return MaskedDirichlet(self.alpha(latent), mask)

    def forward(self, latent: torch.Tensor, mask: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.get_distribution(latent, mask)
        if deterministic:
            return dist.mean
        return dist.sample()

    def log_prob(self, latent: torch.Tensor, mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.get_distribution(latent, mask).log_prob(value)


__all__ = ["PPOActor", "MaskedDirichlet"]
