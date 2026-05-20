from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionalCritic(nn.Module):
    def __init__(self, latent_dim: int, n_quantiles: int = 51):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_quantiles = int(n_quantiles)
        if self.latent_dim <= 0:
            raise ValueError("ERR_DISTRIBUTIONAL_CRITIC_INVALID_CONFIG: latent_dim must be > 0")
        if self.n_quantiles <= 0:
            raise ValueError("ERR_DISTRIBUTIONAL_CRITIC_INVALID_CONFIG: n_quantiles must be > 0")
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.n_quantiles)
        )
        # Tau values: midpoints of n_quantiles buckets
        self.register_buffer(
            "tau",
            torch.linspace(0.5 / self.n_quantiles, 1.0 - 0.5 / self.n_quantiles, self.n_quantiles),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Output: [batch, n_quantiles]
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("ERR_DISTRIBUTIONAL_CRITIC_SHAPE_MISMATCH: latent must be [batch,latent_dim]")
        if not torch.isfinite(latent).all():
            raise ValueError("ERR_DISTRIBUTIONAL_CRITIC_NON_FINITE: latent contains NaN or Inf")
        return self.fc(latent)

    def get_cvar(self, quantiles: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        CVaR_alpha is the average of quantiles below alpha.
        """
        self._validate_quantiles(quantiles)
        if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) <= 1.0:
            raise ValueError("ERR_DISTRIBUTIONAL_CRITIC_INVALID_ALPHA: alpha must be in (0,1]")
        # Number of quantiles to average
        n_below = max(1, int(math.ceil(float(alpha) * self.n_quantiles)))
        
        # Sort quantiles (QR-DQN quantiles are not necessarily sorted, though they should be)
        sorted_quantiles, _ = torch.sort(quantiles, dim=1)
        
        # Take the mean of the first n_below quantiles
        cvar = sorted_quantiles[:, :n_below].mean(dim=1, keepdim=True)
        return cvar

    def get_tail_loss(self, quantiles: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        return torch.clamp(-self.get_cvar(quantiles, alpha=alpha), min=0.0)

    def expected_value(self, quantiles: torch.Tensor) -> torch.Tensor:
        self._validate_quantiles(quantiles)
        return quantiles.mean(dim=1, keepdim=True)

    def quantile_huber_loss(self, current_quantiles: torch.Tensor, target_values: torch.Tensor, huber_kappa: float = 1.0) -> torch.Tensor:
        """
        Quantile Huber Loss for Quantile Regression DQN.
        current_quantiles: [batch, n_quantiles]
        target_values: [batch, 1] or [batch, m_quantiles]
        """
        self._validate_quantiles(current_quantiles)
        if not math.isfinite(float(huber_kappa)) or huber_kappa <= 0.0:
            raise ValueError("ERR_DISTRIBUTIONAL_CRITIC_INVALID_HUBER_KAPPA: huber_kappa must be > 0")
        if target_values.ndim == 1:
            target_values = target_values.view(-1, 1)
        if target_values.ndim != 2 or target_values.shape[0] != current_quantiles.shape[0]:
            raise ValueError(
                "ERR_DISTRIBUTIONAL_CRITIC_SHAPE_MISMATCH: target_values must be [batch,1] or [batch,m_quantiles]"
            )
        if not torch.isfinite(target_values).all():
            raise ValueError("ERR_DISTRIBUTIONAL_CRITIC_NON_FINITE: target_values contain NaN or Inf")

        # Ensure target_values is [batch, 1, m_quantiles] for broadcasting
        if target_values.dim() == 2:
            target_values = target_values.unsqueeze(1) # [batch, 1, m]
        
        # current_quantiles: [batch, n_quantiles, 1]
        current_quantiles = current_quantiles.unsqueeze(2)
        
        # Pairwise differences: [batch, n, m]
        diff = target_values - current_quantiles
        
        # Huber loss
        abs_diff = diff.abs()
        huber_loss = torch.where(abs_diff <= huber_kappa, 0.5 * diff.pow(2), huber_kappa * (abs_diff - 0.5 * huber_kappa))
        
        # Quantile loss weight: |tau - I(diff < 0)|
        # tau: [1, n, 1]
        tau = self.tau.view(1, -1, 1)
        weight = torch.abs(tau - (diff.detach() < 0).float())
        
        # Final loss: mean over quantiles and batch
        loss = (weight * huber_loss).mean(dim=1).sum(dim=1).mean()
        return loss

    def _validate_quantiles(self, quantiles: torch.Tensor) -> None:
        if quantiles.ndim != 2 or quantiles.shape[1] != self.n_quantiles:
            raise ValueError("ERR_DISTRIBUTIONAL_CRITIC_SHAPE_MISMATCH: quantiles must be [batch,n_quantiles]")
        if not torch.isfinite(quantiles).all():
            raise ValueError("ERR_DISTRIBUTIONAL_CRITIC_NON_FINITE: quantiles contain NaN or Inf")


__all__ = ["DistributionalCritic"]
