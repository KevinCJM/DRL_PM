import torch
import torch.nn as nn


DEFAULT_EVALUATION_OMEGAS = (
    (0.20, 0.20, 0.20, 0.20, 0.20),
    (1.00, 0.00, 0.00, 0.00, 0.00),
    (0.00, 1.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 1.00, 0.00, 0.00),
    (0.00, 0.00, 0.00, 1.00, 0.00),
    (0.00, 0.00, 0.00, 0.00, 1.00),
)


class PreferenceConditioner(nn.Module):
    def __init__(self, latent_dim: int, omega_dim: int, sum_tol: float = 1.0e-6):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.omega_dim = int(omega_dim)
        self.sum_tol = float(sum_tol)
        if self.latent_dim <= 0 or self.omega_dim <= 0:
            raise ValueError("ERR_PREFERENCE_CONFIG_INVALID: latent_dim and omega_dim must be > 0")
        
        # FiLM parameters generator
        self.gamma_net = nn.Sequential(
            nn.Linear(self.omega_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.latent_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(self.omega_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.latent_dim)
        )

    def forward(self, latent: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        latent: [batch, latent_dim]
        omega: [batch, omega_dim]
        """
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("ERR_PREFERENCE_SHAPE_MISMATCH: latent must be [batch,latent_dim]")
        omega = self.validate_omega(omega, batch_size=latent.shape[0], device=latent.device, dtype=latent.dtype)
        gamma = self.gamma_net(omega)
        beta = self.beta_net(omega)
        
        # FiLM: gamma * latent + beta
        return gamma * latent + beta

    def validate_omega(
        self,
        omega: torch.Tensor,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not isinstance(omega, torch.Tensor):
            omega = torch.as_tensor(omega, device=device, dtype=dtype or torch.float32)
        if omega.ndim == 1:
            omega = omega.unsqueeze(0)
        if omega.ndim != 2 or omega.shape[1] != self.omega_dim:
            raise ValueError("ERR_PREFERENCE_OMEGA_INVALID: omega must be [batch,omega_dim]")
        if batch_size is not None and omega.shape[0] not in {1, int(batch_size)}:
            raise ValueError("ERR_PREFERENCE_OMEGA_INVALID: omega batch mismatch")
        if batch_size is not None and omega.shape[0] == 1 and int(batch_size) != 1:
            omega = omega.expand(int(batch_size), -1)
        omega = omega.to(device=device or omega.device, dtype=dtype or omega.dtype)
        if not torch.isfinite(omega).all():
            raise ValueError("ERR_PREFERENCE_OMEGA_INVALID: omega contains NaN or Inf")
        if (omega < -self.sum_tol).any():
            raise ValueError("ERR_PREFERENCE_OMEGA_INVALID: omega must be non-negative")
        omega = omega.clamp_min(0.0)
        if not torch.allclose(
            omega.sum(dim=1),
            torch.ones(omega.shape[0], device=omega.device, dtype=omega.dtype),
            atol=self.sum_tol,
            rtol=0.0,
        ):
            raise ValueError("ERR_PREFERENCE_OMEGA_INVALID: omega rows must sum to 1")
        return omega

    def default_omega(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        omega = torch.full(
            (int(batch_size), self.omega_dim),
            1.0 / self.omega_dim,
            device=device,
            dtype=dtype or torch.float32,
        )
        return self.validate_omega(omega)

    def preference_reward(self, reward_vector: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        if reward_vector.ndim != 2 or reward_vector.shape[1] != self.omega_dim:
            raise ValueError("ERR_PREFERENCE_REWARD_VECTOR_INVALID: reward_vector must be [batch,omega_dim]")
        omega = self.validate_omega(
            omega,
            batch_size=reward_vector.shape[0],
            device=reward_vector.device,
            dtype=reward_vector.dtype,
        )
        if not torch.isfinite(reward_vector).all():
            raise ValueError("ERR_PREFERENCE_REWARD_VECTOR_INVALID: reward_vector contains NaN or Inf")
        return (reward_vector * omega).sum(dim=1, keepdim=True)

    def evaluation_omegas(
        self,
        configured_omegas: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if configured_omegas is not None:
            return self.validate_omega(configured_omegas, device=device, dtype=dtype or torch.float32)
        if self.omega_dim == len(DEFAULT_EVALUATION_OMEGAS[0]):
            omegas = torch.tensor(DEFAULT_EVALUATION_OMEGAS, device=device, dtype=dtype or torch.float32)
        else:
            equal = torch.full((1, self.omega_dim), 1.0 / self.omega_dim, device=device, dtype=dtype or torch.float32)
            one_hot = torch.eye(self.omega_dim, device=device, dtype=dtype or torch.float32)
            omegas = torch.cat([equal, one_hot], dim=0)
        return self.validate_omega(omegas)


__all__ = ["PreferenceConditioner", "DEFAULT_EVALUATION_OMEGAS"]
