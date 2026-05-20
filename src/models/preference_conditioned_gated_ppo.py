import torch
from typing import Mapping, Any

from .dqn_gated_multitask_cnn_ppo import FullGatedModel
from .preference_conditioner import PreferenceConditioner

class PreferenceConditionedGatedPPO(FullGatedModel):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        preference_config = _mapping(config.get("preference"))
        configured_omega = preference_config.get("omega")
        self.omega_dim = int(preference_config.get("omega_dim", len(configured_omega or [0, 0, 0, 0, 0])))
        self.conditioner = PreferenceConditioner(self.latent_dim, self.omega_dim)
        default_omega = torch.tensor(
            configured_omega or [1.0 / self.omega_dim] * self.omega_dim,
            dtype=torch.float32,
        )
        self.register_buffer("default_preference_omega", self.conditioner.validate_omega(default_omega), persistent=False)
        evaluation_omegas = preference_config.get("evaluation_omegas")
        configured_evaluation = None
        if evaluation_omegas:
            configured_evaluation = torch.tensor(evaluation_omegas, dtype=torch.float32)
        self.register_buffer(
            "fixed_evaluation_omegas",
            self.conditioner.evaluation_omegas(configured_evaluation),
            persistent=False,
        )

    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor, 
                current_weights: torch.Tensor,
                estimated_turnover: torch.Tensor,
                estimated_cost: torch.Tensor,
                omega: torch.Tensor = None,
                reward_vector: torch.Tensor = None,
                deterministic: bool = False,
                candidate_weights_override: torch.Tensor | None = None,
                rebalance_intensity_override: torch.Tensor | None = None) -> Mapping[str, Any]:
        """
        x: [batch, n_features, window_size, n_assets]
        mask: [batch, n_assets]
        current_weights: [batch, n_assets]
        estimated_turnover: [batch, 1]
        estimated_cost: [batch, 1]
        omega: [batch, omega_dim]
        """
        if omega is None:
            omega = self.default_preference_omega.to(device=x.device, dtype=x.dtype)
        omega = self.conditioner.validate_omega(omega, batch_size=x.shape[0], device=x.device, dtype=x.dtype)
            
        # 1. Encode state
        raw_latent = self.encoder(x)
        
        # 2. Condition latent on preference omega
        latent = self.conditioner(raw_latent, omega)
        
        # 3. PPO Candidate Weights
        dist = self.actor.get_distribution(latent, mask)
        if candidate_weights_override is not None:
            candidate_weights = candidate_weights_override.to(device=x.device, dtype=x.dtype)
            if candidate_weights.shape != (x.shape[0], self.n_assets):
                raise ValueError("ERR_POLICY_SHAPE_MISMATCH: candidate_weights_override must be [batch,n_assets]")
            log_prob = dist.log_prob(candidate_weights)
        elif deterministic:
            candidate_weights = dist.mean
            log_prob = dist.log_prob(candidate_weights)
        else:
            candidate_weights = dist.sample()
            log_prob = dist.log_prob(candidate_weights)
            
        # 4. PPO Value
        value = self.critic(latent)
        
        # 5. DQN Gate Q-values and action
        gate_q, gate_action = self._gate_outputs(
            latent,
            candidate_weights,
            current_weights,
            estimated_turnover,
            estimated_cost,
        )
        
        # 6. Auxiliary outputs (using raw_latent or conditioned? Usually raw is better for self-supervision)
        aux_outputs = self._auxiliary_outputs(raw_latent)

        outputs = {
            "candidate_weights": candidate_weights,
            "log_prob": log_prob,
            "value": value,
            "gate_q": gate_q,
            "gate_action": gate_action,
            "estimated_turnover": estimated_turnover,
            "estimated_cost": estimated_cost,
            "aux_outputs": aux_outputs,
            "latent": latent,
            "raw_latent": raw_latent,
            "omega": omega,
        }
        if reward_vector is not None:
            reward_vector = reward_vector.to(device=x.device, dtype=x.dtype)
            outputs["reward_vector"] = reward_vector
            outputs["preference_reward"] = self.conditioner.preference_reward(reward_vector, omega)
        return outputs

    def evaluation_omegas(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        return self.fixed_evaluation_omegas.to(
            device=device or self.fixed_evaluation_omegas.device,
            dtype=dtype or self.fixed_evaluation_omegas.dtype,
        )

    def compute_preference_reward(self, reward_vector: torch.Tensor, omega: torch.Tensor | None = None) -> torch.Tensor:
        if omega is None:
            omega = self.default_preference_omega
        return self.conditioner.preference_reward(reward_vector, omega)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["PreferenceConditionedGatedPPO"]
