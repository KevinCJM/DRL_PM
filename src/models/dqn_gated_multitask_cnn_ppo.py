import torch
import torch.nn as nn
from typing import Mapping, Any

from .encoders import EncoderFactory
from .ppo_actor import PPOActor
from .ppo_critic import PPOCritic
from .dqn_gate import DQNGate
from .auxiliary_heads import AuxiliaryHeads

class FullGatedModel(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = dict(config)
        model_config = _mapping(config.get("model"))
        self.n_assets = _resolve_int(config, "n_assets")
        self.n_features = _resolve_int(config, "n_features")
        self.window_size = _resolve_int(config, "window_size")
        self.latent_dim = int(config.get("latent_dim", model_config.get("latent_dim", 256)))
        self.use_risk_state = bool(model_config.get("use_risk_state", False))
        self.risk_state_dim = int(model_config.get("risk_state_dim", 8))
        self.effective_latent_dim = (
            self.latent_dim + self.risk_state_dim if self.use_risk_state else self.latent_dim
        )
        resolved_config = dict(config)
        resolved_config.update(
            {
                "n_assets": self.n_assets,
                "n_features": self.n_features,
                "window_size": self.window_size,
                "latent_dim": self.latent_dim,
            }
        )
        
        # 1. Encoder
        self.encoder = EncoderFactory.create(resolved_config)
        
        # 2. PPO Actor
        ppo_config = _mapping(config.get("ppo") or model_config.get("ppo"))
        ppo_hidden_dims = ppo_config.get("hidden_dims")
        max_alpha_val = ppo_config.get("max_alpha")
        self.actor = PPOActor(
            self.effective_latent_dim,
            self.n_assets,
            min_alpha=float(ppo_config.get("min_alpha", ppo_config.get("actor_min_alpha", 1.0e-3))),
            max_alpha=float(max_alpha_val) if max_alpha_val is not None else None,
            hidden_dims=ppo_config.get("actor_hidden_dims", ppo_hidden_dims),
        )
        
        # 3. PPO Critic
        self.critic = PPOCritic(
            self.effective_latent_dim,
            hidden_dims=ppo_config.get("critic_hidden_dims", ppo_hidden_dims),
        )
        
        # 4. DQN Gate
        dqn_config = _mapping(config.get("dqn") or model_config.get("dqn"))
        self.dqn_gate_enabled = bool(dqn_config.get("enabled", True))
        self.q_gap_threshold = float(dqn_config.get("q_gap_threshold", 0.0))
        self.gate = DQNGate(
            latent_dim=self.effective_latent_dim, 
            n_assets=self.n_assets, 
            dueling=dqn_config.get("dueling", True),
            output_dim=int(dqn_config.get("output_dim", 2)),
            hidden_dims=dqn_config.get("hidden_dims"),
            dropout=float(dqn_config.get("dropout", model_config.get("dropout", 0.10))),
        )
        
        # 5. Auxiliary Heads
        aux_config = _mapping(config.get("auxiliary") or model_config.get("auxiliary"))
        self.auxiliary_enabled = bool(aux_config.get("enabled", True))
        self.aux_heads = (
            AuxiliaryHeads(
                latent_dim=self.effective_latent_dim,
                n_assets=self.n_assets,
                n_features=self.n_features,
                window_size=self.window_size,
                config=aux_config,
            )
            if self.auxiliary_enabled
            else None
        )

    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor, 
                current_weights: torch.Tensor,
                estimated_turnover: torch.Tensor,
                estimated_cost: torch.Tensor,
                deterministic: bool = False,
                candidate_weights_override: torch.Tensor | None = None,
                rebalance_intensity_override: torch.Tensor | None = None,
                risk_state: torch.Tensor | None = None) -> Mapping[str, Any]:
        """
        x: [batch, n_features, window_size, n_assets]
        mask: [batch, n_assets]
        current_weights: [batch, n_assets]
        estimated_turnover: [batch, 1]
        estimated_cost: [batch, 1]
        risk_state: [batch, risk_state_dim] or None
        """
        # 1. Encode state
        latent = self.encoder(x)
        if self.use_risk_state and risk_state is None:
            raise ValueError("ERR_OBSERVATION_RISK_STATE_MISSING: use_risk_state=True but risk_state is None")
        if risk_state is not None:
            latent = torch.cat([latent, risk_state], dim=-1)
        
        # 2. PPO Candidate Weights
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
            
        # 3. PPO Value
        value = self.critic(latent)
        
        # 4. DQN Gate Q-values and action
        gate_q, gate_action = self._gate_outputs(
            latent,
            candidate_weights,
            current_weights,
            estimated_turnover,
            estimated_cost,
        )
        
        # 5. Auxiliary outputs
        aux_outputs = self._auxiliary_outputs(latent)
        
        return {
            "candidate_weights": candidate_weights,
            "log_prob": log_prob,
            "value": value,
            "gate_q": gate_q,
            "gate_action": gate_action,
            "estimated_turnover": estimated_turnover,
            "estimated_cost": estimated_cost,
            "aux_outputs": aux_outputs,
            "latent": latent
        }
        
    def get_representation_loss(self, latent: torch.Tensor) -> torch.Tensor:
        if self.aux_heads is None:
            return torch.zeros((), dtype=latent.dtype, device=latent.device)
        return self.aux_heads.get_representation_loss(latent)

    def _gate_outputs(
        self,
        latent: torch.Tensor,
        candidate_weights: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.dqn_gate_enabled:
            return self._always_rebalance_gate(latent)
        gate_q = self.gate(
            latent,
            candidate_weights,
            current_weights,
            estimated_turnover,
            estimated_cost,
        )
        if self.gate.output_dim == 2 and self.q_gap_threshold != 0.0:
            gate_action = self.gate.select_action(gate_q, q_gap_threshold=self.q_gap_threshold)
        else:
            gate_action = torch.argmax(gate_q, dim=1)
        return gate_q, gate_action

    def _always_rebalance_gate(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = latent.shape[0]
        gate_q = torch.zeros((batch_size, 2), dtype=latent.dtype, device=latent.device)
        gate_q[:, 1] = 1.0
        gate_action = torch.ones(batch_size, dtype=torch.long, device=latent.device)
        return gate_q, gate_action

    def _auxiliary_outputs(self, latent: torch.Tensor) -> Mapping[str, torch.Tensor]:
        if self.aux_heads is None:
            return {}
        return self.aux_heads(latent)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_int(config: Mapping[str, Any], key: str) -> int:
    env_config = _mapping(config.get("env"))
    feature_matrix_config = _mapping(config.get("feature_matrix"))
    if key in config:
        value = config[key]
    elif key in env_config:
        value = env_config[key]
    elif key in feature_matrix_config:
        value = feature_matrix_config[key]
    else:
        raise KeyError(key)
    value = int(value)
    if value <= 0:
        raise ValueError(f"ERR_FULL_GATED_MODEL_CONFIG_INVALID: {key} must be > 0")
    return value


__all__ = ["FullGatedModel"]
