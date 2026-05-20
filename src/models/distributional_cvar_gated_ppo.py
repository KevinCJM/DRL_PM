import torch
from typing import Mapping, Any

from src.data.leakage_checks import assert_decision_visibility_contract

from .dqn_gated_multitask_cnn_ppo import FullGatedModel
from .distributional_critic import DistributionalCritic


DISTRIBUTIONAL_GATE_RISK_FIELDS = (
    "candidate_expected_value",
    "hold_expected_value",
    "candidate_cvar",
    "hold_cvar",
    "candidate_tail_loss",
    "hold_tail_loss",
    "delta_U",
)


class DistributionalCVaRGatedPPO(FullGatedModel):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.dist_config = _mapping(config.get("distributional_cvar"))
        self.n_quantiles = int(self.dist_config.get("n_quantiles", 51))
        self.cvar_alpha = float(self.dist_config.get("cvar_alpha", 0.05))
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("ERR_DISTRIBUTIONAL_CVAR_INVALID_ALPHA: cvar_alpha must be in (0,1]")

        self.candidate_dist_critic = DistributionalCritic(self.latent_dim, self.n_quantiles)
        self.hold_dist_critic = DistributionalCritic(self.latent_dim, self.n_quantiles)
        self.dist_critic = self.candidate_dist_critic
        self.gate_risk_head = torch.nn.Linear(len(DISTRIBUTIONAL_GATE_RISK_FIELDS), self.gate.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
        deterministic: bool = False,
        candidate_weights_override: torch.Tensor | None = None,
        rebalance_intensity_override: torch.Tensor | None = None,
    ) -> Mapping[str, Any]:
        outputs = super().forward(
            x,
            mask,
            current_weights,
            estimated_turnover,
            estimated_cost,
            deterministic=deterministic,
            candidate_weights_override=candidate_weights_override,
            rebalance_intensity_override=rebalance_intensity_override,
        )
        latent = outputs["latent"]

        candidate_quantiles = self.candidate_dist_critic(latent)
        hold_quantiles = self.hold_dist_critic(latent)
        candidate_expected_value = self.candidate_dist_critic.expected_value(candidate_quantiles)
        hold_expected_value = self.hold_dist_critic.expected_value(hold_quantiles)
        candidate_cvar = self.candidate_dist_critic.get_cvar(candidate_quantiles, self.cvar_alpha)
        hold_cvar = self.hold_dist_critic.get_cvar(hold_quantiles, self.cvar_alpha)
        candidate_tail_loss = self.candidate_dist_critic.get_tail_loss(candidate_quantiles, self.cvar_alpha)
        hold_tail_loss = self.hold_dist_critic.get_tail_loss(hold_quantiles, self.cvar_alpha)
        candidate_utility = candidate_expected_value - candidate_tail_loss
        hold_utility = hold_expected_value - hold_tail_loss
        delta_u = candidate_utility - hold_utility
        gate_risk_features = torch.cat(
            [
                candidate_expected_value,
                hold_expected_value,
                candidate_cvar,
                hold_cvar,
                candidate_tail_loss,
                hold_tail_loss,
                delta_u,
            ],
            dim=1,
        )
        risk_adjusted_gate_q = (
            outputs["gate_q"] if not self.dqn_gate_enabled else outputs["gate_q"] + self.gate_risk_head(gate_risk_features)
        )
        if not self.dqn_gate_enabled:
            gate_action = torch.ones(x.shape[0], device=x.device, dtype=torch.long)
        elif self.gate.output_dim == 2 and self.q_gap_threshold != 0.0:
            gate_action = self.gate.select_action(risk_adjusted_gate_q, q_gap_threshold=self.q_gap_threshold)
        else:
            gate_action = torch.argmax(risk_adjusted_gate_q, dim=1)

        outputs.update(
            {
                "base_gate_q": outputs["gate_q"],
                "gate_q": risk_adjusted_gate_q,
                "gate_action": gate_action,
                "candidate_quantiles": candidate_quantiles,
                "hold_quantiles": hold_quantiles,
                "quantiles": candidate_quantiles,
                "candidate_expected_value": candidate_expected_value,
                "hold_expected_value": hold_expected_value,
                "expected_value": candidate_expected_value,
                "candidate_cvar": candidate_cvar,
                "hold_cvar": hold_cvar,
                "cvar": candidate_cvar,
                "candidate_tail_loss": candidate_tail_loss,
                "hold_tail_loss": hold_tail_loss,
                "tail_loss": candidate_tail_loss,
                "candidate_utility": candidate_utility,
                "hold_utility": hold_utility,
                "delta_U": delta_u,
                "delta_u": delta_u,
                "gate_risk_features": gate_risk_features,
                "distributional_features": gate_risk_features,
                "gate_input_extensions": distributional_gate_risk_fields(),
                "cvar_alpha": self.cvar_alpha,
            }
        )
        return outputs

    def quantile_huber_loss(
        self,
        current_quantiles: torch.Tensor,
        target_values: torch.Tensor,
        huber_kappa: float = 1.0,
    ) -> torch.Tensor:
        return self.candidate_dist_critic.quantile_huber_loss(current_quantiles, target_values, huber_kappa=huber_kappa)


def distributional_gate_risk_fields() -> tuple[str, ...]:
    assert_decision_visibility_contract(gate_input=DISTRIBUTIONAL_GATE_RISK_FIELDS)
    return DISTRIBUTIONAL_GATE_RISK_FIELDS


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["DistributionalCVaRGatedPPO", "distributional_gate_risk_fields"]
