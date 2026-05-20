from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .dqn_gated_multitask_cnn_ppo import FullGatedModel
from .partial_rebalance_gate import (
    DEFAULT_DISCRETE_RHO_VALUES,
    BetaIntensityActor,
    DiscretePartialGate,
    _validate_rho_values,
)


VALID_PARTIAL_REBALANCE_MODES = {"discrete_dqn", "continuous_beta", "hybrid_dqn_beta"}


class PartialRebalanceGatedPPO(FullGatedModel):
    def __init__(self, config: Mapping[str, Any]):
        self.partial_config = _mapping(config.get("partial_rebalance"))
        self.mode = str(self.partial_config.get("mode", "hybrid_dqn_beta"))
        if self.mode not in VALID_PARTIAL_REBALANCE_MODES:
            raise ValueError("ERR_PARTIAL_REBALANCE_MODE_INVALID: partial_rebalance.mode")

        super().__init__(_force_partial_gate_config(config, self.mode, self.partial_config.get("discrete_rho_values")))
        self.rho_values = self.partial_config.get("discrete_rho_values")

        if self.mode == "discrete_dqn":
            self.rho_values = _validate_rho_values(
                DEFAULT_DISCRETE_RHO_VALUES if self.rho_values is None else self.rho_values
            )
            self.discrete_gate = DiscretePartialGate(self.latent_dim, self.n_assets, self.rho_values)
        elif self.mode == "continuous_beta":
            self.intensity_actor = BetaIntensityActor(
                self.latent_dim,
                min_concentration=self.partial_config.get("beta_min_concentration", 1.0e-3),
            )
        elif self.mode == "hybrid_dqn_beta":
            self.intensity_actor = BetaIntensityActor(
                self.latent_dim,
                min_concentration=self.partial_config.get("beta_min_concentration", 1.0e-3),
            )

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

        batch_size = x.shape[0]
        rebalance_intensity = torch.ones(batch_size, 1, device=x.device)
        intensity_log_prob = torch.zeros(x.shape[0], 1, device=x.device)

        if self.mode == "discrete_dqn":
            rebalance_intensity, gate_q = self.discrete_gate(
                latent,
                outputs["candidate_weights"],
                current_weights,
                estimated_turnover,
                estimated_cost,
            )
            outputs["gate_q"] = gate_q
            outputs["gate_action"] = (rebalance_intensity.squeeze(1) > 0).long()
            outputs["rebalance_values"] = self.discrete_gate.rho_values
            outputs["raw_rebalance_intensity"] = rebalance_intensity

        elif self.mode == "continuous_beta":
            if rebalance_intensity_override is None:
                rebalance_intensity, intensity_log_prob = self.intensity_actor.sample_with_log_prob(
                    latent,
                    deterministic=deterministic,
                )
            else:
                rebalance_intensity = rebalance_intensity_override.to(device=x.device, dtype=x.dtype)
                intensity_log_prob = self.intensity_actor.log_prob(latent, rebalance_intensity)
            outputs["gate_action"] = torch.ones(batch_size, device=x.device, dtype=torch.long)
            outputs["raw_rebalance_intensity"] = rebalance_intensity

        elif self.mode == "hybrid_dqn_beta":
            if rebalance_intensity_override is None:
                intensity_sample, intensity_log_prob = self.intensity_actor.sample_with_log_prob(
                    latent,
                    deterministic=deterministic,
                )
            else:
                intensity_sample = rebalance_intensity_override.to(device=x.device, dtype=x.dtype)
                intensity_log_prob = self.intensity_actor.log_prob(latent, intensity_sample)

            outputs["raw_rebalance_intensity"] = intensity_sample
            rebalance_intensity = intensity_sample * outputs["gate_action"].float().view(-1, 1)

        outputs["rebalance_intensity"] = rebalance_intensity
        outputs["intensity_log_prob"] = intensity_log_prob
        outputs["joint_log_prob"] = _joint_log_prob(outputs["log_prob"], intensity_log_prob)

        return outputs


def _joint_log_prob(policy_log_prob: torch.Tensor, intensity_log_prob: torch.Tensor) -> torch.Tensor:
    if policy_log_prob.ndim != 1 or intensity_log_prob.ndim != 2 or intensity_log_prob.shape[1] != 1:
        raise ValueError("ERR_POLICY_SHAPE_MISMATCH: log_prob must be [batch], intensity_log_prob must be [batch,1]")
    if policy_log_prob.shape[0] != intensity_log_prob.shape[0]:
        raise ValueError("ERR_POLICY_SHAPE_MISMATCH: log_prob batch mismatch")
    return policy_log_prob + intensity_log_prob.squeeze(1)


def _force_partial_gate_config(
    config: Mapping[str, Any],
    mode: str,
    rho_values: Any,
) -> Mapping[str, Any]:
    if mode == "discrete_dqn":
        resolved_rho_values = _validate_rho_values(
            DEFAULT_DISCRETE_RHO_VALUES if rho_values is None else rho_values
        )
        return _force_dqn_output_dim(config, len(resolved_rho_values))
    if mode in {"continuous_beta", "hybrid_dqn_beta"}:
        return _force_dqn_output_dim(config, 2)
    return config


def _force_dqn_output_dim(config: Mapping[str, Any], output_dim: int) -> Mapping[str, Any]:
    if int(output_dim) <= 0:
        raise ValueError("ERR_PARTIAL_GATE_OUTPUT_DIM_INVALID: output_dim must be > 0")

    resolved = dict(config)
    if isinstance(config.get("dqn"), Mapping):
        dqn_config = dict(config["dqn"])
        dqn_config["output_dim"] = int(output_dim)
        resolved["dqn"] = dqn_config
        return resolved

    model_config = config.get("model")
    if isinstance(model_config, Mapping) and isinstance(model_config.get("dqn"), Mapping):
        resolved_model = dict(model_config)
        dqn_config = dict(model_config["dqn"])
        dqn_config["output_dim"] = int(output_dim)
        resolved_model["dqn"] = dqn_config
        resolved["model"] = resolved_model
        return resolved

    resolved["dqn"] = {"output_dim": int(output_dim)}
    return resolved


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["PartialRebalanceGatedPPO", "VALID_PARTIAL_REBALANCE_MODES"]
