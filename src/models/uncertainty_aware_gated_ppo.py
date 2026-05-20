import torch
import torch.nn as nn
from typing import Mapping, Any

from .dqn_gated_multitask_cnn_ppo import FullGatedModel
from .uncertainty_heads import MultiHeadUncertaintyHeads, summarize_uncertainty, uncertainty_gate_input_fields


class UncertaintyAwareGatedPPO(FullGatedModel):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.uncertainty_config = _mapping(config.get("uncertainty"))
        self.method = str(self.uncertainty_config.get("method", "dropout")).lower().replace("-", "_")
        self.n_samples = int(self.uncertainty_config.get("n_samples", 20))
        if self.n_samples <= 0:
            raise ValueError("ERR_UNCERTAINTY_CONFIG_INVALID: n_samples must be > 0")

        if self.method in {"multi_head", "multihead"}:
            n_heads = int(self.uncertainty_config.get("n_heads", self.n_samples))
            self.uncertainty_heads = MultiHeadUncertaintyHeads(
                latent_dim=self.latent_dim,
                n_assets=self.n_assets,
                n_heads=n_heads,
                actor_hidden_dims=self.uncertainty_config.get("actor_hidden_dims"),
                gate_hidden_dims=self.uncertainty_config.get("gate_hidden_dims"),
                gate_output_dim=self.gate.output_dim,
                dueling=self.gate.dueling,
                dropout=float(self.uncertainty_config.get("dropout", 0.10)),
            )
        else:
            self.uncertainty_heads = None

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
        if self.method in {"multi_head", "multihead"}:
            return self._multi_head_forward(
                x,
                mask,
                current_weights,
                estimated_turnover,
                estimated_cost,
                deterministic=deterministic,
                candidate_weights_override=candidate_weights_override,
                rebalance_intensity_override=rebalance_intensity_override,
            )
        if self.method in {"dropout", "mc_dropout"}:
            return self._dropout_forward(
                x,
                mask,
                current_weights,
                estimated_turnover,
                estimated_cost,
                deterministic=deterministic,
                candidate_weights_override=candidate_weights_override,
                rebalance_intensity_override=rebalance_intensity_override,
            )

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
        zero_summary = summarize_uncertainty(outputs["candidate_weights"].unsqueeze(0), outputs["gate_q"].unsqueeze(0))
        return self._apply_uncertainty_outputs(outputs, zero_summary, mask)

    def _dropout_forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
        deterministic: bool,
        candidate_weights_override: torch.Tensor | None,
        rebalance_intensity_override: torch.Tensor | None,
    ) -> Mapping[str, Any]:
        was_training = self.training
        if not was_training:
            self.enable_dropout()

        sampled_outputs = []
        try:
            for _ in range(self.n_samples):
                sampled_outputs.append(
                    super().forward(
                        x,
                        mask,
                        current_weights,
                        estimated_turnover,
                        estimated_cost,
                        deterministic=deterministic,
                        candidate_weights_override=candidate_weights_override,
                        rebalance_intensity_override=rebalance_intensity_override,
                    )
                )
        finally:
            if not was_training:
                self.disable_dropout()

        summary = summarize_uncertainty(
            [outputs["candidate_weights"] for outputs in sampled_outputs],
            [outputs["gate_q"] for outputs in sampled_outputs],
        )
        outputs = super().forward(
            x,
            mask,
            current_weights,
            estimated_turnover,
            estimated_cost,
            deterministic=True,
            candidate_weights_override=candidate_weights_override,
            rebalance_intensity_override=rebalance_intensity_override,
        )
        if candidate_weights_override is None:
            outputs = dict(outputs)
            outputs["candidate_weights"] = summary["mean_candidate_weights"]
        return self._apply_uncertainty_outputs(outputs, summary, mask)

    def _multi_head_forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        current_weights: torch.Tensor,
        estimated_turnover: torch.Tensor,
        estimated_cost: torch.Tensor,
        deterministic: bool,
        candidate_weights_override: torch.Tensor | None,
        rebalance_intensity_override: torch.Tensor | None,
    ) -> Mapping[str, Any]:
        if self.uncertainty_heads is None:
            raise ValueError("ERR_UNCERTAINTY_CONFIG_INVALID: multi_head estimator is not initialized")
        latent = self.encoder(x)
        summary = self.uncertainty_heads(
            latent,
            mask,
            current_weights,
            estimated_turnover,
            estimated_cost,
            deterministic=deterministic,
            candidate_weights_override=candidate_weights_override,
        )
        candidate_weights = summary["mean_candidate_weights"]
        if candidate_weights_override is not None:
            candidate_weights = candidate_weights_override.to(device=x.device, dtype=x.dtype)
        outputs = {
            "candidate_weights": candidate_weights,
            "log_prob": self.uncertainty_heads.log_prob(latent, mask, candidate_weights),
            "value": self.critic(latent),
            "gate_q": summary["mean_gate_q"],
            "gate_action": self._select_gate_action(summary["mean_gate_q"]),
            "estimated_turnover": estimated_turnover,
            "estimated_cost": estimated_cost,
            "aux_outputs": self._auxiliary_outputs(latent),
            "latent": latent,
        }
        return self._apply_uncertainty_outputs(outputs, summary, mask)

    def _apply_uncertainty_outputs(
        self,
        outputs: Mapping[str, Any],
        summary: Mapping[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> dict[str, Any]:
        result = dict(outputs)
        candidate_weights = result["candidate_weights"]
        result["candidate_weights"] = candidate_weights
        result["mean_candidate_weights"] = summary["mean_candidate_weights"]
        result["candidate_weight_variance"] = summary["candidate_weight_variance"]
        result["weight_uncertainty"] = summary["weight_uncertainty"]
        result["gate_q"] = summary["mean_gate_q"]
        result["q_uncertainty"] = summary["q_uncertainty"]
        result["uncertainty_features"] = summary["uncertainty_features"]
        result["gate_input_extensions"] = uncertainty_gate_input_fields()
        result["gate_action"] = self._select_gate_action(summary["mean_gate_q"])
        if self.uncertainty_heads is not None:
            result["log_prob"] = self.uncertainty_heads.log_prob(result["latent"], mask, candidate_weights)
        else:
            result["log_prob"] = self.actor.log_prob(result["latent"], mask, candidate_weights)
        return result

    def _select_gate_action(self, gate_q: torch.Tensor) -> torch.Tensor:
        if not self.dqn_gate_enabled:
            return torch.ones(gate_q.shape[0], device=gate_q.device, dtype=torch.long)
        if self.gate.output_dim == 2 and self.q_gap_threshold != 0.0:
            return self.gate.select_action(gate_q, q_gap_threshold=self.q_gap_threshold)
        return torch.argmax(gate_q, dim=1)

    def enable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()

    def disable_dropout(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.eval()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["UncertaintyAwareGatedPPO"]
