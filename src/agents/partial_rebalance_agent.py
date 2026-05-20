from __future__ import annotations

from src.agents._specialized_agent import SpecializedAgent


class PartialRebalanceAgent(SpecializedAgent):
    module_name = "partial_rebalance"

    @property
    def partial_rebalance_mode(self) -> str:
        mode = self.module_config.get("mode")
        if mode is not None:
            return str(mode)
        model = getattr(self.hybrid_agent.ppo_agent, "model", None)
        return str(getattr(model, "mode", "hybrid_dqn_beta"))


PartialRebalanceGatedAgent = PartialRebalanceAgent

__all__ = ["PartialRebalanceAgent", "PartialRebalanceGatedAgent"]
