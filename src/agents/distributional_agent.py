from __future__ import annotations

from src.agents._specialized_agent import SpecializedAgent


class DistributionalAgent(SpecializedAgent):
    module_name = "distributional_cvar"

    @property
    def cvar_alpha(self) -> float:
        if "cvar_alpha" in self.module_config:
            return float(self.module_config["cvar_alpha"])
        model = getattr(self.hybrid_agent.ppo_agent, "model", None)
        return float(getattr(model, "cvar_alpha", 0.05))


DistributionalCVaRAgent = DistributionalAgent

__all__ = ["DistributionalAgent", "DistributionalCVaRAgent"]
