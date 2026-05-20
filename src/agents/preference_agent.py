from __future__ import annotations

from typing import Any

from src.agents._specialized_agent import SpecializedAgent


class PreferenceAgent(SpecializedAgent):
    module_name = "preference"

    def preference_omegas(self) -> Any:
        model = getattr(self.hybrid_agent.ppo_agent, "model", None)
        if model is not None and hasattr(model, "evaluation_omegas"):
            return model.evaluation_omegas()
        return self.module_config.get("evaluation_omegas", self.module_config.get("omega"))


PreferenceConditionedAgent = PreferenceAgent

__all__ = ["PreferenceAgent", "PreferenceConditionedAgent"]
