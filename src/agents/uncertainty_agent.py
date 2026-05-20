from __future__ import annotations

from src.agents._specialized_agent import SpecializedAgent


class UncertaintyAgent(SpecializedAgent):
    module_name = "uncertainty"

    @property
    def uncertainty_method(self) -> str:
        method = self.module_config.get("method")
        if method is not None:
            return str(method)
        model = getattr(self.hybrid_agent.ppo_agent, "model", None)
        return str(getattr(model, "method", "dropout"))


UncertaintyAwareAgent = UncertaintyAgent

__all__ = ["UncertaintyAgent", "UncertaintyAwareAgent"]
