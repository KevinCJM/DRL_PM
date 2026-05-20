from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.agents.hybrid_agent import HybridAgent
from src.agents.ppo_agent import PPOAgent


class SpecializedAgent:
    module_name = "specialized"

    def __init__(
        self,
        hybrid_agent: HybridAgent | PPOAgent | None = None,
        *args: Any,
        module_config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.module_config = dict(module_config or {})
        if hybrid_agent is None:
            ppo_agent = kwargs.pop("ppo_agent", None)
            if ppo_agent is None:
                raise TypeError("ERR_SPECIALIZED_AGENT_REQUIRES_HYBRID_OR_PPO_AGENT")
            self.hybrid_agent = HybridAgent(ppo_agent, *args, **kwargs)
        elif isinstance(hybrid_agent, HybridAgent):
            extra_config = kwargs.pop("config", None)
            if not self.module_config and isinstance(extra_config, Mapping):
                self.module_config = dict(extra_config)
            if args or _has_hybrid_init_kwargs(kwargs):
                raise TypeError("ERR_SPECIALIZED_AGENT_AMBIGUOUS_HYBRID_AGENT")
            self.hybrid_agent = hybrid_agent
        else:
            self.hybrid_agent = HybridAgent(hybrid_agent, *args, **kwargs)

    @property
    def status(self) -> str:
        return self.hybrid_agent.status

    @property
    def failure_state(self) -> dict[str, Any] | None:
        return self.hybrid_agent.failure_state

    @property
    def history(self) -> list[dict[str, Any]]:
        return self.hybrid_agent.history

    @property
    def device(self):
        return self.hybrid_agent.device

    @property
    def uses_shared_execution_contract(self) -> bool:
        return True

    def shared_execution_contract(self) -> dict[str, Any]:
        return {
            "module": self.module_name,
            "executor": "HybridAgent",
            "ppo_agent": type(self.hybrid_agent.ppo_agent).__name__,
            "dqn_agent": None
            if self.hybrid_agent.dqn_agent is None
            else type(self.hybrid_agent.dqn_agent).__name__,
            "auxiliary_heads": None
            if self.hybrid_agent.auxiliary_heads is None
            else type(self.hybrid_agent.auxiliary_heads).__name__,
        }

    def train(
        self,
        env: Any,
        validation_env: Any | None = None,
        epochs: int | None = None,
        num_epochs: int | None = None,
    ) -> dict[str, Any]:
        return self.hybrid_agent.train(
            env,
            validation_env=validation_env,
            epochs=epochs,
            num_epochs=num_epochs,
        )

    def evaluate(self, env: Any, deterministic: bool = True) -> dict[str, float]:
        return self.hybrid_agent.evaluate(env, deterministic=deterministic)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.hybrid_agent, name)


def _has_hybrid_init_kwargs(kwargs: Mapping[str, Any]) -> bool:
    return any(
        key in kwargs
        for key in (
            "dqn_agent",
            "auxiliary_heads",
            "auxiliary_optimizer",
            "config",
            "checkpoint_callback",
        )
    )


__all__ = ["SpecializedAgent"]
