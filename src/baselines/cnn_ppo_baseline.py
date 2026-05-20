from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .ppo_baseline import PPOBaselineStrategy


class CNNPPOBaselineStrategy(PPOBaselineStrategy):
    strategy_name = "cnn_ppo_baseline"
    default_encoder_type = "cnn"

    def _resolve_model_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        resolved = dict(super()._resolve_model_config(config))
        if "encoder" in resolved:
            encoder_config = dict(resolved["encoder"])
            encoder_config["type"] = "cnn"
            resolved["encoder"] = encoder_config
            return resolved

        model_config = dict(resolved.get("model", {}))
        model_encoder = dict(model_config.get("encoder", {}))
        model_encoder["type"] = "cnn"
        model_config["encoder"] = model_encoder
        resolved["model"] = model_config
        return resolved


__all__ = ["CNNPPOBaselineStrategy"]
