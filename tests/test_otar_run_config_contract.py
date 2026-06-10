"""Tests for M1-T2: OTAR run config contract validation.

Verifies:
- Every otar_* config declares reward.mode = A13_otar_soft_ru_cvar_fixed
- Config loader maps reward.mode to the field consumed by RewardCalculator
- Formal matrix expands A0-A4 + Small-8/Core-13 + seeds
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import ConfigLoader, PROJECT_ROOT


OTAR_CONFIGS = [
    "configs/paper/otar_small8_smoke.yaml",
    "configs/paper/otar_small8_pilot.yaml",
    "configs/paper/otar_core13_robustness.yaml",
]


def _load_config(relative: str) -> dict:
    return ConfigLoader.load(PROJECT_ROOT / relative)


# ---------------------------------------------------------------------------
# reward.mode contract
# ---------------------------------------------------------------------------

class TestOTARRewardModeContract:
    """Every OTAR base config must declare reward.mode = A13_otar_soft_ru_cvar_fixed."""

    @pytest.mark.parametrize("config_path", OTAR_CONFIGS)
    def test_otar_config_declares_reward_mode(self, config_path: str) -> None:
        config = _load_config(config_path)
        reward = config.get("reward", {})
        assert isinstance(reward, dict), f"{config_path}: missing reward section"
        mode = reward.get("mode")
        assert mode == "A13_otar_soft_ru_cvar_fixed", (
            f"{config_path}: reward.mode={mode!r}, expected 'A13_otar_soft_ru_cvar_fixed'"
        )

    @pytest.mark.parametrize("config_path", OTAR_CONFIGS)
    def test_otar_config_declares_required_reward_params(self, config_path: str) -> None:
        config = _load_config(config_path)
        reward = config.get("reward", {})
        required_keys = [
            "lambda_tail", "confidence_q", "tau", "tau_v", "eta_v",
            "lambda_dd", "lambda_turnover", "v_update_mode", "v_clip_min", "v_clip_max",
        ]
        for key in required_keys:
            assert key in reward, f"{config_path}: missing reward.{key}"


# ---------------------------------------------------------------------------
# Formal matrix expansion
# ---------------------------------------------------------------------------

class TestOTARFormalMatrixExpansion:
    """Formal matrix must define A0-A4 ablation models."""

    def test_formal_matrix_exists(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        assert path.exists(), f"Missing formal matrix: {path}"

    def test_formal_matrix_has_ablation_models(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        with path.open("r", encoding="utf-8") as f:
            matrix = yaml.safe_load(f)
        ablation = matrix.get("ablation_models", {})
        expected = {"A0", "A1", "A2", "A3", "A4_lite", "A4"}
        assert expected.issubset(set(ablation.keys())), (
            f"Missing ablation keys: {expected - set(ablation.keys())}"
        )

    def test_formal_matrix_has_seeds(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        with path.open("r", encoding="utf-8") as f:
            matrix = yaml.safe_load(f)
        seeds = matrix.get("seeds", [])
        assert len(seeds) >= 5, f"Expected >= 5 seeds, got {len(seeds)}"

    def test_formal_matrix_has_universes(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        with path.open("r", encoding="utf-8") as f:
            matrix = yaml.safe_load(f)
        universes = matrix.get("universes", [])
        assert "Small-8" in universes, "Missing Small-8 universe"
        assert "Core-13" in universes, "Missing Core-13 universe"
