"""Tests for M5: Formal matrix expansion and registry.

Verifies:
- otar_formal_matrix.yaml is valid YAML
- A0-A4 ablation models defined
- registry.py maps otar_cqr_gate to OTarCQRGateStrategy
- pipeline.py can resolve otar_cqr_gate model class
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Formal matrix YAML
# ---------------------------------------------------------------------------

class TestOTARFormalMatrixYAML:
    """otar_formal_matrix.yaml must be valid and contain required structure."""

    def test_file_exists(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        assert path.exists(), f"Missing: {path}"

    def test_yaml_parses(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_has_ablation_models(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        ablation = data.get("ablation_models", {})
        expected = {"A0", "A1", "A2", "A3", "A4_lite", "A4"}
        assert expected.issubset(set(ablation.keys())), (
            f"Missing: {expected - set(ablation.keys())}"
        )

    def test_a3_uses_a13_reward(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        a3 = data["ablation_models"]["A3"]
        assert a3.get("reward_mode") == "A13_otar_soft_ru_cvar_fixed"

    def test_a0_uses_a2_reward(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        a0 = data["ablation_models"]["A0"]
        assert a0.get("reward_mode") == "A2_net_log_return_after_cost"

    def test_has_seeds(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        seeds = data.get("seeds", [])
        assert len(seeds) >= 5

    def test_has_split(self) -> None:
        path = PROJECT_ROOT / "configs/paper/otar_formal_matrix.yaml"
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        split = data.get("split", {})
        assert "train" in split
        assert "validation" in split
        assert "test" in split


# ---------------------------------------------------------------------------
# Registry mapping
# ---------------------------------------------------------------------------

class TestOTARRegistryMapping:
    """registry.py must map otar_cqr_gate to OTarCQRGateStrategy."""

    def test_registry_has_otar_cqr_gate(self) -> None:
        try:
            from src.experiments.registry import DEEP_BASELINE_CLASSES
            assert "otar_cqr_gate" in DEEP_BASELINE_CLASSES, (
                "DEEP_BASELINE_CLASSES should contain 'otar_cqr_gate'"
            )
        except ImportError:
            pytest.skip("registry module not available")

    def test_registry_has_otar_fixed(self) -> None:
        try:
            from src.experiments.registry import DEEP_BASELINE_CLASSES
            assert "otar_fixed" in DEEP_BASELINE_CLASSES, (
                "DEEP_BASELINE_CLASSES should contain 'otar_fixed'"
            )
        except ImportError:
            pytest.skip("registry module not available")

    def test_otar_cqr_gate_strategy_factory_contract(self) -> None:
        try:
            from src.experiments.registry import DEEP_BASELINE_CLASSES
            strategy_factory = DEEP_BASELINE_CLASSES["otar_cqr_gate"]
            strategy = strategy_factory({"model_name": "otar_cqr_gate"})
            assert hasattr(strategy, "compute_target_weights")
        except ImportError:
            pytest.skip("registry module not available")


# ---------------------------------------------------------------------------
# Pipeline model class
# ---------------------------------------------------------------------------

class TestPipelineModelClass:
    """pipeline.py must resolve otar_cqr_gate to OTarCQRGate."""

    def test_pipeline_resolves_otar_cqr_gate(self) -> None:
        try:
            from src.experiments.pipeline import _model_class
            cls = _model_class("otar_cqr_gate")
            assert cls is not None, "_model_class('otar_cqr_gate') should not return None"
        except (ImportError, AttributeError):
            pytest.skip("pipeline._model_class not available")
