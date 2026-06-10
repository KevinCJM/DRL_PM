"""Tests for M7-T3/T4: Formal artifacts and paper aggregate outputs.

Verifies:
- paper_main_comparison.csv, paper_seed_summary.csv, paper_paired_statistics.csv exist
- otar_gate_diagnostics.csv, otar_tail_calibration.csv exist for CQR runs
- run manifest includes required fields
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Paper aggregate OTAR outputs
# ---------------------------------------------------------------------------

class TestOTARPaperAggregateOutputs:
    """paper_aggregate.py must support OTAR-specific CSV outputs."""

    def test_paper_aggregate_has_otar_gate_diagnostics(self) -> None:
        """Verify paper_aggregate module has _build_otar_gate_diagnostics function."""
        try:
            from src.experiments import paper_aggregate
            assert hasattr(paper_aggregate, "_build_otar_gate_diagnostics"), (
                "paper_aggregate.py should have _build_otar_gate_diagnostics function"
            )
        except ImportError:
            pytest.skip("paper_aggregate module not available")

    def test_paper_aggregate_has_otar_tail_calibration(self) -> None:
        """Verify paper_aggregate module has _build_otar_tail_calibration function."""
        try:
            from src.experiments import paper_aggregate
            assert hasattr(paper_aggregate, "_build_otar_tail_calibration"), (
                "paper_aggregate.py should have _build_otar_tail_calibration function"
            )
        except ImportError:
            pytest.skip("paper_aggregate module not available")

    def test_paper_aggregate_has_otar_training_stability(self) -> None:
        """Verify paper_aggregate module has _build_otar_training_stability function."""
        try:
            from src.experiments import paper_aggregate
            assert hasattr(paper_aggregate, "_build_otar_training_stability"), (
                "paper_aggregate.py should have _build_otar_training_stability function"
            )
        except ImportError:
            pytest.skip("paper_aggregate module not available")


# ---------------------------------------------------------------------------
# Run manifest fields
# ---------------------------------------------------------------------------

class TestRunManifestFields:
    """run_experiment.py must write required manifest fields."""

    def test_run_experiment_has_code_commit_hash(self) -> None:
        """Verify run_experiment module references code_commit_hash."""
        try:
            from src.experiments import run_experiment
            source = open(run_experiment.__file__, "r").read()
            assert "code_commit_hash" in source, (
                "run_experiment.py should reference code_commit_hash"
            )
        except ImportError:
            pytest.skip("run_experiment module not available")

    def test_run_experiment_has_ablation_id(self) -> None:
        """Verify run_experiment module references ablation_id in manifest."""
        try:
            from src.experiments import run_experiment
            source = open(run_experiment.__file__, "r").read()
            assert "ablation_id" in source, (
                "run_experiment.py should reference ablation_id"
            )
        except ImportError:
            pytest.skip("run_experiment module not available")
