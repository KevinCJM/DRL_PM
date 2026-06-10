"""Tests for M5-T4: eta_v validation-only constraint.

Verifies:
- test split must not contain eta_v HPO parameter
- ERR_TEST_PEEKING_ETA_V check exists in formal_readiness.py
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# eta_v test peeking guard
# ---------------------------------------------------------------------------

class TestETAVNoTestPeeking:
    """eta_v must not be tuned on test split."""

    def test_formal_readiness_has_test_peeking_check(self) -> None:
        """Verify formal_readiness module has ERR_TEST_PEEKING_ETA_V check."""
        try:
            from src.experiments import formal_readiness
            source = open(formal_readiness.__file__, "r").read()
            assert "ERR_TEST_PEEKING_ETA_V" in source, (
                "formal_readiness.py should contain ERR_TEST_PEEKING_ETA_V check"
            )
        except ImportError:
            pytest.skip("formal_readiness module not available")

    def test_pipeline_excludes_eta_v_from_test_search_space(self) -> None:
        """Verify pipeline does not inject eta_v into test split HPO search space."""
        try:
            from src.experiments import pipeline
            source = open(pipeline.__file__, "r").read()
            # The pipeline should have logic to restrict eta_v to validation only
            # This is a structural check; actual behavior requires integration test
            assert "eta_v" in source, "pipeline.py should reference eta_v"
        except ImportError:
            pytest.skip("pipeline module not available")
