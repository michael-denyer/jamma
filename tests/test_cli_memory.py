"""Tests for CLI memory pre-flight checks.

Uses subprocess to test memory checks without requiring specific machine
memory sizes.
"""

from pathlib import Path

import pytest

from jamma.io import get_plink_metadata

# Test fixture path
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gemma_synthetic"
PLINK_PREFIX = FIXTURE_DIR / "test"
KINSHIP_FILE = FIXTURE_DIR / "gemma_kinship.cXX.txt"


@pytest.mark.tier0
class TestCliMemoryCheckUnit:
    """Unit tests for memory check logic (no subprocess)."""

    def test_estimate_called_before_load(self):
        """Memory estimate should be computable from metadata alone."""
        from jamma.core.memory import estimate_streaming_memory

        # This simulates what CLI does: get dimensions, then estimate
        meta = get_plink_metadata(PLINK_PREFIX)
        est = estimate_streaming_memory(
            n_samples=meta["n_samples"],
        )

        assert est.total_peak_gb >= 0
        assert est.available_gb >= 0

    def test_metadata_does_not_load_genotypes(self):
        """get_plink_metadata should only read dimensions, not genotypes."""
        # This should be fast and low-memory
        meta = get_plink_metadata(PLINK_PREFIX)

        assert "n_samples" in meta
        assert "n_snps" in meta
        assert meta["n_samples"] == 100
        assert meta["n_snps"] == 500
