"""Pytest fixtures for JAMMA test suite."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from jamma.validation import ToleranceConfig

# =============================================================================
# Test Tier System
# =============================================================================
#
# JAMMA uses a three-tier test system to balance CI speed with thorough validation:
#
# tier0 - Fast Unit Tests (<5s each)
#   - Pure computation tests (no I/O, no GEMMA reference)
#   - Mocked external dependencies
#   - Run on every commit in CI
#   - Example: test_eigenvalue_thresholding, test_pab_computation
#   - Run: pytest -m tier0
#
# tier1 - Parity Tests (<60s each)
#   - Validates numerical output against GEMMA reference data
#   - Uses fixture files in tests/fixtures/
#   - Run on PRs and merges
#   - Example: test_assoc_matches_gemma, test_kinship_matches_gemma
#   - Run: pytest -m tier1
#
# tier2 - Scale Tests (memory/time intensive)
#   - Large sample counts (10k+ samples)
#   - Memory-constrained scenarios
#   - Run manually or in nightly CI with large VMs
#   - Example: test_streaming_large_dataset, test_memory_estimation_accuracy
#   - Run: pytest -m tier2
#
# The existing @pytest.mark.slow is an alias for tier2.
#
# Quick reference:
#   pytest -m tier0           # Fast tests only (~30s total)
#   pytest -m "tier0 or tier1"  # All fast + parity tests
#   pytest -m "not tier2"     # Exclude slow/memory tests
#   pytest                    # All tests
# =============================================================================


def pytest_configure(config):
    """Register custom markers and provide tier documentation."""
    # Markers are registered in pyproject.toml, but we can add runtime config here
    pass


@pytest.fixture
def sample_plink_data() -> Path:
    """Return path prefix for sample PLINK data from test fixtures.

    Returns:
        Path prefix for gemma_synthetic PLINK files (without .bed/.bim/.fam extension)
    """
    return Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory for test results.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        Path to output directory
    """
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def tolerance_config() -> ToleranceConfig:
    """Default tolerance configuration for numerical comparisons.

    Returns:
        ToleranceConfig with default tolerance values for different comparison types
    """
    from jamma.validation import ToleranceConfig

    return ToleranceConfig()
