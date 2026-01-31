"""Pytest fixtures for GEMMA-Next test suite."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_plink_data() -> Path:
    """Return path prefix for sample PLINK data from legacy examples.

    Returns:
        Path prefix for mouse_hs1940 PLINK files (without .bed/.bim/.fam extension)
    """
    example_dir = Path(__file__).parent.parent / "legacy" / "example"
    return example_dir / "mouse_hs1940"


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
def tolerance_config() -> dict:
    """Default tolerance configuration for numerical comparisons.

    Returns:
        Dictionary with tolerance values for different comparison types
    """
    return {
        "beta_rtol": 1e-6,  # Effect size relative tolerance
        "se_rtol": 1e-6,  # Standard error relative tolerance
        "pvalue_rtol": 1e-5,  # P-value relative tolerance (looser due to CDF)
        "kinship_rtol": 1e-8,  # Kinship matrix relative tolerance
        "atol": 1e-12,  # Absolute tolerance for values near zero
    }
