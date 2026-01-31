"""Pytest fixtures for GEMMA-Next test suite."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from jamma.validation import ToleranceConfig


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
def tolerance_config() -> ToleranceConfig:
    """Default tolerance configuration for numerical comparisons.

    Returns:
        ToleranceConfig with default tolerance values for different comparison types
    """
    from jamma.validation import ToleranceConfig

    return ToleranceConfig()
