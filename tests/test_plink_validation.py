"""Tests for PLINK dimension and genotype value validation."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jamma.io.plink import validate_genotype_values, validate_plink_dimensions

# Fixture paths for gemma_synthetic dataset
FIXTURES = Path(__file__).parent / "fixtures" / "gemma_synthetic"
BFILE = FIXTURES / "test"


class TestValidatePlinkDimensions:
    """Tests for validate_plink_dimensions."""

    def test_valid_dimensions(self) -> None:
        """Valid PLINK files pass dimension check without raising."""
        validate_plink_dimensions(BFILE)  # Should not raise

    def test_truncated_bed(self, tmp_path: Path) -> None:
        """Truncated .bed file raises ValueError with dimension mismatch message."""
        # Copy all three PLINK files
        for ext in (".bed", ".bim", ".fam"):
            shutil.copy(FIXTURES / f"test{ext}", tmp_path / f"test{ext}")

        # Truncate the .bed file by 10 bytes
        bed_path = tmp_path / "test.bed"
        original_size = bed_path.stat().st_size
        with open(bed_path, "r+b") as f:
            f.truncate(original_size - 10)

        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_plink_dimensions(tmp_path / "test")

    def test_missing_files(self, tmp_path: Path) -> None:
        """Non-existent PLINK prefix raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            validate_plink_dimensions(tmp_path / "nonexistent")


class TestValidateGenotypeValues:
    """Tests for validate_genotype_values."""

    def test_all_valid(self) -> None:
        """Chunk with only 0, 1, 2, NaN returns 0 unexpected values."""
        chunk = np.array(
            [[0.0, 1.0, 2.0, np.nan], [2.0, 0.0, 1.0, np.nan]], dtype=np.float32
        )
        assert validate_genotype_values(chunk) == 0

    def test_with_unexpected(self) -> None:
        """Chunk with values outside {0, 1, 2, NaN} returns correct count."""
        chunk = np.array([[0.0, 3.0, 1.0], [2.0, -1.0, 0.0]], dtype=np.float32)
        assert validate_genotype_values(chunk) == 2

    def test_all_nan(self) -> None:
        """All-NaN chunk returns 0 unexpected values."""
        chunk = np.full((3, 4), np.nan, dtype=np.float32)
        assert validate_genotype_values(chunk) == 0
