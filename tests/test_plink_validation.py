"""Tests for PLINK dimension and genotype value validation."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jamma.io.plink import (
    _count_lines_fast,
    validate_genotype_values,
    validate_plink_dimensions,
)

# Fixture paths for gemma_synthetic dataset
FIXTURES = Path(__file__).parent / "fixtures" / "gemma_synthetic"
BFILE = FIXTURES / "test"


@pytest.mark.tier0
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


@pytest.mark.tier0
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

    def test_single_unexpected_value(self) -> None:
        """Chunk with exactly one unexpected value returns 1."""
        chunk = np.array([[0.0, 1.0, 2.0], [np.nan, 3.5, 1.0]], dtype=np.float32)
        assert validate_genotype_values(chunk) == 1

    def test_boundary_values_valid(self) -> None:
        """Values 0.0, 1.0, 2.0 are all valid; 0.5, 1.5, 2.5 are not."""
        chunk_valid = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
        assert validate_genotype_values(chunk_valid) == 0

        chunk_invalid = np.array([[0.5, 1.5, 2.5]], dtype=np.float32)
        assert validate_genotype_values(chunk_invalid) == 3


@pytest.mark.tier0
class TestCountLinesFast:
    """Tests for binary line counting."""

    def test_count_matches_text_mode(self, tmp_path: Path) -> None:
        """Binary count matches text-mode sum(1 for _ in f)."""
        content = "line1\nline2\nline3\n"
        path = tmp_path / "test.txt"
        path.write_text(content)

        expected = 3  # 3 newline characters
        assert _count_lines_fast(path) == expected

        # Cross-check with text-mode
        with open(path) as f:
            text_count = sum(1 for _ in f)
        assert _count_lines_fast(path) == text_count

    def test_count_empty_file(self, tmp_path: Path) -> None:
        """Empty file has 0 lines."""
        path = tmp_path / "empty.txt"
        path.write_text("")
        assert _count_lines_fast(path) == 0

    def test_count_no_trailing_newline(self, tmp_path: Path) -> None:
        """File without trailing newline still counts all logical lines."""
        content = "line1\nline2\nline3"  # no trailing newline: 3 logical lines
        path = tmp_path / "test.txt"
        path.write_text(content)
        assert _count_lines_fast(path) == 3

    def test_count_single_line_no_newline(self, tmp_path: Path) -> None:
        """Single line without any newline returns 1."""
        path = tmp_path / "test.txt"
        path.write_text("hello")
        assert _count_lines_fast(path) == 1

    def test_count_single_line_with_newline(self, tmp_path: Path) -> None:
        """Single line file with trailing newline."""
        path = tmp_path / "test.txt"
        path.write_text("single\n")
        assert _count_lines_fast(path) == 1

    def test_count_large_file_chunked(self, tmp_path: Path) -> None:
        """Large file counted correctly across chunk boundaries."""
        # Create file larger than default chunk_size to test multi-chunk path
        lines = [f"chr1\trs{i}\t0\t{i * 1000}\tA\tG" for i in range(10_000)]
        content = "\n".join(lines) + "\n"
        path = tmp_path / "large.bim"
        path.write_text(content)

        # Use small chunk_size to force multiple reads
        assert _count_lines_fast(path, chunk_size=1024) == 10_000
