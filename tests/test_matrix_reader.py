"""Tests for parallel matrix text reader.

Validates:
- Parallel read produces results identical to np.loadtxt
- Small matrices use np.loadtxt fallback (no subprocess spawn)
- Non-square matrices work correctly
- Edge cases: single row, single column, empty file
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.io.matrix_reader import read_matrix_parallel


@pytest.mark.tier0
class TestParallelReaderParity:
    """Verify parallel reader matches np.loadtxt exactly."""

    def test_10x10_matches_loadtxt(self, tmp_path: Path) -> None:
        """10x10 matrix via parallel reader matches np.loadtxt."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((10, 10))
        path = tmp_path / "m10.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))

    def test_100x100_matches_loadtxt(self, tmp_path: Path) -> None:
        """100x100 matrix via parallel reader matches np.loadtxt."""
        rng = np.random.default_rng(123)
        matrix = rng.standard_normal((100, 100))
        path = tmp_path / "m100.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))

    def test_600x600_parallel_matches_loadtxt(self, tmp_path: Path) -> None:
        """600x600 matrix triggers parallel path, matches np.loadtxt."""
        rng = np.random.default_rng(456)
        matrix = rng.standard_normal((600, 600))
        path = tmp_path / "m600.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path, min_rows_for_parallel=500)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))

    def test_non_square_100x50(self, tmp_path: Path) -> None:
        """Non-square matrix (100x50) works correctly."""
        rng = np.random.default_rng(789)
        matrix = rng.standard_normal((100, 50))
        path = tmp_path / "rect.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))

    def test_whitespace_delimiter(self, tmp_path: Path) -> None:
        """Whitespace-delimited file (GEMMA default) works."""
        rng = np.random.default_rng(101)
        matrix = rng.standard_normal((50, 50))
        path = tmp_path / "ws.txt"
        # Default delimiter for savetxt is space
        np.savetxt(path, matrix, fmt="%.10g")

        result = read_matrix_parallel(path, delimiter=None)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))


@pytest.mark.tier0
class TestSmallMatrixFallback:
    """Verify small matrices use np.loadtxt fallback."""

    def test_below_threshold_uses_loadtxt(self, tmp_path: Path) -> None:
        """Matrix with fewer rows than threshold uses np.loadtxt."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 100))
        path = tmp_path / "small.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        # min_rows_for_parallel=500 means 100 rows uses loadtxt
        result = read_matrix_parallel(path, min_rows_for_parallel=500)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))


@pytest.mark.tier0
class TestEdgeCases:
    """Edge cases for matrix reader."""

    def test_single_row(self, tmp_path: Path) -> None:
        """Single-row matrix produces correct 2D shape."""
        path = tmp_path / "single_row.txt"
        np.savetxt(path, np.array([[1.0, 2.0, 3.0]]), fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path)
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result, [[1.0, 2.0, 3.0]])

    def test_single_element(self, tmp_path: Path) -> None:
        """1x1 matrix produces correct 2D shape."""
        path = tmp_path / "single.txt"
        np.savetxt(path, np.array([[42.0]]), fmt="%.10g")

        result = read_matrix_parallel(path)
        assert result.shape == (1, 1)
        np.testing.assert_array_equal(result, [[42.0]])

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """Empty file raises ValueError."""
        path = tmp_path / "empty.txt"
        path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            read_matrix_parallel(path)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Nonexistent file raises FileNotFoundError."""
        path = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            read_matrix_parallel(path)

    def test_result_is_contiguous_float64(self, tmp_path: Path) -> None:
        """Result is C-contiguous float64 regardless of path."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((600, 50))
        path = tmp_path / "contig.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path, min_rows_for_parallel=500)
        assert result.dtype == np.float64
        assert result.flags["C_CONTIGUOUS"]

    def test_parallel_with_2_workers(self, tmp_path: Path) -> None:
        """Explicit n_workers=2 produces correct results."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((600, 30))
        path = tmp_path / "w2.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path, n_workers=2, min_rows_for_parallel=500)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))

    def test_wide_matrix_uses_parallel(self, tmp_path: Path) -> None:
        """Wide matrix (many cols, lines > 64KB) still uses parallel path."""
        rng = np.random.default_rng(99)
        # 600 rows x 5000 cols: each line ~100KB, well beyond old 64KB sample
        matrix = rng.standard_normal((600, 5000))
        path = tmp_path / "wide.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path, min_rows_for_parallel=500)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))


@pytest.mark.tier0
class TestCsvDelimiter:
    """Verify CSV (comma-delimited) files are handled correctly."""

    def test_csv_small_matrix(self, tmp_path: Path) -> None:
        """CSV file with small matrix uses loadtxt fallback correctly."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((10, 5))
        path = tmp_path / "small.csv"
        np.savetxt(path, matrix, fmt="%.10g", delimiter=",")

        result = read_matrix_parallel(path, delimiter=",")
        expected = np.loadtxt(path, dtype=np.float64, delimiter=",")
        np.testing.assert_array_equal(result, np.atleast_2d(expected))

    def test_csv_parallel_path(self, tmp_path: Path) -> None:
        """CSV file via parallel path matches np.loadtxt."""
        rng = np.random.default_rng(123)
        matrix = rng.standard_normal((600, 10))
        path = tmp_path / "par.csv"
        np.savetxt(path, matrix, fmt="%.10g", delimiter=",")

        result = read_matrix_parallel(
            path, delimiter=",", n_workers=2, min_rows_for_parallel=500
        )
        expected = np.loadtxt(path, dtype=np.float64, delimiter=",")
        np.testing.assert_array_equal(result, np.atleast_2d(expected))


@pytest.mark.tier0
class TestBlankAndCommentLines:
    """Verify blank and comment lines are skipped, matching np.loadtxt."""

    def test_blank_lines_small_matrix(self, tmp_path: Path) -> None:
        """Blank lines in a small matrix (loadtxt path) are skipped."""
        path = tmp_path / "blanks.txt"
        path.write_text("1.0 2.0 3.0\n\n4.0 5.0 6.0\n\n\n7.0 8.0 9.0\n")

        result = read_matrix_parallel(path)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))
        assert result.shape == (3, 3)

    def test_comment_lines_small_matrix(self, tmp_path: Path) -> None:
        """Comment lines (starting with #) are skipped."""
        path = tmp_path / "comments.txt"
        path.write_text(
            "# This is a header comment\n1.0 2.0\n# Another comment\n3.0 4.0\n"
        )

        result = read_matrix_parallel(path)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))
        assert result.shape == (2, 2)

    def test_blank_lines_parallel_path(self, tmp_path: Path) -> None:
        """Blank lines in parallel path don't corrupt row offsets."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((600, 5))
        path = tmp_path / "blanks_par.txt"

        # Write matrix with blank lines interspersed every 100 rows
        lines = []
        for i, row in enumerate(matrix):
            lines.append(" ".join(f"{v:.10g}" for v in row))
            if (i + 1) % 100 == 0:
                lines.append("")  # blank line
        path.write_text("\n".join(lines) + "\n")

        result = read_matrix_parallel(path, n_workers=2, min_rows_for_parallel=500)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))
        assert result.shape == (600, 5)

    def test_comment_lines_parallel_path(self, tmp_path: Path) -> None:
        """Comment lines in parallel path don't corrupt row offsets."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((600, 5))
        path = tmp_path / "comments_par.txt"

        # Write matrix with comment lines interspersed every 200 rows
        lines = ["# Matrix data file"]
        for i, row in enumerate(matrix):
            lines.append(" ".join(f"{v:.10g}" for v in row))
            if (i + 1) % 200 == 0:
                lines.append("# checkpoint")
        path.write_text("\n".join(lines) + "\n")

        result = read_matrix_parallel(path, n_workers=2, min_rows_for_parallel=500)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))
        assert result.shape == (600, 5)

    def test_only_comments_and_blanks_parallel_raises(self, tmp_path: Path) -> None:
        """File with only comments/blanks raises ValueError in parallel path."""
        # Build a file large enough to trigger the parallel path: many comment
        # and blank lines so bytes_per_line extrapolation yields >= 500 rows.
        path = tmp_path / "no_data.txt"
        lines = ["# comment line"] * 600 + [""] * 100
        path.write_text("\n".join(lines) + "\n")

        with pytest.raises(ValueError, match="no data rows"):
            read_matrix_parallel(path, min_rows_for_parallel=500)
