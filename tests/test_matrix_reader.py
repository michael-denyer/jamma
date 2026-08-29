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

from jamma.io.matrix_reader import (
    _cleanup_temp_memmap,
    _scan_chunk_boundaries,
    read_matrix_parallel,
)

pytestmark = pytest.mark.tier0


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


class TestBlockCopyMemmap:
    """Verify block-copy memmap-to-dense path for matrices exceeding block size."""

    def test_2048_rows_block_copy(self, tmp_path: Path) -> None:
        """2048-row matrix exercises both max_rows parsing and block-copy (>1024)."""
        rng = np.random.default_rng(2048)
        matrix = rng.standard_normal((2048, 4))
        path = tmp_path / "block2048.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path, n_workers=2, min_rows_for_parallel=500)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))
        assert result.shape == (2048, 4)
        assert result.dtype == np.float64
        assert result.flags["C_CONTIGUOUS"]


class TestBoundedMemoryBehavior:
    """Verify memory-bounded parsing: no BytesIO buffer, correct block copy."""

    def test_block_copy_produces_correct_result(self, tmp_path: Path) -> None:
        """Block-by-block memmap-to-dense copy with known values (>1024 rows)."""
        # Use np.arange reshaped so every element is uniquely identifiable
        n_rows, n_cols = 2048, 4
        matrix = np.arange(n_rows * n_cols, dtype=np.float64).reshape(n_rows, n_cols)
        path = tmp_path / "block_known.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(path, n_workers=2, min_rows_for_parallel=500)
        np.testing.assert_array_equal(result, matrix)

    def test_max_rows_parsing_matches_loadtxt(self, tmp_path: Path) -> None:
        """max_rows parsing path produces identical values to np.loadtxt."""
        rng = np.random.default_rng(777)
        matrix = rng.standard_normal((800, 10))
        path = tmp_path / "max_rows_parity.txt"
        np.savetxt(path, matrix, fmt="%.15g", delimiter="\t")

        result = read_matrix_parallel(path, n_workers=4, min_rows_for_parallel=500)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))

    def test_worker_receives_n_rows_in_args(self, tmp_path: Path) -> None:
        """_scan_chunk_boundaries returns 4-element tuples with n_rows."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((100, 5))
        path = tmp_path / "scan_test.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        n_rows, n_cols, chunks = _scan_chunk_boundaries(path, n_workers=2)
        assert n_rows == 100
        assert n_cols == 5
        assert len(chunks) >= 1

        total_chunk_rows = 0
        for chunk in chunks:
            assert len(chunk) == 4, (
                f"Expected 4-element tuple (start_byte, end_byte, start_row, n_rows), "
                f"got {len(chunk)} elements"
            )
            start_byte, end_byte, start_row, chunk_n_rows = chunk
            assert start_byte >= 0
            assert end_byte > start_byte
            assert start_row >= 0
            assert chunk_n_rows > 0
            total_chunk_rows += chunk_n_rows

        assert total_chunk_rows == 100, (
            f"Sum of chunk n_rows ({total_chunk_rows}) != total rows (100)"
        )


# =============================================================================
# Temp memmap cleanup tests
# =============================================================================


class TestMemmapLifecycle:
    """Verify temp memmap backing files are cleaned up after parsing."""

    def test_copy_true_no_temp_file_leak(self, tmp_path: Path) -> None:
        """Reading leaves no .jamma_mread_ temp directories after completion."""
        rng = np.random.default_rng(47)
        matrix = rng.standard_normal((600, 10))
        path = tmp_path / "no_leak_test.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        read_matrix_parallel(path, n_workers=2, min_rows_for_parallel=500)

        # Verify no .jamma_mread_ temp dirs remain in the matrix file's directory
        leftover_dirs = [
            d
            for d in tmp_path.iterdir()
            if d.is_dir() and d.name.startswith(".jamma_mread_")
        ]
        assert leftover_dirs == [], (
            f"Found leftover temp dirs after read: {leftover_dirs}"
        )

    def test_cleanup_on_parse_failure(self, tmp_path: Path) -> None:
        """Temp dir is cleaned up when a parallel parse fails mid-way.

        Regression coverage for the finally-block cleanup path now that
        the copy=False mode (and its separate weakref.finalize cleanup)
        is gone: this is the only cleanup path left, so a parse failure
        must still not leak the .jamma_mread_ temp directory.
        """
        rng = np.random.default_rng(49)
        matrix = rng.standard_normal((600, 10))
        path = tmp_path / "fail_cleanup.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        # Inject a non-numeric line at row ~400 to cause a parse error
        lines = path.read_text().splitlines()
        lines[400] = "not_a_number\t" * 10
        path.write_text("\n".join(lines) + "\n")

        with pytest.raises(RuntimeError):
            read_matrix_parallel(path, n_workers=2, min_rows_for_parallel=500)

        leftover_dirs = [
            d
            for d in tmp_path.iterdir()
            if d.is_dir() and d.name.startswith(".jamma_mread_")
        ]
        assert leftover_dirs == [], (
            f"Found leftover temp dirs after failed parse: {leftover_dirs}"
        )

    def test_cleanup_temp_memmap_nonexistent_paths(self) -> None:
        """_cleanup_temp_memmap does not raise when files/dirs are already gone."""
        _cleanup_temp_memmap("/nonexistent/dir", "/nonexistent/dir/matrix.dat")

    def test_cleanup_temp_memmap_permission_error(self) -> None:
        """_cleanup_temp_memmap does not raise when os.unlink raises PermissionError.

        Cleanup errors (PermissionError, OSError) are swallowed so that GC
        finalizers and exception handling paths never themselves raise.
        """
        import os
        from unittest.mock import patch

        with patch.object(os, "unlink", side_effect=PermissionError("denied")):
            # Must not raise even though os.unlink will fail
            _cleanup_temp_memmap("/nonexistent/dir", "/nonexistent/dir/matrix.dat")


class TestParseChunkRowCountMismatch:
    """Verify _parse_chunk_to_memmap raises on row-count mismatch."""

    def test_parse_chunk_row_count_mismatch(self, tmp_path: Path) -> None:
        """Row-count mismatch raises RuntimeError."""
        from jamma.io.matrix_reader import _parse_chunk_to_memmap

        # Create a 5-row, 3-column file
        matrix = np.ones((5, 3), dtype=np.float64)
        txt_path = tmp_path / "short.txt"
        np.savetxt(txt_path, matrix, fmt="%.6f", delimiter="\t")

        # Create a memmap target with shape (10, 3) — deliberately wrong
        mm_path = str(tmp_path / "matrix.dat")
        mm = np.memmap(mm_path, dtype=np.float64, mode="w+", shape=(10, 3))
        del mm

        file_size = txt_path.stat().st_size
        # Pass n_rows=10 but file only has 5 rows — triggers row-count mismatch
        args = (
            str(txt_path),
            mm_path,
            (10, 3),
            "float64",
            0,
            file_size,
            0,
            10,
            "\t",
        )
        with pytest.raises(RuntimeError, match="Row count mismatch"):
            _parse_chunk_to_memmap(args)


class TestMultiWorkerCorrectness:
    """Verify multi-worker and single-worker parsing produces correct results."""

    def test_read_matrix_parallel_bounded_memory(self, tmp_path: Path) -> None:
        """Multi-worker parses and reassembles 1000x20 matrix."""
        rng = np.random.default_rng(314)
        matrix = rng.standard_normal((1000, 20))
        path = tmp_path / "bounded.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(
            path, n_workers=2, delimiter="\t", min_rows_for_parallel=500
        )
        assert result.shape == (1000, 20)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(
            result, np.atleast_2d(np.loadtxt(path, dtype=np.float64))
        )

    def test_read_matrix_parallel_large_chunk(self, tmp_path: Path) -> None:
        """Single worker (n_workers=1) handles degenerate single-chunk case."""
        rng = np.random.default_rng(271)
        matrix = rng.standard_normal((500, 10))
        path = tmp_path / "single_chunk.txt"
        np.savetxt(path, matrix, fmt="%.10g", delimiter="\t")

        result = read_matrix_parallel(
            path, n_workers=1, delimiter="\t", min_rows_for_parallel=200
        )
        assert result.shape == (500, 10)
        expected = np.loadtxt(path, dtype=np.float64)
        np.testing.assert_array_equal(result, np.atleast_2d(expected))
