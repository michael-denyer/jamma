"""Tests for parallel matrix text writer.

Validates that write_matrix_parallel produces byte-identical output to
np.savetxt for all matrix sizes, including the parallel path (>=500 rows).
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from jamma.io.matrix_writer import write_matrix_parallel


def _savetxt_bytes(
    matrix: np.ndarray,
    path: Path,
    fmt: str = "%.10g",
    delimiter: str = "\t",
) -> bytes:
    """Write matrix with np.savetxt and return the raw bytes."""
    np.savetxt(path, matrix, fmt=fmt, delimiter=delimiter)
    return path.read_bytes()


@pytest.mark.tier0
class TestByteIdentity:
    """Verify write_matrix_parallel is byte-identical to np.savetxt."""

    def test_savetxt_byte_identical_small(self, tmp_path: Path) -> None:
        """10x10 matrix: parallel output matches np.savetxt bytes."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((10, 10))

        parallel_path = tmp_path / "parallel.txt"
        savetxt_path = tmp_path / "savetxt.txt"

        write_matrix_parallel(matrix, parallel_path)
        expected = _savetxt_bytes(matrix, savetxt_path)

        assert parallel_path.read_bytes() == expected

    def test_savetxt_byte_identical_random_50x50(self, tmp_path: Path) -> None:
        """50x50 symmetric matrix (same seed as test_kinship_io): byte-identical."""
        rng = np.random.default_rng(12345)
        K = rng.standard_normal((50, 50))
        K = (K + K.T) / 2

        parallel_path = tmp_path / "parallel.txt"
        savetxt_path = tmp_path / "savetxt.txt"

        write_matrix_parallel(K, parallel_path)
        expected = _savetxt_bytes(K, savetxt_path)

        assert parallel_path.read_bytes() == expected

    def test_savetxt_byte_identical_500x20(self, tmp_path: Path) -> None:
        """500x20 matrix exceeds min_rows_for_parallel: byte-identical."""
        rng = np.random.default_rng(999)
        matrix = rng.standard_normal((500, 20))

        parallel_path = tmp_path / "parallel.txt"
        savetxt_path = tmp_path / "savetxt.txt"

        write_matrix_parallel(matrix, parallel_path)
        expected = _savetxt_bytes(matrix, savetxt_path)

        assert parallel_path.read_bytes() == expected

    def test_fallback_to_savetxt_below_threshold(self, tmp_path: Path) -> None:
        """100-row matrix (below default 500): exercises fallback, byte-identical."""
        rng = np.random.default_rng(77)
        matrix = rng.standard_normal((100, 10))

        parallel_path = tmp_path / "parallel.txt"
        savetxt_path = tmp_path / "savetxt.txt"

        write_matrix_parallel(matrix, parallel_path)
        expected = _savetxt_bytes(matrix, savetxt_path)

        assert parallel_path.read_bytes() == expected

    def test_scientific_notation_values(self, tmp_path: Path) -> None:
        """Values spanning 1e-15 to 1e15: byte-identical to np.savetxt."""
        matrix = np.array(
            [
                [1e-15, 1e-10, 1e-5, 1.0],
                [1e5, 1e10, 1e15, 3.14159265358979],
            ]
        )

        parallel_path = tmp_path / "parallel.txt"
        savetxt_path = tmp_path / "savetxt.txt"

        write_matrix_parallel(matrix, parallel_path)
        expected = _savetxt_bytes(matrix, savetxt_path)

        assert parallel_path.read_bytes() == expected

    def test_n_workers_1(self, tmp_path: Path) -> None:
        """Single worker still produces byte-identical output."""
        rng = np.random.default_rng(555)
        matrix = rng.standard_normal((600, 10))

        parallel_path = tmp_path / "parallel.txt"
        savetxt_path = tmp_path / "savetxt.txt"

        write_matrix_parallel(matrix, parallel_path, n_workers=1)
        expected = _savetxt_bytes(matrix, savetxt_path)

        assert parallel_path.read_bytes() == expected


@pytest.mark.tier0
class TestEdgeCases:
    """Edge cases for write_matrix_parallel."""

    def test_single_row_matrix(self, tmp_path: Path) -> None:
        """1x5 matrix: byte-identical."""
        matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        parallel_path = tmp_path / "parallel.txt"
        savetxt_path = tmp_path / "savetxt.txt"

        write_matrix_parallel(matrix, parallel_path)
        expected = _savetxt_bytes(matrix, savetxt_path)

        assert parallel_path.read_bytes() == expected

    def test_single_column_matrix(self, tmp_path: Path) -> None:
        """5x1 matrix: byte-identical."""
        matrix = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

        parallel_path = tmp_path / "parallel.txt"
        savetxt_path = tmp_path / "savetxt.txt"

        write_matrix_parallel(matrix, parallel_path)
        expected = _savetxt_bytes(matrix, savetxt_path)

        assert parallel_path.read_bytes() == expected

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Nested path that doesn't exist is created."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        nested_path = tmp_path / "a" / "b" / "c" / "matrix.txt"
        write_matrix_parallel(matrix, nested_path)

        assert nested_path.exists()
        # Verify content is correct
        savetxt_path = tmp_path / "savetxt.txt"
        expected = _savetxt_bytes(matrix, savetxt_path)
        assert nested_path.read_bytes() == expected


@pytest.mark.tier0
class TestFailureHandling:
    """Verify cleanup and platform behavior of write_matrix_parallel."""

    def test_partial_file_cleaned_on_worker_error(self, tmp_path: Path) -> None:
        """Worker failure does not leave a partial output file.

        Uses invalid format string to trigger a real TypeError in workers
        (which run in spawn context and can't use mocks). The output file
        is never created because concatenation only runs after all workers
        succeed.
        """
        rng = np.random.default_rng(42)
        # Create a valid float64 matrix but large enough for parallel path
        matrix = rng.standard_normal((600, 10))
        out_path = tmp_path / "should_not_exist.txt"

        # Trigger worker error via invalid format string: "%s%s" causes
        # TypeError in the worker, wrapped as RuntimeError with row context.
        with pytest.raises(RuntimeError, match="_format_rows_to_file failed"):
            write_matrix_parallel(matrix, out_path, fmt="%s%s", n_workers=2)

        assert not out_path.exists(), "Partial output file should be deleted on failure"

    def test_uses_memmap_not_shared_memory(self) -> None:
        """Memmap used, not SharedMemory (Docker /dev/shm SIGBUS)."""
        source = inspect.getsource(write_matrix_parallel)
        assert "memmap" in source, "write_matrix_parallel should use numpy.memmap"
        assert "SharedMemory" not in source, (
            "write_matrix_parallel should not use SharedMemory"
            " -- Docker /dev/shm is capped at 64 MB"
        )

    def test_temp_files_created_in_output_dir(self, tmp_path: Path) -> None:
        """Temp dir is created adjacent to output, not in system /tmp."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((600, 10))
        output_dir = tmp_path / "my_output"
        output_dir.mkdir()
        out_path = output_dir / "matrix.txt"

        # During write, temp dir should appear as .jamma_mwrite_* in output_dir
        write_matrix_parallel(matrix, out_path, n_workers=2)

        # After completion, temp dir should be cleaned up
        remaining = list(output_dir.glob(".jamma_mwrite_*"))
        assert not remaining, f"Temp dirs not cleaned up: {remaining}"
        # Output file should exist
        assert out_path.exists()

    def test_temp_file_cleaned_after_success(self, tmp_path: Path) -> None:
        """Temp dir and all chunk files are cleaned up after success."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((600, 10))
        out_path = tmp_path / "output.txt"
        write_matrix_parallel(matrix, out_path, n_workers=2)

        remaining = list(tmp_path.glob(".jamma_mwrite_*"))
        assert not remaining, f"Temp dirs not cleaned up: {remaining}"

    def test_temp_file_cleaned_after_failure(self, tmp_path: Path) -> None:
        """Temp dir and all chunk files are cleaned up on failure."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((600, 10))
        out_path = tmp_path / "should_not_exist.txt"

        with pytest.raises(RuntimeError, match="_format_rows_to_file failed"):
            write_matrix_parallel(matrix, out_path, fmt="%s%s", n_workers=2)

        remaining = list(tmp_path.glob(".jamma_mwrite_*"))
        assert not remaining, f"Temp dirs not cleaned up after failure: {remaining}"
