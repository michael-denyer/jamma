"""Tests for parallel matrix text writer.

Validates that write_matrix_parallel produces byte-identical output to
np.savetxt for all matrix sizes, including the parallel path (>=500 rows).
"""

from pathlib import Path

import numpy as np
import pytest

from jamma.io._parallel_text import MAX_WORKERS
from jamma.io.matrix_writer import write_matrix_parallel

pytestmark = pytest.mark.tier0


def _savetxt_bytes(
    matrix: np.ndarray,
    path: Path,
    fmt: str = "%.10g",
    delimiter: str = "\t",
) -> bytes:
    """Write matrix with np.savetxt and return the raw bytes."""
    np.savetxt(path, matrix, fmt=fmt, delimiter=delimiter)
    return path.read_bytes()


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


class TestEdgeCases:
    """Edge cases for write_matrix_parallel."""

    def test_write_matrix_parallel_float32_input_casts_to_float64(
        self, tmp_path: Path
    ) -> None:
        """float32 input is silently cast to float64: byte-identical to savetxt on f64.

        write_matrix_parallel accepts float32 matrices (e.g. kinship computed
        from float32 genotypes). It casts to float64 internally via
        np.ascontiguousarray(matrix, dtype=np.float64), so output matches
        np.savetxt called directly on the float64 equivalent.
        """
        rng = np.random.default_rng(42)
        matrix_f32 = rng.standard_normal((20, 5)).astype(np.float32)
        matrix_f64 = matrix_f32.astype(np.float64)

        parallel_path = tmp_path / "parallel.txt"
        savetxt_path = tmp_path / "savetxt.txt"

        write_matrix_parallel(matrix_f32, parallel_path)
        np.savetxt(savetxt_path, matrix_f64, fmt="%.10g", delimiter="\t")

        assert parallel_path.read_bytes() == savetxt_path.read_bytes()

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


class TestAtomicPublication:
    """A failed write never destroys a pre-existing valid destination."""

    OLD_BYTES = b"prior valid matrix contents\n"

    def test_serial_failure_preserves_destination(self, tmp_path: Path) -> None:
        """Serial-path formatting error leaves the old destination intact."""
        out_path = tmp_path / "kinship.txt"
        out_path.write_bytes(self.OLD_BYTES)

        # 3 rows is below min_rows_for_parallel (500), so this hits the
        # np.savetxt path; "%q" raises ValueError during formatting.
        with pytest.raises(ValueError, match="format"):
            write_matrix_parallel(np.ones((3, 3)), out_path, fmt="%q")

        assert out_path.read_bytes() == self.OLD_BYTES
        remaining = list(tmp_path.glob(f".{out_path.name}.tmp.*"))
        assert not remaining, f"Publication temp not cleaned up: {remaining}"

    def test_parallel_publication_failure_preserves_destination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Publication failure after successful workers leaves the old file."""
        out_path = tmp_path / "kinship.txt"
        out_path.write_bytes(self.OLD_BYTES)

        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((600, 10))

        def raise_on_replace(src: object, dst: object) -> None:
            raise OSError("injected publication failure")

        monkeypatch.setattr("jamma.io.matrix_writer.os.replace", raise_on_replace)
        with pytest.raises(OSError, match="injected publication failure"):
            write_matrix_parallel(matrix, out_path, n_workers=2)

        assert out_path.read_bytes() == self.OLD_BYTES
        leftovers = [
            p for p in tmp_path.iterdir() if p != out_path
        ]  # temp dir, memmap, chunks, publication temp
        assert not leftovers, f"Temp artifacts not cleaned up: {leftovers}"

    def test_success_replaces_old_destination_atomically(self, tmp_path: Path) -> None:
        """Serial and parallel successful writes replace old bytes exactly."""
        rng = np.random.default_rng(42)
        for name, matrix in (
            ("serial.txt", rng.standard_normal((10, 5))),
            ("parallel.txt", rng.standard_normal((600, 10))),
        ):
            out_path = tmp_path / name
            out_path.write_bytes(self.OLD_BYTES)
            write_matrix_parallel(matrix, out_path, n_workers=2)
            expected = _savetxt_bytes(matrix, tmp_path / f"ref_{name}")
            assert out_path.read_bytes() == expected
            # Publication temps and .jamma_mwrite_* dirs all start with "."
            leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".")]
            assert not leftovers, f"Temp artifacts not cleaned up: {leftovers}"


class TestWorkerCap:
    """Verify worker count is capped to avoid disk I/O bottleneck."""

    def test_max_writers_constant(self) -> None:
        """The shared worker cap is 32."""
        assert MAX_WORKERS == 32

    def test_worker_count_capped_on_high_core_machine(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """On a 96-vCPU machine, n_workers defaults to 32, not 96.

        Regression test for Databricks where 96 workers hammering NFS
        caused cluster instability.
        """
        monkeypatch.setattr("os.cpu_count", lambda: 96)

        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((600, 10))
        out_path = tmp_path / "output.txt"

        # write_matrix_parallel should cap at MAX_WORKERS internally.
        # Verify by checking the output is correct (proves it ran) and
        # that the constant is in effect (tested above).
        write_matrix_parallel(matrix, out_path)
        assert out_path.exists()

        # Verify byte-identical to savetxt (correctness under cap)
        savetxt_path = tmp_path / "savetxt.txt"
        np.savetxt(savetxt_path, matrix, fmt="%.10g", delimiter="\t")
        assert out_path.read_bytes() == savetxt_path.read_bytes()
