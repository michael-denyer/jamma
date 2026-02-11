"""Tests for eigendecomposition file I/O and reuse.

Validates:
- GEMMA-compatible file format (.10g precision, no headers)
- Round-trip precision for eigenvalues and eigenvectors
- Dimension validation on read
- Edge cases (empty files, single value, nested dirs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.eigen_io import (
    read_eigen_files,
    read_eigenvalues,
    read_eigenvectors,
    write_eigen_files,
    write_eigenvalues,
    write_eigenvectors,
)

# =============================================================================
# File format tests
# =============================================================================


class TestEigenvalueFormat:
    """Verify eigenvalue file format matches GEMMA .eigenD.txt."""

    def test_write_eigenvalues_format(self, tmp_path: Path) -> None:
        """Eigenvalue file has one value per line, .10g format, no header."""
        values = np.array([0.001, 1.0, 2.5, 100.0, 12345.6789012345])
        path = tmp_path / "test.eigenD.txt"
        write_eigenvalues(values, path)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5

        # Each line should be the .10g formatted value
        for i, line in enumerate(lines):
            expected = f"{values[i]:.10g}"
            assert line == expected, f"Line {i}: got {line!r}, expected {expected!r}"

    def test_write_eigenvectors_format(self, tmp_path: Path) -> None:
        """Eigenvector file has tab-separated rows, .10g format, no header."""
        matrix = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        path = tmp_path / "test.eigenU.txt"
        write_eigenvectors(matrix, path)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3

        for i, line in enumerate(lines):
            parts = line.split("\t")
            assert len(parts) == 3
            for j, part in enumerate(parts):
                expected = f"{matrix[i, j]:.10g}"
                assert part == expected

    def test_eigenvalues_ascending_order_preserved(self, tmp_path: Path) -> None:
        """Ascending eigenvalue order from eigh is preserved through write/read."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((20, 20))
        sym = A + A.T
        eigenvalues, _ = np.linalg.eigh(sym)

        # eigh returns ascending order
        assert np.all(eigenvalues[:-1] <= eigenvalues[1:])

        path = tmp_path / "test.eigenD.txt"
        write_eigenvalues(eigenvalues, path)
        loaded = read_eigenvalues(path)

        # Order is preserved
        assert np.all(loaded[:-1] <= loaded[1:])
        np.testing.assert_allclose(loaded, eigenvalues, rtol=1e-9)


# =============================================================================
# Round-trip precision tests
# =============================================================================


class TestRoundTripPrecision:
    """Verify .10g format preserves sufficient precision for LMM."""

    def test_eigenvalue_round_trip_precision(self, tmp_path: Path) -> None:
        """100 random eigenvalues survive write/read within rtol=1e-9."""
        rng = np.random.default_rng(123)
        # Generate eigenvalues spanning several orders of magnitude
        original = np.sort(rng.uniform(0.001, 1000.0, size=100))

        path = tmp_path / "eigenD.txt"
        write_eigenvalues(original, path)
        loaded = read_eigenvalues(path)

        np.testing.assert_allclose(loaded, original, rtol=1e-9)

    def test_eigenvector_round_trip_precision(self, tmp_path: Path) -> None:
        """50x50 orthogonal matrix survives write/read within rtol=1e-9."""
        rng = np.random.default_rng(456)
        A = rng.standard_normal((50, 50))
        sym = A + A.T
        _, eigenvectors = np.linalg.eigh(sym)

        # Eigenvectors from eigh are orthonormal
        path = tmp_path / "eigenU.txt"
        write_eigenvectors(eigenvectors, path)
        loaded = read_eigenvectors(path)

        np.testing.assert_allclose(loaded, eigenvectors, rtol=1e-9)

    def test_eigen_files_round_trip(self, tmp_path: Path) -> None:
        """write_eigen_files + read_eigen_files round-trip both arrays."""
        rng = np.random.default_rng(789)
        A = rng.standard_normal((30, 30))
        sym = A + A.T
        eigenvalues, eigenvectors = np.linalg.eigh(sym)

        d_path, u_path = write_eigen_files(
            eigenvalues, eigenvectors, tmp_path, prefix="roundtrip"
        )

        loaded_d, loaded_u = read_eigen_files(d_path, u_path)

        np.testing.assert_allclose(loaded_d, eigenvalues, rtol=1e-9)
        np.testing.assert_allclose(loaded_u, eigenvectors, rtol=1e-9)


# =============================================================================
# Dimension validation tests
# =============================================================================


class TestDimensionValidation:
    """Verify read_eigen_files catches dimension mismatches."""

    def test_read_eigen_files_dimension_mismatch(self, tmp_path: Path) -> None:
        """Mismatched eigenvalue count vs eigenvector dimensions raises ValueError."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(8), u_path)

        with pytest.raises(ValueError, match="does not match"):
            read_eigen_files(d_path, u_path)

    def test_read_eigen_files_n_samples_mismatch(self, tmp_path: Path) -> None:
        """n_samples validation catches wrong expected count."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(10), u_path)

        with pytest.raises(ValueError, match="does not match expected n_samples=12"):
            read_eigen_files(d_path, u_path, n_samples=12)

    def test_read_eigen_files_consistent_dimensions(self, tmp_path: Path) -> None:
        """Consistent eigen pair with matching n_samples succeeds."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(10), u_path)

        eigenvalues, eigenvectors = read_eigen_files(d_path, u_path, n_samples=10)
        assert eigenvalues.shape == (10,)
        assert eigenvectors.shape == (10, 10)

    def test_read_eigen_files_no_n_samples_validation(self, tmp_path: Path) -> None:
        """Omitting n_samples skips that validation."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"

        write_eigenvalues(np.ones(10), d_path)
        write_eigenvectors(np.eye(10), u_path)

        eigenvalues, eigenvectors = read_eigen_files(d_path, u_path)
        assert eigenvalues.shape == (10,)
        assert eigenvectors.shape == (10, 10)


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Edge case handling for eigen I/O."""

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """Writing to nested path creates parent directories."""
        nested = tmp_path / "a" / "b" / "c" / "test.eigenD.txt"
        write_eigenvalues(np.array([1.0, 2.0]), nested)
        assert nested.exists()
        loaded = read_eigenvalues(nested)
        assert len(loaded) == 2

    def test_read_eigenvalues_empty_file(self, tmp_path: Path) -> None:
        """Empty eigenD file returns empty array (numpy warns but does not raise)."""
        path = tmp_path / "empty.eigenD.txt"
        path.write_text("")

        # numpy >= 2.0 returns empty float64 array with a UserWarning
        with pytest.warns(UserWarning, match="no data"):
            result = read_eigenvalues(path)

        assert result.size == 0
        assert result.dtype == np.float64

    def test_write_read_single_eigenvalue(self, tmp_path: Path) -> None:
        """1x1 matrix edge case preserves correct shapes."""
        eigenvalues = np.array([3.14])
        eigenvectors = np.array([[1.0]])

        d_path = tmp_path / "single.eigenD.txt"
        u_path = tmp_path / "single.eigenU.txt"

        write_eigenvalues(eigenvalues, d_path)
        write_eigenvectors(eigenvectors, u_path)

        loaded_d = read_eigenvalues(d_path)
        loaded_u = read_eigenvectors(u_path)

        # np.loadtxt on a single-line file returns a 0-d array for 1D
        # and a 1D array for a single-row matrix. Verify shape handling.
        assert loaded_d.ndim <= 1
        assert np.isclose(float(loaded_d), 3.14, rtol=1e-9)

        # Single row eigenvector file: np.loadtxt returns 1D
        assert np.isclose(float(loaded_u.flat[0]), 1.0)
