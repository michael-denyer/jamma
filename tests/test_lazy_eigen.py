"""Tests for LazyEigen class and eigendecompose_kinship_lazy function.

Verifies that the lazy eigendecomposition path (Householder-based rotation)
produces identical results to the standard U.T @ target path.
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma import jlinalg

# D&C pipeline is required for eigh_factored / rotate_via_householder.
_DC_PIPELINE_AVAILABLE = jlinalg.HAS_C_EXTENSION and hasattr(jlinalg, "eigh_factored")

pytestmark = [
    pytest.mark.skipif(
        not _DC_PIPELINE_AVAILABLE, reason="Requires jlinalg D&C pipeline"
    ),
]


def _make_spd_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """Create a symmetric positive-definite matrix."""
    A = rng.standard_normal((n, n))
    return A @ A.T / n + 0.1 * np.eye(n)


def _eigendecompose_standard(K: np.ndarray, threshold: float = 1e-10):
    """Standard eigendecompose_kinship for comparison."""
    from jamma.lmm.eigen import eigendecompose_kinship

    return eigendecompose_kinship(K.copy(), threshold=threshold, check_memory=False)


def _build_U_from_factored(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build full U from eigh_factored for self-consistent comparison.

    Returns (eigenvalues, U) using the D&C pipeline, avoiding sign ambiguity
    between vendor LAPACK and jlinalg D&C.
    """
    K_work = K.copy()
    eigenvalues, tau, V = jlinalg.eigh_factored(K_work)
    # K_work now contains Householder vectors from dsytrd
    eye = np.eye(K.shape[0], dtype=np.float64)
    # rotate_via_householder returns V.T @ (Q^T @ eye) = U.T
    UT = jlinalg.rotate_via_householder(K_work, tau, V, eye)
    return eigenvalues, UT.T


def _try_lazy_or_skip(func, *args, **kwargs):
    """Call func, skip test if NotImplementedError (vendor LAPACK only)."""
    try:
        return func(*args, **kwargs)
    except NotImplementedError:
        pytest.skip("D&C pipeline unreachable (vendor LAPACK active)")


@pytest.mark.tier0
class TestLazyEigen:
    """Tests for LazyEigen rotate() parity with U.T @ target."""

    def test_eigenvalues_match(self):
        """Eigenvalues from lazy path match standard eigendecompose_kinship."""
        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        n = 100
        K = _make_spd_matrix(rng, n)

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), check_memory=False
        )
        evals_std, _ = _eigendecompose_standard(K)

        np.testing.assert_allclose(lazy.eigenvalues, evals_std, rtol=1e-14, atol=1e-14)

    def test_rotate_matrix_parity(self):
        """rotate((100,20) target) matches U.T @ target (self-consistent D&C U)."""
        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        n, m = 100, 20
        K = _make_spd_matrix(rng, n)
        target = rng.standard_normal((n, m))

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), check_memory=False
        )
        _, U = _build_U_from_factored(K)

        result_lazy = lazy.rotate(target)
        result_std = U.T @ target

        np.testing.assert_allclose(result_lazy, result_std, rtol=5e-12)

    def test_rotate_vector_parity(self):
        """rotate((100,) vector) matches U.T @ y (self-consistent D&C U)."""
        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        n = 100
        K = _make_spd_matrix(rng, n)
        y = rng.standard_normal(n)

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), check_memory=False
        )
        _, U = _build_U_from_factored(K)

        result_lazy = lazy.rotate(y)
        result_std = U.T @ y

        assert result_lazy.ndim == 1
        np.testing.assert_allclose(result_lazy, result_std, rtol=5e-12)

    def test_rotate_single_column(self):
        """rotate((100,1) target) matches U.T @ target (self-consistent D&C U)."""
        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        n = 100
        K = _make_spd_matrix(rng, n)
        target = rng.standard_normal((n, 1))

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), check_memory=False
        )
        _, U = _build_U_from_factored(K)

        result_lazy = lazy.rotate(target)
        result_std = U.T @ target

        assert result_lazy.shape == (n, 1)
        np.testing.assert_allclose(result_lazy, result_std, rtol=5e-12)

    def test_threshold_zeroing(self):
        """Eigenvalues below threshold are zeroed, same as standard path."""
        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        n = 100
        K = _make_spd_matrix(rng, n)

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), threshold=1e-10, check_memory=False
        )
        evals_std, _ = _eigendecompose_standard(K, threshold=1e-10)

        # Both paths should have the same zeroed eigenvalues
        np.testing.assert_array_equal(lazy.eigenvalues == 0, evals_std == 0)

    def test_free_tridiag_then_rotate_raises(self):
        """free_tridiag_eigenvectors() then rotate() raises RuntimeError."""
        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        n = 50
        K = _make_spd_matrix(rng, n)
        target = rng.standard_normal((n, 5))

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), check_memory=False
        )
        lazy.free_tridiag_eigenvectors()

        with pytest.raises(RuntimeError, match="freed"):
            lazy.rotate(target)

    def test_free_householder_then_rotate_raises(self):
        """free_householder() then rotate() raises RuntimeError."""
        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        n = 50
        K = _make_spd_matrix(rng, n)
        target = rng.standard_normal((n, 5))

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), check_memory=False
        )
        # Free householder without freeing V first (wrong order)
        lazy.free_householder()

        with pytest.raises(RuntimeError, match="Householder state has been freed"):
            lazy.rotate(target)

    def test_free_householder_reduces_memory(self):
        """memory_gb decreases after free_householder()."""
        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        n = 100
        K = _make_spd_matrix(rng, n)

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), check_memory=False
        )

        mem_before = lazy.memory_gb
        lazy.free_tridiag_eigenvectors()
        mem_after_v = lazy.memory_gb
        lazy.free_householder()
        mem_after_all = lazy.memory_gb

        assert mem_after_v < mem_before
        assert mem_after_all < mem_after_v

    def test_large_scale_parity(self):
        """N=500, M=50 target parity (self-consistent D&C U).

        At N=500 the indirect path (build U then U.T @ target) accumulates
        more FP error than the direct rotate path. Max relative diff ~4e-9
        on CI (Linux OpenBLAS) from the extra N x N matmul in the reference.
        rtol=5e-9 is still very tight and confirms correctness.
        """
        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        n, m = 500, 50
        K = _make_spd_matrix(rng, n)
        target = rng.standard_normal((n, m))

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), check_memory=False
        )
        _, U = _build_U_from_factored(K)

        result_lazy = lazy.rotate(target)
        result_std = U.T @ target

        np.testing.assert_allclose(result_lazy, result_std, rtol=5e-9)


@pytest.mark.tier0
class TestEigendecomposeKinshipLazy:
    """Tests for eigendecompose_kinship_lazy function behavior."""

    def test_returns_lazy_eigen(self):
        """eigendecompose_kinship_lazy returns LazyEigen, not tuple."""
        from jamma.lmm.eigen import eigendecompose_kinship_lazy
        from jamma.lmm.lazy_eigen import LazyEigen

        rng = np.random.default_rng(42)
        K = _make_spd_matrix(rng, 50)

        lazy = _try_lazy_or_skip(
            eigendecompose_kinship_lazy, K.copy(), check_memory=False
        )
        assert isinstance(lazy, LazyEigen)

    def test_not_implemented_without_c_extension(self):
        """Raises NotImplementedError when C extension unavailable."""
        from unittest.mock import patch

        from jamma.lmm.eigen import eigendecompose_kinship_lazy

        rng = np.random.default_rng(42)
        K = _make_spd_matrix(rng, 50)

        with patch("jamma.lmm.eigen.jlinalg") as mock_jlinalg:
            mock_jlinalg.HAS_C_EXTENSION = False
            with pytest.raises(NotImplementedError, match="C extension"):
                eigendecompose_kinship_lazy(K.copy(), check_memory=False)
