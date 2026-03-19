"""Tests for in-place eigendecomposition (jlinalg.eigh inplace=True).

Validates correctness, buffer identity, and memory estimation for the
in-place DSYEVD path added in Phase 86.
"""

import numpy as np
import pytest

from jamma import jlinalg
from jamma.core.memory import (
    _dsyevd_inplace_peak_gb,
    _dsyevd_peak_gb,
    _square_matrix_gb,
)
from jamma.jlinalg import HAS_C_EXTENSION, blas_has_dsyevd

# In-place correctness and buffer identity tests require vendor DSYEVD.
# Without it, jlinalg.eigh(inplace=True) raises RuntimeError from the
# D&C pipeline guard (K==eigenvectors is unsafe for dsytrd+dstedc+dormtr).
_skip_no_dsyevd = pytest.mark.skipif(
    not HAS_C_EXTENSION or not blas_has_dsyevd,
    reason="Requires C extension with vendor DSYEVD for inplace mode",
)


@_skip_no_dsyevd
@pytest.mark.tier0
class TestEighInplaceCorrectness:
    """Verify in-place eigh produces correct eigendecomposition."""

    def test_eigh_inplace_correctness(self):
        """Eigenvalues/eigenvectors from inplace path match numpy within rtol=1e-12."""
        n = 200
        rng = np.random.RandomState(42)
        X = rng.randn(n, n)
        K = X @ X.T
        K_copy = K.copy()

        w_inplace, v_inplace = jlinalg.eigh(K, inplace=True)
        w_ref, v_ref = np.linalg.eigh(K_copy)

        # Eigenvalues match
        np.testing.assert_allclose(w_inplace, w_ref, rtol=1e-12, atol=1e-14)

        # Eigenvector orthogonality: V.T @ V = I
        eye_check = v_inplace.T @ v_inplace
        np.testing.assert_allclose(eye_check, np.eye(n), atol=1e-12, rtol=0)

        # Reconstruction: K_orig @ V ~= V @ diag(w)
        K_orig = K_copy.copy()
        lhs = K_orig @ v_inplace
        rhs = v_inplace * w_inplace[np.newaxis, :]
        norm_K = np.linalg.norm(K_orig)
        np.testing.assert_allclose(lhs, rhs, atol=1e-10 * norm_K, rtol=0)


@_skip_no_dsyevd
@pytest.mark.tier0
class TestEighInplaceBufferIdentity:
    """Verify buffer sharing behavior for inplace=True/False."""

    def test_eigh_inplace_returns_same_buffer(self):
        """When inplace=True, eigenvectors share memory with input K."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 50)
        K = X @ X.T
        K = np.ascontiguousarray(K)

        w, v = jlinalg.eigh(K, inplace=True)
        assert v.ctypes.data == K.ctypes.data, (
            "inplace=True should return eigenvectors in the same buffer as K"
        )

    def test_eigh_default_returns_separate_buffer(self):
        """Default (no inplace arg) returns eigenvectors in a new buffer."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 50)
        K = X @ X.T
        K = np.ascontiguousarray(K)

        w, v = jlinalg.eigh(K)
        assert v.ctypes.data != K.ctypes.data, (
            "default eigh should return eigenvectors in a separate buffer"
        )

    def test_eigh_inplace_false_returns_separate_buffer(self):
        """Explicit inplace=False returns eigenvectors in a new buffer."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 50)
        K = X @ X.T
        K = np.ascontiguousarray(K)

        w, v = jlinalg.eigh(K, inplace=False)
        assert v.ctypes.data != K.ctypes.data, (
            "inplace=False should return eigenvectors in a separate buffer"
        )


@pytest.mark.tier0
class TestMemoryEstimateInplaceVsDefault:
    """Verify in-place memory estimate saves exactly one N x N matrix."""

    def test_inplace_less_than_default(self):
        """In-place peak is strictly less than default peak."""
        n = 1000
        assert _dsyevd_inplace_peak_gb(n) < _dsyevd_peak_gb(n)

    def test_difference_is_one_square_matrix(self):
        """Difference between default and inplace is ~_square_matrix_gb(n)."""
        n = 1000
        diff = _dsyevd_peak_gb(n) - _dsyevd_inplace_peak_gb(n)
        expected = _square_matrix_gb(n)
        assert diff == pytest.approx(expected, rel=0.01), (
            f"Expected saving of {expected:.6f}GB, got {diff:.6f}GB"
        )

    def test_125k_savings(self):
        """At 125k samples, in-place saves ~125GB (one 125k x 125k matrix)."""
        n = 125_000
        diff = _dsyevd_peak_gb(n) - _dsyevd_inplace_peak_gb(n)
        # 125k x 125k x 8 bytes / 1e9 = 125 GB
        assert 124 < diff < 126

    def test_inplace_negative_raises(self):
        """Negative n raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be >= 0"):
            _dsyevd_inplace_peak_gb(-1)
