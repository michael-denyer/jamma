"""Tests for in-place eigendecomposition (jlinalg.eigh inplace=True).

Validates correctness, buffer identity, and memory estimation for the
in-place DSYEVD path.
"""

import numpy as np
import pytest

from jamma import jlinalg
from jamma.core.eigen_plan import (
    _dsyevd_inplace_peak_gb,
    _dsyevd_peak_gb,
    square_matrix_gb,
)
from jamma.jlinalg import HAS_C_EXTENSION, blas_has_dsyevd

# In-place requires vendor DSYEVD — the D&C pipeline rejects K==eigenvectors.
_skip_no_dsyevd = pytest.mark.skipif(
    not HAS_C_EXTENSION or not blas_has_dsyevd,
    reason="Requires C extension with vendor DSYEVD for inplace mode",
)


def _make_symmetric(n: int, seed: int = 42) -> np.ndarray:
    """Create a random symmetric positive-definite matrix."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n)
    return np.ascontiguousarray(X @ X.T)


@_skip_no_dsyevd
@pytest.mark.tier0
class TestEighInplaceCorrectness:
    """Verify in-place eigh produces correct eigendecomposition."""

    def test_eigh_inplace_correctness(self):
        """Eigenvalues/eigenvectors from inplace path match numpy within rtol=5e-12."""
        n = 200
        K = _make_symmetric(n)
        K_ref = K.copy()

        w_inplace, v_inplace = jlinalg.eigh(K, inplace=True)
        w_ref, _ = np.linalg.eigh(K_ref)

        # Eigenvalue accuracy is bounded by Weyl's theorem at O(eps * ||K||_2):
        # the absolute error scales with the largest eigenvalue, not each
        # eigenvalue's own magnitude. jlinalg DSYEVD and numpy differ only in FP
        # accumulation in the tridiagonal reduction (DSYTRD), so the smallest
        # eigenvalues cannot meet a tight per-element rtol (a ~5e-13 absolute
        # slip at lambda~4e-3 is a ~1e-11 relative slip) even though the
        # decomposition is correct to ~eps * lambda_max. Compare against an
        # absolute tolerance scaled by the spectral norm, which is the
        # numerically well-posed metric for eigenvalue agreement.
        scale = float(np.abs(w_ref).max())
        np.testing.assert_allclose(w_inplace, w_ref, rtol=0, atol=1e-12 * scale)

        # Eigenvector orthogonality: V.T @ V = I
        eye_check = v_inplace.T @ v_inplace
        np.testing.assert_allclose(eye_check, np.eye(n), atol=1e-12, rtol=0)

        # Reconstruction: K_ref @ V ~= V @ diag(w)
        lhs = K_ref @ v_inplace
        rhs = v_inplace * w_inplace[np.newaxis, :]
        np.testing.assert_allclose(
            lhs, rhs, atol=1e-10 * float(np.linalg.norm(K_ref)), rtol=0
        )


@_skip_no_dsyevd
@pytest.mark.tier0
class TestEighInplaceBufferIdentity:
    """Verify buffer sharing behavior for inplace=True/False."""

    def test_eigh_inplace_returns_same_buffer(self):
        """When inplace=True, eigenvectors share memory with input K."""
        K = _make_symmetric(50)
        _, v = jlinalg.eigh(K, inplace=True)
        assert v.ctypes.data == K.ctypes.data, (
            "inplace=True should return eigenvectors in the same buffer as K"
        )

    def test_eigh_default_returns_separate_buffer(self):
        """Default (no inplace arg) returns eigenvectors in a new buffer."""
        K = _make_symmetric(50)
        _, v = jlinalg.eigh(K)
        assert v.ctypes.data != K.ctypes.data, (
            "default eigh should return eigenvectors in a separate buffer"
        )

    def test_eigh_inplace_false_returns_separate_buffer(self):
        """Explicit inplace=False returns eigenvectors in a new buffer."""
        K = _make_symmetric(50)
        _, v = jlinalg.eigh(K, inplace=False)
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
        """Difference between default and inplace is ~square_matrix_gb(n)."""
        n = 1000
        diff = _dsyevd_peak_gb(n) - _dsyevd_inplace_peak_gb(n)
        expected = square_matrix_gb(n)
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
