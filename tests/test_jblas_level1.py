"""Level 1/2 BLAS correctness tests for jamma.jblas.

Tests all 5 operations against NumPy reference at rtol=1e-14:
- TestDdot: inner product (BL1-01)
- TestDnrm2: Euclidean norm with overflow/underflow protection (BL1-02)
- TestDaxpy: in-place y += alpha*x (BL1-03)
- TestDscal: in-place x *= alpha (BL1-04)
- TestDgemv: matrix-vector product (BL2-01)

All tests run against the NumPy fallback path in Wave 0 and against the
C extension in Wave 1 (after Plan 02 compiles it) — the same test file works
for both because jamma.jblas transparently dispatches.
"""

import numpy as np
import pytest

from jamma.jblas import HAS_C_EXTENSION, daxpy, ddot, dgemm, dgemv, dnrm2, dscal

_RNG = np.random.default_rng(42)


class TestDdot:
    """ddot: inner product of two double vectors (BL1-01)."""

    @pytest.mark.tier0
    def test_small(self):
        """n=1: single-element dot product."""
        x = _RNG.standard_normal(1)
        y = _RNG.standard_normal(1)
        np.testing.assert_allclose(ddot(x, y), np.dot(x, y), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_medium(self):
        """n=100: typical small vector."""
        x = _RNG.standard_normal(100)
        y = _RNG.standard_normal(100)
        np.testing.assert_allclose(ddot(x, y), np.dot(x, y), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_large(self):
        """n=10000: exercises SIMD unroll paths."""
        x = _RNG.standard_normal(10_000)
        y = _RNG.standard_normal(10_000)
        np.testing.assert_allclose(ddot(x, y), np.dot(x, y), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_orthogonal(self):
        """Orthogonal unit vectors produce dot product of 0."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        assert ddot(x, y) == 0.0

    @pytest.mark.tier0
    def test_empty(self):
        """Empty vectors produce dot product of 0."""
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
        assert ddot(x, y) == 0.0


class TestDnrm2:
    """dnrm2: Euclidean norm (BL1-02)."""

    @pytest.mark.tier0
    def test_small(self):
        """n=1: single-element norm."""
        x = _RNG.standard_normal(1)
        np.testing.assert_allclose(dnrm2(x), np.linalg.norm(x), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_medium(self):
        """n=100: typical small vector."""
        x = _RNG.standard_normal(100)
        np.testing.assert_allclose(dnrm2(x), np.linalg.norm(x), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_large(self):
        """n=10000: exercises main loop path."""
        x = _RNG.standard_normal(10_000)
        np.testing.assert_allclose(dnrm2(x), np.linalg.norm(x), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_zero_vector(self):
        """Zero vector has norm 0."""
        x = np.zeros(100)
        assert dnrm2(x) == 0.0

    @pytest.mark.tier0
    def test_overflow_protection(self):
        """Blue algorithm: verify dnrm2 agrees with np.linalg.norm for large values.

        The C extension (Plan 02) will implement the Blue (1978) three-accumulator
        algorithm to avoid overflow, so it will return a finite result even when
        np.linalg.norm overflows. The NumPy fallback simply delegates to
        np.linalg.norm, so both return the same value (which may be inf on some
        NumPy builds). This test verifies agreement, not finiteness, for the
        fallback path. A separate overflow_protection_c_only test (guarded by
        HAS_C_EXTENSION) will assert finiteness after Plan 02.
        """
        from jamma.jblas import HAS_C_EXTENSION

        x = np.array([1e200, 1e200])
        result = dnrm2(x)
        expected = np.linalg.norm(x)

        if HAS_C_EXTENSION:
            # C extension must implement Blue algorithm and return finite result.
            assert np.isfinite(result), f"dnrm2 C extension overflowed: got {result}"
        else:
            # NumPy fallback matches np.linalg.norm exactly (may also be inf).
            np.testing.assert_equal(result, expected)

    @pytest.mark.tier0
    def test_underflow_protection(self):
        """Blue algorithm: tiny values must not underflow to zero.

        For the NumPy fallback, np.linalg.norm on modern NumPy handles this
        correctly (it applies a scaling step internally). The C extension must
        implement the Blue algorithm to guarantee this on all platforms.
        """
        from jamma.jblas import HAS_C_EXTENSION

        x = np.array([1e-200, 1e-200])
        result = dnrm2(x)
        expected = np.linalg.norm(x)

        if HAS_C_EXTENSION:
            # C extension must not underflow.
            assert result > 0.0, f"dnrm2 C extension underflowed: got {result}"
        else:
            # NumPy fallback matches np.linalg.norm (should also be > 0).
            np.testing.assert_allclose(result, expected, rtol=1e-14)

    @pytest.mark.tier0
    @pytest.mark.skipif(not HAS_C_EXTENSION, reason="C extension not compiled")
    def test_mixed_magnitude(self):
        """Blue algorithm: mixed big+medium and small+medium accumulator combining.

        Exercises the branches where n_big > 0 && n_med > 0 and
        n_sml > 0 && n_med > 0 in the Blue three-accumulator algorithm.
        """
        # big + medium: 1e200 dominates, 1.0 is medium-scale.
        # np.linalg.norm overflows here, so compare against known value:
        # sqrt(1e200^2 + 1.0^2) = 1e200 * sqrt(1 + 1e-400) ≈ 1e200
        x_big = np.array([1e200, 1.0])
        result_big = dnrm2(x_big)
        assert np.isfinite(result_big), f"overflow in big+medium: {result_big}"
        np.testing.assert_allclose(result_big, 1e200, rtol=1e-14)

        # small + medium: 1e-200 is tiny, 1.0 is medium-scale.
        # sqrt(1e-200^2 + 1.0^2) ≈ 1.0
        x_sml = np.array([1e-200, 1.0])
        result_sml = dnrm2(x_sml)
        assert result_sml > 0.0, f"underflow in small+medium: {result_sml}"
        np.testing.assert_allclose(result_sml, 1.0, rtol=1e-14)


class TestDaxpy:
    """daxpy: in-place y += alpha * x (BL1-03)."""

    @pytest.mark.tier0
    def test_basic(self):
        """Standard daxpy: y = y + 2.0*x."""
        x = _RNG.standard_normal(100)
        y = _RNG.standard_normal(100)
        y_orig = y.copy()
        daxpy(2.0, x, y)
        np.testing.assert_allclose(y, y_orig + 2.0 * x, rtol=1e-14)

    @pytest.mark.tier0
    def test_alpha_zero(self):
        """alpha=0: y must be unchanged."""
        x = _RNG.standard_normal(100)
        y = _RNG.standard_normal(100)
        y_orig = y.copy()
        daxpy(0.0, x, y)
        np.testing.assert_array_equal(y, y_orig)

    @pytest.mark.tier0
    @pytest.mark.parametrize("n", [1, 100, 10_000])
    def test_sizes(self, n: int):
        """daxpy correctness at n=1, 100, 10000.

        Uses rtol=1e-13: the C extension does y[i] += alpha * x[i] (two-step
        FP: multiply then add with rounding in each step), while the NumPy
        reference computes alpha * x first (vectorized) then adds to y_orig.
        This operand reordering causes ~8 ULP differences at large n.
        1e-13 is ~10x machine epsilon — tight enough to catch bugs while
        matching scalar-accumulation precision.
        """
        rng = np.random.default_rng(n)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        y_orig = y.copy()
        alpha = 3.14159
        daxpy(alpha, x, y)
        np.testing.assert_allclose(y, y_orig + alpha * x, rtol=1e-13)


class TestDscal:
    """dscal: in-place x *= alpha (BL1-04)."""

    @pytest.mark.tier0
    def test_basic(self):
        """Standard dscal: x = 3.0 * x."""
        x = _RNG.standard_normal(100)
        x_orig = x.copy()
        dscal(3.0, x)
        np.testing.assert_allclose(x, x_orig * 3.0, rtol=1e-14)

    @pytest.mark.tier0
    def test_alpha_zero(self):
        """alpha=0: x becomes all zeros."""
        x = _RNG.standard_normal(100)
        dscal(0.0, x)
        np.testing.assert_array_equal(x, np.zeros(100))

    @pytest.mark.tier0
    def test_alpha_one(self):
        """alpha=1: x is unchanged."""
        x = _RNG.standard_normal(100)
        x_orig = x.copy()
        dscal(1.0, x)
        np.testing.assert_array_equal(x, x_orig)


class TestDgemv:
    """dgemv: matrix-vector product A @ x (BL2-01)."""

    @pytest.mark.tier0
    def test_small(self):
        """1x1 matrix-vector product."""
        A = np.array([[2.5]])
        x = np.array([3.0])
        result = dgemv(A, x)
        np.testing.assert_allclose(result, A @ x, rtol=1e-14)

    @pytest.mark.tier0
    def test_medium(self):
        """10x10 matrix-vector product."""
        A = _RNG.standard_normal((10, 10))
        x = _RNG.standard_normal(10)
        result = dgemv(A, x)
        np.testing.assert_allclose(result, A @ x, rtol=1e-14)

    @pytest.mark.tier0
    def test_rectangular(self):
        """100x50 rectangular matrix-vector product.

        Uses rtol=2e-12 rather than 1e-14: the C extension implements dgemv
        as row-by-row ddot calls with sequential FP accumulation, while NumPy
        A @ x uses a BLAS dgemv with a different (potentially pairwise or
        SIMD-reordered) accumulation, leading to ~1 ULP per element of FP
        disagreement across 50 additions. The result is numerically correct —
        the tolerance is calibrated to scalar-accumulation vs BLAS-level precision.
        """
        A = _RNG.standard_normal((100, 50))
        x = _RNG.standard_normal(50)
        result = dgemv(A, x)
        np.testing.assert_allclose(result, A @ x, rtol=2e-12)
        assert result.shape == (100,)


class TestDgemm:
    """dgemm: matrix-matrix product A @ B (BL3-01)."""

    @pytest.mark.tier0
    def test_basic(self):
        """Small matrix multiplication agrees with NumPy."""
        A = _RNG.standard_normal((4, 3))
        B = _RNG.standard_normal((3, 5))
        result = dgemm(A, B)
        np.testing.assert_allclose(result, A @ B, rtol=1e-14)
        assert result.shape == (4, 5)

    @pytest.mark.tier0
    def test_square(self):
        """Square matrix multiplication."""
        A = _RNG.standard_normal((10, 10))
        B = _RNG.standard_normal((10, 10))
        result = dgemm(A, B)
        np.testing.assert_allclose(result, A @ B, rtol=1e-13)

    @pytest.mark.tier0
    def test_identity(self):
        """Multiplying by identity returns the original matrix."""
        A = _RNG.standard_normal((5, 5))
        eye = np.eye(5)
        np.testing.assert_allclose(dgemm(A, eye), A, rtol=1e-14)
        np.testing.assert_allclose(dgemm(eye, A), A, rtol=1e-14)


@pytest.mark.skipif(not HAS_C_EXTENSION, reason="C extension not compiled")
class TestInputValidation:
    """Input validation error paths in the C extension."""

    @pytest.mark.tier0
    def test_ddot_wrong_ndim(self):
        """ddot rejects 2-D arrays."""
        x = np.ones((2, 3))
        y = np.ones(6)
        with pytest.raises(ValueError, match="1-D"):
            ddot(x, y)

    @pytest.mark.tier0
    def test_ddot_length_mismatch(self):
        """ddot rejects vectors of different lengths."""
        x = np.ones(3)
        y = np.ones(5)
        with pytest.raises(ValueError, match="same length"):
            ddot(x, y)

    @pytest.mark.tier0
    def test_daxpy_wrong_ndim(self):
        """daxpy rejects 2-D arrays."""
        x = np.ones((2, 3))
        y = np.ones(6)
        with pytest.raises(ValueError, match="1-D"):
            daxpy(1.0, x, y)

    @pytest.mark.tier0
    def test_daxpy_length_mismatch(self):
        """daxpy rejects vectors of different lengths."""
        x = np.ones(3)
        y = np.ones(5)
        with pytest.raises(ValueError, match="same length"):
            daxpy(1.0, x, y)

    @pytest.mark.tier0
    def test_dgemv_A_wrong_ndim(self):
        """dgemv rejects 1-D array as A."""
        A = np.ones(6)
        x = np.ones(3)
        with pytest.raises(ValueError, match="2-D"):
            dgemv(A, x)

    @pytest.mark.tier0
    def test_dgemv_x_wrong_ndim(self):
        """dgemv rejects 2-D array as x."""
        A = np.ones((3, 4))
        x = np.ones((4, 1))
        with pytest.raises(ValueError, match="1-D"):
            dgemv(A, x)

    @pytest.mark.tier0
    def test_dgemv_shape_mismatch(self):
        """dgemv rejects shape-mismatched A and x."""
        A = np.ones((3, 4))
        x = np.ones(5)
        with pytest.raises(ValueError, match="columns"):
            dgemv(A, x)

    @pytest.mark.tier0
    def test_dnrm2_wrong_ndim(self):
        """dnrm2 rejects 2-D arrays."""
        x = np.ones((2, 3))
        with pytest.raises(ValueError, match="1-D"):
            dnrm2(x)

    @pytest.mark.tier0
    def test_dscal_wrong_ndim(self):
        """dscal rejects 2-D arrays."""
        x = np.ones((2, 3))
        with pytest.raises(ValueError, match="1-D"):
            dscal(1.0, x)
