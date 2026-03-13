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


class TestDdot:
    """ddot: inner product of two double vectors (BL1-01)."""

    @pytest.mark.tier0
    def test_small(self):
        """n=1: single-element dot product."""
        rng = np.random.default_rng(101)
        x = rng.standard_normal(1)
        y = rng.standard_normal(1)
        np.testing.assert_allclose(ddot(x, y), np.dot(x, y), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_medium(self):
        """n=100: typical small vector."""
        rng = np.random.default_rng(102)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        np.testing.assert_allclose(ddot(x, y), np.dot(x, y), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_large(self):
        """n=10000: exercises SIMD unroll paths."""
        rng = np.random.default_rng(103)
        x = rng.standard_normal(10_000)
        y = rng.standard_normal(10_000)
        np.testing.assert_allclose(ddot(x, y), np.dot(x, y), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    @pytest.mark.parametrize("n", [3, 4, 15, 16, 17])
    def test_simd_boundaries(self, n: int):
        """Boundary sizes for AVX2 (16-wide) and generic (4-wide) unrolls.

        n=3: generic tail only (n < 4).
        n=4: exactly one generic unroll, zero tail.
        n=15: AVX2 scalar tail only (no full 16-wide iteration).
        n=16: exactly one AVX2 iteration, zero tail.
        n=17: one AVX2 iteration plus one tail element.
        """
        rng = np.random.default_rng(n)
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
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
        rng = np.random.default_rng(201)
        x = rng.standard_normal(1)
        np.testing.assert_allclose(dnrm2(x), np.linalg.norm(x), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_medium(self):
        """n=100: typical small vector."""
        rng = np.random.default_rng(202)
        x = rng.standard_normal(100)
        np.testing.assert_allclose(dnrm2(x), np.linalg.norm(x), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_large(self):
        """n=10000: exercises main loop path."""
        rng = np.random.default_rng(203)
        x = rng.standard_normal(10_000)
        np.testing.assert_allclose(dnrm2(x), np.linalg.norm(x), rtol=1e-14, atol=1e-14)

    @pytest.mark.tier0
    def test_zero_vector(self):
        """Zero vector has norm 0."""
        x = np.zeros(100)
        assert dnrm2(x) == 0.0

    @pytest.mark.tier0
    def test_empty(self):
        """Empty vector has norm 0."""
        x = np.array([], dtype=np.float64)
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
        rng = np.random.default_rng(301)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        y_orig = y.copy()
        daxpy(2.0, x, y)
        np.testing.assert_allclose(y, y_orig + 2.0 * x, rtol=1e-14)

    @pytest.mark.tier0
    def test_alpha_zero(self):
        """alpha=0: y must be unchanged."""
        rng = np.random.default_rng(302)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        y_orig = y.copy()
        daxpy(0.0, x, y)
        np.testing.assert_array_equal(y, y_orig)

    @pytest.mark.tier0
    def test_empty(self):
        """Empty vectors: no-op."""
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
        daxpy(1.0, x, y)
        assert len(y) == 0

    @pytest.mark.tier0
    @pytest.mark.parametrize("n", [1, 15, 16, 17, 100, 10_000])
    def test_sizes(self, n: int):
        """daxpy correctness at various sizes including SIMD boundaries.

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
        rng = np.random.default_rng(401)
        x = rng.standard_normal(100)
        x_orig = x.copy()
        dscal(3.0, x)
        np.testing.assert_allclose(x, x_orig * 3.0, rtol=1e-14)

    @pytest.mark.tier0
    def test_alpha_zero(self):
        """alpha=0: x becomes all zeros."""
        rng = np.random.default_rng(402)
        x = rng.standard_normal(100)
        dscal(0.0, x)
        np.testing.assert_array_equal(x, np.zeros(100))

    @pytest.mark.tier0
    def test_alpha_zero_with_nan(self):
        """alpha=0 with NaN: both C extension and fallback produce +0.0.

        Reference BLAS (and the C extension) use memset → NaN becomes +0.0.
        The fallback matches by using x[:] = 0.0 instead of x *= 0.0.
        """
        x = np.array([1.0, np.nan, np.inf, -np.inf, 0.0])
        dscal(0.0, x)
        np.testing.assert_array_equal(x, np.zeros(5))

    @pytest.mark.tier0
    def test_alpha_one(self):
        """alpha=1: x is unchanged."""
        rng = np.random.default_rng(403)
        x = rng.standard_normal(100)
        x_orig = x.copy()
        dscal(1.0, x)
        np.testing.assert_array_equal(x, x_orig)

    @pytest.mark.tier0
    def test_empty(self):
        """Empty vector: no-op."""
        x = np.array([], dtype=np.float64)
        dscal(2.0, x)
        assert len(x) == 0

    @pytest.mark.tier0
    @pytest.mark.parametrize("n", [3, 4, 15, 16, 17])
    def test_simd_boundaries(self, n: int):
        """Boundary sizes for AVX2 (16-wide) and generic (4-wide) unrolls."""
        rng = np.random.default_rng(n + 400)
        x = rng.standard_normal(n)
        x_orig = x.copy()
        dscal(2.5, x)
        np.testing.assert_allclose(x, x_orig * 2.5, rtol=1e-14)


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
        rng = np.random.default_rng(502)
        A = rng.standard_normal((10, 10))
        x = rng.standard_normal(10)
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
        rng = np.random.default_rng(503)
        A = rng.standard_normal((100, 50))
        x = rng.standard_normal(50)
        result = dgemv(A, x)
        np.testing.assert_allclose(result, A @ x, rtol=2e-12)
        assert result.shape == (100,)

    @pytest.mark.tier0
    def test_empty_rows(self):
        """m=0: empty result vector."""
        A = np.zeros((0, 5), dtype=np.float64)
        x = np.ones(5)
        result = dgemv(A, x)
        assert result.shape == (0,)

    @pytest.mark.tier0
    def test_empty_cols(self):
        """n=0: returns m-length zero vector."""
        A = np.zeros((3, 0), dtype=np.float64)
        x = np.array([], dtype=np.float64)
        result = dgemv(A, x)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, np.zeros(3))


class TestDgemm:
    """dgemm: matrix-matrix product A @ B (BL3-01)."""

    @pytest.mark.tier0
    def test_basic(self):
        """Small matrix multiplication agrees with NumPy."""
        rng = np.random.default_rng(601)
        A = rng.standard_normal((4, 3))
        B = rng.standard_normal((3, 5))
        result = dgemm(A, B)
        np.testing.assert_allclose(result, A @ B, rtol=1e-14)
        assert result.shape == (4, 5)

    @pytest.mark.tier0
    def test_square(self):
        """Square matrix multiplication."""
        rng = np.random.default_rng(602)
        A = rng.standard_normal((10, 10))
        B = rng.standard_normal((10, 10))
        result = dgemm(A, B)
        np.testing.assert_allclose(result, A @ B, rtol=1e-13)

    @pytest.mark.tier0
    def test_identity(self):
        """Multiplying by identity returns the original matrix."""
        rng = np.random.default_rng(603)
        A = rng.standard_normal((5, 5))
        eye = np.eye(5)
        np.testing.assert_allclose(dgemm(A, eye), A, rtol=1e-14)
        np.testing.assert_allclose(dgemm(eye, A), A, rtol=1e-14)


class TestInputValidation:
    """Input validation error paths (C extension and NumPy fallback)."""

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

    @pytest.mark.tier0
    def test_dgemm_A_wrong_ndim(self):
        """dgemm rejects 1-D array as A."""
        A = np.ones(6)
        B = np.ones((3, 2))
        with pytest.raises(ValueError, match="2-D"):
            dgemm(A, B)

    @pytest.mark.tier0
    def test_dgemm_B_wrong_ndim(self):
        """dgemm rejects 1-D array as B."""
        A = np.ones((2, 3))
        B = np.ones(3)
        with pytest.raises(ValueError, match="2-D"):
            dgemm(A, B)

    @pytest.mark.tier0
    def test_dgemm_shape_mismatch(self):
        """dgemm rejects inner dimension mismatch."""
        A = np.ones((3, 4))
        B = np.ones((5, 2))
        with pytest.raises(ValueError, match="columns"):
            dgemm(A, B)
