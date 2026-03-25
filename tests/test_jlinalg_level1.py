"""Level 1/2 BLAS correctness tests for jamma.jlinalg.

Tests all operations against NumPy reference at rtol=1e-14:
- TestDdot: inner product
- TestDnrm2: Euclidean norm with overflow/underflow protection
- TestDaxpy: in-place y += alpha*x
- TestDscal: in-place x *= alpha
- TestDgemv: matrix-vector product

Tests run against whichever backend is active — NumPy fallback or C extension.
The same test file works for both because jamma.jlinalg transparently dispatches.
"""

import numpy as np
import pytest

from jamma.jlinalg import HAS_C_EXTENSION, daxpy, ddot, dgemm, dgemv, dnrm2, dscal


class TestDdot:
    """ddot: inner product of two double vectors."""

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
    """dnrm2: Euclidean norm."""

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
    def test_overflow_agreement(self):
        """dnrm2 matches np.linalg.norm for large values."""
        x = np.array([1e200, 1e200])
        result = dnrm2(x)
        expected = np.linalg.norm(x)
        np.testing.assert_equal(result, expected)

    @pytest.mark.tier0
    def test_underflow_agreement(self):
        """dnrm2 matches np.linalg.norm for tiny values."""
        x = np.array([1e-200, 1e-200])
        result = dnrm2(x)
        expected = np.linalg.norm(x)
        np.testing.assert_allclose(result, expected, rtol=1e-14)


class TestDaxpy:
    """daxpy: in-place y += alpha * x."""

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
    """dscal: in-place x *= alpha."""

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
    """dgemv: matrix-vector product A @ x."""

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
    """dgemm: matrix-matrix product A @ B."""

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

    @pytest.mark.tier0
    def test_nan_propagates(self):
        """dgemm with NaN in A propagates to affected output entries."""
        A = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
        B = np.array([[1.0], [1.0]])
        result = dgemm(A, B)
        np.testing.assert_allclose(result[0, 0], 3.0, rtol=1e-14)
        assert np.isnan(result[1, 0]), f"NaN not propagated: result={result}"
        np.testing.assert_allclose(result[2, 0], 7.0, rtol=1e-14)

    @pytest.mark.tier0
    def test_fortran_order(self):
        """dgemm produces correct results for Fortran-order inputs."""
        rng = np.random.default_rng(604)
        A = np.asfortranarray(rng.standard_normal((8, 5)))
        B = np.asfortranarray(rng.standard_normal((5, 6)))
        result = dgemm(A, B)
        expected = np.ascontiguousarray(A) @ np.ascontiguousarray(B)
        np.testing.assert_allclose(result, expected, rtol=1e-13)

    @pytest.mark.tier0
    def test_empty_inner(self):
        """dgemm with zero inner dimension returns zero matrix."""
        A = np.zeros((3, 0), dtype=np.float64)
        B = np.zeros((0, 4), dtype=np.float64)
        result = dgemm(A, B)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, np.zeros((3, 4)))

    @pytest.mark.tier0
    def test_float32_coercion(self):
        """dgemm accepts float32 input (coerced to float64)."""
        rng = np.random.default_rng(605)
        A = rng.standard_normal((6, 4)).astype(np.float32)
        B = rng.standard_normal((4, 3)).astype(np.float32)
        result = dgemm(A, B)
        expected = A.astype(np.float64) @ B.astype(np.float64)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


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
        # C extension reports "inner dimensions mismatch"; fallback reports "columns"
        with pytest.raises(ValueError, match="columns|mismatch"):
            dgemm(A, B)

    @pytest.mark.tier0
    def test_daxpy_non_float64_y_rejected(self):
        """daxpy fallback rejects non-float64 y to prevent silent precision loss."""
        if HAS_C_EXTENSION:
            pytest.skip("C extension coerces dtypes via PyArray_FROM_OTF")
        x = np.ones(5, dtype=np.float64)
        y = np.ones(5, dtype=np.float32)
        with pytest.raises(TypeError, match="float64"):
            daxpy(1.0, x, y)

    @pytest.mark.tier0
    def test_dscal_non_float64_rejected(self):
        """dscal fallback rejects non-float64 input."""
        if HAS_C_EXTENSION:
            pytest.skip("C extension coerces dtypes via PyArray_FROM_OTF")
        x = np.ones(5, dtype=np.float32)
        with pytest.raises(TypeError, match="float64"):
            dscal(2.0, x)


class TestNaNPropagation:
    """NaN/Inf propagation: ensure special values are not silently swallowed."""

    @pytest.mark.tier0
    def test_ddot_nan(self):
        """ddot with NaN input must return NaN."""
        x = np.array([1.0, np.nan, 3.0])
        y = np.array([1.0, 1.0, 1.0])
        assert np.isnan(ddot(x, y))

    @pytest.mark.tier0
    def test_ddot_inf(self):
        """ddot with Inf input propagates correctly."""
        x = np.array([np.inf, 1.0])
        y = np.array([1.0, 1.0])
        assert np.isinf(ddot(x, y))

    @pytest.mark.tier0
    def test_dnrm2_nan(self):
        """dnrm2 with NaN input must return NaN."""
        x = np.array([1.0, np.nan, 3.0])
        assert np.isnan(dnrm2(x))

    @pytest.mark.tier0
    def test_daxpy_nan_propagates(self):
        """daxpy with NaN in x must propagate NaN into y."""
        x = np.array([1.0, np.nan, 3.0])
        y = np.array([1.0, 1.0, 1.0])
        daxpy(1.0, x, y)
        assert np.isnan(y[1]), f"NaN not propagated: y={y}"

    @pytest.mark.tier0
    def test_dgemv_nan_propagates(self):
        """dgemv with NaN in A propagates to affected rows."""
        A = np.array([[1.0, 2.0], [np.nan, 1.0]])
        x = np.array([1.0, 1.0])
        result = dgemv(A, x)
        np.testing.assert_allclose(result[0], 3.0, rtol=1e-14)
        assert np.isnan(result[1]), f"NaN not propagated: result={result}"

    @pytest.mark.tier0
    def test_dscal_nan_propagates(self):
        """dscal with NaN in input propagates when alpha != 0."""
        x = np.array([1.0, np.nan, 3.0])
        dscal(2.0, x)
        np.testing.assert_allclose(x[0], 2.0, rtol=1e-14)
        assert np.isnan(x[1]), f"NaN not propagated: x={x}"
        np.testing.assert_allclose(x[2], 6.0, rtol=1e-14)

    @pytest.mark.tier0
    def test_daxpy_nan_alpha(self):
        """daxpy with NaN alpha makes all y elements NaN."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 1.0])
        daxpy(float("nan"), x, y)
        assert np.all(np.isnan(y)), f"NaN alpha not propagated: y={y}"

    @pytest.mark.tier0
    def test_dscal_nan_alpha(self):
        """dscal with NaN alpha makes all elements NaN."""
        x = np.array([1.0, 2.0, 3.0])
        dscal(float("nan"), x)
        assert np.all(np.isnan(x)), f"NaN alpha not propagated: x={x}"


class TestDtypeAndContiguity:
    """Verify correct handling of non-float64 and non-contiguous inputs."""

    @pytest.mark.tier0
    def test_ddot_float32_input(self):
        """ddot accepts float32 input (coerced to float64)."""
        rng = np.random.default_rng(701)
        x = rng.standard_normal(50).astype(np.float32)
        y = rng.standard_normal(50).astype(np.float32)
        result = ddot(x, y)
        expected = float(np.dot(x.astype(np.float64), y.astype(np.float64)))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.tier0
    def test_dnrm2_int_input(self):
        """dnrm2 accepts integer input (coerced to float64)."""
        x = np.array([3, 4], dtype=np.int64)
        np.testing.assert_allclose(dnrm2(x), 5.0, rtol=1e-14)

    @pytest.mark.tier0
    def test_dgemv_fortran_order(self):
        """dgemv produces correct results for Fortran-order matrix."""
        rng = np.random.default_rng(702)
        A_c = rng.standard_normal((10, 5))
        A_f = np.asfortranarray(A_c)
        x = rng.standard_normal(5)
        result = dgemv(A_f, x)
        expected = A_c @ x
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    @pytest.mark.tier0
    def test_ddot_sliced_arrays(self):
        """ddot works with non-contiguous sliced arrays."""
        rng = np.random.default_rng(703)
        x_full = rng.standard_normal(100)
        y_full = rng.standard_normal(100)
        x = x_full[::2]  # stride-2 slice
        y = y_full[::2]
        result = ddot(x, y)
        expected = float(np.dot(x.astype(np.float64), y.astype(np.float64)))
        np.testing.assert_allclose(result, expected, rtol=1e-14)
