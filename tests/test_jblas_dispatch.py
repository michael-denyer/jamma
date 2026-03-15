"""Tests for jblas dgemm dispatch chain and backend detection."""

import numpy as np
import pytest

from jamma.jblas import HAS_C_EXTENSION, blas_backend

pytestmark = pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="jblas C extension not compiled",
)


class TestBlasBackend:
    """Verify blas_backend attribute is a non-empty string."""

    def test_blas_backend_is_string(self):
        assert isinstance(blas_backend, str)
        assert len(blas_backend) > 0

    def test_blas_backend_known_value(self):
        """Backend string must be one of the known values."""
        known = {
            "MKL-ILP64",
            "MKL-LP64",
            "OpenBLAS-ILP64",
            "OpenBLAS-LP64",
            "Accelerate",
            "BLIS",
            "jblas-own",
            "system-BLAS-ILP64",
            "system-BLAS-LP64",
        }
        assert blas_backend in known, f"Unknown blas_backend: {blas_backend}"


class TestDgemmDispatchCorrectness:
    """Verify dgemm produces correct results regardless of backend."""

    @pytest.mark.parametrize(
        "m,n,k",
        [
            (1, 1, 1),
            (7, 11, 5),  # Non-square, non-MR-aligned
            (100, 100, 100),  # Medium square
            (1, 1000, 1),  # Degenerate row vector
            (1000, 1, 1000),  # Degenerate column output
        ],
    )
    def test_dgemm_matches_numpy(self, m, n, k):
        """dgemm with external dispatch matches numpy matmul."""
        from jamma.jblas import dgemm

        rng = np.random.default_rng(42)
        A = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float64)
        B = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float64)
        C = dgemm(A, B)
        expected = A @ B
        np.testing.assert_allclose(C, expected, rtol=1e-10)

    @pytest.mark.parametrize("transa,transb", [("N", "N"), ("T", "N"), ("N", "T")])
    def test_dgemm_transpose_combinations(self, transa, transb):
        """Transpose dispatch works with external backend."""
        from jamma.jblas import dgemm

        rng = np.random.default_rng(123)
        # Build compatible shapes for each transpose combination:
        #   N,N: A(50,30) @ B(30,40)         -> C(50,40)
        #   T,N: A_in(30,50).T @ B(50,40)    -> C(50,40)  [A_in is 30x50]
        #   N,T: A(50,30) @ B_in(40,30).T    -> C(50,40)  [B_in is 40x30]
        m, k, n = 50, 30, 40
        if transa == "T":
            A_in = np.ascontiguousarray(rng.standard_normal((k, m)), dtype=np.float64)
            opA = A_in.T  # (m, k)
        else:
            A_in = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float64)
            opA = A_in
        if transb == "T":
            B_in = np.ascontiguousarray(rng.standard_normal((n, k)), dtype=np.float64)
            opB = B_in.T  # (k, n)
        else:
            B_in = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float64)
            opB = B_in
        C = dgemm(A_in, B_in, transa=transa, transb=transb)
        expected = opA @ opB
        np.testing.assert_allclose(C, expected, rtol=1e-10)


class TestBlasBackendFallback:
    """Test that fallback path provides correct results."""

    def test_fallback_blas_backend_string(self):
        """When C extension is available, blas_backend is not 'numpy-fallback'."""
        assert blas_backend != "numpy-fallback"
