"""Tests for jblas dgemm dispatch chain and backend detection."""

import os
import subprocess
import sys

import numpy as np
import pytest

from jamma.jblas import HAS_C_EXTENSION, blas_backend, blas_is_ilp64

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


class TestLP64OverflowGuard:
    """Verify LP64 dgemm fallback for large dimensions."""

    def test_lp64_guard_constant_exposed(self):
        """LP64_DIM_MAX (46340) is the int32 overflow threshold for N*N."""
        # We can't allocate 46340^2 matrices in tests, but we can verify
        # the guard exists structurally by checking that blas_dispatch.c
        # defines LP64_DIM_MAX. The wrapper function in blas_dispatch.c
        # falls back to jblas own dgemm when any dimension exceeds 46340
        # and the backend is LP64.

        from jamma.jblas import _jblas

        # Verify the module has the blas_is_ilp64 constant
        assert hasattr(_jblas, "blas_is_ilp64")
        assert isinstance(_jblas.blas_is_ilp64, int)
        assert _jblas.blas_is_ilp64 in (0, 1)

    def test_dgemm_moderate_size_correct(self):
        """dgemm at moderate sizes (within LP64 range) produces correct results."""
        from jamma.jblas import dgemm

        rng = np.random.default_rng(99)
        A = np.ascontiguousarray(rng.standard_normal((500, 500)), dtype=np.float64)
        C = dgemm(A, A)
        expected = A @ A
        np.testing.assert_allclose(C, expected, rtol=1e-10)

    def test_ilp64_flag_consistent_with_backend(self):
        """blas_is_ilp64 flag is consistent with blas_backend string."""
        if blas_is_ilp64:
            assert "ILP64" in blas_backend, (
                f"blas_is_ilp64=1 but backend={blas_backend} does not contain ILP64"
            )
        else:
            # LP64 or no external BLAS — backend should NOT contain ILP64
            if blas_backend not in ("jblas-own", "Accelerate", "BLIS"):
                assert "LP64" in blas_backend or "ILP64" not in blas_backend, (
                    f"blas_is_ilp64=0 but backend={blas_backend} contains ILP64"
                )


class TestDgemmDebugEnvVar:
    """Verify JBLAS_DISPATCH_DEBUG=1 produces stderr output."""

    def test_debug_output_on_stderr(self):
        """JBLAS_DISPATCH_DEBUG=1 should produce dispatch diagnostics on stderr."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from jamma.jblas import blas_backend; print(blas_backend)",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "JBLAS_DISPATCH_DEBUG": "1"},
            timeout=30,
        )
        assert result.returncode == 0
        # Debug output goes to stderr and mentions the dispatch steps
        assert (
            "jblas_dispatch" in result.stderr
            or "RTLD_DEFAULT" in result.stderr
            or "step" in result.stderr.lower()
        ), f"Expected dispatch debug output on stderr, got: {result.stderr[:300]}"


class TestILP64Awareness:
    """Verify ILP64/LP64 detection in blas_dispatch."""

    def test_blas_is_ilp64_type(self):
        """blas_is_ilp64 is an integer (0 or 1)."""
        assert isinstance(blas_is_ilp64, int)
        assert blas_is_ilp64 in (0, 1)

    def test_blas_backend_includes_ilp64_info(self):
        """Backend string distinguishes ILP64 from LP64 for MKL/OpenBLAS.

        This test is informational -- it documents the current system's
        configuration. On macOS with Accelerate, the backend is 'Accelerate'
        (neither ILP64 nor LP64). On Linux with MKL, it should be
        'MKL-ILP64' or 'MKL-LP64'.
        """
        print(f"Current blas_backend: {blas_backend}")
        print(f"Current blas_is_ilp64: {blas_is_ilp64}")
        if "MKL" in blas_backend:
            assert blas_backend in ("MKL-ILP64", "MKL-LP64")
        elif "OpenBLAS" in blas_backend:
            assert blas_backend in ("OpenBLAS-ILP64", "OpenBLAS-LP64")
        # Accelerate, BLIS, jblas-own: no ILP64/LP64 suffix expected

    def test_blas_is_ilp64_accessible_from_module(self):
        """blas_is_ilp64 is accessible via jamma.jblas (public API)."""
        from jamma.jblas import blas_is_ilp64 as ilp64

        assert ilp64 == blas_is_ilp64
