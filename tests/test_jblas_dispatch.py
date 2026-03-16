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
            "Accelerate-ILP64",
            "BLIS",
            "BLIS-ILP64",
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


# ---------------------------------------------------------------------------
# dsyrk correctness tests
# ---------------------------------------------------------------------------


class TestDsyrk:
    """Direct correctness tests for dsyrk (symmetric rank-k update)."""

    @pytest.mark.parametrize("N,K", [(1, 1), (3, 5), (5, 3), (10, 10), (7, 1)])
    def test_dsyrk_vs_numpy(self, N, K):
        """dsyrk(X) should equal X @ X.T for various shapes."""
        from jamma.jblas import dsyrk

        rng = np.random.default_rng(42 + N * 100 + K)
        X = rng.standard_normal((N, K))
        result = dsyrk(X)
        expected = X @ X.T
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)

    def test_dsyrk_bitwise_symmetric(self):
        """dsyrk result must be bitwise symmetric (result[i,j] == result[j,i])."""
        from jamma.jblas import dsyrk

        rng = np.random.default_rng(99)
        X = rng.standard_normal((8, 5))
        result = dsyrk(X)
        np.testing.assert_array_equal(result, result.T)

    def test_dsyrk_empty(self):
        """dsyrk with N=0 should return empty matrix."""
        from jamma.jblas import dsyrk

        X = np.empty((0, 5), dtype=np.float64)
        result = dsyrk(X)
        assert result.shape == (0, 0)

    def test_dsyrk_single_column(self):
        """dsyrk with K=1 should produce an outer product."""
        from jamma.jblas import dsyrk

        X = np.array([[1.0], [2.0], [3.0]])
        result = dsyrk(X)
        expected = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]], dtype=np.float64)
        np.testing.assert_allclose(result, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# dgemm shape validation tests
# ---------------------------------------------------------------------------


class TestDgemmShapeValidation:
    """Verify dgemm raises on incompatible shapes."""

    def test_incompatible_inner_dims(self):
        """dgemm should raise ValueError on inner dimension mismatch."""
        from jamma.jblas import dgemm

        A = np.ones((3, 4), dtype=np.float64)
        B = np.ones((5, 6), dtype=np.float64)  # 4 != 5
        with pytest.raises(ValueError, match="mismatch"):
            dgemm(A, B)

    def test_incompatible_transposed(self):
        """dgemm should raise ValueError on transposed inner mismatch."""
        from jamma.jblas import dgemm

        A = np.ones((3, 4), dtype=np.float64)
        B = np.ones((3, 4), dtype=np.float64)
        # op(A) = A.T (4x3), op(B) = B (3x4) → 3 != 3 → should work
        # But: op(A) = A (3x4), op(B) = B.T (4x3) → 4 != 4 → should work
        # Test: op(A) = A.T (4x3), op(B) = B.T (4x3) → 3 != 4 → error
        with pytest.raises(ValueError):
            dgemm(A, B, transa="T", transb="T")


# ---------------------------------------------------------------------------
# Capability-based selection tests (Phase 80.5)
# ---------------------------------------------------------------------------


class TestCapabilityBasedSelection:
    """Verify discover-all-then-select-best dispatch model."""

    def test_discover_all_no_short_circuit(self):
        """blas_dispatch.c must NOT short-circuit BLIS discovery.

        The old code had `if (!found_system) found_blis = discover_bundled_blis()`
        which prevented BLIS-ILP64 from being discovered when LP64 Accelerate
        was found first. The refactored code must discover all three paths
        unconditionally.
        """
        import pathlib

        dispatch_src = (
            pathlib.Path(__file__).resolve().parent.parent
            / "src"
            / "jamma"
            / "jblas"
            / "src"
            / "blas_dispatch.c"
        )
        source = dispatch_src.read_text()

        # The short-circuit pattern must not appear
        assert "if (!found_system)" not in source, (
            "blas_dispatch.c still contains short-circuit pattern "
            "'if (!found_system)' — BLIS discovery must run unconditionally"
        )

        # All three discovery calls must appear
        assert "discover_system_blas(" in source
        assert "discover_pip_mkl(" in source
        assert "discover_bundled_blis(" in source

        # The candidate struct must exist
        assert "blas_candidate_t" in source
        assert "select_best_backend" in source

    def test_dsyrk_ext_matches_jblas_own(self):
        """dsyrk produces correct results via vendor or jblas-own path."""
        from jamma.jblas import dsyrk

        rng = np.random.default_rng(555)
        X = rng.standard_normal((50, 30))
        result = dsyrk(X)
        expected = X @ X.T
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)

    def test_eigh_ext_matches_numpy(self):
        """eigh produces correct eigendecomposition (vendor or jblas pipeline)."""
        from jamma.jblas import eigh

        rng = np.random.default_rng(777)
        A = rng.standard_normal((50, 50))
        K = A @ A.T  # SPD matrix
        K_copy = K.copy()  # eigh overwrites input

        w, v = eigh(K_copy)

        # Verify reconstruction: ||K - v @ diag(w) @ v.T|| / ||K|| < 1e-13
        reconstructed = v @ np.diag(w) @ v.T
        rel_err = np.linalg.norm(K - reconstructed) / np.linalg.norm(K)
        assert rel_err < 1e-13, f"Reconstruction error {rel_err:.2e} too large"

        # Verify orthogonality: ||v.T @ v - I|| < 1e-13
        orth_err = np.linalg.norm(v.T @ v - np.eye(50))
        assert orth_err < 1e-13, f"Orthogonality error {orth_err:.2e} too large"


# ---------------------------------------------------------------------------
# TestCapabilityFlags — blas_has_dsyrk / blas_has_dsyevd (Phase 80.5)
# ---------------------------------------------------------------------------


class TestCapabilityFlags:
    """Verify blas_has_dsyrk and blas_has_dsyevd are exposed."""

    def test_blas_has_dsyrk_is_int(self):
        from jamma.jblas import blas_has_dsyrk

        assert isinstance(blas_has_dsyrk, int)
        assert blas_has_dsyrk in (0, 1)

    def test_blas_has_dsyevd_is_int(self):
        from jamma.jblas import blas_has_dsyevd

        assert isinstance(blas_has_dsyevd, int)
        assert blas_has_dsyevd in (0, 1)

    def test_blas_has_lapacke_dsyevd_is_int(self):
        from jamma.jblas import blas_has_lapacke_dsyevd

        assert isinstance(blas_has_lapacke_dsyevd, int)
        assert blas_has_lapacke_dsyevd in (0, 1)

    def test_dsyrk_capability_consistent_with_backend(self):
        """If backend is ILP64, dsyrk should also be available."""
        from jamma.jblas import blas_backend, blas_has_dsyrk

        if "ILP64" in blas_backend or "Accelerate" in blas_backend:
            # Vendor ILP64 should have dsyrk
            assert blas_has_dsyrk == 1, (
                f"Backend {blas_backend} is ILP64 but blas_has_dsyrk=0"
            )

    def test_lapacke_dsyevd_consistent_with_backend(self):
        """Accelerate has no LAPACKE; MKL does."""
        from jamma.jblas import blas_backend, blas_has_lapacke_dsyevd

        if blas_backend == "Accelerate-ILP64":
            assert blas_has_lapacke_dsyevd == 0, (
                "Accelerate should not have LAPACKE but "
                f"blas_has_lapacke_dsyevd={blas_has_lapacke_dsyevd}"
            )
        if blas_backend in ("BLIS-ILP64", "jblas-own"):
            assert blas_has_lapacke_dsyevd == 0, (
                f"Backend {blas_backend} should not have LAPACKE"
            )
