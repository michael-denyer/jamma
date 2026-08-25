"""Tests for jlinalg dgemm dispatch chain and backend detection."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from jamma.jlinalg import HAS_C_EXTENSION, blas_backend, blas_is_ilp64

pytestmark = [
    pytest.mark.tier0,
    pytest.mark.skipif(
        not HAS_C_EXTENSION,
        reason="jlinalg C extension not compiled",
    ),
]


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
            "system-BLAS-ILP64",
            "system-BLAS-LP64",
            "numpy-fallback",
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
        from jamma.jlinalg import dgemm

        rng = np.random.default_rng(42)
        A = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float64)
        B = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float64)
        C = dgemm(A, B)
        expected = A @ B
        np.testing.assert_allclose(C, expected, rtol=1e-10)

    @pytest.mark.parametrize("transa,transb", [("N", "N"), ("T", "N"), ("N", "T")])
    def test_dgemm_transpose_combinations(self, transa, transb):
        """Transpose dispatch works with external backend."""
        from jamma.jlinalg import dgemm

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


class TestIlp64FlagConsistency:
    """Verify blas_is_ilp64 stays consistent with the backend string and dgemm
    computes correctly at moderate sizes."""

    def test_dgemm_moderate_size_correct(self):
        """dgemm at moderate sizes produces correct results."""
        from jamma.jlinalg import dgemm

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
            if blas_backend not in ("Accelerate", "numpy-fallback"):
                assert "LP64" in blas_backend or "ILP64" not in blas_backend, (
                    f"blas_is_ilp64=0 but backend={blas_backend} contains ILP64"
                )


class TestDgemmDebugEnvVar:
    """Verify JLINALG_DISPATCH_DEBUG=1 produces stderr output."""

    def test_debug_output_on_stderr(self):
        """JLINALG_DISPATCH_DEBUG=1 should produce dispatch diagnostics on stderr."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from jamma.jlinalg import blas_backend; print(blas_backend)",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "JLINALG_DISPATCH_DEBUG": "1"},
            timeout=30,
        )
        assert result.returncode == 0
        # Debug output goes to stderr and mentions the dispatch steps
        assert (
            "jlinalg_dispatch" in result.stderr
            or "RTLD_DEFAULT" in result.stderr
            or "step" in result.stderr.lower()
        ), f"Expected dispatch debug output on stderr, got: {result.stderr[:300]}"


class TestILP64Awareness:
    """Verify ILP64/LP64 detection in blas_dispatch."""

    def test_blas_is_ilp64_type(self):
        """blas_is_ilp64 is an integer (0 or 1)."""
        assert isinstance(blas_is_ilp64, int)
        assert blas_is_ilp64 in (0, 1)

    def test_blas_backend_string_has_known_value(self):
        """Backend string is one of the documented values.

        MKL and OpenBLAS backends MUST advertise their ILP64/LP64 split
        in the suffix. Accelerate is ILP64-only on macOS 13.3+ and is
        reported without a suffix. ``numpy-fallback`` is the catch-all
        when no vendor BLAS is wired in.
        """
        known_backends = {
            "MKL-ILP64",
            "MKL-LP64",
            "OpenBLAS-ILP64",
            "OpenBLAS-LP64",
            "Accelerate",
            "Accelerate-ILP64",
            # ``system-BLAS-{I,}LP64`` is returned by blas_dispatch.c when
            # a vendor library is loaded but path-string detection fails to
            # identify it (typically Linux distributions linking against an
            # alias-only libblas.so).
            "system-BLAS-ILP64",
            "system-BLAS-LP64",
            "numpy-fallback",
        }
        assert blas_backend in known_backends, (
            f"Unknown blas_backend {blas_backend!r}; expected one of "
            f"{sorted(known_backends)}. Update this list and "
            f"jlinalg/__init__.py if a new backend was added."
        )

    def test_blas_backend_ilp64_flag_consistent_with_string(self):
        """``blas_is_ilp64`` value matches the backend-string suffix."""
        if "ILP64" in blas_backend:
            assert blas_is_ilp64 == 1, (
                f"backend={blas_backend!r} but blas_is_ilp64={blas_is_ilp64}"
            )
        elif "LP64" in blas_backend:
            assert blas_is_ilp64 == 0, (
                f"backend={blas_backend!r} but blas_is_ilp64={blas_is_ilp64}"
            )
        # Accelerate / numpy-fallback: no LP64/ILP64 suffix to cross-check.

    def test_blas_is_ilp64_accessible_from_module(self):
        """blas_is_ilp64 is accessible via jamma.jlinalg (public API)."""
        from jamma.jlinalg import blas_is_ilp64 as ilp64

        assert ilp64 == blas_is_ilp64


# ---------------------------------------------------------------------------
# dsyrk correctness tests
# ---------------------------------------------------------------------------


class TestDsyrk:
    """Direct correctness tests for dsyrk (symmetric rank-k update)."""

    @pytest.mark.parametrize("N,K", [(1, 1), (3, 5), (5, 3), (10, 10), (7, 1)])
    def test_dsyrk_vs_numpy(self, N, K):
        """dsyrk(X) should equal X @ X.T for various shapes."""
        from jamma.jlinalg import dsyrk

        rng = np.random.default_rng(42 + N * 100 + K)
        X = rng.standard_normal((N, K))
        result = dsyrk(X)
        expected = X @ X.T
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)

    def test_dsyrk_bitwise_symmetric(self):
        """dsyrk result must be bitwise symmetric (result[i,j] == result[j,i])."""
        from jamma.jlinalg import dsyrk

        rng = np.random.default_rng(99)
        X = rng.standard_normal((8, 5))
        result = dsyrk(X)
        np.testing.assert_array_equal(result, result.T)

    def test_dsyrk_empty(self):
        """dsyrk with N=0 should return empty matrix."""
        from jamma.jlinalg import dsyrk

        X = np.empty((0, 5), dtype=np.float64)
        result = dsyrk(X)
        assert result.shape == (0, 0)

    def test_dsyrk_single_column(self):
        """dsyrk with K=1 should produce an outer product."""
        from jamma.jlinalg import dsyrk

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
        from jamma.jlinalg import dgemm

        A = np.ones((3, 4), dtype=np.float64)
        B = np.ones((5, 6), dtype=np.float64)  # 4 != 5
        with pytest.raises(ValueError, match="mismatch"):
            dgemm(A, B)

    def test_incompatible_transposed(self):
        """dgemm should raise ValueError on transposed inner mismatch."""
        from jamma.jlinalg import dgemm

        A = np.ones((3, 4), dtype=np.float64)
        B = np.ones((3, 4), dtype=np.float64)
        # op(A) = A.T (4x3), op(B) = B (3x4) → 3 != 3 → should work
        # But: op(A) = A (3x4), op(B) = B.T (4x3) → 4 != 4 → should work
        # Test: op(A) = A.T (4x3), op(B) = B.T (4x3) → 3 != 4 → error
        with pytest.raises(ValueError):
            dgemm(A, B, transa="T", transb="T")


# ---------------------------------------------------------------------------
# Capability-based selection tests
# ---------------------------------------------------------------------------


class TestCapabilityBasedSelection:
    """Verify discover-all-then-select-best dispatch model."""

    def test_dsyrk_ext_matches_numpy(self):
        """dsyrk produces correct results via vendor or numpy-fallback path."""
        from jamma.jlinalg import dsyrk

        rng = np.random.default_rng(555)
        X = rng.standard_normal((50, 30))
        result = dsyrk(X)
        expected = X @ X.T
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)

    def test_eigh_ext_matches_numpy(self):
        """eigh produces correct eigendecomposition (vendor or jlinalg pipeline)."""
        from jamma.jlinalg import eigh

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
# TestCapabilityFlags — blas_has_dsyrk / blas_has_dsyevd
# ---------------------------------------------------------------------------


class TestCapabilityFlags:
    """Verify blas_has_dsyrk and blas_has_dsyevd are exposed."""

    def test_blas_has_dsyrk_is_int(self):
        from jamma.jlinalg import blas_has_dsyrk

        assert isinstance(blas_has_dsyrk, int)
        assert blas_has_dsyrk in (0, 1)

    def test_blas_has_dsyevd_is_int(self):
        from jamma.jlinalg import blas_has_dsyevd

        assert isinstance(blas_has_dsyevd, int)
        assert blas_has_dsyevd in (0, 1)

    def test_blas_has_lapacke_dsyevd_is_int(self):
        from jamma.jlinalg import blas_has_lapacke_dsyevd

        assert isinstance(blas_has_lapacke_dsyevd, int)
        assert blas_has_lapacke_dsyevd in (0, 1)

    def test_dsyrk_capability_consistent_with_backend(self):
        """If backend is ILP64 (except OpenBLAS), dsyrk should be available."""
        from jamma.jlinalg import blas_backend, blas_has_dsyrk

        # OpenBLAS-ILP64 may not expose cblas_dsyrk with ILP64 symbols
        if "OpenBLAS" in blas_backend:
            return
        if "ILP64" in blas_backend or "Accelerate" in blas_backend:
            assert blas_has_dsyrk == 1, (
                f"Backend {blas_backend} is ILP64 but blas_has_dsyrk=0"
            )

    def test_lapacke_dsyevd_consistent_with_backend(self):
        """Accelerate has no LAPACKE; MKL does."""
        from jamma.jlinalg import blas_backend, blas_has_lapacke_dsyevd

        if blas_backend == "Accelerate-ILP64":
            assert blas_has_lapacke_dsyevd == 0, (
                "Accelerate should not have LAPACKE but "
                f"blas_has_lapacke_dsyevd={blas_has_lapacke_dsyevd}"
            )
        if blas_backend == "numpy-fallback":
            assert blas_has_lapacke_dsyevd == 0, (
                f"Backend {blas_backend} should not have LAPACKE"
            )


# ---------------------------------------------------------------------------
# The dgemm vendor gate
# ---------------------------------------------------------------------------


class TestDgemmVendorGate:
    """``dgemm`` must reach NumPy when vendor dgemm is not wired.

    ``py_dgemm`` raises ``RuntimeError`` the moment ``blas_has_external()`` is
    false, so binding ``dgemm`` straight off the C module is only safe while a
    vendor BLAS is wired. An LP64-only host (distro or conda numpy) sits in the
    unwired state permanently, and the first chunk rotation in
    ``chunk_runner_numpy`` is where it lands. CI never sees it because PyPI
    numpy ships ILP64 ``scipy_openblas64``, so ``JLINALG_NO_VENDOR_DGEMM``
    reproduces the state here.
    """

    @staticmethod
    def _run_unwired(body: str) -> subprocess.CompletedProcess[str]:
        """Run ``body`` in a fresh interpreter with vendor dgemm unwired.

        A subprocess is required: dispatch resolves once, at extension import.
        """
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(body)],
            capture_output=True,
            text=True,
            env={**os.environ, "JLINALG_NO_VENDOR_DGEMM": "1"},
            timeout=60,
        )

    def test_rotation_still_computes_without_vendor_dgemm(self):
        """The chunk-rotation call shape returns the right numbers, not an error."""
        proc = self._run_unwired("""
            import numpy as np
            from jamma.jlinalg import HAS_C_EXTENSION, dgemm

            assert HAS_C_EXTENSION, "this test needs the compiled extension"
            rng = np.random.default_rng(7)
            G = np.ascontiguousarray(rng.standard_normal((40, 12)))
            U = np.ascontiguousarray(rng.standard_normal((40, 40)))
            out = np.empty((12, 40))
            returned = dgemm(G, U, transa="T", out=out)
            assert returned is out
            np.testing.assert_allclose(out, G.T @ U, rtol=1e-12, atol=1e-14)
            print("OK")
        """)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().splitlines()[-1] == "OK", proc.stdout

    def test_capability_flag_reports_the_unwired_state(self):
        """``blas_has_dgemm`` is the gate; ``blas_backend`` cannot stand in for it.

        The backend string names what dispatch resolved, not what it wired.
        Here it still reads as a vendor backend while vendor dgemm is absent,
        which is why the binding keys on the flag.
        """
        proc = self._run_unwired("""
            from jamma.jlinalg import HAS_C_EXTENSION, blas_backend, blas_has_dgemm

            assert HAS_C_EXTENSION, "this test needs the compiled extension"
            print(f"{blas_has_dgemm} {blas_backend}")
        """)
        assert proc.returncode == 0, proc.stderr
        flag, backend = proc.stdout.strip().splitlines()[-1].split(" ", 1)
        assert flag == "0", proc.stdout
        assert backend != "numpy-fallback", (
            "the knob is meant to leave a vendor backend resolved; without that "
            "this test no longer shows why the string is not a usable gate"
        )

    def test_numpy_dgemm_keeps_the_c_input_contract(self):
        """The fallback rejects what the C entry point rejects."""
        proc = self._run_unwired("""
            import numpy as np
            from jamma.jlinalg import dgemm

            A = np.ones((3, 4))
            for bad, match in [
                (lambda: dgemm(np.ones(3), A), "2-D"),
                (lambda: dgemm(A, np.ones((5, 6))), "must match"),
                (lambda: dgemm(A, A, transa="X"), "'N' or 'T'"),
                (lambda: dgemm(A, A, transb="T", out=np.empty((3, 4))), "shape"),
            ]:
                try:
                    bad()
                except ValueError as exc:
                    assert match in str(exc), f"{match!r} not in {exc}"
                else:
                    raise AssertionError(f"no ValueError for {match!r}")
            print("OK")
        """)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().splitlines()[-1] == "OK", proc.stdout


class TestDgemmCapabilityFlag:
    """``blas_has_dgemm`` parity with the rest of the dispatch state."""

    def test_flag_is_int(self):
        from jamma.jlinalg import blas_has_dgemm

        assert isinstance(blas_has_dgemm, int)
        assert blas_has_dgemm in (0, 1)

    def test_numpy_fallback_backend_never_wires_dgemm(self):
        """No vendor backend resolved means no vendor dgemm."""
        from jamma.jlinalg import blas_has_dgemm

        if blas_backend == "numpy-fallback":
            assert blas_has_dgemm == 0, (
                f"backend={blas_backend!r} but blas_has_dgemm={blas_has_dgemm}"
            )
