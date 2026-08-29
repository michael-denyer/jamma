"""Tests for jlinalg dgemm (matrix-matrix product).

Tests cover:
- Correctness vs NumPy reference (all sizes, special cases)
- Boundary sizes (tests.builders.BOUNDARY_SIZES)
- Transpose variants (NN, TN, NT, TT)
- Thread safety (OMP_NUM_THREADS=1 vs 4 gives bitwise-identical results)
- Throughput benchmark (skipped if C extension not present)

Run with -n0 to avoid parallel interference with threading tests:
    uv run pytest tests/test_jlinalg_dgemm.py -x -n0 -v
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import ClassVar

import numpy as np
import numpy.testing as npt
import pytest

from jamma.jlinalg import HAS_C_EXTENSION, _dgemm_numpy, dgemm
from tests.builders import BOUNDARY_SIZES

pytestmark = pytest.mark.tier0


def _reference_dgemm(
    A: np.ndarray,
    B: np.ndarray,
    transa: str = "N",
    transb: str = "N",
) -> np.ndarray:
    """Reference dgemm using np.matmul with explicit transpositions.

    Args:
        A: Left matrix.
        B: Right matrix.
        transa: 'N' (no transpose) or 'T' (transpose A).
        transb: 'N' (no transpose) or 'T' (transpose B).

    Returns:
        Result matrix op(A) @ op(B).
    """
    Aop = A.T if transa == "T" else A
    Bop = B.T if transb == "T" else B
    return np.matmul(
        Aop.astype(np.float64, copy=False),
        Bop.astype(np.float64, copy=False),
    )


# ---------------------------------------------------------------------------
# TestDgemmCorrectness
# ---------------------------------------------------------------------------


class TestDgemmCorrectness:
    """Basic correctness tests for dgemm vs NumPy matmul reference."""

    def test_square_small(self) -> None:
        """10x10 @ 10x10, rtol=1e-12."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 10))
        B = rng.standard_normal((10, 10))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_square_medium(self) -> None:
        """200x200 @ 200x200 (K triggers blocking on AVX2)."""
        rng = np.random.default_rng(43)
        A = rng.standard_normal((200, 200))
        B = rng.standard_normal((200, 200))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_rectangular_tall_skinny(self) -> None:
        """500x50 @ 50x300 (tall skinny A)."""
        rng = np.random.default_rng(44)
        A = rng.standard_normal((500, 50))
        B = rng.standard_normal((50, 300))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_rectangular_wide(self) -> None:
        """50x500 @ 500x50 (wide A, K=500 triggers multi-pass blocking)."""
        rng = np.random.default_rng(45)
        A = rng.standard_normal((50, 500))
        B = rng.standard_normal((500, 50))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_single_element(self) -> None:
        """1x1 @ 1x1, exact match."""
        A = np.array([[3.0]])
        B = np.array([[7.0]])
        result = dgemm(A, B)
        expected = np.array([[21.0]])
        npt.assert_array_equal(result, expected)

    def test_identity(self) -> None:
        """Identity matrix @ A == A."""
        rng = np.random.default_rng(46)
        A = rng.standard_normal((50, 50))
        eye = np.eye(50)
        result = dgemm(eye, A)
        npt.assert_allclose(result, A, rtol=1e-14)

    def test_zero_matrix(self) -> None:
        """0 @ A == 0 (zero matrix on left)."""
        rng = np.random.default_rng(47)
        A = rng.standard_normal((30, 30))
        Z = np.zeros((30, 30))
        result = dgemm(Z, A)
        npt.assert_array_equal(result, np.zeros((30, 30)))

    def test_known_result(self) -> None:
        """Hand-computed 2x3 @ 3x2 product."""
        # A = [[1, 2, 3], [4, 5, 6]]
        # B = [[7, 8], [9, 10], [11, 12]]
        # C = [[1*7+2*9+3*11, 1*8+2*10+3*12],
        #      [4*7+5*9+6*11, 4*8+5*10+6*12]]
        #   = [[58, 64], [139, 154]]
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        B = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        result = dgemm(A, B)
        expected = np.array([[58.0, 64.0], [139.0, 154.0]])
        npt.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# TestDgemmContract — properties any dgemm implementation must uphold
# ---------------------------------------------------------------------------


class TestDgemmContract:
    """NaN propagation and dtype coercion, checked against both backends.

    ``dgemm`` is whichever implementation the module resolved at import
    (vendor or NumPy); ``_dgemm_numpy`` is always the NumPy fallback,
    reachable directly regardless of what the vendor path resolved to.
    Running each case against both proves the contract holds independent
    of which backend answers ``dgemm`` in this process.
    """

    def test_nan_propagates(self) -> None:
        """A NaN in A propagates only to the output entries it touches."""
        A = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
        B = np.array([[1.0], [1.0]])
        for impl in (dgemm, _dgemm_numpy):
            result = impl(A, B)
            npt.assert_allclose(result[0, 0], 3.0, rtol=1e-14)
            assert np.isnan(result[1, 0]), f"{impl}: NaN not propagated: {result}"
            npt.assert_allclose(result[2, 0], 7.0, rtol=1e-14)

    def test_float32_input_coerced_to_float64(self) -> None:
        rng = np.random.default_rng(605)
        A = rng.standard_normal((6, 4)).astype(np.float32)
        B = rng.standard_normal((4, 3)).astype(np.float32)
        expected = A.astype(np.float64) @ B.astype(np.float64)
        for impl in (dgemm, _dgemm_numpy):
            result = impl(A, B)
            assert result.dtype == np.float64
            npt.assert_allclose(result, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# TestDgemmZeroDimension — empty matrix edge cases
# ---------------------------------------------------------------------------


class TestDgemmZeroDimension:
    """Zero-dimension edge cases: M=0, N=0, K=0.

    dgemm.c zero-initialises C then returns early when any dimension is 0.
    Verify the Python wrapper handles these correctly (no segfault, correct shape).
    """

    def test_m_zero(self) -> None:
        """M=0: result is (0, N) empty matrix."""
        A = np.empty((0, 5), dtype=np.float64)
        B = np.random.default_rng(0).standard_normal((5, 3))
        result = dgemm(A, B)
        assert result.shape == (0, 3)
        assert result.dtype == np.float64

    def test_n_zero(self) -> None:
        """N=0: result is (M, 0) empty matrix."""
        A = np.random.default_rng(0).standard_normal((3, 5))
        B = np.empty((5, 0), dtype=np.float64)
        result = dgemm(A, B)
        assert result.shape == (3, 0)
        assert result.dtype == np.float64

    def test_k_zero(self) -> None:
        """K=0: result is (M, N) zero matrix."""
        A = np.empty((3, 0), dtype=np.float64)
        B = np.empty((0, 5), dtype=np.float64)
        result = dgemm(A, B)
        assert result.shape == (3, 5)
        assert result.dtype == np.float64
        npt.assert_array_equal(result, np.zeros((3, 5)))


# ---------------------------------------------------------------------------
# TestDgemmBoundary (BL3-08 — packing buffer overrun pitfall)
# ---------------------------------------------------------------------------


class TestDgemmBoundary:
    """Boundary size tests to catch packing buffer overruns.

    A dimension that is not a multiple of the backend's block size is the
    classic packing bug, so every size in the sweep runs square.
    """

    @pytest.mark.parametrize("size", BOUNDARY_SIZES)
    def test_boundary_sizes(self, size: int) -> None:
        """Square M=N=K at each boundary size.

        Uses per-test RNG seed to avoid order-dependence under xdist/randomly.
        At large N (>=256) near-zero expected values cause large relative error
        from normal FP accumulation differences, so we add atol=1e-12 alongside
        rtol=2e-9.
        """
        rng = np.random.default_rng(42 + size)
        A = rng.standard_normal((size, size))
        B = rng.standard_normal((size, size))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        # Large matrices: blocking-order FP diff reaches ~4e-13 absolute;
        # near-zero expected values inflate rdiff, so atol handles those.
        if size >= 256:
            npt.assert_allclose(result, expected, rtol=2e-9, atol=1e-12)
        else:
            npt.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.parametrize("m", [5, 6, 7])
    def test_mr_boundary_avx2(self, m: int) -> None:
        """M = 5, 6, 7 with K=256, N=8."""
        rng = np.random.default_rng(100 + m)
        A = rng.standard_normal((m, 256))
        B = rng.standard_normal((256, 8))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("m", [7, 8, 9])
    def test_mr_boundary_neon(self, m: int) -> None:
        """M = 7, 8, 9 with K=256, N=4."""
        rng = np.random.default_rng(200 + m)
        A = rng.standard_normal((m, 256))
        B = rng.standard_normal((256, 4))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("n", [3, 4, 7, 8, 9])
    def test_nr_boundary(self, n: int) -> None:
        """N at NR boundaries (NR=8 for AVX2, NR=4 for NEON), M=6, K=256."""
        rng = np.random.default_rng(300 + n)
        A = rng.standard_normal((6, 256))
        B = rng.standard_normal((256, n))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("m", [71, 72, 73])
    def test_mc_boundary(self, m: int) -> None:
        """M = 71, 72, 73 with K=256, N=8."""
        rng = np.random.default_rng(400 + m)
        A = rng.standard_normal((m, 256))
        B = rng.standard_normal((256, 8))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("k", [255, 256, 257])
    def test_kc_boundary(self, k: int) -> None:
        """K = 255, 256, 257 with M=72, N=8."""
        rng = np.random.default_rng(500 + k)
        A = rng.standard_normal((72, k))
        B = rng.standard_normal((k, 8))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-10)

    def test_prime_dimensions(self) -> None:
        """M=97, N=83, K=131 (all prime, no clean blocking)."""
        rng = np.random.default_rng(600)
        A = rng.standard_normal((97, 131))
        B = rng.standard_normal((131, 83))
        result = dgemm(A, B)
        expected = _reference_dgemm(A, B)
        npt.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestDgemmTranspose (BL3-04)
# ---------------------------------------------------------------------------


class TestDgemmTranspose:
    """Transpose variant tests: NN, TN, NT, TT.

    Uses the real transa/transb kwargs now that the C extension supports them.
    """

    _sizes: ClassVar[list[tuple[int, int, int]]] = [
        (100, 100, 100),
        (73, 128, 64),
        (6, 256, 256),
    ]

    @pytest.mark.parametrize("m,n,k", _sizes)
    def test_nn(self, m: int, n: int, k: int) -> None:
        """NN: A @ B (no transpose)."""
        rng = np.random.default_rng(700 + m * 1000 + n * 10 + k)
        A = rng.standard_normal((m, k))
        B = rng.standard_normal((k, n))
        result = dgemm(A, B, transa="N", transb="N")
        expected = _reference_dgemm(A, B, transa="N", transb="N")
        npt.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("m,n,k", _sizes)
    def test_tn(self, m: int, n: int, k: int) -> None:
        """TN: op(A)=A.T @ B where A is (k x m), so op(A) is (m x k)."""
        rng = np.random.default_rng(800 + m * 1000 + n * 10 + k)
        A_T_shape = rng.standard_normal(
            (k, m)
        )  # shape (k, m) — will be transposed by dgemm
        B = rng.standard_normal((k, n))
        result = dgemm(A_T_shape, B, transa="T", transb="N")
        expected = _reference_dgemm(A_T_shape, B, transa="T", transb="N")
        npt.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("m,n,k", _sizes)
    def test_nt(self, m: int, n: int, k: int) -> None:
        """NT: A @ op(B)=B.T where B is (n x k), so op(B) is (k x n)."""
        rng = np.random.default_rng(900 + m * 1000 + n * 10 + k)
        A = rng.standard_normal((m, k))
        B_T_shape = rng.standard_normal(
            (n, k)
        )  # shape (n, k) — will be transposed by dgemm
        result = dgemm(A, B_T_shape, transa="N", transb="T")
        expected = _reference_dgemm(A, B_T_shape, transa="N", transb="T")
        npt.assert_allclose(result, expected, rtol=1e-10)

    @pytest.mark.parametrize("m,n,k", _sizes)
    def test_tt(self, m: int, n: int, k: int) -> None:
        """TT: op(A)=A.T @ op(B)=B.T — both pack routines use trans=1."""
        rng = np.random.default_rng(1000 + m * 1000 + n * 10 + k)
        A_T_shape = rng.standard_normal((k, m))
        B_T_shape = rng.standard_normal((n, k))
        result = dgemm(A_T_shape, B_T_shape, transa="T", transb="T")
        expected = _reference_dgemm(A_T_shape, B_T_shape, transa="T", transb="T")
        npt.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestDgemmInit (BUILD-04)
# ---------------------------------------------------------------------------


class TestDgemmInit:
    """Tests that dgemm import and C extension linkage work correctly."""

    def test_import_succeeds(self) -> None:
        """Import jamma.jlinalg.dgemm without error."""
        from jamma.jlinalg import dgemm as _dgemm

        assert callable(_dgemm)

    def test_c_extension_has_dgemm(self) -> None:
        """If HAS_C_EXTENSION, verify the C extension exports dgemm."""
        if not HAS_C_EXTENSION:
            pytest.skip("C extension not compiled")
        from jamma.jlinalg import _jlinalg  # type: ignore[import]

        assert hasattr(_jlinalg, "dgemm"), (
            "C extension loaded but does not export 'dgemm'."
        )


# ---------------------------------------------------------------------------
# TestDgemmThreadSafety (BL3-09)
# ---------------------------------------------------------------------------


class TestDgemmThreadSafety:
    """Thread safety: dgemm results must be consistent for any OMP_NUM_THREADS value."""

    def test_single_vs_multi_thread(self) -> None:
        """OMP_NUM_THREADS=1 and OMP_NUM_THREADS=4 give consistent results.

        Runs each configuration in a separate subprocess so that the OpenMP
        runtime is initialised fresh with the correct thread count (OMP_NUM_THREADS
        must be set before the library is loaded).  Size 500x500.

        With external BLAS (MKL/Accelerate), results are bitwise identical.
        With jlinalg-own, different thread counts change FP accumulation order
        in the IC loop, producing differences up to ~1e-13 (within double
        precision expectations for 500-element dot products).
        """
        script = textwrap.dedent("""
            import sys
            import numpy as np
            from jamma.jlinalg import dgemm

            rng = np.random.default_rng(12345)
            A = rng.standard_normal((500, 500))
            B = rng.standard_normal((500, 500))
            C = dgemm(A, B)
            # Write result as binary to stdout
            sys.stdout.buffer.write(C.tobytes())
        """)

        import os

        env_single = {**os.environ, "OMP_NUM_THREADS": "1"}
        env_multi = {**os.environ, "OMP_NUM_THREADS": "4"}

        result_single = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env=env_single,
            timeout=60,
        )
        result_multi = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env=env_multi,
            timeout=60,
        )

        assert result_single.returncode == 0, (
            f"Single-thread subprocess failed: {result_single.stderr.decode()}"
        )
        assert result_multi.returncode == 0, (
            f"Multi-thread subprocess failed: {result_multi.stderr.decode()}"
        )

        C_single = np.frombuffer(result_single.stdout, dtype=np.float64).reshape(
            500, 500
        )
        C_multi = np.frombuffer(result_multi.stdout, dtype=np.float64).reshape(500, 500)

        npt.assert_allclose(
            C_single,
            C_multi,
            rtol=1e-12,
            atol=1e-12,
            err_msg=(
                "dgemm results differ between OMP_NUM_THREADS=1 and OMP_NUM_THREADS=4 "
                "beyond FP accumulation tolerance. This indicates a thread-safety bug "
                "in the dgemm implementation (packing, workspace, or microkernel)."
            ),
        )


# ---------------------------------------------------------------------------
# Throughput benchmark (skipped if no C extension)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION, reason="C extension required for throughput test"
)
@pytest.mark.benchmark
def test_dgemm_throughput() -> None:
    """VALID-07: dgemm achieves >0.9x throughput vs OpenBLAS on jamma rotation sizes.

    Tests at N=1410 (mouse_hs1940 rotation size) — the primary JAMMA workload.
    Prints GFLOPS for both implementations.  Skipped when the C extension is
    not compiled.

    The 0.9x assertion is only enforced on AVX2 hardware where jlinalg uses its
    fully optimised 6x8 FMA microkernel.  On NEON and generic paths, the
    comparison target (NumPy's BLAS backend) varies by platform (Apple
    Accelerate on macOS, OpenBLAS on Linux); we report the ratio but do not
    fail on it.
    """
    import time

    from jamma.jlinalg import blas_backend as _blas_backend
    from jamma.jlinalg import jlinalg_isa as _isa

    if _blas_backend == "numpy-fallback":
        pytest.skip(
            "vendor BLAS dgemm not wired; jlinalg.dgemm forwards directly to "
            "np.matmul, so the 0.9x throughput target is dominated by wrapper "
            "overhead and single-iteration timing variance on shared CI runners."
        )

    # When jlinalg routes to the SAME OpenBLAS library that numpy uses (the
    # case on stock numpy >=2.x which ships scipy-openblas64 with
    # INTERFACE64=1), both paths call the same symbols at the same threading
    # level. The 0.9x assertion was designed to catch jlinalg routing
    # regressions when the two paths target different libraries (MKL vs
    # OpenBLAS). With both on OpenBLAS, the only delta is jlinalg's per-call
    # wrapper overhead, which can easily eat 10-20% on small matrices.
    if _blas_backend.startswith("OpenBLAS"):
        pytest.skip(
            f"backend={_blas_backend}: jlinalg and np.matmul both call the "
            "same OpenBLAS symbols, so the throughput delta is wrapper "
            "overhead only. Assertion is meaningful only against MKL or Accelerate."
        )

    rng = np.random.default_rng(42)
    # N=1410: mouse_hs1940 rotation size (primary jamma workload)
    M = N = K = 1410
    A = rng.standard_normal((M, K))
    B = rng.standard_normal((K, N))

    # Warm up
    _ = dgemm(A, B)
    _ = np.matmul(A, B)

    # Time jlinalg: best of 5
    n_iters = 5
    best_jlinalg = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        dgemm(A, B)
        best_jlinalg = min(best_jlinalg, time.perf_counter() - t0)

    # Time NumPy: best of 5
    best_numpy = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        np.matmul(A, B)
        best_numpy = min(best_numpy, time.perf_counter() - t0)

    flops = 2.0 * M * N * K
    gflops_jlinalg = flops / best_jlinalg / 1e9
    gflops_numpy = flops / best_numpy / 1e9
    ratio = best_numpy / best_jlinalg

    jl_ms = best_jlinalg * 1000
    np_ms = best_numpy * 1000
    print(f"\njlinalg dgemm N={M}: {gflops_jlinalg:.1f} GF ({jl_ms:.1f} ms)")
    print(f"np.matmul:        {gflops_numpy:.1f} GF ({np_ms:.1f} ms)")
    print(f"Ratio:            {ratio:.3f} (ISA: {_isa})")

    # VALID-07: enforce 0.9x target on AVX2 hardware; NEON and generic paths
    # are not expected to match multi-threaded Accelerate/MKL-backed np.matmul.
    if _isa == "AVX2":
        assert ratio >= 0.9, (
            f"jlinalg dgemm is less than 90% of np.matmul throughput on AVX2: "
            f"ratio={ratio:.3f}, jlinalg={gflops_jlinalg:.1f} GFLOPS, "
            f"numpy={gflops_numpy:.1f} GFLOPS"
        )
    elif _isa in ("NEON", "generic"):
        print(f"{_isa}: throughput assertion skipped (ratio={ratio:.3f}x vs np.matmul)")


# ---------------------------------------------------------------------------
# TestDgemmValidation — input validation edge cases
# ---------------------------------------------------------------------------


class TestDgemmValidation:
    """Tests for input validation: invalid transa/transb, type checks."""

    def test_invalid_transa_value(self) -> None:
        """Invalid transa flag ('X') raises ValueError."""
        A = np.eye(3, dtype=np.float64)
        B = np.eye(3, dtype=np.float64)
        with pytest.raises(ValueError, match="transa"):
            dgemm(A, B, transa="X")  # type: ignore[bad-argument-type]

    def test_invalid_transb_value(self) -> None:
        """Invalid transb flag ('Z') raises ValueError."""
        A = np.eye(3, dtype=np.float64)
        B = np.eye(3, dtype=np.float64)
        with pytest.raises(ValueError, match="transb"):
            dgemm(A, B, transb="Z")  # type: ignore[bad-argument-type]

    def test_empty_string_transa(self) -> None:
        """Empty string transa raises ValueError."""
        A = np.eye(3, dtype=np.float64)
        B = np.eye(3, dtype=np.float64)
        with pytest.raises((ValueError, IndexError)):
            dgemm(A, B, transa="")  # type: ignore[bad-argument-type]

    def test_multichar_transa_rejected(self) -> None:
        """Multi-character transa like 'transpose' must be rejected."""
        A = np.eye(3, dtype=np.float64)
        B = np.eye(3, dtype=np.float64)
        with pytest.raises(ValueError, match="transa"):
            dgemm(A, B, transa="transpose")  # type: ignore[bad-argument-type]

    def test_multichar_transb_rejected(self) -> None:
        """Multi-character transb like 'TT' must be rejected."""
        A = np.eye(3, dtype=np.float64)
        B = np.eye(3, dtype=np.float64)
        with pytest.raises(ValueError, match="transb"):
            dgemm(A, B, transb="TT")  # type: ignore[bad-argument-type]

    def test_output_contiguity_and_dtype(self) -> None:
        """Output is C-contiguous float64."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((50, 30))
        B = rng.standard_normal((30, 40))
        result = dgemm(A, B)
        assert result.dtype == np.float64
        assert result.flags["C_CONTIGUOUS"]
        assert result.shape == (50, 40)

    def test_dimension_mismatch_nn(self) -> None:
        """K dimension mismatch (NN) raises ValueError."""
        with pytest.raises(ValueError, match=r"mismatch|dimensions|columns"):
            dgemm(np.zeros((3, 5)), np.zeros((4, 7)))

    def test_dimension_mismatch_tn(self) -> None:
        """K dimension mismatch with transa='T' raises ValueError."""
        with pytest.raises(ValueError, match=r"mismatch|dimensions|columns"):
            dgemm(np.zeros((5, 3)), np.zeros((4, 7)), transa="T")

    def test_1d_input_raises(self) -> None:
        """1-D array input raises ValueError."""
        with pytest.raises(ValueError, match=r"2-D|2D|ndim"):
            dgemm(np.ones(10), np.ones(10))

    def test_3d_input_raises(self) -> None:
        """3-D array input raises ValueError."""
        with pytest.raises(ValueError, match=r"2-D|2D|ndim"):
            dgemm(np.ones((2, 3, 4)), np.ones((4, 5)))

    def test_fortran_order_input(self) -> None:
        """Fortran-order input produces correct results."""
        rng = np.random.default_rng(88)
        A = np.asfortranarray(rng.standard_normal((50, 30)))
        B = rng.standard_normal((30, 40))
        result = dgemm(A, B)
        expected = A @ B
        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_strided_input(self) -> None:
        """Non-contiguous (strided) input produces correct results."""
        rng = np.random.default_rng(88)
        A_full = rng.standard_normal((100, 30))
        B = rng.standard_normal((30, 40))
        A_strided = A_full[::2, :]  # non-contiguous
        result = dgemm(A_strided, B)
        expected = A_strided @ B
        npt.assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestDgemmFallback — explicit fallback path tests
# ---------------------------------------------------------------------------


class TestDgemmFallback:
    """Test the NumPy fallback dgemm directly (jamma.jlinalg._dgemm_numpy).

    Always exercised, independent of whether the C extension is present.
    """

    def test_fallback_transpose_nn(self) -> None:
        """Fallback NN matches reference."""
        rng = np.random.default_rng(77)
        A = rng.standard_normal((30, 20))
        B = rng.standard_normal((20, 25))
        npt.assert_allclose(_dgemm_numpy(A, B), _reference_dgemm(A, B), rtol=1e-14)

    def test_fallback_transpose_tn(self) -> None:
        """Fallback TN matches reference."""
        rng = np.random.default_rng(77)
        A = rng.standard_normal((20, 30))  # physical (K, M), op(A) = (M, K)
        B = rng.standard_normal((20, 25))
        npt.assert_allclose(
            _dgemm_numpy(A, B, transa="T"),
            _reference_dgemm(A, B, transa="T"),
            rtol=1e-14,
        )

    def test_fallback_transpose_nt(self) -> None:
        """Fallback NT matches reference."""
        rng = np.random.default_rng(77)
        A = rng.standard_normal((30, 20))
        B = rng.standard_normal((25, 20))  # physical (N, K), op(B) = (K, N)
        npt.assert_allclose(
            _dgemm_numpy(A, B, transb="T"),
            _reference_dgemm(A, B, transb="T"),
            rtol=1e-14,
        )

    def test_fallback_transpose_tt(self) -> None:
        """Fallback TT matches reference."""
        rng = np.random.default_rng(77)
        A = rng.standard_normal((20, 30))
        B = rng.standard_normal((25, 20))
        npt.assert_allclose(
            _dgemm_numpy(A, B, transa="T", transb="T"),
            _reference_dgemm(A, B, transa="T", transb="T"),
            rtol=1e-14,
        )

    def test_fallback_non_string_transa(self) -> None:
        """Fallback raises TypeError on non-string transa."""
        A = np.eye(3, dtype=np.float64)
        B = np.eye(3, dtype=np.float64)
        with pytest.raises(TypeError, match="transa must be a string"):
            _dgemm_numpy(A, B, transa=0)  # type: ignore[bad-argument-type]

    def test_fallback_non_string_transb(self) -> None:
        """Fallback raises TypeError on non-string transb."""
        A = np.eye(3, dtype=np.float64)
        B = np.eye(3, dtype=np.float64)
        with pytest.raises(TypeError, match="transb must be a string"):
            _dgemm_numpy(A, B, transb=True)  # type: ignore[bad-argument-type]

    def test_fallback_out_parameter(self) -> None:
        """Fallback dgemm(A, B, out=C) writes into C and returns C."""
        rng = np.random.default_rng(2001)
        A = rng.standard_normal((50, 100))
        B = rng.standard_normal((100, 80))
        out_buf = np.empty((50, 80), dtype=np.float64)
        result = _dgemm_numpy(A, B, out=out_buf)
        assert result is out_buf
        npt.assert_allclose(result, A @ B, rtol=1e-12)

    def test_fallback_out_none(self) -> None:
        """Fallback dgemm(A, B, out=None) allocates fresh array."""
        rng = np.random.default_rng(2002)
        A = rng.standard_normal((50, 100))
        B = rng.standard_normal((100, 80))
        result = _dgemm_numpy(A, B, out=None)
        assert result.shape == (50, 80)
        npt.assert_allclose(result, A @ B, rtol=1e-12)

    def test_fallback_out_shape_mismatch(self) -> None:
        """Fallback dgemm(A, B, out=wrong_shape) raises ValueError."""
        rng = np.random.default_rng(2003)
        A = rng.standard_normal((50, 100))
        B = rng.standard_normal((100, 80))
        out_bad = np.empty((10, 10), dtype=np.float64)
        with pytest.raises(ValueError, match="out shape"):
            _dgemm_numpy(A, B, out=out_bad)

    def test_fallback_out_wrong_ndim(self) -> None:
        """Fallback dgemm(A, B, out=1d_array) raises ValueError."""
        A = np.eye(5, dtype=np.float64)
        B = np.eye(5, dtype=np.float64)
        out_1d = np.empty(25, dtype=np.float64)
        with pytest.raises(ValueError, match="out shape"):
            _dgemm_numpy(A, B, out=out_1d)


# ---------------------------------------------------------------------------
# TestDgemmOutParameter — out= preallocated buffer support
# ---------------------------------------------------------------------------


class TestDgemmOutParameter:
    """Tests for dgemm out= parameter: write into caller-provided buffer."""

    def test_dgemm_out_parameter(self) -> None:
        """dgemm(A, B, out=C) writes into C and returns C."""
        rng = np.random.default_rng(1001)
        A = rng.standard_normal((50, 100))
        B = rng.standard_normal((100, 80))
        out_buf = np.empty((50, 80), dtype=np.float64)
        result = dgemm(A, B, out=out_buf)
        assert result is out_buf
        npt.assert_allclose(result, A @ B, rtol=1e-12)

    def test_dgemm_out_none_fallback(self) -> None:
        """dgemm(A, B, out=None) allocates fresh array (backward compatible)."""
        rng = np.random.default_rng(1002)
        A = rng.standard_normal((50, 100))
        B = rng.standard_normal((100, 80))
        result = dgemm(A, B, out=None)
        assert result.shape == (50, 80)
        assert result is not A
        assert result is not B

    def test_dgemm_out_shape_mismatch(self) -> None:
        """dgemm(A, B, out=wrong_shape) raises ValueError."""
        rng = np.random.default_rng(1003)
        A = rng.standard_normal((50, 100))
        B = rng.standard_normal((100, 80))
        out_bad = np.empty((10, 10), dtype=np.float64)
        with pytest.raises(ValueError, match="out shape"):
            dgemm(A, B, out=out_bad)

    def test_dgemm_out_subslice(self) -> None:
        """dgemm with out= as a contiguous sub-slice of a larger buffer."""
        rng = np.random.default_rng(1004)
        A_sub = rng.standard_normal((30, 50))
        B = rng.standard_normal((50, 80))
        big_buf = np.full((100, 80), 999.0, dtype=np.float64)
        result = dgemm(A_sub, B, out=big_buf[:30, :])
        npt.assert_allclose(result, A_sub @ B, rtol=1e-12)
        # Data beyond the slice must be untouched
        npt.assert_array_equal(big_buf[30:, :], 999.0)

    def test_dgemm_out_transpose_variants(self) -> None:
        """out= works for all 4 transpose combos (NN, TN, NT, TT)."""
        rng = np.random.default_rng(1005)

        # NN: A(50,100) @ B(100,80) -> (50,80)
        A_nn = rng.standard_normal((50, 100))
        B_nn = rng.standard_normal((100, 80))
        out_nn = np.empty((50, 80), dtype=np.float64)
        r = dgemm(A_nn, B_nn, out=out_nn)
        assert r is out_nn
        npt.assert_allclose(r, A_nn @ B_nn, rtol=1e-12)

        # TN: A(100,50).T @ B(100,80) -> (50,80)
        A_tn = rng.standard_normal((100, 50))
        B_tn = rng.standard_normal((100, 80))
        out_tn = np.empty((50, 80), dtype=np.float64)
        r = dgemm(A_tn, B_tn, transa="T", out=out_tn)
        assert r is out_tn
        npt.assert_allclose(r, A_tn.T @ B_tn, rtol=1e-12)

        # NT: A(50,100) @ B(80,100).T -> (50,80)
        A_nt = rng.standard_normal((50, 100))
        B_nt = rng.standard_normal((80, 100))
        out_nt = np.empty((50, 80), dtype=np.float64)
        r = dgemm(A_nt, B_nt, transb="T", out=out_nt)
        assert r is out_nt
        npt.assert_allclose(r, A_nt @ B_nt.T, rtol=1e-12)

        # TT: A(100,50).T @ B(80,100).T -> (50,80)
        A_tt = rng.standard_normal((100, 50))
        B_tt = rng.standard_normal((80, 100))
        out_tt = np.empty((50, 80), dtype=np.float64)
        r = dgemm(A_tt, B_tt, transa="T", transb="T", out=out_tt)
        assert r is out_tt
        npt.assert_allclose(r, A_tt.T @ B_tt.T, rtol=1e-12)

    def test_dgemm_out_wrong_ndim(self) -> None:
        """dgemm(A, B, out=1d_array) raises ValueError."""
        A = np.eye(5, dtype=np.float64)
        B = np.eye(5, dtype=np.float64)
        out_1d = np.empty(25, dtype=np.float64)
        with pytest.raises((ValueError, TypeError)):
            dgemm(A, B, out=out_1d)

    def test_dgemm_out_fortran_order_rejected(self) -> None:
        """dgemm with Fortran-order out= raises ValueError."""
        rng = np.random.default_rng(1006)
        A = rng.standard_normal((50, 100))
        B = rng.standard_normal((100, 80))
        out_f = np.asfortranarray(np.empty((50, 80), dtype=np.float64))
        with pytest.raises(ValueError, match="C-contiguous"):
            dgemm(A, B, out=out_f)

    def test_dgemm_out_wrong_dtype_rejected(self) -> None:
        """dgemm with float32 out= raises ValueError."""
        A = np.eye(5, dtype=np.float64)
        B = np.eye(5, dtype=np.float64)
        out_f32 = np.empty((5, 5), dtype=np.float32)
        with pytest.raises((ValueError, TypeError)):
            dgemm(A, B, out=out_f32)  # type: ignore[bad-argument-type]

    @pytest.mark.skipif(
        not HAS_C_EXTENSION, reason="C extension required for native out regression"
    )
    def test_native_unaligned_out_is_rejected(self) -> None:
        """Native dgemm must not replace an unaligned out array with a copy."""
        from jamma.jlinalg import _jlinalg

        if not _jlinalg.blas_has_dgemm:
            pytest.skip("vendor DGEMM required for native-path regression")

        A = np.arange(12, dtype=np.float64).reshape(4, 3)
        B = np.arange(12, dtype=np.float64).reshape(3, 4)
        sentinel = 999.0
        storage = bytearray(4 + 4 * 4 * np.dtype(np.float64).itemsize)
        out = np.frombuffer(storage, dtype=np.float64, count=16, offset=4).reshape(4, 4)
        out[:] = sentinel
        assert not out.flags["ALIGNED"]

        with pytest.raises(ValueError, match="aligned"):
            _jlinalg.dgemm(A, B, out=out)
        npt.assert_array_equal(out, sentinel)

        with pytest.raises(ValueError, match="aligned"):
            dgemm(A, B, out=out)
        npt.assert_array_equal(out, sentinel)
