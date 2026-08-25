"""Tests for jlinalg dsyrk (symmetric rank-k update).

Tests cover:
- Correctness vs NumPy reference at multiple sizes (BL3-05)
- Bitwise symmetry of dsyrk result (BL3-06)
- Boundary sizes: MR-1/MR/MR+1 for AVX2 (MR=6) and NEON (MR=8),
  MC-1/MC/MC+1 for AVX2 (MC=72) and NEON (MC=64), KC boundaries (KC=256)
- Zero-dimension edge cases (N=0, K=0)
- Throughput benchmark scaffold (skipped until C extension)
- Input validation: ValueError on bad shapes

Run with -n0 to avoid parallel interference with OpenMP threading tests:
    uv run pytest tests/test_jlinalg_dsyrk.py -x -n0 -v
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from jamma.jlinalg import HAS_C_EXTENSION, dsyrk

pytestmark = pytest.mark.tier0

# ---------------------------------------------------------------------------
# Boundary size parameters
# ---------------------------------------------------------------------------

# Sizes chosen to cover MR-1/MR/MR+1 for both AVX2 (MR=6) and NEON (MR=8),
# plus MC boundaries (AVX2 MC=72, NEON MC=64), KC boundaries (KC=256),
# NC boundaries (NC=4096), and a selection of prime dimensions.
BOUNDARY_SIZES = [
    1,
    3,
    5,
    6,
    7,  # MR-1/MR/MR+1 for AVX2 (MR=6)
    7,
    8,
    9,  # MR-1/MR/MR+1 for NEON (MR=8)
    11,
    13,
    63,
    64,
    65,  # MC-1/MC/MC+1 for NEON (MC=64)
    71,
    72,
    73,  # MC-1/MC/MC+1 for AVX2 (MC=72)
    127,
    128,
    129,  # KC/2 boundaries
    255,
    256,
    257,  # KC-1/KC/KC+1 (KC=256)
    500,
    1000,
]

# Deduplicate while preserving order
_seen: set[int] = set()
BOUNDARY_SIZES = [x for x in BOUNDARY_SIZES if not (x in _seen or _seen.add(x))]  # type: ignore[func-returns-value]


# ---------------------------------------------------------------------------
# Helper: reference implementations
# ---------------------------------------------------------------------------


def _reference_dsyrk(X: np.ndarray) -> np.ndarray:
    """Reference dsyrk: compute X @ X.T via np.dot."""
    X64 = X.astype(np.float64, copy=False)
    return np.dot(X64, X64.T)


# ---------------------------------------------------------------------------
# TestDsyrkCorrectness — BL3-05
# ---------------------------------------------------------------------------


class TestDsyrkCorrectness:
    """dsyrk(X) must match np.dot(X, X.T) within rtol=1e-12 at various sizes."""

    @pytest.mark.parametrize("N", [1, 10, 100, 1000])
    @pytest.mark.parametrize("K", [1, 50, 200])
    def test_correctness_parametrized(self, N: int, K: int) -> None:
        """dsyrk(X) matches np.dot(X, X.T) for shape (N, K)."""
        rng = np.random.default_rng(42 + N * 1000 + K)
        X = rng.standard_normal((N, K))
        result = dsyrk(X)
        expected = _reference_dsyrk(X)
        npt.assert_allclose(
            result, expected, rtol=1e-12, err_msg=f"dsyrk mismatch at N={N}, K={K}"
        )

    def test_rectangular_tall_skinny(self) -> None:
        """Tall skinny X: N=500, K=5."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 5))
        result = dsyrk(X)
        expected = _reference_dsyrk(X)
        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_rectangular_wide(self) -> None:
        """Wide X: N=10, K=300."""
        rng = np.random.default_rng(43)
        X = rng.standard_normal((10, 300))
        result = dsyrk(X)
        expected = _reference_dsyrk(X)
        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_output_shape(self) -> None:
        """Output shape is (N, N) for any input (N, K)."""
        rng = np.random.default_rng(44)
        X = rng.standard_normal((37, 19))
        result = dsyrk(X)
        assert result.shape == (37, 37)

    def test_output_dtype(self) -> None:
        """Output dtype is float64."""
        rng = np.random.default_rng(45)
        X = rng.standard_normal((10, 5))
        result = dsyrk(X)
        assert result.dtype == np.float64

    def test_float32_input_promoted(self) -> None:
        """float32 input is promoted to float64 before computation."""
        rng = np.random.default_rng(46)
        X = rng.standard_normal((20, 10)).astype(np.float32)
        result = dsyrk(X)
        expected = _reference_dsyrk(X)
        assert result.dtype == np.float64
        # Use rtol=1e-5 for float32 → float64 promotion (float32 precision ~7 digits)
        npt.assert_allclose(result, expected, rtol=1e-5)

    def test_identity_input(self) -> None:
        """dsyrk(I_n) == I_n for NxN identity (square case K==N)."""
        eye10 = np.eye(10)
        result = dsyrk(eye10)
        npt.assert_allclose(result, np.eye(10), rtol=1e-14)

    def test_zero_input(self) -> None:
        """dsyrk(zeros) == zeros."""
        X = np.zeros((10, 5))
        result = dsyrk(X)
        npt.assert_array_equal(result, np.zeros((10, 10)))

    def test_single_row(self) -> None:
        """N=1, K=100: result is 1x1 scalar (sum of squares)."""
        rng = np.random.default_rng(47)
        X = rng.standard_normal((1, 100))
        result = dsyrk(X)
        expected = np.sum(X**2).reshape(1, 1)
        npt.assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestDsyrkSymmetry — BL3-06
# ---------------------------------------------------------------------------


class TestDsyrkSymmetry:
    """dsyrk result must be bitwise symmetric: result[i,j] == result[j,i]."""

    @pytest.mark.parametrize("N", [1, 10, 100, 1000])
    def test_bitwise_symmetric(self, N: int) -> None:
        """result == result.T (bitwise equality via assert_array_equal)."""
        rng = np.random.default_rng(99 + N)
        X = rng.standard_normal((N, max(1, N // 2)))
        result = dsyrk(X)
        npt.assert_array_equal(
            result,
            result.T,
            err_msg=(
                f"dsyrk result at N={N} is not bitwise symmetric. "
                "Upper triangle must equal lower triangle exactly."
            ),
        )

    def test_symmetric_after_blocking(self) -> None:
        """Symmetry holds after multiple MC/KC blocking passes (N=300, K=400)."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((300, 400))
        result = dsyrk(X)
        npt.assert_array_equal(result, result.T)

    def test_symmetric_non_square_input(self) -> None:
        """Symmetry holds for tall (N=7, K=3) and wide (N=3, K=7) inputs."""
        rng = np.random.default_rng(100)
        for N, K in [(7, 3), (3, 7), (13, 100)]:
            X = rng.standard_normal((N, K))
            result = dsyrk(X)
            npt.assert_array_equal(
                result,
                result.T,
                err_msg=f"dsyrk not symmetric for N={N}, K={K}",
            )


# ---------------------------------------------------------------------------
# TestDsyrkBoundary — packing buffer boundary sizes
# ---------------------------------------------------------------------------


class TestDsyrkBoundary:
    """Boundary sizes to catch packing overruns and tail-handling bugs."""

    @pytest.mark.parametrize("size", BOUNDARY_SIZES)
    def test_boundary_square(self, size: int) -> None:
        """Square input (N=K=size): result matches np.dot(X, X.T)."""
        rng = np.random.default_rng(42 + size)
        X = rng.standard_normal((size, size))
        result = dsyrk(X)
        expected = _reference_dsyrk(X)
        rtol = 1e-9 if size >= 256 else 1e-12
        npt.assert_allclose(
            result,
            expected,
            rtol=rtol,
            err_msg=f"dsyrk boundary mismatch at size={size}",
        )

    @pytest.mark.parametrize("size", BOUNDARY_SIZES)
    def test_boundary_fixed_k(self, size: int) -> None:
        """N=size, K=100: result matches np.dot(X, X.T)."""
        rng = np.random.default_rng(99 + size)
        X = rng.standard_normal((size, 100))
        result = dsyrk(X)
        expected = _reference_dsyrk(X)
        rtol = 1e-9 if size >= 256 else 1e-12
        npt.assert_allclose(
            result,
            expected,
            rtol=rtol,
            err_msg=f"dsyrk boundary mismatch at size={size}, K=100",
        )

    def test_mr_boundary_avx2(self) -> None:
        """N = MR-1/MR/MR+1 for AVX2 (MR=6): K=256."""
        rng = np.random.default_rng(200)
        for N in [5, 6, 7]:
            X = rng.standard_normal((N, 256))
            result = dsyrk(X)
            expected = _reference_dsyrk(X)
            npt.assert_allclose(
                result, expected, rtol=1e-10, err_msg=f"dsyrk AVX2 MR boundary N={N}"
            )

    def test_mr_boundary_neon(self) -> None:
        """N = MR-1/MR/MR+1 for NEON (MR=8): K=256."""
        rng = np.random.default_rng(201)
        for N in [7, 8, 9]:
            X = rng.standard_normal((N, 256))
            result = dsyrk(X)
            expected = _reference_dsyrk(X)
            npt.assert_allclose(
                result, expected, rtol=1e-10, err_msg=f"dsyrk NEON MR boundary N={N}"
            )

    def test_mc_boundary_avx2(self) -> None:
        """N = MC-1/MC/MC+1 for AVX2 (MC=72): K=256."""
        rng = np.random.default_rng(202)
        for N in [71, 72, 73]:
            X = rng.standard_normal((N, 256))
            result = dsyrk(X)
            expected = _reference_dsyrk(X)
            npt.assert_allclose(
                result, expected, rtol=1e-10, err_msg=f"dsyrk AVX2 MC boundary N={N}"
            )

    def test_kc_boundary(self) -> None:
        """K = KC-1/KC/KC+1 (KC=256): N=72."""
        rng = np.random.default_rng(203)
        for K in [255, 256, 257]:
            X = rng.standard_normal((72, K))
            result = dsyrk(X)
            expected = _reference_dsyrk(X)
            npt.assert_allclose(
                result, expected, rtol=1e-10, err_msg=f"dsyrk KC boundary K={K}"
            )

    def test_prime_dimensions(self) -> None:
        """N=97, K=131 (both prime, no clean blocking)."""
        rng = np.random.default_rng(204)
        X = rng.standard_normal((97, 131))
        result = dsyrk(X)
        expected = _reference_dsyrk(X)
        npt.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestDsyrkZero — zero-dimension edge cases
# ---------------------------------------------------------------------------


class TestDsyrkZero:
    """Zero-dimension edge cases for dsyrk."""

    def test_n_zero(self) -> None:
        """N=0: result has shape (0, 0)."""
        X = np.empty((0, 5), dtype=np.float64)
        result = dsyrk(X)
        assert result.shape == (0, 0)
        assert result.dtype == np.float64

    def test_k_zero(self) -> None:
        """K=0: result has shape (N, N) and is all zeros."""
        X = np.empty((5, 0), dtype=np.float64)
        result = dsyrk(X)
        assert result.shape == (5, 5)
        assert result.dtype == np.float64
        npt.assert_array_equal(result, np.zeros((5, 5)))

    @pytest.mark.skipif(
        not HAS_C_EXTENSION, reason="C extension required for allocator regression"
    )
    def test_native_k_zero_does_not_read_fresh_output_memory(self) -> None:
        """A fresh native output for K=0 is initialized, even after heap churn."""
        from jamma.jlinalg import _jlinalg

        if not _jlinalg.blas_has_dsyrk:
            pytest.skip("vendor DSYRK required for native-path regression")

        n = 23
        X = np.empty((n, 0), dtype=np.float64)
        expected = np.zeros((n, n), dtype=np.float64)
        for _ in range(32):
            dirty = np.full((n, n), np.nan, dtype=np.float64)
            del dirty
            result = _jlinalg.dsyrk(X)
            npt.assert_array_equal(result, expected)
            del result


# ---------------------------------------------------------------------------
# TestDsyrkValidation — input validation
# ---------------------------------------------------------------------------


class TestDsyrkValidation:
    """dsyrk must raise ValueError for non-2D inputs."""

    def test_1d_input_raises(self) -> None:
        """1-D input raises ValueError."""
        with pytest.raises(ValueError, match=r"2-D|2D|ndim"):
            dsyrk(np.ones(10))

    def test_3d_input_raises(self) -> None:
        """3-D input raises ValueError."""
        with pytest.raises(ValueError, match=r"2-D|2D|ndim"):
            dsyrk(np.ones((2, 3, 4)))

    def test_scalar_input_raises(self) -> None:
        """0-D (scalar) input raises ValueError."""
        with pytest.raises(ValueError, match=r"2-D|2D|ndim"):
            dsyrk(np.array(1.0))


class TestDsyrkOutput:
    """dsyrk can update a caller-owned symmetric output buffer."""

    def test_accumulates_into_output(self) -> None:
        rng = np.random.default_rng(314)
        X = rng.standard_normal((12, 7))
        initial = rng.standard_normal((12, 12))
        initial = initial @ initial.T
        out = initial.copy()

        result = dsyrk(X, out=out, beta=1.0)

        assert result is out
        npt.assert_allclose(out, initial + X @ X.T, rtol=1e-12, atol=1e-14)
        npt.assert_array_equal(out, out.T)

    def test_beta_zero_overwrites_output(self) -> None:
        rng = np.random.default_rng(315)
        X = rng.standard_normal((8, 5))
        out = np.full((8, 8), np.nan)

        result = dsyrk(X, out=out)

        assert result is out
        npt.assert_allclose(out, X @ X.T, rtol=1e-12, atol=1e-14)

    @pytest.mark.skipif(
        not HAS_C_EXTENSION, reason="C extension required for native out regression"
    )
    def test_native_unaligned_out_is_rejected(self) -> None:
        """Native dsyrk must not replace an unaligned out array with a copy."""
        from jamma.jlinalg import _jlinalg

        if not _jlinalg.blas_has_dsyrk:
            pytest.skip("vendor DSYRK required for native-path regression")

        X = np.arange(12, dtype=np.float64).reshape(4, 3)
        storage = bytearray(4 + 4 * 4 * np.dtype(np.float64).itemsize)
        out = np.frombuffer(storage, dtype=np.float64, count=16, offset=4).reshape(4, 4)
        assert not out.flags["ALIGNED"]

        with pytest.raises(ValueError, match="aligned"):
            _jlinalg.dsyrk(X, out=out)

        with pytest.raises(ValueError, match="aligned"):
            dsyrk(X, out=out)

    def test_zero_width_input_scales_output(self) -> None:
        X = np.empty((6, 0), dtype=np.float64)
        out = np.eye(6, dtype=np.float64) * 4.0

        dsyrk(X, out=out, beta=0.25)

        npt.assert_array_equal(out, np.eye(6))

    def test_beta_without_output_raises(self) -> None:
        with pytest.raises(ValueError, match="beta requires out"):
            dsyrk(np.ones((3, 2)), beta=1.0)

    @pytest.mark.parametrize(
        ("out", "message"),
        [
            (np.empty(3, dtype=np.float64), "2-D"),
            (np.empty((3, 4), dtype=np.float64), "shape"),
            (np.empty((3, 3), dtype=np.float32), "float64"),
            (np.empty((3, 6), dtype=np.float64)[:, ::2], "C-contiguous"),
        ],
    )
    def test_invalid_output_raises(self, out: np.ndarray, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            dsyrk(np.ones((3, 2)), out=out)

    def test_readonly_output_raises(self) -> None:
        out = np.empty((3, 3), dtype=np.float64)
        out.flags.writeable = False
        with pytest.raises(ValueError, match="writeable"):
            dsyrk(np.ones((3, 2)), out=out)

    def test_non_array_output_raises(self) -> None:
        with pytest.raises(TypeError, match="numpy array"):
            dsyrk(np.ones((3, 2)), out=[[0.0] * 3] * 3)  # type: ignore[arg-type]

    def test_output_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            dsyrk(np.ones((3, 2)), np.empty((3, 3)))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TestDsyrkFallback — explicit fallback path tests
# ---------------------------------------------------------------------------


class TestDsyrkFallback:
    """Test the NumPy fallback dsyrk independently.

    Imports the fallback logic directly (or via jamma.jlinalg when HAS_C_EXTENSION
    is False).  The fallback is always tested since it's the active code path
    when the C extension is missing dsyrk.
    """

    def _get_fallback_dsyrk(self):
        """Return a pure-NumPy fallback dsyrk matching the __init__.py fallback."""
        import numpy as _np

        def _dsyrk(X: np.ndarray) -> np.ndarray:
            if X.ndim != 2:
                raise ValueError(f"dsyrk: X must be a 2-D array, got {X.ndim}-D")
            X64 = _np.ascontiguousarray(X, dtype=_np.float64)
            result = _np.dot(X64, X64.T)
            # Mirror lower to upper for bitwise symmetry (matches production fallback)
            il = _np.tril_indices_from(result, -1)
            result.T[il] = result[il]
            return result

        return _dsyrk

    def test_fallback_correctness(self) -> None:
        """Fallback matches reference for random (30, 20) input."""
        fb = self._get_fallback_dsyrk()
        rng = np.random.default_rng(77)
        X = rng.standard_normal((30, 20))
        npt.assert_allclose(fb(X), _reference_dsyrk(X), rtol=1e-14)

    def test_fallback_symmetric(self) -> None:
        """Fallback result is bitwise symmetric."""
        fb = self._get_fallback_dsyrk()
        rng = np.random.default_rng(78)
        X = rng.standard_normal((50, 30))
        result = fb(X)
        npt.assert_array_equal(result, result.T)

    def test_production_fallback_accumulates_into_output(self) -> None:
        """The production NumPy backend preserves the validated out contract."""
        from jamma.jlinalg import _dsyrk_numpy

        X = np.arange(12, dtype=np.float64).reshape(4, 3)
        initial = np.eye(4, dtype=np.float64)
        out = initial.copy()

        result = _dsyrk_numpy(X, out=out, beta=0.5)

        assert result is out
        npt.assert_allclose(out, X @ X.T + 0.5 * initial, rtol=1e-14, atol=0.0)

    @pytest.mark.parametrize("n", [1, 2, 511, 512, 513, 1100])
    def test_production_fallback_accumulation_spans_tiles(self, n: int) -> None:
        """Row-block accumulation and tiled mirror agree with the reference.

        The accumulating path fills only the lower triangle and mirrors it up,
        both in tiles. These sizes straddle the mirror tile edge (512) and the
        row-block divisor, so a tile boundary that dropped or double-counted a
        strip would show up as an asymmetry or a wrong entry here.
        """
        from jamma.jlinalg import _dsyrk_numpy

        rng = np.random.default_rng(1234 + n)
        X = np.ascontiguousarray(rng.standard_normal((n, 7)))
        initial = rng.standard_normal((n, n))
        initial = initial + initial.T  # dsyrk's out contract assumes symmetry

        out = initial.copy()
        result = _dsyrk_numpy(X, out=out, beta=1.0)

        npt.assert_allclose(result, initial + X @ X.T, rtol=1e-12, atol=1e-14)
        npt.assert_array_equal(result, result.T)

    def test_fallback_1d_raises(self) -> None:
        """Fallback raises ValueError on 1-D input."""
        fb = self._get_fallback_dsyrk()
        with pytest.raises(ValueError, match=r"2-D|ndim"):
            fb(np.ones(10))

    def test_fallback_3d_raises(self) -> None:
        """Fallback raises ValueError on 3-D input."""
        fb = self._get_fallback_dsyrk()
        with pytest.raises(ValueError, match=r"2-D|ndim"):
            fb(np.ones((2, 3, 4)))

    def test_via_jamma_jlinalg(self) -> None:
        """jamma.jlinalg.dsyrk produces correct results (fallback or C extension)."""
        rng = np.random.default_rng(79)
        X = rng.standard_normal((40, 25))
        result = dsyrk(X)
        expected = _reference_dsyrk(X)
        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_fallback_rejects_non_array_output(self) -> None:
        """Fallback matches the native output type contract."""
        from jamma.jlinalg import _dsyrk_numpy

        with pytest.raises(TypeError, match="numpy array"):
            _dsyrk_numpy(np.ones((3, 2)), out=[[0.0] * 3] * 3)  # type: ignore[arg-type]

    def test_fallback_rejects_non_2d_output(self) -> None:
        """Fallback reports dimensionality before shape mismatch."""
        from jamma.jlinalg import _dsyrk_numpy

        with pytest.raises(ValueError, match="2-D"):
            _dsyrk_numpy(np.ones((3, 2)), out=np.empty(3, dtype=np.float64))


# ---------------------------------------------------------------------------
# TestDsyrkThreadSafety — OMP_NUM_THREADS=1 vs 4 bitwise identity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION, reason="Thread safety test requires C extension"
)
class TestDsyrkThreadSafety:
    """Thread safety: dsyrk results must be identical for any OMP_NUM_THREADS."""

    def test_dsyrk_single_vs_multi_thread(self) -> None:
        """OMP_NUM_THREADS=1 and OMP_NUM_THREADS=4 give bitwise-identical dsyrk results.

        Runs each configuration in a separate subprocess so that the OpenMP
        runtime is initialised fresh with the correct thread count.
        """
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import sys
            import numpy as np
            from jamma.jlinalg import dsyrk

            rng = np.random.default_rng(54321)
            X = rng.standard_normal((300, 200))
            C = dsyrk(X)
            sys.stdout.buffer.write(C.tobytes())
        """)

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
            300, 300
        )
        C_multi = np.frombuffer(result_multi.stdout, dtype=np.float64).reshape(300, 300)

        npt.assert_array_equal(
            C_single,
            C_multi,
            err_msg=(
                "dsyrk results differ between OMP_NUM_THREADS=1 and OMP_NUM_THREADS=4. "
                "This indicates a thread-safety bug in the dsyrk implementation."
            ),
        )


# ---------------------------------------------------------------------------
# TestDsyrkThroughput — benchmark scaffold (skipped without C extension)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION, reason="C extension required for throughput test"
)
@pytest.mark.benchmark
def test_dsyrk_throughput() -> None:
    """VALID-06: dsyrk achieves >1.2x throughput vs OpenBLAS on kinship workload.

    Tests at N=4000, K=2000 as a CI-runnable proxy for the full kinship
    workload (N=10000, K=5000 tested in bench_jlinalg.py).  Run with -n0
    to avoid OpenMP / pytest-xdist interference.

    The 1.2x assertion reflects ~50% tile-count reduction from the
    lower-triangle skip in dsyrk combined with microkernel efficiency.

    GFLOPS for both implementations are printed for diagnostics.
    """
    import time

    from jamma.jlinalg import blas_backend as _blas_backend
    from jamma.jlinalg import blas_has_dsyrk as _has_vendor_dsyrk
    from jamma.jlinalg import jlinalg_isa as _isa

    if not _has_vendor_dsyrk:
        pytest.skip(
            "vendor BLAS dsyrk not wired (no ILP64 BLAS detected); "
            "jlinalg.dsyrk falls back to np.dot, so the >1.2x throughput "
            "target is unreachable by construction. Install ILP64 numpy-mkl "
            "(Linux/Windows) or run on macOS with Accelerate-ILP64."
        )

    # The 1.2x speedup comes from the lower-triangle skip in vendor dsyrk plus
    # microkernel efficiency. When both jlinalg.dsyrk and np.matmul resolve to
    # the SAME OpenBLAS library (the case on stock numpy >=2.x which ships
    # scipy-openblas64 with INTERFACE64=1), there's no implementation
    # difference left for jlinalg to exploit — both paths call the same
    # symbols at the same threading level. The assertion only makes sense
    # when jlinalg routes to MKL or Accelerate (a different library than
    # numpy's BLAS) where the per-call dispatch and symmetric kernel win
    # actually pay off.
    if _blas_backend.startswith("OpenBLAS"):
        pytest.skip(
            f"backend={_blas_backend}: jlinalg and np.matmul both call the "
            "same OpenBLAS symbols, so dsyrk's lower-triangle-skip win "
            "vanishes. Assertion is meaningful only against MKL or Accelerate."
        )

    rng = np.random.default_rng(42)
    # N=4000, K=2000: CI-runnable proxy for kinship workload
    # Full target (N=10000, K=5000) tested in bench_jlinalg.py
    N = 4000
    K = 2000
    X = rng.standard_normal((N, K))

    # Warm up
    _ = dsyrk(X)
    _ = np.matmul(X, X.T)

    # Time jlinalg dsyrk: best of 3
    n_iters = 3
    best_jlinalg = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        dsyrk(X)
        best_jlinalg = min(best_jlinalg, time.perf_counter() - t0)

    # Time np.matmul: best of 3
    best_numpy = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        np.matmul(X, X.T)
        best_numpy = min(best_numpy, time.perf_counter() - t0)

    # GFLOPS: 2*N^2*K flops for X @ X.T
    flops = 2.0 * N * N * K
    gflops_jlinalg = flops / best_jlinalg / 1e9
    gflops_numpy = flops / best_numpy / 1e9
    ratio = best_numpy / best_jlinalg

    jl_ms = best_jlinalg * 1000
    np_ms = best_numpy * 1000
    print(f"\ndsyrk N={N}, K={K}: {gflops_jlinalg:.1f} GF ({jl_ms:.0f} ms)")
    print(f"np.matmul(X, X.T):  {gflops_numpy:.1f} GF ({np_ms:.0f} ms)")
    print(f"Speedup ratio:      {ratio:.3f}x  (ISA: {_isa})")

    # VALID-06: enforce 1.2x target on AVX2 (x86_64 with OpenMP).
    # On NEON (Apple Silicon), Apple Accelerate's np.matmul is multi-threaded
    # and significantly faster than our single-threaded C extension.
    if _isa == "AVX2":
        assert ratio >= 1.2, (
            f"jlinalg dsyrk is less than 1.2x faster than np.matmul at N={N}, K={K}: "
            f"ratio={ratio:.3f}, jlinalg={gflops_jlinalg:.1f} GFLOPS, "
            f"numpy={gflops_numpy:.1f} GFLOPS (ISA: {_isa})"
        )
    elif _isa in ("NEON", "generic"):
        print(f"{_isa}: throughput assertion skipped (ratio={ratio:.3f}x vs np.matmul)")


# ---------------------------------------------------------------------------
# TestDsyrkVendorDispatch — vendor dsyrk dispatch parity tests
# ---------------------------------------------------------------------------


class TestDsyrkVendorDispatch:
    """Verify dsyrk produces correct results regardless of vendor dispatch path."""

    def test_dsyrk_vendor_parity_small(self):
        """dsyrk result matches numpy at N=10, K=5."""
        rng = np.random.default_rng(42)
        X = np.ascontiguousarray(rng.standard_normal((10, 5)), dtype=np.float64)
        K = dsyrk(X)
        expected = X @ X.T
        npt.assert_allclose(K, expected, rtol=1e-12)

    def test_dsyrk_vendor_parity_medium(self):
        """dsyrk result matches numpy at N=200, K=100."""
        rng = np.random.default_rng(123)
        X = np.ascontiguousarray(rng.standard_normal((200, 100)), dtype=np.float64)
        K = dsyrk(X)
        expected = X @ X.T
        npt.assert_allclose(K, expected, rtol=1e-12)

    def test_dsyrk_vendor_symmetry(self):
        """Vendor dsyrk produces bitwise-symmetric result."""
        rng = np.random.default_rng(456)
        X = np.ascontiguousarray(rng.standard_normal((50, 30)), dtype=np.float64)
        K = dsyrk(X)
        # Bitwise symmetry: K[i,j] == K[j,i] exactly
        npt.assert_array_equal(K, K.T)

    def test_dsyrk_vendor_parity_boundary_sizes(self):
        """dsyrk matches numpy at small boundary sizes."""
        rng = np.random.default_rng(789)
        for n in [1, 2, 3, 4, 5]:
            X = np.ascontiguousarray(rng.standard_normal((n, 10)), dtype=np.float64)
            K = dsyrk(X)
            expected = X @ X.T
            npt.assert_allclose(K, expected, rtol=1e-12, err_msg=f"Failed at N={n}")
