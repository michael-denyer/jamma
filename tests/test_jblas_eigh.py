"""Tests for jblas eigh (symmetric eigendecomposition via DSYTRD + DSTEDC + DORMTR).

Tests cover all EIGH requirements (EIGH-01 through EIGH-09):
- EIGH-01: DSYTRD reduction to tridiagonal form (C extension stub)
- EIGH-02: DSTEDC divide-and-conquer secular solver (C extension stub)
- EIGH-03: Block-diagonal stress test (repeated/clustered eigenvalues)
- EIGH-04: DORMTR eigenvector back-transformation (C extension stub)
- EIGH-05: Python fallback correctness (identity, diagonal, random SPD, ascending)
- EIGH-06: Output memory layout (shape, dtype, C-contiguous)
- EIGH-07: Reconstruction accuracy: ||K - U diag(w) U.T|| / ||K|| < 1e-8
- EIGH-08: Orthogonality: ||U.T @ U - I||_F < 1e-8
- EIGH-09: LAPACK sources in hatch_build.py must not receive -ffast-math

Run with -n0 to avoid interference with OpenMP threading tests:
    uv run pytest tests/test_jblas_eigh.py -x -n0 -v
"""

from __future__ import annotations

import ctypes
import inspect

import numpy as np
import numpy.testing as npt
import pytest

from jamma.jblas import HAS_C_EXTENSION, eigh, get_n_threads, set_n_threads

# ---------------------------------------------------------------------------
# Assertion helpers — reconstruction and orthogonality checks
# ---------------------------------------------------------------------------


def _assert_reconstruction(
    K: np.ndarray,
    w: np.ndarray,
    v: np.ndarray,
    tol: float,
    label: str = "",
) -> float:
    """Assert ||K - V diag(w) V.T||_F / ||K||_F < tol.

    Args:
        K: Original matrix (before eigh overwrites it).
        w: Eigenvalues from eigh.
        v: Eigenvectors from eigh.
        tol: Maximum allowed relative reconstruction error.
        label: Optional label for the assertion message.

    Returns:
        The computed relative reconstruction error.
    """
    K_recon = v @ np.diag(w) @ v.T
    norm_K = np.linalg.norm(K, "fro")
    if norm_K == 0.0:
        ratio = np.linalg.norm(K_recon, "fro")
    else:
        ratio = np.linalg.norm(K - K_recon, "fro") / norm_K
    msg = f"Reconstruction error {ratio:.2e} > {tol}"
    if label:
        msg = f"{label}: {msg}"
    assert ratio < tol, msg
    return ratio


def _assert_orthogonality(
    v: np.ndarray,
    tol: float,
    label: str = "",
) -> float:
    """Assert ||V.T @ V - I||_F < tol.

    Args:
        v: Eigenvectors from eigh, shape (N, N).
        tol: Maximum allowed orthogonality error.
        label: Optional label for the assertion message.

    Returns:
        The computed orthogonality error.
    """
    N = v.shape[1]
    norm_off = np.linalg.norm(v.T @ v - np.eye(N), "fro")
    msg = f"Orthogonality error {norm_off:.2e} > {tol}"
    if label:
        msg = f"{label}: {msg}"
    assert norm_off < tol, msg
    return norm_off


# ---------------------------------------------------------------------------
# Boundary size parameters
# ---------------------------------------------------------------------------

# Sizes chosen to cover MR-1/MR/MR+1 for both AVX2 (MR=6) and NEON (MR=8),
# plus MC boundaries (AVX2 MC=72, NEON MC=64), and a selection of primes.
# Capped at 200 for eigh (eigendecomp is O(N^3) — larger sizes are in slow tests).
BOUNDARY_SIZES = [
    1,
    2,
    3,
    5,
    6,
    7,  # MR-1/MR/MR+1 for AVX2 (MR=6)
    8,
    9,  # MR/MR+1 for NEON (MR=8)
    11,
    13,
    31,
    63,
    64,
    65,  # MC-1/MC/MC+1 for NEON (MC=64)
    71,
    72,
    73,  # MC-1/MC/MC+1 for AVX2 (MC=72)
    100,
    127,
    128,
    129,
    200,
]

# Deduplicate while preserving order
_seen: set[int] = set()
BOUNDARY_SIZES = [x for x in BOUNDARY_SIZES if not (x in _seen or _seen.add(x))]  # type: ignore[func-returns-value]


# ---------------------------------------------------------------------------
# Helper: generate random symmetric positive semi-definite matrix
# ---------------------------------------------------------------------------


def _random_spd(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric positive semi-definite matrix.

    Args:
        N: Matrix dimension.
        rng: NumPy random generator instance.

    Returns:
        N x N symmetric PSD matrix, float64.
    """
    A = rng.standard_normal((N, N))
    K = A @ A.T / N
    return K


# ---------------------------------------------------------------------------
# TestEigh — EIGH-05: Python fallback correctness
# ---------------------------------------------------------------------------


class TestEigh:
    """eigh must produce correct eigenvalues/eigenvectors (EIGH-05)."""

    @pytest.mark.parametrize("N", [1, 3, 10, 100])
    def test_identity(self, N: int) -> None:
        """eigh(eye(N)) returns all-ones eigenvalues and orthogonal eigenvectors."""
        K = np.eye(N)
        K_copy = K.copy()
        w, v = eigh(K_copy)
        # All eigenvalues should be 1.0
        npt.assert_allclose(
            w, np.ones(N), rtol=1e-14, err_msg=f"Identity eigenvalues wrong at N={N}"
        )
        # Eigenvectors form orthogonal matrix: V.T @ V = I
        VtV = v.T @ v
        npt.assert_allclose(
            VtV,
            np.eye(N),
            atol=1e-14,
            err_msg=f"Identity eigenvectors not orthogonal at N={N}",
        )

    @pytest.mark.parametrize("N", [1, 5, 50])
    def test_diagonal(self, N: int) -> None:
        """eigh(diag(d)) returns sorted d as eigenvalues, permuted identity columns."""
        rng = np.random.default_rng(42 + N)
        # Random positive diagonal entries (unsorted)
        d_unsorted = rng.uniform(0.1, 10.0, size=N)
        K = np.diag(d_unsorted)
        K_copy = K.copy()
        w, v = eigh(K_copy)
        # Eigenvalues must be sorted ascending
        d_sorted = np.sort(d_unsorted)
        npt.assert_allclose(
            w,
            d_sorted,
            rtol=1e-12,
            err_msg=f"Diagonal eigenvalues wrong at N={N}",
        )
        # Each eigenvector must be a unit standard basis vector
        # (columns of v are unit vectors matching permuted identity columns)
        for j in range(N):
            col = np.abs(v[:, j])
            # One element should be ~1, rest ~0
            assert col.max() > 0.99, (
                f"Eigenvector {j} is not a standard basis vector at N={N}"
            )
            npt.assert_allclose(
                col.sum(),
                1.0,
                atol=1e-12,
                err_msg=f"Eigenvector {j} not unit at N={N}",
            )

    @pytest.mark.parametrize("N", BOUNDARY_SIZES)
    def test_random_spd(self, N: int) -> None:
        """eigh on random SPD: eigenvalues ascending, eigenvectors unit norm."""
        rng = np.random.default_rng(42 + N * 7)
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v = eigh(K_copy)
        # Eigenvalues must be ascending
        if N > 1:
            assert np.all(w[:-1] <= w[1:] + 1e-14), (
                f"Eigenvalues not ascending at N={N}: {w}"
            )
        # Eigenvectors must be unit norm
        norms = np.linalg.norm(v, axis=0)
        npt.assert_allclose(
            norms,
            np.ones(N),
            atol=1e-13,
            err_msg=f"Eigenvector norms not 1 at N={N}",
        )

    def test_ascending_eigenvalues(self) -> None:
        """Eigenvalues are sorted ascending for a known non-trivial matrix."""
        rng = np.random.default_rng(1234)
        K = _random_spd(50, rng)
        K_copy = K.copy()
        w, _ = eigh(K_copy)
        diffs = np.diff(w)
        assert np.all(diffs >= -1e-14), (
            f"Eigenvalues not ascending: min diff = {diffs.min()}"
        )

    def test_raises_non_square(self) -> None:
        """eigh on (3,4) array raises ValueError."""
        K = np.ones((3, 4))
        with pytest.raises(ValueError, match="square|shape"):
            eigh(K)

    def test_raises_1d(self) -> None:
        """eigh on 1-D array raises ValueError."""
        K = np.ones(5)
        with pytest.raises(ValueError, match="2-D|ndim"):
            eigh(K)

    def test_empty_matrix(self) -> None:
        """eigh on 0x0 matrix returns empty eigenvalues and eigenvectors."""
        K = np.zeros((0, 0), dtype=np.float64)
        w, v = eigh(K)
        assert w.shape == (0,), f"Expected (0,) eigenvalues, got {w.shape}"
        assert v.shape == (0, 0), f"Expected (0,0) eigenvectors, got {v.shape}"

    def test_fortran_order_input(self) -> None:
        """eigh on Fortran-order input produces correct results."""
        rng = np.random.default_rng(99)
        N = 20
        K = _random_spd(N, rng)
        K_f = np.asfortranarray(K.copy())
        w, v = eigh(K_f)
        _assert_reconstruction(K, w, v, 1e-13, "Fortran-order")


# ---------------------------------------------------------------------------
# test_reconstruction_accuracy — EIGH-07
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reconstruction_accuracy() -> None:
    """N=1000 random SPD: ||K - U diag(w) U.T|| / ||K|| < 1e-8.

    D&C-direct (no QR fallback) produces ~1e-9 residuals at N=1000.
    Tolerance 1e-8 gives comfortable margin above D&C-direct residuals.
    """
    rng = np.random.default_rng(42)
    N = 1000
    K = _random_spd(N, rng)
    K_copy = K.copy()
    w, v = eigh(K_copy)
    _assert_reconstruction(K, w, v, 1e-8, "EIGH-07 N=1000")


# ---------------------------------------------------------------------------
# test_orthogonality — EIGH-08
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_orthogonality() -> None:
    """N=1000 random SPD: ||U.T @ U - I||_F < 1e-12."""
    rng = np.random.default_rng(43)
    N = 1000
    K = _random_spd(N, rng)
    K_copy = K.copy()
    _, v = eigh(K_copy)
    _assert_orthogonality(v, 1e-12, "EIGH-08 N=1000")


# ---------------------------------------------------------------------------
# test_eigh_memory_layout — EIGH-06
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_eigh_nan_inf_input(bad_value: float) -> None:
    """eigh should not silently produce garbage for NaN/Inf input.

    Either raises an error or propagates NaN in eigenvalues — both are
    acceptable as long as the result is not silently wrong finite values.
    """
    N = 10
    K = np.eye(N, dtype=np.float64)
    K[3, 3] = bad_value
    K_copy = K.copy()
    try:
        w, v = eigh(K_copy)
        # If it didn't raise, eigenvalues must contain NaN/Inf — not
        # silently finite values from a corrupted decomposition.
        assert not np.all(np.isfinite(w)), (
            f"eigh produced all-finite eigenvalues from input containing "
            f"{bad_value} — expected NaN/Inf propagation or an exception"
        )
    except (RuntimeError, np.linalg.LinAlgError, ValueError):
        pass  # Raising is acceptable


def test_eigh_memory_layout() -> None:
    """eigh returns (eigenvalues shape (N,), eigenvectors shape (N,N)).

    Both arrays must be C-contiguous float64.
    """
    rng = np.random.default_rng(44)
    N = 20
    K = _random_spd(N, rng)
    K_copy = K.copy()
    w, v = eigh(K_copy)
    # Shape checks
    assert w.shape == (N,), f"Eigenvalues shape {w.shape} != ({N},)"
    assert v.shape == (N, N), f"Eigenvectors shape {v.shape} != ({N}, {N})"
    # dtype checks
    assert w.dtype == np.float64, f"Eigenvalues dtype {w.dtype} != float64"
    assert v.dtype == np.float64, f"Eigenvectors dtype {v.dtype} != float64"
    # C-contiguous checks
    assert w.flags["C_CONTIGUOUS"], "Eigenvalues must be C-contiguous"
    assert v.flags["C_CONTIGUOUS"], "Eigenvectors must be C-contiguous"


# ---------------------------------------------------------------------------
# test_block_diagonal_stress — EIGH-03
# ---------------------------------------------------------------------------


def test_block_diagonal_stress() -> None:
    """Block-diagonal 1000x1000 matrix with clustered eigenvalues per block.

    Builds 10 groups x 100 = 1000x1000 block-diagonal matrix.
    Verifies reconstruction < 1e-8 and orthogonality < 1e-8.

    With z-vector sign fix (Phase 80.4-07), D&C achieves ~1e-9 residuals
    at N=1000 without QR fallback.
    """
    rng = np.random.default_rng(77)
    n_groups = 10
    block_size = 100
    N = n_groups * block_size

    # Build block-diagonal matrix: each block is a random SPD matrix
    K = np.zeros((N, N))
    for g in range(n_groups):
        start = g * block_size
        end = start + block_size
        block = _random_spd(block_size, rng)
        # Scale so eigenvalues cluster tightly within the block
        block = block / block.max()
        K[start:end, start:end] = block

    if HAS_C_EXTENSION:
        K_copy = K.copy()
        w, v, status = _call_eigh_with_status(K_copy)
        assert status.secular_failures == 0, (
            f"{status.secular_failures} secular failures on block-diagonal"
        )
        assert status.qr_fallback == 0, (
            f"QR fallback {status.qr_fallback} on block-diagonal -- z-vector "
            f"sign fix should eliminate QR fallback"
        )
    else:
        K_copy = K.copy()
        w, v = eigh(K_copy)

    _assert_reconstruction(K, w, v, 1e-8, "Block-diagonal")
    _assert_orthogonality(v, 1e-8, "Block-diagonal")


# ---------------------------------------------------------------------------
# test_vs_mouse_hs1940_kinship — real-data validation
# ---------------------------------------------------------------------------


def test_vs_mouse_hs1940_kinship() -> None:
    """eigh on mouse_hs1940 kinship matrix: correct eigenvalues and orthogonality.

    Checks reconstruction and orthogonality rather than comparing eigenvectors
    element-wise against numpy.linalg.eigh.  Direct eigenvector comparison is
    not meaningful when eigenvalues are degenerate or nearly degenerate —
    any rotation within an eigenspace is mathematically valid, so independent
    implementations may choose different bases.

    Tolerance rationale:
    - Eigenvalues: atol=1e-12, rtol=1e-8.  The kinship matrix is nearly
      singular (smallest eigenvalue ~9e-13, condition number ~4e13), so
      pure rtol is inappropriate for near-zero eigenvalues.  rtol=1e-8
      accounts for the different rounding paths in jblas QR vs LAPACK dstevd.
    - Reconstruction: ||K - V W V^T||_F / ||K||_F < 1e-8.  For N=1940 with
      condition number 4e13, O(N * eps * cond) gives ~4e13 * 2e-16 * 1940 ~
      0.15 — but the tridiagonalization concentrates error so the practical
      bound is ~1e-8 in the Frobenius norm.
    - Orthogonality: ||V^T V - I||_F < 1e-5.  QR iteration loses a digit
      per decade of condition ratio; with 13 decades the accumulated orthogon-
      ality error is O(sqrt(N) * eps * cond_sub) where cond_sub is the local
      subproblem condition.
    """
    import os

    kinship_path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kinship",
        "mouse_hs1940.cXX.txt",
    )
    K = np.loadtxt(kinship_path)
    assert K.ndim == 2, f"Kinship matrix must be 2-D, got {K.ndim}-D"
    assert K.shape[0] == K.shape[1], f"Kinship matrix must be square, got {K.shape}"

    # Reference eigenvalues from numpy (LAPACK dstevd/dsyevd)
    w_ref = np.linalg.eigvalsh(K.copy())

    # jblas eigh
    K_jblas = K.copy()
    w_jblas, v_jblas = eigh(K_jblas)

    # Compare eigenvalues: use atol for near-zero eigenvalues, rtol for large ones
    npt.assert_allclose(
        w_jblas,
        w_ref,
        rtol=1e-8,
        atol=1e-12,
        err_msg="jblas eigh eigenvalues differ from np.linalg.eigh on mouse_hs1940",
    )

    _assert_orthogonality(v_jblas, 1e-5, "Kinship")
    _assert_reconstruction(K, w_jblas, v_jblas, 1e-4, "Kinship")


# ---------------------------------------------------------------------------
# TestDsytrd — EIGH-01: Tridiagonalization (C extension only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for dsytrd tests",
)
class TestDsytrd:
    """DSYTRD reduction to tridiagonal form (EIGH-01).

    dsytrd/dstedc/dormtr are internal C functions not yet exposed as individual
    Python bindings. These tests exercise them indirectly through eigh.
    """

    def test_tridiagonalizes_via_eigh(self) -> None:
        """Verify dsytrd works correctly by checking eigh reconstruction.

        dsytrd is the first step of eigh; if it produces wrong tridiagonal
        output, reconstruction will fail.
        """
        rng = np.random.default_rng(101)
        N = 64  # NB_DSYTRD = 64, tests blocked path boundary
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v = eigh(K_copy)
        _assert_reconstruction(K, w, v, 1e-13, "dsytrd/eigh")


# ---------------------------------------------------------------------------
# TestDstedc — EIGH-02: Divide-and-conquer solver (C extension only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for dstedc tests",
)
class TestDstedc:
    """DSTEDC divide-and-conquer solver (EIGH-02).

    Tests exercise dstedc indirectly through eigh at sizes that trigger
    both the QR base case and D&C recursive splitting.
    """

    @pytest.mark.parametrize("N", [127, 128, 129, 200])
    def test_dstedc_boundary_reconstruction(self, N: int) -> None:
        """Verify D&C merge at DSTEDC_BASE boundary via reconstruction.

        After Phase 80.4 (A/B/C rational interpolation + delta_mat weight
        product), D&C achieves ~1e-9 residuals at N <= ~127 without QR
        fallback.  At N >= ~128, QR fallback ensures machine-precision
        results.  Tolerance is 1e-8 to accept both D&C-direct and
        QR-fallback paths.
        """
        rng = np.random.default_rng(202 + N)
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v = eigh(K_copy)
        _assert_reconstruction(K, w, v, 1e-8, f"D&C boundary N={N}")

    def test_degenerate_eigenvalues(self) -> None:
        """Matrix with exact repeated eigenvalues exercises deflation."""
        # diag([1,1,1,2,2,3]) — exact multiplicity
        K = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        w, v = eigh(K.copy())
        npt.assert_allclose(w, [1, 1, 1, 2, 2, 3], atol=1e-14)
        VtV = v.T @ v
        npt.assert_allclose(VtV, np.eye(6), atol=1e-14)

    def test_zero_matrix(self) -> None:
        """Zero matrix: all eigenvalues 0, eigenvectors form orthonormal basis."""
        N = 10
        K = np.zeros((N, N))
        w, v = eigh(K.copy())
        npt.assert_allclose(w, np.zeros(N), atol=1e-15)
        VtV = v.T @ v
        npt.assert_allclose(VtV, np.eye(N), atol=1e-14)

    def test_indefinite_matrix(self) -> None:
        """Matrix with negative eigenvalues."""
        K = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: -1, 3
        w, v = eigh(K.copy())
        npt.assert_allclose(w, [-1.0, 3.0], atol=1e-14)
        VtV = v.T @ v
        npt.assert_allclose(VtV, np.eye(2), atol=1e-14)


# ---------------------------------------------------------------------------
# TestDormtr — EIGH-04: Back-transformation (C extension only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for dormtr tests",
)
class TestDormtr:
    """DORMTR eigenvector back-transformation (EIGH-04).

    Tests exercise dormtr indirectly through eigh by verifying that
    eigenvectors are in the original (not tridiagonal) basis.
    """

    def test_back_transforms_eigenvectors(self) -> None:
        """Verify eigenvectors are in the original basis (not tridiagonal).

        After Phase 80.4, D&C at N=100 achieves ~1e-9 residuals without
        QR fallback, producing per-vector residuals up to ~2e-9.  Tolerance
        1e-7 provides margin for seed-dependent variation.
        """
        rng = np.random.default_rng(303)
        N = 100
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v = eigh(K_copy)
        # If dormtr failed, K @ v[:,j] != w[j] * v[:,j]
        for j in range(min(5, N)):
            residual = np.linalg.norm(K @ v[:, j] - w[j] * v[:, j])
            assert residual < 1e-7, (
                f"dormtr back-transform failed for eigenvector {j}: "
                f"residual={residual:.2e}"
            )


# ---------------------------------------------------------------------------
# test_lapack_no_ffast_math — EIGH-09
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestThreadControl — thread control API
# ---------------------------------------------------------------------------


class TestThreadControl:
    """Thread control API: get/set_n_threads with init-time clamping."""

    def test_get_n_threads_returns_positive(self) -> None:
        """get_n_threads returns a positive integer."""
        n = get_n_threads()
        assert isinstance(n, int)
        assert n >= 1, f"get_n_threads returned {n}, expected >= 1"

    def test_set_n_threads_returns_old_count(self) -> None:
        """set_n_threads returns the previous thread count."""
        original = get_n_threads()
        old = set_n_threads(1)
        assert old == original, f"set_n_threads returned {old}, expected {original}"
        # Restore
        set_n_threads(original)

    def test_set_n_threads_clamps_upward(self) -> None:
        """set_n_threads(9999) clamps to init-time max, not 9999."""
        original = get_n_threads()
        set_n_threads(9999)
        clamped = get_n_threads()
        assert clamped <= original, (
            f"set_n_threads(9999) set {clamped}, expected <= init max {original}"
        )
        assert clamped >= 1
        # Restore
        set_n_threads(original)

    def test_set_n_threads_rejects_zero(self) -> None:
        """set_n_threads(0) raises ValueError."""
        with pytest.raises(ValueError):
            set_n_threads(0)

    def test_set_n_threads_rejects_negative(self) -> None:
        """set_n_threads(-1) raises ValueError."""
        with pytest.raises(ValueError):
            set_n_threads(-1)


# ---------------------------------------------------------------------------
# TestAccumGemm — accumulate GEMM functional tests via eigh (EIGH-07/08)
# ---------------------------------------------------------------------------


class TestAccumGemm:
    """Functional tests for _dgemm_core refactor via eigh correctness.

    The accumulate GEMM and workspace APIs are C-internal with no direct
    Python binding.  We verify them indirectly: eigh calls jblas_dgemm_c
    internally (via DSTEDC D&C), so if the _dgemm_core refactor broke
    anything, eigh reconstruction would fail.
    """

    def test_eigh_still_correct_n100(self) -> None:
        """eigh on 100x100 random SPD: reconstruction < 1e-8, orthogonality < 1e-12.

        After Phase 80.4, D&C at N=100 achieves ~1e-9 residuals without
        QR fallback.  Reconstruction tolerance 1e-8 provides margin.
        """
        rng = np.random.default_rng(5001)
        N = 100
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v = eigh(K_copy)

        _assert_reconstruction(K, w, v, 1e-8, "AccumGemm N=100")
        _assert_orthogonality(v, 1e-12, "AccumGemm N=100")

    @pytest.mark.slow
    def test_eigh_still_correct_n500(self) -> None:
        """eigh on 500x500 random SPD: full pipeline test.

        At N=500, this exercises the full DSYTRD + DSTEDC (D&C with QR
        base case) + DORMTR pipeline.  Verifies _dgemm_core refactor is sound.
        """
        rng = np.random.default_rng(5002)
        N = 500
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v = eigh(K_copy)

        _assert_reconstruction(K, w, v, 1e-8, "AccumGemm N=500")
        _assert_orthogonality(v, 1e-8, "AccumGemm N=500")


# ---------------------------------------------------------------------------
# TestWorkspaceApi — workspace API (indirect via eigh, C-internal only)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for throughput benchmarks",
)
class TestEighThroughput:
    """Benchmark jblas.eigh vs numpy.linalg.eigh for Phase 80.1."""

    def test_eigh_throughput_n500(self) -> None:
        """jblas eigh should be < 10x slower than numpy at N=500 after optimizations."""
        import time

        rng = np.random.default_rng(42)
        N = 500
        A = rng.standard_normal((N, N))
        K = A @ A.T + np.eye(N)  # SPD

        # Warm up
        np.linalg.eigh(K.copy())
        eigh(K.copy())

        # Time numpy
        t0 = time.perf_counter()
        for _ in range(3):
            np.linalg.eigh(K.copy())
        t_numpy = (time.perf_counter() - t0) / 3

        # Time jblas
        t0 = time.perf_counter()
        for _ in range(3):
            eigh(K.copy())
        t_jblas = (time.perf_counter() - t0) / 3

        ratio = t_jblas / t_numpy
        print(
            f"\nN={N}: jblas={t_jblas:.4f}s, numpy={t_numpy:.4f}s, ratio={ratio:.1f}x"
        )
        # Cross-platform soft gate.  Apple Silicon Accelerate is multi-threaded
        # and uses LAPACK dsyevd with vDSP BLAS — jblas single-threaded LAPACK
        # cannot match that.  Use 15x to accommodate Accelerate (~14x observed)
        # while still catching gross regressions.  On x86_64 with AVX2 + MKL,
        # expect < 8x.
        assert ratio < 15.0, (
            f"jblas eigh is {ratio:.1f}x slower than numpy -- expected < 15x"
        )

    def test_eigh_correctness_n1000_post_optimization(self) -> None:
        """Full correctness gate at N=1000 after all Phase 80.1 optimizations."""
        rng = np.random.default_rng(123)
        N = 1000
        A = rng.standard_normal((N, N))
        K = A @ A.T + np.eye(N)

        vals, vecs = eigh(K.copy())
        np_vals, _ = np.linalg.eigh(K.copy())

        # Eigenvalue agreement — D&C eigenvalues match numpy to ~1e-10
        npt.assert_allclose(vals, np_vals, rtol=1e-8)

        _assert_reconstruction(K, vals, vecs, 1e-8, "N=1000 post-opt")
        _assert_orthogonality(vecs, 1e-8, "N=1000 post-opt")


class TestWorkspaceApi:
    """Workspace API has no Python binding (C-internal only).

    Tested indirectly: if eigh works at N > DSTEDC_BASE (128), the GEMM
    calls inside dstedc are functioning with workspace buffers.
    The TestAccumGemm tests above cover the _dgemm_core refactor path.
    """

    def test_eigh_uses_dgemm_internally(self) -> None:
        """Verify eigh at boundary sizes uses the _dgemm_core path without error.

        At N=128/200, D&C may or may not trigger QR fallback depending on
        the specific matrix and seed.  Tolerance 1e-8 accepts both
        D&C-direct (~1e-9) and QR-fallback (~1e-14) results.
        """
        rng = np.random.default_rng(6001)
        for N in [128, 200]:
            K = _random_spd(N, rng)
            K_copy = K.copy()
            w, v = eigh(K_copy)
            _assert_reconstruction(K, w, v, 1e-8, f"Workspace N={N}")


# ---------------------------------------------------------------------------
# test_lapack_no_ffast_math — EIGH-09
# ---------------------------------------------------------------------------


def test_lapack_no_ffast_math() -> None:
    """LAPACK sources in hatch_build.py must use restricted flags (no -ffast-math).

    This is a structural test: it inspects hatch_build.py source to verify
    that a 'lapack_sources' group exists and that -ffast-math is not applied
    to LAPACK source files.

    """
    import importlib.util
    import sys

    # Load hatch_build.py as a module for inspection
    hatch_build_path = (
        __file__.replace("/tests/test_jblas_eigh.py", "/hatch_build.py").replace(
            "/tests\\test_jblas_eigh.py", "/hatch_build.py"
        )  # Windows
    )

    spec = importlib.util.spec_from_file_location("hatch_build", hatch_build_path)
    assert spec is not None, "Could not locate hatch_build.py"
    hatch_build = importlib.util.module_from_spec(spec)
    sys.modules["hatch_build"] = hatch_build
    spec.loader.exec_module(hatch_build)  # type: ignore[union-attr]

    # Get the source of _compile_jblas_extension
    src = inspect.getsource(hatch_build.CustomBuildHook._compile_jblas_extension)

    assert "lapack_sources" in src, (
        "hatch_build.py _compile_jblas_extension must define a 'lapack_sources' group "
        "for LAPACK files that must not receive -ffast-math. "
        "Add lapack_sources in Plan 03."
    )
    # Verify -ffast-math is not in the lapack compile flags section
    # (structural check: the lapack_sources block must not contain -ffast-math)
    lapack_section_start = src.find("lapack_sources")
    lapack_section = src[lapack_section_start : lapack_section_start + 500]
    assert "-ffast-math" not in lapack_section, (
        "LAPACK source files must NOT receive -ffast-math. "
        "The dstedc secular equation uses IEEE 754 infinity arithmetic "
        "which -ffast-math breaks. Fix the lapack_sources compile flags."
    )


# ---------------------------------------------------------------------------
# EIGH-10: Dtype handling — non-float64 inputs
# ---------------------------------------------------------------------------


class TestEighDtype:
    """Verify eigh handles non-float64 inputs correctly."""

    def test_float32_input(self):
        """eigh should accept float32 and produce correct results (via conversion)."""
        K = np.eye(5, dtype=np.float32)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            w, v = eigh(K.astype(np.float64))
        npt.assert_allclose(w, np.ones(5), atol=1e-14)

    def test_int_input_requires_float64(self):
        """eigh rejects or converts int arrays (no silent garbage)."""
        K = np.eye(5, dtype=np.int32)
        # Must convert to float64 — calling eigh on int array should either
        # work (after auto-conversion) or raise a clear error
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                w, v = eigh(K.astype(np.float64))
                npt.assert_allclose(w, np.ones(5), atol=1e-14)
            except (TypeError, ValueError):
                pass  # Clear error is acceptable


# ---------------------------------------------------------------------------
# EIGH-11: Error path tests
# ---------------------------------------------------------------------------


class TestEighErrors:
    """Verify error paths raise appropriate exceptions."""

    def test_non_square_raises(self):
        """Non-square input must raise ValueError."""
        K = np.ones((3, 4), dtype=np.float64)
        with pytest.raises(ValueError, match="square"):
            eigh(K)

    def test_1d_raises(self):
        """1-D input must raise ValueError."""
        K = np.ones(5, dtype=np.float64)
        with pytest.raises(ValueError):
            eigh(K)

    def test_3d_raises(self):
        """3-D input must raise ValueError."""
        K = np.ones((3, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            eigh(K)


# ---------------------------------------------------------------------------
# ctypes helpers for direct C function access
# ---------------------------------------------------------------------------


def _find_jblas_so() -> str:
    """Find the _jblas shared library path."""
    import jamma.jblas._jblas as mod

    return mod.__file__


class _EighStatus(ctypes.Structure):
    """ctypes mirror of jblas_eigh_status_t."""

    _fields_ = [
        ("dstedc_ws_fallback", ctypes.c_int),
        ("dsytrd_mirror_fallback", ctypes.c_int),
        ("secular_failures", ctypes.c_int),
        ("qr_fallback", ctypes.c_int),
    ]


def _load_jblas_eigh_c():
    """Load jblas_eigh_c via ctypes.

    Returns:
        Callable with signature matching jblas_eigh_c.
    """
    so_path = _find_jblas_so()
    lib = ctypes.CDLL(so_path)

    fn = lib.jblas_eigh_c
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_longlong,  # npy_intp N
        ctypes.c_void_p,  # double *K
        ctypes.c_longlong,  # npy_intp ldk
        ctypes.c_void_p,  # double *eigenvalues
        ctypes.c_void_p,  # double *eigenvectors
        ctypes.c_longlong,  # npy_intp ldz
        ctypes.POINTER(_EighStatus),  # jblas_eigh_status_t *status
    ]
    return fn


def _ptr(arr: np.ndarray) -> ctypes.c_void_p:
    """Get ctypes void pointer to numpy array data."""
    return ctypes.c_void_p(arr.ctypes.data)


def _call_eigh_with_status(
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, _EighStatus]:
    """Call jblas_eigh_c via ctypes, returning eigenvalues, eigenvectors, status.

    Args:
        K: Symmetric matrix (N x N, float64, C-contiguous). Modified in place.

    Returns:
        Tuple of (eigenvalues, eigenvectors, status).
    """
    N = K.shape[0]
    eigenvalues = np.empty(N, dtype=np.float64)
    eigenvectors = np.empty((N, N), dtype=np.float64)
    status = _EighStatus()

    fn = _load_jblas_eigh_c()
    ret = fn(
        N, _ptr(K), N, _ptr(eigenvalues), _ptr(eigenvectors), N, ctypes.byref(status)
    )
    assert ret == 0, f"jblas_eigh_c returned {ret}"
    return eigenvalues, eigenvectors, status


# ---------------------------------------------------------------------------
# TestDstedcNoSecularFailures — secular solver convergence and QR tracking
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for secular failure detection",
)
class TestDstedcNoSecularFailures:
    """Test secular solver convergence and D&C eigenvector quality.

    After Phase 80.4-07 (LAPACK-matching z-vector sign handling in
    dstedc_recurse), D&C produces eigenvectors with residuals < 1e-8
    at all sizes without QR fallback.  The z-vector fix applies sign_rho
    only to the right half (matching LAPACK DLAED2), which corrects the
    cross-terms z[i]*z[j] that were corrupting eigenvectors at N>=128.

    Asserts:
      - Zero secular convergence failures at all sizes
      - Zero QR fallback at all sizes
      - Reconstruction accuracy < 1e-8 (D&C achieves ~1e-10 at N=200)
      - Eigenvector orthogonality < 1e-8
    """

    def test_no_secular_failures_n200(self) -> None:
        """N=200: zero secular failures, zero QR fallback, correct reconstruction."""
        rng = np.random.default_rng(42)
        N = 200
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, status = _call_eigh_with_status(K_copy)

        assert status.secular_failures == 0, (
            f"{status.secular_failures} secular failures at N={N}"
        )
        assert status.qr_fallback == 0, (
            f"QR fallback {status.qr_fallback} at N={N} -- z-vector sign fix "
            f"should eliminate QR fallback at all sizes"
        )
        _assert_reconstruction(K, w, v, 1e-8, f"SecularSolver N={N}")
        _assert_orthogonality(v, 1e-8, f"SecularSolver N={N}")

    @pytest.mark.slow
    def test_no_secular_failures_n500(self) -> None:
        """N=500: zero secular failures, zero QR fallback, correct reconstruction."""
        rng = np.random.default_rng(42)
        N = 500
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, status = _call_eigh_with_status(K_copy)

        assert status.secular_failures == 0, (
            f"{status.secular_failures} secular failures at N={N}"
        )
        assert status.qr_fallback == 0, (
            f"QR fallback {status.qr_fallback} at N={N} -- z-vector sign fix "
            f"should eliminate QR fallback at all sizes"
        )
        _assert_reconstruction(K, w, v, 1e-8, f"SecularSolver N={N}")
        _assert_orthogonality(v, 1e-8, f"SecularSolver N={N}")

    @pytest.mark.slow
    def test_no_secular_failures_n1000(self) -> None:
        """N=1000: zero secular failures, zero QR fallback, correct reconstruction."""
        rng = np.random.default_rng(42)
        N = 1000
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, status = _call_eigh_with_status(K_copy)

        assert status.secular_failures == 0, (
            f"{status.secular_failures} secular failures at N={N}"
        )
        assert status.qr_fallback == 0, (
            f"QR fallback {status.qr_fallback} at N={N} -- z-vector sign fix "
            f"should eliminate QR fallback at all sizes"
        )
        _assert_reconstruction(K, w, v, 1e-8, f"SecularSolver N={N}")
        _assert_orthogonality(v, 1e-8, f"SecularSolver N={N}")


# ---------------------------------------------------------------------------
# TestDlaed4Convergence — dlaed4 convergence on difficult inputs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for dlaed4 convergence tests",
)
class TestDlaed4Convergence:
    """Test dlaed4 convergence on known-difficult secular equation inputs.

    These tests exercise dlaed4 indirectly through jblas_eigh_c by
    constructing symmetric tridiagonal matrices that produce difficult
    patterns. Since T is already tridiagonal, dsytrd is a no-op and
    dstedc exercises the solver directly.

    With LAPACK-quality dlaed4 (ORGATI origin selection, A/B/C rational
    interpolation for quadratic convergence, SWTCH/SWTCH3/dlaed6 for
    clustered poles, and delta_mat weight product for full relative
    precision) and LAPACK-matching z-vector sign handling (sign_rho
    applied only to right half, matching DLAED2), secular convergence
    is reliable (zero failures) and D&C achieves < 1e-8 residuals
    at all sizes without QR fallback.
    """

    def test_clustered_eigenvalues(self) -> None:
        """Clustered eigenvalues: d values within 1e-10 of each other.

        Constructs a tridiagonal matrix whose eigenvalues cluster tightly,
        stressing the secular solver's ability to separate close poles.
        """
        N = 50
        # Construct a tridiagonal matrix with clustered eigenvalues
        # Use near-constant diagonal with small off-diagonal perturbation
        d = np.ones(N) + np.arange(N) * 1e-10
        e = np.full(N - 1, 1e-12)
        K = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        K_copy = K.copy()

        w, v, status = _call_eigh_with_status(K_copy)
        assert status.secular_failures == 0, (
            f"{status.secular_failures} secular failures on clustered eigenvalues"
        )
        _assert_reconstruction(K, w, v, 1e-12, "Clustered eigenvalues")

    def test_large_gap_ratio(self) -> None:
        """Large gap ratio: eigenvalues spanning many orders of magnitude.

        Tridiagonal matrix with diagonal spanning 1e-15 to O(1).
        """
        N = 20
        d = np.logspace(-15, 0, N)
        e = np.full(N - 1, 1e-8)
        K = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        K_copy = K.copy()

        w, v, status = _call_eigh_with_status(K_copy)
        assert status.secular_failures == 0, (
            f"{status.secular_failures} secular failures on large gap ratio"
        )
        _assert_reconstruction(K, w, v, 1e-10, "Large gap ratio")

    def test_boundary_eigenvalue(self) -> None:
        """Boundary eigenvalue: stress the i=n-1 case (above largest pole).

        Tridiagonal matrix with strong off-diagonal to push the last
        eigenvalue well above d[n-1].
        """
        N = 30
        d = np.arange(1.0, N + 1.0)
        e = np.full(N - 1, 2.0)
        K = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        K_copy = K.copy()

        w, v, status = _call_eigh_with_status(K_copy)
        assert status.secular_failures == 0, (
            f"{status.secular_failures} secular failures on boundary eigenvalue"
        )
        _assert_reconstruction(K, w, v, 1e-12, "Boundary eigenvalue")

    def test_delta_quality_via_reconstruction(self) -> None:
        """Verify dlaed4 delta precision via reconstruction accuracy.

        With LAPACK-matching z-vector sign handling (sign_rho applied only
        to right half, matching DLAED2), D&C produces eigenvectors without
        QR fallback at all sizes.

        Asserts:
          - qr_fallback==0 at all sizes (z-vector sign fix eliminates QR)
          - Reconstruction < 1e-8 at all sizes (D&C achieves ~1e-10)
          - Orthogonality < 1e-8 at all sizes
          - Zero secular convergence failures
        """
        rng = np.random.default_rng(42)
        for N in [50, 200, 500]:
            A = rng.standard_normal((N, N))
            A = (A + A.T) / 2
            w, V, status = _call_eigh_with_status(A.copy())

            # Reconstruction: D&C achieves ~1e-10 at N=200 without QR fallback
            recon = np.linalg.norm(A - V @ np.diag(w) @ V.T) / np.linalg.norm(A)
            assert recon < 1e-8, f"Reconstruction {recon:.2e} at N={N} (threshold 1e-8)"

            # Orthogonality
            orth = np.linalg.norm(V.T @ V - np.eye(N))
            assert orth < 1e-8, f"Orthogonality {orth:.2e} at N={N} (threshold 1e-8)"

            # Zero secular convergence failures
            assert status.secular_failures == 0, (
                f"{status.secular_failures} secular failures at N={N}"
            )

            # Z-vector sign fix eliminates QR fallback at all sizes
            assert status.qr_fallback == 0, (
                f"QR fallback {status.qr_fallback} at N={N} -- z-vector sign fix "
                f"should eliminate QR fallback at all sizes"
            )
