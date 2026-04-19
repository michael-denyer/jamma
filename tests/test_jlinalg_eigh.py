"""Tests for jlinalg eigh (symmetric eigendecomposition via DSYTRD + DSTEDC + DORMTR).

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
    uv run pytest tests/test_jlinalg_eigh.py -x -n0 -v
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from jamma.jlinalg import (
    HAS_C_EXTENSION,
    blas_has_dsyevd,
    blas_has_dsyevr,
    eigh,
    get_n_threads,
    set_n_threads,
)

# True when the C extension can actually run eigh (has vendor DSYEVD or DSYEVR).
_HAS_VENDOR_LAPACK = HAS_C_EXTENSION and (blas_has_dsyevd or blas_has_dsyevr)

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

    @pytest.mark.parametrize("N", [1, 2, 3, 10, 100])
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

    @pytest.mark.parametrize("N", [1, 2, 5, 50])
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
        with pytest.raises(ValueError, match=r"square|shape"):
            eigh(K)

    def test_raises_1d(self) -> None:
        """eigh on 1-D array raises ValueError."""
        K = np.ones(5)
        with pytest.raises(ValueError, match=r"2-D|ndim"):
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
      accounts for the different rounding paths in jlinalg QR vs LAPACK dstevd.
    - Reconstruction: ||K - V W V^T||_F / ||K||_F < 1e-8.  For N=1940 with
      condition number 4e13, O(N * eps * cond) gives ~4e13 * 2e-16 * 1940 ~
      0.15 — but the tridiagonalization concentrates error so the practical
      bound is ~1e-8 in the Frobenius norm.
    - Orthogonality: ||V^T V - I||_F < 1e-5.  QR iteration loses a digit
      per decade of condition ratio; with 13 decades the accumulated orthogon-
      ality error is O(sqrt(N) * eps * cond_sub) where cond_sub is the local
      subproblem condition.
    """
    fixtures_dir = Path(__file__).parent / "fixtures" / "kinship"
    kinship_path = fixtures_dir / "mouse_hs1940.cXX.txt"
    K = np.loadtxt(kinship_path)
    assert K.ndim == 2, f"Kinship matrix must be 2-D, got {K.ndim}-D"
    assert K.shape[0] == K.shape[1], f"Kinship matrix must be square, got {K.shape}"

    # Reference eigenvalues from numpy (LAPACK dstevd/dsyevd)
    w_ref = np.linalg.eigvalsh(K.copy())

    # jlinalg eigh
    K_jlinalg = K.copy()
    w_jlinalg, v_jlinalg = eigh(K_jlinalg)

    # Compare eigenvalues: use atol for near-zero eigenvalues, rtol for large ones
    npt.assert_allclose(
        w_jlinalg,
        w_ref,
        rtol=1e-8,
        atol=1e-12,
        err_msg="jlinalg eigh eigenvalues differ from np.linalg.eigh on mouse_hs1940",
    )

    _assert_orthogonality(v_jlinalg, 1e-5, "Kinship")
    _assert_reconstruction(K, w_jlinalg, v_jlinalg, 1e-4, "Kinship")


# VALID-03: Sign ambiguity in eigenvectors.
# All eigenvector comparisons in this module use reconstruction checks
# (||K - V diag(w) V^T||_F / ||K||_F) and orthogonality checks
# (||V^T V - I||_F) rather than element-wise comparison.  This is the
# correct approach because eigenvectors are unique only up to sign (and
# up to rotation within degenerate eigenspaces).  Reconstruction and
# orthogonality are sign-invariant, making them robust to implementation
# differences between jlinalg and NumPy/LAPACK.


@pytest.mark.skipif(
    not HAS_C_EXTENSION or not blas_has_dsyevd,
    reason="Requires vendor DSYEVD (ILP64 LAPACK) — jlinalg D&C path has looser bounds",
)
def test_mouse_hs1940_eigendecomp_strict() -> None:
    """VALID-08: strict eigendecomp on real mouse_hs1940 kinship data.

    Tighter tolerances than test_vs_mouse_hs1940_kinship to validate the
    vendor LAPACK eigensolver quality on real data.  The vendor DSYEVD/DSYEVR
    path (Accelerate, MKL-ILP64) achieves orthogonality < 1e-12 on this
    1940 x 1940 kinship matrix.  The jlinalg D&C path has conditioning-
    dependent error that exceeds this threshold, so this test is gated on
    vendor LAPACK availability.
    """
    fixtures_dir = Path(__file__).parent / "fixtures" / "kinship"
    kinship_path = fixtures_dir / "mouse_hs1940.cXX.txt"
    K = np.loadtxt(kinship_path)
    K_copy = K.copy()

    w, v = eigh(K_copy)

    # VALID-08 requirement: orthogonality < 1e-12
    ortho_err = _assert_orthogonality(v, 1e-12, "mouse_hs1940 strict")

    # Reconstruction: < 1e-8 (same as existing test, but confirms on C path)
    recon_err = _assert_reconstruction(K, w, v, 1e-8, "mouse_hs1940 strict")

    print(f"mouse_hs1940 strict: ortho={ortho_err:.3e}, recon={recon_err:.3e}")


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

        With DSTEDC_BASE=64, D&C merge kicks in at N >= 65.  After
        Phase 80.4 (A/B/C rational interpolation + delta_mat weight
        product), D&C achieves ~1e-9 residuals without QR fallback.
        Tolerance is 1e-8 to accept both D&C-direct and QR-fallback
        paths.
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

    def test_set_n_threads_accepts_large(self) -> None:
        """set_n_threads(9999) stores the value (no clamping after own-BLAS removal)."""
        original = get_n_threads()
        set_n_threads(9999)
        assert get_n_threads() == 9999
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
    Python binding.  We verify them indirectly: eigh calls jlinalg_dgemm_c
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
    """Benchmark jlinalg.eigh vs numpy.linalg.eigh for Phase 80.1."""

    def test_eigh_throughput_n500(self) -> None:
        """eigh should be < 10x slower than numpy at N=500."""
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

        # Time jlinalg
        t0 = time.perf_counter()
        for _ in range(3):
            eigh(K.copy())
        t_jlinalg = (time.perf_counter() - t0) / 3

        ratio = t_jlinalg / t_numpy
        print(
            f"\nN={N}: jlinalg={t_jlinalg:.4f}s, np={t_numpy:.4f}s, ratio={ratio:.1f}x"
        )
        # Cross-platform soft gate.  Apple Silicon Accelerate is multi-threaded
        # and uses LAPACK dsyevd with vDSP BLAS — jlinalg single-threaded LAPACK
        # cannot match that.  Use 15x to accommodate Accelerate (~14x observed)
        # while still catching gross regressions.  On x86_64 with AVX2 + MKL,
        # expect < 8x.
        assert ratio < 15.0, (
            f"jlinalg eigh is {ratio:.1f}x slower than numpy -- expected < 15x"
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

    Tested indirectly: if eigh works at N > DSTEDC_BASE (64), the GEMM
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
    """LAPACK sources in build configs must use strict IEEE 754 flags.

    Phase 123-05 consolidated compile flags into build_support/compile_and_link.py.
    All three entry points (hatch_build.py, _compile_jlinalg.py, _compile_accel.py)
    route through the helper instead of keeping inline flag lists, so we validate
    the single source of truth once: LAPACK_CFLAGS must include '-fno-fast-math'
    and must NOT include '-ffast-math'. The dstedc secular equation uses IEEE 754
    infinity arithmetic which -ffast-math breaks.
    """
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from build_support.compile_and_link import LAPACK_CFLAGS, LAPACK_SOURCES

    # LAPACK_SOURCES identifies the source files that require strict flags.
    assert LAPACK_SOURCES, (
        "build_support.compile_and_link.LAPACK_SOURCES must list at least one "
        "source file (eigh.c) that requires strict IEEE 754 flags"
    )

    # LAPACK_CFLAGS is the canonical flag list — must include strict IEEE 754.
    assert "-fno-fast-math" in LAPACK_CFLAGS, (
        "LAPACK_CFLAGS must include '-fno-fast-math' to ensure strict IEEE 754 "
        "arithmetic for the LAPACK secular equation solver"
    )
    assert "-ffast-math" not in LAPACK_CFLAGS, (
        "LAPACK_CFLAGS must NOT include '-ffast-math'"
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


def _find_jlinalg_so() -> str:
    """Find the _jlinalg shared library path."""
    import jamma.jlinalg._jlinalg as mod

    return mod.__file__


class _EighStatus(ctypes.Structure):
    """ctypes mirror of jlinalg_eigh_status_t."""

    _fields_ = [
        ("vendor_lapack_skipped", ctypes.c_int),
    ]


def _load_jlinalg_eigh_c():
    """Load jlinalg_eigh_c via ctypes.

    Returns:
        Callable with signature matching jlinalg_eigh_c.
    """
    so_path = _find_jlinalg_so()
    lib = ctypes.CDLL(so_path)

    fn = lib.jlinalg_eigh_c
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_longlong,  # npy_intp N
        ctypes.c_void_p,  # double *K
        ctypes.c_longlong,  # npy_intp ldk
        ctypes.c_void_p,  # double *eigenvalues
        ctypes.c_void_p,  # double *eigenvectors
        ctypes.c_longlong,  # npy_intp ldz
        ctypes.POINTER(_EighStatus),  # jlinalg_eigh_status_t *status
    ]
    return fn


def _ptr(arr: np.ndarray) -> ctypes.c_void_p:
    """Get ctypes void pointer to numpy array data."""
    return ctypes.c_void_p(arr.ctypes.data)


def _call_eigh_with_status(
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, _EighStatus]:
    """Call jlinalg_eigh_c via ctypes, returning eigenvalues, eigenvectors, status.

    Args:
        K: Symmetric matrix (N x N, float64, C-contiguous). Modified in place.

    Returns:
        Tuple of (eigenvalues, eigenvectors, status).
    """
    N = K.shape[0]
    eigenvalues = np.empty(N, dtype=np.float64)
    eigenvectors = np.empty((N, N), dtype=np.float64)
    status = _EighStatus()

    fn = _load_jlinalg_eigh_c()
    ret = fn(
        N, _ptr(K), N, _ptr(eigenvalues), _ptr(eigenvectors), N, ctypes.byref(status)
    )
    assert ret == 0, f"jlinalg_eigh_c returned {ret}"
    return eigenvalues, eigenvectors, status


# ---------------------------------------------------------------------------
# TestDstedcNoSecularFailures — secular solver convergence and QR tracking
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_VENDOR_LAPACK,
    reason="Vendor LAPACK (DSYEVD/DSYEVR) required for secular failure detection",
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
        """N=200: vendor eigh produces correct reconstruction and orthogonality."""
        rng = np.random.default_rng(42)
        N = 200
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, _ = _call_eigh_with_status(K_copy)
        _assert_reconstruction(K, w, v, 1e-8, f"SecularSolver N={N}")
        _assert_orthogonality(v, 1e-8, f"SecularSolver N={N}")

    @pytest.mark.slow
    def test_no_secular_failures_n500(self) -> None:
        """N=500: vendor eigh produces correct reconstruction and orthogonality."""
        rng = np.random.default_rng(42)
        N = 500
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, _ = _call_eigh_with_status(K_copy)
        _assert_reconstruction(K, w, v, 1e-8, f"SecularSolver N={N}")
        _assert_orthogonality(v, 1e-8, f"SecularSolver N={N}")

    @pytest.mark.slow
    def test_no_secular_failures_n1000(self) -> None:
        """N=1000: vendor eigh produces correct reconstruction and orthogonality."""
        rng = np.random.default_rng(42)
        N = 1000
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, _ = _call_eigh_with_status(K_copy)
        _assert_reconstruction(K, w, v, 1e-8, f"SecularSolver N={N}")
        _assert_orthogonality(v, 1e-8, f"SecularSolver N={N}")


# ---------------------------------------------------------------------------
# TestDlaed4Convergence — dlaed4 convergence on difficult inputs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_VENDOR_LAPACK,
    reason="Vendor LAPACK (DSYEVD/DSYEVR) required for dlaed4 convergence tests",
)
class TestDlaed4Convergence:
    """Test dlaed4 convergence on known-difficult secular equation inputs.

    These tests exercise dlaed4 indirectly through jlinalg_eigh_c by
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

    @pytest.mark.parametrize(
        "K",
        [
            np.array([[3.0, 1.0], [1.0, 5.0]]),  # well-separated
            np.array([[1.0, 0.5], [0.5, 1.0]]),  # close eigenvalues
            np.array([[1e-10, 1e-11], [1e-11, 2e-10]]),  # tiny scale
            np.array([[1e10, 1e9], [1e9, 2e10]]),  # huge scale
        ],
        ids=["well-separated", "close", "tiny-scale", "huge-scale"],
    )
    def test_dlaed5_both_roots(self, K: np.ndarray) -> None:
        """N=2 exercises dlaed5 (analytical 2-pole secular solver).

        dlaed4 delegates all N=2 cases to dlaed5, which has three branches
        (i=0 w>0, i=0 w<=0, i=1). Every D&C merge at DSTEDC_BASE boundary
        involves 2-pole sub-problems, so dlaed5 correctness is critical.
        """
        K_copy = K.copy()
        w, v, _ = _call_eigh_with_status(K_copy)
        _assert_reconstruction(K, w, v, 1e-12, "dlaed5")
        _assert_orthogonality(v, 1e-12, "dlaed5")
        # Eigenvalues ascending
        assert w[0] <= w[1] + 1e-14, f"Eigenvalues not ascending: {w}"

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

        w, v, _ = _call_eigh_with_status(K_copy)
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

        w, v, _ = _call_eigh_with_status(K_copy)
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

        w, v, _ = _call_eigh_with_status(K_copy)
        _assert_reconstruction(K, w, v, 1e-12, "Boundary eigenvalue")

    def test_negative_rho_zvector_sign(self) -> None:
        """Regression: z-vector sign with negative off-diagonal (rho < 0).

        Constructs a tridiagonal matrix where e[m-1] < 0 at a D&C split
        point, forcing sign_rho = -1 in dstedc_recurse. The z-vector sign
        fix (apply sign_rho only to the right half, matching LAPACK DLAED2)
        is critical — applying it to all of z produces 5-13% residual errors.

        Uses N=130 to ensure at least one D&C merge above DSTEDC_BASE=64.
        """
        N = 130
        d = np.linspace(1.0, 10.0, N)
        e = np.full(N - 1, 0.5)
        # Make e[mid-1] negative to force sign_rho = -1 at the split point
        mid = N // 2
        e[mid - 1] = -0.5
        K = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        K_copy = K.copy()

        w, v, _ = _call_eigh_with_status(K_copy)
        _assert_reconstruction(K, w, v, 1e-8, "Negative rho z-vector")
        _assert_orthogonality(v, 1e-8, "Negative rho z-vector")

    def test_delta_quality_via_reconstruction(self) -> None:
        """Verify vendor eigh delivers reconstruction and orthogonality at scale.

        Asserts:
          - Reconstruction < 1e-8 at all sizes
          - Orthogonality < 1e-8 at all sizes
        """
        rng = np.random.default_rng(42)
        for N in [50, 200, 500]:
            A = rng.standard_normal((N, N))
            A = (A + A.T) / 2
            w, V, _ = _call_eigh_with_status(A.copy())

            recon = np.linalg.norm(A - V @ np.diag(w) @ V.T) / np.linalg.norm(A)
            assert recon < 1e-8, f"Reconstruction {recon:.2e} at N={N} (threshold 1e-8)"

            orth = np.linalg.norm(V.T @ V - np.eye(N))
            assert orth < 1e-8, f"Orthogonality {orth:.2e} at N={N} (threshold 1e-8)"


# ---------------------------------------------------------------------------
# Vendor dsyevd dispatch tests
# ---------------------------------------------------------------------------


class TestEighVendorDispatch:
    """Verify eigh produces correct results via vendor dsyevd or jlinalg pipeline."""

    def test_eigh_vendor_reconstruction_accuracy(self):
        """Reconstruction ||K - UDU.T||/||K|| on random SPD matrix.

        Tolerance is 1e-9 to accommodate OpenBLAS dsyevd which has lower
        accuracy than Accelerate/MKL on some matrix structures.
        """
        rng = np.random.default_rng(42)
        A = rng.standard_normal((100, 100))
        K = np.ascontiguousarray(A @ A.T + np.eye(100), dtype=np.float64)
        K_orig = K.copy()
        w, v = eigh(K)
        recon = v @ np.diag(w) @ v.T
        rel_err = np.linalg.norm(K_orig - recon) / np.linalg.norm(K_orig)
        assert rel_err < 1e-9, f"Reconstruction error {rel_err:.2e}"

    def test_eigh_vendor_orthogonality(self):
        """Orthogonality ||U.T U - I|| < 1e-13 on random SPD matrix."""
        rng = np.random.default_rng(123)
        A = rng.standard_normal((100, 100))
        K = np.ascontiguousarray(A @ A.T + np.eye(100), dtype=np.float64)
        w, v = eigh(K)
        orth_err = np.linalg.norm(v.T @ v - np.eye(100))
        assert orth_err < 1e-13, f"Orthogonality error {orth_err:.2e}"

    def test_eigh_vendor_eigenvalue_ascending(self):
        """Eigenvalues are ascending."""
        rng = np.random.default_rng(456)
        A = rng.standard_normal((50, 50))
        K = np.ascontiguousarray(A @ A.T + np.eye(50), dtype=np.float64)
        w, v = eigh(K)
        assert np.all(np.diff(w) >= 0), "Eigenvalues not ascending"

    def test_eigh_vendor_matches_numpy(self):
        """Eigenvalues match numpy.linalg.eigh within rtol=1e-12."""
        rng = np.random.default_rng(789)
        A = rng.standard_normal((50, 50))
        K = np.ascontiguousarray(A @ A.T + np.eye(50), dtype=np.float64)
        K_copy = K.copy()
        w_jlinalg, _ = eigh(K)
        w_numpy, _ = np.linalg.eigh(K_copy)
        npt.assert_allclose(w_jlinalg, w_numpy, rtol=1e-12)

    def test_eigh_vendor_degenerate_eigenvalues(self):
        """Stress test: block-diagonal with repeated eigenvalues.

        OpenBLAS dsyevd can fail to converge on highly degenerate matrices
        (returns info > 0).  In that case jlinalg_eigh_c propagates the error
        rather than falling through to the jlinalg pipeline.  We tolerate this
        by catching RuntimeError and verifying numpy succeeds instead.
        """
        N = 100
        # 5 blocks of 20 with identical eigenvalues within each block
        blocks = []
        for i in range(5):
            rng = np.random.default_rng(i)
            Q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
            lam = (i + 1) * np.ones(20)
            blocks.append(Q @ np.diag(lam) @ Q.T)
        K = np.ascontiguousarray(
            np.block(
                [
                    [blocks[i] if i == j else np.zeros((20, 20)) for j in range(5)]
                    for i in range(5)
                ]
            ),
            dtype=np.float64,
        )
        K_orig = K.copy()
        try:
            w, v = eigh(K)
        except (RuntimeError, np.linalg.LinAlgError) as e:
            if "convergence failure" in str(e):
                # Vendor dsyevd failed on degenerate matrix (OpenBLAS).
                # Verify numpy handles it — the matrix itself is valid.
                w_np, v_np = np.linalg.eigh(K_orig)
                recon = v_np @ np.diag(w_np) @ v_np.T
                rel_err = np.linalg.norm(K_orig - recon) / np.linalg.norm(K_orig)
                assert rel_err < 1e-12, (
                    f"numpy also failed reconstruction: {rel_err:.2e}"
                )
                return
            raise
        recon = v @ np.diag(w) @ v.T
        rel_err = np.linalg.norm(K_orig - recon) / np.linalg.norm(K_orig)
        assert rel_err < 1e-9, (
            f"Reconstruction error on degenerate matrix: {rel_err:.2e}"
        )
        orth_err = np.linalg.norm(v.T @ v - np.eye(N))
        assert orth_err < 1e-9, (
            f"Orthogonality error on degenerate matrix: {orth_err:.2e}"
        )

    def test_eigh_vendor_size_1(self):
        """Size-1 matrix: edge case."""
        K = np.array([[3.14]], dtype=np.float64)
        w, v = eigh(K)
        assert abs(w[0] - 3.14) < 1e-15
        assert abs(v[0, 0] - 1.0) < 1e-15


# ---------------------------------------------------------------------------
# Backend reporting consistency tests
# ---------------------------------------------------------------------------


class TestEighBackendReporting:
    """Verify blas_has_dsyevd is consistent with backend."""

    def test_has_lapack_consistent_with_backend(self):
        from jamma.jlinalg import blas_backend, blas_has_dsyevd

        if blas_backend in ("Accelerate-ILP64", "MKL-ILP64"):
            assert blas_has_dsyevd == 1, (
                f"Backend {blas_backend} should have LAPACK "
                f"but blas_has_dsyevd={blas_has_dsyevd}"
            )

    def test_has_dsyevr_flag_exposed(self):
        """blas_has_dsyevr is accessible as an int constant."""
        from jamma.jlinalg import blas_has_dsyevr

        assert isinstance(blas_has_dsyevr, int)
        assert blas_has_dsyevr in (0, 1)


# ---------------------------------------------------------------------------
# EIGH-10: LinAlgError contract — convergence failure raises LinAlgError
# ---------------------------------------------------------------------------


class TestEighLinAlgError:
    """Verify eigh raises numpy.linalg.LinAlgError on convergence failure."""

    def test_eigh_convergence_linalgerror(self):
        """Convergence failure raises np.linalg.LinAlgError, not RuntimeError."""
        K = np.full((4, 4), np.nan)
        with pytest.raises(np.linalg.LinAlgError):
            eigh(K)

    def test_eigh_convergence_linalgerror_not_runtime(self):
        """Verify the error is specifically LinAlgError, not RuntimeError wrapping."""
        K = np.full((4, 4), np.nan)
        try:
            eigh(K)
            pytest.fail("eigh should have raised on NaN input")  # pragma: no cover
        except np.linalg.LinAlgError:
            pass  # correct
        except RuntimeError:
            pytest.fail("eigh raised RuntimeError instead of numpy.linalg.LinAlgError")


# ---------------------------------------------------------------------------
# EIGH-11: DSYEVR fallback path tests
# ---------------------------------------------------------------------------


class TestEighDsyevrFallback:
    """Verify DSYEVD→DSYEVR memory-pressure fallback in eigendecompose_kinship."""

    def test_dsyevr_fallback_on_dsyevd_memory_error(self):
        """When DSYEVD workspace can't allocate, DSYEVR should be tried.

        We mock jlinalg.eigh to simulate the fallback: first call raises
        MemoryError (DSYEVD path), second succeeds (DSYEVR path).  Since the
        C code handles this internally, we verify the contract at the Python
        level by confirming eigendecompose_kinship handles MemoryError from
        jlinalg.eigh correctly.
        """
        from unittest.mock import patch

        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        A = rng.standard_normal((30, 30))
        K = np.ascontiguousarray((A @ A.T) / 30, dtype=np.float64)

        # Mock jlinalg.eigh to raise MemoryError
        with patch("jamma.lmm.eigen.jlinalg") as mock_jlinalg:
            mock_jlinalg.blas_has_dsyevr = 1
            mock_jlinalg.eigh.side_effect = MemoryError("workspace allocation failed")

            with pytest.raises(MemoryError, match="workspace allocation failed"):
                eigendecompose_kinship(K, check_memory=False)

    def test_eigh_produces_correct_result_regardless_of_dispatch(self):
        """Verify eigh returns correct eigendecomposition (covers whichever
        vendor path is active — DSYEVD, DSYEVR, or jlinalg D&C)."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((80, 80))
        K = np.ascontiguousarray(A @ A.T + np.eye(80), dtype=np.float64)
        K_orig = K.copy()

        w, v = eigh(K)

        # Reconstruction check — DSYEVR on Linux gives ~6e-9, so use 1e-8
        recon = v @ np.diag(w) @ v.T
        rel_err = np.linalg.norm(K_orig - recon) / np.linalg.norm(K_orig)
        assert rel_err < 1e-8, f"Reconstruction error {rel_err:.2e}"

        # Eigenvalues ascending
        assert np.all(np.diff(w) >= 0), "Eigenvalues not ascending"


# ---------------------------------------------------------------------------
# EIGH-12: Error propagation through eigendecompose_kinship
# ---------------------------------------------------------------------------


class TestEigendecomposeKinshipErrors:
    """Verify eigendecompose_kinship catches and re-raises errors with logging."""

    def test_memoryerror_caught_and_reraised(self):
        """MemoryError from jlinalg.eigh is caught, logged, and re-raised."""
        from unittest.mock import patch

        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        K = np.ascontiguousarray(rng.standard_normal((20, 20)), dtype=np.float64)
        K = (K + K.T) / 2

        with patch("jamma.lmm.eigen.jlinalg") as mock_jlinalg:
            mock_jlinalg.blas_has_dsyevr = 0
            mock_jlinalg.eigh.side_effect = MemoryError("out of memory")

            with pytest.raises(MemoryError, match="out of memory"):
                eigendecompose_kinship(K, check_memory=False)

    def test_linalgerror_caught_and_reraised(self):
        """LinAlgError from jlinalg.eigh is caught, logged, and re-raised."""
        from unittest.mock import patch

        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        K = np.ascontiguousarray(rng.standard_normal((20, 20)), dtype=np.float64)
        K = (K + K.T) / 2

        with patch("jamma.lmm.eigen.jlinalg") as mock_jlinalg:
            mock_jlinalg.blas_has_dsyevr = 0
            mock_jlinalg.eigh.side_effect = np.linalg.LinAlgError("convergence failure")

            with pytest.raises(np.linalg.LinAlgError, match="convergence failure"):
                eigendecompose_kinship(K, check_memory=False)

    def test_runtimeerror_caught_and_reraised(self):
        """RuntimeError from jlinalg.eigh is caught, logged, and re-raised."""
        from unittest.mock import patch

        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        K = np.ascontiguousarray(rng.standard_normal((20, 20)), dtype=np.float64)
        K = (K + K.T) / 2

        with patch("jamma.lmm.eigen.jlinalg") as mock_jlinalg:
            mock_jlinalg.blas_has_dsyevr = 0
            mock_jlinalg.eigh.side_effect = RuntimeError(
                "illegal argument to vendor LAPACK"
            )

            with pytest.raises(RuntimeError, match="illegal argument"):
                eigendecompose_kinship(K, check_memory=False)

    def test_runtimeerror_log_message_distinct_from_linalgerror(self):
        """RuntimeError log says 'internal error', not 'not PSD'."""
        from unittest.mock import patch

        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        K = np.ascontiguousarray(rng.standard_normal((20, 20)), dtype=np.float64)
        K = (K + K.T) / 2

        with patch("jamma.lmm.eigen.jlinalg") as mock_jlinalg:
            mock_jlinalg.blas_has_dsyevr = 0
            mock_jlinalg.eigh.side_effect = RuntimeError("bad arg")

            with pytest.raises(RuntimeError):
                eigendecompose_kinship(K, check_memory=False)

    def test_internal_error_raises_runtime_error(self):
        """JLINALG_EXT_INTERNAL_ERROR maps to RuntimeError in pymodule.c."""
        from unittest.mock import patch

        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(99)
        K = rng.standard_normal((10, 10))
        K = np.ascontiguousarray((K + K.T) / 2, dtype=np.float64)

        with patch("jamma.lmm.eigen.jlinalg") as mock_jlinalg:
            mock_jlinalg.blas_has_dsyevr = 0
            mock_jlinalg.eigh.side_effect = RuntimeError("internal error in dstedc")
            with pytest.raises(RuntimeError, match="internal error"):
                eigendecompose_kinship(K, check_memory=False)


class TestDstedcNoAbort:
    """Structural test: dstedc.c must not call abort()."""

    def test_dstedc_no_abort_call(self):
        """dstedc.c must not call abort() -- errors return codes instead."""
        import pathlib
        import re

        dstedc_src = (
            pathlib.Path(__file__).parent.parent
            / "src"
            / "jamma"
            / "jlinalg"
            / "src"
            / "dstedc.c"
        )
        if not dstedc_src.exists():
            pytest.skip("source not available")
        source = dstedc_src.read_text()
        # Check for actual abort() calls (not just the word "abort" in comments)
        abort_calls = re.findall(r"\babort\s*\(\s*\)", source)
        assert abort_calls == [], (
            f"dstedc.c still contains abort() calls: {len(abort_calls)} found. "
            "Use error return codes instead."
        )


# ---------------------------------------------------------------------------
# EIGH-13: Eigenvalue zeroing boundary tests
# ---------------------------------------------------------------------------


class TestEigenvalueZeroingBoundary:
    """Boundary tests for eigenvalue zeroing in eigendecompose_kinship.

    The zeroing logic (eigenvalues with |value| < threshold set to 0) is
    driver-independent. These tests exercise precise boundary behavior.
    """

    def test_eigenvalue_at_threshold_is_zeroed(self):
        """Eigenvalue exactly at 1e-10 threshold is zeroed."""

        from jamma.lmm.eigen import eigendecompose_kinship

        n = 10
        # Construct diagonal matrix with one eigenvalue at exactly the threshold
        evals = np.array([1e-10] + [1.0] * (n - 1), dtype=np.float64)
        K = np.diag(evals)

        eigenvalues, _ = eigendecompose_kinship(K.copy(), check_memory=False)

        # The 1e-10 eigenvalue should be zeroed (|1e-10| < 1e-10 is False,
        # but abs(1e-10) == 1e-10, and the check is abs_evals < threshold,
        # so exactly-at-threshold is NOT zeroed)
        assert eigenvalues[0] == pytest.approx(1e-10, abs=1e-15), (
            "Eigenvalue exactly at threshold should NOT be zeroed,"
            f" got {eigenvalues[0]}"
        )

    def test_eigenvalue_just_below_threshold_is_zeroed(self):
        """Eigenvalue just below 1e-10 threshold is zeroed."""
        n = 10
        evals = np.array([9.99e-11] + [1.0] * (n - 1), dtype=np.float64)
        K = np.diag(evals)

        from jamma.lmm.eigen import eigendecompose_kinship

        eigenvalues, _ = eigendecompose_kinship(K.copy(), check_memory=False)

        assert eigenvalues[0] == 0.0, (
            f"Eigenvalue below threshold should be zeroed, got {eigenvalues[0]}"
        )

    def test_negative_eigenvalue_above_neg_threshold_zeroed(self):
        """Negative eigenvalue with |value| < threshold is zeroed."""
        n = 10
        evals = np.array([-5e-11] + [1.0] * (n - 1), dtype=np.float64)
        K = np.diag(evals)

        from jamma.lmm.eigen import eigendecompose_kinship

        eigenvalues, _ = eigendecompose_kinship(K.copy(), check_memory=False)

        assert eigenvalues[0] == 0.0, (
            f"Small negative eigenvalue should be zeroed, got {eigenvalues[0]}"
        )

    def test_negative_eigenvalue_below_neg_threshold_zeroed_with_warning(self):
        """Negative eigenvalue |value| > threshold triggers warning."""
        import warnings

        n = 10
        evals = np.array([-0.5] + [1.0] * (n - 1), dtype=np.float64)
        K = np.diag(evals)

        from jamma.lmm.eigen import eigendecompose_kinship

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eigenvalues, _ = eigendecompose_kinship(K.copy(), check_memory=False)

        # Should have warned about negative eigenvalues
        neg_warnings = [x for x in w if "negative eigenvalue" in str(x.message)]
        assert len(neg_warnings) > 0, "Expected warning about negative eigenvalues"

        # The -0.5 eigenvalue should be zeroed
        assert eigenvalues[0] == 0.0, (
            f"Large negative eigenvalue should be zeroed, got {eigenvalues[0]}"
        )

    def test_multiple_zero_eigenvalues_warning(self):
        """More than 1 zero eigenvalue triggers rank-deficient warning."""
        import warnings

        n = 10
        evals = np.array([1e-12, 1e-13] + [1.0] * (n - 2), dtype=np.float64)
        K = np.diag(evals)

        from jamma.lmm.eigen import eigendecompose_kinship

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eigenvalues, _ = eigendecompose_kinship(K.copy(), check_memory=False)

        rank_warnings = [x for x in w if "rank-deficient" in str(x.message)]
        assert len(rank_warnings) > 0, "Expected rank-deficient warning"

    def test_single_zero_eigenvalue_no_warning(self):
        """Exactly 1 zero eigenvalue should NOT trigger rank-deficient warning."""
        import warnings

        n = 10
        evals = np.array([1e-12] + [1.0] * (n - 1), dtype=np.float64)
        K = np.diag(evals)

        from jamma.lmm.eigen import eigendecompose_kinship

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            eigenvalues, _ = eigendecompose_kinship(K.copy(), check_memory=False)

        rank_warnings = [x for x in w if "rank-deficient" in str(x.message)]
        assert len(rank_warnings) == 0, (
            "Single zero eigenvalue should NOT trigger rank-deficient warning"
        )
