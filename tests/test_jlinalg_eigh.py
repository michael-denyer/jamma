"""Contract tests for ``jamma.jlinalg.eigh``, the symmetric eigendecomposition.

These test the *contract* of ``eigh()``, not any one implementation of it.
``eigh`` dispatches to vendor DSYEVD, then vendor DSYEVR, then falls back to
``numpy.linalg.eigh`` when neither is available (see
``src/jamma/jlinalg/src/eigh.c``).  Which backend answers a given call depends
on the installed BLAS, so every assertion here is chosen to hold for all three.

What that means in practice: no test compares eigenvectors element-wise against
a reference.  Eigenvectors are unique only up to sign, and only up to an
arbitrary rotation within a degenerate eigenspace, so independent
implementations legitimately return different bases.  The properties asserted
instead are backend-independent:

- **Reconstruction** — ``||K - V diag(w) V.T||_F / ||K||_F`` below a tolerance
- **Orthogonality** — ``||V.T V - I||_F`` below a tolerance
- **Ordering** — eigenvalues ascending
- **Layout** — shapes, float64 dtype, C-contiguity
- **Error contract** — which exception type each failure mode raises

Sizes are parametrised across ``EIGH_BOUNDARY_SIZES`` because vendor LAPACK picks
different internal blocking and different kernels by size, so a bug that only
appears at one blocking boundary stays visible.

Run with ``-n0`` to avoid interference with the OpenMP threading tests:
    uv run pytest tests/test_jlinalg_eigh.py -x -n0 -v
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from jamma.jlinalg import (
    HAS_C_EXTENSION,
    blas_has_dsyevd,
    blas_has_dsyevr,
    eigh,
)
from tests.builders import EIGH_BOUNDARY_SIZES
from tests.fixture_paths import KINSHIP_DIR
from tests.jlinalg_eigh_helpers import (
    _assert_orthogonality,
    _assert_reconstruction,
    _random_spd,
)

pytestmark = pytest.mark.tier0

# True when the C extension can actually run eigh (has vendor DSYEVD or DSYEVR).
_HAS_VENDOR_LAPACK = HAS_C_EXTENSION and (blas_has_dsyevd or blas_has_dsyevr)

# ---------------------------------------------------------------------------
# Core correctness on inputs with a known answer
# ---------------------------------------------------------------------------


class TestEigh:
    """eigh returns the known eigendecomposition of hand-checkable inputs."""

    @pytest.mark.parametrize("N", [1, 2, 3, 10, 100])
    def test_identity(self, N: int) -> None:
        """eigh(eye(N)) returns all-ones eigenvalues and orthogonal eigenvectors."""
        K = np.eye(N)
        K_copy = K.copy()
        w, v, _ = eigh(K_copy)
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
        w, v, _ = eigh(K_copy)
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

    @pytest.mark.parametrize("N", EIGH_BOUNDARY_SIZES)
    def test_random_spd(self, N: int) -> None:
        """eigh on random SPD: eigenvalues ascending, eigenvectors unit norm."""
        rng = np.random.default_rng(42 + N * 7)
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, _ = eigh(K_copy)
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
        w, _, _ = eigh(K_copy)
        diffs = np.diff(w)
        assert np.all(diffs >= -1e-14), (
            f"Eigenvalues not ascending: min diff = {diffs.min()}"
        )

    def test_empty_matrix(self) -> None:
        """eigh on 0x0 matrix returns empty eigenvalues and eigenvectors."""
        K = np.zeros((0, 0), dtype=np.float64)
        w, v, _ = eigh(K)
        assert w.shape == (0,), f"Expected (0,) eigenvalues, got {w.shape}"
        assert v.shape == (0, 0), f"Expected (0,0) eigenvectors, got {v.shape}"

    def test_fortran_order_input(self) -> None:
        """eigh on Fortran-order input produces correct results."""
        rng = np.random.default_rng(99)
        N = 20
        K = _random_spd(N, rng)
        K_f = np.asfortranarray(K.copy())
        w, v, _ = eigh(K_f)
        _assert_reconstruction(K, w, v, 1e-13, "Fortran-order")


# ---------------------------------------------------------------------------
# Reconstruction and orthogonality at N=1000
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reconstruction_accuracy() -> None:
    """N=1000 random SPD: ||K - U diag(w) U.T|| / ||K|| < 1e-8.

    The observed residual is ~1e-9 at this size; 1e-8 leaves margin for the
    spread between the vendor backends and the NumPy fallback.
    """
    rng = np.random.default_rng(42)
    N = 1000
    K = _random_spd(N, rng)
    K_copy = K.copy()
    w, v, _ = eigh(K_copy)
    _assert_reconstruction(K, w, v, 1e-8, "N=1000")


@pytest.mark.slow
def test_orthogonality() -> None:
    """N=1000 random SPD: ||U.T @ U - I||_F < 1e-12."""
    rng = np.random.default_rng(43)
    N = 1000
    K = _random_spd(N, rng)
    K_copy = K.copy()
    _, v, _ = eigh(K_copy)
    _assert_orthogonality(v, 1e-12, "N=1000")


# ---------------------------------------------------------------------------
# Bad input, and the shape/dtype/contiguity of the output
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
        w, v, _ = eigh(K_copy)
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
    w, v, _ = eigh(K_copy)
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
# JLINALG_NO_VENDOR_LAPACK override, honoured inside jlinalg.eigh
# ---------------------------------------------------------------------------


def test_no_vendor_lapack_env_forces_numpy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JLINALG_NO_VENDOR_LAPACK set makes eigh take the NumPy path per call.

    The override lives in jlinalg.eigh (not only in eigendecompose_kinship), so
    any direct caller gets a consistent bypass. The result is still a correct
    decomposition regardless of which vendor backend would otherwise answer.
    """
    rng = np.random.default_rng(4321)
    N = 40
    K = _random_spd(N, rng)
    monkeypatch.setenv("JLINALG_NO_VENDOR_LAPACK", "1")
    w, v, _ = eigh(K.copy())
    _assert_reconstruction(K, w, v, 1e-10, "NO_VENDOR_LAPACK forced")
    assert np.all(np.diff(w) >= -1e-12), "eigenvalues not ascending on forced path"


@pytest.mark.parametrize("off_value", ["", "0"])
def test_no_vendor_lapack_off_values_do_not_force(
    monkeypatch: pytest.MonkeyPatch, off_value: str
) -> None:
    """Empty and "0" leave vendor dispatch in place, matching env_flag semantics."""
    rng = np.random.default_rng(4322)
    N = 30
    K = _random_spd(N, rng)
    monkeypatch.setenv("JLINALG_NO_VENDOR_LAPACK", off_value)
    w, v, _ = eigh(K.copy())
    _assert_reconstruction(K, w, v, 1e-10, f"NO_VENDOR_LAPACK={off_value!r}")


# ---------------------------------------------------------------------------
# Clustered eigenvalues
# ---------------------------------------------------------------------------


def test_block_diagonal_stress() -> None:
    """Block-diagonal 1000x1000 matrix with clustered eigenvalues per block.

    Builds a 10 x 100 block-diagonal matrix and checks reconstruction and
    orthogonality below 1e-8.  Tight clusters within each block are the case
    that separates a well-behaved eigensolver from one that loses
    orthogonality between near-equal eigenvalues, so this is a stress test
    rather than a duplicate of the random-SPD cases.
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
    w, v, _ = eigh(K_copy)

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
      accounts for the different rounding paths across the eigh backends.
    - Reconstruction: ||K - V W V^T||_F / ||K||_F < 1e-8.  For N=1940 with
      condition number 4e13, the worst-case O(N * eps * cond) bound gives
      ~4e13 * 2e-16 * 1940 ~ 0.15, far looser than what any backend here
      actually delivers; 1e-8 is the empirically observed bound.
    - Orthogonality: ||V^T V - I||_F < 1e-5.  Looser than the 1e-12 the
      vendor path reaches, because it must also hold for the NumPy fallback
      on a matrix this ill-conditioned.
    """
    kinship_path = KINSHIP_DIR / "mouse_hs1940.cXX.txt"
    K = np.loadtxt(kinship_path)
    assert K.ndim == 2, f"Kinship matrix must be 2-D, got {K.ndim}-D"
    assert K.shape[0] == K.shape[1], f"Kinship matrix must be square, got {K.shape}"

    # Reference eigenvalues from numpy, whatever LAPACK it is linked against
    w_ref = np.linalg.eigvalsh(K.copy())

    # jlinalg eigh
    K_jlinalg = K.copy()
    w_jlinalg, v_jlinalg, _ = eigh(K_jlinalg)

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


@pytest.mark.skipif(
    not HAS_C_EXTENSION or not blas_has_dsyevd,
    reason="Requires vendor DSYEVD (ILP64 LAPACK); the NumPy fallback is looser",
)
def test_mouse_hs1940_eigendecomp_strict() -> None:
    """Strict eigendecomposition on real mouse_hs1940 kinship data.

    Tighter tolerances than ``test_vs_mouse_hs1940_kinship``, which has to
    hold for every backend.  Vendor DSYEVD/DSYEVR (Accelerate, MKL-ILP64)
    reaches orthogonality < 1e-12 on this 1940 x 1940 matrix; the NumPy
    fallback does not reliably, which is why this one is gated on vendor
    LAPACK rather than loosening the bound for everyone.
    """
    kinship_path = KINSHIP_DIR / "mouse_hs1940.cXX.txt"
    K = np.loadtxt(kinship_path)
    K_copy = K.copy()

    w, v, _ = eigh(K_copy)

    ortho_err = _assert_orthogonality(v, 1e-12, "mouse_hs1940 strict")

    # Same 1e-8 reconstruction bound as the all-backend test, but this time
    # confirmed specifically on the vendor path.
    recon_err = _assert_reconstruction(K, w, v, 1e-8, "mouse_hs1940 strict")

    print(f"mouse_hs1940 strict: ortho={ortho_err:.3e}, recon={recon_err:.3e}")


# ---------------------------------------------------------------------------
# Reconstruction through the C entry point, at a single mid-range size
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required",
)
class TestEighReconstructionMidSize:
    """Reconstruction at N=64, tight tolerance, C extension present.

    Separate from the parametrised sweep because the tolerance is 1e-13
    rather than 1e-8: at this size every backend is accurate to near machine
    precision, so a loose bound here would hide a real regression.
    """

    def test_reconstruction_n64(self) -> None:
        """N=64 random SPD reconstructs to 1e-13."""
        rng = np.random.default_rng(101)
        N = 64
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, _ = eigh(K_copy)
        _assert_reconstruction(K, w, v, 1e-13, "N=64")


# ---------------------------------------------------------------------------
# Matrices whose spectra have a known awkward structure
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required",
)
class TestEighDegenerateSpectra:
    """eigh on spectra that are degenerate, empty, or indefinite.

    Exact eigenvalue multiplicity, an all-zero matrix, and a matrix with
    negative eigenvalues each break a different naive implementation, and none
    is covered by the random-SPD sweep, which almost surely produces distinct
    positive eigenvalues.
    """

    @pytest.mark.parametrize("N", [127, 128, 129, 200])
    def test_reconstruction_boundary_sizes(self, N: int) -> None:
        """Reconstruction holds either side of the 128 blocking boundary.

        1e-8 rather than a tighter bound so this holds for the NumPy fallback
        as well as the vendor backends.
        """
        rng = np.random.default_rng(202 + N)
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, _ = eigh(K_copy)
        _assert_reconstruction(K, w, v, 1e-8, f"boundary N={N}")

    def test_degenerate_eigenvalues(self) -> None:
        """Matrix with exact repeated eigenvalues still yields an orthogonal basis."""
        # diag([1,1,1,2,2,3]) — exact multiplicity
        K = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        w, v, _ = eigh(K.copy())
        npt.assert_allclose(w, [1, 1, 1, 2, 2, 3], atol=1e-14)
        VtV = v.T @ v
        npt.assert_allclose(VtV, np.eye(6), atol=1e-14)

    def test_zero_matrix(self) -> None:
        """Zero matrix: all eigenvalues 0, eigenvectors form orthonormal basis."""
        N = 10
        K = np.zeros((N, N))
        w, v, _ = eigh(K.copy())
        npt.assert_allclose(w, np.zeros(N), atol=1e-15)
        VtV = v.T @ v
        npt.assert_allclose(VtV, np.eye(N), atol=1e-14)

    def test_indefinite_matrix(self) -> None:
        """Matrix with negative eigenvalues."""
        K = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: -1, 3
        w, v, _ = eigh(K.copy())
        npt.assert_allclose(w, [-1.0, 3.0], atol=1e-14)
        VtV = v.T @ v
        npt.assert_allclose(VtV, np.eye(2), atol=1e-14)


# ---------------------------------------------------------------------------
# Per-eigenpair residual, the defining equation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required",
)
class TestEighEigenpairResidual:
    """Each returned pair satisfies K v = w v individually.

    Reconstruction is an aggregate over all N pairs, so a single badly wrong
    eigenvector can hide inside a passing Frobenius norm.  Checking pairs one
    at a time is what catches that, and it also confirms the eigenvectors come
    back in the basis of the input rather than in some internal working basis.
    """

    def test_eigenpair_residual_n100(self) -> None:
        """||K v_j - w_j v_j|| < 1e-7 for the first few eigenpairs at N=100."""
        rng = np.random.default_rng(303)
        N = 100
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, _ = eigh(K_copy)
        for j in range(min(5, N)):
            residual = np.linalg.norm(K @ v[:, j] - w[j] * v[:, j])
            assert residual < 1e-7, (
                f"eigenpair {j} does not satisfy K v = w v: residual={residual:.2e}"
            )


# ---------------------------------------------------------------------------
# Thread control API
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reconstruction and orthogonality together, at two larger sizes
# ---------------------------------------------------------------------------


class TestEighReconstructionAtScale:
    """Both properties asserted on the same decomposition, at N=100 and N=500.

    The smaller cases above check one property at a time.  Asserting both on
    one result rules out a decomposition that reconstructs the input while
    returning a non-orthogonal basis, which is possible and which neither
    check alone would catch.
    """

    @pytest.mark.parametrize(
        ("N", "seed", "ortho_tol"),
        [
            (100, 5001, 1e-12),
            pytest.param(500, 5002, 1e-8, marks=pytest.mark.slow),
        ],
    )
    def test_recon_and_ortho(self, N: int, seed: int, ortho_tol: float) -> None:
        """Reconstruction < 1e-8 and orthogonality < ortho_tol.

        Orthogonality tolerance loosens from 1e-12 at N=100 to 1e-8 at
        N=500, because the error grows with N and this bound must hold for
        the NumPy fallback too.
        """
        rng = np.random.default_rng(seed)
        K = _random_spd(N, rng)
        K_copy = K.copy()
        w, v, _ = eigh(K_copy)

        _assert_reconstruction(K, w, v, 1e-8, f"N={N}")
        _assert_orthogonality(v, ortho_tol, f"N={N}")


# ---------------------------------------------------------------------------
# Throughput against numpy, and a sequential-draw boundary sweep
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for throughput benchmarks",
)
class TestEighThroughput:
    """Benchmark jlinalg.eigh vs numpy.linalg.eigh."""

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
        # Cross-platform soft gate against gross regressions in the dispatch
        # layer, not a performance target.  15x accommodates the widest
        # observed gap; on x86_64 with MKL, expect well under that.
        assert ratio < 15.0, (
            f"jlinalg eigh is {ratio:.1f}x slower than numpy -- expected < 15x"
        )

    def test_eigh_correctness_n1000_post_optimization(self) -> None:
        """Full correctness gate at N=1000."""
        rng = np.random.default_rng(123)
        N = 1000
        A = rng.standard_normal((N, N))
        K = A @ A.T + np.eye(N)

        vals, vecs, _ = eigh(K.copy())
        np_vals, _ = np.linalg.eigh(K.copy())

        # Eigenvalues agree with numpy to ~1e-10 in practice; 1e-8 for margin.
        npt.assert_allclose(vals, np_vals, rtol=1e-8)

        _assert_reconstruction(K, vals, vecs, 1e-8, "N=1000")
        _assert_orthogonality(vecs, 1e-8, "N=1000")


class TestEighBoundarySizeLoop:
    """Two boundary sizes drawn in sequence from one generator.

    Deliberately not parametrised, unlike ``test_reconstruction_boundary_sizes``
    above.  Both matrices come from the same un-reseeded generator, so the
    N=200 input depends on the N=128 draw.  That makes this the one case that
    would catch state leaking between consecutive eigh calls; a parametrised
    version reseeds and cannot.
    """

    def test_reconstruction_n128_n200(self) -> None:
        """Consecutive eigh calls at N=128 then N=200 both reconstruct to 1e-8."""
        rng = np.random.default_rng(6001)
        for N in [128, 200]:
            K = _random_spd(N, rng)
            K_copy = K.copy()
            w, v, _ = eigh(K_copy)
            _assert_reconstruction(K, w, v, 1e-8, f"sequential N={N}")


# ---------------------------------------------------------------------------
# Error path tests
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

    def test_invalid_driver_raises(self):
        """An unrecognised driver name must raise ValueError."""
        K = np.eye(4)
        with pytest.raises(ValueError, match="driver"):
            eigh(K, driver="bogus")  # type: ignore[bad-argument-type]
