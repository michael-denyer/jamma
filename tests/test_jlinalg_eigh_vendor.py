"""Direct-C and vendor-dispatch contracts for ``jamma.jlinalg.eigh``."""

from __future__ import annotations

import ctypes

import numpy as np
import numpy.testing as npt
import pytest

from jamma.jlinalg import (
    HAS_C_EXTENSION,
    blas_has_dsyevd,
    blas_has_dsyevr,
    eigh,
)
from tests.jlinalg_eigh_helpers import (
    _assert_orthogonality,
    _assert_reconstruction,
    _random_spd,
)

pytestmark = pytest.mark.tier0

_HAS_VENDOR_LAPACK = HAS_C_EXTENSION and (blas_has_dsyevd or blas_has_dsyevr)

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
        ("driver_used", ctypes.c_int),
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
        ctypes.c_int,  # int prefer_dsyevr
        ctypes.POINTER(_EighStatus),  # jlinalg_eigh_status_t *status
    ]
    return fn


def _ptr(arr: np.ndarray) -> ctypes.c_void_p:
    """Get ctypes void pointer to numpy array data."""
    return ctypes.c_void_p(arr.ctypes.data)


def _call_eigh_with_status(
    K: np.ndarray, *, prefer_dsyevr: bool = False
) -> tuple[np.ndarray, np.ndarray, _EighStatus]:
    """Call jlinalg_eigh_c via ctypes, returning eigenvalues, eigenvectors, status.

    Args:
        K: Symmetric matrix (N x N, float64, C-contiguous). Modified in place.
        prefer_dsyevr: Skip the DSYEVD attempt and require DSYEVR directly.

    Returns:
        Tuple of (eigenvalues, eigenvectors, status).
    """
    N = K.shape[0]
    eigenvalues = np.empty(N, dtype=np.float64)
    eigenvectors = np.empty((N, N), dtype=np.float64)
    status = _EighStatus()

    fn = _load_jlinalg_eigh_c()
    ret = fn(
        N,
        _ptr(K),
        N,
        _ptr(eigenvalues),
        _ptr(eigenvectors),
        N,
        int(prefer_dsyevr),
        ctypes.byref(status),
    )
    assert ret == 0, f"jlinalg_eigh_c returned {ret}"
    return eigenvalues, eigenvectors, status


# ---------------------------------------------------------------------------
# The same contract, asserted through the raw C entry point
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_VENDOR_LAPACK,
    reason="Vendor LAPACK (DSYEVD/DSYEVR) required to reach jlinalg_eigh_c",
)
class TestEighViaCtypesAtScale:
    """Reconstruction, orthogonality, and driver outcome via the raw C entry point.

    Only ``test_recon_and_ortho`` needs the ctypes call: it is the sole
    assertion in this file that vendor LAPACK actually ran, rather than
    silently falling through to the NumPy path, which the ``status`` struct
    exposes and the Python ``eigh()`` wrapper does not.

    Sizes 5, 64, 65 straddle vendor LAPACK's internal blocking switch at 64.
    """

    @pytest.mark.parametrize("n", [5, 64, 65])
    def test_recon_and_ortho(self, n: int) -> None:
        """Reconstruction, orthogonality, and driver outcome at each size."""
        rng = np.random.default_rng(42)
        K = _random_spd(n, rng)
        K_copy = K.copy()
        w, v, status = _call_eigh_with_status(K_copy)
        _assert_reconstruction(K, w, v, 1e-8, f"C API N={n}")
        _assert_orthogonality(v, 1e-8, f"C API N={n}")
        assert status.vendor_lapack_skipped == 0, (
            f"C API N={n}: vendor LAPACK was skipped (fell through to the "
            "NumPy fallback path) when it should have run"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [500, 1000])
    def test_recon_and_ortho_large(self, n: int) -> None:
        """Reconstruction and orthogonality at larger N through ``eigh()``."""
        rng = np.random.default_rng(42)
        K = _random_spd(n, rng)
        K_copy = K.copy()
        w, v, _ = eigh(K_copy)
        _assert_reconstruction(K, w, v, 1e-8, f"N={n}")
        _assert_orthogonality(v, 1e-8, f"N={n}")


_JLINALG_EXT_BAD_STRIDE = -1004


class TestEighPaddedStrideRejected:
    """A padded stride is a contract violation, not a case jlinalg_eigh_c services.

    Every production caller passes ldk == ldz == N. jlinalg_eigh_c used to carry
    a second code path for ldk != N or ldz != N with no exerciser anywhere in the
    tree; it is gone, replaced by one check that rejects the mismatch outright.
    """

    def test_padded_ldk_rejected(self) -> None:
        """ldk != N returns JLINALG_EXT_BAD_STRIDE without touching LAPACK."""
        n = 5
        rng = np.random.default_rng(7)
        K = _random_spd(n, rng)
        eigenvalues = np.empty(n, dtype=np.float64)
        eigenvectors = np.empty((n, n), dtype=np.float64)
        status = _EighStatus()

        fn = _load_jlinalg_eigh_c()
        ret = fn(
            n,
            _ptr(K),
            n + 1,  # ldk != N
            _ptr(eigenvalues),
            _ptr(eigenvectors),
            n,
            0,  # prefer_dsyevr
            ctypes.byref(status),
        )
        assert ret == _JLINALG_EXT_BAD_STRIDE

    def test_padded_ldz_rejected(self) -> None:
        """ldz != N returns JLINALG_EXT_BAD_STRIDE without touching LAPACK."""
        n = 5
        rng = np.random.default_rng(8)
        K = _random_spd(n, rng)
        eigenvalues = np.empty(n, dtype=np.float64)
        eigenvectors = np.empty((n, n), dtype=np.float64)
        status = _EighStatus()

        fn = _load_jlinalg_eigh_c()
        ret = fn(
            n,
            _ptr(K),
            n,
            _ptr(eigenvalues),
            _ptr(eigenvectors),
            n + 1,  # ldz != N
            0,  # prefer_dsyevr
            ctypes.byref(status),
        )
        assert ret == _JLINALG_EXT_BAD_STRIDE


# ---------------------------------------------------------------------------
# Numerically awkward inputs
# ---------------------------------------------------------------------------


class TestEighDifficultInputs:
    """eigh holds up on inputs chosen to be numerically hostile.

    Each case targets a different way an eigensolver degrades: eigenvalues too
    close to separate, a spectrum spanning fifteen orders of magnitude, an
    eigenvalue pushed well outside the diagonal's range, and a scale far from
    unity in both directions.  Random SPD matrices exhibit none of these, so
    without this class the whole family would go untested.

    All inputs are symmetric tridiagonal, which keeps them cheap to build and
    easy to reason about while still being fully general as far as ``eigh`` is
    concerned.
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
    def test_size_2_matrices(self, K: np.ndarray) -> None:
        """N=2 across four spectra: reconstruction and orthogonality to 1e-12.

        N=2 has a closed-form answer, so any backend should be accurate to
        near machine precision regardless of scale.  A failure here is a
        genuine defect and not a conditioning artefact, which is why the
        tolerance is far tighter than the large-N cases.
        """
        K_copy = K.copy()
        w, v, _ = eigh(K_copy)
        _assert_reconstruction(K, w, v, 1e-12, "N=2")
        _assert_orthogonality(v, 1e-12, "N=2")
        # Eigenvalues ascending
        assert w[0] <= w[1] + 1e-14, f"Eigenvalues not ascending: {w}"

    def test_clustered_eigenvalues(self) -> None:
        """Diagonal entries within 1e-10 of each other still reconstruct to 1e-12.

        Near-equal eigenvalues are the case where a solver is most likely to
        return a basis that is not orthogonal, because the eigenvectors it must
        separate are nearly parallel.
        """
        N = 50
        # Construct a tridiagonal matrix with clustered eigenvalues
        # Use near-constant diagonal with small off-diagonal perturbation
        d = np.ones(N) + np.arange(N) * 1e-10
        e = np.full(N - 1, 1e-12)
        K = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        K_copy = K.copy()

        w, v, _ = eigh(K_copy)
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

        w, v, _ = eigh(K_copy)
        _assert_reconstruction(K, w, v, 1e-10, "Large gap ratio")

    def test_boundary_eigenvalue(self) -> None:
        """Largest eigenvalue pushed well above the largest diagonal entry.

        A strong off-diagonal moves the top eigenvalue outside the range of
        the diagonal, so a solver that implicitly assumes eigenvalues bracket
        within the diagonal returns a wrong answer here.
        """
        N = 30
        d = np.arange(1.0, N + 1.0)
        e = np.full(N - 1, 2.0)
        K = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        K_copy = K.copy()

        w, v, _ = eigh(K_copy)
        _assert_reconstruction(K, w, v, 1e-12, "Boundary eigenvalue")

    def test_negative_offdiagonal(self) -> None:
        """A negative off-diagonal entry mid-matrix does not corrupt the result.

        Every other case here uses uniformly positive off-diagonals.  A sign
        flip partway along changes the matrix meaningfully while leaving it
        symmetric, and mishandling that sign historically produced residual
        errors in the percent range rather than a clean failure — the kind of
        wrongness a loose tolerance would wave through.
        """
        N = 130
        d = np.linspace(1.0, 10.0, N)
        e = np.full(N - 1, 0.5)
        # Flip the sign of one off-diagonal entry, halfway along.
        mid = N // 2
        e[mid - 1] = -0.5
        K = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        K_copy = K.copy()

        w, v, _ = eigh(K_copy)
        _assert_reconstruction(K, w, v, 1e-8, "negative off-diagonal")
        _assert_orthogonality(v, 1e-8, "negative off-diagonal")

    def test_non_psd_symmetric_at_scale(self) -> None:
        """Symmetric but indefinite matrices reconstruct at N=50, 200 and 500.

        Built by symmetrising a Gaussian matrix rather than forming A @ A.T, so
        unlike ``_random_spd`` these have negative eigenvalues and eigenvalues
        near zero.  Sweeping three sizes in one test keeps the three cases
        above free to stay size-specific.
        """
        rng = np.random.default_rng(42)
        for N in [50, 200, 500]:
            A = rng.standard_normal((N, N))
            A = (A + A.T) / 2
            w, V, _ = eigh(A.copy())

            recon = np.linalg.norm(A - V @ np.diag(w) @ V.T) / np.linalg.norm(A)
            assert recon < 1e-8, f"Reconstruction {recon:.2e} at N={N} (threshold 1e-8)"

            orth = np.linalg.norm(V.T @ V - np.eye(N))
            assert orth < 1e-8, f"Orthogonality {orth:.2e} at N={N} (threshold 1e-8)"


# ---------------------------------------------------------------------------
# Vendor dsyevd dispatch tests
# ---------------------------------------------------------------------------


class TestEighVendorDispatch:
    """eigh is correct whichever backend the installed BLAS causes it to pick."""

    def test_eigh_vendor_reconstruction_accuracy(self):
        """Reconstruction ||K - UDU.T||/||K|| on random SPD matrix.

        Tolerance is 1e-9 to accommodate OpenBLAS dsyevd, which is less
        accurate than Accelerate or MKL on some matrix structures.
        """
        rng = np.random.default_rng(42)
        A = rng.standard_normal((100, 100))
        K = np.ascontiguousarray(A @ A.T + np.eye(100), dtype=np.float64)
        K_orig = K.copy()
        w, v, _ = eigh(K)
        recon = v @ np.diag(w) @ v.T
        rel_err = np.linalg.norm(K_orig - recon) / np.linalg.norm(K_orig)
        assert rel_err < 1e-9, f"Reconstruction error {rel_err:.2e}"

    def test_eigh_vendor_orthogonality(self):
        """Orthogonality ||U.T U - I|| < 1e-13 on random SPD matrix."""
        rng = np.random.default_rng(123)
        A = rng.standard_normal((100, 100))
        K = np.ascontiguousarray(A @ A.T + np.eye(100), dtype=np.float64)
        w, v, _ = eigh(K)
        orth_err = np.linalg.norm(v.T @ v - np.eye(100))
        assert orth_err < 1e-13, f"Orthogonality error {orth_err:.2e}"

    def test_eigh_vendor_eigenvalue_ascending(self):
        """Eigenvalues are ascending."""
        rng = np.random.default_rng(456)
        A = rng.standard_normal((50, 50))
        K = np.ascontiguousarray(A @ A.T + np.eye(50), dtype=np.float64)
        w, v, _ = eigh(K)
        assert np.all(np.diff(w) >= 0), "Eigenvalues not ascending"

    def test_eigh_vendor_matches_numpy(self):
        """Eigenvalues match numpy.linalg.eigh within rtol=1e-12."""
        rng = np.random.default_rng(789)
        A = rng.standard_normal((50, 50))
        K = np.ascontiguousarray(A @ A.T + np.eye(50), dtype=np.float64)
        K_copy = K.copy()
        w_jlinalg, _, _ = eigh(K)
        w_numpy, _ = np.linalg.eigh(K_copy)
        npt.assert_allclose(w_jlinalg, w_numpy, rtol=1e-12)

    def test_eigh_vendor_degenerate_eigenvalues(self):
        """Stress test: block-diagonal with repeated eigenvalues.

        OpenBLAS dsyevd can fail to converge on highly degenerate matrices
        (returns info > 0).  jlinalg_eigh_c propagates that error rather than
        silently substituting another backend, so the test accepts either a
        correct decomposition or a raised error, and in the error case confirms
        numpy handles the same matrix — proving the input itself is valid.
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
            w, v, _ = eigh(K)
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
        w, v, _ = eigh(K)
        assert abs(w[0] - 3.14) < 1e-15
        assert abs(v[0, 0] - 1.0) < 1e-15


# ---------------------------------------------------------------------------
# LinAlgError contract — convergence failure raises LinAlgError
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
# DSYEVR fallback path tests
# ---------------------------------------------------------------------------


class TestEighDsyevrFallback:
    """eigh is correct whichever driver the C layer selected."""

    def test_eigh_produces_correct_result_regardless_of_dispatch(self):
        """eigh is correct on whichever path is active: DSYEVD, DSYEVR, or NumPy."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((80, 80))
        K = np.ascontiguousarray(A @ A.T + np.eye(80), dtype=np.float64)
        K_orig = K.copy()

        w, v, _ = eigh(K)

        # Reconstruction check — DSYEVR on Linux gives ~6e-9, so use 1e-8
        recon = v @ np.diag(w) @ v.T
        rel_err = np.linalg.norm(K_orig - recon) / np.linalg.norm(K_orig)
        assert rel_err < 1e-8, f"Reconstruction error {rel_err:.2e}"

        # Eigenvalues ascending
        assert np.all(np.diff(w) >= 0), "Eigenvalues not ascending"


# ---------------------------------------------------------------------------
# driver= plumbing: the memory plan's DSYEVR choice must reach jlinalg_eigh_c
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (HAS_C_EXTENSION and blas_has_dsyevr),
    reason="Vendor DSYEVR required to force the driver='dsyevr' path",
)
class TestEighDriverForcesDsyevr:
    """driver='dsyevr' skips DSYEVD and reports DSYEVR as the driver used.

    Guards the C1 fix: the memory plan picks DSYEVR for its smaller O(N)
    footprint, and jlinalg_eigh_c must run DSYEVR directly rather than
    silently retrying DSYEVD first and only falling back on an allocation
    failure the plan already ruled out.
    """

    def test_forced_dsyevr_reports_driver_used(self):
        rng = np.random.default_rng(2024)
        A = rng.standard_normal((64, 64))
        K = np.ascontiguousarray(A @ A.T + np.eye(64), dtype=np.float64)

        _, _, status = eigh(K.copy(), driver="dsyevr")

        assert status.driver_used == "dsyevr"

    def test_forced_dsyevr_matches_dsyevd_eigenvalues(self):
        """DSYEVR and DSYEVD agree on the same matrix within 1e-12 relative."""
        rng = np.random.default_rng(2024)
        A = rng.standard_normal((64, 64))
        K = np.ascontiguousarray(A @ A.T + np.eye(64), dtype=np.float64)

        w_dsyevr, _, status_dsyevr = eigh(K.copy(), driver="dsyevr")
        w_dsyevd, _, status_dsyevd = eigh(K.copy(), driver="dsyevd")

        assert status_dsyevr.driver_used == "dsyevr"
        assert status_dsyevd.driver_used == "dsyevd"
        npt.assert_allclose(w_dsyevr, w_dsyevd, rtol=1e-12)

    def test_default_driver_reports_dsyevd(self):
        """driver='auto' (the default) runs DSYEVD when it is available."""
        rng = np.random.default_rng(7)
        A = rng.standard_normal((64, 64))
        K = np.ascontiguousarray(A @ A.T + np.eye(64), dtype=np.float64)

        _, _, status = eigh(K)

        assert status.driver_used == "dsyevd"
