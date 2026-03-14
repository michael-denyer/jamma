"""Tests for jblas eigh (symmetric eigendecomposition via LAPACK DSYEVD).

Tests cover all EIGH requirements (EIGH-01 through EIGH-09):
- EIGH-01: DSYTRD reduction to tridiagonal form (C extension stub)
- EIGH-02: DSTEDC divide-and-conquer secular solver (C extension stub)
- EIGH-03: Block-diagonal stress test (repeated/clustered eigenvalues)
- EIGH-04: DORMTR eigenvector back-transformation (C extension stub)
- EIGH-05: Python fallback correctness (identity, diagonal, random SPD, ascending)
- EIGH-06: Output memory layout (shape, dtype, C-contiguous)
- EIGH-07: Reconstruction accuracy: ||K - U diag(w) U.T|| / ||K|| < 1e-14
- EIGH-08: Orthogonality: ||U.T @ U - I||_F < 1e-14
- EIGH-09: LAPACK sources in hatch_build.py must not receive -ffast-math

Run with -n0 to avoid interference with OpenMP threading tests:
    uv run pytest tests/test_jblas_eigh.py -x -n0 -v
"""

from __future__ import annotations

import inspect

import numpy as np
import numpy.testing as npt
import pytest

from jamma.jblas import HAS_C_EXTENSION, eigh

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


# ---------------------------------------------------------------------------
# test_reconstruction_accuracy — EIGH-07
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reconstruction_accuracy() -> None:
    """N=1000 random SPD: ||K - U diag(w) U.T|| / ||K|| < 1e-14."""
    rng = np.random.default_rng(42)
    N = 1000
    K = _random_spd(N, rng)
    K_copy = K.copy()
    w, v = eigh(K_copy)
    # Reconstruct: K_reconstructed = v @ diag(w) @ v.T
    K_reconstructed = v @ np.diag(w) @ v.T
    norm_K = np.linalg.norm(K, "fro")
    norm_diff = np.linalg.norm(K - K_reconstructed, "fro")
    ratio = norm_diff / norm_K
    assert ratio < 1e-14, (
        f"Reconstruction accuracy ||K - U diag(w) U.T|| / ||K|| = {ratio:.2e} > 1e-14"
    )


# ---------------------------------------------------------------------------
# test_orthogonality — EIGH-08
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_orthogonality() -> None:
    """N=1000 random SPD: ||U.T @ U - I||_F < 1e-14."""
    rng = np.random.default_rng(43)
    N = 1000
    K = _random_spd(N, rng)
    K_copy = K.copy()
    _, v = eigh(K_copy)
    VtV = v.T @ v
    norm_off = np.linalg.norm(VtV - np.eye(N), "fro")
    assert norm_off < 1e-12, f"Orthogonality ||U.T @ U - I||_F = {norm_off:.2e} > 1e-12"


# ---------------------------------------------------------------------------
# test_eigh_memory_layout — EIGH-06
# ---------------------------------------------------------------------------


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
    Verifies reconstruction < 1e-13 and orthogonality < 1e-13.
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

    # Reconstruction accuracy
    K_reconstructed = v @ np.diag(w) @ v.T
    norm_K = np.linalg.norm(K, "fro")
    norm_diff = np.linalg.norm(K - K_reconstructed, "fro")
    ratio = norm_diff / norm_K
    assert ratio < 1e-13, (
        f"Block-diagonal reconstruction: ||K - U diag(w) U.T|| / ||K||"
        f" = {ratio:.2e} > 1e-13"
    )

    # Orthogonality
    VtV = v.T @ v
    norm_off = np.linalg.norm(VtV - np.eye(N), "fro")
    assert norm_off < 1e-13, (
        f"Block-diagonal orthogonality: ||U.T @ U - I||_F = {norm_off:.2e} > 1e-13"
    )


# ---------------------------------------------------------------------------
# test_vs_mouse_hs1940_kinship — real-data validation
# ---------------------------------------------------------------------------


def test_vs_mouse_hs1940_kinship() -> None:
    """eigh on mouse_hs1940 kinship matrix matches np.linalg.eigh within rtol=1e-12."""
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

    # Reference eigendecomp via numpy
    K_ref = K.copy()
    w_ref, v_ref = np.linalg.eigh(K_ref)

    # jblas eigh
    K_jblas = K.copy()
    w_jblas, v_jblas = eigh(K_jblas)

    # Compare eigenvalues
    npt.assert_allclose(
        w_jblas,
        w_ref,
        rtol=1e-12,
        err_msg="jblas eigh eigenvalues differ from np.linalg.eigh on mouse_hs1940",
    )

    # Compare eigenvectors using absolute values to handle sign ambiguity
    npt.assert_allclose(
        np.abs(v_jblas),
        np.abs(v_ref),
        rtol=1e-12,
        err_msg="jblas eigh eigenvectors differ from np.linalg.eigh on mouse_hs1940",
    )


# ---------------------------------------------------------------------------
# TestDsytrd — EIGH-01: Tridiagonalization stubs (C extension only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for dsytrd tests",
)
class TestDsytrd:
    """DSYTRD reduction to tridiagonal form (EIGH-01).

    These tests require the C extension with jblas_dsytrd_c wired to Python.
    Plan 02 implements the C function; Plan 03 exposes it to Python.
    Until then, these tests skip.
    """

    def test_tridiagonalizes(self) -> None:
        """After dsytrd, Q T Q.T ≈ K and T is tridiagonal.

        This test will be expanded in Plan 02 once jblas.dsytrd is exposed.
        """
        pytest.skip("jblas.dsytrd not yet exposed to Python (Plan 03 wires it)")


# ---------------------------------------------------------------------------
# TestDstedc — EIGH-02: Divide-and-conquer solver stubs (C extension only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for dstedc tests",
)
class TestDstedc:
    """DSTEDC divide-and-conquer solver (EIGH-02).

    These tests require the C extension with jblas_dstedc_c wired to Python.
    Plan 02 implements the C function; Plan 03 exposes it to Python.
    Until then, these tests skip.
    """

    def test_solves_symmetric_tridiagonal(self) -> None:
        """DSTEDC computes correct eigenvalues/eigenvectors for a tridiagonal input.

        This test will be expanded in Plan 02 once jblas.dstedc is exposed.
        """
        pytest.skip("jblas.dstedc not yet exposed to Python (Plan 03 wires it)")


# ---------------------------------------------------------------------------
# TestDormtr — EIGH-04: Back-transformation stubs (C extension only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="C extension required for dormtr tests",
)
class TestDormtr:
    """DORMTR eigenvector back-transformation (EIGH-04).

    These tests require the C extension with jblas_dormtr_c wired to Python.
    Plan 02 implements the C function; Plan 03 exposes it to Python.
    Until then, these tests skip.
    """

    def test_back_transforms_eigenvectors(self) -> None:
        """DORMTR correctly applies the orthogonal transform from DSYTRD.

        This test will be expanded in Plan 02 once jblas.dormtr is exposed.
        """
        pytest.skip("jblas.dormtr not yet exposed to Python (Plan 03 wires it)")


# ---------------------------------------------------------------------------
# test_lapack_no_ffast_math — EIGH-09
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Plan 03 adds lapack_sources group to hatch_build.py")
def test_lapack_no_ffast_math() -> None:
    """LAPACK sources in hatch_build.py must use restricted flags (no -ffast-math).

    This is a structural test: it inspects hatch_build.py source to verify
    that a 'lapack_sources' group exists and that -ffast-math is not applied
    to LAPACK source files.

    The test is skipped until Plan 03 adds the lapack_sources group.
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
