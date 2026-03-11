"""Tests for _secular_accel C extension (rank-1 secular equation solver).

Tests cover:
  1. Import/ABI check
  2. Positive rho rank-1 update
  3. Negative rho rank-1 downdate (critical for LOCO)
  4. Near-degenerate d (deflation)
  5. Eigenvector orthogonality
  6. Automatic z normalization
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Extension availability
# ---------------------------------------------------------------------------

try:
    import jamma.lmm._secular_accel as _ext

    _EXT_AVAILABLE = True
    _EXT_ERROR = None
except ImportError as _e:
    _ext = None  # type: ignore[assignment]
    _EXT_AVAILABLE = False
    _EXT_ERROR = str(_e)

_skip_no_ext = pytest.mark.skipif(
    not _EXT_AVAILABLE,
    reason=f"_secular_accel not compiled: {_EXT_ERROR}",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)  # Legacy — avoid in new tests; use per-test RNG

_EXPECTED_ABI = 2  # Must match ABI_VERSION in _secular_accel.c


def _make_rank1_matrix(d: np.ndarray, rho: float, z: np.ndarray) -> np.ndarray:
    """Build the full matrix D + rho * z @ z.T for reference eigendecomposition."""
    return np.diag(d) + rho * np.outer(z, z)


def _eigh_reference(
    d: np.ndarray, rho: float, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reference eigendecomposition via np.linalg.eigh."""
    M = _make_rank1_matrix(d, rho, z)
    vals, vecs = np.linalg.eigh(M)
    return vals, vecs


# ---------------------------------------------------------------------------
# Test 1: Import and ABI version
# ---------------------------------------------------------------------------


def test_secular_accel_import_and_abi():
    """Extension imports (or is clearly unavailable) and ABI_VERSION is correct."""
    if not _EXT_AVAILABLE:
        pytest.skip(f"_secular_accel not compiled: {_EXT_ERROR}")

    assert hasattr(_ext, "ABI_VERSION"), "_secular_accel missing ABI_VERSION"
    assert _ext.ABI_VERSION == _EXPECTED_ABI, (
        f"ABI mismatch: got {_ext.ABI_VERSION}, expected {_EXPECTED_ABI}"
    )
    assert hasattr(_ext, "IS_ILP64"), "_secular_accel missing IS_ILP64"
    assert isinstance(_ext.IS_ILP64, int)
    assert hasattr(_ext, "rank1_eigenvalue_update"), (
        "_secular_accel missing rank1_eigenvalue_update"
    )


# ---------------------------------------------------------------------------
# Test 2: Positive rho
# ---------------------------------------------------------------------------


@_skip_no_ext
def test_rank1_update_positive_rho():
    """rank1_eigenvalue_update with positive rho matches np.linalg.eigh (rtol=1e-12)."""
    rng = np.random.default_rng(100)
    n = 8
    d = np.sort(rng.uniform(0.1, 5.0, n))
    z = rng.standard_normal(n)
    z = z / np.linalg.norm(z)  # unit norm
    rho = 0.5

    vals, vecs = _ext.rank1_eigenvalue_update(d, rho, z)
    ref_vals, ref_vecs = _eigh_reference(d, rho, z)

    assert vals.shape == (n,), f"eigenvalue shape: {vals.shape}"
    assert vecs.shape == (n, n), f"eigenvector shape: {vecs.shape}"

    # Eigenvalues must match (ascending order guaranteed by dlaed4)
    np.testing.assert_allclose(
        vals,
        ref_vals,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Eigenvalues mismatch (positive rho)",
    )

    # Eigenvector orthogonality: V^T V = I
    orth = vecs.T @ vecs
    np.testing.assert_allclose(
        orth,
        np.eye(n),
        atol=1e-10,
        err_msg="Eigenvectors not orthogonal (positive rho)",
    )


# ---------------------------------------------------------------------------
# Test 3: Negative rho (LOCO downdate case)
# ---------------------------------------------------------------------------


@_skip_no_ext
def test_rank1_update_negative_rho():
    """rank1_eigenvalue_update with negative rho (downdate) matches np.linalg.eigh."""
    rng = np.random.default_rng(101)
    n = 6
    # Wider spread to avoid degenerate case
    d = np.array([1.0, 2.0, 3.5, 5.0, 7.0, 10.0])
    z = rng.standard_normal(n)
    z = z / np.linalg.norm(z)
    rho = -0.3

    vals, vecs = _ext.rank1_eigenvalue_update(d, rho, z)
    ref_vals, ref_vecs = _eigh_reference(d, rho, z)

    # Eigenvalues match
    np.testing.assert_allclose(
        vals,
        ref_vals,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Eigenvalues mismatch (negative rho / downdate)",
    )

    # Orthogonality
    orth = vecs.T @ vecs
    np.testing.assert_allclose(
        orth,
        np.eye(n),
        atol=1e-10,
        err_msg="Eigenvectors not orthogonal (negative rho)",
    )


# ---------------------------------------------------------------------------
# Test 4: Near-degenerate d (deflation path in dlaed4)
# ---------------------------------------------------------------------------


@_skip_no_ext
def test_rank1_update_near_degenerate_d():
    """dlaed4 handles near-equal d values without info != 0 (deflation)."""
    rng = np.random.default_rng(102)
    n = 5
    # Two pairs of near-equal elements
    d = np.array([1.0, 1.0 + 1e-10, 3.0, 3.0 + 1e-10, 6.0])
    z = rng.standard_normal(n)
    z = z / np.linalg.norm(z)
    rho = 0.8

    # Should not raise
    vals, vecs = _ext.rank1_eigenvalue_update(d, rho, z)
    ref_vals, _ = _eigh_reference(d, rho, z)

    assert vals.shape == (n,)
    # Near-degenerate case: relax tolerance slightly (deflation may slightly shift)
    np.testing.assert_allclose(
        vals,
        ref_vals,
        rtol=1e-10,
        atol=1e-12,
        err_msg="Eigenvalues mismatch (near-degenerate d)",
    )

    # Orthogonality must hold regardless
    orth = vecs.T @ vecs
    np.testing.assert_allclose(
        orth,
        np.eye(n),
        atol=1e-8,
        err_msg="Eigenvectors not orthogonal (near-degenerate d)",
    )


# ---------------------------------------------------------------------------
# Test 5: Eigenvector orthogonality (standalone, larger n)
# ---------------------------------------------------------------------------


@_skip_no_ext
def test_eigenvectors_orthogonal_larger():
    """Eigenvectors satisfy V^T V = I within 1e-10 for n=20."""
    rng = np.random.default_rng(103)
    n = 20
    d = np.sort(rng.uniform(0.5, 20.0, n))
    z = rng.standard_normal(n)
    z = z / np.linalg.norm(z)
    rho = 1.5

    vals, vecs = _ext.rank1_eigenvalue_update(d, rho, z)

    # Orthogonality
    orth = vecs.T @ vecs
    np.testing.assert_allclose(
        orth, np.eye(n), atol=1e-10, err_msg="Eigenvectors V^T V != I for n=20"
    )

    # Also verify eigenvalue equation: M @ v_j ≈ lambda_j * v_j for each column
    M = _make_rank1_matrix(d, rho, z)
    for j in range(n):
        lhs = M @ vecs[:, j]
        rhs = vals[j] * vecs[:, j]
        np.testing.assert_allclose(
            lhs, rhs, atol=1e-9, err_msg=f"Eigenvalue equation fails for j={j}"
        )


# ---------------------------------------------------------------------------
# Test 6: z normalization — non-unit z is handled internally
# ---------------------------------------------------------------------------


@_skip_no_ext
def test_rank1_update_z_normalization():
    """Non-unit-norm z is normalized internally; rho is adjusted accordingly."""
    n = 5
    d = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    z_raw = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
    rho_effective = 0.4

    # The extension receives z_raw and rho_effective.
    # Internally: z_unit = z_raw / ||z_raw||, rho_internal = rho_effective * ||z_raw||^2
    # Result should match np.linalg.eigh(D + rho_effective * z_raw @ z_raw.T)
    ref_vals, _ = _eigh_reference(d, rho_effective, z_raw)

    vals, vecs = _ext.rank1_eigenvalue_update(d, rho_effective, z_raw)

    np.testing.assert_allclose(
        vals,
        ref_vals,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Eigenvalues mismatch when z is not unit norm",
    )

    # Orthogonality
    orth = vecs.T @ vecs
    np.testing.assert_allclose(
        orth, np.eye(n), atol=1e-10, err_msg="Eigenvectors not orthogonal (non-unit z)"
    )


# ---------------------------------------------------------------------------
# Tests for rank1_eigenvalues_and_norms (new function, Plan 69.3-01)
# ---------------------------------------------------------------------------


@_skip_no_ext
def test_rank1_eigenvalues_and_norms_positive_rho():
    """rank1_eigenvalues_and_norms eigenvalues match rank1_eigenvalue_update."""
    rng = np.random.default_rng(100)
    n = 8
    d = np.sort(rng.uniform(0.1, 5.0, n))
    z = rng.standard_normal(n)
    z = z / np.linalg.norm(z)
    rho = 0.5

    assert hasattr(_ext, "rank1_eigenvalues_and_norms"), (
        "_secular_accel missing rank1_eigenvalues_and_norms"
    )

    vals_norms, norms = _ext.rank1_eigenvalues_and_norms(d, rho, z)
    vals_ref, _ = _ext.rank1_eigenvalue_update(d, rho, z)

    assert vals_norms.shape == (n,), f"eigenvalues shape: {vals_norms.shape}"
    assert norms.shape == (n,), f"norms shape: {norms.shape}"

    np.testing.assert_allclose(
        vals_norms,
        vals_ref,
        rtol=1e-12,
        atol=1e-14,
        err_msg=(
            "Eigenvalues mismatch between rank1_eigenvalues_and_norms"
            " and rank1_eigenvalue_update (positive rho)"
        ),
    )

    assert np.all(norms > 0), "All norms must be positive"
    assert np.all(np.isfinite(norms)), "All norms must be finite"


@_skip_no_ext
def test_rank1_eigenvalues_and_norms_negative_rho():
    """rank1_eigenvalues_and_norms works for negative rho (LOCO downdate)."""
    rng = np.random.default_rng(200)
    n = 6
    d = np.array([1.0, 2.0, 3.5, 5.0, 7.0, 10.0])
    z = rng.standard_normal(n)
    z = z / np.linalg.norm(z)
    rho = -0.3

    vals_norms, norms = _ext.rank1_eigenvalues_and_norms(d, rho, z)
    vals_ref, _ = _ext.rank1_eigenvalue_update(d, rho, z)

    assert vals_norms.shape == (n,)
    assert norms.shape == (n,)

    np.testing.assert_allclose(
        vals_norms,
        vals_ref,
        rtol=1e-12,
        atol=1e-14,
        err_msg="Eigenvalues mismatch for negative rho",
    )

    assert np.all(np.isfinite(norms)), "All norms must be finite (negative rho)"
    assert np.all(norms > 0), "All norms must be positive (negative rho)"


@_skip_no_ext
def test_rank1_eigenvalues_and_norms_norm_reconstruction():
    """Norms from rank1_eigenvalues_and_norms match ||z_unit / delta_k||_2."""
    rng = np.random.default_rng(300)
    n = 10
    d = np.sort(rng.uniform(0.5, 10.0, n))
    z = rng.standard_normal(n)
    z = z / np.linalg.norm(z)
    rho = 1.2

    vals_new, norms = _ext.rank1_eigenvalues_and_norms(d, rho, z)
    vals_old, vecs = _ext.rank1_eigenvalue_update(d, rho, z)

    # Verify eigenvalues agree (sanity check)
    np.testing.assert_allclose(vals_new, vals_old, rtol=1e-12, atol=1e-14)

    # Each eigenvector column from rank1_eigenvalue_update must be unit norm
    for j in range(n):
        col_norm = np.linalg.norm(vecs[:, j])
        np.testing.assert_allclose(
            col_norm,
            1.0,
            atol=1e-10,
            err_msg=f"Column {j} of eigenvectors not unit norm",  # noqa: E501
        )

    # The norms returned by rank1_eigenvalues_and_norms should be positive and finite.
    # Verify that for each eigenvalue k: norms[k] = ||z_unit / delta_k||_2
    # where delta_k = d - vals[k] (the secular equation denominator).
    z_unit = z / np.linalg.norm(z)  # already unit norm, but be explicit
    for k in range(n):
        delta_k = d - vals_new[k]
        expected_norm = np.linalg.norm(z_unit / delta_k)
        np.testing.assert_allclose(
            norms[k],
            expected_norm,
            rtol=1e-10,
            err_msg=f"norm[{k}] = {norms[k]:.6e} != expected {expected_norm:.6e}",
        )


@_skip_no_ext
def test_rank1_eigenvalues_and_norms_deflation():
    """Deflation (near-equal d values) produces finite norms without inf/NaN."""
    rng = np.random.default_rng(400)
    n = 5
    d = np.array([1.0, 1.0 + 1e-10, 3.0, 3.0 + 1e-10, 6.0])
    z = rng.standard_normal(n)
    z = z / np.linalg.norm(z)
    rho = 0.8

    vals, norms = _ext.rank1_eigenvalues_and_norms(d, rho, z)
    ref_vals, _ = _eigh_reference(d, rho, z)

    assert np.all(np.isfinite(norms)), (
        f"Norms contain inf/NaN in deflation case: {norms}"
    )
    assert np.all(norms > 0), f"Norms not all positive in deflation case: {norms}"

    np.testing.assert_allclose(
        vals,
        ref_vals,
        rtol=1e-10,
        atol=1e-12,
        err_msg="Eigenvalues mismatch (near-degenerate d)",
    )


@_skip_no_ext
def test_rank1_eigenvalues_and_norms_z_normalization():
    """Non-unit z is normalized internally by rank1_eigenvalues_and_norms."""
    d = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    z_raw = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
    rho = 0.4

    ref_vals, _ = _eigh_reference(d, rho, z_raw)
    vals, norms = _ext.rank1_eigenvalues_and_norms(d, rho, z_raw)

    np.testing.assert_allclose(
        vals,
        ref_vals,
        rtol=1e-12,
        atol=1e-14,
        err_msg=(
            "Eigenvalues mismatch when z is not unit norm (rank1_eigenvalues_and_norms)"
        ),
    )

    assert np.all(np.isfinite(norms)), "Norms must be finite for non-unit z"
    assert np.all(norms > 0), "Norms must be positive for non-unit z"


# ---------------------------------------------------------------------------
# C extension edge cases: input validation & mutation safety
# ---------------------------------------------------------------------------


@_skip_no_ext
class TestCExtensionEdgeCases:
    """Tests for C extension input validation and edge cases."""

    def test_rank1_update_wrong_dtype_int32(self) -> None:
        """C extension rejects int32 arrays."""
        d = np.array([1.0, 2.0, 3.0], dtype=np.int32)
        z = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        with pytest.raises(TypeError):
            _ext.rank1_eigenvalue_update(d, 0.5, z)

    def test_rank1_update_wrong_dtype_float32(self) -> None:
        """C extension rejects float32 arrays."""
        d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        z = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        with pytest.raises(TypeError):
            _ext.rank1_eigenvalue_update(d, 0.5, z)

    def test_rank1_update_2d_array(self) -> None:
        """C extension rejects 2-D arrays."""
        d = np.array([[1.0, 2.0], [3.0, 4.0]])
        z = np.array([1.0, 1.0])
        with pytest.raises(ValueError):
            _ext.rank1_eigenvalue_update(d, 0.5, z)

    def test_rank1_update_size_mismatch(self) -> None:
        """C extension rejects mismatched array lengths."""
        d = np.array([1.0, 2.0, 3.0])
        z = np.array([1.0, 1.0])
        with pytest.raises(ValueError):
            _ext.rank1_eigenvalue_update(d, 0.5, z)

    def test_rank1_update_n1(self) -> None:
        """C extension handles n=1 (degenerate secular equation)."""
        d = np.array([2.0])
        z = np.array([1.0])
        rho = 0.5
        vals, vecs = _ext.rank1_eigenvalue_update(d, rho, z)
        expected = d[0] + rho  # D + rho*z*z^T = [[2.5]]
        np.testing.assert_allclose(vals, [expected], rtol=1e-14)
        assert vecs.shape == (1, 1)

    def test_rank1_eigs_norms_wrong_dtype(self) -> None:
        """rank1_eigenvalues_and_norms rejects non-float64 arrays."""
        d = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        z = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        with pytest.raises(TypeError):
            _ext.rank1_eigenvalues_and_norms(d, 0.5, z)

    def test_rank1_eigs_norms_n1(self) -> None:
        """rank1_eigenvalues_and_norms handles n=1."""
        d = np.array([2.0])
        z = np.array([1.0])
        rho = 0.5
        vals, norms = _ext.rank1_eigenvalues_and_norms(d, rho, z)
        expected = d[0] + rho
        np.testing.assert_allclose(vals, [expected], rtol=1e-14)
        assert len(norms) == 1
        assert np.isfinite(norms[0])

    def test_rank1_update_rho_zero(self) -> None:
        """rho=0 returns eigenvalues=d and eigenvectors=I."""
        d = np.array([1.0, 2.0, 3.0, 4.0])
        z = np.array([0.5, 0.5, 0.5, 0.5])
        vals, vecs = _ext.rank1_eigenvalue_update(d, 0.0, z)
        np.testing.assert_allclose(vals, d, rtol=1e-14)
        np.testing.assert_allclose(vecs, np.eye(4), atol=1e-14)

    def test_rank1_eigs_norms_rho_zero(self) -> None:
        """rho=0 returns eigenvalues=d and norms=0."""
        d = np.array([1.0, 2.0, 3.0, 4.0])
        z = np.array([0.5, 0.5, 0.5, 0.5])
        vals, norms = _ext.rank1_eigenvalues_and_norms(d, 0.0, z)
        np.testing.assert_allclose(vals, d, rtol=1e-14)
        np.testing.assert_allclose(norms, 0.0, atol=1e-14)

    def test_rank1_update_z_zero(self) -> None:
        """z=0 vector returns eigenvalues=d and eigenvectors=I."""
        d = np.array([1.0, 2.0, 3.0])
        z = np.zeros(3)
        vals, vecs = _ext.rank1_eigenvalue_update(d, 1.0, z)
        np.testing.assert_allclose(vals, d, rtol=1e-14)
        np.testing.assert_allclose(vecs, np.eye(3), atol=1e-14)

    def test_rank1_update_does_not_mutate_z(self) -> None:
        """C extension must not modify caller's z array."""
        rng = np.random.default_rng(42)
        d = np.sort(rng.random(10))
        z = rng.standard_normal(10)
        z_copy = z.copy()
        _ext.rank1_eigenvalue_update(d, 0.5, z)
        np.testing.assert_array_equal(z, z_copy, err_msg="z was mutated")

    def test_rank1_eigs_norms_does_not_mutate_z(self) -> None:
        """rank1_eigenvalues_and_norms must not modify caller's z array."""
        rng = np.random.default_rng(42)
        d = np.sort(rng.random(10))
        z = rng.standard_normal(10)
        z_copy = z.copy()
        _ext.rank1_eigenvalues_and_norms(d, 0.5, z)
        np.testing.assert_array_equal(z, z_copy, err_msg="z was mutated")

    def test_rank1_update_non_contiguous(self) -> None:
        """C extension handles non-contiguous arrays (e.g., sliced)."""
        rng = np.random.default_rng(42)
        d_full = np.sort(rng.random(20))
        d = d_full[::2]  # non-contiguous stride-2 view
        z = rng.standard_normal(20)[::2]
        assert not d.flags["C_CONTIGUOUS"]
        # Should not crash — PyArray_ContiguousFromAny handles the copy
        vals, vecs = _ext.rank1_eigenvalue_update(d, 0.5, z)
        assert len(vals) == len(d)
        assert vecs.shape == (len(d), len(d))
