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

_RNG = np.random.default_rng(42)

_EXPECTED_ABI = 1  # Must match ABI_VERSION in _secular_accel.c


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
    n = 8
    d = np.sort(_RNG.uniform(0.1, 5.0, n))
    z = _RNG.standard_normal(n)
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
    n = 6
    # Wider spread to avoid degenerate case
    d = np.array([1.0, 2.0, 3.5, 5.0, 7.0, 10.0])
    z = _RNG.standard_normal(n)
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
    n = 5
    # Two pairs of near-equal elements
    d = np.array([1.0, 1.0 + 1e-10, 3.0, 3.0 + 1e-10, 6.0])
    z = _RNG.standard_normal(n)
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
    n = 20
    d = np.sort(_RNG.uniform(0.5, 20.0, n))
    z = _RNG.standard_normal(n)
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
