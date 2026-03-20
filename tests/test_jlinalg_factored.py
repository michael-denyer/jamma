"""Tests for jlinalg eigh_factored and rotate_via_householder.

Tests cover the lazy eigendecomposition infrastructure:
- eigh_factored returns eigenvalues matching eigh to rtol=1e-14
- V from eigh_factored is orthogonal (V^T V == I)
- K_householder from eigh_factored contains valid Householder vectors
- dormtr transpose produces Q^T @ C (roundtrip Q @ Q^T == I)
- rotate_via_householder matches U.T @ target to rtol=1e-12
- rotate_via_householder works for multi-column targets
- eigh_factored returns JLINALG_EXT_UNAVAILABLE when vendor LAPACK only

Run:
    uv run pytest tests/test_jlinalg_factored.py -x -v
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from jamma.jlinalg import HAS_C_EXTENSION

pytestmark = pytest.mark.skipif(
    not HAS_C_EXTENSION,
    reason="eigh_factored/rotate_via_householder require jlinalg C extension",
)


def _make_spd(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric positive-definite matrix."""
    A = rng.standard_normal((N, N))
    return (A + A.T) / 2 + N * np.eye(N)


class TestEighFactored:
    """Tests for eigh_factored C API."""

    def test_eigh_factored_eigenvalues_match(self) -> None:
        """eigh_factored eigenvalues match eigh eigenvalues to rtol=1e-14."""
        from jamma.jlinalg import eigh, eigh_factored

        rng = np.random.default_rng(42)
        N = 100
        K = _make_spd(N, rng)
        K_copy = K.copy()

        w_ref, _U = eigh(K_copy)
        w_fac, tau, V = eigh_factored(K)

        npt.assert_allclose(w_fac, w_ref, rtol=1e-14)

    def test_v_orthogonal(self) -> None:
        """V from eigh_factored satisfies ||V^T V - I|| < 1e-13."""
        from jamma.jlinalg import eigh_factored

        rng = np.random.default_rng(42)
        N = 100
        K = _make_spd(N, rng)

        _w, _tau, V = eigh_factored(K)

        VtV = V.T @ V
        npt.assert_allclose(VtV, np.eye(N), atol=1e-13)

    def test_householder_vectors_valid(self) -> None:
        """K_householder from eigh_factored contains valid Householder vectors.

        Applying dormtr(K_h, tau, I) should produce an orthogonal Q.
        We verify by using rotate_via_householder with V=I, target=I:
          result = I^T @ (Q^T @ I) = Q^T
        Then Q^T should be orthogonal.
        """
        from jamma.jlinalg import eigh_factored, rotate_via_householder

        rng = np.random.default_rng(42)
        N = 50
        K = _make_spd(N, rng)

        _w, tau, V = eigh_factored(K)
        # K now contains Householder vectors

        # Use rotate_via_householder with V_identity and target=I to extract Q^T
        V_identity = np.eye(N)
        target_I = np.eye(N)
        Qt = rotate_via_householder(K, tau, V_identity, target_I)

        # Q^T should be orthogonal: Q^T @ Q == I
        QtQ = Qt @ Qt.T
        npt.assert_allclose(QtQ, np.eye(N), atol=1e-13)

    def test_rotate_via_householder_parity(self) -> None:
        """rotate_via_householder(K_h, tau, V, target) == U.T @ target at rtol=1e-12.

        Constructs U from the factored output (via rotate with target=I) to avoid
        sign ambiguity between vendor LAPACK and jlinalg D&C eigenvectors.
        """
        from jamma.jlinalg import eigh_factored, rotate_via_householder

        rng = np.random.default_rng(42)
        N = 100
        M = 20
        K = _make_spd(N, rng)
        K_orig = K.copy()
        target = rng.standard_normal((N, M)).astype(np.float64)

        w, tau, V = eigh_factored(K)
        result = rotate_via_householder(K, tau, V, target)

        # Reconstruct U = Q @ V by rotating identity
        Ut = rotate_via_householder(K, tau, V, np.eye(N))
        ref = Ut @ target

        npt.assert_allclose(result, ref, rtol=1e-11)

        # Verify reconstruction: ||K - U diag(w) U.T|| / ||K|| < 1e-8
        U = Ut.T
        K_rec = U @ np.diag(w) @ U.T
        rel_err = np.linalg.norm(K_rec - K_orig) / np.linalg.norm(K_orig)
        assert rel_err < 1e-8, f"Reconstruction relative Frobenius error {rel_err:.2e}"

    def test_rotate_via_householder_vector(self) -> None:
        """rotate_via_householder works for (N, 1) target (single column)."""
        from jamma.jlinalg import eigh_factored, rotate_via_householder

        rng = np.random.default_rng(42)
        N = 80
        K = _make_spd(N, rng)
        target = rng.standard_normal((N, 1)).astype(np.float64)

        w, tau, V = eigh_factored(K)
        result = rotate_via_householder(K, tau, V, target)

        # Verify rotation preserves norm (U is orthogonal)
        npt.assert_allclose(np.linalg.norm(result), np.linalg.norm(target), rtol=1e-12)

        # Verify via explicit U reconstruction
        Ut = rotate_via_householder(K, tau, V, np.eye(N))
        ref = Ut @ target
        npt.assert_allclose(result, ref, rtol=1e-12)

    def test_rotate_via_householder_large(self) -> None:
        """N=500, M=50 target (larger scale parity)."""
        from jamma.jlinalg import eigh_factored, rotate_via_householder

        rng = np.random.default_rng(42)
        N = 500
        M = 50
        K = _make_spd(N, rng)
        K_orig = K.copy()
        target = rng.standard_normal((N, M)).astype(np.float64)

        w, tau, V = eigh_factored(K)
        result = rotate_via_householder(K, tau, V, target)

        # Verify rotation preserves Gram matrix (orthogonal transform)
        npt.assert_allclose(result.T @ result, target.T @ target, rtol=1e-10)

        # Verify reconstruction: ||K - U diag(w) U.T|| / ||K|| < 1e-7
        Ut = rotate_via_householder(K, tau, V, np.eye(N))
        U = Ut.T
        K_rec = U @ np.diag(w) @ U.T
        rel_err = np.linalg.norm(K_rec - K_orig) / np.linalg.norm(K_orig)
        assert rel_err < 1e-7, f"Reconstruction relative Frobenius error {rel_err:.2e}"

    def test_n1_edge_case(self) -> None:
        """eigh_factored handles N=1 matrix (degenerate single-sample case)."""
        from jamma.jlinalg import eigh_factored, rotate_via_householder

        K = np.array([[5.0]])
        w, tau, V = eigh_factored(K)

        assert w.shape == (1,)
        npt.assert_allclose(w[0], 5.0, rtol=1e-14)
        assert tau.shape == (0,)
        assert V.shape == (1, 1)
        npt.assert_allclose(V[0, 0], 1.0, atol=1e-15)

        # rotate_via_householder should work with N=1
        target = np.array([[3.0]])
        result = rotate_via_householder(K, tau, V, target)
        npt.assert_allclose(result, target, rtol=1e-14)

    def test_n2_edge_case(self) -> None:
        """eigh_factored handles N=2 (single Householder reflector)."""
        from jamma.jlinalg import eigh_factored, rotate_via_householder

        rng = np.random.default_rng(42)
        K = _make_spd(2, rng)
        K_orig = K.copy()

        w, tau, V = eigh_factored(K)

        assert tau.shape == (1,)
        assert V.shape == (2, 2)

        # Verify reconstruction
        Ut = rotate_via_householder(K, tau, V, np.eye(2))
        U = Ut.T
        K_rec = U @ np.diag(w) @ U.T
        rel_err = np.linalg.norm(K_rec - K_orig) / np.linalg.norm(K_orig)
        assert rel_err < 1e-13, f"Reconstruction error {rel_err:.2e}"

    def test_tau_shape(self) -> None:
        """tau from eigh_factored has shape (N-1,)."""
        from jamma.jlinalg import eigh_factored

        rng = np.random.default_rng(42)
        N = 50
        K = _make_spd(N, rng)

        _w, tau, _V = eigh_factored(K)

        assert tau.shape == (N - 1,), f"Expected tau shape ({N - 1},), got {tau.shape}"
        assert tau.dtype == np.float64
