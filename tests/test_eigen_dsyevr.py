"""Tests for DSYEVR eigendecomposition via C extension."""

from unittest.mock import patch

import numpy as np
import pytest

# Skip entire module if C extension not available
dsyevr_available = False
try:
    from jamma.lmm._eigen_accel import eigh_dsyevr

    dsyevr_available = True
except ImportError:
    pass

pytestmark = [
    pytest.mark.tier0,
    pytest.mark.skipif(not dsyevr_available, reason="DSYEVR C extension not compiled"),
]


@pytest.mark.tier0
class TestDsyevrCorrectness:
    """Eigenvalue/eigenvector accuracy against np.linalg.eigh (DSYEVD)."""

    def test_identity_matrix(self):
        """eigh_dsyevr(eye(n)) should return all eigenvalues == 1.0."""
        n = 50
        K = np.eye(n, dtype=np.float64)
        w, v = eigh_dsyevr(K)
        assert w.shape == (n,)
        assert v.shape == (n, n)
        np.testing.assert_allclose(w, np.ones(n), rtol=1e-14)

    def test_diagonal_matrix(self):
        """Known eigenvalues for diagonal matrix (eigenvalues == diagonal)."""
        diag_vals = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        K = np.diag(diag_vals)
        w, v = eigh_dsyevr(K)
        # DSYEVR returns eigenvalues ascending
        expected = np.sort(diag_vals)
        np.testing.assert_allclose(w, expected, rtol=1e-14)

    def test_random_spd_100x100(self):
        """Eigenvalues from DSYEVR match np.linalg.eigh to rtol=1e-12 (100x100)."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((100, 100))
        K = (A @ A.T) / 100
        w_dsyevr, v_dsyevr = eigh_dsyevr(K.copy())
        w_numpy, v_numpy = np.linalg.eigh(K.copy())
        np.testing.assert_allclose(w_dsyevr, w_numpy, rtol=1e-12, atol=1e-14)

    def test_random_spd_1000x1000(self):
        """Eigenvalues from DSYEVR match np.linalg.eigh to rtol=1e-12 (1000x1000)."""
        rng = np.random.default_rng(123)
        A = rng.standard_normal((1000, 1000))
        K = (A @ A.T) / 1000
        w_dsyevr, _ = eigh_dsyevr(K.copy())
        w_numpy, _ = np.linalg.eigh(K.copy())
        np.testing.assert_allclose(w_dsyevr, w_numpy, rtol=1e-12, atol=1e-14)

    def test_reconstruction(self):
        """V @ diag(w) @ V.T reconstructs original matrix to rtol=1e-10."""
        rng = np.random.default_rng(42)
        n = 200
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        K_ref = K.copy()
        w, v = eigh_dsyevr(K.copy())
        K_recon = v @ np.diag(w) @ v.T
        np.testing.assert_allclose(K_recon, K_ref, rtol=1e-10, atol=1e-14)

    def test_eigenvalues_ascending(self):
        """DSYEVR returns eigenvalues in ascending order (RANGE='A')."""
        rng = np.random.default_rng(42)
        n = 150
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        w, _ = eigh_dsyevr(K.copy())
        assert np.all(np.diff(w) >= 0), "Eigenvalues should be sorted ascending"


@pytest.mark.tier0
class TestDsyevrSignConsistency:
    """Downstream invariants that hold regardless of eigenvector sign convention."""

    def test_utranspose_y_invariant(self):
        """(U.T @ y)**2 is sign-invariant: matches DSYEVR and DSYEVD."""
        rng = np.random.default_rng(42)
        n = 100
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        y = rng.standard_normal(n)

        w_dsyevr, v_dsyevr = eigh_dsyevr(K.copy())
        w_numpy, v_numpy = np.linalg.eigh(K.copy())

        # Squared projections are sign-invariant (tiny numerical diffs remain
        # from DSYEVR vs DSYEVD eigenvector precision, hence rtol=1e-10 not 1e-12)
        proj_dsyevr = (v_dsyevr.T @ y) ** 2
        proj_numpy = (v_numpy.T @ y) ** 2
        np.testing.assert_allclose(proj_dsyevr, proj_numpy, rtol=1e-10)

    def test_eigenvalue_order_matches_numpy(self):
        """Eigenvalues from DSYEVR match numpy's ascending order."""
        rng = np.random.default_rng(99)
        n = 80
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        w_dsyevr, _ = eigh_dsyevr(K.copy())
        w_numpy, _ = np.linalg.eigh(K.copy())
        np.testing.assert_allclose(w_dsyevr, w_numpy, rtol=1e-12, atol=1e-14)


@pytest.mark.tier0
class TestDsyevrEdgeCases:
    """Boundary conditions and error handling."""

    def test_n_equals_1(self):
        """1x1 matrix: single element is the eigenvalue."""
        K = np.array([[3.5]], dtype=np.float64)
        w, v = eigh_dsyevr(K.copy())
        assert w.shape == (1,)
        assert v.shape == (1, 1)
        np.testing.assert_allclose(w, [3.5], rtol=1e-14)

    def test_n_equals_2(self):
        """2x2 matrix: eigenvalues match analytical solution."""
        # [[2, 1], [1, 2]]: eigenvalues are 1 and 3
        K = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        w, v = eigh_dsyevr(K.copy())
        np.testing.assert_allclose(w, [1.0, 3.0], rtol=1e-14)
        # Reconstruct to verify eigenvectors
        K_recon = v @ np.diag(w) @ v.T
        np.testing.assert_allclose(K_recon, K, atol=1e-14)

    def test_near_singular(self):
        """Near-singular matrix (cond ~1e15): non-zero eigenvalues match rtol=1e-10."""
        rng = np.random.default_rng(42)
        n = 100
        A = rng.standard_normal((n, n))
        K_full = (A @ A.T) / n
        w_full, v_full = np.linalg.eigh(K_full)
        # Scale smallest eigenvalue way down (condition ~1e15)
        w_ill = w_full.copy()
        w_ill[0] = w_full[-1] * 1e-15
        K_ill = v_full @ np.diag(w_ill) @ v_full.T
        # Make symmetric (remove numerical asymmetry)
        K_ill = (K_ill + K_ill.T) / 2

        w_dsyevr, _ = eigh_dsyevr(K_ill.copy())
        w_numpy, _ = np.linalg.eigh(K_ill.copy())
        # Non-zero eigenvalues should still match to rtol=1e-10
        mask = np.abs(w_numpy) > 1e-12 * np.max(np.abs(w_numpy))
        np.testing.assert_allclose(w_dsyevr[mask], w_numpy[mask], rtol=1e-10)

    def test_zero_eigenvalues(self):
        """Rank-deficient matrix (outer product): should have (n-1) near-zero eigs."""
        rng = np.random.default_rng(42)
        n = 20
        v = rng.standard_normal(n)
        K = np.outer(v, v)  # rank 1: n-1 eigenvalues near 0
        w, _ = eigh_dsyevr(K.copy())
        # All but 1 should be near zero
        n_near_zero = int(np.sum(np.abs(w) < 1e-10 * np.max(np.abs(w) + 1e-30)))
        assert n_near_zero == n - 1, (
            f"Expected {n - 1} near-zero eigenvalues, got {n_near_zero}. "
            f"Eigenvalues: {w}"
        )

    def test_rejects_non_float64(self):
        """eigh_dsyevr raises TypeError for non-float64 input."""
        with pytest.raises(TypeError):
            eigh_dsyevr(np.eye(10, dtype=np.float32))

    def test_rejects_non_square(self):
        """eigh_dsyevr raises ValueError for non-square input."""
        with pytest.raises(ValueError):
            eigh_dsyevr(np.ones((3, 4), dtype=np.float64))

    def test_rejects_1d(self):
        """eigh_dsyevr raises ValueError for 1D input."""
        with pytest.raises(ValueError):
            eigh_dsyevr(np.ones(10, dtype=np.float64))


@pytest.mark.tier0
class TestDsyevrFallback:
    """Dispatch behavior: DSYEVR path vs DSYEVD fallback."""

    def test_fallback_when_unavailable(self):
        """When _DSYEVR_AVAILABLE=False, eigendecompose_kinship uses DSYEVD path."""
        import jamma.lmm.eigen as eigen_mod
        from jamma.lmm.eigen import eigendecompose_kinship

        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        K_ref = K.copy()

        # Patch _DSYEVR_AVAILABLE to False — dispatch falls back to DSYEVD path
        with patch.object(eigen_mod, "_DSYEVR_AVAILABLE", False):
            eigenvalues, eigenvectors = eigendecompose_kinship(
                K.copy(), check_memory=False
            )

        # Results should still be correct via DSYEVD path
        K_recon = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(K_recon, K_ref, rtol=1e-10, atol=1e-14)

    def test_dsyevr_used_when_available(self):
        """When _DSYEVR_AVAILABLE=True, _eigh_dsyevr is called."""
        import jamma.lmm.eigen as eigen_mod
        from jamma.lmm.eigen import _DSYEVR_AVAILABLE, eigendecompose_kinship

        if not _DSYEVR_AVAILABLE:
            pytest.skip("DSYEVR not available — can't test dispatch to it")

        rng = np.random.default_rng(42)
        n = 30
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n

        call_count = []
        original = eigen_mod._eigh_dsyevr

        def tracking_wrapper(K_in):
            call_count.append(1)
            return original(K_in)

        with patch.object(eigen_mod, "_eigh_dsyevr", tracking_wrapper):
            eigendecompose_kinship(K.copy(), check_memory=False)

        assert len(call_count) == 1, (
            "Expected _eigh_dsyevr to be called once when _DSYEVR_AVAILABLE is True"
        )
