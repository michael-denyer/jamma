"""Python integration tests for Rust eigendecomposition.

These tests verify the Rust implementation produces numerically identical
results to scipy.linalg.eigh within the existing JAMMA tolerances.
"""

import numpy as np
import pytest
from scipy import linalg

# Import will fail if maturin develop hasn't been run
jamma_core = pytest.importorskip("jamma_core")


class TestEigendecompBasic:
    """Basic functionality tests."""

    def test_identity_matrix(self):
        """Identity matrix should have all eigenvalues = 1."""
        n = 100
        K = np.eye(n, dtype=np.float64)

        eigenvalues, eigenvectors = jamma_core.eigendecompose_kinship(K)

        assert eigenvalues.shape == (n,)
        assert eigenvectors.shape == (n, n)
        # All eigenvalues should be 1.0
        np.testing.assert_allclose(eigenvalues, np.ones(n), rtol=1e-10)

    def test_diagonal_matrix(self):
        """Diagonal matrix eigenvalues are the diagonal elements."""
        n = 50
        diag = np.arange(1, n + 1, dtype=np.float64)
        K = np.diag(diag)

        eigenvalues, eigenvectors = jamma_core.eigendecompose_kinship(K)

        # Eigenvalues should match diagonal (sorted ascending)
        np.testing.assert_allclose(eigenvalues, np.sort(diag), rtol=1e-10)

    def test_symmetric_random(self):
        """Random symmetric matrix reconstruction."""
        np.random.seed(42)
        n = 100
        A = np.random.randn(n, n)
        K = (A + A.T) / 2  # Make symmetric

        eigenvalues, eigenvectors = jamma_core.eigendecompose_kinship(K)

        # Verify reconstruction: K = U @ diag(S) @ U.T
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        np.testing.assert_allclose(reconstructed, K, rtol=1e-10, atol=1e-10)

    def test_threshold_zeroing(self):
        """Small eigenvalues below threshold should be zeroed."""
        n = 10
        # Matrix with one very small eigenvalue
        K = np.eye(n, dtype=np.float64)
        K[0, 0] = 1e-12  # Below default threshold of 1e-10

        eigenvalues, _ = jamma_core.eigendecompose_kinship(K)

        # Smallest eigenvalue should be zeroed
        assert eigenvalues[0] == 0.0

    def test_custom_threshold(self):
        """Custom threshold should control zeroing."""
        n = 10
        K = np.eye(n, dtype=np.float64)
        K[0, 0] = 1e-6  # Above default threshold

        # With default threshold, should NOT be zeroed
        eigenvalues_default, _ = jamma_core.eigendecompose_kinship(K)
        assert eigenvalues_default[0] != 0.0

        # With higher threshold, should be zeroed
        eigenvalues_custom, _ = jamma_core.eigendecompose_kinship(K, threshold=1e-5)
        assert eigenvalues_custom[0] == 0.0


class TestEigendecompScipyParity:
    """Tests verifying parity with scipy.linalg.eigh."""

    def test_eigenvalues_match_scipy(self):
        """Eigenvalues should match scipy within tolerance."""
        np.random.seed(123)
        n = 200
        A = np.random.randn(n, n)
        K = (A + A.T) / 2

        # Rust implementation
        rust_eigenvalues, _ = jamma_core.eigendecompose_kinship(K, threshold=0.0)

        # scipy reference
        scipy_eigenvalues, _ = linalg.eigh(K)

        # JAMMA tolerance for eigenvalues
        np.testing.assert_allclose(
            rust_eigenvalues, scipy_eigenvalues, rtol=1e-10, atol=1e-14
        )

    def test_eigenvectors_orthonormal(self):
        """Eigenvectors should be orthonormal."""
        np.random.seed(456)
        n = 100
        A = np.random.randn(n, n)
        K = (A + A.T) / 2

        _, eigenvectors = jamma_core.eigendecompose_kinship(K)

        # U @ U.T should be identity
        orthogonality = eigenvectors @ eigenvectors.T
        np.testing.assert_allclose(orthogonality, np.eye(n), rtol=1e-10, atol=1e-10)

    def test_eigenvector_equation(self):
        """Each eigenvector should satisfy K @ v = lambda @ v."""
        np.random.seed(789)
        n = 50
        A = np.random.randn(n, n)
        K = (A + A.T) / 2

        eigenvalues, eigenvectors = jamma_core.eigendecompose_kinship(K, threshold=0.0)

        for i in range(n):
            v = eigenvectors[:, i]
            lam = eigenvalues[i]
            # K @ v should equal lambda * v
            np.testing.assert_allclose(K @ v, lam * v, rtol=1e-10, atol=1e-10)

    def test_larger_matrix_parity(self):
        """Test with larger matrix (1000x1000) for numerical stability."""
        np.random.seed(999)
        n = 1000
        A = np.random.randn(n, n)
        K = (A + A.T) / 2

        rust_eigenvalues, rust_eigenvectors = jamma_core.eigendecompose_kinship(
            K, threshold=0.0
        )
        scipy_eigenvalues, scipy_eigenvectors = linalg.eigh(K)

        # Eigenvalues should match
        np.testing.assert_allclose(
            rust_eigenvalues, scipy_eigenvalues, rtol=1e-10, atol=1e-12
        )

        # Eigenvectors may have sign differences - compare absolute values of projection
        # K @ U should equal U @ diag(S)
        rust_projection = K @ rust_eigenvectors
        expected_projection = rust_eigenvectors * rust_eigenvalues
        np.testing.assert_allclose(
            rust_projection, expected_projection, rtol=1e-9, atol=1e-10
        )


class TestEigendecompEdgeCases:
    """Edge case and error handling tests."""

    def test_non_square_raises_error(self):
        """Non-square matrix should raise ValueError."""
        K = np.ones((10, 20), dtype=np.float64)

        with pytest.raises(ValueError, match="square"):
            jamma_core.eigendecompose_kinship(K)

    def test_empty_matrix_raises_error(self):
        """Empty matrix should raise ValueError."""
        K = np.zeros((0, 0), dtype=np.float64)

        with pytest.raises(ValueError, match="empty"):
            jamma_core.eigendecompose_kinship(K)

    def test_single_element(self):
        """1x1 matrix should work."""
        K = np.array([[5.0]], dtype=np.float64)

        eigenvalues, eigenvectors = jamma_core.eigendecompose_kinship(K)

        assert eigenvalues.shape == (1,)
        assert eigenvectors.shape == (1, 1)
        np.testing.assert_allclose(eigenvalues, [5.0], rtol=1e-10)

    def test_f_contiguous_input(self):
        """Fortran-ordered input should work (tests memory layout handling)."""
        np.random.seed(111)
        n = 50
        A = np.random.randn(n, n)
        K = np.asfortranarray((A + A.T) / 2)  # Fortran (column-major) order

        eigenvalues, eigenvectors = jamma_core.eigendecompose_kinship(K)

        # Should still produce valid eigendecomposition
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        K_c = np.ascontiguousarray(K)
        np.testing.assert_allclose(reconstructed, K_c, rtol=1e-10, atol=1e-10)

    def test_nan_input_raises_error(self):
        """Matrix containing NaN should raise ValueError."""
        n = 10
        K = np.eye(n, dtype=np.float64)
        K[5, 5] = np.nan

        with pytest.raises(ValueError, match="NaN or Inf"):
            jamma_core.eigendecompose_kinship(K)

    def test_inf_input_raises_error(self):
        """Matrix containing Inf should raise ValueError."""
        n = 10
        K = np.eye(n, dtype=np.float64)
        K[3, 3] = np.inf

        with pytest.raises(ValueError, match="NaN or Inf"):
            jamma_core.eigendecompose_kinship(K)

    def test_negative_inf_input_raises_error(self):
        """Matrix containing -Inf should raise ValueError."""
        n = 10
        K = np.eye(n, dtype=np.float64)
        K[0, 0] = -np.inf

        with pytest.raises(ValueError, match="NaN or Inf"):
            jamma_core.eigendecompose_kinship(K)

    def test_negative_threshold_raises_error(self):
        """Negative threshold should raise ValueError."""
        n = 10
        K = np.eye(n, dtype=np.float64)

        with pytest.raises(ValueError, match="non-negative"):
            jamma_core.eigendecompose_kinship(K, threshold=-1e-5)

    def test_nan_threshold_raises_error(self):
        """NaN threshold should raise ValueError."""
        n = 10
        K = np.eye(n, dtype=np.float64)

        with pytest.raises(ValueError, match="finite"):
            jamma_core.eigendecompose_kinship(K, threshold=np.nan)

    def test_inf_threshold_raises_error(self):
        """Inf threshold should raise ValueError."""
        n = 10
        K = np.eye(n, dtype=np.float64)

        with pytest.raises(ValueError, match="finite"):
            jamma_core.eigendecompose_kinship(K, threshold=np.inf)
