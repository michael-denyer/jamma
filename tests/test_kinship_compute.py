"""Tests for kinship matrix computation.

These tests verify the JAX-accelerated kinship matrix computation,
including symmetry, scaling, missing data handling, and determinism.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jamma.core import configure_jax
from jamma.kinship import compute_centered_kinship, impute_and_center


@pytest.fixture(autouse=True)
def setup_jax():
    """Configure JAX with 64-bit precision before each test."""
    configure_jax(enable_x64=True)


@pytest.fixture
def simple_genotypes():
    """Small test genotype matrix (4 samples, 5 SNPs)."""
    return np.array(
        [
            [0, 1, 2, 0, 1],
            [1, 1, 1, 1, 1],
            [2, 1, 0, 2, 1],
            [1, 0, 1, 1, 0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def genotypes_with_missing():
    """Genotype matrix with missing values."""
    return np.array(
        [
            [0, np.nan, 2, 0, 1],
            [1, 1, np.nan, 1, 1],
            [2, 1, 0, np.nan, 1],
            [1, 0, 1, 1, np.nan],
        ],
        dtype=np.float64,
    )


class TestKinshipSymmetry:
    """Tests for kinship matrix symmetry."""

    def test_kinship_symmetric(self, simple_genotypes):
        """Kinship matrix should be symmetric: K == K.T."""
        K = compute_centered_kinship(simple_genotypes)
        assert np.allclose(K, K.T), "Kinship matrix should be symmetric"

    def test_kinship_symmetric_with_missing(self, genotypes_with_missing):
        """Symmetry should hold even with missing data."""
        K = compute_centered_kinship(genotypes_with_missing)
        assert np.allclose(
            K, K.T
        ), "Kinship matrix should be symmetric with missing data"


class TestKinshipShape:
    """Tests for kinship matrix dimensions."""

    def test_kinship_shape(self, simple_genotypes):
        """Kinship matrix should be (n_samples, n_samples)."""
        K = compute_centered_kinship(simple_genotypes)
        n_samples = simple_genotypes.shape[0]
        assert K.shape == (n_samples, n_samples)

    def test_kinship_shape_various_sizes(self):
        """Test shape with different sample/SNP counts."""
        rng = np.random.default_rng(42)
        for n_samples, n_snps in [(10, 20), (5, 100), (50, 10)]:
            X = rng.integers(0, 3, size=(n_samples, n_snps)).astype(np.float64)
            K = compute_centered_kinship(X)
            assert K.shape == (n_samples, n_samples)


class TestKinshipDiagonal:
    """Tests for kinship matrix diagonal properties."""

    def test_kinship_positive_diagonal(self, simple_genotypes):
        """Diagonal elements should be non-negative."""
        K = compute_centered_kinship(simple_genotypes)
        assert np.all(np.diag(K) >= 0), "Diagonal should be non-negative"

    def test_kinship_diagonal_with_missing(self, genotypes_with_missing):
        """Diagonal should remain non-negative with missing data."""
        K = compute_centered_kinship(genotypes_with_missing)
        # Allow small numerical error for non-negative check
        assert np.all(np.diag(K) >= -1e-10), "Diagonal should be non-negative"


class TestKinshipMissingData:
    """Tests for missing data handling."""

    def test_kinship_with_missing_data(self, genotypes_with_missing):
        """Kinship computation should handle NaN values without errors."""
        K = compute_centered_kinship(genotypes_with_missing)
        # Should produce valid result (no NaN)
        assert not np.any(np.isnan(K)), "Result should not contain NaN"

    def test_kinship_missing_imputed_correctly(self):
        """Missing values should be imputed to SNP mean."""
        # Simple case: one SNP with known mean
        X = np.array([[0.0], [np.nan], [2.0]], dtype=np.float64)
        K = compute_centered_kinship(X)
        # Mean is 1.0, so X_centered = [-1, 0, 1]
        # K = 1/1 * X_c @ X_c.T
        # K[0,0] = (-1)^2 = 1, K[1,1] = 0^2 = 0, K[2,2] = 1^2 = 1
        expected_diag = np.array([1.0, 0.0, 1.0])
        assert np.allclose(np.diag(K), expected_diag)


class TestKinshipScaling:
    """Tests for kinship matrix scaling."""

    def test_kinship_scaling(self):
        """Kinship matrix should be scaled by number of SNPs."""
        rng = np.random.default_rng(42)

        # Create genotypes with known properties
        n_samples, n_snps = 5, 10
        X = rng.integers(0, 3, size=(n_samples, n_snps)).astype(np.float64)

        K = compute_centered_kinship(X)

        # Manually compute expected K
        X_jax = jnp.array(X)
        X_centered = X_jax - jnp.mean(X_jax, axis=0, keepdims=True)
        K_expected = np.array(jnp.matmul(X_centered, X_centered.T)) / n_snps

        assert np.allclose(K, K_expected, rtol=1e-10)


class TestKinshipDeterminism:
    """Tests for reproducibility."""

    def test_kinship_deterministic(self, simple_genotypes):
        """Same input should produce identical output."""
        K1 = compute_centered_kinship(simple_genotypes)
        K2 = compute_centered_kinship(simple_genotypes)
        assert np.allclose(K1, K2), "Results should be deterministic"

    def test_kinship_deterministic_with_missing(self, genotypes_with_missing):
        """Determinism should hold with missing data."""
        K1 = compute_centered_kinship(genotypes_with_missing)
        K2 = compute_centered_kinship(genotypes_with_missing)
        assert np.allclose(K1, K2), "Results should be deterministic with missing data"


class TestKinshipBatching:
    """Tests for batch processing."""

    def test_kinship_batch_size_invariant(self, simple_genotypes):
        """Different batch sizes should produce same result."""
        K1 = compute_centered_kinship(simple_genotypes, batch_size=1)
        K2 = compute_centered_kinship(simple_genotypes, batch_size=2)
        K3 = compute_centered_kinship(simple_genotypes, batch_size=100)
        assert np.allclose(K1, K2), "Batch size 1 vs 2 should match"
        assert np.allclose(K1, K3), "Batch size 1 vs 100 should match"

    def test_kinship_large_batch_size(self):
        """Batch size larger than n_snps should work correctly."""
        rng = np.random.default_rng(42)
        X = rng.integers(0, 3, size=(10, 20)).astype(np.float64)
        K = compute_centered_kinship(X, batch_size=1000)
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T)


class TestImputeAndCenter:
    """Tests for the impute_and_center function."""

    def test_impute_and_center_no_missing(self):
        """Centering without missing values."""
        X = jnp.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
        X_centered = impute_and_center(X)

        # Mean of col 0: 1.0, col 1: 1.0
        # Centered: [[-1, 1], [0, 0], [1, -1]]
        expected = jnp.array([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]])
        assert jnp.allclose(X_centered, expected)

    def test_impute_and_center_column_mean_zero(self):
        """Centered columns should have mean approximately zero."""
        X = jnp.array([[0.0, 1.0, 2.0], [1.0, 2.0, 0.0], [2.0, 0.0, 1.0]])
        X_centered = impute_and_center(X)

        # Column means should be zero (or very close)
        col_means = jnp.mean(X_centered, axis=0)
        assert jnp.allclose(col_means, 0.0, atol=1e-10)

    def test_impute_and_center_with_missing(self):
        """NaN values should be replaced with column mean, then centered."""
        X = jnp.array([[0.0, jnp.nan], [jnp.nan, 2.0], [2.0, 2.0]])

        X_centered = impute_and_center(X)

        # Col 0: [0, nan, 2] -> mean=1.0, imputed=[0,1,2], centered=[-1,0,1]
        # Col 1: [nan, 2, 2] -> mean=2.0, imputed=[2,2,2], centered=[0,0,0]
        expected = jnp.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        assert jnp.allclose(X_centered, expected)

    def test_impute_and_center_preserves_dtype(self):
        """Output dtype should match input dtype."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)
        X_centered = impute_and_center(X)
        assert X_centered.dtype == jnp.float64


class TestKinshipNumerical:
    """Tests for numerical properties."""

    def test_kinship_trace_related_to_variance(self):
        """Trace of K should be related to total genetic variance."""
        rng = np.random.default_rng(42)
        X = rng.integers(0, 3, size=(20, 50)).astype(np.float64)
        K = compute_centered_kinship(X)

        # Trace should be positive (sum of individual variances)
        assert np.trace(K) > 0

    def test_kinship_eigenvalues_reasonable(self):
        """Eigenvalues should be non-negative for a valid kinship matrix."""
        rng = np.random.default_rng(42)
        X = rng.integers(0, 3, size=(10, 100)).astype(np.float64)
        K = compute_centered_kinship(X)

        eigenvalues = np.linalg.eigvalsh(K)
        # Allow small negative due to numerical error
        assert np.all(eigenvalues >= -1e-8), f"Min eigenvalue: {eigenvalues.min()}"
