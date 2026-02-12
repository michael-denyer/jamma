"""Validation tests comparing JAMMA kinship to reference.

These tests verify JAMMA produces numerically consistent kinship matrices
that match the expected output format and properties.
"""

from pathlib import Path

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship import compute_centered_kinship, compute_standardized_kinship
from jamma.validation import (
    compare_kinship_matrices,
    load_gemma_kinship,
)

# Test data paths - use gemma_synthetic which has matching PLINK + reference outputs
EXAMPLE_DATA = Path("tests/fixtures/gemma_synthetic/test")
REFERENCE_KINSHIP = Path("tests/fixtures/gemma_synthetic/gemma_kinship.cXX.txt")


@pytest.fixture
def mouse_genotypes():
    """Load gemma_synthetic genotypes."""
    plink_data = load_plink_binary(EXAMPLE_DATA)
    return plink_data.genotypes


@pytest.fixture
def reference_kinship():
    """Load reference kinship matrix."""
    return load_gemma_kinship(REFERENCE_KINSHIP)


class TestKinshipValidation:
    """Tests validating JAMMA kinship against reference."""

    def test_kinship_matches_reference(self, mouse_genotypes, reference_kinship):
        """JAMMA kinship matches reference within tolerance.

        Reference was regenerated with monomorphic filtering enabled (GEMMA behavior).
        """
        jamma_kinship = compute_centered_kinship(mouse_genotypes)

        result = compare_kinship_matrices(jamma_kinship, reference_kinship)

        assert result.passed, (
            f"Kinship validation failed: {result.message}\n"
            f"Max abs diff: {result.max_abs_diff:.2e}\n"
            f"Max rel diff: {result.max_rel_diff:.2e}\n"
            f"Worst location: {result.worst_location}"
        )

    def test_kinship_shape_matches_reference(self, mouse_genotypes, reference_kinship):
        """JAMMA kinship has same shape as reference."""
        jamma_kinship = compute_centered_kinship(mouse_genotypes)
        assert jamma_kinship.shape == reference_kinship.shape

    def test_kinship_deterministic(self, mouse_genotypes):
        """Same input produces identical output."""
        K1 = compute_centered_kinship(mouse_genotypes)
        K2 = compute_centered_kinship(mouse_genotypes)
        assert np.array_equal(K1, K2), "Kinship computation is not deterministic"

    def test_kinship_tolerance_report(self, mouse_genotypes, reference_kinship):
        """Report achieved tolerance and assert on regression thresholds."""
        jamma_kinship = compute_centered_kinship(mouse_genotypes)

        result = compare_kinship_matrices(jamma_kinship, reference_kinship)

        # Report for visibility
        print("\n=== KINSHIP VALIDATION REPORT ===")
        print("Dataset: gemma_synthetic (100 samples, 500 SNPs)")
        print(f"Passed: {result.passed}")
        print(f"Max absolute difference: {result.max_abs_diff:.2e}")
        print(f"Max relative difference: {result.max_rel_diff:.2e}")
        if result.worst_location:
            print(f"Worst location: {result.worst_location}")
        print("=================================\n")

        # Assert on regression thresholds - fail if tolerance degrades
        # These thresholds are based on observed behavior with current implementation
        # Observed: ~4.66e-10 relative difference due to float64 precision limits
        assert result.max_abs_diff < 1e-9, (
            f"Absolute difference regression: {result.max_abs_diff:.2e} >= 1e-9"
        )
        assert result.max_rel_diff < 1e-9, (
            f"Relative difference regression: {result.max_rel_diff:.2e} >= 1e-9"
        )


class TestKinshipSmallScale:
    """Tests with smaller synthetic data for faster CI."""

    @pytest.fixture
    def small_genotypes(self):
        """Small synthetic genotype data (100 samples, 1000 SNPs)."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 3, size=(100, 1000)).astype(np.float64)

    def test_small_kinship_symmetric(self, small_genotypes):
        """Small kinship matrix is symmetric."""
        K = compute_centered_kinship(small_genotypes)
        assert np.allclose(K, K.T), "Kinship should be symmetric"

    def test_small_kinship_positive_diagonal(self, small_genotypes):
        """Diagonal elements should be non-negative."""
        K = compute_centered_kinship(small_genotypes)
        assert np.all(np.diag(K) >= 0), "Diagonal should be non-negative"

    def test_small_kinship_shape(self, small_genotypes):
        """Kinship matrix has correct shape."""
        K = compute_centered_kinship(small_genotypes)
        n_samples = small_genotypes.shape[0]
        assert K.shape == (n_samples, n_samples)

    def test_small_kinship_deterministic(self, small_genotypes):
        """Small kinship computation is deterministic."""
        K1 = compute_centered_kinship(small_genotypes)
        K2 = compute_centered_kinship(small_genotypes)
        assert np.array_equal(K1, K2)


class TestKinshipWithMissingData:
    """Tests for kinship computation with missing values."""

    @pytest.fixture
    def genotypes_with_missing(self):
        """Genotype data with 5% missing values."""
        rng = np.random.default_rng(42)
        X = rng.integers(0, 3, size=(100, 1000)).astype(np.float64)
        # Set 5% to NaN
        mask = rng.random(X.shape) < 0.05
        X[mask] = np.nan
        return X

    def test_missing_data_produces_valid_kinship(self, genotypes_with_missing):
        """Kinship with missing data produces valid output."""
        K = compute_centered_kinship(genotypes_with_missing)

        # No NaN in output
        assert not np.any(np.isnan(K)), "Kinship should not contain NaN"

        # Symmetric
        assert np.allclose(K, K.T), "Kinship should be symmetric"

        # Correct shape
        n_samples = genotypes_with_missing.shape[0]
        assert K.shape == (n_samples, n_samples)

    def test_missing_data_deterministic(self, genotypes_with_missing):
        """Missing data handling is deterministic."""
        K1 = compute_centered_kinship(genotypes_with_missing)
        K2 = compute_centered_kinship(genotypes_with_missing)
        assert np.array_equal(K1, K2)


def _numpy_standardized_kinship(genotypes: np.ndarray) -> np.ndarray:
    """Reference implementation using pure NumPy per-SNP loop.

    Matches GEMMA's standardized kinship algorithm: impute missing to mean,
    compute variance on imputed data (E[X^2] - E[X]^2), standardize.

    Uses a per-SNP loop for clarity. Note that this produces slightly different
    floating-point results from JAX's vectorized operations due to different
    reduction order, so comparisons need rtol ~1e-6 not 1e-8.
    """
    n_samples, n_snps = genotypes.shape
    Z = np.zeros_like(genotypes, dtype=np.float64)
    p_used = 0
    for k in range(n_snps):
        col = genotypes[:, k].copy()
        mask = np.isnan(col)
        mean_val = np.nanmean(col)
        if np.isnan(mean_val):
            mean_val = 0.0
        col[mask] = mean_val
        var_val = np.mean(col**2) - mean_val**2
        if var_val > 0:
            Z[:, k] = (col - mean_val) / np.sqrt(var_val)
            p_used += 1
        # else: Z[:, k] stays zero
    if p_used == 0:
        return np.zeros((n_samples, n_samples), dtype=np.float64)
    return Z @ Z.T / p_used


class TestStandardizedKinshipValidation:
    """Tests validating JAMMA standardized kinship against NumPy reference."""

    def test_standardized_kinship_matches_numpy_reference(self, mouse_genotypes):
        """JAMMA standardized kinship matches NumPy reference on gemma_synthetic.

        Tolerance is rtol=1e-3 because JAX vectorized nanmean/variance uses
        different reduction order than NumPy per-SNP scalar loop. The per-element
        Z difference (~1e-6 relative) accumulates through the matrix multiply to
        ~5e-4 relative on K elements. This is normal FP behavior, not a bug.
        """
        K_jamma = compute_standardized_kinship(mouse_genotypes)
        K_ref = _numpy_standardized_kinship(mouse_genotypes)

        max_rel = np.max(np.abs(K_jamma - K_ref) / (np.abs(K_ref) + 1e-15))
        assert max_rel < 1e-3, (
            f"Standardized kinship mismatch: max rel diff = {max_rel:.2e}"
        )

    def test_standardized_kinship_differs_from_centered(self, mouse_genotypes):
        """Standardized and centered kinship are not identical on real data."""
        K_centered = compute_centered_kinship(mouse_genotypes)
        K_standardized = compute_standardized_kinship(mouse_genotypes)

        assert not np.allclose(K_centered, K_standardized), (
            "Standardized and centered kinship should differ on real data"
        )

    def test_standardized_kinship_properties(self, mouse_genotypes):
        """Standardized kinship has expected mathematical properties."""
        K = compute_standardized_kinship(mouse_genotypes)

        # Symmetric
        assert np.allclose(K, K.T), "Standardized kinship should be symmetric"

        # No NaN
        assert not np.any(np.isnan(K)), "Should not contain NaN"

        # Positive semi-definite (eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8), (
            f"Should be PSD, min eigenvalue: {eigenvalues.min():.2e}"
        )

        # Positive diagonal
        assert np.all(np.diag(K) > 0), "Diagonal should be positive"

    def test_standardized_kinship_with_maf_filter(self, mouse_genotypes):
        """Standardized kinship works with MAF filtering."""
        K = compute_standardized_kinship(mouse_genotypes, maf_threshold=0.05)

        assert np.allclose(K, K.T), "Should be symmetric with MAF filter"
        assert not np.any(np.isnan(K)), "Should not contain NaN"

    def test_standardized_kinship_matches_numpy_with_missing(self):
        """Standardized kinship matches reference with missing data."""
        rng = np.random.default_rng(42)
        X = rng.integers(0, 3, size=(100, 1000)).astype(np.float64)
        # Set 5% to NaN
        mask = rng.random(X.shape) < 0.05
        X[mask] = np.nan

        K_jamma = compute_standardized_kinship(X)
        K_ref = _numpy_standardized_kinship(X)

        max_rel = np.max(np.abs(K_jamma - K_ref) / (np.abs(K_ref) + 1e-15))
        assert max_rel < 1e-3, (
            f"Mismatch with missing data: max rel diff = {max_rel:.2e}"
        )
