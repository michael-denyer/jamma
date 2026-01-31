"""Validation tests comparing JAMMA kinship to reference.

These tests verify JAMMA produces numerically consistent kinship matrices
that match the expected output format and properties.
"""

import numpy as np
import pytest
from pathlib import Path

from jamma.kinship import compute_centered_kinship
from jamma.io import load_plink_binary
from jamma.validation import (
    ToleranceConfig,
    compare_kinship_matrices,
    load_gemma_kinship,
)


# Test data paths
EXAMPLE_DATA = Path("legacy/example/mouse_hs1940")
REFERENCE_KINSHIP = Path("tests/fixtures/kinship/mouse_hs1940.cXX.txt")


@pytest.fixture
def mouse_genotypes():
    """Load mouse_hs1940 genotypes."""
    plink_data = load_plink_binary(EXAMPLE_DATA)
    return plink_data.genotypes


@pytest.fixture
def reference_kinship():
    """Load reference kinship matrix."""
    return load_gemma_kinship(REFERENCE_KINSHIP)


class TestKinshipValidation:
    """Tests validating JAMMA kinship against reference."""

    def test_kinship_matches_reference(self, mouse_genotypes, reference_kinship):
        """JAMMA kinship matches reference within tolerance."""
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
        """Report achieved tolerance for decision gate."""
        jamma_kinship = compute_centered_kinship(mouse_genotypes)

        result = compare_kinship_matrices(jamma_kinship, reference_kinship)

        # This test always passes but reports the divergence for manual review
        print(f"\n=== KINSHIP VALIDATION REPORT ===")
        print(f"Dataset: mouse_hs1940 (1940 samples, 12226 SNPs)")
        print(f"Passed: {result.passed}")
        print(f"Max absolute difference: {result.max_abs_diff:.2e}")
        print(f"Max relative difference: {result.max_rel_diff:.2e}")
        if result.worst_location:
            print(f"Worst location: {result.worst_location}")
        print(f"=================================\n")


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
