"""Tests for heterogeneous missingness patterns and edge cases.

Validates that SNP filtering and statistics handle real-world missingness
patterns: varying rates per SNP, high-missingness samples, checkerboard
patterns, and degenerate cases (all-NaN, near-all-NaN SNPs).

All tests are tier0 (no I/O, no GEMMA reference).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats


@pytest.mark.tier0
class TestHeterogeneousMissingness:
    """Tests for varying missing rates across SNPs and samples."""

    def test_heterogeneous_missingness_snp_stats(self):
        """Verify per-SNP stats with varying missing rates across 10 SNPs."""
        rng = np.random.default_rng(42)
        n_samples, n_snps = 50, 10
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))

        # SNP 0: 0% missing
        # SNP 1: 5% missing (2-3 samples)
        # SNP 2: 20% missing (10 samples)
        # SNP 3: 50% missing (25 samples)
        # SNPs 4-9: varying rates 0-30%
        miss_counts_expected = [0, 3, 10, 25, 0, 5, 8, 12, 15, 2]
        for snp_idx, n_miss in enumerate(miss_counts_expected):
            if n_miss > 0:
                miss_samples = rng.choice(n_samples, size=n_miss, replace=False)
                genotypes[miss_samples, snp_idx] = np.nan

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)

        # miss_counts match expected
        assert_array_equal(miss_counts, miss_counts_expected)

        # col_means computed on non-missing values only
        # SNP 0: no missing, mean = nanmean of all 50 values
        expected_mean_0 = np.nanmean(genotypes[:, 0])
        assert_allclose(col_means[0], expected_mean_0, atol=1e-14)

        # SNP 3: 50% missing, mean computed from 25 non-missing values
        expected_mean_3 = np.nanmean(genotypes[:, 3])
        assert_allclose(col_means[3], expected_mean_3, atol=1e-14)

        # All variances non-negative
        assert np.all(col_vars >= 0)

    def test_heterogeneous_missingness_filter_mask(self):
        """Verify filter with miss_threshold=0.25 excludes high-missingness SNPs."""
        rng = np.random.default_rng(42)
        n_samples, n_snps = 50, 10
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))

        miss_counts_expected = [0, 3, 10, 25, 0, 5, 8, 12, 15, 2]
        for snp_idx, n_miss in enumerate(miss_counts_expected):
            if n_miss > 0:
                miss_samples = rng.choice(n_samples, size=n_miss, replace=False)
                genotypes[miss_samples, snp_idx] = np.nan

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)
        snp_mask, _allele_freqs, _mafs = compute_snp_filter_mask(
            col_means,
            miss_counts,
            col_vars,
            n_samples,
            maf_threshold=0.0,
            miss_threshold=0.25,
        )

        # SNP 3 (50% missing) must be excluded
        assert not snp_mask[3], "SNP 3 with 50% missing should be filtered out"

        # SNPs with <=25% missing rate should be retained (if polymorphic)
        # miss_rates: [0, 0.06, 0.20, 0.50, 0, 0.10, 0.16, 0.24, 0.30, 0.04]
        # SNP 8 (30% missing) should also be excluded
        assert not snp_mask[8], "SNP 8 with 30% missing should be filtered out"

        # Mask shape matches n_snps
        assert snp_mask.shape == (n_snps,)

    def test_high_missingness_sample_pattern(self):
        """Verify per-SNP stats when specific samples have high missingness."""
        rng = np.random.default_rng(123)
        n_samples, n_snps = 20, 5
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))

        # Sample 0: missing for 80% of SNPs (4 out of 5)
        genotypes[0, :4] = np.nan

        # Sample 1: missing for 0% of SNPs
        # (no changes needed)

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)

        # Per-SNP miss counts: SNPs 0-3 each have 1 missing (sample 0), SNP 4 has 0
        assert_array_equal(miss_counts, [1, 1, 1, 1, 0])

        # Means should be computed from 19 non-missing values for SNPs 0-3
        for snp_idx in range(4):
            expected = np.nanmean(genotypes[:, snp_idx])
            assert_allclose(col_means[snp_idx], expected, atol=1e-14)

        # SNP 4: all 20 values present
        expected_4 = np.mean(genotypes[:, 4])
        assert_allclose(col_means[4], expected_4, atol=1e-14)

    def test_cross_pattern_missingness(self):
        """Verify checkerboard missingness produces correct, deterministic results."""
        n_samples, n_snps = 20, 6

        # Checkerboard: odd samples missing for even SNPs, vice versa
        genotypes = np.full((n_samples, n_snps), 1.0)
        for i in range(n_samples):
            for j in range(n_snps):
                if (i % 2 == 1 and j % 2 == 0) or (i % 2 == 0 and j % 2 == 1):
                    genotypes[i, j] = np.nan
                else:
                    # Set non-missing values to vary so SNPs are polymorphic
                    genotypes[i, j] = float(i % 3)  # 0, 1, 2 pattern

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)

        # Each SNP has exactly 50% missing (10 out of 20)
        assert_array_equal(miss_counts, [10, 10, 10, 10, 10, 10])

        # Mean imputation computes from available values
        for snp_idx in range(n_snps):
            expected = np.nanmean(genotypes[:, snp_idx])
            assert_allclose(col_means[snp_idx], expected, atol=1e-14)

        # Deterministic: run twice, compare
        col_means2, miss_counts2, col_vars2 = compute_snp_stats(genotypes)
        assert_array_equal(col_means, col_means2)
        assert_array_equal(miss_counts, miss_counts2)
        assert_array_equal(col_vars, col_vars2)


@pytest.mark.tier0
class TestMissingSNPEdgeCases:
    """Tests for all-missing and near-all-missing SNP edge cases."""

    def test_all_missing_snp_returns_zero_stats(self):
        """Verify all-NaN column produces zero mean and zero variance."""
        n_samples = 30
        genotypes = np.array(
            [
                [1.0, np.nan],
                [0.0, np.nan],
                [2.0, np.nan],
            ]
            + [[1.0, np.nan]] * (n_samples - 3)
        )

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)

        # All-NaN SNP (column 1)
        assert col_means[1] == 0.0, "All-NaN SNP mean should be 0.0 (nan_to_num)"
        assert col_vars[1] == 0.0, "All-NaN SNP variance should be 0.0 (nan_to_num)"
        assert miss_counts[1] == n_samples

    def test_all_missing_snp_filtered_out(self):
        """Verify all-NaN SNP is filtered by both missingness and polymorphic checks."""
        n_samples = 20
        rng = np.random.default_rng(99)
        # Normal SNP + all-NaN SNP
        normal = rng.choice([0.0, 1.0, 2.0], size=(n_samples, 1))
        all_nan = np.full((n_samples, 1), np.nan)
        genotypes = np.hstack([normal, all_nan])

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)
        snp_mask, allele_freqs, mafs = compute_snp_filter_mask(
            col_means,
            miss_counts,
            col_vars,
            n_samples,
            maf_threshold=0.01,
            miss_threshold=0.1,
        )

        # All-NaN SNP must be filtered out
        assert not snp_mask[1], "All-NaN SNP should be filtered out"

        # Filtered by miss_threshold (100% > 10%)
        miss_rate = miss_counts[1] / n_samples
        assert miss_rate > 0.1

        # Also filtered by polymorphic check (var=0)
        assert col_vars[1] == 0.0

    def test_near_all_missing_snp(self):
        """Verify single non-missing value SNP has zero variance, is filtered."""
        n_samples = 20
        genotypes = np.full((n_samples, 2), np.nan)
        # SNP 0: all NaN except sample 0 = 1.0
        genotypes[0, 0] = 1.0
        # SNP 1: normal data for comparison
        genotypes[:, 1] = np.tile([0.0, 1.0, 2.0, 1.0], 5)

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)

        # Single-value SNP
        assert_allclose(col_means[0], 1.0, atol=1e-14)
        assert_allclose(col_vars[0], 0.0, atol=1e-14)
        assert miss_counts[0] == n_samples - 1

        # Should be filtered by polymorphic check
        snp_mask, _, _ = compute_snp_filter_mask(
            col_means,
            miss_counts,
            col_vars,
            n_samples,
            maf_threshold=0.0,
            miss_threshold=1.0,  # permissive thresholds
        )
        assert not snp_mask[0], (
            "Single-value SNP should be filtered by polymorphic check"
        )

    def test_near_all_missing_two_values(self):
        """Verify SNP with only 2 non-missing values has well-defined statistics."""
        n_samples = 20
        genotypes = np.full((n_samples, 1), np.nan)
        genotypes[0, 0] = 0.0
        genotypes[1, 0] = 2.0

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)

        assert_allclose(col_means[0], 1.0, atol=1e-14)
        assert_allclose(col_vars[0], 1.0, atol=1e-14)
        assert miss_counts[0] == n_samples - 2

        # With permissive miss_threshold, this SNP passes (polymorphic, var>0)
        snp_mask, _, _ = compute_snp_filter_mask(
            col_means,
            miss_counts,
            col_vars,
            n_samples,
            maf_threshold=0.0,
            miss_threshold=1.0,
        )
        assert snp_mask[0], (
            "Two-value polymorphic SNP should pass with permissive thresholds"
        )

    def test_all_snps_all_missing(self):
        """Verify all-NaN matrix produces all-False mask with no errors."""
        n_samples, n_snps = 15, 4
        genotypes = np.full((n_samples, n_snps), np.nan)

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)

        # All means and vars should be 0.0 (nan_to_num)
        assert_array_equal(col_means, np.zeros(n_snps))
        assert_array_equal(col_vars, np.zeros(n_snps))
        assert_array_equal(miss_counts, np.full(n_snps, n_samples))

        snp_mask, allele_freqs, mafs = compute_snp_filter_mask(
            col_means,
            miss_counts,
            col_vars,
            n_samples,
            maf_threshold=0.01,
            miss_threshold=0.1,
        )

        # All SNPs filtered out
        assert not np.any(snp_mask), "All-NaN matrix should produce all-False mask"

        # Allele freqs and MAFs are all 0.0
        assert_array_equal(allele_freqs, np.zeros(n_snps))
        assert_array_equal(mafs, np.zeros(n_snps))
