"""Tests for SNP filter functions including HWE computation.

Validates compute_hwe_pvalues for equilibrium, deviation, degenerate,
vectorized, and textbook cases. Also tests filter composition (SNP list
+ HWE + MAF/miss filters via boolean AND).
"""

import numpy as np
import pytest

from jamma.core.snp_filter import compute_hwe_pvalues, compute_snp_filter_mask


@pytest.mark.tier0
class TestComputeHwePvalues:
    """Tests for Hardy-Weinberg equilibrium chi-squared p-values."""

    def test_hwe_perfect_equilibrium(self):
        """Perfect HWE (p=0.5): n_aa=25, n_ab=50, n_bb=25 -> p-value ~1.0."""
        n_aa = np.array([25])
        n_ab = np.array([50])
        n_bb = np.array([25])

        pvalues = compute_hwe_pvalues(n_aa, n_ab, n_bb)
        assert pvalues.shape == (1,)
        assert pvalues[0] >= 0.99

    def test_hwe_strong_deviation(self):
        """Strong HWE deviation: heterozygote deficit -> p-value near 0.

        n_aa=50, n_ab=0, n_bb=50: p=q=0.5, expected AB=50, observed AB=0.
        Extreme deficit of heterozygotes.
        """
        n_aa = np.array([50])
        n_ab = np.array([0])
        n_bb = np.array([50])

        pvalues = compute_hwe_pvalues(n_aa, n_ab, n_bb)
        assert pvalues[0] < 0.001

    def test_hwe_vectorized(self):
        """Pass arrays of 100 SNPs, verify output shape matches."""
        rng = np.random.default_rng(42)
        n_aa = rng.integers(10, 50, size=100)
        n_ab = rng.integers(10, 50, size=100)
        n_bb = rng.integers(10, 50, size=100)

        pvalues = compute_hwe_pvalues(n_aa, n_ab, n_bb)
        assert pvalues.shape == (100,)
        assert np.all(pvalues >= 0)
        assert np.all(pvalues <= 1)

    def test_hwe_degenerate_snp(self):
        """All same genotype (monomorphic): should return p=1.0, not NaN."""
        n_aa = np.array([100])
        n_ab = np.array([0])
        n_bb = np.array([0])

        pvalues = compute_hwe_pvalues(n_aa, n_ab, n_bb)
        assert not np.isnan(pvalues[0])
        # Degenerate SNPs pass HWE by convention (chi_sq=0 -> p=1.0)
        assert pvalues[0] >= 0.99

    def test_hwe_known_value(self):
        """Textbook example with known chi-squared value.

        n_aa=735, n_ab=210, n_bb=55 (N=1000, p=0.84, q=0.16)
        Expected: e_aa=705.6, e_ab=268.8, e_bb=25.6
        chi_sq ~ 48.3 -> p-value essentially 0.
        """
        n_aa = np.array([735])
        n_ab = np.array([210])
        n_bb = np.array([55])

        pvalues = compute_hwe_pvalues(n_aa, n_ab, n_bb)
        assert pvalues[0] < 1e-10

    def test_hwe_all_zero_counts(self):
        """All zero counts: degenerate case should not crash."""
        n_aa = np.array([0])
        n_ab = np.array([0])
        n_bb = np.array([0])

        pvalues = compute_hwe_pvalues(n_aa, n_ab, n_bb)
        assert not np.isnan(pvalues[0])


@pytest.mark.tier0
class TestFilterComposition:
    """Integration tests for filter composition (SNP list + HWE + MAF/miss)."""

    def _make_polymorphic_genotypes(
        self, n_samples: int, n_snps: int, seed: int = 42
    ) -> np.ndarray:
        """Create a genotype matrix with all SNPs polymorphic."""
        rng = np.random.default_rng(seed)
        geno = rng.choice(
            [0.0, 1.0, 2.0], size=(n_samples, n_snps), p=[0.25, 0.5, 0.25]
        )
        return geno

    def test_snp_list_mask_composition(self):
        """SNP list mask ANDs with MAF/miss filter mask."""
        geno = self._make_polymorphic_genotypes(10, 20)
        n_samples = geno.shape[0]

        # Compute basic stats and filter mask
        miss_counts = np.sum(np.isnan(geno), axis=0)
        with np.errstate(invalid="ignore"):
            col_means = np.nanmean(geno, axis=0)
            col_vars = np.nanvar(geno, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        col_vars = np.nan_to_num(col_vars, nan=0.0)

        snp_mask, _, _ = compute_snp_filter_mask(
            col_means,
            miss_counts,
            col_vars,
            n_samples,
            maf_threshold=0.0,
            miss_threshold=1.0,
        )

        # Create SNP list restriction: only indices 0, 5, 10, 15
        snps_indices = np.array([0, 5, 10, 15])
        snp_list_mask = np.zeros(20, dtype=bool)
        snp_list_mask[snps_indices] = True
        combined = snp_mask & snp_list_mask

        # Only SNPs in list AND passing filter survive
        passing_indices = np.where(combined)[0]
        for idx in passing_indices:
            assert idx in snps_indices
            assert snp_mask[idx]  # Must also pass base filter

    def test_hwe_filter_composition(self):
        """HWE filter removes deviating SNPs from mask."""
        # 4 SNPs: 2 in perfect HWE, 2 with extreme deviation
        # Perfect HWE: p=q=0.5 -> n_aa=25, n_ab=50, n_bb=25
        # Extreme deviation: all hom, no het -> n_aa=50, n_ab=0, n_bb=50
        n_aa = np.array([25, 50, 25, 50])
        n_ab = np.array([50, 0, 50, 0])
        n_bb = np.array([25, 50, 25, 50])

        hwe_pvalues = compute_hwe_pvalues(n_aa, n_ab, n_bb)
        hwe_pass = hwe_pvalues >= 0.001

        # SNPs 0 and 2 should pass, 1 and 3 should fail
        assert hwe_pass[0]
        assert not hwe_pass[1]
        assert hwe_pass[2]
        assert not hwe_pass[3]

        # Compose with an all-True base mask
        base_mask = np.ones(4, dtype=bool)
        combined = base_mask & hwe_pass
        assert np.sum(combined) == 2

    def test_snp_list_and_hwe_compose(self):
        """Both SNP list and HWE filters active simultaneously (AND semantics)."""
        # 6 SNPs: indices 0-5
        # SNP list: only 0, 2, 4
        # HWE: SNPs 0 and 4 pass, SNP 2 fails (extreme deviation)
        n_aa = np.array([25, 25, 50, 25, 25, 50])
        n_ab = np.array([50, 50, 0, 50, 50, 0])
        n_bb = np.array([25, 25, 50, 25, 25, 50])

        hwe_pvalues = compute_hwe_pvalues(n_aa, n_ab, n_bb)
        hwe_pass = hwe_pvalues >= 0.001

        snps_indices = np.array([0, 2, 4])
        snp_list_mask = np.zeros(6, dtype=bool)
        snp_list_mask[snps_indices] = True

        base_mask = np.ones(6, dtype=bool)
        combined = base_mask & snp_list_mask & hwe_pass

        # Only SNPs 0 and 4 should survive (in list AND pass HWE)
        surviving = np.where(combined)[0]
        np.testing.assert_array_equal(surviving, [0, 4])
