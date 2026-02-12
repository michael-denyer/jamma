"""Tests for SNP filter functions including HWE computation.

Validates compute_hwe_pvalues for equilibrium, deviation, degenerate,
vectorized, and textbook cases.
"""

import numpy as np
import pytest

from jamma.core.snp_filter import compute_hwe_pvalues


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
