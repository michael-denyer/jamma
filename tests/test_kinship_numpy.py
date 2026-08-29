"""Tests for kinship computation helpers.

These tests exercise impute_and_center, impute_center_and_standardize, and
compute_centered_kinship. The property-based classes (filtering, standardized
kinship) moved here from test_hypothesis.py, which is organised by technique
rather than subsystem.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jamma.io import load_plink_binary
from jamma.kinship import impute_and_center, impute_center_and_standardize
from jamma.validation import compare_kinship_matrices, load_gemma_kinship
from tests.conftest import require_fixture
from tests.fixture_paths import SYNTHETIC
from tests.reference.kinship import compute_centered_kinship


@st.composite
def genotype_matrix(draw, min_samples=10, max_samples=100, min_snps=5, max_snps=50):
    """Generate realistic genotype matrices (values in {0, 1, 2}).

    Uses a random seed to generate genotypes with realistic variance.
    """
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_snps = draw(st.integers(min_value=min_snps, max_value=max_snps))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    rng = np.random.default_rng(seed)

    # Use varying MAFs to ensure non-constant columns
    mafs = rng.uniform(0.1, 0.5, n_snps)
    genotypes = np.zeros((n_samples, n_snps), dtype=np.float64)

    for j in range(n_snps):
        p = mafs[j]
        # Hardy-Weinberg genotype frequencies
        probs = [(1 - p) ** 2, 2 * p * (1 - p), p**2]
        genotypes[:, j] = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=probs)

    return genotypes


@pytest.mark.tier0
class TestMonomorphismMaskBasis:
    """The single-pass kinship loop selects columns via compute_snp_stats.

    It used np.nanvar > 0 to drop monomorphic columns per chunk; that mask is
    equal to compute_snp_stats(chunk).col_vars > 0 on genotype data, so the loop
    now uses the canonical stats path. These tests pin that equality on the edge
    cases the swap depends on, so the single-pass filter cannot silently diverge
    from the two-pass filter.
    """

    @staticmethod
    def _nanvar_mask(chunk: np.ndarray) -> np.ndarray:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanvar(chunk, axis=0) > 0

    @staticmethod
    def _stats_mask(chunk: np.ndarray) -> np.ndarray:
        from jamma.core.snp_filter import compute_snp_stats

        _means, _miss, col_vars = compute_snp_stats(chunk)
        return col_vars > 0

    def test_masks_agree_on_genotype_data(self) -> None:
        rng = np.random.default_rng(1)
        chunk = rng.integers(0, 3, size=(64, 40)).astype(np.float64)
        chunk[rng.random((64, 40)) < 0.15] = np.nan
        np.testing.assert_array_equal(self._nanvar_mask(chunk), self._stats_mask(chunk))

    def test_masks_agree_on_edge_columns(self) -> None:
        # Columns crafted to hit every monomorphism edge case.
        chunk = np.array(
            [
                [np.nan, 1.0, 1.0, 2.0, 0.0],
                [np.nan, 1.0, np.nan, 2.0, 0.0],
                [np.nan, 1.0, np.nan, np.nan, 2.0],
                [np.nan, 1.0, np.nan, 2.0, 2.0],
            ]
        )
        # col 0 all-NaN, col 1 constant, col 2 single value, col 3 constant+NaN,
        # col 4 polymorphic.
        expected = np.array([False, False, False, False, True])
        np.testing.assert_array_equal(self._nanvar_mask(chunk), expected)
        np.testing.assert_array_equal(self._stats_mask(chunk), expected)


@pytest.mark.tier0
class TestImputeAndCenterInPlace:
    """Tests for KIN-03: impute_and_center modifies input in-place."""

    def test_returns_same_object(self):
        """Return value is the same object as input (no copy)."""
        X = np.array([[0.0, 1.0], [np.nan, 2.0], [2.0, 1.0]])
        result = impute_and_center(X)
        assert result is X, "impute_and_center must return the same array object"

    def test_numerical_correctness_with_nans(self):
        """Numerical output matches expected values after in-place imputation."""
        X = np.array([[0.0, 1.0], [np.nan, 2.0], [2.0, 1.0]])
        result = impute_and_center(X)
        # Column 0: mean = (0+2)/2 = 1.0, NaN->1.0, centered: [-1, 0, 1]
        # Column 1: mean = (1+2+1)/3 = 4/3, centered: [1-4/3, 2-4/3, 1-4/3]
        expected_col0 = np.array([-1.0, 0.0, 1.0])
        expected_col1 = np.array([1.0, 2.0, 1.0]) - 4.0 / 3.0
        np.testing.assert_allclose(result[:, 0], expected_col0, atol=1e-14)
        np.testing.assert_allclose(result[:, 1], expected_col1, atol=1e-14)

    def test_all_missing_column(self):
        """All-NaN column produces zeros after centering."""
        X = np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 1.0]])
        result = impute_and_center(X)
        np.testing.assert_array_equal(result[:, 0], 0.0)

    def test_no_missing_values(self):
        """Matrix without NaN is centered correctly in-place."""
        X = np.array(
            [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0]], dtype=np.float64
        )
        X_ref = X.copy()
        result = impute_and_center(X)
        assert result is X
        expected = X_ref - X_ref.mean(axis=0, keepdims=True)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_read_only_array_uses_fallback(self):
        """Read-only numpy array takes the copy-based fallback path."""
        X = np.array([[0.0, 1.0], [np.nan, 2.0], [2.0, 1.0]])
        X_ref = X.copy()
        X.flags.writeable = False

        result = impute_and_center(X)

        # Must return a new object (not the read-only input)
        assert result is not X
        # Original must be unmodified
        np.testing.assert_array_equal(X_ref[~np.isnan(X_ref)], X[~np.isnan(X)])
        # Numerical output must match the writable path
        expected = impute_and_center(X_ref)
        np.testing.assert_allclose(result, expected, atol=1e-14)


@pytest.mark.tier0
class TestImputeCenterStandardizeEinsum:
    """Tests for KIN-06: einsum variance replaces X**2 intermediate."""

    def test_numerical_equivalence(self):
        """einsum variance produces identical output to np.mean(X**2, axis=0)."""
        rng = np.random.default_rng(42)
        X = rng.choice([0.0, 1.0, 2.0], size=(100, 50))
        # Inject some NaN
        X[0, 3] = np.nan
        X[10, 20] = np.nan
        X[50, 0] = np.nan

        result = impute_center_and_standardize(X.copy())

        # Compute reference using the old method (inline)
        X_ref = X.copy()
        snp_means = np.nanmean(X_ref, axis=0, keepdims=True)
        snp_means = np.nan_to_num(snp_means, nan=0.0)
        X_imputed = np.where(np.isnan(X_ref), snp_means, X_ref)
        X_centered = X_imputed - snp_means
        snp_var = np.mean(X_centered**2, axis=0, keepdims=True)
        snp_sd = np.sqrt(snp_var)
        expected = np.where(snp_sd > 0, X_centered / snp_sd, 0.0)

        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_zero_variance_snp(self):
        """Monomorphic SNP (zero variance) produces zero column."""
        X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        result = impute_center_and_standardize(X)
        # Column 0 is constant (variance=0) -> should be all zeros
        np.testing.assert_array_equal(result[:, 0], 0.0)
        # Column 1 has variance -> should be non-zero
        assert np.any(result[:, 1] != 0.0)


@pytest.mark.tier0
class TestKinshipNoCopy:
    """Tests for KIN-02: no redundant float64 copy in _compute_kinship_inmemory."""

    def test_float64_input_produces_correct_kinship(self):
        """compute_centered_kinship on float64 input produces correct result.

        When input is already float64, the dtype check avoids creating a second
        copy. This test verifies the function still produces numerically correct
        output after the optimization.
        """
        X = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 0]], dtype=np.float64)
        K = compute_centered_kinship(X, check_memory=False)
        assert K.shape == (3, 3)
        assert K.dtype == np.float64
        np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_float32_input_produces_correct_kinship(self):
        """compute_centered_kinship on float32 input produces same result as float64.

        When input is float32, the single astype(float64) allocation avoids the
        intermediate float32 boolean-indexed copy followed by astype. This test
        verifies the result matches reference computed from float64 input.
        """
        rng = np.random.default_rng(42)
        X_f64 = rng.integers(0, 3, size=(10, 20)).astype(np.float64)
        X_f32 = X_f64.astype(np.float32)

        K_from_f64 = compute_centered_kinship(X_f64, check_memory=False)
        K_from_f32 = compute_centered_kinship(X_f32, check_memory=False)

        # float32 input should produce identical result (both converted to float64)
        np.testing.assert_allclose(K_from_f32, K_from_f64, rtol=1e-10, atol=1e-12)


@pytest.mark.tier1
class TestImputationGemmaEquivalence:
    """TEST-01: GEMMA equivalence and property tests for imputation functions.

    The centered kinship test proves equivalence transitively: impute genotypes
    -> compute kinship -> compare to GEMMA reference. Remaining tests validate
    structural properties of standardized kinship and NaN imputation completeness.
    """

    def test_impute_and_center_kinship_matches_gemma(self):
        """Centered kinship from impute_and_center matches GEMMA -gk 1 reference."""
        require_fixture(SYNTHETIC.bed, SYNTHETIC.kinship)
        plink = load_plink_binary(SYNTHETIC.bfile)
        X = plink.genotypes.astype(np.float64)
        X_centered = impute_and_center(X)
        K_jamma = X_centered @ X_centered.T / X_centered.shape[1]
        K_gemma = load_gemma_kinship(SYNTHETIC.kinship)
        result = compare_kinship_matrices(K_jamma, K_gemma)
        assert result.passed, (
            f"Kinship from impute_and_center does not match GEMMA reference:\n"
            f"Max abs diff: {result.max_abs_diff:.2e}\n"
            f"Max rel diff: {result.max_rel_diff:.2e}\n"
            f"Worst location: {result.worst_location}"
        )

    def test_impute_center_standardize_produces_valid_standardized_kinship(self):
        """Standardized kinship has correct structural properties."""
        require_fixture(SYNTHETIC.bed)
        plink = load_plink_binary(SYNTHETIC.bfile)
        X = plink.genotypes.astype(np.float64)
        X_std = impute_center_and_standardize(X.copy())
        K_std = X_std @ X_std.T / X_std.shape[1]
        # Shape and symmetry
        n = X_std.shape[0]
        assert K_std.shape == (n, n)
        np.testing.assert_allclose(K_std, K_std.T, atol=1e-14)
        # Diagonal positive (each sample has non-zero self-similarity)
        assert np.all(K_std.diagonal() > 0)
        # Standardized kinship differs from centered kinship
        X2 = plink.genotypes.astype(np.float64)
        X_centered = impute_and_center(X2)
        K_centered = X_centered @ X_centered.T / X_centered.shape[1]
        assert not np.allclose(K_std, K_centered, atol=1e-6), (
            "Standardized and centered kinship should differ"
        )

    def test_impute_and_center_no_nans_remain(self):
        """All NaN values imputed after impute_and_center."""
        require_fixture(SYNTHETIC.bed)
        plink = load_plink_binary(SYNTHETIC.bfile)
        X = plink.genotypes.astype(np.float64)
        # Inject missing values if fixture has none, ensuring imputation is exercised
        if not np.isnan(X).any():
            rng = np.random.default_rng(42)
            mask = rng.random(X.shape) < 0.05
            X[mask] = np.nan
        assert np.isnan(X).any(), "Expected NaN values before imputation"
        result = impute_and_center(X)
        assert not np.isnan(result).any(), "NaN values remain after impute_and_center"


@pytest.mark.tier0
class TestFilteringProperties:
    """Property-based tests for MAF/missing/monomorphic filtering."""

    @given(
        n_samples=st.integers(min_value=50, max_value=100),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_monomorphic_snps_always_filtered(self, n_samples, seed):
        """Monomorphic SNPs should always be filtered regardless of MAF threshold."""
        from tests.reference.kinship import _filter_snps

        rng = np.random.default_rng(seed)

        # Create genotypes with some monomorphic SNPs
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, 10))
        # Make first 3 SNPs monomorphic
        genotypes[:, 0] = 0.0
        genotypes[:, 1] = 1.0
        genotypes[:, 2] = 2.0

        # Even with maf=0.0 (no MAF filtering), monomorphic should be filtered
        filtered, n_filtered, n_original = _filter_snps(
            genotypes, maf_threshold=0.0, miss_threshold=1.0
        )

        # Verify monomorphic SNPs are excluded
        assert n_filtered <= n_original - 3, (
            f"Expected at least 3 monomorphic SNPs filtered, "
            f"got {n_original - n_filtered}"
        )

        # Check that all remaining columns have variance > 0
        if n_filtered > 0:
            col_vars = np.var(filtered, axis=0)
            assert np.all(col_vars > 0), "Filtered genotypes contain monomorphic SNPs"

    @given(
        n_samples=st.integers(min_value=50, max_value=100),
        maf_threshold=st.floats(min_value=0.0, max_value=0.49),
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_filtered_snps_meet_maf_threshold(self, n_samples, maf_threshold):
        """All SNPs passing filter should have MAF >= threshold."""
        from tests.reference.kinship import _filter_snps

        rng = np.random.default_rng(42)

        # Generate genotypes with varying MAF
        genotypes = np.zeros((n_samples, 20), dtype=np.float64)
        for j in range(20):
            maf = rng.uniform(0.01, 0.5)
            p = maf
            probs = [(1 - p) ** 2, 2 * p * (1 - p), p**2]
            genotypes[:, j] = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=probs)

        filtered, n_filtered, _ = _filter_snps(
            genotypes, maf_threshold=maf_threshold, miss_threshold=1.0
        )

        if n_filtered > 0:
            # Compute MAF of filtered SNPs
            col_means = np.nanmean(filtered, axis=0)
            allele_freqs = col_means / 2.0
            mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)

            # All should be >= threshold (with small tolerance for numerical error)
            assert np.all(mafs >= maf_threshold - 1e-10), (
                f"SNP with MAF {mafs.min():.4f} passed threshold {maf_threshold}"
            )

    @given(
        n_samples=st.integers(min_value=50, max_value=100),
        miss_threshold=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_filtered_snps_meet_miss_threshold(self, n_samples, miss_threshold):
        """All SNPs passing filter should have missing rate <= threshold."""
        from tests.reference.kinship import _filter_snps

        rng = np.random.default_rng(42)

        # Generate genotypes with varying missing rates
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, 20)).astype(np.float64)

        # Add random missing values
        for j in range(20):
            miss_rate = rng.uniform(0.0, 0.3)
            n_miss = int(n_samples * miss_rate)
            if n_miss > 0:
                miss_idx = rng.choice(n_samples, size=n_miss, replace=False)
                genotypes[miss_idx, j] = np.nan

        filtered, n_filtered, _ = _filter_snps(
            genotypes, maf_threshold=0.0, miss_threshold=miss_threshold
        )

        if n_filtered > 0:
            # Compute missing rate of filtered SNPs
            miss_counts = np.sum(np.isnan(filtered), axis=0)
            miss_rates = miss_counts / n_samples

            # All should be <= threshold
            assert np.all(miss_rates <= miss_threshold + 1e-10), (
                f"SNP with missing rate {miss_rates.max():.4f} "
                f"passed threshold {miss_threshold}"
            )

    @given(
        n_samples=st.integers(min_value=50, max_value=100),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(
        max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_filtering_deterministic(self, n_samples, seed):
        """Filtering should be deterministic given same input."""
        from tests.reference.kinship import _filter_snps

        rng = np.random.default_rng(seed)
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, 20)).astype(np.float64)

        # Add some missing
        miss_mask = rng.random(genotypes.shape) < 0.05
        genotypes[miss_mask] = np.nan

        # Run twice
        filtered1, n1, _ = _filter_snps(genotypes, 0.01, 0.1)
        filtered2, n2, _ = _filter_snps(genotypes, 0.01, 0.1)

        assert n1 == n2, "Filtering produced different counts"
        np.testing.assert_array_equal(
            np.isnan(filtered1),
            np.isnan(filtered2),
            err_msg="Filtering not deterministic",
        )


@pytest.mark.slow
@pytest.mark.tier2
class TestFilteringEquivalence:
    """Tests for filtering equivalence across different code paths."""

    @given(
        n_samples=st.integers(min_value=50, max_value=80),
        n_snps=st.integers(min_value=20, max_value=50),
        maf_threshold=st.sampled_from([0.0, 0.01, 0.05]),
        miss_threshold=st.sampled_from([0.05, 0.1, 1.0]),
    )
    @settings(
        max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_kinship_full_load_vs_filter_function(
        self, n_samples, n_snps, maf_threshold, miss_threshold
    ):
        """Full-load kinship should use same filtering as _filter_snps."""
        from tests.reference.kinship import _filter_snps, compute_centered_kinship

        rng = np.random.default_rng(42)

        # Generate genotypes with varying MAF
        genotypes = np.zeros((n_samples, n_snps), dtype=np.float64)
        for j in range(n_snps):
            maf = rng.uniform(0.02, 0.5)
            p = maf
            probs = [(1 - p) ** 2, 2 * p * (1 - p), p**2]
            genotypes[:, j] = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=probs)

        # Add some missing
        miss_mask = rng.random(genotypes.shape) < 0.03
        genotypes[miss_mask] = np.nan

        # Get filter result
        _, n_filtered, _ = _filter_snps(genotypes, maf_threshold, miss_threshold)

        if n_filtered == 0:
            # Should raise ValueError
            with pytest.raises(ValueError, match="No SNPs passed filtering"):
                compute_centered_kinship(
                    genotypes,
                    maf_threshold=maf_threshold,
                    miss_threshold=miss_threshold,
                    check_memory=False,
                )
        else:
            # Should succeed and produce valid kinship
            K = compute_centered_kinship(
                genotypes,
                maf_threshold=maf_threshold,
                miss_threshold=miss_threshold,
                check_memory=False,
            )
            assert K.shape == (n_samples, n_samples)
            assert np.allclose(K, K.T), "Kinship not symmetric"


@pytest.mark.tier0
class TestStandardizedKinshipProperties:
    """Property tests for standardized kinship matrix (-gk 2)."""

    @given(
        genotypes=genotype_matrix(
            min_samples=20, max_samples=40, min_snps=20, max_snps=40
        )
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_standardized_kinship_is_symmetric(self, genotypes):
        """Standardized kinship must be symmetric."""
        from tests.reference.kinship import compute_standardized_kinship

        K = compute_standardized_kinship(genotypes, check_memory=False)
        np.testing.assert_allclose(K, K.T, rtol=1e-10, atol=1e-14)

    @given(
        genotypes=genotype_matrix(
            min_samples=20, max_samples=40, min_snps=20, max_snps=40
        )
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_standardized_kinship_psd(self, genotypes):
        """Standardized kinship eigenvalues should be non-negative."""
        from tests.reference.kinship import compute_standardized_kinship

        K = compute_standardized_kinship(genotypes, check_memory=False)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8), (
            f"Large negative eigenvalue: {eigenvalues.min()}"
        )

    @given(
        genotypes=genotype_matrix(
            min_samples=20, max_samples=40, min_snps=30, max_snps=50
        )
    )
    @settings(
        max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_standardized_kinship_trace_approximates_n(self, genotypes):
        """Standardized kinship trace should approximate n_samples.

        K = (1/p) * Z @ Z.T where each column of Z has unit variance.
        So trace(K) = (1/p) * sum of column norms^2. Each column has
        variance ~1 and n samples, so norm^2 ~ n, giving trace ~ n.
        """
        from tests.reference.kinship import compute_standardized_kinship

        K = compute_standardized_kinship(genotypes, check_memory=False)
        n_samples = genotypes.shape[0]
        trace = np.trace(K)

        # Trace should be approximately n_samples (within ~30% for small samples)
        assert trace > 0, f"Trace should be positive, got {trace}"
        assert abs(trace - n_samples) / n_samples < 0.5, (
            f"Trace {trace:.2f} too far from n_samples {n_samples}"
        )

    @given(
        genotypes=genotype_matrix(
            min_samples=20, max_samples=40, min_snps=20, max_snps=40
        )
    )
    @settings(
        max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_centered_and_standardized_same_shape(self, genotypes):
        """Centered and standardized kinship should produce same-shaped matrices."""
        from tests.reference.kinship import (
            compute_centered_kinship,
            compute_standardized_kinship,
        )

        K_centered = compute_centered_kinship(genotypes, check_memory=False)
        K_standardized = compute_standardized_kinship(genotypes, check_memory=False)

        assert K_centered.shape == K_standardized.shape
        # Both should be symmetric
        np.testing.assert_allclose(K_centered, K_centered.T, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(
            K_standardized, K_standardized.T, rtol=1e-10, atol=1e-14
        )
