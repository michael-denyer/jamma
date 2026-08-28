"""Tests for kinship computation helpers.

These tests exercise impute_and_center, impute_center_and_standardize, and
compute_centered_kinship.
"""

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship import impute_and_center, impute_center_and_standardize
from jamma.validation import compare_kinship_matrices, load_gemma_kinship
from tests.conftest import require_fixture
from tests.fixture_paths import SYNTHETIC
from tests.reference.kinship import compute_centered_kinship


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
