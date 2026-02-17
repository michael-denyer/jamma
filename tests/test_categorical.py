"""Tests for categorical covariate encoding."""

import numpy as np
import pytest

from jamma.io.covariate import encode_categorical_covariates


class TestEncodeCategoricalBasic:
    """Tests for basic categorical encoding behavior."""

    def test_encode_basic(self) -> None:
        """Three-level categorical produces 2 dummy columns with reference dropped."""
        # 5 samples, 3 columns: intercept, continuous, categorical(1,2,3,1,2)
        covariates = np.array(
            [
                [1.0, 0.5, 1.0],
                [1.0, 1.2, 2.0],
                [1.0, 0.8, 3.0],
                [1.0, 1.5, 1.0],
                [1.0, 0.3, 2.0],
            ]
        )

        result = encode_categorical_covariates(covariates, cat_columns=[3])

        # Column 3 (values 1,2,3) -> reference=1 dropped, dummies for 2 and 3
        # Result: intercept, continuous, dummy_2, dummy_3
        assert result.shape == (5, 4)

        # Intercept preserved
        np.testing.assert_array_equal(result[:, 0], [1.0, 1.0, 1.0, 1.0, 1.0])
        # Continuous preserved
        np.testing.assert_array_equal(result[:, 1], [0.5, 1.2, 0.8, 1.5, 0.3])
        # Dummy for level 2: samples 1,4 have value 2
        np.testing.assert_array_equal(result[:, 2], [0.0, 1.0, 0.0, 0.0, 1.0])
        # Dummy for level 3: sample 2 has value 3
        np.testing.assert_array_equal(result[:, 3], [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_encode_binary(self) -> None:
        """Binary categorical produces 1 dummy column."""
        covariates = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        )

        result = encode_categorical_covariates(covariates, cat_columns=[2])

        # Column 2 (values 0,1) -> reference=0 dropped, 1 dummy for level 1
        assert result.shape == (4, 2)
        # Intercept preserved
        np.testing.assert_array_equal(result[:, 0], [1.0, 1.0, 1.0, 1.0])
        # Dummy for level 1
        np.testing.assert_array_equal(result[:, 1], [0.0, 1.0, 0.0, 1.0])

    def test_encode_multiple_columns(self) -> None:
        """Encoding columns 1 and 3 expands both correctly."""
        # 4 samples, 3 columns: cat_A(1,2), continuous, cat_B(10,20,30)
        covariates = np.array(
            [
                [1.0, 5.0, 10.0],
                [2.0, 6.0, 20.0],
                [1.0, 7.0, 30.0],
                [2.0, 8.0, 10.0],
            ]
        )

        result = encode_categorical_covariates(covariates, cat_columns=[1, 3])

        # Col 1 (1,2) -> 1 dummy (ref=1, dummy for 2)
        # Col 3 (10,20,30) -> 2 dummies (ref=10, dummies for 20 and 30)
        # Total: 1 dummy + continuous + 2 dummies = 4 columns
        assert result.shape == (4, 4)

        # Continuous column (originally col 2) should be in the middle
        np.testing.assert_array_equal(result[:, 1], [5.0, 6.0, 7.0, 8.0])

    def test_encode_preserves_nan(self) -> None:
        """NaN in categorical column propagated to all dummy columns."""
        covariates = np.array(
            [
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, np.nan],
                [1.0, 3.0],
            ]
        )

        result = encode_categorical_covariates(covariates, cat_columns=[2])

        # Column 2 (1,2,NaN,3) -> ref=1, dummies for 2 and 3
        assert result.shape == (4, 3)
        # Row with NaN should have NaN in all dummy columns
        assert np.isnan(result[2, 1])
        assert np.isnan(result[2, 2])
        # Non-NaN rows should be fine
        np.testing.assert_array_equal(result[0, 1:], [0.0, 0.0])  # level 1 = ref
        np.testing.assert_array_equal(result[1, 1:], [1.0, 0.0])  # level 2
        np.testing.assert_array_equal(result[3, 1:], [0.0, 1.0])  # level 3

    def test_encode_single_level_warns_and_drops(self) -> None:
        """Single-level categorical with no NaN is removed entirely."""
        covariates = np.array(
            [
                [1.0, 5.0, 7.0],
                [1.0, 5.0, 7.0],
                [1.0, 5.0, 7.0],
            ]
        )

        # Column 3 has all same value (7.0) -> 0 dummies
        result = encode_categorical_covariates(covariates, cat_columns=[3])

        # The categorical column is removed entirely (no dummies)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[:, 0], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result[:, 1], [5.0, 5.0, 5.0])

    def test_encode_single_level_with_nan_preserves_missingness(self) -> None:
        """Single-level categorical with NaN keeps a NaN marker column.

        Regression test: previously the column was deleted entirely, removing
        the NaN signal. Pipeline valid-mask (np.all(~np.isnan(covariates), axis=1))
        would then include rows that should be excluded.
        """
        # Column 2: single non-NaN level (1.0) plus NaN -> 0 dummies,
        # but NaN must survive for pipeline filtering
        covariates = np.array(
            [
                [1.0, 1.0],
                [1.0, np.nan],
                [1.0, 1.0],
            ]
        )

        result = encode_categorical_covariates(covariates, cat_columns=[2])

        # Column should be replaced with NaN marker, not deleted
        assert result.shape == (3, 2)
        # NaN row must still contain NaN
        assert np.isnan(result[1, 1])
        # Non-NaN rows should be 0 (marker value)
        assert result[0, 1] == 0.0
        assert result[2, 1] == 0.0
        # Pipeline valid-mask check: the NaN row is excluded
        valid_mask = np.all(~np.isnan(result), axis=1)
        np.testing.assert_array_equal(valid_mask, [True, False, True])

    def test_encode_invalid_column_index_too_high(self) -> None:
        """Column index > n_cvt raises ValueError."""
        covariates = np.array([[1.0, 2.0], [1.0, 3.0]])

        with pytest.raises(ValueError, match="out of range"):
            encode_categorical_covariates(covariates, cat_columns=[3])

    def test_encode_invalid_column_index_zero(self) -> None:
        """Column index 0 raises ValueError (must be 1-indexed)."""
        covariates = np.array([[1.0, 2.0], [1.0, 3.0]])

        with pytest.raises(ValueError, match="out of range"):
            encode_categorical_covariates(covariates, cat_columns=[0])

    def test_encode_intercept_preserved(self) -> None:
        """First column (intercept) is not altered by encoding other columns."""
        covariates = np.array(
            [
                [1.0, 0.5, 1.0],
                [1.0, 1.2, 2.0],
                [1.0, 0.8, 1.0],
                [1.0, 1.5, 2.0],
            ]
        )

        result = encode_categorical_covariates(covariates, cat_columns=[3])

        # Intercept column should be unchanged
        np.testing.assert_array_equal(result[:, 0], [1.0, 1.0, 1.0, 1.0])

    def test_encode_duplicate_cat_columns_deduplicated(self) -> None:
        """Duplicate -cat indices are deduplicated, not double-encoded.

        Regression test: passing [2, 2] would encode column 2 twice,
        second pass operating on dummies from the first pass.
        """
        covariates = np.array(
            [
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
            ]
        )

        result_dedup = encode_categorical_covariates(covariates, cat_columns=[2, 2])
        result_single = encode_categorical_covariates(covariates, cat_columns=[2])

        np.testing.assert_array_equal(result_dedup, result_single)
