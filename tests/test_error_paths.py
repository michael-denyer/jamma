"""Error-path tests for covariate.py and pipeline.py.

Covers failure conditions (malformed inputs, dimension mismatches, missing data)
that are currently untested, closing the most significant coverage gaps
in JAMMA's two most undertested modules.

ERRP-01: covariate.py error paths
ERRP-02: pipeline.py error paths
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.io.covariate import encode_categorical_covariates, read_covariate_file

FIXTURES = Path(__file__).parent / "fixtures" / "gemma_synthetic"
BFILE = FIXTURES / "test"


@pytest.mark.tier0
class TestCovariateErrorPaths:
    """Error-path tests for read_covariate_file and encode_categorical_covariates."""

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """read_covariate_file raises ValueError for an empty file."""
        cov_path = tmp_path / "empty.txt"
        cov_path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            read_covariate_file(cov_path)

    def test_whitespace_only_file_raises(self, tmp_path: Path) -> None:
        """read_covariate_file raises ValueError for a whitespace-only file."""
        cov_path = tmp_path / "whitespace.txt"
        cov_path.write_text("   \n\n   \n")

        with pytest.raises(ValueError, match="empty"):
            read_covariate_file(cov_path)

    def test_ragged_rows_raises(self, tmp_path: Path) -> None:
        """read_covariate_file raises ValueError when rows have inconsistent columns."""
        cov_path = tmp_path / "ragged.txt"
        cov_path.write_text("1 2 3\n1 2\n")

        with pytest.raises(ValueError, match="row 2"):
            read_covariate_file(cov_path)

    def test_non_numeric_raises(self, tmp_path: Path) -> None:
        """read_covariate_file raises ValueError for non-numeric values (not 'NA')."""
        cov_path = tmp_path / "nonnumeric.txt"
        cov_path.write_text("1 foo 3\n")

        with pytest.raises(ValueError, match="foo|cannot parse"):
            read_covariate_file(cov_path)

    def test_na_cascades_to_indicator(self, tmp_path: Path) -> None:
        """NA values set indicator=0 and covariates are NaN; non-NA values preserved."""
        cov_path = tmp_path / "na.txt"
        cov_path.write_text("1.0 2.0\n1.0 NA\n1.0 3.0\n")

        covariates, indicator = read_covariate_file(cov_path)

        assert indicator[0] == 1
        assert indicator[1] == 0
        assert indicator[2] == 1
        assert np.isnan(covariates[1, 1])
        assert covariates[0, 1] == pytest.approx(2.0)
        assert covariates[2, 1] == pytest.approx(3.0)

    def test_encode_categorical_out_of_range_raises(self) -> None:
        """encode_categorical_covariates raises ValueError for column index > n_cvt."""
        cov = np.array([[1.0, 0.0], [1.0, 1.0]])

        with pytest.raises(ValueError, match="out of range"):
            encode_categorical_covariates(cov, cat_columns=[5])

    def test_encode_categorical_single_level_removed(self) -> None:
        """Single-level categorical column (no NaN) is removed entirely."""
        # 2-column array: intercept + constant categorical (all 1.0)
        cov = np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        result = encode_categorical_covariates(cov, cat_columns=[2])

        # Column 2 had only one level (1.0), no dummies created, column removed
        assert result.shape == (3, 1), (
            f"Expected shape (3, 1) after removing constant categorical column, "
            f"got {result.shape}"
        )
        np.testing.assert_array_equal(result[:, 0], [1.0, 1.0, 1.0])

    def test_encode_categorical_single_level_with_nan_kept_as_marker(self) -> None:
        """Single-level categorical with NaN rows is kept as NaN marker column."""
        # 2-column array: intercept + categorical with 1 level + NaN rows
        cov = np.array(
            [
                [1.0, 2.0],
                [1.0, np.nan],  # NaN row
                [1.0, 2.0],
                [1.0, np.nan],  # NaN row
            ]
        )

        result = encode_categorical_covariates(cov, cat_columns=[2])

        # Column has 1 non-NaN level + NaN rows — kept as NaN marker
        assert result.shape == (4, 2), (
            f"Expected shape (4, 2) with NaN marker column kept, got {result.shape}"
        )
        # Intercept preserved
        np.testing.assert_array_equal(result[:, 0], [1.0, 1.0, 1.0, 1.0])
        # NaN rows remain NaN; valid rows become 0 (marker column)
        assert result[0, 1] == pytest.approx(0.0)
        assert np.isnan(result[1, 1])
        assert result[2, 1] == pytest.approx(0.0)
        assert np.isnan(result[3, 1])
