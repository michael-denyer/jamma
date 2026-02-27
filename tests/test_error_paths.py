"""Error-path tests for covariate.py and pipeline.py.

Covers failure conditions (malformed inputs, dimension mismatches, missing data)
that are currently untested, closing the most significant coverage gaps
in JAMMA's two most undertested modules.

ERRP-01: covariate.py error paths
ERRP-02: pipeline.py error paths
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jamma.io.covariate import encode_categorical_covariates, read_covariate_file
from jamma.pipeline import PipelineConfig, PipelineResult, PipelineRunner

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

        with pytest.raises(ValueError, match="cannot parse"):
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


@pytest.mark.tier0
class TestPipelineErrorPaths:
    """Error-path tests for PipelineConfig, PipelineResult, and PipelineRunner."""

    def test_output_prefix_with_separator_raises(self) -> None:
        """PipelineConfig raises ValueError when output_prefix has path separators."""
        with pytest.raises(ValueError, match="path separators"):
            PipelineConfig(bfile=Path("test"), output_prefix="dir/prefix")

    def test_pipeline_result_invalid_backend_raises(self) -> None:
        """PipelineResult raises ValueError for unrecognised backend value."""
        with pytest.raises(ValueError, match="must be"):
            PipelineResult(
                associations=[],
                n_samples=10,
                n_snps_tested=5,
                assoc_path=Path("t.txt"),
                backend="invalid",  # type: ignore[arg-type]
            )

    def test_pipeline_config_invalid_backend_raises(self) -> None:
        """PipelineConfig raises ValueError for unrecognised backend value."""
        with pytest.raises(ValueError, match="backend must be"):
            PipelineConfig(bfile=Path("test"), backend="gpu")  # type: ignore[arg-type]

    def test_all_phenotypes_missing_raises(self, tmp_path: Path) -> None:
        """parse_phenotypes raises ValueError when all phenotypes are -9 (missing)."""
        # Copy .bed and .bim from fixture; overwrite .fam with all-missing phenotypes
        for ext in (".bed", ".bim", ".fam"):
            shutil.copy(FIXTURES / f"test{ext}", tmp_path / f"test{ext}")

        fam_path = tmp_path / "test.fam"
        with open(fam_path) as f:
            n_samples = sum(1 for _ in f)

        with open(fam_path, "w") as f:
            for i in range(n_samples):
                f.write(f"FAM{i:03d} IND{i:03d} 0 0 0 -9\n")

        config = PipelineConfig(
            bfile=tmp_path / "test",
            check_memory=False,
        )
        with pytest.raises(ValueError, match="No samples"):
            PipelineRunner(config).parse_phenotypes()

    def test_covariate_dimension_mismatch_raises(self, tmp_path: Path) -> None:
        """load_covariates raises ValueError when covariate row count != n_samples."""
        cov_path = tmp_path / "cov.txt"
        cov_path.write_text("1 2\n3 4\n")  # 2 rows

        config = PipelineConfig(
            bfile=BFILE,
            covariate_file=cov_path,
            check_memory=False,
        )
        with pytest.raises(ValueError, match="rows"):
            PipelineRunner(config).load_covariates(n_samples=100)

    def test_missing_intercept_warning(self, tmp_path: Path) -> None:
        """load_covariates emits a loguru warning when first column is not all 1s."""
        from loguru import logger

        # Determine sample count from BFILE .fam
        fam_path = Path(f"{BFILE}.fam")
        with open(fam_path) as f:
            n_samples = sum(1 for _ in f)

        cov_path = tmp_path / "cov.txt"
        with open(cov_path, "w") as f:
            for i in range(n_samples):
                f.write(f"{2 + i} {3 + i}\n")  # First column is NOT all 1s

        config = PipelineConfig(
            bfile=BFILE,
            covariate_file=cov_path,
            check_memory=False,
        )

        # Capture loguru messages directly (compatible with pytest-xdist workers)
        captured_messages: list[str] = []
        handler_id = logger.add(
            lambda msg: captured_messages.append(msg),
            level="WARNING",
            format="{message}",
        )
        try:
            PipelineRunner(config).load_covariates(n_samples=n_samples)
        finally:
            logger.remove(handler_id)

        assert any("intercept" in m for m in captured_messages), (
            f"Expected 'intercept' warning from loguru, got: {captured_messages!r}"
        )
