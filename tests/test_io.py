"""Tests for I/O functionality (PLINK, covariates)."""

from pathlib import Path

import numpy as np
import pytest

from jamma.genotype.dataset import GenotypeDataset
from jamma.io import read_covariate_file

pytestmark = pytest.mark.tier0


class TestOpenPlink:
    """PLINK metadata read through GenotypeDataset.open_plink."""

    def test_open_plink_shape(self, sample_plink_data: Path) -> None:
        """Verify sample and variant counts match expected dimensions."""
        dataset = GenotypeDataset.open_plink(sample_plink_data)

        # gemma_synthetic has 100 samples and 500 SNPs
        assert (dataset.n_samples, dataset.n_variants) == (100, 500)

    def test_open_plink_metadata_lengths(self, sample_plink_data: Path) -> None:
        """Verify metadata arrays have correct lengths."""
        dataset = GenotypeDataset.open_plink(sample_plink_data)
        variants = dataset.variants

        assert len(dataset.samples.iid) == 100, "iid length should match n_samples"
        assert len(variants.rs) == 500, "rs length should match n_snps"
        assert len(variants.chr) == 500, "chr length should match n_snps"
        assert len(variants.pos) == 500, "pos length should match n_snps"
        assert len(variants.a1) == 500, "a1 length should match n_snps"
        assert len(variants.a0) == 500, "a0 length should match n_snps"

    def test_open_plink_missing_file(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match=r"PLINK .bed file not found"):
            GenotypeDataset.open_plink(nonexistent)


class TestReadCovariateFile:
    """Tests for read_covariate_file function."""

    def test_covariate_basic_parsing(self, tmp_path: Path) -> None:
        """Verify basic parsing with intercept and covariates."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1 35.0 0\n1 42.0 1\n1 28.0 0\n1 55.0 1\n")

        covariates, indicator = read_covariate_file(cov_file)

        assert covariates.shape == (4, 3)
        assert indicator.shape == (4,)
        assert covariates.dtype == np.float64
        assert indicator.dtype == np.int32
        # All rows valid (no NA)
        np.testing.assert_array_equal(indicator, [1, 1, 1, 1])

    def test_covariate_na_handling(self, tmp_path: Path) -> None:
        """Verify NA values are converted to NaN and indicator set to 0."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1 35.0 0\n1 NA 1\n1 28.0 NA\n1 55.0 1\n")

        covariates, indicator = read_covariate_file(cov_file)

        # Rows 2 and 3 have NA, should be marked invalid
        np.testing.assert_array_equal(indicator, [1, 0, 0, 1])
        # NaN values in correct positions
        assert np.isnan(covariates[1, 1])  # Row 2, col 2
        assert np.isnan(covariates[2, 2])  # Row 3, col 3
        # Other values still valid
        assert covariates[0, 1] == 35.0
        assert covariates[3, 2] == 1.0

    def test_covariate_tab_delimited(self, tmp_path: Path) -> None:
        """Verify tab-delimited files are parsed correctly."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1\t35.0\t0\n1\t42.0\t1\n")

        covariates, indicator = read_covariate_file(cov_file)

        assert covariates.shape == (2, 3)
        assert covariates[0, 1] == 35.0
        assert covariates[1, 2] == 1.0

    def test_covariate_mixed_whitespace(self, tmp_path: Path) -> None:
        """Verify mixed spaces and tabs are handled."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1   35.0\t0\n1\t42.0   1\n")

        covariates, indicator = read_covariate_file(cov_file)

        assert covariates.shape == (2, 3)
        np.testing.assert_array_equal(indicator, [1, 1])

    # Covariate error-path tests (empty, whitespace, column mismatch, non-numeric)
    # live in test_error_paths.py::TestCovariateErrorPaths to avoid duplication.

    def test_covariate_na_case_sensitive(self, tmp_path: Path) -> None:
        """Verify NA is case-sensitive (lowercase 'na' is invalid)."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1 na 0\n")

        # Lowercase 'na' should fail (not recognized as missing)
        with pytest.raises(ValueError, match="na"):
            read_covariate_file(cov_file)

    def test_covariate_intercept_detection(self, tmp_path: Path) -> None:
        """Verify first column with all 1s is parseable (intercept column)."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1 10.0\n1 20.0\n1 30.0\n")

        covariates, indicator = read_covariate_file(cov_file)

        # First column should be all 1s
        np.testing.assert_array_equal(covariates[:, 0], [1.0, 1.0, 1.0])
        # All rows valid
        np.testing.assert_array_equal(indicator, [1, 1, 1])

    def test_covariate_skip_empty_lines(self, tmp_path: Path) -> None:
        """Verify empty lines are skipped (GEMMA behavior)."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1 35.0\n\n1 42.0\n\n\n1 28.0\n")

        covariates, indicator = read_covariate_file(cov_file)

        # Only 3 data rows (empty lines skipped)
        assert covariates.shape == (3, 2)
        np.testing.assert_array_equal(covariates[:, 1], [35.0, 42.0, 28.0])

    def test_covariate_scientific_notation(self, tmp_path: Path) -> None:
        """Verify scientific notation values are parsed correctly."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1 1.5e-3 2.0E+02\n1 -3.14e0 0\n")

        covariates, indicator = read_covariate_file(cov_file)

        assert covariates.shape == (2, 3)
        np.testing.assert_allclose(covariates[0, 1], 1.5e-3)
        np.testing.assert_allclose(covariates[0, 2], 200.0)
        np.testing.assert_allclose(covariates[1, 1], -3.14)
