"""Tests for I/O functionality (PLINK, covariates)."""

from pathlib import Path

import numpy as np
import pytest

from jamma.io import load_plink_binary, read_covariate_file


class TestLoadPlinkBinary:
    """Tests for load_plink_binary function."""

    def test_load_plink_binary_shape(self, sample_plink_data: Path) -> None:
        """Verify genotypes shape matches expected dimensions."""
        data = load_plink_binary(sample_plink_data)

        # gemma_synthetic has 100 samples and 500 SNPs
        assert data.genotypes.shape == (100, 500)

    def test_load_plink_binary_metadata_lengths(self, sample_plink_data: Path) -> None:
        """Verify metadata arrays have correct lengths."""
        data = load_plink_binary(sample_plink_data)

        assert len(data.iid) == 100, "iid length should match n_samples"
        assert len(data.sid) == 500, "sid length should match n_snps"
        assert len(data.chromosome) == 500, "chromosome length should match n_snps"
        assert len(data.bp_position) == 500, "bp_position length should match n_snps"
        assert len(data.allele_1) == 500, "allele_1 length should match n_snps"
        assert len(data.allele_2) == 500, "allele_2 length should match n_snps"

    def test_load_plink_binary_genotype_values(self, sample_plink_data: Path) -> None:
        """Verify genotype values are in valid set {0, 1, 2, NaN}."""
        data = load_plink_binary(sample_plink_data)

        # Get non-NaN values
        non_nan_values = data.genotypes[~np.isnan(data.genotypes)]

        # All non-NaN values should be 0, 1, or 2
        unique_values = np.unique(non_nan_values)
        valid_values = {0.0, 1.0, 2.0}
        assert set(unique_values).issubset(
            valid_values
        ), f"Genotype values should be in {{0, 1, 2}}, got {unique_values}"

    def test_load_plink_binary_dtype(self, sample_plink_data: Path) -> None:
        """Verify genotypes are returned as float32."""
        data = load_plink_binary(sample_plink_data)

        assert data.genotypes.dtype == np.float32

    def test_load_plink_binary_missing_file(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="PLINK .bed file not found"):
            load_plink_binary(nonexistent)


class TestPlinkDataProperties:
    """Tests for PlinkData dataclass properties."""

    def test_n_samples_property(self, sample_plink_data: Path) -> None:
        """Verify n_samples property returns correct count."""
        data = load_plink_binary(sample_plink_data)

        assert data.n_samples == 100

    def test_n_snps_property(self, sample_plink_data: Path) -> None:
        """Verify n_snps property returns correct count."""
        data = load_plink_binary(sample_plink_data)

        assert data.n_snps == 500

    def test_properties_match_shape(self, sample_plink_data: Path) -> None:
        """Verify n_samples and n_snps match genotypes shape."""
        data = load_plink_binary(sample_plink_data)

        assert data.n_samples == data.genotypes.shape[0]
        assert data.n_snps == data.genotypes.shape[1]


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

    def test_covariate_empty_file_error(self, tmp_path: Path) -> None:
        """Verify ValueError for empty file."""
        cov_file = tmp_path / "empty.txt"
        cov_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            read_covariate_file(cov_file)

    def test_covariate_whitespace_only_file_error(self, tmp_path: Path) -> None:
        """Verify ValueError for file with only whitespace."""
        cov_file = tmp_path / "whitespace.txt"
        cov_file.write_text("   \n\t\n  \n")

        with pytest.raises(ValueError, match="empty"):
            read_covariate_file(cov_file)

    def test_covariate_column_mismatch_error(self, tmp_path: Path) -> None:
        """Verify ValueError for inconsistent column counts."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1 35.0 0\n1 42.0\n")  # Row 2 missing column

        with pytest.raises(ValueError, match="column"):
            read_covariate_file(cov_file)

    def test_covariate_non_numeric_error(self, tmp_path: Path) -> None:
        """Verify ValueError for non-numeric values (not NA)."""
        cov_file = tmp_path / "covariates.txt"
        cov_file.write_text("1 abc 0\n")

        with pytest.raises(ValueError, match="abc"):
            read_covariate_file(cov_file)

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
