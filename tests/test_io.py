"""Tests for PLINK I/O functionality."""

from pathlib import Path

import numpy as np
import pytest

from jamma.io import load_plink_binary


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
