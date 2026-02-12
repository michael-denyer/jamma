"""Tests for PipelineRunner service class."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jamma.pipeline import PipelineConfig, PipelineRunner

# Fixture paths for gemma_synthetic dataset
FIXTURES = Path(__file__).parent / "fixtures" / "gemma_synthetic"
BFILE = FIXTURES / "test"


class TestPipelineConfig:
    """Tests for PipelineConfig defaults."""

    def test_defaults(self) -> None:
        """PipelineConfig has expected default values."""
        config = PipelineConfig(bfile=Path("test"))
        assert config.kinship_file is None
        assert config.covariate_file is None
        assert config.lmm_mode == 1
        assert config.maf == 0.01
        assert config.miss == 0.05
        assert config.output_dir == Path("output")
        assert config.output_prefix == "result"
        assert config.save_kinship is False
        assert config.check_memory is True
        assert config.show_progress is True
        assert config.mem_budget is None

    def test_custom_values(self) -> None:
        """PipelineConfig accepts custom values."""
        config = PipelineConfig(
            bfile=Path("data/study"),
            kinship_file=Path("k.txt"),
            lmm_mode=4,
            maf=0.05,
            miss=0.1,
            output_dir=Path("results"),
            output_prefix="my_run",
            save_kinship=True,
            check_memory=False,
            mem_budget=64.0,
        )
        assert config.bfile == Path("data/study")
        assert config.kinship_file == Path("k.txt")
        assert config.lmm_mode == 4
        assert config.maf == 0.05
        assert config.mem_budget == 64.0


class TestValidateInputs:
    """Tests for PipelineRunner.validate_inputs."""

    def test_missing_plink_files(self, tmp_path: Path) -> None:
        """validate_inputs raises FileNotFoundError for missing PLINK files."""
        config = PipelineConfig(
            bfile=tmp_path / "nonexistent",
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(FileNotFoundError, match="PLINK .bed file"):
            runner.validate_inputs()

    def test_invalid_lmm_mode(self) -> None:
        """validate_inputs raises ValueError for invalid lmm_mode."""
        config = PipelineConfig(
            bfile=BFILE,
            lmm_mode=99,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="lmm_mode must be"):
            runner.validate_inputs()

    def test_valid_lmm_modes(self) -> None:
        """validate_inputs accepts all valid lmm_mode values."""
        for mode in (1, 2, 3, 4):
            config = PipelineConfig(
                bfile=BFILE,
                lmm_mode=mode,
                check_memory=False,
            )
            runner = PipelineRunner(config)
            runner.validate_inputs()  # Should not raise

    def test_missing_kinship_file(self, tmp_path: Path) -> None:
        """validate_inputs raises FileNotFoundError for missing kinship file."""
        config = PipelineConfig(
            bfile=BFILE,
            kinship_file=tmp_path / "nonexistent.cXX.txt",
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(FileNotFoundError, match="Kinship matrix file"):
            runner.validate_inputs()

    def test_missing_covariate_file(self, tmp_path: Path) -> None:
        """validate_inputs raises FileNotFoundError for missing covariate file."""
        config = PipelineConfig(
            bfile=BFILE,
            covariate_file=tmp_path / "nonexistent.txt",
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(FileNotFoundError, match="Covariate file"):
            runner.validate_inputs()


class TestParsePhenotypes:
    """Tests for PipelineRunner.parse_phenotypes."""

    def test_parse_phenotypes_from_fixture(self, sample_plink_data: Path) -> None:
        """parse_phenotypes reads phenotypes from .fam file."""
        config = PipelineConfig(
            bfile=sample_plink_data,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        phenotypes, n_analyzed = runner.parse_phenotypes()

        assert len(phenotypes) == 100  # gemma_synthetic has 100 samples
        assert n_analyzed > 0
        assert n_analyzed <= 100


class TestCheckMemory:
    """Tests for PipelineRunner.check_memory_requirements."""

    def test_returns_none_when_disabled(self) -> None:
        """check_memory_requirements returns None when check_memory=False."""
        config = PipelineConfig(
            bfile=BFILE,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        result = runner.check_memory_requirements(100, 500)
        assert result is None

    def test_returns_breakdown_when_enabled(self) -> None:
        """check_memory_requirements returns StreamingMemoryBreakdown."""
        from jamma.core.memory import StreamingMemoryBreakdown

        config = PipelineConfig(
            bfile=BFILE,
            check_memory=True,
        )
        runner = PipelineRunner(config)
        result = runner.check_memory_requirements(100, 500)

        assert isinstance(result, StreamingMemoryBreakdown)
        assert result.total_peak_gb >= 0
        assert result.available_gb >= 0


def _copy_plink_genotypes(dest: Path) -> Path:
    """Copy .bed and .bim from gemma_synthetic fixture to dest directory.

    Returns:
        bfile prefix (dest / "test")
    """
    for ext in (".bed", ".bim"):
        shutil.copy(FIXTURES / f"test{ext}", dest / f"test{ext}")
    return dest / "test"


class TestPhenotypeColumnSelection:
    """Tests for phenotype column selection via PipelineConfig.phenotype_column."""

    def test_default_phenotype_column(self, sample_plink_data: Path) -> None:
        """PipelineConfig default phenotype_column=1 produces same result as before."""
        config = PipelineConfig(
            bfile=sample_plink_data,
            check_memory=False,
        )
        assert config.phenotype_column == 1

        runner = PipelineRunner(config)
        phenotypes, n_analyzed = runner.parse_phenotypes()

        assert len(phenotypes) == 100
        assert n_analyzed > 0
        # Verify first value matches fixture (column 6, 0-indexed 5)
        fam_path = f"{sample_plink_data}.fam"
        raw = np.loadtxt(fam_path, dtype=str, usecols=(5,))
        expected_first = float(raw[0])
        assert phenotypes[0] == pytest.approx(expected_first)

    def test_phenotype_column_selects_different_data(self, tmp_path: Path) -> None:
        """Different phenotype_column values return different phenotype vectors."""
        bfile = _copy_plink_genotypes(tmp_path)

        # Write a custom .fam with 3 phenotype columns (8 total columns)
        # Column 6 (pheno 1): 1.0, 2.0, 3.0, ...
        # Column 7 (pheno 2): 4.0, 5.0, 6.0, ...
        # Column 8 (pheno 3): 7.0, 8.0, 9.0, ...
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                pheno1 = 1.0 + i
                pheno2 = 4.0 + i
                pheno3 = 7.0 + i
                f.write(
                    f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno1}\t{pheno2}\t{pheno3}\n"
                )

        # phenotype_column=1 -> first phenotype (column 6)
        config1 = PipelineConfig(bfile=bfile, check_memory=False, phenotype_column=1)
        pheno1, _ = PipelineRunner(config1).parse_phenotypes()

        # phenotype_column=2 -> second phenotype (column 7)
        config2 = PipelineConfig(bfile=bfile, check_memory=False, phenotype_column=2)
        pheno2, _ = PipelineRunner(config2).parse_phenotypes()

        # phenotype_column=3 -> third phenotype (column 8)
        config3 = PipelineConfig(bfile=bfile, check_memory=False, phenotype_column=3)
        pheno3, _ = PipelineRunner(config3).parse_phenotypes()

        # All should be different
        assert not np.array_equal(pheno1, pheno2)
        assert not np.array_equal(pheno2, pheno3)

        # Verify actual values
        assert pheno1[0] == pytest.approx(1.0)
        assert pheno2[0] == pytest.approx(4.0)
        assert pheno3[0] == pytest.approx(7.0)

    def test_phenotype_column_zero_raises(self) -> None:
        """phenotype_column=0 raises ValueError."""
        config = PipelineConfig(
            bfile=BFILE,
            check_memory=False,
            phenotype_column=0,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="phenotype_column must be >= 1"):
            runner.parse_phenotypes()

    def test_phenotype_column_negative_raises(self) -> None:
        """Negative phenotype_column raises ValueError in validate_inputs."""
        config = PipelineConfig(
            bfile=BFILE,
            check_memory=False,
            phenotype_column=-1,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="phenotype_column must be >= 1"):
            runner.validate_inputs()

    def test_phenotype_column_too_large_raises(self) -> None:
        """phenotype_column exceeding .fam columns raises ValueError."""
        config = PipelineConfig(
            bfile=BFILE,
            check_memory=False,
            phenotype_column=99,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="exceeds available columns"):
            runner.parse_phenotypes()


class TestPipelineConfigSnpsFields:
    """Tests for PipelineConfig SNP filtering fields."""

    def test_snps_fields_defaults(self) -> None:
        """PipelineConfig has correct defaults for SNP filtering fields."""
        config = PipelineConfig(bfile=Path("test"))
        assert config.snps_file is None
        assert config.ksnps_file is None
        assert config.hwe_threshold == 0.0

    def test_snps_fields_custom(self) -> None:
        """PipelineConfig accepts custom SNP filtering values."""
        config = PipelineConfig(
            bfile=Path("test"),
            snps_file=Path("snps.txt"),
            ksnps_file=Path("ksnps.txt"),
            hwe_threshold=0.001,
        )
        assert config.snps_file == Path("snps.txt")
        assert config.ksnps_file == Path("ksnps.txt")
        assert config.hwe_threshold == 0.001


class TestValidateInputsSnpsFields:
    """Tests for validate_inputs SNP filtering validation."""

    def test_snps_file_not_found(self, tmp_path: Path) -> None:
        """validate_inputs raises FileNotFoundError for missing snps_file."""
        config = PipelineConfig(
            bfile=BFILE,
            snps_file=tmp_path / "nonexistent_snps.txt",
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(FileNotFoundError, match="SNP list file not found"):
            runner.validate_inputs()

    def test_ksnps_file_not_found(self, tmp_path: Path) -> None:
        """validate_inputs raises FileNotFoundError for missing ksnps_file."""
        config = PipelineConfig(
            bfile=BFILE,
            ksnps_file=tmp_path / "nonexistent_ksnps.txt",
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(FileNotFoundError, match="Kinship SNP list file not found"):
            runner.validate_inputs()

    def test_negative_hwe_raises(self) -> None:
        """validate_inputs raises ValueError for negative hwe_threshold."""
        config = PipelineConfig(
            bfile=BFILE,
            hwe_threshold=-0.1,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="hwe_threshold must be >= 0"):
            runner.validate_inputs()


class TestPhenotypeColumnMissingValues:
    """Tests for missing value handling in non-default phenotype columns."""

    def test_phenotype_column_with_missing_values(self, tmp_path: Path) -> None:
        """Missing value handling works correctly for non-default phenotype columns."""
        bfile = _copy_plink_genotypes(tmp_path)

        # Write .fam with 2 phenotype columns; column 7 (pheno 2) has missing values
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                pheno1 = 1.0 + i
                if i == 0:
                    pheno2_str = "NA"
                elif i == 1:
                    pheno2_str = "-9"
                else:
                    pheno2_str = str(10.0 + i)
                f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno1}\t{pheno2_str}\n")

        config = PipelineConfig(bfile=bfile, check_memory=False, phenotype_column=2)
        phenotypes, n_analyzed = PipelineRunner(config).parse_phenotypes()

        # First two samples should be NaN (NA and -9)
        assert np.isnan(phenotypes[0])
        assert np.isnan(phenotypes[1])
        # Third sample should be valid
        assert phenotypes[2] == pytest.approx(12.0)
        # n_analyzed should exclude the 2 missing
        assert n_analyzed == n_samples - 2
