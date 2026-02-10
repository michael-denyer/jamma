"""Tests for PipelineRunner service class."""

from __future__ import annotations

from pathlib import Path

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
