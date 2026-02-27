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


@pytest.mark.tier1
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

    def test_profile_dir_default(self) -> None:
        """profile_dir defaults to None."""
        config = PipelineConfig(bfile=Path("/tmp/test"))
        assert config.profile_dir is None

    def test_profile_dir_set(self, tmp_path: Path) -> None:
        """profile_dir can be set to a Path."""
        config = PipelineConfig(
            bfile=Path("/tmp/test"), profile_dir=tmp_path / "traces"
        )
        assert config.profile_dir == tmp_path / "traces"

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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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

    def test_hwe_upper_bound_raises(self) -> None:
        """validate_inputs raises ValueError for hwe_threshold > 1.0."""
        config = PipelineConfig(
            bfile=BFILE,
            hwe_threshold=1.5,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="hwe_threshold must be in"):
            runner.validate_inputs()

    def test_hwe_with_loco_raises(self) -> None:
        """validate_inputs raises ValueError for -hwe combined with -loco."""
        config = PipelineConfig(
            bfile=BFILE,
            hwe_threshold=0.001,
            loco=True,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="not yet supported with -loco"):
            runner.validate_inputs()


@pytest.mark.tier1
class TestPipelineConfigLambdaBounds:
    """Tests for PipelineConfig lambda bounds (l_min, l_max)."""

    def test_lambda_bounds_defaults(self) -> None:
        """PipelineConfig has correct defaults for lambda bounds."""
        config = PipelineConfig(bfile=Path("test"))
        assert config.l_min == 1e-5
        assert config.l_max == 1e5

    def test_lambda_bounds_custom(self) -> None:
        """PipelineConfig accepts custom lambda bounds."""
        config = PipelineConfig(
            bfile=BFILE,
            l_min=1e-3,
            l_max=1e3,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        runner.validate_inputs()  # Should not raise
        assert config.l_min == 1e-3
        assert config.l_max == 1e3

    def test_lambda_lmin_zero_raises(self) -> None:
        """validate_inputs raises ValueError for l_min=0."""
        config = PipelineConfig(
            bfile=BFILE,
            l_min=0,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="l_min must be > 0"):
            runner.validate_inputs()

    def test_lambda_lmin_negative_raises(self) -> None:
        """validate_inputs raises ValueError for negative l_min."""
        config = PipelineConfig(
            bfile=BFILE,
            l_min=-1e-5,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="l_min must be > 0"):
            runner.validate_inputs()

    def test_lambda_lmax_less_than_lmin_raises(self) -> None:
        """validate_inputs raises ValueError when l_max <= l_min."""
        config = PipelineConfig(
            bfile=BFILE,
            l_min=1e-3,
            l_max=1e-4,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="l_max must be > l_min"):
            runner.validate_inputs()

    def test_lambda_lmax_equals_lmin_raises(self) -> None:
        """validate_inputs raises ValueError when l_max == l_min."""
        config = PipelineConfig(
            bfile=BFILE,
            l_min=1.0,
            l_max=1.0,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="l_max must be > l_min"):
            runner.validate_inputs()


@pytest.mark.tier1
class TestPipelineConfigWeightFile:
    """Tests for PipelineConfig weight_file and pipeline weight application."""

    def test_weight_file_default_none(self) -> None:
        """PipelineConfig weight_file defaults to None."""
        config = PipelineConfig(bfile=Path("test"))
        assert config.weight_file is None

    def test_weight_file_not_found(self, tmp_path: Path) -> None:
        """validate_inputs raises FileNotFoundError for missing weight file."""
        config = PipelineConfig(
            bfile=BFILE,
            weight_file=tmp_path / "nonexistent_weights.txt",
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(FileNotFoundError, match="Weight file not found"):
            runner.validate_inputs()

    def test_weight_file_with_loco_raises(self) -> None:
        """validate_inputs raises ValueError for -widv with -loco."""
        weight_path = FIXTURES / "test.fam"  # Use any existing file
        config = PipelineConfig(
            bfile=BFILE,
            weight_file=weight_path,
            loco=True,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="not yet supported with -loco"):
            runner.validate_inputs()

    def test_weight_file_with_eigen_raises(self, tmp_path: Path) -> None:
        """validate_inputs raises ValueError for -widv with -d/-u."""
        weight_path = FIXTURES / "test.fam"  # Use any existing file
        # Create dummy eigen files
        d_file = tmp_path / "test.eigenD.txt"
        u_file = tmp_path / "test.eigenU.txt"
        d_file.write_text("1.0\n")
        u_file.write_text("1.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            weight_file=weight_path,
            eigenvalue_file=d_file,
            eigenvector_file=u_file,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="cannot be used with -d/-u"):
            runner.validate_inputs()

    def test_weight_file_applied_to_kinship(self, tmp_path: Path) -> None:
        """Pipeline applies weights to kinship matrix when weight_file is set."""
        # Create a weight file with non-trivial weights
        weight_file = tmp_path / "weights.txt"
        with open(weight_file, "w") as f:
            for _ in range(100):
                f.write("4.0\n")  # All weights = 4.0

        # Run pipeline with weights
        config_weighted = PipelineConfig(
            bfile=BFILE,
            lmm_mode=1,
            maf=0.01,
            miss=0.05,
            output_dir=tmp_path / "weighted",
            check_memory=False,
            show_progress=False,
            weight_file=weight_file,
        )

        # Run pipeline without weights for comparison
        config_unweighted = PipelineConfig(
            bfile=BFILE,
            lmm_mode=1,
            maf=0.01,
            miss=0.05,
            output_dir=tmp_path / "unweighted",
            check_memory=False,
            show_progress=False,
        )

        runner_w = PipelineRunner(config_weighted)
        runner_u = PipelineRunner(config_unweighted)

        # Load kinship with and without weights
        K_weighted = runner_w.load_kinship(100)
        K_unweighted = runner_u.load_kinship(100)

        # With uniform weights=4.0, K_weighted[i,j] = K[i,j] / sqrt(4*4) = K[i,j] / 4
        np.testing.assert_allclose(K_weighted, K_unweighted / 4.0, rtol=1e-10)


@pytest.mark.tier1
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


# ===========================================================================
# Regression Tests for Correctness Bugs (Phase 29.6)
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.requires_jax
class TestX64GuaranteedWithPrecomputedEigen:
    """Regression test for Bug 1 (jamma-4x8): x64 not guaranteed when
    using precomputed eigen files, which bypass kinship/compute.py.
    """

    def test_x64_guaranteed_with_precomputed_eigen(
        self, sample_plink_data: Path, tmp_path: Path
    ) -> None:
        """PipelineRunner.run() enables x64 even when eigendecomposition is precomputed.

        This reproduces the exact failure scenario: precomputed eigen files bypass
        kinship compute, which was the only place ensure_jax_configured() was called.
        """
        import jax

        import jamma.core.jax_config as jc

        # Step 1: Generate valid eigen files from the fixture data
        from jamma.kinship import compute_kinship_streaming
        from jamma.lmm.eigen import eigendecompose_kinship
        from jamma.lmm.eigen_io import write_eigen_files

        K = compute_kinship_streaming(
            sample_plink_data, check_memory=False, show_progress=False
        )
        eigenvalues, eigenvectors = eigendecompose_kinship(K, check_memory=False)
        d_path, u_path = write_eigen_files(eigenvalues, eigenvectors, tmp_path, "test")

        # Step 2: Reset the JAX config guard to simulate fresh process
        original_state = jc._jax_configured
        jc._jax_configured = False

        try:
            # Step 3: Run pipeline with precomputed eigen (no kinship compute path)
            config = PipelineConfig(
                bfile=sample_plink_data,
                eigenvalue_file=d_path,
                eigenvector_file=u_path,
                output_dir=tmp_path / "output",
                check_memory=False,
                show_progress=False,
            )
            result = PipelineRunner(config).run()

            # Step 4: Verify x64 is enabled after the run
            assert jax.config.jax_enable_x64 is True, (
                "JAX x64 should be enabled after pipeline run with precomputed eigen"
            )
            assert jc._jax_configured is True, (
                "ensure_jax_configured should have been called"
            )
            assert result.n_samples > 0
            assert result.n_snps_tested > 0
        finally:
            # Restore the original state
            jc._jax_configured = original_state


@pytest.mark.tier1
@pytest.mark.requires_jax
class TestNSamplesReflectsCovariateFiltering:
    """Regression test for Bug 2 (jamma-ri0): n_samples reported
    phenotype-only count instead of post-covariate-filter count.
    """

    def test_n_samples_reflects_covariate_filtering(self, tmp_path: Path) -> None:
        """PipelineResult.n_samples excludes samples with NaN covariates.

        Creates a covariate file where some rows have NaN values, then
        verifies that n_samples is strictly less than the total sample count
        and equals the expected valid count after both phenotype and
        covariate filtering.
        """
        bfile = _copy_plink_genotypes(tmp_path)

        # Write a .fam with all valid phenotypes
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                pheno = 1.0 + i * 0.1
                f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno}\n")

        # Write a covariate file with intercept + one covariate, 10 rows have NaN
        n_nan_covariates = 10
        cov_path = tmp_path / "covariates.txt"
        with open(cov_path, "w") as f:
            for i in range(n_samples):
                intercept = 1.0
                if i < n_nan_covariates:
                    cov = "NA"  # These samples should be filtered out
                else:
                    cov = str(0.5 + i * 0.01)
                f.write(f"{intercept}\t{cov}\n")

        config = PipelineConfig(
            bfile=bfile,
            covariate_file=cov_path,
            output_dir=tmp_path / "output",
            check_memory=False,
            show_progress=False,
        )
        result = PipelineRunner(config).run()

        expected_valid = n_samples - n_nan_covariates
        assert result.n_samples == expected_valid, (
            f"n_samples should be {expected_valid} (after covariate NaN filtering), "
            f"got {result.n_samples}"
        )
        assert result.n_samples < n_samples, (
            "n_samples should be less than total sample count when covariates have NaN"
        )


@pytest.mark.tier1
@pytest.mark.requires_jax
class TestNSnpsTestedReflectsFiltering:
    """Regression test for Bug 3 (jamma-bza): n_snps_tested reported
    dataset total instead of post-filter count.
    """

    def test_n_snps_tested_reflects_maf_filtering(
        self, sample_plink_data: Path, tmp_path: Path
    ) -> None:
        """PipelineResult.n_snps_tested is less than dataset total with strict MAF.

        Uses a high MAF threshold to filter out some SNPs, then verifies
        that n_snps_tested reflects the filtered count, not the dataset total.
        """
        from jamma.io.plink import get_plink_metadata

        meta = get_plink_metadata(sample_plink_data)
        total_snps_in_dataset = meta["n_snps"]

        # Run pipeline with a restrictive MAF threshold
        config = PipelineConfig(
            bfile=sample_plink_data,
            maf=0.1,  # Restrictive MAF to filter out some SNPs
            output_dir=tmp_path / "output",
            check_memory=False,
            show_progress=False,
        )
        result = PipelineRunner(config).run()

        assert result.n_snps_tested < total_snps_in_dataset, (
            f"n_snps_tested ({result.n_snps_tested}) should be less than "
            f"dataset total ({total_snps_in_dataset}) with maf=0.1 filter"
        )
        assert result.n_snps_tested > 0, "Should still have some SNPs passing filter"

    def test_n_snps_tested_with_snps_file(
        self, sample_plink_data: Path, tmp_path: Path
    ) -> None:
        """PipelineResult.n_snps_tested reflects SNP list restriction.

        Creates a -snps file that restricts to a subset, then verifies
        n_snps_tested matches the restricted count (not dataset total).
        """
        from jamma.io.plink import get_plink_metadata

        meta = get_plink_metadata(sample_plink_data)
        total_snps_in_dataset = meta["n_snps"]

        # Restrict to first 50 SNPs via -snps file
        n_restrict = 50
        snps_path = tmp_path / "snps.txt"
        snps_path.write_text("\n".join(meta["sid"][:n_restrict]) + "\n")

        config = PipelineConfig(
            bfile=sample_plink_data,
            snps_file=snps_path,
            maf=0.0,  # Permissive MAF to isolate the SNP list effect
            miss=1.0,  # Permissive miss to isolate the SNP list effect
            output_dir=tmp_path / "output",
            check_memory=False,
            show_progress=False,
        )
        result = PipelineRunner(config).run()

        assert result.n_snps_tested <= n_restrict, (
            f"n_snps_tested ({result.n_snps_tested}) should be <= "
            f"{n_restrict} (SNP list size)"
        )
        assert result.n_snps_tested < total_snps_in_dataset, (
            f"n_snps_tested ({result.n_snps_tested}) should be less than "
            f"dataset total ({total_snps_in_dataset})"
        )


# ===========================================================================
# Backend Routing Tests (Phase 36-02)
# ===========================================================================


@pytest.mark.tier1
def test_pipeline_numpy_backend(sample_plink_data: Path, output_dir: Path) -> None:
    """Pipeline completes with NumPy backend and writes assoc file."""
    kinship_file = sample_plink_data.parent / "gemma_kinship.cXX.txt"
    config = PipelineConfig(
        bfile=sample_plink_data,
        kinship_file=kinship_file,
        lmm_mode=1,
        output_dir=output_dir,
        check_memory=False,
        show_progress=False,
        backend="numpy",
    )
    result = PipelineRunner(config).run()
    assert result.n_snps_tested > 0
    assert result.assoc_path.exists()
    assert result.backend == "numpy"


@pytest.mark.tier1
def test_pipeline_loco_numpy(tmp_path: Path) -> None:
    """LOCO + NumPy backend completes end-to-end without error."""
    # gemma_loco fixture: 100 samples, 500 SNPs across 3 chromosomes
    loco_bfile = Path(__file__).parent / "fixtures" / "gemma_loco" / "test"
    config = PipelineConfig(
        bfile=loco_bfile,
        lmm_mode=1,
        loco=True,
        backend="numpy",
        output_dir=tmp_path / "output",
        check_memory=False,
        show_progress=False,
    )
    result = PipelineRunner(config).run()
    assert result.backend == "numpy"
    assert result.n_snps_tested > 0
    assert result.assoc_path.exists()


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_pipeline_backend_in_timing(sample_plink_data: Path, output_dir: Path) -> None:
    """PipelineResult.backend is 'jax' when JAX backend is used."""
    kinship_file = sample_plink_data.parent / "gemma_kinship.cXX.txt"
    config = PipelineConfig(
        bfile=sample_plink_data,
        kinship_file=kinship_file,
        lmm_mode=1,
        output_dir=output_dir,
        check_memory=False,
        show_progress=False,
        backend="jax",
    )
    result = PipelineRunner(config).run()
    assert result.backend == "jax"


@pytest.mark.tier0
def test_pipeline_config_backend_validation() -> None:
    """PipelineConfig raises ValueError for invalid backend value."""
    with pytest.raises(ValueError, match="backend must be"):
        PipelineConfig(bfile=Path("dummy"), backend="invalid")


@pytest.mark.tier1
@pytest.mark.parametrize("lmm_mode", [2, 3, 4], ids=["LRT", "Score", "All"])
def test_pipeline_numpy_backend_modes(
    sample_plink_data: Path, tmp_path: Path, lmm_mode: int
) -> None:
    """T3: Pipeline completes with NumPy backend for modes 2, 3, and 4."""
    kinship_file = sample_plink_data.parent / "gemma_kinship.cXX.txt"
    out = tmp_path / f"output_mode{lmm_mode}"
    out.mkdir()
    config = PipelineConfig(
        bfile=sample_plink_data,
        kinship_file=kinship_file,
        lmm_mode=lmm_mode,
        output_dir=out,
        check_memory=False,
        show_progress=False,
        backend="numpy",
    )
    result = PipelineRunner(config).run()
    assert result.n_snps_tested > 0
    assert result.assoc_path.exists()
    assert result.backend == "numpy"


@pytest.mark.tier1
def test_pipeline_numpy_with_snps_file(sample_plink_data: Path, tmp_path: Path) -> None:
    """T8: Pipeline NumPy backend works with -snps file filtering."""
    from jamma.io.plink import get_plink_metadata

    meta = get_plink_metadata(sample_plink_data)
    total_snps = meta["n_snps"]

    # Restrict to first 30 SNPs
    n_restrict = 30
    snps_path = tmp_path / "snps.txt"
    snps_path.write_text("\n".join(meta["sid"][:n_restrict]) + "\n")

    kinship_file = sample_plink_data.parent / "gemma_kinship.cXX.txt"
    out = tmp_path / "output_snps"
    out.mkdir()
    config = PipelineConfig(
        bfile=sample_plink_data,
        kinship_file=kinship_file,
        snps_file=snps_path,
        lmm_mode=1,
        maf=0.0,
        miss=1.0,
        output_dir=out,
        check_memory=False,
        show_progress=False,
        backend="numpy",
    )
    result = PipelineRunner(config).run()
    assert result.n_snps_tested <= n_restrict
    assert result.n_snps_tested < total_snps
    assert result.assoc_path.exists()
    assert result.backend == "numpy"
