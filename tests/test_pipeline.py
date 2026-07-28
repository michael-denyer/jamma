"""Tests for PipelineRunner service class."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.schema import MIN_N_GRID
from jamma.pipeline import PipelineConfig, PipelineRunner
from jamma.pipeline_memory import check_streaming_memory

# Fixture paths for gemma_synthetic dataset
FIXTURES = Path(__file__).parent / "fixtures" / "gemma_synthetic"
BFILE = FIXTURES / "test"


@pytest.mark.tier0
def test_data_classes_still_importable_from_jamma_pipeline():
    """The three data classes moved to ``pipeline_config`` but the old path stays.

    ``from jamma.pipeline import PipelineConfig`` is not a courtesy re-export.
    It is what ``jamma.cli`` and ``jamma.gwas`` use, and it is also the import
    the jamma-databricks notebooks use
    (``databricks_jamma_vs_gemma_numpy.py`` builds a ``PipelineConfig`` and
    hands it to ``PipelineRunner``). That consumer lives outside this repo, so
    nothing here would fail if the re-export were dropped during a later tidy-up.

    Identity rather than importability: a second class definition with the same
    name would satisfy an import check while breaking ``isinstance``.
    """
    from jamma import pipeline, pipeline_config

    for name in ("PipelineConfig", "PipelineResult", "KinshipResult"):
        assert hasattr(pipeline, name), (
            f"jamma.pipeline.{name} disappeared; jamma-databricks imports it"
        )
        assert getattr(pipeline, name) is getattr(pipeline_config, name), (
            f"jamma.pipeline.{name} is not the same object as "
            f"jamma.pipeline_config.{name}; isinstance checks across the two "
            "import paths would silently disagree"
        )


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
        assert config.legacy_text is False
        assert config.phenotype_columns == [1]

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
        with pytest.raises(FileNotFoundError, match=r"PLINK .bed file"):
            runner.validate_inputs()

    def test_invalid_lmm_mode(self) -> None:
        """Construction raises ValueError for invalid lmm_mode."""
        with pytest.raises(ValueError, match="lmm_mode must be"):
            PipelineConfig(bfile=BFILE, lmm_mode=99, check_memory=False)

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
    """Tests for pipeline_memory.check_streaming_memory."""

    def test_returns_none_when_disabled(self) -> None:
        """check_streaming_memory returns None when check_memory=False."""
        config = PipelineConfig(
            bfile=BFILE,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        result = check_streaming_memory(runner.config, 100, 500)
        assert result is None

    def test_returns_breakdown_when_enabled(self) -> None:
        """check_streaming_memory returns StreamingMemoryBreakdown."""
        from jamma.core.memory import StreamingMemoryBreakdown

        config = PipelineConfig(
            bfile=BFILE,
            check_memory=True,
        )
        runner = PipelineRunner(config)
        result = check_streaming_memory(runner.config, 100, 500)

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
    """Tests for phenotype column selection via PipelineConfig.phenotype_columns."""

    def test_default_phenotype_column(self, sample_plink_data: Path) -> None:
        """The default phenotype_columns=[1] reads the standard .fam phenotype."""
        config = PipelineConfig(
            bfile=sample_plink_data,
            check_memory=False,
        )
        assert config.phenotype_columns == [1]

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
        """Different phenotype_columns values return different phenotype vectors."""
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

        config1 = PipelineConfig(bfile=bfile, check_memory=False, phenotype_columns=[1])
        pheno1, _ = PipelineRunner(config1).parse_phenotypes()

        config2 = PipelineConfig(bfile=bfile, check_memory=False, phenotype_columns=[2])
        pheno2, _ = PipelineRunner(config2).parse_phenotypes()

        config3 = PipelineConfig(bfile=bfile, check_memory=False, phenotype_columns=[3])
        pheno3, _ = PipelineRunner(config3).parse_phenotypes()

        # All should be different
        assert not np.array_equal(pheno1, pheno2)
        assert not np.array_equal(pheno2, pheno3)

        # Verify actual values
        assert pheno1[0] == pytest.approx(1.0)
        assert pheno2[0] == pytest.approx(4.0)
        assert pheno3[0] == pytest.approx(7.0)

    def test_phenotype_column_zero_raises(self) -> None:
        """A 0 index is rejected at construction, before any file is read."""
        with pytest.raises(ValueError, match="phenotype_columns indices must be >= 1"):
            PipelineConfig(
                bfile=BFILE,
                check_memory=False,
                phenotype_columns=[0],
            )

    def test_phenotype_column_negative_raises(self) -> None:
        """A negative index is rejected at construction."""
        with pytest.raises(ValueError, match="phenotype_columns indices must be >= 1"):
            PipelineConfig(
                bfile=BFILE,
                check_memory=False,
                phenotype_columns=[-1],
            )

    def test_phenotype_column_too_large_raises(self) -> None:
        """A column the .fam lacks is only detectable once the .fam is read."""
        config = PipelineConfig(
            bfile=BFILE,
            check_memory=False,
            phenotype_columns=[99],
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

    @pytest.mark.parametrize("l_min", [0, -1e-5], ids=["zero", "negative"])
    def test_lambda_lmin_not_positive_raises(self, l_min: float) -> None:
        """Construction rejects a non-positive l_min."""
        with pytest.raises(ValueError, match="l_min must be positive"):
            PipelineConfig(bfile=BFILE, l_min=l_min, check_memory=False)

    @pytest.mark.parametrize(
        ("l_min", "l_max"), [(1e-3, 1e-4), (1.0, 1.0)], ids=["less", "equal"]
    )
    def test_lambda_lmax_not_above_lmin_raises(
        self, l_min: float, l_max: float
    ) -> None:
        """Construction rejects l_max <= l_min."""
        with pytest.raises(ValueError, match="must be greater than l_min"):
            PipelineConfig(bfile=BFILE, l_min=l_min, l_max=l_max, check_memory=False)


@pytest.mark.tier1
class TestPipelineConfigGridResolution:
    """Tests for PipelineConfig n_grid validation.

    Regression: LmmConfig rejected n_grid < 2, but the LOCO branch used to
    forward PipelineConfig.n_grid to run_lmm_loco without ever building an
    LmmConfig, so a one-point grid reached the kernel — after kinship and
    eigendecomposition on the C path, and silently as lambda = l_min on the
    NumPy path. Both branches now build one; PipelineConfig.__post_init__
    builds a throwaway too, which is what makes the failure land here at
    construction rather than mid-run.
    """

    @pytest.mark.parametrize("n_grid", [-5, 0, 1])
    @pytest.mark.parametrize("loco", [False, True], ids=["batch", "loco"])
    def test_n_grid_below_two_raises(self, n_grid: int, loco: bool) -> None:
        """Construction fails for both branches, before any compute happens."""
        with pytest.raises(ValueError, match="n_grid must be >= 2"):
            PipelineConfig(bfile=Path("test"), n_grid=n_grid, loco=loco)

    def test_minimum_grid_accepted(self) -> None:
        """n_grid == MIN_N_GRID is the smallest grid that still brackets."""
        config = PipelineConfig(bfile=Path("test"), n_grid=MIN_N_GRID)
        assert config.n_grid == 2


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


_N_SAMPLES = 100
_NAN_INDICES = {5, 10, 15}


def _write_fam(
    fam_path: Path, n_samples: int = _N_SAMPLES, nan_indices: set[int] | None = None
) -> None:
    """Write a .fam file with optional NaN phenotypes at specified indices."""
    with open(fam_path, "w") as f:
        for i in range(n_samples):
            pheno = "NA" if nan_indices and i in nan_indices else str(1.0 + i * 0.1)
            f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno}\n")


def _valid_indices_excluding(
    n_samples: int = _N_SAMPLES, exclude: set[int] | None = None
) -> np.ndarray:
    """Return sorted array of sample indices excluding the given set."""
    exclude = exclude or _NAN_INDICES
    return np.array([i for i in range(n_samples) if i not in exclude])


@pytest.mark.tier1
class TestEarlySampleFiltering:
    """Tests for early sample filtering before kinship computation."""

    def test_early_sample_filter_pipeline(self, tmp_path: Path) -> None:
        """Early filtering: NaN phenotypes + save_kinship=False.

        Verifies the pipeline computes valid_mask before kinship and
        passes valid_indices, producing identical eigenvalues to a
        direct valid-subset kinship computation.
        """
        from jamma.kinship.compute import compute_kinship_streaming

        bfile = _copy_plink_genotypes(tmp_path)
        _write_fam(tmp_path / "test.fam", nan_indices=_NAN_INDICES)
        valid_indices = _valid_indices_excluding()

        out = tmp_path / "output_early"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            save_kinship=False,
            backend="numpy",
        )

        runner = PipelineRunner(config)
        K_with_vi = runner.load_kinship(_N_SAMPLES, valid_indices=valid_indices)
        n_valid = len(valid_indices)
        assert K_with_vi.shape == (n_valid, n_valid), (
            f"Expected ({n_valid}, {n_valid}) kinship, got {K_with_vi.shape}"
        )

        K_ref = compute_kinship_streaming(
            bfile,
            check_memory=False,
            show_progress=False,
            valid_indices=valid_indices,
        )
        np.testing.assert_allclose(
            K_with_vi,
            K_ref,
            rtol=1e-12,
            err_msg="load_kinship with valid_indices must match direct streaming",
        )

    def test_save_kinship_full_size(self, tmp_path: Path) -> None:
        """save_kinship=True: load_kinship still returns filtered shape."""
        bfile = _copy_plink_genotypes(tmp_path)
        _write_fam(tmp_path / "test.fam")

        out = tmp_path / "output_save"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            save_kinship=True,
            backend="numpy",
        )

        valid_indices = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9])
        K = PipelineRunner(config).load_kinship(_N_SAMPLES, valid_indices=valid_indices)
        assert K.shape == (len(valid_indices), len(valid_indices))

    def test_weight_file_valid_indices(self, tmp_path: Path) -> None:
        """Weights filtered to match valid_indices under early filtering."""
        bfile = _copy_plink_genotypes(tmp_path)
        _write_fam(tmp_path / "test.fam")

        weight_file = tmp_path / "weights.txt"
        np.savetxt(weight_file, np.arange(1.0, _N_SAMPLES + 1.0))

        out = tmp_path / "output_wt"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            weight_file=weight_file,
        )

        valid_indices = _valid_indices_excluding()
        K = PipelineRunner(config).load_kinship(_N_SAMPLES, valid_indices=valid_indices)
        n_valid = len(valid_indices)
        assert K.shape == (n_valid, n_valid), (
            f"Expected ({n_valid}, {n_valid}) with valid_indices, got {K.shape}"
        )

    def test_precomputed_kinship_still_works(self, tmp_path: Path) -> None:
        """Pre-computed kinship from file is subsetted post-load with valid_indices."""
        from jamma.kinship.compute import compute_kinship_streaming

        bfile = _copy_plink_genotypes(tmp_path)
        _write_fam(tmp_path / "test.fam")

        K_full = compute_kinship_streaming(
            bfile, check_memory=False, show_progress=False
        )
        kinship_file = tmp_path / "kinship.cXX.txt"
        np.savetxt(kinship_file, K_full)

        out = tmp_path / "output_precomp"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            kinship_file=kinship_file,
        )

        valid_indices = _valid_indices_excluding()
        K = PipelineRunner(config).load_kinship(_N_SAMPLES, valid_indices=valid_indices)

        n_valid = len(valid_indices)
        assert K.shape == (n_valid, n_valid)
        np.testing.assert_allclose(
            K,
            K_full[np.ix_(valid_indices, valid_indices)],
            rtol=1e-12,
            err_msg="Pre-computed kinship with valid_indices must match np.ix_",
        )

    def test_run_end_to_end_with_nan_phenotypes(self, tmp_path: Path) -> None:
        """Full run() with NaN phenotypes triggers early filtering and completes."""
        bfile = _copy_plink_genotypes(tmp_path)
        _write_fam(tmp_path / "test.fam", nan_indices=_NAN_INDICES)
        n_valid = _N_SAMPLES - len(_NAN_INDICES)

        out = tmp_path / "output_e2e"
        out.mkdir()
        config = PipelineConfig(
            bfile=bfile,
            lmm_mode=1,
            output_dir=out,
            check_memory=False,
            show_progress=False,
            save_kinship=False,
            backend="numpy",
        )

        result = PipelineRunner(config).run()

        assert result.n_samples == n_valid, (
            f"Expected {n_valid} samples after NaN filtering, got {result.n_samples}"
        )
        assert result.n_snps_tested > 0, "Should test at least some SNPs"

    def test_run_end_to_end_save_kinship_with_nan(self, tmp_path: Path) -> None:
        """Full run() with save_kinship=True and NaN phenotypes.

        Verifies save_kinship does not change statistical results: the
        filtered kinship is saved and eigenpairs match the non-save path.
        """
        bfile = _copy_plink_genotypes(tmp_path)
        _write_fam(tmp_path / "test.fam", nan_indices=_NAN_INDICES)
        n_valid = _N_SAMPLES - len(_NAN_INDICES)

        # backend stays out of the dict: splatting it would widen the literal
        # to str and no longer satisfy PipelineConfig's Literal[...] field.
        common_kwargs = {
            "bfile": bfile,
            "lmm_mode": 1,
            "check_memory": False,
            "show_progress": False,
        }

        out_no_save = tmp_path / "output_nosave"
        out_no_save.mkdir()
        result_no_save = PipelineRunner(
            PipelineConfig(
                **common_kwargs,
                backend="numpy",
                output_dir=out_no_save,
                save_kinship=False,
            )
        ).run()

        out_save = tmp_path / "output_save"
        out_save.mkdir()
        result_save = PipelineRunner(
            PipelineConfig(
                **common_kwargs,
                backend="numpy",
                output_dir=out_save,
                save_kinship=True,
            )
        ).run()

        assert result_save.n_samples == result_no_save.n_samples == n_valid
        assert result_save.n_snps_tested == result_no_save.n_snps_tested

        # Saved kinship should be full (n_samples, n_samples) for reuse
        K_saved = np.load(out_save / "result.cXX.npy")
        assert K_saved.shape == (_N_SAMPLES, _N_SAMPLES), (
            f"save_kinship must write full ({_N_SAMPLES}, {_N_SAMPLES}) "
            f"kinship for reuse, got {K_saved.shape}"
        )


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

        config = PipelineConfig(bfile=bfile, check_memory=False, phenotype_columns=[2])
        phenotypes, n_analyzed = PipelineRunner(config).parse_phenotypes()

        # First two samples should be NaN (NA and -9)
        assert np.isnan(phenotypes[0])
        assert np.isnan(phenotypes[1])
        # Third sample should be valid
        assert phenotypes[2] == pytest.approx(12.0)
        # n_analyzed should exclude the 2 missing
        assert n_analyzed == n_samples - 2


# ===========================================================================
# Regression Tests for Correctness Bugs
# ===========================================================================


# ===========================================================================
# Backend Routing Tests
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
def test_pipeline_pve_se_populated(sample_plink_data: Path, output_dir: Path) -> None:
    """PipelineResult.pve_estimate is in (0, 1) and pve_se is None or positive."""
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

    assert result.pve_estimate is not None, (
        "pve_estimate should be populated after a standard run"
    )
    assert 0 < result.pve_estimate < 1, (
        f"pve_estimate should be in (0, 1), got {result.pve_estimate}"
    )
    assert result.pve_se is not None, "pve_se should be populated for synthetic data"
    assert result.pve_se > 0, f"pve_se should be positive, got {result.pve_se}"


@pytest.mark.tier1
def test_pipeline_output_path_content_matches_n_tested(
    sample_plink_data: Path, output_dir: Path
) -> None:
    """Disk file line count matches n_snps_tested and path formula is correct."""
    from jamma.validation import load_gemma_assoc

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

    expected_path = config.output_dir / f"{config.output_prefix}.assoc.txt"
    assert result.assoc_path.exists(), "assoc_path file should exist on disk"
    assert result.assoc_path == expected_path, (
        f"assoc_path {result.assoc_path} does not match expected {expected_path}"
    )
    assert result.assoc_paths == [result.assoc_path], (
        f"assoc_paths should be a single-element list for single-phenotype runs, "
        f"got {result.assoc_paths}"
    )

    disk_results = load_gemma_assoc(result.assoc_path)
    assert len(disk_results) == result.n_snps_tested, (
        f"Disk file has {len(disk_results)} rows but "
        f"n_snps_tested={result.n_snps_tested}"
    )


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


@pytest.mark.tier0
def test_pipeline_config_backend_validation() -> None:
    """PipelineConfig raises ValueError for invalid backend value."""
    with pytest.raises(ValueError, match="backend must be"):
        PipelineConfig(bfile=Path("dummy"), backend="invalid")  # type: ignore[bad-argument-type]


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


# ===========================================================================
# Multi-Phenotype Tests
# ===========================================================================


@pytest.mark.tier1
class TestMultiPhenotypeConfig:
    """Tests for PipelineConfig multi-phenotype support."""

    def test_phenotype_columns_default_single(self) -> None:
        """PipelineConfig() has phenotype_columns==[1] by default."""
        config = PipelineConfig(bfile=Path("test"))
        assert config.phenotype_columns == [1]

    def test_phenotype_columns_explicit(self) -> None:
        """An explicit list is kept in the order it was given."""
        config = PipelineConfig(bfile=Path("test"), phenotype_columns=[1, 2, 3])
        assert config.phenotype_columns == [1, 2, 3]

    def test_empty_phenotype_columns_raises(self) -> None:
        """An empty list is a config error, not a stand-in for the default."""
        with pytest.raises(ValueError, match="must name at least one column"):
            PipelineConfig(bfile=Path("test"), phenotype_columns=[])

    def test_every_phenotype_column_is_range_checked(self) -> None:
        """A bad index is caught wherever it sits, not only at the front."""
        with pytest.raises(ValueError, match="phenotype_columns indices must be >= 1"):
            PipelineConfig(bfile=Path("test"), phenotype_columns=[1, 0])

    def test_loco_multi_phenotype_error(self) -> None:
        """PipelineConfig(loco=True, phenotype_columns=[1,2]) raises ValueError."""
        with pytest.raises(
            ValueError, match=r"LOCO mode.*does not support multi-phenotype"
        ):
            PipelineConfig(bfile=Path("test"), loco=True, phenotype_columns=[1, 2])

    def test_loco_single_phenotype_ok(self) -> None:
        """PipelineConfig(loco=True, phenotype_columns=[1]) is valid."""
        config = PipelineConfig(bfile=Path("test"), loco=True, phenotype_columns=[1])
        assert config.phenotype_columns == [1]


@pytest.mark.tier1
class TestMultiPhenotypeOutputNaming:
    """Tests for multi-phenotype output file naming."""

    def test_single_phenotype_no_suffix(
        self, sample_plink_data: Path, tmp_path: Path
    ) -> None:
        """phenotype_columns=[1] produces result.assoc.txt (no .pheno suffix)."""
        config = PipelineConfig(
            bfile=sample_plink_data,
            phenotype_columns=[1],
            output_dir=tmp_path / "output",
            check_memory=False,
            show_progress=False,
            backend="numpy",
        )
        result = PipelineRunner(config).run()
        assert result.assoc_path.name == "result.assoc.txt"

    def test_multi_phenotype_output_naming(self, tmp_path: Path) -> None:
        """Multi-phenotype produces per-pheno output files."""
        bfile = _copy_plink_genotypes(tmp_path)

        # Write a .fam with 2 phenotype columns
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                pheno1 = 1.0 + i * 0.1
                pheno2 = 2.0 + i * 0.1
                f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno1}\t{pheno2}\n")

        out = tmp_path / "output"
        config = PipelineConfig(
            bfile=bfile,
            phenotype_columns=[1, 2],
            output_dir=out,
            check_memory=False,
            show_progress=False,
            backend="numpy",
        )
        PipelineRunner(config).run()

        # Both per-phenotype output files should exist
        assert (out / "result.pheno1.assoc.txt").exists()
        assert (out / "result.pheno2.assoc.txt").exists()
        # No plain result.assoc.txt
        assert not (out / "result.assoc.txt").exists()


@pytest.mark.tier1
class TestMultiPhenotypeSingleEigen:
    """Tests for eigendecomposition reuse across phenotypes."""

    def test_multi_phenotype_single_eigen(self, tmp_path: Path) -> None:
        """Multi-phenotype mode calls eigendecompose_kinship exactly once."""
        bfile = _copy_plink_genotypes(tmp_path)

        # Write a .fam with 2 phenotype columns
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                pheno1 = 1.0 + i * 0.1
                pheno2 = 2.0 + i * 0.1
                f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno1}\t{pheno2}\n")

        from unittest.mock import patch

        out = tmp_path / "output"
        config = PipelineConfig(
            bfile=bfile,
            phenotype_columns=[1, 2],
            output_dir=out,
            check_memory=False,
            show_progress=False,
            backend="numpy",
        )

        with patch(
            "jamma.pipeline.eigendecompose_kinship",
            wraps=eigendecompose_kinship,
        ) as mock_eigen:
            PipelineRunner(config).run()
            assert mock_eigen.call_count == 1, (
                f"eigendecompose_kinship should be called once, "
                f"got {mock_eigen.call_count}"
            )


@pytest.mark.tier1
class TestMultiPhenotypeMaskIntersection:
    """Tests for multi-phenotype missing value mask intersection."""

    def test_multi_phenotype_shared_missing_excluded(self, tmp_path: Path) -> None:
        """Multi-phenotype with shared missing samples excludes them correctly."""
        bfile = _copy_plink_genotypes(tmp_path)

        # Write .fam with 2 phenotype columns; samples 0 and 1 missing in BOTH
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                if i < 2:
                    pheno1 = "NA"
                    pheno2 = "NA"
                else:
                    pheno1 = str(1.0 + i * 0.1)
                    pheno2 = str(2.0 + i * 0.1)
                f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno1}\t{pheno2}\n")

        out = tmp_path / "output"
        config = PipelineConfig(
            bfile=bfile,
            phenotype_columns=[1, 2],
            output_dir=out,
            check_memory=False,
            show_progress=False,
            backend="numpy",
        )
        result = PipelineRunner(config).run()
        # Both samples 0 and 1 should be excluded
        assert result.n_samples == n_samples - 2

    def test_multi_phenotype_all_missing_raises(self, tmp_path: Path) -> None:
        """Multi-phenotype with complementary missing patterns raises ValueError."""
        bfile = _copy_plink_genotypes(tmp_path)

        # Write .fam where every sample is missing in at least one phenotype
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                if i % 2 == 0:
                    pheno1 = "NA"
                    pheno2 = str(2.0 + i * 0.1)
                else:
                    pheno1 = str(1.0 + i * 0.1)
                    pheno2 = "NA"
                f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno1}\t{pheno2}\n")

        out = tmp_path / "output"
        config = PipelineConfig(
            bfile=bfile,
            phenotype_columns=[1, 2],
            output_dir=out,
            check_memory=False,
            show_progress=False,
            backend="numpy",
        )
        with pytest.raises(ValueError, match="No samples have valid values"):
            PipelineRunner(config).run()

    def test_assoc_paths_multi_phenotype(self, tmp_path: Path) -> None:
        """PipelineResult.assoc_paths contains all per-phenotype output files."""
        bfile = _copy_plink_genotypes(tmp_path)

        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                pheno1 = 1.0 + i * 0.1
                pheno2 = 2.0 + i * 0.1
                f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno1}\t{pheno2}\n")

        out = tmp_path / "output"
        config = PipelineConfig(
            bfile=bfile,
            phenotype_columns=[1, 2],
            output_dir=out,
            check_memory=False,
            show_progress=False,
            backend="numpy",
        )
        result = PipelineRunner(config).run()
        assert len(result.assoc_paths) == 2
        assert all(p.exists() for p in result.assoc_paths)
        assert result.assoc_path == result.assoc_paths[-1]

    def test_assoc_paths_single_phenotype(
        self, sample_plink_data: Path, tmp_path: Path
    ) -> None:
        """Single-phenotype run has assoc_paths == [assoc_path]."""
        config = PipelineConfig(
            bfile=sample_plink_data,
            phenotype_columns=[1],
            output_dir=tmp_path / "output",
            check_memory=False,
            show_progress=False,
            backend="numpy",
        )
        result = PipelineRunner(config).run()
        assert result.assoc_paths == [result.assoc_path]

    def test_duplicate_phenotype_columns_raises(self) -> None:
        """PipelineConfig rejects duplicate phenotype columns."""
        with pytest.raises(ValueError, match="duplicate"):
            PipelineConfig(bfile=Path("test"), phenotype_columns=[1, 1, 3])

    def test_out_of_range_phenotype_column_raises(self, tmp_path: Path) -> None:
        """Multi-phenotype with out-of-range column index raises ValueError."""
        bfile = _copy_plink_genotypes(tmp_path)

        # Write .fam with only 1 phenotype column (column index 5)
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{1.0 + i * 0.1}\n")

        out = tmp_path / "output"
        config = PipelineConfig(
            bfile=bfile,
            phenotype_columns=[1, 5],
            output_dir=out,
            check_memory=False,
            show_progress=False,
            backend="numpy",
        )
        with pytest.raises(ValueError, match="exceeds available columns"):
            PipelineRunner(config).run()


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


@pytest.mark.tier1
def test_pipeline_re_evaluation_passes_n_cvt(
    tmp_path: Path, sample_plink_data: Path
) -> None:
    """BCKAUTO-04: Re-evaluation passes n_cvt from loaded covariates."""
    from unittest.mock import patch

    from jamma.lmm.runner import select_execution_mode

    n_samples = 100  # gemma_synthetic fixture has 100 samples

    # Write a covariate file with 2 columns (intercept + one covariate).
    # GEMMA format: whitespace-separated values, one row per sample, no header.
    cov_path = tmp_path / "covariates.txt"
    rng = np.random.default_rng(42)
    cov_data = np.column_stack([np.ones(n_samples), rng.standard_normal(n_samples)])
    np.savetxt(str(cov_path), cov_data, fmt="%.6f")

    kinship_file = sample_plink_data.parent / "gemma_kinship.cXX.txt"
    out = tmp_path / "output_ncvt"
    out.mkdir()

    # Spy on select_execution_mode to capture all call kwargs
    calls: list[dict] = []
    original_sem = select_execution_mode

    def spy_sem(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_sem(*args, **kwargs)

    config = PipelineConfig(
        bfile=sample_plink_data,
        kinship_file=kinship_file,
        covariate_file=cov_path,
        lmm_mode=1,
        maf=0.0,
        miss=1.0,
        output_dir=out,
        check_memory=False,
        show_progress=False,
        backend="numpy",
    )
    with patch("jamma.pipeline.select_execution_mode", side_effect=spy_sem):
        PipelineRunner(config).run()

    # The re-evaluation call (post-covariate-load) must have n_cvt=2
    assert any(c.get("n_cvt", 1) == 2 for c in calls), (
        f"No call to select_execution_mode had n_cvt=2; calls={calls}"
    )


@pytest.mark.tier1
def test_pipeline_emits_telemetry(tmp_path: Path, sample_plink_data: Path) -> None:
    """Telemetry record is emitted with expected fields after a pipeline run."""
    from unittest.mock import patch

    from jamma.core.telemetry import append_benchmark_record

    records: list[dict] = []
    original_append = append_benchmark_record

    def spy_append(record, **kwargs):
        records.append(dict(record))
        return original_append(record, **kwargs)

    kinship_file = sample_plink_data.parent / "gemma_kinship.cXX.txt"
    out = tmp_path / "output_tel"
    out.mkdir()

    config = PipelineConfig(
        bfile=sample_plink_data,
        kinship_file=kinship_file,
        lmm_mode=1,
        maf=0.0,
        miss=1.0,
        output_dir=out,
        check_memory=False,
        show_progress=False,
        backend="numpy",
    )
    with patch("jamma.core.telemetry.append_benchmark_record", side_effect=spy_append):
        PipelineRunner(config).run()

    assert len(records) == 1, f"Expected 1 telemetry record, got {len(records)}"
    rec = records[0]
    # Required fields
    assert "timestamp" in rec
    assert "jamma_version" in rec
    assert rec["n_samples"] > 0
    assert rec["n_snps"] > 0
    assert "backend" in rec
    # Optional but expected from pipeline
    assert rec["lmm_mode"] == 1
    assert rec["loco"] is False
    assert "total_s" in rec


# ---------------------------------------------------------------------------
# Covariate NaN filtering
# ---------------------------------------------------------------------------


@pytest.mark.tier1
class TestNSamplesReflectsCovariateFiltering:
    """Regression test: n_samples must exclude samples with NaN covariates."""

    def test_n_samples_reflects_covariate_filtering(self, tmp_path: Path) -> None:
        """PipelineResult.n_samples excludes samples with NaN covariates."""
        bfile = _copy_plink_genotypes(tmp_path)

        # Write a .fam with all valid phenotypes
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        with open(fam_path, "w") as f:
            for i in range(n_samples):
                pheno = 1.0 + i * 0.1
                f.write(f"FAM{i:03d}\tIND{i:03d}\t0\t0\t0\t{pheno}\n")

        # Write a covariate file with intercept + one covariate, 10 rows NaN
        n_nan_covariates = 10
        cov_path = tmp_path / "covariates.txt"
        with open(cov_path, "w") as f:
            for i in range(n_samples):
                intercept = 1.0
                cov = "NA" if i < n_nan_covariates else str(0.5 + i * 0.01)
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
        assert result.n_samples < n_samples
