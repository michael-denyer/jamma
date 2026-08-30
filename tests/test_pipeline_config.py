"""Tests for PipelineConfig construction and validation.

Split from test_pipeline.py: these tests build a PipelineConfig or call
validate_inputs() against paths that need not exist, and never touch real
PLINK/kinship fixture data. Component-level ``run()`` tests stay in
test_pipeline.py; load_kinship behaviour moved to test_pipeline_kinship.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma.lmm.association_plan import plan_association
from jamma.lmm.schema import MIN_N_GRID
from jamma.pipeline import PipelineConfig, PipelineRunner
from jamma.pipeline_memory import memory_preflight
from tests.fixture_paths import SYNTHETIC

BFILE = SYNTHETIC.bfile


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


@pytest.mark.tier0
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

    def test_prefix_with_separator_rejected(self) -> None:
        """A prefix containing a path separator is rejected at construction.

        The output-config dataclass this rule used to live on is gone; the
        message must stay identical since the CLI matches on it.
        """
        with pytest.raises(ValueError, match="must not contain path separators"):
            PipelineConfig(bfile=BFILE, output_prefix="a/b")


@pytest.mark.tier0
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


@pytest.mark.tier0
class TestCheckMemory:
    """Tests for pipeline_memory.memory_preflight."""

    def test_returns_none_when_disabled(self) -> None:
        """memory_preflight returns None when check_memory=False."""
        config = PipelineConfig(
            bfile=BFILE,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        result = memory_preflight(
            runner.config,
            plan_association(
                100,
                500,
                requested="numpy-streaming",
                _require_streaming_accel=False,
            ),
        )
        assert result is None

    def test_returns_plan_when_enabled(self) -> None:
        """memory_preflight returns a MemoryPlan for the streaming mode."""
        from jamma.pipeline_memory import MemoryPlan

        config = PipelineConfig(
            bfile=BFILE,
            check_memory=True,
        )
        runner = PipelineRunner(config)
        result = memory_preflight(
            runner.config,
            plan_association(
                100,
                500,
                requested="numpy-streaming",
                _require_streaming_accel=False,
            ),
        )

        assert isinstance(result, MemoryPlan)
        assert result.total_peak_gb >= 0
        assert result.available_gb >= 0


@pytest.mark.tier0
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


@pytest.mark.tier0
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

    @pytest.mark.parametrize("hwe", [-0.1, 1.5])
    def test_hwe_outside_unit_interval_fails_at_construction(self, hwe: float):
        """hwe_threshold is a p-value; anything outside [0, 1] is a config error."""
        with pytest.raises(ValueError, match="hwe_threshold must be in"):
            PipelineConfig(bfile=BFILE, hwe_threshold=hwe, check_memory=False)

    def test_hwe_with_loco_fails_at_construction(self) -> None:
        """-hwe combined with -loco is rejected before any file is read."""
        with pytest.raises(ValueError, match="not yet supported with -loco"):
            PipelineConfig(
                bfile=BFILE, hwe_threshold=0.001, loco=True, check_memory=False
            )

    def test_cat_requires_covariate_file_at_construction(self) -> None:
        with pytest.raises(ValueError, match="-cat requires -c"):
            PipelineConfig(bfile=BFILE, cat_columns=[1], check_memory=False)

    def test_cat_column_below_one_fails_at_construction(self, tmp_path: Path):
        with pytest.raises(ValueError, match=r"-cat column indices must be >= 1"):
            PipelineConfig(
                bfile=BFILE,
                covariate_file=tmp_path / "cov.txt",
                cat_columns=[0],
                check_memory=False,
            )


@pytest.mark.tier0
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


@pytest.mark.tier0
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


@pytest.mark.tier0
def test_pipeline_config_backend_validation() -> None:
    """PipelineConfig raises ValueError for invalid backend value."""
    with pytest.raises(ValueError, match="backend must be"):
        PipelineConfig(bfile=Path("dummy"), backend="invalid")  # type: ignore[bad-argument-type]


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


# ---------------------------------------------------------------------------
# Flag interaction tests (unit-level)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
class TestFlagInteractions:
    """Verify flag validation rules for eigen reuse."""

    def test_validate_d_without_u_raises(self, tmp_path: Path) -> None:
        """Eigenvalue file without eigenvector file raises ValueError."""
        # Create a dummy eigenvalue file
        d_path = tmp_path / "test.eigenD.txt"
        d_path.write_text("1.0\n2.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=d_path,
            eigenvector_file=None,
            check_memory=False,
        )
        with pytest.raises(ValueError, match=r"Both -d.*and -u.*must be provided"):
            PipelineRunner(config).validate_inputs()

    def test_validate_u_without_d_raises(self, tmp_path: Path) -> None:
        """Eigenvector file without eigenvalue file raises ValueError."""
        u_path = tmp_path / "test.eigenU.txt"
        u_path.write_text("1.0\t0.0\n0.0\t1.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=None,
            eigenvector_file=u_path,
            check_memory=False,
        )
        with pytest.raises(ValueError, match=r"Both -d.*and -u.*must be provided"):
            PipelineRunner(config).validate_inputs()

    def test_validate_eigen_with_loco_raises(self, tmp_path: Path) -> None:
        """Eigen files with -loco raises ValueError (use --eigen-dir instead)."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"
        d_path.write_text("1.0\n")
        u_path.write_text("1.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=d_path,
            eigenvector_file=u_path,
            loco=True,
            check_memory=False,
        )
        with pytest.raises(ValueError, match="not supported with -loco"):
            PipelineRunner(config).validate_inputs()

    def test_validate_eigen_files_not_found_raises(self, tmp_path: Path) -> None:
        """Nonexistent eigenvalue file raises FileNotFoundError."""
        d_path = tmp_path / "nonexistent.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"
        u_path.write_text("1.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=d_path,
            eigenvector_file=u_path,
            check_memory=False,
        )
        with pytest.raises(FileNotFoundError, match="Eigenvalue file not found"):
            PipelineRunner(config).validate_inputs()

    def test_kinship_not_required_with_eigen_files(self, tmp_path: Path) -> None:
        """Kinship is optional when eigen files are provided."""
        d_path = tmp_path / "test.eigenD.txt"
        u_path = tmp_path / "test.eigenU.txt"
        d_path.write_text("1.0\n2.0\n")
        u_path.write_text("1.0\t0.0\n0.0\t1.0\n")

        config = PipelineConfig(
            bfile=BFILE,
            eigenvalue_file=d_path,
            eigenvector_file=u_path,
            kinship_file=None,
            check_memory=False,
        )
        # Should NOT raise -- kinship is optional with eigen files
        PipelineRunner(config).validate_inputs()
