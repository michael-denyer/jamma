"""Tests for PipelineRunner service class."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.pipeline import PipelineConfig, PipelineRunner
from tests.builders import write_fam
from tests.fixture_paths import LOCO, SYNTHETIC

BFILE = SYNTHETIC.bfile


def _first_phenotype(runner: PipelineRunner) -> tuple[np.ndarray, int]:
    """Read the runner's first configured phenotype column the way run() does."""
    columns = runner.config.phenotype_columns
    data, _mask, _n_valid = runner._load_phenotypes_and_intersect_masks(columns, None)
    return data[columns[0]]


@pytest.mark.tier0
class TestParsePhenotypes:
    """Tests for the .fam phenotype parsing behind run()."""

    def test_parse_phenotypes_from_fixture(self, sample_plink_data: Path) -> None:
        """The phenotype loader reads phenotypes from the .fam file."""
        config = PipelineConfig(
            bfile=sample_plink_data,
            check_memory=False,
        )
        runner = PipelineRunner(config)
        phenotypes, n_analyzed = _first_phenotype(runner)

        assert len(phenotypes) == 100  # gemma_synthetic has 100 samples
        assert n_analyzed > 0
        assert n_analyzed <= 100


def _copy_plink_genotypes(dest: Path) -> Path:
    """Copy .bed and .bim from gemma_synthetic fixture to dest directory.

    Returns:
        bfile prefix (dest / "test")
    """
    for ext in (".bed", ".bim"):
        shutil.copy(SYNTHETIC.dir / f"test{ext}", dest / f"test{ext}")
    return dest / "test"


@pytest.mark.tier0
class TestPhenotypeColumnSelection:
    """Tests for phenotype column selection via PipelineConfig.phenotype_columns."""

    def test_default_phenotype_column(self, sample_plink_data: Path) -> None:
        """The default phenotype_columns=[1] reads the standard .fam phenotype."""
        config = PipelineConfig(
            bfile=sample_plink_data,
            check_memory=False,
        )
        assert config.phenotype_columns == (1,)

        runner = PipelineRunner(config)
        phenotypes, n_analyzed = _first_phenotype(runner)

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
        write_fam(
            fam_path,
            [1.0 + i for i in range(n_samples)],
            [4.0 + i for i in range(n_samples)],
            [7.0 + i for i in range(n_samples)],
        )

        config1 = PipelineConfig(bfile=bfile, check_memory=False, phenotype_columns=[1])
        pheno1, _ = _first_phenotype(PipelineRunner(config1))

        config2 = PipelineConfig(bfile=bfile, check_memory=False, phenotype_columns=[2])
        pheno2, _ = _first_phenotype(PipelineRunner(config2))

        config3 = PipelineConfig(bfile=bfile, check_memory=False, phenotype_columns=[3])
        pheno3, _ = _first_phenotype(PipelineRunner(config3))

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
            _first_phenotype(runner)


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
        weight_path = SYNTHETIC.fam  # Use any existing file
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
        weight_path = SYNTHETIC.fam  # Use any existing file
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


@pytest.mark.tier1
class TestPhenotypeColumnMissingValues:
    """Tests for missing value handling in non-default phenotype columns."""

    def test_phenotype_column_with_missing_values(self, tmp_path: Path) -> None:
        """Missing value handling works correctly for non-default phenotype columns."""
        bfile = _copy_plink_genotypes(tmp_path)

        # Write .fam with 2 phenotype columns; column 7 (pheno 2) has missing values
        n_samples = 100
        fam_path = tmp_path / "test.fam"
        write_fam(
            fam_path,
            [1.0 + i for i in range(n_samples)],
            ["NA", "-9", *[10.0 + i for i in range(2, n_samples)]],
        )

        config = PipelineConfig(bfile=bfile, check_memory=False, phenotype_columns=[2])
        phenotypes, n_analyzed = _first_phenotype(PipelineRunner(config))

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
def test_pipeline_builds_association_plan_once(
    sample_plink_data: Path, output_dir: Path, monkeypatch
) -> None:
    """PipelineRunner.run() builds executable association policy exactly once.

    A prior version called it twice: once before the phenotype/covariate
    masks existed (pricing the pre-mask n_samples), and again after masking
    to catch a post-filter mode flip. Both calls ran estimate_lmm_memory.
    Now there is a single call, made after the masks exist so it never needs
    a second pass.
    """
    import jamma.pipeline as pipeline_module

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

    call_count = 0
    real_plan_association = pipeline_module.plan_association

    def _counting_plan_association(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_plan_association(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "plan_association", _counting_plan_association)

    result = PipelineRunner(config).run()

    assert result.n_snps_tested > 0
    assert call_count == 1, f"expected plan_association once, got {call_count}"


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
def test_pipeline_loco_prices_and_threads_one_plan(tmp_path: Path, monkeypatch):
    """The LOCO branch runs the shared preflight and hands run_lmm_loco its plan."""
    import jamma.lmm
    from jamma.core import memory

    seen: dict = {}
    real_run_lmm_loco = jamma.lmm.run_lmm_loco

    def _spy(*args, **kwargs):
        seen["execution"] = kwargs["execution"]
        return real_run_lmm_loco(*args, **kwargs)

    monkeypatch.setattr(jamma.lmm, "run_lmm_loco", _spy)
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 1000.0)
    quotes: list[str] = []
    handle = logger.add(quotes.append, format="{message}", level="INFO")
    try:
        result = PipelineRunner(
            PipelineConfig(
                bfile=LOCO.bfile,
                loco=True,
                backend="numpy",
                output_dir=tmp_path / "output",
                check_memory=True,
                show_progress=False,
            )
        ).run()
    finally:
        logger.remove(handle)

    assert result.n_snps_tested > 0
    assert seen["execution"].summary.mode == "loco"
    assert any(m.startswith("Memory estimate (numpy-loco)") for m in quotes), quotes


@pytest.mark.tier1
def test_pipeline_loco_numpy(tmp_path: Path) -> None:
    """LOCO + NumPy backend completes end-to-end without error."""
    # gemma_loco fixture: 100 samples, 500 SNPs across 3 chromosomes
    loco_bfile = LOCO.bfile
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
    assert result.n_snps_tested > 0
    assert result.assoc_path.exists()


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


# ===========================================================================
# Multi-Phenotype Tests
# ===========================================================================


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
        write_fam(
            fam_path,
            [1.0 + i * 0.1 for i in range(n_samples)],
            [2.0 + i * 0.1 for i in range(n_samples)],
        )

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
        write_fam(
            fam_path,
            [1.0 + i * 0.1 for i in range(n_samples)],
            [2.0 + i * 0.1 for i in range(n_samples)],
        )

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

        # allow-patch: dispatch spy; one eigendecomposition per run is the contract
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
        write_fam(
            fam_path,
            ["NA" if i < 2 else 1.0 + i * 0.1 for i in range(n_samples)],
            ["NA" if i < 2 else 2.0 + i * 0.1 for i in range(n_samples)],
        )

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
        write_fam(
            fam_path,
            ["NA" if i % 2 == 0 else 1.0 + i * 0.1 for i in range(n_samples)],
            [2.0 + i * 0.1 if i % 2 == 0 else "NA" for i in range(n_samples)],
        )

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
        write_fam(
            fam_path,
            [1.0 + i * 0.1 for i in range(n_samples)],
            [2.0 + i * 0.1 for i in range(n_samples)],
        )

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
        write_fam(fam_path, [1.0 + i * 0.1 for i in range(n_samples)])

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
    total_snps = meta.n_snps

    # Restrict to first 30 SNPs
    n_restrict = 30
    snps_path = tmp_path / "snps.txt"
    snps_path.write_text("\n".join(meta.sid[:n_restrict]) + "\n")

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


@pytest.mark.tier1
def test_pipeline_planning_passes_n_cvt(
    tmp_path: Path, sample_plink_data: Path
) -> None:
    """BCKAUTO-04: Re-evaluation passes n_cvt from loaded covariates."""
    from unittest.mock import patch

    from jamma.lmm.association_plan import plan_association

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

    # Spy on executable planning to capture its dimensions.
    calls: list[dict] = []
    original_plan = plan_association

    def spy_plan(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_plan(*args, **kwargs)

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
    with patch("jamma.pipeline.plan_association", side_effect=spy_plan):
        PipelineRunner(config).run()

    # The re-evaluation call (post-covariate-load) must have n_cvt=2
    assert any(c.get("n_cvt", 1) == 2 for c in calls), (
        f"No call to plan_association had n_cvt=2; calls={calls}"
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
        write_fam(fam_path, [1.0 + i * 0.1 for i in range(n_samples)])

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
