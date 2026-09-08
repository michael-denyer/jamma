"""Observable contracts for shared multi-phenotype preparation."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from jamma.lmm.schema import LmmMode
from jamma.pipeline import (
    BackendRequest,
    PipelineConfig,
    PipelineResult,
    PipelineRunner,
)
from tests.builders import write_fam
from tests.fixture_paths import SYNTHETIC

pytestmark = pytest.mark.tier1


def _copy_genotypes(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    bfile = destination / "test"
    for extension in (".bed", ".bim"):
        shutil.copy(SYNTHETIC.dir / f"test{extension}", Path(f"{bfile}{extension}"))
    return bfile


def _study(tmp_path: Path) -> tuple[Path, list[list[float | str]], Path]:
    bfile = _copy_genotypes(tmp_path / "shared")
    rng = np.random.default_rng(20260904)
    phenotypes: list[list[float | str]] = [
        rng.standard_normal(100).tolist(),
        rng.standard_normal(100).tolist(),
    ]
    phenotypes[0][0] = "NA"
    phenotypes[1][1] = "NA"
    write_fam(
        Path(f"{bfile}.fam"),
        *phenotypes,
    )
    covariates = np.column_stack((np.ones(100), rng.standard_normal(100)))
    covariates[2, 1] = np.nan
    covariate_path = tmp_path / "covariates.txt"
    np.savetxt(covariate_path, covariates)
    return bfile, phenotypes, covariate_path


def _config(
    bfile: Path,
    output_dir: Path,
    columns: list[int],
    covariate_path: Path,
    backend: BackendRequest,
    lmm_mode: LmmMode,
) -> PipelineConfig:
    return PipelineConfig(
        bfile=bfile,
        phenotype_columns=columns,
        output_dir=output_dir,
        covariate_file=covariate_path,
        check_memory=False,
        show_progress=False,
        no_telemetry=True,
        backend=backend,
        lmm_mode=lmm_mode,
    )


def _run_logged(config: PipelineConfig) -> tuple[PipelineResult, list[str]]:
    """Run the pipeline and return its result with the INFO lines it logged."""
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO", format="{message}")
    try:
        return PipelineRunner(config).run(), messages
    finally:
        logger.remove(sink_id)


def _genotype_passes(messages: list[str], *, filtered_only: bool = False) -> int:
    """Count the ``Reading N SNPs in K chunks`` lines, one per pass over the file.

    ``filtered_only`` keeps just the association pass over the SNPs that
    survived filtering.
    """
    marker = " filtered SNPs in " if filtered_only else " SNPs in "
    return sum(
        1
        for message in messages
        if message.startswith("Reading ") and marker in message
    )


@pytest.mark.parametrize(
    ("backend", "lmm_mode", "force_fallback"),
    [
        ("numpy", 1, False),
        ("numpy-streaming", 1, False),
        ("numpy", 4, False),
        ("numpy-streaming", 4, False),
        ("numpy", 4, True),
    ],
)
def test_multi_phenotype_reads_genotypes_once_and_preserves_each_result(
    tmp_path: Path,
    backend: BackendRequest,
    lmm_mode: LmmMode,
    force_fallback: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A two-phenotype run reads the genotype file as often as a one-phenotype run.

    The PLINK reader logs one ``Reading N SNPs in K chunks`` line per pass
    over the file, so the log is where a user sees whether the streaming
    backend collected SNP statistics and streamed the association chunks
    once for both phenotypes or once per phenotype. The in-memory backend's
    statistics pass and the covariate rotation produce no output, so this
    test does not observe them; it checks their results against isolated
    single-phenotype runs instead.
    """
    bfile, phenotypes, covariate_path = _study(tmp_path)
    if force_fallback:
        from jamma.lmm import accel

        monkeypatch.setattr(accel, "_accel", None)

    combined, combined_log = _run_logged(
        _config(
            bfile,
            tmp_path / "combined",
            [1, 2],
            covariate_path,
            backend,
            lmm_mode,
        )
    )
    expected_association_passes = 1 if backend == "numpy-streaming" else 0
    assert (
        _genotype_passes(combined_log, filtered_only=True)
        == expected_association_passes
    )

    individual = []
    for index, column in enumerate((1, 2)):
        single_bfile = _copy_genotypes(tmp_path / f"single-input-{column}")
        shared_masked = list(phenotypes[index])
        for missing_index in (0, 1):
            shared_masked[missing_index] = "NA"
        write_fam(Path(f"{single_bfile}.fam"), shared_masked)
        result, single_log = _run_logged(
            _config(
                single_bfile,
                tmp_path / f"single-{column}",
                [1],
                covariate_path,
                backend,
                lmm_mode,
            )
        )
        assert _genotype_passes(single_log) == _genotype_passes(combined_log)
        individual.append(result)

    assert [result.column for result in combined.phenotype_results] == [1, 2]
    assert combined.n_snps_tested == sum(
        result.n_snps_tested for result in combined.phenotype_results
    )
    assert combined.pve_estimate is None
    assert combined.pve_se is None

    for combined_result, individual_result in zip(
        combined.phenotype_results, individual, strict=True
    ):
        assert combined_result.n_snps_tested == individual_result.n_snps_tested
        assert (
            combined_result.assoc_path.read_text()
            == individual_result.assoc_path.read_text()
        )
        assert combined_result.pve_estimate == individual_result.pve_estimate
        assert combined_result.pve_se == individual_result.pve_se

    timing = combined.phenotype_results
    assert combined.timing.rotation_s == sum(item.timing.rotation_s for item in timing)


def test_prepared_genotypes_reject_same_size_different_sample_basis() -> None:
    """Prepared data cannot be reused for different source-row positions."""
    from jamma.lmm.association_plan import plan_association
    from jamma.lmm.genotype_source import SampleBasis
    from jamma.lmm.prepare_common import EigenPairs
    from jamma.lmm.runner_numpy import (
        LmmRunSpec,
        MatrixSource,
        prepare_genotypes,
        run_lmm_association_prepared,
    )
    from jamma.lmm.schema import LmmConfig, SnpMeta

    genotypes = np.arange(18, dtype=np.float64).reshape(6, 3) % 3
    meta = SnpMeta(
        chr=np.full(3, "1"),
        rs=np.array(["rs1", "rs2", "rs3"]),
        pos=np.arange(3),
        a1=np.full(3, "A"),
        a0=np.full(3, "G"),
    )
    config = LmmConfig(check_memory=False, show_progress=False, maf_threshold=0.0)
    spec = LmmRunSpec(
        config=config,
        execution=plan_association(4, 3, requested="numpy"),
    )
    prepared = prepare_genotypes(
        MatrixSource(genotypes, meta), spec, SampleBasis(np.array([0, 1, 2, 3]), 6)
    )
    phenotype = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, np.nan])

    with pytest.raises(ValueError, match="sample basis does not match"):
        run_lmm_association_prepared(
            prepared,
            spec,
            phenotypes=phenotype,
            eigen_input=EigenPairs(np.ones(4), np.eye(4)),
            covariates=None,
        )
