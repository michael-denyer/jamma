"""Observable contracts for shared multi-phenotype preparation."""

from __future__ import annotations

import collections
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.schema import LmmMode
from jamma.pipeline import BackendRequest, PipelineConfig, PipelineRunner
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
def test_multi_phenotype_reuses_preparation_and_preserves_each_result(
    tmp_path: Path,
    backend: BackendRequest,
    lmm_mode: LmmMode,
    force_fallback: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One shared preparation produces the same files as isolated runs."""
    bfile, phenotypes, covariate_path = _study(tmp_path)
    if force_fallback:
        from jamma.lmm import accel

        monkeypatch.setattr(accel, "_accel", None)
    calls: collections.Counter[str] = collections.Counter()

    def profile(frame, event, _arg):  # type: ignore[no-untyped-def]
        if event != "call":
            return
        filename = frame.f_code.co_filename
        name = frame.f_code.co_name
        if (
            Path(filename).name in {"runner_numpy.py", "runner_numpy_streaming.py"}
            and name == "prepare"
        ):
            calls["source preparation"] += 1
        caller = frame.f_back.f_code.co_filename if frame.f_back is not None else ""
        if (
            filename.endswith("snp_stats.py")
            and name == "collect_snp_stats_from_chunks"
            and Path(caller).name == "runner_numpy.py"
        ):
            calls["collect_snp_stats_from_chunks"] += 1
        if (
            filename.endswith("snp_stats.py")
            and name == "collect_streamed_snp_stats"
            and Path(caller).name == "runner_numpy_streaming.py"
        ):
            calls["collect_streamed_snp_stats"] += 1
        if filename.endswith("chunk_runner_numpy.py") and name == "prepare":
            calls["chunk rotations"] += 1
        if filename.endswith("prepare_common.py") and name == "_build_covariate_matrix":
            calls["covariate preparation"] += 1
        if (
            filename.endswith("prepare_common.py")
            and name == "prepare_rotated_covariates"
        ):
            calls["covariate rotation"] += 1

    sys.setprofile(profile)
    try:
        combined = PipelineRunner(
            _config(
                bfile,
                tmp_path / "combined",
                [1, 2],
                covariate_path,
                backend,
                lmm_mode,
            )
        ).run()
    finally:
        sys.setprofile(None)

    individual = []
    for index, column in enumerate((1, 2)):
        single_bfile = _copy_genotypes(tmp_path / f"single-input-{column}")
        shared_masked = list(phenotypes[index])
        for missing_index in (0, 1):
            shared_masked[missing_index] = "NA"
        write_fam(Path(f"{single_bfile}.fam"), shared_masked)
        individual.append(
            PipelineRunner(
                _config(
                    single_bfile,
                    tmp_path / f"single-{column}",
                    [1],
                    covariate_path,
                    backend,
                    lmm_mode,
                )
            ).run()
        )

    assert calls["source preparation"] == 1
    expected_stats_key = (
        "collect_snp_stats_from_chunks"
        if backend == "numpy"
        else "collect_streamed_snp_stats"
    )
    assert calls[expected_stats_key] == 1
    assert calls["covariate preparation"] == 1
    assert calls["covariate rotation"] == 1
    from jamma.lmm.association_plan import plan_association

    execution = plan_association(
        97,
        500,
        requested=backend,
        n_input_samples=100,
        n_cvt=2,
        lmm_mode=lmm_mode,
        n_phenotypes=2,
    )
    assert execution.phenotype_group_size == 2
    assert calls["chunk rotations"] == execution.conservative_chunks.n_chunks
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
