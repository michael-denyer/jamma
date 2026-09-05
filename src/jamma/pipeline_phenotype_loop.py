"""The per-phenotype LMM loop and the genotype source it runs over.

Split out of ``pipeline.py``: ``PipelineRunner.run`` is the only caller, the
loop calls nothing else in the pipeline, and it reads nothing from the runner
but the config. So this is where the question "how does one phenotype reach
the shared LMM body" is answered, without the surrounding orchestration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from loguru import logger

from jamma.io.plink import PlinkMetadata
from jamma.lmm.association_plan import DEFAULT_STATS_CHUNK, ExecutionMode
from jamma.lmm.genotype_source import GenotypeSource, SampleBasis
from jamma.lmm.prepare_common import prepare_rotated_covariates
from jamma.lmm.runner_numpy import BATCH_LABELS, STREAMING_LABELS
from jamma.lmm.schema import ChunkRunStats, SnpMeta
from jamma.lmm.stats import AssocResult
from jamma.pipeline_config import PhenotypeResult, PipelineConfig
from jamma.pipeline_plan import StandardAnalysisPlan

__all__ = ["PhenoLoopOutcome", "run_phenotype_loop"]


class PhenoLoopOutcome(NamedTuple):
    """Aggregated results of the per-phenotype LMM loop.

    Returned by ``run_phenotype_loop`` so ``PipelineRunner.run`` can
    assemble the final ``PipelineResult`` without holding the loop's locals.
    """

    associations: list[AssocResult]
    n_tested: int
    assoc_paths: list[Path]
    phenotype_results: list[PhenotypeResult]
    lmm_s: float
    runner_timing: ChunkRunStats
    pve: float | None
    pve_se: float | None


def run_phenotype_loop(
    config: PipelineConfig,
    analysis: StandardAnalysisPlan,
    all_pheno_data: dict[int, tuple[np.ndarray, int]],
    valid_mask: np.ndarray,
    covariates: np.ndarray | None,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    assoc_path: Path,
    meta: PlinkMetadata,
) -> PhenoLoopOutcome:
    """Run the per-phenotype LMM loop and aggregate its results.

    Builds one genotype source for the plan's mode, then iterates the
    configured phenotype columns, masking each to the shared valid-sample
    intersection and running the shared LMM body over one prepared genotype
    selection. Aggregate runner timing sums every phenotype's work.
    ``meta`` is the pipeline's already-parsed PLINK metadata, so the
    streaming source never re-reads the .bim per phenotype. The ``-snps``
    restriction reaches the body as ``snps_indices`` in both modes, where it
    joins the MAF, missingness and HWE filters.

    Returns:
        A PhenoLoopOutcome bundling associations, total SNPs tested, the
        per-phenotype output paths, the loop wall time, runner timing, and
        the PVE estimate.
    """
    from jamma.lmm.runner_numpy import (
        LmmRunSpec,
        prepare_genotypes,
    )

    pheno_columns = config.phenotype_columns
    is_multi = len(pheno_columns) > 1
    plan = analysis.execution.summary

    t_lmm = time.perf_counter()
    all_results: list[AssocResult] = []
    total_tested = 0
    all_assoc_paths: list[Path] = []
    phenotype_results: list[PhenotypeResult] = []

    source = _genotype_source(plan.mode, plan.runner_name, config.bfile, meta, analysis)
    spec = LmmRunSpec(
        config=analysis.lmm,
        execution=analysis.execution,
        snps_indices=analysis.snps_indices,
        hwe_threshold=config.hwe_threshold,
        labels=_LABELS[plan.mode],
    )
    genotypes = prepare_genotypes(source, spec, SampleBasis.from_mask(valid_mask))
    if genotypes.n_unexpected > 0:
        logger.warning(
            f"Genotype validation: {genotypes.n_unexpected} values outside "
            "expected range {0, 1, 2, NaN}"
        )
    if genotypes.n_filtered == 0:
        logger.warning("All SNPs were filtered out. No association tests will run.")
    filtered_covariates = covariates[valid_mask, :] if covariates is not None else None
    prepared_covariates = prepare_rotated_covariates(
        eigenvectors, filtered_covariates, genotypes.analyzed_sample_count
    )

    prefix = config.output_prefix
    from jamma.lmm.runner_numpy import (
        PreparedPhenotypeSpec,
        run_lmm_association_group_prepared,
    )

    shared_rotation_s = 0.0
    group_size = analysis.execution.phenotype_group_size
    for group_start in range(0, len(pheno_columns), group_size):
        columns = pheno_columns[group_start : group_start + group_size]
        group_specs = []
        group_paths = []
        for col in columns:
            if is_multi:
                logger.info(f"Starting LMM for phenotype column {col}")
            phenotypes_col = all_pheno_data[col][0][valid_mask]
            col_path = (
                config.output_dir / f"{prefix}.pheno{col}.assoc.txt"
                if is_multi
                else assoc_path
            )
            group_specs.append(PreparedPhenotypeSpec(phenotypes_col, col_path))
            group_paths.append(col_path)

        grouped = run_lmm_association_group_prepared(
            genotypes,
            spec,
            tuple(group_specs),
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            prepared_covariates=prepared_covariates,
        )
        shared_rotation_s += grouped.rotation_s
        rotation_shares = [grouped.rotation_s / len(grouped.results)] * len(
            grouped.results
        )
        rotation_shares[-1] = grouped.rotation_s - sum(rotation_shares[:-1])
        for col, col_path, run_result, rotation_share in zip(
            columns, group_paths, grouped.results, rotation_shares, strict=True
        ):
            all_results.extend(run_result.associations)
            total_tested += run_result.n_tested
            all_assoc_paths.append(col_path)
            phenotype_results.append(
                PhenotypeResult(
                    column=col,
                    associations=run_result.associations,
                    n_snps_tested=run_result.n_tested,
                    assoc_path=col_path,
                    timing=ChunkRunStats(
                        processed=run_result.timing.processed,
                        rotation_s=rotation_share,
                        compute_s=run_result.timing.compute_s,
                        result_write_s=run_result.timing.result_write_s,
                    ),
                    pve_estimate=run_result.pve,
                    pve_se=run_result.pve_se,
                )
            )
            logger.info(
                f"Phenotype {col}: {run_result.n_tested} SNPs tested -> {col_path}"
            )

    lmm_s = time.perf_counter() - t_lmm
    runner_timing = _sum_chunk_stats(phenotype_results, shared_rotation_s)
    single = phenotype_results[0] if len(phenotype_results) == 1 else None

    return PhenoLoopOutcome(
        associations=all_results,
        n_tested=total_tested,
        assoc_paths=all_assoc_paths,
        phenotype_results=phenotype_results,
        lmm_s=lmm_s,
        runner_timing=runner_timing,
        pve=single.pve_estimate if single is not None else None,
        pve_se=single.pve_se if single is not None else None,
    )


def _sum_chunk_stats(
    results: list[PhenotypeResult], shared_rotation_s: float
) -> ChunkRunStats:
    """Sum work and stage timings across every phenotype run."""
    return ChunkRunStats(
        processed=sum(result.timing.processed for result in results),
        rotation_s=shared_rotation_s,
        compute_s=sum(result.timing.compute_s for result in results),
        result_write_s=sum(result.timing.result_write_s for result in results),
    )


_LABELS = {"batch": BATCH_LABELS, "streaming": STREAMING_LABELS}


def _genotype_source(
    mode: ExecutionMode,
    runner_name: str,
    bfile: Path,
    meta: PlinkMetadata,
    analysis: StandardAnalysisPlan,
) -> GenotypeSource:
    """Build the one genotype source every phenotype in this run reads from."""
    snp_meta = SnpMeta.from_plink_meta(meta)
    if mode == "streaming":
        from jamma.lmm.runner_numpy_streaming import BedSource

        return BedSource(
            bfile,
            snp_meta=snp_meta,
            n_samples=meta.n_samples,
            n_snps=meta.n_snps,
            stats_chunk_size=DEFAULT_STATS_CHUNK,
            validate_genotypes=True,
            show_progress=analysis.lmm.show_progress,
        )

    from jamma.io import load_plink_binary
    from jamma.lmm.runner_numpy import MatrixSource

    logger.info(
        f"{runner_name}: loading all genotypes into memory"
        " (for large datasets, use --backend numpy-streaming)"
    )
    return MatrixSource(load_plink_binary(bfile).genotypes, snp_meta)
