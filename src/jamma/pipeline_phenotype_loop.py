"""The per-phenotype LMM loop and the genotype source it runs over.

Split out of ``pipeline.py``: ``PipelineRunner.run`` is the only caller, the
loop calls nothing else in the pipeline, and it reads nothing from the runner
but the config. So this is where the question "how does one phenotype reach
the shared LMM body" is answered, without the surrounding orchestration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from jamma.io.plink import PlinkMetadata, load_plink_binary
from jamma.lmm.association_plan import DEFAULT_STATS_CHUNK, ExecutionMode
from jamma.lmm.genotype_source import GenotypeSource
from jamma.lmm.prepare_common import prepare_rotated_covariates
from jamma.lmm.runner_numpy import (
    BATCH_LABELS,
    STREAMING_LABELS,
    LmmRunSpec,
    MatrixSource,
    PreparedPhenotypeSpec,
    prepare_genotypes,
    run_lmm_association_group_prepared,
)
from jamma.lmm.runner_numpy_streaming import BedSource
from jamma.lmm.schema import ChunkRunStats, SnpMeta
from jamma.pipeline_config import PhenotypeResult, PipelineConfig
from jamma.pipeline_plan import StandardAnalysisPlan
from jamma.pipeline_samples import AnalysedSamples

__all__ = ["run_phenotype_loop"]


def run_phenotype_loop(
    config: PipelineConfig,
    analysis: StandardAnalysisPlan,
    samples: AnalysedSamples,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    assoc_path: Path,
    meta: PlinkMetadata,
) -> tuple[list[PhenotypeResult], float]:
    """Run the per-phenotype LMM loop over the analysed samples.

    Builds one genotype source for the plan's mode, then iterates the
    configured phenotype columns, masking each to the shared valid-sample
    intersection and running the shared LMM body over one prepared genotype
    selection. ``meta`` is the pipeline's already-parsed PLINK metadata, so
    the streaming source never re-reads the .bim per phenotype. The ``-snps``
    restriction reaches the body as ``snps_indices`` in both modes, where it
    joins the MAF, missingness and HWE filters.

    Returns:
        One ``PhenotypeResult`` per configured column, in column order, and
        the genotype rotation time shared across them.
    """
    pheno_columns = config.phenotype_columns
    is_multi = len(pheno_columns) > 1
    plan = analysis.execution.summary

    phenotype_results: list[PhenotypeResult] = []

    source = _genotype_source(plan.mode, plan.runner_name, config.bfile, meta, analysis)
    spec = LmmRunSpec(
        config=analysis.lmm,
        execution=analysis.execution,
        snps_indices=analysis.snps_indices,
        hwe_threshold=config.hwe_threshold,
        labels=_LABELS[plan.mode],
    )
    genotypes = prepare_genotypes(source, spec, samples.basis)
    # Prepared chunks retain their analyzed rows. Release the original batch
    # matrix when sample filtering replaced it with a smaller allocation.
    del source
    if genotypes.n_unexpected > 0:
        logger.warning(
            f"Genotype validation: {genotypes.n_unexpected} values outside "
            "expected range {0, 1, 2, NaN}"
        )
    if genotypes.n_filtered == 0:
        logger.warning("All SNPs were filtered out. No association tests will run.")
    covariates = samples.covariates
    filtered_covariates = (
        covariates[samples.valid_mask, :] if covariates is not None else None
    )
    prepared_covariates = prepare_rotated_covariates(
        eigenvectors, filtered_covariates, genotypes.analyzed_sample_count
    )

    prefix = config.output_prefix

    shared_rotation_s = 0.0
    group_size = analysis.execution.phenotype_group_size
    for group_start in range(0, len(pheno_columns), group_size):
        columns = pheno_columns[group_start : group_start + group_size]
        group_specs = []
        group_paths = []
        for col in columns:
            if is_multi:
                logger.info(f"Starting LMM for phenotype column {col}")
            phenotypes_col = samples.phenotypes[col][samples.valid_mask]
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

    return phenotype_results, shared_rotation_s


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
        return BedSource(
            bfile,
            snp_meta=snp_meta,
            n_samples=meta.n_samples,
            n_snps=meta.n_snps,
            stats_chunk_size=DEFAULT_STATS_CHUNK,
            validate_genotypes=True,
            show_progress=analysis.lmm.show_progress,
        )

    logger.info(
        f"{runner_name}: loading all genotypes into memory"
        " (for large datasets, use --backend numpy-streaming)"
    )
    return MatrixSource(load_plink_binary(bfile).genotypes, snp_meta)
