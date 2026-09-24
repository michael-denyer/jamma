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

from jamma.genotype.dataset import GenotypeDataset
from jamma.lmm.association_plan import ExecutionMode
from jamma.lmm.prepare_common import _build_covariate_matrix, rotate_basis
from jamma.lmm.runner_numpy import (
    BATCH_LABELS,
    STREAMING_LABELS,
    LmmRunSpec,
    PhenotypeRun,
    prepare_genotypes,
    run_association,
)
from jamma.lmm.schema import ChunkRunStats
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
    dataset: GenotypeDataset,
) -> tuple[list[PhenotypeResult], float]:
    """Run the per-phenotype LMM loop over the analysed samples.

    Builds one genotype source for the plan's mode, then iterates the
    configured phenotype columns, masking each to the shared valid-sample
    intersection and running the shared LMM body over one prepared genotype
    selection. ``dataset`` is the pipeline's already-opened genotypes, so
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

    spec = LmmRunSpec(
        config=analysis.lmm,
        execution=analysis.execution,
        snps_indices=analysis.snps_indices,
        hwe_threshold=config.hwe_threshold,
        labels=_LABELS[plan.mode],
    )
    genotypes = prepare_genotypes(
        _genotype_dataset(plan.mode, plan.runner_name, dataset),
        samples.basis,
        spec.snp_filters,
        progress=spec.stats_progress,
    )
    if genotypes.n_filtered == 0:
        logger.warning("All SNPs were filtered out. No association tests will run.")
    covariates = samples.covariates
    filtered_covariates = (
        covariates[samples.valid_mask, :] if covariates is not None else None
    )
    W, _n_cvt = _build_covariate_matrix(
        filtered_covariates, genotypes.analyzed_sample_count
    )
    basis = rotate_basis(eigenvalues, eigenvectors, W)

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
            group_specs.append(PhenotypeRun(phenotypes_col, col_path))
            group_paths.append(col_path)

        grouped = run_association(genotypes, spec, basis, group_specs)
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


def _genotype_dataset(
    mode: ExecutionMode, runner_name: str, dataset: GenotypeDataset
) -> GenotypeDataset:
    """Return the one dataset every phenotype in this run reads from."""
    if mode == "streaming":
        return dataset
    logger.info(
        f"{runner_name}: loading all genotypes into memory"
        " (for large datasets, use --backend numpy-streaming)"
    )
    return dataset.materialize()
