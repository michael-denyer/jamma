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
from jamma.lmm.genotype_source import GenotypeSource
from jamma.lmm.schema import RunnerTiming, SnpMeta
from jamma.lmm.stats import AssocResult
from jamma.pipeline_config import PipelineConfig
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
    lmm_s: float
    runner_timing: RunnerTiming
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
    intersection and running the shared LMM body over that source. PVE and
    the runner timing breakdown are taken from the final phenotype's result.
    ``meta`` is the pipeline's already-parsed PLINK metadata, so the
    streaming source never re-reads the .bim per phenotype. The ``-snps``
    restriction reaches the body as ``snps_indices`` in both modes, where it
    joins the MAF, missingness and HWE filters.

    Returns:
        A PhenoLoopOutcome bundling associations, total SNPs tested, the
        per-phenotype output paths, the loop wall time, runner timing, and
        the PVE estimate.
    """
    from jamma.lmm.runner_numpy import _run_numpy_lmm

    pheno_columns = config.phenotype_columns
    is_multi = len(pheno_columns) > 1
    plan = analysis.execution.summary

    t_lmm = time.perf_counter()
    all_results: list[AssocResult] = []
    total_tested = 0
    all_assoc_paths: list[Path] = []

    source = _genotype_source(plan.mode, plan.runner_name, config.bfile, meta, analysis)

    # The loop's last run carries the PVE estimate; both stay None if
    # pheno_columns is empty, which PipelineConfig already rejects.
    prefix = config.output_prefix
    pve: float | None = None
    pve_se: float | None = None
    runner_timing: RunnerTiming = {}
    for col in pheno_columns:
        if is_multi:
            logger.info(f"Starting LMM for phenotype column {col}")
        # Mark samples outside the shared intersection as NaN so the
        # runner computes the same valid_mask used for eigendecomposition.
        # We pass full-length arrays (not pre-filtered) because the
        # streaming runner indexes genotypes streamed from disk using
        # the mask it computes internally.
        phenotypes_col = all_pheno_data[col][0].copy()
        phenotypes_col[~valid_mask] = np.nan

        if is_multi:
            col_path = config.output_dir / f"{prefix}.pheno{col}.assoc.txt"
        else:
            col_path = assoc_path

        run_result = _run_numpy_lmm(
            source,
            phenotypes=phenotypes_col,
            # The body takes the eigenpairs; the pipeline consumes the
            # kinship matrix during eigendecomposition and has none left.
            kinship=None,
            covariates=covariates,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            config=analysis.lmm,
            output_path=col_path,
            snps_indices=analysis.snps_indices,
            hwe_threshold=config.hwe_threshold,
            execution=analysis.execution,
            banner=_BANNER[plan.mode],
            label=_LABEL[plan.mode],
            progress_label=_PROGRESS_LABEL[plan.mode],
        )

        all_results.extend(run_result.associations)
        total_tested += run_result.n_tested
        all_assoc_paths.append(col_path)
        pve, pve_se = run_result.pve, run_result.pve_se
        runner_timing = run_result.timing
        logger.info(f"Phenotype {col}: {run_result.n_tested} SNPs tested -> {col_path}")

    lmm_s = time.perf_counter() - t_lmm

    return PhenoLoopOutcome(
        associations=all_results,
        n_tested=total_tested,
        assoc_paths=all_assoc_paths,
        lmm_s=lmm_s,
        runner_timing=runner_timing,
        pve=pve,
        pve_se=pve_se,
    )


_BANNER = {"batch": "NumPy batch", "streaming": "NumPy streaming"}
_LABEL = {"batch": "lmm_numpy", "streaming": "lmm_numpy_streaming"}
_PROGRESS_LABEL = {
    "batch": "LMM association",
    "streaming": "LMM association (streaming)",
}


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
