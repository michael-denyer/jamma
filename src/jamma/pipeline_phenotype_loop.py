"""The per-phenotype LMM loop and the two runner call sites it dispatches to.

Split out of ``pipeline.py``: ``PipelineRunner._run_inner`` is the only caller,
these three functions call nothing else in the pipeline, and none of them read
anything from the runner but the config. So this is where the question "how does
one phenotype reach a runner" is answered, without the surrounding
orchestration.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from loguru import logger

from jamma.io.plink import PlinkData
from jamma.lmm.runner import ExecutionPlan
from jamma.lmm.schema import LmmRunResult, RunnerTiming
from jamma.lmm.stats import AssocResult
from jamma.pipeline_config import PipelineConfig

__all__ = ["PhenoLoopOutcome", "run_phenotype_loop"]


class PhenoLoopOutcome(NamedTuple):
    """Aggregated results of the per-phenotype LMM loop.

    Returned by ``run_phenotype_loop`` so ``PipelineRunner._run_inner`` can
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
    plan: ExecutionPlan,
    all_pheno_data: dict[int, tuple[np.ndarray, int]],
    valid_mask: np.ndarray,
    covariates: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    assoc_path: Path,
    snps_indices: np.ndarray | None,
) -> PhenoLoopOutcome:
    """Run the per-phenotype LMM loop and aggregate its results.

    Iterates the configured phenotype columns, masking each to the shared
    valid-sample intersection, dispatching to the batch or streaming runner
    per the plan, and collecting associations, counts, and output paths.
    Captures PVE and runner rotation timing from the final phenotype.

    Returns:
        A PhenoLoopOutcome bundling associations, total SNPs tested, the
        per-phenotype output paths, the loop wall time, runner timing, and
        the PVE estimate.
    """
    pheno_columns = config.phenotype_columns
    is_multi = len(pheno_columns) > 1

    t_lmm = time.perf_counter()
    all_results: list[AssocResult] = []
    total_tested = 0
    all_assoc_paths: list[Path] = []

    # Pre-load PLINK data once for batch multi-phenotype runs
    _plink_data = None
    if plan.mode == "batch" and is_multi:
        from jamma.io import load_plink_binary

        logger.info(
            f"{plan.runner_name}: loading all genotypes into memory"
            " (for large datasets, use --backend numpy-streaming)"
        )
        _plink_data = load_plink_binary(config.bfile)

    # The loop's last run carries the PVE estimate; both stay None if
    # pheno_columns is empty, which PipelineConfig already rejects.
    prefix = config.output_prefix
    pve: float | None = None
    pve_se: float | None = None
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

        if plan.mode == "streaming":
            run_result, n_tested = _run_streaming(
                config,
                phenotypes_col,
                covariates,
                eigenvalues,
                eigenvectors,
                col_path,
                snps_indices,
            )
        else:
            run_result, n_tested = _run_batch(
                config,
                phenotypes_col,
                covariates,
                eigenvalues,
                eigenvectors,
                col_path,
                snps_indices,
                plink_data=_plink_data,
            )

        all_results.extend(run_result.associations)
        total_tested += n_tested
        all_assoc_paths.append(col_path)
        pve, pve_se = run_result.pve, run_result.pve_se
        logger.info(f"Phenotype {col}: {n_tested} SNPs tested -> {col_path}")

    lmm_s = time.perf_counter() - t_lmm

    # Pull runner-level rotation timing from the most recent runner call.
    runner_timing: RunnerTiming = {}
    if plan.mode == "streaming":
        from jamma.lmm.runner_numpy_streaming import (
            get_last_run_timing as _np_stream_timing,
        )

        runner_timing = _np_stream_timing()

    return PhenoLoopOutcome(
        associations=all_results,
        n_tested=total_tested,
        assoc_paths=all_assoc_paths,
        lmm_s=lmm_s,
        runner_timing=runner_timing,
        pve=pve,
        pve_se=pve_se,
    )


def _run_batch(
    config: PipelineConfig,
    phenotypes: np.ndarray,
    covariates: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    assoc_path: Path,
    snps_indices: np.ndarray | None,
    plink_data: PlinkData | None = None,
) -> tuple[LmmRunResult, int]:
    """Run LMM association using the pure-NumPy batch backend.

    Args:
        plink_data: Pre-loaded PLINK data. If None, loads from disk.
            Pass this to avoid reloading genotypes in multi-phenotype runs.
    """
    from jamma.io import load_plink_binary
    from jamma.lmm import run_lmm_association_numpy

    if plink_data is None:
        logger.info(
            "NumPy backend: loading all genotypes into memory "
            "(for large datasets, use --backend numpy-streaming)"
        )
        plink_data = load_plink_binary(config.bfile)

    genotypes = plink_data.genotypes

    # Apply snps_indices filter before passing to runner
    indices = snps_indices if snps_indices is not None else range(plink_data.n_snps)
    if snps_indices is not None:
        genotypes = genotypes[:, snps_indices]
    snp_info = [
        {
            "chr": str(plink_data.chromosome[i]),
            "rs": plink_data.sid[i],
            "pos": int(plink_data.bp_position[i]),
            "a1": plink_data.allele_1[i],
            "a0": plink_data.allele_2[i],
        }
        for i in indices
    ]

    run_result = run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        # The runner takes the eigenpairs; the pipeline consumes the kinship
        # matrix during eigendecomposition and has none left to pass.
        kinship=None,
        snp_info=snp_info,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        config=config.lmm_config(),
        output_path=assoc_path,
    )

    return run_result, run_result.snp_count


def _run_streaming(
    config: PipelineConfig,
    phenotypes: np.ndarray,
    covariates: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    assoc_path: Path,
    snps_indices: np.ndarray | None,
) -> tuple[LmmRunResult, int]:
    """Run LMM via NumPy streaming backend (disk I/O + C extension)."""
    from jamma.lmm.runner_numpy_streaming import (
        run_lmm_association_numpy_streaming,
    )

    return run_lmm_association_numpy_streaming(
        bed_path=config.bfile,
        phenotypes=phenotypes,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        output_path=assoc_path,
        snps_indices=snps_indices,
        hwe_threshold=config.hwe_threshold,
        config=config.lmm_config(),
    )
