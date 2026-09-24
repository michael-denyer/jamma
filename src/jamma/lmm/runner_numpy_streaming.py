"""Disk-streaming NumPy LMM association runner.

Streams a ``GenotypeDataset`` twice (statistics pass, float64 association
pass) without ever allocating the full genotype matrix; the run itself is the
shared body in ``runner_numpy``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jamma.genotype.dataset import GenotypeDataset
from jamma.genotype.snp_filter import validate_snp_indices
from jamma.lmm.association_plan import DEFAULT_STATS_CHUNK, plan_association
from jamma.lmm.prepare_common import (
    AnalysedPhenotype,
    parse_eigen_input,
    restrict_eigen_input,
)
from jamma.lmm.runner_numpy import (
    STREAMING_LABELS,
    LmmRunSpec,
    run_single,
)
from jamma.lmm.schema import (
    DEFAULT_LMM_CONFIG,
    LmmConfig,
    LmmRunResult,
)


def run_lmm_association_numpy_streaming(
    dataset: GenotypeDataset,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None = None,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    chunk_size: int | None = None,
    output_path: Path | None = None,
    snps_indices: np.ndarray | None = None,
    hwe_threshold: float = 0.0,
    config: LmmConfig = DEFAULT_LMM_CONFIG,
) -> LmmRunResult:
    """Run LMM association tests by streaming genotypes from disk.

    Two-pass disk streaming: pass 1 computes SNP statistics for filtering,
    pass 2 runs the shared chunk engine per chunk. Never allocates the full
    genotype matrix.

    Args:
        dataset: The genotypes, streamed from their file; for a PLINK
            prefix pass ``GenotypeDataset.open_plink(prefix)``.
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples), or None when
            pre-computed eigenvalues and eigenvectors are provided. Consumed:
            centred in place, then overwritten by the eigendecomposition
            (zeroed on the NumPy fallback). Must be writeable; pass
            kinship.copy() to keep the original matrix.
        covariates: Covariate matrix (n_samples, n_cvt) or None for
            intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        chunk_size: Cap on SNPs per chunk, for both the statistics pass and
            the association pass. None (default) reads statistics in
            DEFAULT_STATS_CHUNK blocks and lets the chunk engine size the
            association chunks against the RAM budget.
        output_path: Path for incremental result writing, or None for
            in-memory.
        snps_indices: Pre-resolved column indices for -snps restriction,
            or None.
        hwe_threshold: HWE p-value threshold; SNPs with p < threshold are
            removed. 0.0 disables HWE filtering (default).
        config: LmmConfig with thresholds, lambda bounds, test type,
            memory check and progress settings.

    Returns:
        LmmRunResult with associations (empty if output_path is set --
        results on disk), PVE from the null model, n_tested counting the
        SNPs that passed filtering and were tested, and the run's timing
        breakdown.
    """
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1 or None, got {chunk_size}")

    validate_snp_indices(snps_indices, dataset.n_variants)
    samples = AnalysedPhenotype.from_inputs(phenotypes, covariates)
    execution = plan_association(
        samples.n_samples,
        dataset.n_variants,
        config=config,
        backend="numpy-streaming",
        n_cvt=samples.n_cvt,
        n_input_samples=dataset.n_samples,
        max_chunk_size=chunk_size,
    )
    return run_single(
        dataset,
        LmmRunSpec(
            config=config,
            execution=execution,
            snps_indices=snps_indices,
            hwe_threshold=hwe_threshold,
            labels=STREAMING_LABELS,
            stats_block_size=DEFAULT_STATS_CHUNK if chunk_size is None else chunk_size,
        ),
        samples,
        restrict_eigen_input(
            parse_eigen_input(kinship, eigenvalues, eigenvectors), samples.valid_mask
        ),
        output_path if output_path is not None else [],
    )
