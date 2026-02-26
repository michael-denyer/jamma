"""Pure-NumPy batch LMM association runner.

No JAX dependency. Input genotypes must fit in memory.
"""

from __future__ import annotations

import gc
import time

import numpy as np
from loguru import logger

from jamma.core.memory import estimate_lmm_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.core.threading import blas_threads, get_physical_core_count
from jamma.lmm.compute_numpy import _compute_lmm_chunk_numpy
from jamma.lmm.likelihood_numpy import batch_compute_uab_numpy
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    _eigendecompose_or_reuse,
)
from jamma.lmm.results import (
    _build_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory

# Default chunk cap (same as JAX runner)
_MAX_CHUNK = 50_000


def _compute_chunk_size_numpy(n_samples: int, n_filtered: int, n_cvt: int = 1) -> int:
    """Compute chunk size based on RAM budget (no int32 constraint for NumPy).

    Memory per SNP: n_samples * n_index * 8 bytes (Uab dominates).
    Target: stay under ~2 GB per chunk for Uab allocation.

    Args:
        n_samples: Number of samples.
        n_filtered: Number of filtered SNPs.
        n_cvt: Number of covariates.

    Returns:
        Chunk size (number of SNPs per chunk).
    """
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    bytes_per_snp = n_samples * n_index * 8
    if bytes_per_snp == 0:
        return n_filtered
    mem_budget = 2e9  # 2 GB
    chunk_from_memory = int(mem_budget / bytes_per_snp)
    chunk = max(100, min(chunk_from_memory, n_filtered, _MAX_CHUNK))
    return chunk


def run_lmm_association_numpy(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    use_gpu: bool = False,
    check_memory: bool = True,
    show_progress: bool = True,
    lmm_mode: int = 1,
) -> list[AssocResult]:
    """Run LMM association tests using pure-NumPy batch processing.

    Processes SNPs in memory-bounded chunks using BLAS-backed NumPy operations.
    No JAX dependency. Input genotypes must fit in memory; for disk streaming
    use run_lmm_association_streaming.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2.
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples). WARNING: may be
            overwritten in-place during eigendecomposition (buffer reused for
            eigenvectors). Treat as consumed; pass kinship.copy() if you need
            the original matrix after this call.
        snp_info: List of dicts with keys: chr, rs, pos, a1, a0.
        covariates: Covariate matrix (n_samples, n_cvt) or None for intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations (clamped to min 20
            internally for ~1e-5 tolerance).
        use_gpu: Accepted but silently ignored — NumPy backend is CPU-only.
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.

    Returns:
        List of AssocResult for each SNP that passes filtering.

    Raises:
        MemoryError: If check_memory=True and insufficient memory.
        ValueError: If only one of eigenvalues/eigenvectors is provided,
            or if no valid samples remain after filtering.
    """
    # Validate eigendecomposition params - must provide both or neither
    if (eigenvalues is None) != (eigenvectors is None):
        raise ValueError(
            "Must provide both eigenvalues and eigenvectors, or neither. "
            f"Got eigenvalues={eigenvalues is not None}, "
            f"eigenvectors={eigenvectors is not None}"
        )

    if lmm_mode not in (1, 2, 3, 4):
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    if use_gpu:
        logger.debug("use_gpu ignored for NumPy backend")

    # Memory check before workflow
    n_samples, n_snps = genotypes.shape
    start_time = time.perf_counter()

    if show_progress:
        logger.info("Performing LMM Association Test (NumPy batch)")
        logger.info(f"  Total individuals: {n_samples:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.debug(
            f"MAF threshold = {maf_threshold}, missing threshold = {miss_threshold}"
        )

    # Always log memory estimate (useful even without hard check)
    est = estimate_lmm_memory(n_samples, n_snps)
    logger.info(
        f"LMM memory: estimated {est.total_gb:.1f}GB, "
        f"available {est.available_gb:.1f}GB"
    )
    if check_memory and not est.sufficient:
        raise MemoryError(
            f"Insufficient memory for LMM workflow with {n_samples:,} samples × "
            f"{n_snps:,} SNPs.\n"
            f"Need: {est.total_gb:.1f}GB, Available: {est.available_gb:.1f}GB\n"
            f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
            f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
            f"genotypes={est.genotypes_gb:.1f}GB"
        )

    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9.0)
    if covariates is not None:
        valid_covariate = np.all(~np.isnan(covariates), axis=1)
        valid_mask = valid_mask & valid_covariate
    if not np.all(valid_mask):
        genotypes = genotypes[valid_mask, :]
        phenotypes = phenotypes[valid_mask]
        kinship = kinship[np.ix_(valid_mask, valid_mask)]
        if covariates is not None:
            covariates = covariates[valid_mask, :]

    n_samples, n_snps = genotypes.shape
    if n_samples == 0:
        raise ValueError(
            "No valid samples: all phenotypes are missing or -9"
            + (", or all have missing covariates" if covariates is not None else "")
        )

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Vectorized SNP stats and filtering using shared functions
    col_means, missing_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, missing_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )
    snp_indices = np.where(snp_mask)[0]

    if len(snp_indices) == 0:
        return []

    # Extract filtered stats as numpy arrays (use allele_freqs for output, not mafs)
    filtered_afs = allele_freqs[snp_indices]
    filtered_miss = missing_counts[snp_indices].astype(int)

    t_eigen_start = time.perf_counter()
    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship,
        eigenvalues,
        eigenvectors,
        show_progress,
        "lmm_numpy",
        check_memory=check_memory,
    )
    del kinship
    gc.collect()

    # Use all physical cores for BLAS rotation
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = U.T @ W
        Uty = U.T @ phenotypes

    logl_H0, lambda_null_mle, Hi_eval_null = _compute_null_model_common(
        lmm_mode,
        eigenvalues_np,
        UtW,
        Uty,
        n_cvt,
        show_progress,
        l_min=l_min,
        l_max=l_max,
    )
    t_eigen_end = time.perf_counter()

    n_filtered = len(snp_indices)
    chunk_size = _compute_chunk_size_numpy(n_samples, n_filtered, n_cvt)

    n_chunks = (n_filtered + chunk_size - 1) // chunk_size
    if show_progress:
        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"  Processing in {n_chunks} chunks ({chunk_size:,} SNPs/chunk)"
            )

    # Pre-allocate result arrays driven by _RESULT_FIELDS mapping
    write_offset = 0
    arrays_out: dict[str, np.ndarray] = {
        key: np.empty(n_filtered, dtype=np.float64) for key in _RESULT_FIELDS[lmm_mode]
    }

    # Timing accumulators for per-chunk phases
    t_rotation_total = 0.0
    t_numpy_compute_total = 0.0
    t_result_write_total = 0.0

    chunk_starts = list(range(0, n_filtered, chunk_size))

    if show_progress and n_chunks > 1:
        chunk_iterator = progress_iterator(
            chunk_starts, total=n_chunks, desc="LMM association"
        )
    else:
        chunk_iterator = iter(chunk_starts)

    for chunk_start in chunk_iterator:
        chunk_end = min(chunk_start + chunk_size, n_filtered)
        chunk_indices = snp_indices[chunk_start:chunk_end]
        geno_chunk = genotypes[:, chunk_indices]

        # Mean-impute missing genotypes
        chunk_means = col_means[chunk_indices]
        missing = np.isnan(geno_chunk)
        geno_chunk = np.where(missing, chunk_means[None, :], geno_chunk)

        # Rotate genotypes
        t_rot_start = time.perf_counter()
        with blas_threads(rotation_threads):
            UtG = U.T @ geno_chunk
        t_rotation_total += time.perf_counter() - t_rot_start

        # Compute Uab batch
        t_compute_start = time.perf_counter()
        Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG)

        # Mode dispatch
        cr = _compute_lmm_chunk_numpy(
            lmm_mode,
            n_cvt,
            eigenvalues_np,
            Uab_batch,
            n_samples,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_refine=n_refine,
            Hi_eval_null=Hi_eval_null,
            logl_H0=logl_H0,
        )
        t_numpy_compute_total += time.perf_counter() - t_compute_start

        # Write results
        t_write_start = time.perf_counter()
        actual_len = chunk_end - chunk_start
        s = slice(write_offset, write_offset + actual_len)
        for key in arrays_out:
            arrays_out[key][s] = cr[key][:actual_len]
        write_offset += actual_len
        t_result_write_total += time.perf_counter() - t_write_start

    # Validate all results were written
    if write_offset != n_filtered:
        raise RuntimeError(
            f"Pre-allocated array size mismatch: wrote {write_offset} results,"
            f" expected {n_filtered}. This is an internal error — please report"
            f" this issue with your dataset dimensions."
        )

    # Log memory after all chunks processed
    if show_progress:
        log_rss_memory("lmm_numpy", "after_all_chunks")

    # Lambda boundary convergence diagnostics
    n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
        lmm_mode, arrays_out, l_min, l_max
    )
    log_lambda_boundary_warning(n_at_lmin, n_at_lmax, l_min, l_max)

    # Log completion
    elapsed = time.perf_counter() - start_time
    if show_progress:
        t_eigen = t_eigen_end - t_eigen_start
        accounted = (
            t_eigen + t_rotation_total + t_numpy_compute_total + t_result_write_total
        )
        logger.info("Timing breakdown:")
        logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
        logger.info(f"  UT@G rotation:       {t_rotation_total:.2f}s")
        logger.info(f"  NumPy compute:       {t_numpy_compute_total:.2f}s")
        logger.info(f"  Result write:        {t_result_write_total:.2f}s")
        logger.info("  ----")
        logger.info(f"  Accounted:           {accounted:.2f}s")
        logger.info(f"  Total:               {elapsed:.2f}s")
        logger.info(f"LMM Association completed in {elapsed:.2f}s")

    return _build_results(
        lmm_mode, snp_indices, filtered_afs, filtered_miss, snp_info, arrays_out
    )
