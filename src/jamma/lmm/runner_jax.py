"""JAX-optimized batch LMM association runner.

Batch-optimized LMM association testing on CPU (XLA) or GPU (JAX).
Input genotypes must fit in memory; for disk streaming use runner_streaming.py.
"""

import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jamma.core.memory import estimate_lmm_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.core.threading import blas_threads
from jamma.lmm.chunk import _compute_chunk_size
from jamma.lmm.compute import _compute_lmm_chunk
from jamma.lmm.likelihood_jax import batch_compute_uab
from jamma.lmm.prepare import (
    _build_covariate_matrix,
    _compute_null_model,
    _eigendecompose_or_reuse,
    _select_jax_device,
)
from jamma.lmm.results import (
    _build_results_all,
    _build_results_lrt,
    _build_results_score,
    _build_results_wald,
    _count_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory


def run_lmm_association_jax(
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
    """Run LMM association tests using JAX-optimized batch processing.

    Processes all SNPs in parallel via JAX vectorization and JIT compilation.
    SNPs are processed in chunks to avoid JAX int32 buffer overflow. Input
    genotypes must fit in memory; for disk streaming use run_lmm_association_streaming.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2.
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples).
        snp_info: List of dicts with keys: chr, rs, pos, a1, a0.
        covariates: Covariate matrix (n_samples, n_cvt) or None for intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations (min 20 for 1e-5 tolerance).
        use_gpu: Whether to use GPU acceleration.
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

    # Memory check before workflow
    n_samples, n_snps = genotypes.shape
    start_time = time.perf_counter()

    if show_progress:
        logger.info("Performing LMM Association Test (JAX batch)")
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

    device = _select_jax_device(use_gpu)

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

    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship,
        eigenvalues,
        eigenvectors,
        show_progress,
        "lmm_jax",
        check_memory=check_memory,
    )
    # Free kinship ref and LAPACK workspace before LMM phase
    del kinship
    gc.collect()

    with blas_threads():
        UtW = U.T @ W
        Uty = U.T @ phenotypes

    logl_H0, lambda_null_mle, Hi_eval_null_jax = _compute_null_model(
        lmm_mode,
        eigenvalues_np,
        UtW,
        Uty,
        n_cvt,
        device,
        show_progress,
        l_min=l_min,
        l_max=l_max,
    )

    # Determine chunk size to avoid int32 buffer overflow
    n_filtered = len(snp_indices)
    chunk_size = _compute_chunk_size(n_samples, n_filtered, n_grid, n_cvt)

    # Device-resident shared arrays - placed on device ONCE before chunk loop
    eigenvalues = jax.device_put(eigenvalues_np, device)
    UtW_jax = jax.device_put(UtW, device)
    Uty_jax = jax.device_put(Uty, device)

    # Process in chunks if needed
    n_chunks = (n_filtered + chunk_size - 1) // chunk_size
    if show_progress:
        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"  Processing in {n_chunks} chunks ({chunk_size:,} SNPs/chunk)"
            )

    # Pre-allocate result arrays (replaces list accumulators)
    write_offset = 0

    if lmm_mode == 1:  # Wald
        lambdas_out = np.empty(n_filtered, dtype=np.float64)
        logls_out = np.empty(n_filtered, dtype=np.float64)
        betas_out = np.empty(n_filtered, dtype=np.float64)
        ses_out = np.empty(n_filtered, dtype=np.float64)
        pwalds_out = np.empty(n_filtered, dtype=np.float64)
    elif lmm_mode == 3:  # Score
        betas_out = np.empty(n_filtered, dtype=np.float64)
        ses_out = np.empty(n_filtered, dtype=np.float64)
        p_scores_out = np.empty(n_filtered, dtype=np.float64)
    elif lmm_mode == 2:  # LRT
        lambdas_mle_out = np.empty(n_filtered, dtype=np.float64)
        p_lrts_out = np.empty(n_filtered, dtype=np.float64)
    elif lmm_mode == 4:  # All tests
        lambdas_out = np.empty(n_filtered, dtype=np.float64)
        logls_out = np.empty(n_filtered, dtype=np.float64)
        betas_out = np.empty(n_filtered, dtype=np.float64)
        ses_out = np.empty(n_filtered, dtype=np.float64)
        pwalds_out = np.empty(n_filtered, dtype=np.float64)
        lambdas_mle_out = np.empty(n_filtered, dtype=np.float64)
        p_lrts_out = np.empty(n_filtered, dtype=np.float64)
        p_scores_out = np.empty(n_filtered, dtype=np.float64)

    def _prepare_chunk(start: int) -> tuple[jnp.ndarray, int, bool]:
        """Prepare a chunk for device transfer (CPU work)."""
        end = min(start + chunk_size, n_filtered)
        actual_len = end - start

        chunk_indices = snp_indices[start:end]
        geno_chunk = genotypes[:, chunk_indices]
        chunk_means_local = col_means[chunk_indices]
        missing_mask = np.isnan(geno_chunk)
        geno_chunk = np.where(missing_mask, chunk_means_local[None, :], geno_chunk)

        needs_pad = actual_len < chunk_size
        if needs_pad:
            pad_width = chunk_size - actual_len
            geno_chunk = np.pad(geno_chunk, ((0, 0), (0, pad_width)), mode="constant")

        with blas_threads():
            UtG_chunk = np.ascontiguousarray(U.T @ geno_chunk)
        return UtG_chunk, actual_len, needs_pad

    # Double buffering: overlap device transfer with computation
    chunk_starts = list(range(0, n_filtered, chunk_size))

    # Prepare first chunk
    UtG_np, actual_len, needs_pad = _prepare_chunk(chunk_starts[0])
    UtG_jax = jax.device_put(UtG_np, device)
    del UtG_np

    # Create progress bar iterator
    if show_progress and n_chunks > 1:
        chunk_iterator = progress_iterator(
            enumerate(chunk_starts), total=n_chunks, desc="LMM association"
        )
    else:
        chunk_iterator = enumerate(chunk_starts)

    for i, _chunk_start in chunk_iterator:
        actual_chunk_len = actual_len
        needs_padding = needs_pad
        current_UtG = UtG_jax

        # Start async transfer of next chunk while computing current
        if i + 1 < len(chunk_starts):
            next_UtG_np, actual_len, needs_pad = _prepare_chunk(chunk_starts[i + 1])
            # device_put is async - transfer starts immediately, overlaps with compute
            UtG_jax = jax.device_put(next_UtG_np, device)
            del next_UtG_np

        try:
            # Batch compute Uab for this chunk (shared across all modes)
            Uab_batch = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, current_UtG)

            chunk_result = _compute_lmm_chunk(
                lmm_mode, n_cvt, eigenvalues, Uab_batch, n_samples,
                l_min=l_min, l_max=l_max, n_grid=n_grid, n_refine=n_refine,
                Hi_eval_null=Hi_eval_null_jax, logl_H0=logl_H0,
            )
            best_lambdas = chunk_result["best_lambdas"]
            best_logls = chunk_result["best_logls"]
            betas = chunk_result["betas"]
            ses = chunk_result["ses"]
            p_walds = chunk_result["p_walds"]
            best_lambdas_mle = chunk_result["best_lambdas_mle"]
            p_lrts = chunk_result["p_lrts"]
            p_scores = chunk_result["p_scores"]

        except Exception as e:
            error_msg = str(e)
            # Check for int32 overflow error
            if "exceeds the maximum representable value" in error_msg:
                n_index = (n_cvt + 3) * (n_cvt + 2) // 2
                buffer_elements = n_samples * chunk_size * n_index
                logger.error(
                    f"JAX int32 buffer overflow during LMM computation.\n"
                    f"  Chunk {i + 1}/{n_chunks}: {chunk_size:,} SNPs x "
                    f"{n_samples:,} samples\n"
                    f"  Buffer elements: {buffer_elements:,} (limit: ~2.1B)\n"
                    f"  This should not happen with automatic chunking.\n"
                    f"  Please report this issue with your dataset dimensions."
                )
            else:
                logger.error(
                    f"JAX computation failed on chunk {i + 1}/{n_chunks}:\n"
                    f"  {type(e).__name__}: {error_msg}\n"
                    f"  Chunk size: {chunk_size:,} SNPs, Samples: {n_samples:,}"
                )
            raise

        # Write results into pre-allocated arrays by index (no list append)
        if lmm_mode == 1:
            slice_len = actual_chunk_len if needs_padding else len(best_lambdas)
            s = slice(write_offset, write_offset + slice_len)
            lambdas_out[s] = np.asarray(best_lambdas[:slice_len])
            logls_out[s] = np.asarray(best_logls[:slice_len])
            betas_out[s] = np.asarray(betas[:slice_len])
            ses_out[s] = np.asarray(ses[:slice_len])
            pwalds_out[s] = np.asarray(p_walds[:slice_len])
        elif lmm_mode == 3:
            slice_len = actual_chunk_len if needs_padding else len(betas)
            s = slice(write_offset, write_offset + slice_len)
            betas_out[s] = np.asarray(betas[:slice_len])
            ses_out[s] = np.asarray(ses[:slice_len])
            p_scores_out[s] = np.asarray(p_scores[:slice_len])
        elif lmm_mode == 2:
            slice_len = actual_chunk_len if needs_padding else len(best_lambdas_mle)
            s = slice(write_offset, write_offset + slice_len)
            lambdas_mle_out[s] = np.asarray(best_lambdas_mle[:slice_len])
            p_lrts_out[s] = np.asarray(p_lrts[:slice_len])
        elif lmm_mode == 4:
            slice_len = actual_chunk_len if needs_padding else len(best_lambdas)
            s = slice(write_offset, write_offset + slice_len)
            # Wald
            lambdas_out[s] = np.asarray(best_lambdas[:slice_len])
            logls_out[s] = np.asarray(best_logls[:slice_len])
            betas_out[s] = np.asarray(betas[:slice_len])
            ses_out[s] = np.asarray(ses[:slice_len])
            pwalds_out[s] = np.asarray(p_walds[:slice_len])
            # LRT
            lambdas_mle_out[s] = np.asarray(best_lambdas_mle[:slice_len])
            p_lrts_out[s] = np.asarray(p_lrts[:slice_len])
            # Score
            p_scores_out[s] = np.asarray(p_scores[:slice_len])

        write_offset += slice_len

    # Validate all results were written
    assert write_offset == n_filtered, (
        f"Pre-allocated array size mismatch: wrote {write_offset},"
        f" expected {n_filtered}"
    )

    # Log memory after all chunks processed
    if show_progress:
        log_rss_memory("lmm_jax", "after_all_chunks")

    # Lambda boundary convergence diagnostics
    n_at_lmin, n_at_lmax = 0, 0
    if lmm_mode in (1, 4):
        lmin_hits, lmax_hits = _count_boundary_hits(lambdas_out, l_min, l_max)
        n_at_lmin += lmin_hits
        n_at_lmax += lmax_hits
    if lmm_mode in (2, 4):
        lmin_hits, lmax_hits = _count_boundary_hits(lambdas_mle_out, l_min, l_max)
        n_at_lmin += lmin_hits
        n_at_lmax += lmax_hits
    log_lambda_boundary_warning(n_at_lmin, n_at_lmax, l_min, l_max)

    # Explicit cleanup of JAX arrays before returning to prevent SIGSEGV
    # from race conditions between Python GC and JAX background threads
    del eigenvalues, UtW_jax, Uty_jax
    jax.clear_caches()

    # Log completion
    elapsed = time.perf_counter() - start_time
    if show_progress:
        logger.info(f"LMM Association completed in {elapsed:.2f}s")

    if lmm_mode == 1:
        return _build_results_wald(
            snp_indices,
            filtered_afs,
            filtered_miss,
            snp_info,
            lambdas_out,
            logls_out,
            betas_out,
            ses_out,
            pwalds_out,
        )
    elif lmm_mode == 3:
        return _build_results_score(
            snp_indices,
            filtered_afs,
            filtered_miss,
            snp_info,
            betas_out,
            ses_out,
            p_scores_out,
        )
    elif lmm_mode == 2:
        return _build_results_lrt(
            snp_indices,
            filtered_afs,
            filtered_miss,
            snp_info,
            lambdas_mle_out,
            p_lrts_out,
        )
    else:  # lmm_mode == 4 (validated at top)
        return _build_results_all(
            snp_indices,
            filtered_afs,
            filtered_miss,
            snp_info,
            lambdas_out,
            logls_out,
            betas_out,
            ses_out,
            pwalds_out,
            lambdas_mle_out,
            p_lrts_out,
            p_scores_out,
        )
