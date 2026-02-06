"""JAX-optimized batch LMM association runner.

Batch-optimized LMM association testing on CPU (XLA) or GPU (JAX).
Input genotypes must fit in memory; for disk streaming use runner_streaming.py.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jamma.core.memory import estimate_workflow_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.lmm.chunk import (
    _MAX_BUFFER_ELEMENTS,  # noqa: F401 - re-export for backward compatibility
    MAX_SAFE_CHUNK,  # noqa: F401 - re-export for backward compatibility
    _compute_chunk_size,
    auto_tune_chunk_size,  # noqa: F401 - re-export for backward compatibility
)
from jamma.lmm.likelihood_jax import (
    batch_calc_score_stats,
    batch_calc_wald_stats,
    batch_compute_iab,
    batch_compute_uab,
    calc_lrt_pvalue_jax,
    golden_section_optimize_lambda_mle,
)
from jamma.lmm.prepare import (
    _build_covariate_matrix,
    _compute_null_model,
    _eigendecompose_or_reuse,
    _grid_optimize_lambda_batched,
    _select_jax_device,
)
from jamma.lmm.results import (
    _build_results_all,
    _build_results_lrt,
    _build_results_score,
    _build_results_wald,
)
from jamma.lmm.runner_streaming import (
    run_lmm_association_streaming,  # noqa: F401 - re-export for backward compatibility
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
        ValueError: If only one of eigenvalues/eigenvectors is provided.
    """
    # Validate eigendecomposition params - must provide both or neither
    if (eigenvalues is None) != (eigenvectors is None):
        raise ValueError(
            "Must provide both eigenvalues and eigenvectors, or neither. "
            f"Got eigenvalues={eigenvalues is not None}, "
            f"eigenvectors={eigenvectors is not None}"
        )

    # Memory check before workflow
    n_samples, n_snps = genotypes.shape
    start_time = time.perf_counter()

    if show_progress:
        logger.info("## Performing LMM Association Test (JAX)")
        logger.info(f"number of total individuals = {n_samples:,}")
        logger.info(f"number of total SNPs/variants = {n_snps:,}")
        logger.debug(
            f"MAF threshold = {maf_threshold}, missing threshold = {miss_threshold}"
        )

    if check_memory:
        est = estimate_workflow_memory(n_samples, n_snps)
        if not est.sufficient:
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

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Vectorized SNP stats and filtering using shared functions
    col_means, missing_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, missing_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )
    snp_indices = np.where(snp_mask)[0]

    if len(snp_indices) == 0:
        return []

    # Extract filtered stats (use allele_freqs for output, not mafs)
    snp_stats = list(
        zip(
            allele_freqs[snp_indices],
            missing_counts[snp_indices].astype(int),
            strict=False,
        )
    )

    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship, eigenvalues, eigenvectors, show_progress, "lmm_jax"
    )

    UtW = U.T @ W
    Uty = U.T @ phenotypes

    logl_H0, lambda_null_mle, Hi_eval_null_jax = _compute_null_model(
        lmm_mode, eigenvalues_np, UtW, Uty, n_cvt, device, show_progress
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
        logger.info(f"number of analyzed individuals = {n_samples:,}")
        logger.info(f"number of analyzed SNPs = {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"Processing in {n_chunks} chunks "
                f"({chunk_size:,} SNPs/chunk) to avoid buffer overflow"
            )

    # Mode-aware accumulators
    if lmm_mode == 1:  # Wald
        all_lambdas = []
        all_logls = []
        all_betas = []
        all_ses = []
        all_pwalds = []
    elif lmm_mode == 3:  # Score
        all_betas = []
        all_ses = []
        all_p_scores = []
    elif lmm_mode == 2:  # LRT
        all_lambdas_mle = []
        all_logls_mle = []
        all_p_lrts = []
    elif lmm_mode == 4:  # All tests
        # Wald accumulators
        all_lambdas = []
        all_logls = []
        all_betas = []
        all_ses = []
        all_pwalds = []
        # LRT accumulators
        all_lambdas_mle = []
        all_p_lrts = []
        # Score accumulator
        all_p_scores = []

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

            if lmm_mode == 1:  # Wald (existing, unchanged)
                Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
                best_lambdas, best_logls = _grid_optimize_lambda_batched(
                    n_cvt,
                    eigenvalues,
                    Uab_batch,
                    Iab_batch,
                    l_min,
                    l_max,
                    n_grid,
                    n_refine,
                )
                betas, ses, p_walds = batch_calc_wald_stats(
                    n_cvt, best_lambdas, eigenvalues, Uab_batch, n_samples
                )

            elif lmm_mode == 3:  # Score
                betas, ses, p_scores = batch_calc_score_stats(
                    n_cvt, Hi_eval_null_jax, Uab_batch, n_samples
                )

            elif lmm_mode == 2:  # LRT
                best_lambdas_mle, best_logls_mle = golden_section_optimize_lambda_mle(
                    n_cvt,
                    eigenvalues,
                    Uab_batch,
                    l_min=l_min,
                    l_max=l_max,
                    n_grid=n_grid,
                    n_iter=max(n_refine, 20),
                )
                p_lrts = jax.vmap(calc_lrt_pvalue_jax)(
                    best_logls_mle, jnp.full_like(best_logls_mle, logl_H0)
                )

            elif lmm_mode == 4:  # All tests
                # Score test (cheapest, no optimization, reads Uab_batch)
                _, _, p_scores = batch_calc_score_stats(
                    n_cvt, Hi_eval_null_jax, Uab_batch, n_samples
                )

                # MLE optimization for LRT
                best_lambdas_mle, best_logls_mle = golden_section_optimize_lambda_mle(
                    n_cvt,
                    eigenvalues,
                    Uab_batch,
                    l_min=l_min,
                    l_max=l_max,
                    n_grid=n_grid,
                    n_iter=max(n_refine, 20),
                )
                p_lrts = jax.vmap(calc_lrt_pvalue_jax)(
                    best_logls_mle, jnp.full_like(best_logls_mle, logl_H0)
                )

                # REML optimization for Wald
                Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
                best_lambdas, best_logls = _grid_optimize_lambda_batched(
                    n_cvt,
                    eigenvalues,
                    Uab_batch,
                    Iab_batch,
                    l_min,
                    l_max,
                    n_grid,
                    n_refine,
                )
                betas, ses, p_walds = batch_calc_wald_stats(
                    n_cvt, best_lambdas, eigenvalues, Uab_batch, n_samples
                )

        except Exception as e:
            error_msg = str(e)
            # Check for int32 overflow error
            if "exceeds the maximum representable value" in error_msg:
                n_index = (n_cvt + 3) * (n_cvt + 2) // 2
                buffer_elements = n_samples * chunk_size * n_index
                logger.error(
                    f"JAX int32 buffer overflow during LMM computation.\n"
                    f"  Chunk {i+1}/{n_chunks}: {chunk_size:,} SNPs x "
                    f"{n_samples:,} samples\n"
                    f"  Buffer elements: {buffer_elements:,} (limit: ~2.1B)\n"
                    f"  This should not happen with automatic chunking.\n"
                    f"  Please report this issue with your dataset dimensions."
                )
            else:
                logger.error(
                    f"JAX computation failed on chunk {i+1}/{n_chunks}:\n"
                    f"  {type(e).__name__}: {error_msg}\n"
                    f"  Chunk size: {chunk_size:,} SNPs, Samples: {n_samples:,}"
                )
            raise

        # Strip padding if needed, keep as JAX arrays to avoid per-chunk host transfer
        if lmm_mode == 1:
            slice_len = actual_chunk_len if needs_padding else len(best_lambdas)
            all_lambdas.append(best_lambdas[:slice_len])
            all_logls.append(best_logls[:slice_len])
            all_betas.append(betas[:slice_len])
            all_ses.append(ses[:slice_len])
            all_pwalds.append(p_walds[:slice_len])
        elif lmm_mode == 3:
            slice_len = actual_chunk_len if needs_padding else len(betas)
            all_betas.append(betas[:slice_len])
            all_ses.append(ses[:slice_len])
            all_p_scores.append(p_scores[:slice_len])
        elif lmm_mode == 2:
            slice_len = actual_chunk_len if needs_padding else len(best_lambdas_mle)
            all_lambdas_mle.append(best_lambdas_mle[:slice_len])
            all_logls_mle.append(best_logls_mle[:slice_len])
            all_p_lrts.append(p_lrts[:slice_len])
        elif lmm_mode == 4:
            slice_len = actual_chunk_len if needs_padding else len(best_lambdas)
            # Wald
            all_lambdas.append(best_lambdas[:slice_len])
            all_logls.append(best_logls[:slice_len])
            all_betas.append(betas[:slice_len])
            all_ses.append(ses[:slice_len])
            all_pwalds.append(p_walds[:slice_len])
            # LRT
            all_lambdas_mle.append(best_lambdas_mle[:slice_len])
            all_p_lrts.append(p_lrts[:slice_len])
            # Score
            all_p_scores.append(p_scores[:slice_len])

    # Log memory after all chunks processed
    if show_progress:
        log_rss_memory("lmm_jax", "after_all_chunks")

    # Concatenate on device, then single host transfer (mode-aware)
    if lmm_mode == 1:
        best_lambdas_np = np.asarray(jnp.concatenate(all_lambdas))
        best_logls_np = np.asarray(jnp.concatenate(all_logls))
        betas_np = np.asarray(jnp.concatenate(all_betas))
        ses_np = np.asarray(jnp.concatenate(all_ses))
        p_walds_np = np.asarray(jnp.concatenate(all_pwalds))
        del all_lambdas, all_logls, all_betas, all_ses, all_pwalds
    elif lmm_mode == 3:
        betas_np = np.asarray(jnp.concatenate(all_betas))
        ses_np = np.asarray(jnp.concatenate(all_ses))
        p_scores_np = np.asarray(jnp.concatenate(all_p_scores))
        del all_betas, all_ses, all_p_scores
    elif lmm_mode == 2:
        lambdas_mle_np = np.asarray(jnp.concatenate(all_lambdas_mle))
        p_lrts_np = np.asarray(jnp.concatenate(all_p_lrts))
        del all_lambdas_mle, all_logls_mle, all_p_lrts
    elif lmm_mode == 4:
        best_lambdas_np = np.asarray(jnp.concatenate(all_lambdas))
        best_logls_np = np.asarray(jnp.concatenate(all_logls))
        betas_np = np.asarray(jnp.concatenate(all_betas))
        ses_np = np.asarray(jnp.concatenate(all_ses))
        p_walds_np = np.asarray(jnp.concatenate(all_pwalds))
        lambdas_mle_np = np.asarray(jnp.concatenate(all_lambdas_mle))
        p_lrts_np = np.asarray(jnp.concatenate(all_p_lrts))
        p_scores_np = np.asarray(jnp.concatenate(all_p_scores))
        del all_lambdas, all_logls, all_betas, all_ses, all_pwalds
        del all_lambdas_mle, all_p_lrts, all_p_scores

    # Explicit cleanup of JAX arrays before returning to prevent SIGSEGV
    # from race conditions between Python GC and JAX background threads
    del eigenvalues, UtW_jax, Uty_jax
    # Force synchronization - ensures all JAX operations complete before returning
    if lmm_mode == 1:
        jax.block_until_ready(betas_np)
    elif lmm_mode == 3:
        jax.block_until_ready(p_scores_np)
    elif lmm_mode == 2:
        jax.block_until_ready(p_lrts_np)
    elif lmm_mode == 4:
        jax.block_until_ready(betas_np)

    # Log completion
    elapsed = time.perf_counter() - start_time
    if show_progress:
        logger.info("## LMM Association completed")
        logger.info(f"time elapsed = {elapsed:.2f} seconds")

    if lmm_mode == 1:
        return _build_results_wald(
            snp_indices,
            snp_stats,
            snp_info,
            best_lambdas_np,
            best_logls_np,
            betas_np,
            ses_np,
            p_walds_np,
        )
    elif lmm_mode == 3:
        return _build_results_score(
            snp_indices,
            snp_stats,
            snp_info,
            betas_np,
            ses_np,
            p_scores_np,
        )
    elif lmm_mode == 2:
        return _build_results_lrt(
            snp_indices,
            snp_stats,
            snp_info,
            lambdas_mle_np,
            p_lrts_np,
        )
    elif lmm_mode == 4:
        return _build_results_all(
            snp_indices,
            snp_stats,
            snp_info,
            best_lambdas_np,
            best_logls_np,
            betas_np,
            ses_np,
            p_walds_np,
            lambdas_mle_np,
            p_lrts_np,
            p_scores_np,
        )
    else:
        raise ValueError(f"Unsupported lmm_mode: {lmm_mode}. Use 1, 2, 3, or 4.")
