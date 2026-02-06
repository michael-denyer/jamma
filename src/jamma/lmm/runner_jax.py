"""JAX-optimized LMM association runner.

Provides batch-optimized LMM association testing that works efficiently
on both CPU (via XLA) and GPU (via JAX device abstraction).

Key optimizations vs sequential NumPy version:
1. Batch genotype rotation: O(1) matrix multiply vs O(n_snps) loop
2. Batch Uab computation: vmap parallelizes across SNPs
3. Hybrid grid + golden section optimization: All SNPs in parallel
4. Batch Wald statistics: vmap parallelizes final computations

Mathematical Equivalence to GEMMA
=================================
This implementation produces scientifically identical results to GEMMA:
- P-value rank correlation: 1.000000 (perfect)
- Significance agreement: 100% at all thresholds (0.05, 0.01, 0.001, 5e-8)
- Effect direction agreement: 100%

Numerical differences (within tolerance) arise from:

1. **Lambda optimization method**:
   - GEMMA/NumPy: Brent's method (derivative-free, ~50 function evaluations)
   - JAX: Grid search + golden section (~70 function evaluations)
   - Both converge to identical optima (max rel diff < 1e-5)

2. **Why golden section matches Brent**:
   - REML surface ℓ(λ) is unimodal and smooth
   - Grid search brackets the optimum within ±1 grid cell on log scale
   - Golden section achieves O(0.618^n) convergence rate
   - 20 iterations: 0.618^20 ≈ 6.6e-5 relative tolerance

3. **F-distribution CDF**:
   - GEMMA: GSL gsl_cdf_fdist_Q
   - JAX: scipy.special.betainc
   - Max p-value difference: ~4e-5 (statistically negligible)

Performance vs GEMMA
====================
On 1940 samples × 12K SNPs (CPU):
- GEMMA (C++/LAPACK): ~19s
- JAMMA NumPy (Brent): ~24s (1.26x slower)
- JAMMA JAX (golden): ~10s (1.9x faster than GEMMA)

GPU acceleration provides additional speedup for larger datasets.

Chunked Processing
==================
For large-scale analyses (>25K samples), SNPs are processed in chunks to:
1. Avoid JAX int32 buffer index overflow (keeps elements below INT32_MAX)
2. Avoid materializing full rotated genotype matrix UtG (n_samples × n_snps)

Note: The input genotypes array must still fit in memory. Chunking only reduces
peak memory for intermediate arrays (UtG, Uab). For true streaming from disk,
use run_lmm_association_streaming() which streams genotypes from disk.

Streaming Mode
==============
run_lmm_association_streaming() reads genotypes from disk per-chunk:
1. Never allocates full (n_samples, n_snps) genotype array
2. Two-pass approach: SNP stats pass, then association pass
3. Memory: eigenvectors O(n^2) + chunk O(n * chunk_size)

Combined with compute_kinship_streaming(), enables full GWAS workflow
without ever loading the complete genotype matrix.

Usage:
    from jamma.lmm.runner_jax import run_lmm_association_jax
    from jamma.lmm.runner_jax import run_lmm_association_streaming

    # Full-load version (genotypes in memory)
    results = run_lmm_association_jax(
        genotypes, phenotypes, kinship, snp_info,
        use_gpu=True  # Set False for CPU-only
    )

    # Streaming version (genotypes from disk)
    results = run_lmm_association_streaming(
        bed_path, phenotypes, kinship, snp_info,
        chunk_size=10_000
    )
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jamma.core.memory import (
    estimate_streaming_memory,
    estimate_workflow_memory,
)
from jamma.core.progress import progress_iterator
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.lmm.chunk import (
    _MAX_BUFFER_ELEMENTS,  # noqa: F401 - re-export for backward compatibility
    MAX_SAFE_CHUNK,  # noqa: F401 - re-export for backward compatibility
    _compute_chunk_size,
    auto_tune_chunk_size,  # noqa: F401 - re-export for backward compatibility
)
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.likelihood import compute_null_model_mle
from jamma.lmm.likelihood_jax import (
    batch_calc_score_stats,
    batch_calc_wald_stats,
    batch_compute_iab,
    batch_compute_uab,
    calc_lrt_pvalue_jax,
    golden_section_optimize_lambda,
    golden_section_optimize_lambda_mle,
)
from jamma.lmm.results import (
    _build_results_all,
    _build_results_lrt,
    _build_results_score,
    _build_results_wald,
    _snp_metadata,
)
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory


def _select_jax_device(use_gpu: bool) -> jax.Device:
    """Select JAX compute device with safe GPU detection.

    Falls back to CPU if GPU backend is unavailable.

    Args:
        use_gpu: Whether to attempt GPU selection.

    Returns:
        JAX device to use for computation.
    """
    device = jax.devices("cpu")[0]
    if use_gpu:
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                device = gpu_devices[0]
        except RuntimeError:
            pass
    return device


def _build_covariate_matrix(
    covariates: np.ndarray | None, n_samples: int
) -> tuple[np.ndarray, int]:
    """Construct covariate matrix W and return (W, n_cvt).

    If covariates is None, uses intercept-only model. Warns if provided
    covariates lack an intercept column.

    Args:
        covariates: Optional covariate matrix (n_samples, n_covariates).
        n_samples: Number of samples (for intercept construction).

    Returns:
        Tuple of (W, n_cvt) where W is the covariate matrix.
    """
    if covariates is None:
        W = np.ones((n_samples, 1))
    else:
        W = covariates.astype(np.float64)
        if not np.allclose(W[:, 0], 1.0):
            logger.warning(
                "Covariate matrix does not have intercept column "
                "(first column is not all 1s). "
                "Model will NOT include an intercept term."
            )
    return W, W.shape[1]


def _eigendecompose_or_reuse(
    kinship: np.ndarray,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    show_progress: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigendecomposition, computing it if not provided.

    Args:
        kinship: Kinship matrix (n_samples, n_samples).
        eigenvalues: Pre-computed eigenvalues or None.
        eigenvectors: Pre-computed eigenvectors or None.
        show_progress: Whether to log memory usage.
        label: Label for memory logging (e.g. "lmm_jax", "lmm_streaming").

    Returns:
        Tuple of (eigenvalues, eigenvectors).
    """
    if eigenvalues is not None and eigenvectors is not None:
        if show_progress:
            logger.debug("Using pre-computed eigendecomposition")
        return eigenvalues, eigenvectors

    if show_progress:
        log_rss_memory(label, "before_eigendecomp")
    eigenvalues_np, U = eigendecompose_kinship(kinship)
    if show_progress:
        log_rss_memory(label, "after_eigendecomp")
    return eigenvalues_np, U


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

    This version processes all SNPs in parallel using JAX's vectorization
    and JIT compilation. Significantly faster than sequential NumPy version
    for large datasets, especially on GPU.

    Supports arbitrary covariates. If covariates=None, uses intercept-only
    model (n_cvt=1). Covariate matrix should include an intercept column
    (first column all 1s) if desired, matching GEMMA -c flag behavior.

    Memory Scaling:
        SNPs are processed in chunks to bound intermediate array sizes:
        - Uab array: (chunk_size, n_samples, n_index) for projection computation
          where n_index = (n_cvt+3)*(n_cvt+2)//2
        - Lambda grid: (n_grid, chunk_size) for optimization
        - UtG_chunk: (n_samples, chunk_size) rotated genotypes per chunk

        Chunk size is computed to avoid JAX int32 buffer overflow. Note that:
        - Input genotypes array must still fit in memory (O(n_samples x n_snps))
        - Kinship and eigenvectors require O(n_samples^2) memory
        - GPU mode transfers each chunk from CPU to device (rotation is CPU)

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2
        phenotypes: Phenotype vector (n_samples,)
        kinship: Kinship matrix (n_samples, n_samples)
        snp_info: List of dicts with keys: chr, rs, pos, a1, a0
        covariates: Optional covariate matrix (n_samples, n_covariates).
            If None, uses intercept-only model. If provided, should include
            intercept column if desired (GEMMA -c flag behavior).
            Samples with NaN covariates are excluded from analysis.
        eigenvalues: Pre-computed eigenvalues from kinship decomposition. If provided
            along with eigenvectors, skips eigendecomposition (saves significant time
            for large matrices). Must be sorted ascending.
        eigenvectors: Pre-computed eigenvectors from kinship decomposition. Columns
            are eigenvectors corresponding to eigenvalues.
        maf_threshold: Minimum MAF for SNP inclusion
        miss_threshold: Maximum missing rate for SNP inclusion
        l_min: Minimum lambda for optimization
        l_max: Maximum lambda for optimization
        n_grid: Grid search resolution for initial lambda bracketing
        n_refine: Golden section iterations for lambda refinement (min 20 for 1e-5 tol)
        use_gpu: Whether to use GPU acceleration (requires JAX GPU setup)
        check_memory: If True (default), check available memory before workflow
            and raise MemoryError if insufficient.
        show_progress: If True (default), show progress bars and GEMMA-style logging.
        lmm_mode: Test type (default 1).
            - 1 (Wald): Per-SNP REML optimization, full statistics.
            - 2 (LRT): Null MLE once, per-SNP MLE optimization, LRT p-values.
            - 3 (Score): Null MLE once, no per-SNP optimization (fastest).
            - 4 (All): All three tests in a single pass (~2x mode 1 cost).

    Returns:
        List of AssocResult for each SNP that passes filtering

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
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

    # Filter samples with missing phenotypes or missing covariates
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

    # Vectorized SNP stats computation (replaces per-SNP Python loop)
    missing_counts = np.sum(np.isnan(genotypes), axis=0)  # (n_snps,)
    miss_rates = missing_counts / n_samples

    # Compute allele frequencies handling missing values
    with np.errstate(invalid="ignore"):  # Suppress warnings for all-NaN columns
        col_means = np.nanmean(genotypes, axis=0)  # Mean of non-missing
        col_vars = np.nanvar(genotypes, axis=0)  # Variance for monomorphic detection
    col_means = np.nan_to_num(col_means, nan=0.0)  # Handle all-missing columns
    col_vars = np.nan_to_num(col_vars, nan=0.0)
    allele_freqs = col_means / 2.0
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)

    # Detect monomorphic SNPs (variance == 0 means constant genotype)
    is_polymorphic = col_vars > 0

    # Filter SNPs by MAF, missing rate, and monomorphism
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic
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

    # Prepare rotated matrices
    UtW = U.T @ W
    Uty = U.T @ phenotypes

    logl_H0, lambda_null_mle, Hi_eval_null_jax = _compute_null_model(
        lmm_mode, eigenvalues_np, UtW, Uty, n_cvt, device, show_progress
    )

    # Determine chunk size to avoid int32 buffer overflow
    n_filtered = len(snp_indices)
    chunk_size = _compute_chunk_size(n_samples, n_filtered, n_grid, n_cvt)

    # Device-resident shared arrays - placed on device ONCE before chunk loop
    # These arrays are used across all chunks and should not be re-transferred
    # - eigenvalues: Used in every lambda optimization and Wald stat calculation
    # - UtW_jax: Rotated covariates (constant across SNPs)
    # - Uty_jax: Rotated phenotypes (constant across SNPs)
    # Direct device_put - JAX handles numpy conversion efficiently
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
    # While GPU computes on buffer A, CPU prepares and transfers buffer B
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


def _compute_null_model(
    lmm_mode: int,
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    device: jax.Device,
    show_progress: bool,
) -> tuple[float | None, float | None, jnp.ndarray | None]:
    """Compute null model MLE for Score, LRT, and All-tests modes.

    Score test (mode 3) and All-tests (mode 4) additionally precompute Hi_eval
    at the null lambda. Wald (mode 1) skips this entirely.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        eigenvalues_np: Kinship eigenvalues as numpy array.
        UtW: Rotated covariates.
        Uty: Rotated phenotype.
        n_cvt: Number of covariates.
        device: JAX device for Hi_eval placement.
        show_progress: Whether to log results.

    Returns:
        Tuple of (logl_H0, lambda_null_mle, Hi_eval_null_jax).
        All None for Wald mode.
    """
    if lmm_mode not in (2, 3, 4):
        return None, None, None

    lambda_null_mle, logl_H0 = compute_null_model_mle(eigenvalues_np, UtW, Uty, n_cvt)
    if show_progress:
        logger.info(
            f"Null model MLE: lambda={lambda_null_mle:.6f}, logl_H0={logl_H0:.6f}"
        )

    Hi_eval_null_jax = None
    if lmm_mode in (3, 4):
        Hi_eval_null = 1.0 / (lambda_null_mle * eigenvalues_np + 1.0)
        Hi_eval_null_jax = jax.device_put(Hi_eval_null, device)

    return logl_H0, lambda_null_mle, Hi_eval_null_jax


def _grid_optimize_lambda_batched(
    n_cvt: int,
    eigenvalues: jnp.ndarray,
    Uab_batch: jnp.ndarray,
    Iab_batch: jnp.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batch lambda optimization using grid search + golden section refinement.

    Delegates to golden_section_optimize_lambda with precomputed Iab and at
    least 20 iterations to achieve ~1e-5 relative tolerance.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Eigenvalues (n_samples,)
        Uab_batch: Uab matrices (n_snps, n_samples, n_index)
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index)
        l_min, l_max: Lambda bounds
        n_grid: Coarse grid points
        n_refine: Golden section iterations
    """
    return golden_section_optimize_lambda(
        n_cvt,
        eigenvalues,
        Uab_batch,
        Iab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=max(n_refine, 20),
    )


def run_lmm_association_streaming(
    bed_path: Path,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list | None = None,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    chunk_size: int = 10_000,
    use_gpu: bool = False,
    check_memory: bool = True,
    show_progress: bool = True,
    output_path: Path | None = None,
    lmm_mode: int = 1,
) -> list[AssocResult]:
    """Run LMM association tests by streaming genotypes from disk.

    This version reads genotypes per-chunk from disk, never allocating the full
    (n_samples, n_snps) genotype matrix. Combined with compute_kinship_streaming(),
    enables full GWAS workflow without ever loading the complete genotype matrix.

    Supports arbitrary covariates. If covariates=None, uses intercept-only
    model (n_cvt=1). Covariate matrix should include an intercept column
    (first column all 1s) if desired, matching GEMMA -c flag behavior.

    Two-pass approach:
    1. SNP statistics pass: Compute MAF and missing rate for filtering
    2. Association pass: For each chunk, rotate genotypes and compute stats

    Memory Scaling:
        Peak memory is dominated by eigendecomposition:
        - Kinship + eigenvectors: 2 * n_samples^2 * 8 bytes
        - Chunk buffer: n_samples * chunk_size * 8 bytes
        - Never allocates full genotype matrix

    Args:
        bed_path: Path prefix for PLINK files (without .bed/.bim/.fam extension).
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples).
        snp_info: Optional list of dicts with keys: chr, rs, pos, a1, a0.
            If None, builds from PLINK metadata.
        covariates: Optional covariate matrix (n_samples, n_covariates).
            If None, uses intercept-only model. If provided, should include
            intercept column if desired (GEMMA -c flag behavior).
            Samples with NaN covariates are excluded from analysis.
        eigenvalues: Pre-computed eigenvalues from kinship decomposition. If provided
            along with eigenvectors, skips eigendecomposition (saves significant time
            for large matrices). Must be sorted ascending.
        eigenvectors: Pre-computed eigenvectors from kinship decomposition. Columns
            are eigenvectors corresponding to eigenvalues.
        maf_threshold: Minimum MAF for SNP inclusion (default: 0.01).
        miss_threshold: Maximum missing rate for SNP inclusion (default: 0.05).
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for initial lambda bracketing.
        n_refine: Golden section iterations for lambda refinement.
        chunk_size: Number of SNPs per chunk (default: 10,000).
        use_gpu: Whether to use GPU acceleration.
        check_memory: If True (default), check available memory before workflow.
        show_progress: If True (default), show progress bars and GEMMA-style logging.
        output_path: Optional path for incremental result writing. If provided,
            results are written to disk as computed instead of being accumulated
            in memory. Returns empty list when output_path is set (results are
            on disk). If None (default), returns list of AssocResult as before.
        lmm_mode: Test type (default 1).
            - 1 (Wald): Per-SNP REML optimization, full statistics.
            - 2 (LRT): Null MLE once, per-SNP MLE optimization, LRT p-values.
            - 3 (Score): Null MLE once, no per-SNP optimization (fastest).
            - 4 (All): All three tests in a single pass (~2x mode 1 cost).

    Returns:
        List of AssocResult for each SNP that passes filtering. Empty list if
        output_path is provided (results are written to disk instead).

    Raises:
        MemoryError: If check_memory=True and insufficient memory available.
        FileNotFoundError: If the .bed file does not exist.
        ValueError: If only one of eigenvalues/eigenvectors is provided.
    """
    start_time = time.perf_counter()

    # Validate eigendecomposition params - must provide both or neither
    if (eigenvalues is None) != (eigenvectors is None):
        raise ValueError(
            "Must provide both eigenvalues and eigenvectors, or neither. "
            f"Got eigenvalues={eigenvalues is not None}, "
            f"eigenvectors={eigenvectors is not None}"
        )

    # Get metadata without loading genotypes
    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps = meta["n_snps"]

    # Build snp_info from metadata if not provided
    if snp_info is None:
        snp_info = [
            {
                "chr": str(meta["chromosome"][i]),
                "rs": meta["sid"][i],
                "pos": int(meta["bp_position"][i]),
                "a1": meta["allele_1"][i],
                "a0": meta["allele_2"][i],
            }
            for i in range(n_snps)
        ]

    # Filter samples with missing phenotypes or missing covariates
    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9.0)
    if covariates is not None:
        valid_covariate = np.all(~np.isnan(covariates), axis=1)
        valid_mask = valid_mask & valid_covariate
    n_valid = int(np.sum(valid_mask))
    if not np.all(valid_mask):
        phenotypes = phenotypes[valid_mask]
        kinship = kinship[np.ix_(valid_mask, valid_mask)]
        if covariates is not None:
            covariates = covariates[valid_mask, :]

    n_samples = phenotypes.shape[0]

    # Memory check using streaming estimation
    if check_memory:
        est = estimate_streaming_memory(n_samples, n_snps, chunk_size=chunk_size)
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory for streaming LMM with {n_samples:,} samples "
                f"x {n_snps:,} SNPs (chunk_size={chunk_size:,}).\n"
                f"Peak: {est.total_peak_gb:.1f}GB, "
                f"Available: {est.available_gb:.1f}GB\n"
                f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
                f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"eigendecomp_workspace={est.eigendecomp_workspace_gb:.1f}GB"
            )

    # GEMMA-style logging
    if show_progress:
        logger.info("## Performing LMM Association Test (Streaming)")
        logger.info(f"number of total individuals = {n_samples_total}")
        logger.info(f"number of analyzed individuals = {n_valid}")
        logger.info(f"number of total SNPs/variants = {n_snps}")
        logger.info(f"lambda range = [{l_min:.2e}, {l_max:.2e}]")

    device = _select_jax_device(use_gpu)

    # === PASS 1: SNP statistics ===
    # Compute per-SNP stats without loading all genotypes at once
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.int32)
    all_vars = np.zeros(n_snps, dtype=np.float64)  # For monomorphic detection

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = progress_iterator(
            stats_iterator, total=n_chunks, desc="Computing SNP statistics"
        )

    for chunk, start, end in stats_iterator:
        # Apply sample filtering
        if not np.all(valid_mask):
            chunk = chunk[valid_mask, :]

        # Compute stats for this chunk
        chunk_miss_counts = np.sum(np.isnan(chunk), axis=0)
        with np.errstate(invalid="ignore"):
            chunk_means = np.nanmean(chunk, axis=0)
            chunk_vars = np.nanvar(chunk, axis=0)  # For monomorphic detection
        chunk_means = np.nan_to_num(chunk_means, nan=0.0)
        chunk_vars = np.nan_to_num(chunk_vars, nan=0.0)

        all_means[start:end] = chunk_means
        all_miss_counts[start:end] = chunk_miss_counts
        all_vars[start:end] = chunk_vars

    # Compute MAF and filter SNPs
    miss_rates = all_miss_counts / n_samples
    allele_freqs = all_means / 2.0
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)

    # Detect monomorphic SNPs (variance == 0 means constant genotype)
    is_polymorphic = all_vars > 0

    # Filter SNPs by MAF, missing rate, and monomorphism
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic
    snp_indices = np.where(snp_mask)[0]
    n_filtered = len(snp_indices)

    if show_progress:
        logger.info(f"number of analyzed SNPs = {n_filtered}")

    if n_filtered == 0:
        if show_progress:
            elapsed = time.perf_counter() - start_time
            logger.info("## LMM Association completed")
            logger.info(f"time elapsed = {elapsed:.2f} seconds")
        return []

    # Precompute filtered stats for result building
    # Use allele_freqs (not mafs) for output to match GEMMA format
    snp_stats = list(
        zip(
            allele_freqs[snp_indices],
            all_miss_counts[snp_indices].astype(int),
            strict=False,
        )
    )
    filtered_means = all_means[snp_indices]

    # === SETUP: Eigendecomposition ===
    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship, eigenvalues, eigenvectors, show_progress, "lmm_streaming"
    )

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Prepare rotated matrices
    UtW = U.T @ W
    Uty = U.T @ phenotypes

    logl_H0, lambda_null_mle, Hi_eval_null_jax = _compute_null_model(
        lmm_mode, eigenvalues_np, UtW, Uty, n_cvt, device, show_progress
    )

    # Device-resident shared arrays - placed on device ONCE before chunk loop
    # These are NOT re-transferred inside the file chunk or JAX chunk loops
    # Only UtG (rotated genotypes) is transferred per-chunk as it differs by SNP
    # Direct device_put - JAX handles numpy conversion efficiently
    eigenvalues = jax.device_put(eigenvalues_np, device)
    UtW_jax = jax.device_put(UtW, device)
    Uty_jax = jax.device_put(Uty, device)

    # Compute chunk size for JAX buffer limits
    jax_chunk_size = _compute_chunk_size(n_samples, n_filtered, n_grid, n_cvt)

    # === PASS 2: Association ===
    # Open writer at start if output_path provided (per-chunk writing)
    writer = None
    if output_path is not None:
        test_type_map = {1: "wald", 2: "lrt", 3: "score", 4: "all"}
        test_type = test_type_map.get(lmm_mode, "wald")
        writer = IncrementalAssocWriter(output_path, test_type=test_type)
        writer.__enter__()

    # Track results for in-memory mode (when output_path is None)
    all_results: list[AssocResult] = []

    try:
        # Map filtered SNP indices to original indices for chunk extraction
        # Group filtered SNPs by which file chunk they belong to
        assoc_iterator = stream_genotype_chunks(
            bed_path, chunk_size=chunk_size, dtype=np.float64, show_progress=False
        )
        if show_progress:
            n_chunks = (n_snps + chunk_size - 1) // chunk_size
            assoc_iterator = progress_iterator(
                assoc_iterator, total=n_chunks, desc="Running LMM association"
            )

        for chunk, file_start, file_end in assoc_iterator:
            # Apply sample filtering
            if not np.all(valid_mask):
                chunk = chunk[valid_mask, :]

            # Find filtered SNPs in this file chunk
            chunk_filtered_indices = []
            chunk_filtered_local_idx = []  # Index within filtered SNPs
            chunk_filtered_col_idx = []  # Column index within this chunk

            # Scan filtered SNPs that fall in this chunk range
            for i, snp_idx in enumerate(snp_indices):
                if file_start <= snp_idx < file_end:
                    chunk_filtered_indices.append(snp_idx)
                    chunk_filtered_local_idx.append(i)
                    chunk_filtered_col_idx.append(snp_idx - file_start)

            if len(chunk_filtered_indices) == 0:
                continue

            # Extract columns for filtered SNPs in this chunk
            chunk_filtered_col_idx_arr = np.array(chunk_filtered_col_idx)
            geno_subset = chunk[:, chunk_filtered_col_idx_arr].copy()

            # Impute missing to mean
            for j, local_idx in enumerate(chunk_filtered_local_idx):
                missing_mask = np.isnan(geno_subset[:, j])
                if np.any(missing_mask):
                    geno_subset[missing_mask, j] = filtered_means[local_idx]

            # Process in JAX chunks if needed (for buffer limit compliance)
            # Double buffering: overlap device transfer with computation
            n_subset = geno_subset.shape[1]
            jax_starts = list(range(0, n_subset, jax_chunk_size))

            def _prepare_jax_chunk(
                start: int, geno: np.ndarray, total: int
            ) -> tuple[np.ndarray, int, bool]:
                """Prepare a JAX chunk for device transfer (CPU work)."""
                end = min(start + jax_chunk_size, total)
                actual_len = end - start

                geno_jax_chunk = geno[:, start:end]

                needs_pad = actual_len < jax_chunk_size
                if needs_pad:
                    pad_width = jax_chunk_size - actual_len
                    geno_jax_chunk = np.pad(
                        geno_jax_chunk, ((0, 0), (0, pad_width)), mode="constant"
                    )

                UtG_chunk = np.ascontiguousarray(U.T @ geno_jax_chunk)
                return UtG_chunk, actual_len, needs_pad

            # Collect results for this FILE chunk (mode-aware accumulators)
            if lmm_mode == 1:  # Wald
                file_chunk_lambdas = []
                file_chunk_logls = []
                file_chunk_betas = []
                file_chunk_ses = []
                file_chunk_pwalds = []
            elif lmm_mode == 3:  # Score
                file_chunk_betas = []
                file_chunk_ses = []
                file_chunk_p_scores = []
            elif lmm_mode == 2:  # LRT
                file_chunk_lambdas_mle = []
                file_chunk_logls_mle = []
                file_chunk_p_lrts = []
            elif lmm_mode == 4:  # All tests
                file_chunk_lambdas = []
                file_chunk_logls = []
                file_chunk_betas = []
                file_chunk_ses = []
                file_chunk_pwalds = []
                file_chunk_lambdas_mle = []
                file_chunk_logls_mle = []
                file_chunk_p_lrts = []
                file_chunk_p_scores = []

            # Prepare first JAX chunk
            UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
                jax_starts[0], geno_subset, n_subset
            )
            UtG_jax = jax.device_put(UtG_np, device)
            del UtG_np

            for i, _jax_start in enumerate(jax_starts):
                current_actual_len = actual_jax_len
                current_needs_padding = needs_padding
                current_UtG = UtG_jax

                # Start async transfer of next JAX chunk while computing current
                if i + 1 < len(jax_starts):
                    UtG_np, actual_jax_len, needs_padding = _prepare_jax_chunk(
                        jax_starts[i + 1], geno_subset, n_subset
                    )
                    UtG_jax = jax.device_put(UtG_np, device)
                    del UtG_np

                # Batch compute Uab (shared across all modes)
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
                    best_lambdas_mle, best_logls_mle = (
                        golden_section_optimize_lambda_mle(
                            n_cvt,
                            eigenvalues,
                            Uab_batch,
                            l_min=l_min,
                            l_max=l_max,
                            n_grid=n_grid,
                            n_iter=max(n_refine, 20),
                        )
                    )
                    p_lrts = jax.vmap(calc_lrt_pvalue_jax)(
                        best_logls_mle, jnp.full_like(best_logls_mle, logl_H0)
                    )

                elif lmm_mode == 4:  # All tests
                    # Score test (cheapest, no optimization)
                    _, _, p_scores = batch_calc_score_stats(
                        n_cvt, Hi_eval_null_jax, Uab_batch, n_samples
                    )

                    # MLE optimization for LRT
                    best_lambdas_mle, best_logls_mle = (
                        golden_section_optimize_lambda_mle(
                            n_cvt,
                            eigenvalues,
                            Uab_batch,
                            l_min=l_min,
                            l_max=l_max,
                            n_grid=n_grid,
                            n_iter=max(n_refine, 20),
                        )
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

                # Strip padding, keep as JAX arrays for file chunk concatenation
                if lmm_mode == 1:
                    slice_len = (
                        current_actual_len
                        if current_needs_padding
                        else len(best_lambdas)
                    )
                    file_chunk_lambdas.append(best_lambdas[:slice_len])
                    file_chunk_logls.append(best_logls[:slice_len])
                    file_chunk_betas.append(betas[:slice_len])
                    file_chunk_ses.append(ses[:slice_len])
                    file_chunk_pwalds.append(p_walds[:slice_len])
                elif lmm_mode == 3:
                    slice_len = (
                        current_actual_len if current_needs_padding else len(betas)
                    )
                    file_chunk_betas.append(betas[:slice_len])
                    file_chunk_ses.append(ses[:slice_len])
                    file_chunk_p_scores.append(p_scores[:slice_len])
                elif lmm_mode == 2:
                    slice_len = (
                        current_actual_len
                        if current_needs_padding
                        else len(best_lambdas_mle)
                    )
                    file_chunk_lambdas_mle.append(best_lambdas_mle[:slice_len])
                    file_chunk_logls_mle.append(best_logls_mle[:slice_len])
                    file_chunk_p_lrts.append(p_lrts[:slice_len])
                elif lmm_mode == 4:
                    slice_len = (
                        current_actual_len
                        if current_needs_padding
                        else len(best_lambdas)
                    )
                    # Wald
                    file_chunk_lambdas.append(best_lambdas[:slice_len])
                    file_chunk_logls.append(best_logls[:slice_len])
                    file_chunk_betas.append(betas[:slice_len])
                    file_chunk_ses.append(ses[:slice_len])
                    file_chunk_pwalds.append(p_walds[:slice_len])
                    # LRT
                    file_chunk_lambdas_mle.append(best_lambdas_mle[:slice_len])
                    file_chunk_logls_mle.append(best_logls_mle[:slice_len])
                    file_chunk_p_lrts.append(p_lrts[:slice_len])
                    # Score
                    file_chunk_p_scores.append(p_scores[:slice_len])

            # After ALL JAX chunks in this file chunk are processed:
            # Concatenate file chunk results and transfer to host
            _has_results = (
                (lmm_mode == 1 and file_chunk_lambdas)
                or (lmm_mode == 3 and file_chunk_betas)
                or (lmm_mode == 2 and file_chunk_lambdas_mle)
                or (lmm_mode == 4 and file_chunk_lambdas)
            )
            if _has_results:
                if lmm_mode == 1:
                    chunk_lambdas_np = np.asarray(jnp.concatenate(file_chunk_lambdas))
                    chunk_logls_np = np.asarray(jnp.concatenate(file_chunk_logls))
                    chunk_betas_np = np.asarray(jnp.concatenate(file_chunk_betas))
                    chunk_ses_np = np.asarray(jnp.concatenate(file_chunk_ses))
                    chunk_pwalds_np = np.asarray(jnp.concatenate(file_chunk_pwalds))
                elif lmm_mode == 3:
                    chunk_betas_np = np.asarray(jnp.concatenate(file_chunk_betas))
                    chunk_ses_np = np.asarray(jnp.concatenate(file_chunk_ses))
                    chunk_p_scores_np = np.asarray(jnp.concatenate(file_chunk_p_scores))
                elif lmm_mode == 2:
                    chunk_lambdas_mle_np = np.asarray(
                        jnp.concatenate(file_chunk_lambdas_mle)
                    )
                    chunk_p_lrts_np = np.asarray(jnp.concatenate(file_chunk_p_lrts))
                elif lmm_mode == 4:
                    chunk_lambdas_np = np.asarray(jnp.concatenate(file_chunk_lambdas))
                    chunk_logls_np = np.asarray(jnp.concatenate(file_chunk_logls))
                    chunk_betas_np = np.asarray(jnp.concatenate(file_chunk_betas))
                    chunk_ses_np = np.asarray(jnp.concatenate(file_chunk_ses))
                    chunk_pwalds_np = np.asarray(jnp.concatenate(file_chunk_pwalds))
                    chunk_lambdas_mle_np = np.asarray(
                        jnp.concatenate(file_chunk_lambdas_mle)
                    )
                    chunk_p_lrts_np = np.asarray(jnp.concatenate(file_chunk_p_lrts))
                    chunk_p_scores_np = np.asarray(jnp.concatenate(file_chunk_p_scores))

                # Build and write/accumulate results for this file chunk
                for j, local_idx in enumerate(chunk_filtered_local_idx):
                    snp_idx = snp_indices[local_idx]
                    af, n_miss = snp_stats[local_idx]
                    info = snp_info[snp_idx]

                    meta = _snp_metadata(info, af, n_miss)
                    if lmm_mode == 1:
                        result = AssocResult(
                            **meta,
                            beta=float(chunk_betas_np[j]),
                            se=float(chunk_ses_np[j]),
                            logl_H1=float(chunk_logls_np[j]),
                            l_remle=float(chunk_lambdas_np[j]),
                            p_wald=float(chunk_pwalds_np[j]),
                        )
                    elif lmm_mode == 3:
                        result = AssocResult(
                            **meta,
                            beta=float(chunk_betas_np[j]),
                            se=float(chunk_ses_np[j]),
                            p_score=float(chunk_p_scores_np[j]),
                        )
                    elif lmm_mode == 2:
                        result = AssocResult(
                            **meta,
                            beta=float("nan"),
                            se=float("nan"),
                            l_mle=float(chunk_lambdas_mle_np[j]),
                            p_lrt=float(chunk_p_lrts_np[j]),
                        )
                    elif lmm_mode == 4:
                        result = AssocResult(
                            **meta,
                            beta=float(chunk_betas_np[j]),
                            se=float(chunk_ses_np[j]),
                            logl_H1=float(chunk_logls_np[j]),
                            l_remle=float(chunk_lambdas_np[j]),
                            l_mle=float(chunk_lambdas_mle_np[j]),
                            p_wald=float(chunk_pwalds_np[j]),
                            p_lrt=float(chunk_p_lrts_np[j]),
                            p_score=float(chunk_p_scores_np[j]),
                        )

                    if writer is not None:
                        writer.write(result)
                    else:
                        all_results.append(result)

                # Clear file chunk arrays to free memory
                if lmm_mode == 1:
                    del chunk_lambdas_np, chunk_logls_np, chunk_betas_np
                    del chunk_ses_np, chunk_pwalds_np
                    del file_chunk_lambdas, file_chunk_logls, file_chunk_betas
                    del file_chunk_ses, file_chunk_pwalds
                elif lmm_mode == 3:
                    del chunk_betas_np, chunk_ses_np, chunk_p_scores_np
                    del file_chunk_betas, file_chunk_ses, file_chunk_p_scores
                elif lmm_mode == 2:
                    del chunk_lambdas_mle_np, chunk_p_lrts_np
                    del file_chunk_lambdas_mle, file_chunk_logls_mle
                    del file_chunk_p_lrts
                elif lmm_mode == 4:
                    del chunk_lambdas_np, chunk_logls_np, chunk_betas_np
                    del chunk_ses_np, chunk_pwalds_np
                    del chunk_lambdas_mle_np, chunk_p_lrts_np, chunk_p_scores_np
                    del file_chunk_lambdas, file_chunk_logls, file_chunk_betas
                    del file_chunk_ses, file_chunk_pwalds
                    del file_chunk_lambdas_mle, file_chunk_logls_mle
                    del file_chunk_p_lrts, file_chunk_p_scores

        # Log memory after association pass completes
        if show_progress:
            log_rss_memory("lmm_streaming", "after_association")

        # Explicit cleanup of JAX device-resident arrays before returning
        # to prevent SIGSEGV from race conditions between Python GC and JAX threads
        del eigenvalues, UtW_jax, Uty_jax

    finally:
        if writer is not None:
            writer.__exit__(None, None, None)
            if show_progress:
                logger.info(f"Wrote {writer.count:,} results to {output_path}")

    # Force synchronization - ensures all JAX operations complete before returning
    # This prevents SIGSEGV from background threads accessing freed memory
    jax.clear_caches()

    # GEMMA-style completion logging
    if show_progress:
        elapsed = time.perf_counter() - start_time
        logger.info("## LMM Association completed")
        logger.info(f"time elapsed = {elapsed:.2f} seconds")

    # When output_path is set, results are on disk; return empty list
    return [] if output_path is not None else all_results
