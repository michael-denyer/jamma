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
from jamma.lmm.compute import _compute_lmm_chunk, block_chunk_result
from jamma.lmm.likelihood_jax import batch_compute_uab
from jamma.lmm.prepare import (
    _build_covariate_matrix,
    _compute_null_model,
    _eigendecompose_or_reuse,
    _select_jax_device,
    _setup_cpu_sharding,
)
from jamma.lmm.results import (
    _RESULT_FIELDS,
    _build_results,
    count_lambda_boundary_hits,
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
        n_refine: Golden section iterations (clamped to min 20
            internally for ~1e-5 tolerance).
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
    snp_spec, rep_spec = _setup_cpu_sharding()
    n_devices = len(jax.devices("cpu"))

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
        "lmm_jax",
        check_memory=check_memory,
    )
    # Free kinship ref and LAPACK workspace before LMM phase
    del kinship
    gc.collect()

    with blas_threads():
        UtW = U.T @ W
        Uty = U.T @ phenotypes

    # Determine chunk size to avoid int32 buffer overflow (before null model to
    # use n_devices-aligned chunk_size for placement decisions)
    n_filtered = len(snp_indices)
    chunk_size = _compute_chunk_size(n_samples, n_filtered, n_grid, n_cvt, n_devices)

    # Use sharding whenever multiple devices are available. Chunks that aren't
    # evenly divisible by n_devices get zero-padded before device_put (the
    # padded SNP results are discarded via actual_len slicing).
    effective_snp_spec = snp_spec
    effective_rep_spec = rep_spec

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
        rep_spec=effective_rep_spec,
    )
    t_eigen_end = time.perf_counter()

    # Device-resident shared arrays - placed on device ONCE before chunk loop
    if effective_rep_spec is not None:
        eigenvalues = jax.device_put(eigenvalues_np, effective_rep_spec)
        UtW_jax = jax.device_put(UtW, effective_rep_spec)
        Uty_jax = jax.device_put(Uty, effective_rep_spec)
    else:
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

    # Pre-allocate result arrays driven by _RESULT_FIELDS mapping
    write_offset = 0
    arrays_out: dict[str, np.ndarray] = {
        key: np.empty(n_filtered, dtype=np.float64) for key in _RESULT_FIELDS[lmm_mode]
    }

    # Timing accumulators for per-chunk phases
    t_rotation_total = 0.0
    t_jax_compute_total = 0.0
    t_result_write_total = 0.0

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

        # Pad to device-count multiple for even NamedSharding distribution
        if effective_snp_spec is not None and UtG_chunk.shape[1] % n_devices != 0:
            dev_pad = n_devices - (UtG_chunk.shape[1] % n_devices)
            UtG_chunk = np.pad(UtG_chunk, ((0, 0), (0, dev_pad)), mode="constant")

        return UtG_chunk, actual_len, needs_pad

    # Double buffering: overlap device transfer with computation
    chunk_starts = list(range(0, n_filtered, chunk_size))

    # Prepare first chunk (includes BLAS rotation U.T @ G)
    t_rot_start = time.perf_counter()
    UtG_np, actual_len, needs_pad = _prepare_chunk(chunk_starts[0])
    t_rot_end = time.perf_counter()
    t_rotation_total += t_rot_end - t_rot_start
    if effective_snp_spec is not None:
        UtG_jax = jax.device_put(UtG_np, effective_snp_spec)
    else:
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
        current_UtG = UtG_jax

        # Start async transfer of next chunk while computing current
        if i + 1 < len(chunk_starts):
            t_rot_start = time.perf_counter()
            next_UtG_np, actual_len, needs_pad = _prepare_chunk(chunk_starts[i + 1])
            t_rot_end = time.perf_counter()
            t_rotation_total += t_rot_end - t_rot_start
            # device_put is async - transfer starts immediately, overlaps with compute
            if effective_snp_spec is not None:
                UtG_jax = jax.device_put(next_UtG_np, effective_snp_spec)
            else:
                UtG_jax = jax.device_put(next_UtG_np, device)
            del next_UtG_np

        # --- JAX compute timing ---
        t_jax_start = time.perf_counter()
        try:
            # Batch compute Uab for this chunk (shared across all modes)
            Uab_batch = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, current_UtG)

            cr = _compute_lmm_chunk(
                lmm_mode,
                n_cvt,
                eigenvalues,
                Uab_batch,
                n_samples,
                l_min=l_min,
                l_max=l_max,
                n_grid=n_grid,
                n_refine=n_refine,
                Hi_eval_null=Hi_eval_null_jax,
                logl_H0=logl_H0,
            )
            # Explicit sync before timing result write (np.asarray below also syncs,
            # but this isolates JAX compute time accurately)
            block_chunk_result(cr, lmm_mode)

        except Exception as e:
            error_msg = str(e)
            # JAX int32 overflow (observed in JAX 0.4.x). String-based
            # detection is fragile — update if JAX changes the wording.
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

        t_jax_end = time.perf_counter()
        t_jax_compute_total += t_jax_end - t_jax_start

        # Write results into pre-allocated arrays by index (no list append)
        t_write_start = time.perf_counter()
        # Always slice to actual_chunk_len — strips both tail-chunk padding
        # and device-alignment padding from NamedSharding.
        slice_len = actual_chunk_len
        s = slice(write_offset, write_offset + slice_len)

        for key in arrays_out:
            arrays_out[key][s] = np.asarray(cr[key][:slice_len])

        write_offset += slice_len
        t_write_end = time.perf_counter()
        t_result_write_total += t_write_end - t_write_start

    # Validate all results were written
    if write_offset != n_filtered:
        raise RuntimeError(
            f"Pre-allocated array size mismatch: wrote {write_offset} results,"
            f" expected {n_filtered}. This is an internal error — please report"
            f" this issue with your dataset dimensions."
        )

    # Log memory after all chunks processed
    if show_progress:
        log_rss_memory("lmm_jax", "after_all_chunks")

    # Lambda boundary convergence diagnostics
    n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
        lmm_mode, arrays_out, l_min, l_max
    )
    log_lambda_boundary_warning(n_at_lmin, n_at_lmax, l_min, l_max)

    # Explicit cleanup of JAX arrays before returning to prevent SIGSEGV
    # from race conditions between Python GC and JAX background threads
    del eigenvalues, UtW_jax, Uty_jax
    jax.clear_caches()

    # Log completion
    elapsed = time.perf_counter() - start_time
    if show_progress:
        t_eigen = t_eigen_end - t_eigen_start
        accounted = (
            t_eigen + t_rotation_total + t_jax_compute_total + t_result_write_total
        )
        logger.info("Timing breakdown:")
        logger.info(f"  Eigendecomp+setup:   {t_eigen:.2f}s")
        logger.info(f"  UT@G rotation:       {t_rotation_total:.2f}s")
        logger.info(f"  JAX compute:         {t_jax_compute_total:.2f}s")
        logger.info(f"  Result write:        {t_result_write_total:.2f}s")
        logger.info("  ----")
        logger.info(f"  Accounted:           {accounted:.2f}s")
        logger.info(f"  Total:               {elapsed:.2f}s")
        logger.info(f"LMM Association completed in {elapsed:.2f}s")

    return _build_results(
        lmm_mode, snp_indices, filtered_afs, filtered_miss, snp_info, arrays_out
    )
