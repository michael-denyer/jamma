"""JAX-optimized batch LMM association runner.

Batch-optimized LMM association testing on CPU (XLA) or GPU (JAX).
Input genotypes must fit in memory; for disk streaming use runner_streaming.py.
"""

import gc
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import numpy as np
from loguru import logger

from jamma.core.memory import estimate_lmm_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.core.threading import blas_threads, get_physical_core_count
from jamma.lmm.chunk import _compute_chunk_size
from jamma.lmm.compute import (
    _compute_lmm_chunk,
    block_chunk_result,
    exposed_rotation_time,
    log_jax_error,
)
from jamma.lmm.likelihood_jax import batch_compute_uab
from jamma.lmm.prepare import (
    _build_covariate_matrix,
    _compute_null_model,
    _eigendecompose_or_reuse,
    prepare_utg_chunk,
    resolve_device_placement,
)
from jamma.lmm.prepare_common import validate_runner_inputs
from jamma.lmm.results import (
    _build_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import LmmConfig, RunnerTiming
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory

# Module-level timing from the last run, for direct callers (tests, notebooks).
# The pipeline reads runner_streaming.last_run_timing for the JAX streaming path.
# Not thread-safe: concurrent calls will corrupt this dict.
# Cleared at function entry; repopulated at function exit on success.
last_run_timing: RunnerTiming = {}


def run_lmm_association_jax(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
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
    config: LmmConfig | None = None,
) -> list[AssocResult]:
    """Run LMM association tests using JAX-optimized batch processing.

    Processes all SNPs in parallel via JAX vectorization and JIT compilation.
    Ensures JAX is configured for 64-bit precision (required for GEMMA equivalence).
    SNPs are processed in memory-budget-sized chunks. Input genotypes must fit
    in memory; for disk streaming use run_lmm_association_streaming.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2.
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples) or None when
            pre-computed eigenvalues/eigenvectors are provided. WARNING: may
            be overwritten in-place during eigendecomposition (buffer reused
            for eigenvectors). Treat as consumed; pass kinship.copy() if you
            need the original matrix after this call.
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
    # Unpack config if provided (config takes precedence over individual kwargs)
    if config is not None:
        kw = config.as_kwargs()
        maf_threshold = kw["maf_threshold"]
        miss_threshold = kw["miss_threshold"]
        l_min, l_max = kw["l_min"], kw["l_max"]
        n_grid, n_refine = kw["n_grid"], kw["n_refine"]
        use_gpu, check_memory = kw["use_gpu"], kw["check_memory"]
        show_progress, lmm_mode = kw["show_progress"], kw["lmm_mode"]

    from jamma.core.jax_config import ensure_jax_configured

    ensure_jax_configured()

    # Memory check before workflow (uses genotype shape, runner-specific)
    n_samples, n_snps = genotypes.shape
    start_time = time.perf_counter()

    if show_progress:
        logger.info("Performing LMM Association Test (JAX batch)")
        logger.info(f"  Total individuals: {n_samples:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.debug(
            f"MAF threshold = {maf_threshold}, missing threshold = {miss_threshold}"
        )

    if check_memory:
        actual_chunk = _compute_chunk_size(n_snps, n_samples=n_samples)
        est = estimate_lmm_memory(n_samples, n_snps, lmm_batch_size=actual_chunk)
        # available reports free system RAM, but this process is already
        # using memory (eigenvectors, etc.). Subtract current process
        # usage to get realistic headroom for new allocations.
        import psutil

        rss_gb = psutil.Process().memory_info().rss / 1e9
        effective_available = est.available_gb
        logger.info(
            f"LMM memory: estimated {est.total_gb:.1f}GB, "
            f"available {effective_available:.1f}GB "
            f"(process using {rss_gb:.1f}GB)"
        )
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory for LMM with {n_samples:,} samples × "
                f"{n_snps:,} SNPs.\n"
                f"Need: {est.total_gb:.1f}GB, "
                f"Available: {effective_available:.1f}GB "
                f"(process using {rss_gb:.1f}GB)\n"
                f"Breakdown: eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"genotypes={est.genotypes_gb:.1f}GB, "
                f"batch buffers={est.lmm_batch_gb:.1f}GB\n"
                f"Consider: JAMMA_BACKEND=numpy for lower memory usage"
            )

    placement = resolve_device_placement(use_gpu)

    # Validate inputs and apply sample filtering (shared logic for all runners)
    setup = validate_runner_inputs(
        phenotypes, kinship, covariates, eigenvalues, eigenvectors, lmm_mode
    )
    phenotypes = setup.phenotypes
    kinship = setup.kinship
    covariates = setup.covariates
    eigenvalues = setup.eigenvalues
    eigenvectors = setup.eigenvectors
    n_samples = setup.n_samples

    # Apply the same valid-mask to genotypes (runner-specific: genotypes in memory)
    if not np.all(setup.valid_mask):
        genotypes = genotypes[setup.valid_mask, :]

    n_samples, n_snps = genotypes.shape

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Vectorized SNP stats and filtering using shared functions
    col_means, missing_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        col_means, missing_counts, col_vars, n_samples, maf_threshold, miss_threshold
    )
    snp_indices = np.where(snp_mask)[0]

    if len(snp_indices) == 0:
        logger.warning(
            f"All {n_snps} SNPs filtered out (MAF>{maf_threshold}, "
            f"miss<{miss_threshold}). No association tests to run. "
            f"Consider relaxing --maf or --miss thresholds."
        )
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
    del kinship
    gc.collect()

    # Use all physical cores for BLAS rotation (no JAX contention)
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = U.T @ W
        Uty = U.T @ phenotypes

    n_filtered = len(snp_indices)
    chunk_size = _compute_chunk_size(
        n_filtered, placement.n_devices, n_samples=n_samples, pipeline_buffers=2
    )

    logl_H0, lambda_null_mle, Hi_eval_null_jax = _compute_null_model(
        lmm_mode,
        eigenvalues_np,
        UtW,
        Uty,
        n_cvt,
        placement.rep,
        show_progress,
        l_min=l_min,
        l_max=l_max,
    )
    t_eigen_end = time.perf_counter()

    eigenvalues = jax.device_put(eigenvalues_np, placement.rep)
    UtW_jax = jax.device_put(UtW, placement.rep)
    Uty_jax = jax.device_put(Uty, placement.rep)

    # Process in chunks if needed
    n_chunks = (n_filtered + chunk_size - 1) // chunk_size
    if show_progress:
        from jamma.core.estimates import estimate_lmm_time

        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"  Processing in {n_chunks} chunks ({chunk_size:,} SNPs/chunk)"
            )
        est = estimate_lmm_time(n_samples, n_filtered, rotation_threads)
        logger.info(f"  Estimated time: {est}")

    # Pre-allocate result arrays driven by _RESULT_FIELDS mapping
    write_offset = 0
    arrays_out: dict[str, np.ndarray] = {
        key: np.empty(n_filtered, dtype=np.float64) for key in _RESULT_FIELDS[lmm_mode]
    }

    # Invalidate stale timing immediately so callers never see prior-run data
    # if this run raises mid-execution.
    last_run_timing.clear()

    # Timing accumulators for per-chunk phases
    t_rotation_total = 0.0
    t_rotation_exposed_total = 0.0
    t_jax_compute_total = 0.0
    t_result_write_total = 0.0

    def _impute_and_prepare(start: int) -> tuple[np.ndarray, int]:
        """Mean-impute a genotype slice and prepare UtG for device transfer."""
        chunk_indices = snp_indices[start : start + chunk_size]
        geno_chunk = genotypes[:, chunk_indices]
        chunk_means_local = col_means[chunk_indices]
        missing = np.isnan(geno_chunk)
        if missing.any():  # RUN-06: skip O(n*chunk) np.where on clean data
            geno_chunk = np.where(missing, chunk_means_local[None, :], geno_chunk)
        del missing
        return prepare_utg_chunk(geno_chunk, U, placement, rotation_threads)

    def _impute_and_prepare_timed(start: int) -> tuple[np.ndarray, int, float]:
        """_impute_and_prepare with internal duration measurement for background thread.

        Duration is measured inside because the caller on the main thread cannot
        observe start/end timestamps of work running on the background thread.
        """
        t0 = time.perf_counter()
        UtG_np, actual_len = _impute_and_prepare(start)
        return UtG_np, actual_len, time.perf_counter() - t0

    chunk_starts = list(range(0, n_filtered, chunk_size))

    # prev_compute_end tracks the perf_counter timestamp of the last JAX compute
    # sync, used to compute how much rotation time was exposed (not overlapped).
    prev_compute_end: float | None = None

    # Prepare first chunk (includes BLAS rotation U.T @ G)
    t_rot_start = time.perf_counter()
    UtG_np, actual_len = _impute_and_prepare(chunk_starts[0])
    t_rot_end = time.perf_counter()
    rot_dur = t_rot_end - t_rot_start
    t_rotation_total += rot_dur
    t_rotation_exposed_total += exposed_rotation_time(
        rot_dur, t_rot_end, prev_compute_end
    )
    UtG_jax = jax.device_put(UtG_np, placement.snp)
    del UtG_np  # Safe: JAX holds internal ref during async transfer

    # Create progress bar iterator
    if show_progress and n_chunks > 1:
        chunk_iterator = progress_iterator(
            enumerate(chunk_starts), total=n_chunks, desc="LMM association"
        )
    else:
        chunk_iterator = enumerate(chunk_starts)

    # Rotation-compute pipeline: while JAX processes chunk N on XLA, a
    # background thread runs BLAS DGEMM (U.T @ G) for chunk N+1. DGEMM
    # releases the GIL, so both run truly concurrently. max_workers=1
    # ensures at most one prefetch is in flight (double-buffering, not
    # unbounded prefetch). Memory budget is halved via pipeline_buffers=2
    # to account for two live UtG arrays.
    with ThreadPoolExecutor(max_workers=1) as executor:
        for i, _chunk_start in chunk_iterator:
            actual_chunk_len = actual_len
            current_UtG = UtG_jax

            # Submit next rotation to BACKGROUND THREAD.
            # BLAS DGEMM (U.T @ G) releases the GIL and runs concurrently
            # with JAX compute dispatched below.
            future = None
            if i + 1 < len(chunk_starts):
                future = executor.submit(_impute_and_prepare_timed, chunk_starts[i + 1])

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
                # Explicit sync before timing result write (np.asarray below also
                # syncs, but this isolates JAX compute time accurately)
                block_chunk_result(cr, lmm_mode)

            except Exception as e:
                # Best-effort cancel: only succeeds if rotation hasn't started.
                # If already running, executor.__exit__ will wait for completion.
                if future is not None:
                    future.cancel()
                log_jax_error(
                    e,
                    chunk_label=f"{i + 1}/{n_chunks}",
                    chunk_snps=chunk_size,
                    n_samples=n_samples,
                    n_cvt=n_cvt,
                )
                raise

            t_jax_end = time.perf_counter()
            t_jax_compute_total += t_jax_end - t_jax_start
            prev_compute_end = t_jax_end

            # Collect background rotation result (may be ready by now).
            # future.result() blocks only for the remaining time after JAX sync.
            if future is not None:
                try:
                    UtG_np, actual_len, rot_dur = future.result()
                except MemoryError:
                    raise  # Let MemoryError propagate directly for OOM handling
                except Exception as exc:
                    raise RuntimeError(
                        f"Background rotation failed for chunk starting at "
                        f"index {chunk_starts[i + 1]}. "
                        f"Processed {write_offset + actual_chunk_len} SNPs "
                        f"before failure."
                    ) from exc
                t_rot_end = time.perf_counter()
                t_rotation_total += rot_dur
                # Exposed = time main thread waited for future AFTER JAX sync.
                # Near zero when JAX compute takes longer than rotation.
                # Capped at rot_dur to prevent GC/scheduling jitter inflation.
                t_rotation_exposed_total += min(
                    rot_dur, max(0.0, t_rot_end - t_jax_end)
                )
                UtG_jax = jax.device_put(UtG_np, placement.snp)
                del UtG_np  # Safe: JAX holds internal ref during async transfer

            # Write results, stripping padding from tail/device-alignment
            t_write_start = time.perf_counter()
            s = slice(write_offset, write_offset + actual_chunk_len)
            for key in arrays_out:
                arrays_out[key][s] = np.asarray(cr[key][:actual_chunk_len])
            write_offset += actual_chunk_len
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

    # NaN diagnostic: warn if any output arrays contain NaN results
    for key, arr in arrays_out.items():
        n_nan = int(np.sum(np.isnan(arr)))
        if n_nan > 0:
            logger.warning(
                f"{n_nan}/{n_filtered} SNPs have NaN {key} — "
                "check kinship matrix quality"
            )

    # Lambda boundary convergence diagnostics
    n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
        lmm_mode, arrays_out, l_min, l_max
    )
    log_lambda_boundary_warning(n_at_lmin, n_at_lmax, l_min, l_max)

    # Explicit cleanup prevents SIGSEGV from GC/JAX thread race conditions
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
        logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
        logger.info(f"  UT@G rotation:       {t_rotation_total:.2f}s")
        logger.info(f"  UT@G exposed:        {t_rotation_exposed_total:.2f}s")
        logger.info(f"  JAX compute:         {t_jax_compute_total:.2f}s")
        logger.info(f"  Result write:        {t_result_write_total:.2f}s")
        logger.info("  ----")
        logger.info(f"  Accounted:           {accounted:.2f}s")
        logger.info(f"  Total:               {elapsed:.2f}s")
        logger.info(f"LMM Association completed in {elapsed:.2f}s")

    last_run_timing.clear()
    last_run_timing.update(
        {
            "rotation_s": t_rotation_total,
            "rotation_exposed_s": t_rotation_exposed_total,
            "jax_compute_s": t_jax_compute_total,
            "result_write_s": t_result_write_total,
        }
    )

    return _build_results(
        lmm_mode, snp_indices, filtered_afs, filtered_miss, snp_info, arrays_out
    )
