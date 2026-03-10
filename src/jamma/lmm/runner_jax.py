"""JAX-optimized batch LMM association runner.

Batch-optimized LMM association testing on CPU (XLA) or GPU (JAX).
Input genotypes must fit in memory; for disk streaming use runner_streaming.py.
"""

import gc
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

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
from jamma.lmm.prepare_common import (
    compute_and_log_pve,
    validate_runner_inputs,
)
from jamma.lmm.results import (
    _build_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
    write_streaming_chunk,
)
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.lmm.schema import LmmConfig, LmmRunResult, RunnerTiming
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
    output_path: Path | None = None,
    clear_caches: bool = True,
) -> LmmRunResult:
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
        config: LmmConfig instance. When provided, overrides individual
            threshold/mode kwargs above.
        output_path: Path for per-chunk disk streaming. When set, results
            are written incrementally and the returned LmmRunResult has
            empty associations and n_tested populated instead.
        clear_caches: Clear JAX compilation caches on exit. Set to False
            when calling repeatedly with identical shapes (e.g. multi-phenotype
            loops) to avoid redundant JIT recompilation.

    Returns:
        LmmRunResult with per-SNP associations and PVE from null model.
            When output_path is set, associations is empty (results on
            disk) and n_tested contains the count of SNPs written.

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
        # JAX's XLA allocator creates transient intermediates during matmul
        # that aren't captured in the static estimate. Apply a 1.25x safety
        # factor and check against available memory minus current process
        # usage (eigenvectors, eigenvalues, etc. already resident).
        import psutil

        rss_gb = psutil.Process().memory_info().rss / 1e9
        jax_safety_factor = 1.25
        safe_estimate = est.total_gb * jax_safety_factor
        sufficient = (safe_estimate + rss_gb) < est.available_gb
        headroom = est.available_gb - safe_estimate - rss_gb
        logger.info(
            f"LMM memory: estimated {est.total_gb:.1f}GB "
            f"(x{jax_safety_factor} safety = {safe_estimate:.0f}GB), "
            f"process using {rss_gb:.1f}GB, "
            f"{est.available_gb:.1f}GB available"
        )
        if not sufficient:
            raise MemoryError(
                f"Insufficient memory for JAX batch LMM with "
                f"{n_samples:,} samples × {n_snps:,} SNPs.\n"
                f"Need: ~{safe_estimate:.0f}GB "
                f"({est.total_gb:.1f}GB x{jax_safety_factor} JAX safety), "
                f"process using {rss_gb:.1f}GB, "
                f"{est.available_gb:.1f}GB available "
                f"({headroom:.0f}GB shortfall)\n"
                f"Breakdown: eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"genotypes={est.genotypes_gb:.1f}GB, "
                f"batch buffers={est.lmm_batch_gb:.1f}GB\n"
                f"Consider: JAMMA_BACKEND=numpy for lower memory usage"
            )

    # Enforce minimum 20 golden section iterations for ~1e-5 lambda tolerance
    n_refine = max(n_refine, 20)

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
        if output_path is not None:
            from jamma.lmm.io import IncrementalAssocWriter

            with IncrementalAssocWriter(
                output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
            ):
                pass  # Header-only file, matching streaming runner behavior
        return LmmRunResult(associations=[], n_tested=0)

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

    eigenvalues_jax = None
    UtW_jax = None
    Uty_jax = None
    try:
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

        pve, pve_se = compute_and_log_pve(eigenvalues_np, UtW, Uty, n_cvt, l_min, l_max)

        eigenvalues_jax = jax.device_put(eigenvalues_np, placement.rep)
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

        # Streaming mode: write per-chunk to disk, skip arrays_out allocation.
        # Non-streaming: accumulate into arrays_out, build AssocResult list at end.
        streaming = output_path is not None
        if streaming:
            from jamma.lmm.io import IncrementalAssocWriter

            writer_ctx = IncrementalAssocWriter(
                output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
            )
            arrays_out = None
        else:
            writer_ctx = None
            arrays_out = {
                key: np.empty(n_filtered, dtype=np.float64)
                for key in _RESULT_FIELDS[lmm_mode]
            }

        write_offset = 0

        # Invalidate stale timing immediately so callers never see prior-run data
        # if this run raises mid-execution.
        last_run_timing.clear()

        # Timing accumulators for per-chunk phases
        t_rotation_total = 0.0
        t_rotation_exposed_total = 0.0
        t_jax_compute_total = 0.0
        t_result_write_total = 0.0

        # Per-chunk diagnostic accumulators (used in streaming mode where
        # arrays_out is not available for post-loop inspection).
        nan_counts: dict[str, int] = {}
        n_at_lmin_accum = 0
        n_at_lmax_accum = 0

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
            """Measure background-thread preparation time inside the worker."""
            t0 = time.perf_counter()
            UtG_np, actual_len = _impute_and_prepare(start)
            return UtG_np, actual_len, time.perf_counter() - t0

        chunk_starts = list(range(0, n_filtered, chunk_size))

        # prev_compute_end tracks the perf_counter timestamp of the last JAX
        # compute sync, used to compute how much rotation time was exposed.
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

        writer_cm = writer_ctx if streaming else nullcontext()

        # Rotation-compute pipeline: while JAX processes chunk N on XLA, a
        # background thread runs BLAS DGEMM (U.T @ G) for chunk N+1.
        with writer_cm as writer, ThreadPoolExecutor(max_workers=1) as executor:
            for i, _chunk_start in chunk_iterator:
                actual_chunk_len = actual_len
                current_UtG = UtG_jax

                future = None
                if i + 1 < len(chunk_starts):
                    future = executor.submit(
                        _impute_and_prepare_timed, chunk_starts[i + 1]
                    )

                t_jax_start = time.perf_counter()
                try:
                    Uab_batch = batch_compute_uab(n_cvt, UtW_jax, Uty_jax, current_UtG)

                    cr = _compute_lmm_chunk(
                        lmm_mode,
                        n_cvt,
                        eigenvalues_jax,
                        Uab_batch,
                        n_samples,
                        l_min=l_min,
                        l_max=l_max,
                        n_grid=n_grid,
                        n_refine=n_refine,
                        Hi_eval_null=Hi_eval_null_jax,
                        logl_H0=logl_H0,
                    )
                    # Explicit sync before timing result write
                    block_chunk_result(cr, lmm_mode)

                except Exception as e:
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

                if future is not None:
                    try:
                        UtG_np, actual_len, rot_dur = future.result()
                    except MemoryError:
                        raise
                    except Exception as exc:
                        raise RuntimeError(
                            f"Background rotation failed for chunk starting at "
                            f"index {chunk_starts[i + 1]}. "
                            f"Processed {write_offset + actual_chunk_len} SNPs "
                            f"before failure."
                        ) from exc
                    t_rot_end = time.perf_counter()
                    t_rotation_total += rot_dur
                    t_rotation_exposed_total += min(
                        rot_dur, max(0.0, t_rot_end - t_jax_end)
                    )
                    UtG_jax = jax.device_put(UtG_np, placement.snp)
                    del UtG_np  # Safe: JAX holds internal ref during async transfer

                # Write results, stripping padding from tail/device-alignment
                t_write_start = time.perf_counter()
                chunk_arrays = {
                    key: np.asarray(cr[key][:actual_chunk_len])
                    for key in _RESULT_FIELDS[lmm_mode]
                }
                if streaming:
                    n_at_lmin_accum, n_at_lmax_accum = write_streaming_chunk(
                        writer,
                        lmm_mode,
                        snp_indices[write_offset : write_offset + actual_chunk_len],
                        snp_info,
                        filtered_afs[write_offset : write_offset + actual_chunk_len],
                        filtered_miss[write_offset : write_offset + actual_chunk_len],
                        chunk_arrays,
                        l_min,
                        l_max,
                        nan_counts,
                        n_at_lmin_accum,
                        n_at_lmax_accum,
                    )
                else:
                    s = slice(write_offset, write_offset + actual_chunk_len)
                    for key in arrays_out:
                        arrays_out[key][s] = chunk_arrays[key]
                write_offset += actual_chunk_len
                t_write_end = time.perf_counter()
                t_result_write_total += t_write_end - t_write_start

        if write_offset != n_filtered:
            raise RuntimeError(
                f"Pre-allocated array size mismatch: wrote {write_offset} results,"
                f" expected {n_filtered}. This is an internal error — please report"
                f" this issue with your dataset dimensions."
            )

        if show_progress:
            log_rss_memory("lmm_jax", "after_all_chunks")

        # Diagnostics: use accumulated per-chunk counts for streaming,
        # post-loop arrays_out inspection for non-streaming.
        if streaming:
            for key, n_nan in nan_counts.items():
                logger.warning(
                    f"{n_nan}/{n_filtered} SNPs have NaN {key} — "
                    "check for degenerate (constant) genotypes "
                    "and kinship matrix quality"
                )
            log_lambda_boundary_warning(n_at_lmin_accum, n_at_lmax_accum, l_min, l_max)
        else:
            for key, arr in arrays_out.items():
                n_nan = int(np.sum(np.isnan(arr)))
                if n_nan > 0:
                    logger.warning(
                        f"{n_nan}/{n_filtered} SNPs have NaN {key} — "
                        "check for degenerate (constant) genotypes "
                        "and kinship matrix quality"
                    )
            n_at_lmin, n_at_lmax = count_lambda_boundary_hits(
                lmm_mode, arrays_out, l_min, l_max
            )
            log_lambda_boundary_warning(n_at_lmin, n_at_lmax, l_min, l_max)

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

        if streaming:
            return LmmRunResult(
                associations=[],
                pve=pve,
                pve_se=pve_se,
                n_tested=write_offset,
            )

        return LmmRunResult(
            associations=_build_results(
                lmm_mode,
                snp_indices,
                filtered_afs,
                filtered_miss,
                snp_info,
                arrays_out,
            ),
            pve=pve,
            pve_se=pve_se,
        )
    finally:
        del eigenvalues_jax, UtW_jax, Uty_jax
        if clear_caches:
            try:
                jax.clear_caches()
            except Exception:
                logger.warning(
                    "Failed to clear JAX caches during cleanup", exc_info=True
                )
