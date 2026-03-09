"""Streaming LMM association runner.

Two-pass disk streaming: (1) SNP statistics, (2) association per chunk.
Never allocates the full genotype matrix.
"""

import contextlib
import gc
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import numpy as np
from loguru import logger

from jamma.core.memory import estimate_lmm_streaming_memory
from jamma.core.progress import progress_iterator
from jamma.core.snp_filter import compute_snp_filter_mask
from jamma.core.threading import blas_threads, get_physical_core_count
from jamma.io.plink import (
    get_plink_metadata,
    prefetch_iterator,
    stream_genotype_chunks,
    validate_genotype_values,
)
from jamma.lmm.chunk import _compute_chunk_size, compute_subchunk_starts
from jamma.lmm.compute import (
    _compute_lmm_chunk,
    block_chunk_result,
    exposed_rotation_time,
    log_jax_error,
)
from jamma.lmm.io import IncrementalAssocWriter
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
    _chunk_result_to_numpy,
    _yield_chunk_results,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.schema import ACCUM_KEYS as _ACCUM_KEYS
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.lmm.schema import LazySnpMeta as _LazySnpMeta
from jamma.lmm.schema import LmmConfig, LmmRunResult, RunnerTiming
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory

# Module-level timing from the last run, for programmatic access by pipeline/benchmarks.
# Not thread-safe: concurrent calls will corrupt this dict.
# Cleared at function entry; repopulated at function exit on success.
# Use get_last_run_timing() for a safe snapshot copy.
last_run_timing: RunnerTiming = {}


def get_last_run_timing() -> RunnerTiming:
    """Return a snapshot copy of the last run's timing data.

    Safe to call from any thread — returns an independent dict.
    """
    return dict(last_run_timing)


def run_lmm_association_streaming(
    bed_path: Path,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None = None,
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
    snps_indices: np.ndarray | None = None,
    hwe_threshold: float = 0.0,
    validate_genotypes: bool = True,
    config: LmmConfig | None = None,
    clear_caches: bool = True,
) -> tuple[LmmRunResult, int]:
    """Run LMM association tests by streaming genotypes from disk.

    Reads genotypes per-chunk, never allocating the full genotype matrix.
    Two-pass: (1) SNP statistics for filtering, (2) association per chunk.

    Args:
        bed_path: PLINK file prefix (without .bed/.bim/.fam extension).
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples), or None when
            pre-computed eigenvalues and eigenvectors are provided. WARNING:
            may be overwritten in-place during eigendecomposition (buffer
            reused for eigenvectors). Treat as consumed; pass kinship.copy()
            if you need the original matrix.
        snp_info: List of SNP metadata dicts, or None to build from PLINK.
        covariates: Covariate matrix (n_samples, n_cvt) or None for intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        maf_threshold: Minimum MAF for SNP inclusion.
        miss_threshold: Maximum missing rate for SNP inclusion.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations for lambda refinement.
        chunk_size: Number of SNPs per disk chunk (default: 10,000).
        use_gpu: Whether to use GPU acceleration.
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        output_path: Path for incremental result writing, or None for in-memory.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        snps_indices: Pre-resolved column indices for -snps restriction, or None.
        hwe_threshold: HWE p-value threshold; SNPs with p < threshold are
            removed. 0.0 disables HWE filtering (default).
        validate_genotypes: Check for unexpected genotype values during pass-1
            (default True).
        config: LmmConfig instance. When provided, overrides individual
            threshold/mode kwargs above.
        clear_caches: Clear JAX compilation caches on exit. Set to False
            when calling repeatedly with identical shapes (e.g. multi-phenotype
            loops) to avoid redundant JIT recompilation.

    Returns:
        Tuple of (LmmRunResult, n_tested) where LmmRunResult contains
        associations (empty if output_path is set -- results on disk) and
        PVE from null model. n_tested is the number of SNPs that passed
        filtering and were tested.

    Raises:
        MemoryError: If check_memory=True and insufficient memory.
        ValueError: If only one of eigenvalues/eigenvectors is provided.
    """
    # Unpack config if provided (config takes precedence over individual kwargs).
    # Streaming-specific params (bed_path, chunk_size, output_path, snps_indices,
    # hwe_threshold, validate_genotypes) remain as separate kwargs.
    if config is not None:
        kw = config.as_kwargs()
        maf_threshold = kw["maf_threshold"]
        miss_threshold = kw["miss_threshold"]
        l_min, l_max = kw["l_min"], kw["l_max"]
        n_grid, n_refine = kw["n_grid"], kw["n_refine"]
        use_gpu, check_memory = kw["use_gpu"], kw["check_memory"]
        show_progress, lmm_mode = kw["show_progress"], kw["lmm_mode"]

    start_time = time.perf_counter()

    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps = meta["n_snps"]

    if snp_info is None:
        snp_info = _LazySnpMeta(meta)

    # Validate inputs and apply sample filtering (shared logic for all runners)
    setup = validate_runner_inputs(
        phenotypes, kinship, covariates, eigenvalues, eigenvectors, lmm_mode
    )
    phenotypes = setup.phenotypes
    kinship = setup.kinship
    covariates = setup.covariates
    eigenvalues = setup.eigenvalues
    eigenvectors = setup.eigenvectors
    n_valid = setup.n_samples
    valid_mask = setup.valid_mask

    n_samples = phenotypes.shape[0]

    # Always log memory estimate (useful even without hard check).
    # jax_chunk_size is computed from n_snps (upper bound; n_filtered unknown
    # until after pass 1) — same logic as runtime _compute_chunk_size call.
    est_jax_chunk = _compute_chunk_size(n_snps, n_samples=n_samples, pipeline_buffers=2)
    est = estimate_lmm_streaming_memory(
        n_samples,
        n_snps,
        chunk_size=chunk_size,
        pipeline_buffers=2,
        jax_chunk_size=est_jax_chunk,
    )
    logger.info(
        f"LMM streaming memory: estimated peak {est.total_peak_gb:.1f}GB, "
        f"available {est.available_gb:.1f}GB"
    )
    if check_memory and not est.sufficient:
        raise MemoryError(
            f"Insufficient memory for streaming LMM with {n_samples:,} samples "
            f"x {n_snps:,} SNPs (chunk_size={chunk_size:,}).\n"
            f"Peak: {est.total_peak_gb:.1f}GB, "
            f"Available: {est.available_gb:.1f}GB\n"
            f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
            f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
            f"eigendecomp_workspace={est.eigendecomp_workspace_gb:.1f}GB"
        )

    if show_progress:
        logger.info("Performing LMM Association Test (streaming)")
        logger.info(f"  Total individuals: {n_samples_total:,}")
        logger.info(f"  Analyzed individuals: {n_valid:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.info(f"  Lambda range: [{l_min:.2e}, {l_max:.2e}]")

    placement = resolve_device_placement(use_gpu)
    needs_sample_filter = not np.all(valid_mask)

    # === PASS 1: SNP statistics (without loading all genotypes) ===
    t_io_start = time.perf_counter()
    all_means = np.zeros(n_snps, dtype=np.float64)
    all_miss_counts = np.zeros(n_snps, dtype=np.int32)
    all_vars = np.zeros(n_snps, dtype=np.float64)

    # HWE genotype count accumulators (only when threshold > 0)
    if hwe_threshold > 0:
        all_n_aa = np.zeros(n_snps, dtype=np.int64)
        all_n_ab = np.zeros(n_snps, dtype=np.int64)
        all_n_bb = np.zeros(n_snps, dtype=np.int64)

    # Genotype validation accumulator
    n_unexpected_total = 0

    stats_iterator = stream_genotype_chunks(
        bed_path, chunk_size=chunk_size, dtype=np.float32, show_progress=False
    )
    if show_progress:
        n_chunks = (n_snps + chunk_size - 1) // chunk_size
        stats_iterator = progress_iterator(
            stats_iterator, total=n_chunks, desc="Computing SNP statistics"
        )

    with jax.profiler.TraceAnnotation("pass1_snp_statistics"):
        for chunk, start, end in stats_iterator:
            # Apply sample filtering
            if needs_sample_filter:
                chunk = chunk[valid_mask, :]

            # Compute stats for this chunk
            chunk_miss_counts = np.sum(np.isnan(chunk), axis=0)
            with np.errstate(invalid="ignore"):
                chunk_means = np.nanmean(chunk, axis=0)
                chunk_vars = np.nanvar(chunk, axis=0)
            chunk_means = np.nan_to_num(chunk_means, nan=0.0)
            chunk_vars = np.nan_to_num(chunk_vars, nan=0.0)

            all_means[start:end] = chunk_means
            all_miss_counts[start:end] = chunk_miss_counts
            all_vars[start:end] = chunk_vars

            # Accumulate HWE genotype counts (no extra disk pass)
            if hwe_threshold > 0:
                valid_geno = ~np.isnan(chunk)
                all_n_aa[start:end] += np.sum((chunk == 0) & valid_geno, axis=0)
                all_n_ab[start:end] += np.sum((chunk == 1) & valid_geno, axis=0)
                all_n_bb[start:end] += np.sum((chunk == 2) & valid_geno, axis=0)

            if validate_genotypes:
                n_unexpected_total += validate_genotype_values(chunk)

    if validate_genotypes and n_unexpected_total > 0:
        logger.warning(
            f"Genotype validation: {n_unexpected_total} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    t_io_end = time.perf_counter()

    # === SNP statistics: filtering + stats construction ===
    t_snp_start = time.perf_counter()
    snp_mask, allele_freqs, _mafs = compute_snp_filter_mask(
        all_means, all_miss_counts, all_vars, n_samples, maf_threshold, miss_threshold
    )
    del all_vars  # Only used by compute_snp_filter_mask

    # Apply SNP list restriction (if -snps provided)
    if snps_indices is not None:
        from jamma.core.snp_filter import apply_snp_list_mask

        apply_snp_list_mask(snp_mask, snps_indices, n_snps, "SNP list filter")

    # Apply HWE filter (if -hwe threshold > 0)
    if hwe_threshold > 0:
        from jamma.core.snp_filter import compute_hwe_pvalues

        hwe_pvalues = compute_hwe_pvalues(all_n_aa, all_n_ab, all_n_bb)
        hwe_pass = hwe_pvalues >= hwe_threshold
        n_hwe_fail = int(np.sum(~hwe_pass & snp_mask))
        snp_mask &= hwe_pass
        logger.info(f"HWE filter: {n_hwe_fail} SNPs removed (p < {hwe_threshold})")

    snp_indices = np.where(snp_mask)[0]
    n_filtered = len(snp_indices)

    if show_progress:
        from jamma.core.estimates import estimate_lmm_time

        logger.info(f"  Analyzed SNPs: {n_filtered:,}")
        logger.info(f"  Estimated time: {estimate_lmm_time(n_samples, n_filtered)}")

    if output_path is None and n_filtered > 100_000:
        logger.warning(
            f"In-memory mode with {n_filtered:,} SNPs. Results will accumulate "
            f"in memory. Provide output_path to stream results to disk."
        )

    if n_filtered == 0:
        if output_path is not None:
            with IncrementalAssocWriter(
                output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
            ):
                pass  # Context manager writes header, no data rows
        if show_progress:
            elapsed = time.perf_counter() - start_time
            logger.info(
                f"LMM Association completed in {elapsed:.2f}s (no SNPs passed filter)"
            )
        return LmmRunResult(associations=[]), 0

    filtered_afs = allele_freqs[snp_indices]
    filtered_miss = all_miss_counts[snp_indices].astype(int)
    del all_miss_counts, allele_freqs
    filtered_means = all_means[snp_indices]
    del all_means
    if hwe_threshold > 0:
        del all_n_aa, all_n_ab, all_n_bb

    t_snp_end = time.perf_counter()

    # === Eigendecomp + setup ===
    t_eigen_start = time.perf_counter()
    with jax.profiler.TraceAnnotation("eigendecomp_and_setup"):
        eigenvalues_np, U = _eigendecompose_or_reuse(
            kinship,
            eigenvalues,
            eigenvectors,
            show_progress,
            "lmm_streaming",
            check_memory=check_memory,
        )
        if kinship is not None:
            del kinship
        gc.collect()

        W, n_cvt = _build_covariate_matrix(covariates, n_samples)

        # Use all physical cores for BLAS rotation (no JAX contention)
        rotation_threads = get_physical_core_count()

        with blas_threads(rotation_threads):
            UtW = U.T @ W
            Uty = U.T @ phenotypes

        jax_chunk_size = _compute_chunk_size(
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

        eigenvalues_jax = jax.device_put(eigenvalues_np, placement.rep)
        UtW_jax = jax.device_put(UtW, placement.rep)
        Uty_jax = jax.device_put(Uty, placement.rep)

        pve, pve_se = compute_and_log_pve(eigenvalues_np, UtW, Uty, n_cvt, l_min, l_max)

        last_run_timing.clear()

        t_rotation_total = 0.0
        t_rotation_exposed_total = 0.0
        t_jax_compute_total = 0.0
        t_result_write_total = 0.0

        prev_compute_end: float | None = None
        all_results: list[AssocResult] = []
        n_at_lmin = 0
        n_at_lmax = 0

        with contextlib.ExitStack() as stack:
            writer = None
            if output_path is not None:
                writer = stack.enter_context(
                    IncrementalAssocWriter(
                        output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
                    )
                )
            assoc_iterator = stream_genotype_chunks(
                bed_path,
                chunk_size=chunk_size,
                dtype=np.float64,
                show_progress=False,
                snp_indices=snp_indices,
            )
            assoc_iterator = prefetch_iterator(assoc_iterator)
            if show_progress:
                n_chunks = (n_filtered + chunk_size - 1) // chunk_size
                assoc_iterator = progress_iterator(
                    assoc_iterator, total=n_chunks, desc="Running LMM association"
                )

            def _prepare_jax_chunk(
                start: int, end: int, geno: np.ndarray
            ) -> tuple[np.ndarray, int]:
                """Slice a genotype subset and prepare UtG for device transfer."""
                geno_slice = geno[:, start:end]
                return prepare_utg_chunk(geno_slice, U, placement, rotation_threads)

            def _prepare_jax_chunk_timed(
                start: int, end: int, geno: np.ndarray
            ) -> tuple[np.ndarray, int, float]:
                """Measure background-thread preparation time inside the worker."""
                t0 = time.perf_counter()
                UtG_np, actual_len = _prepare_jax_chunk(start, end, geno)
                return UtG_np, actual_len, time.perf_counter() - t0

            with ThreadPoolExecutor(max_workers=1) as executor:
                for chunk, filt_start, filt_end in assoc_iterator:
                    if needs_sample_filter:
                        chunk = chunk[valid_mask, :]

                    if filt_end <= filt_start:
                        continue

                    filtered_means_broadcast = filtered_means[
                        filt_start:filt_end
                    ].reshape(1, -1)
                    missing_mask = np.isnan(chunk)
                    if missing_mask.any():
                        chunk = np.where(missing_mask, filtered_means_broadcast, chunk)
                    del missing_mask

                    n_subset = chunk.shape[1]
                    jax_starts = compute_subchunk_starts(
                        n_subset, jax_chunk_size, placement.n_devices
                    )
                    jax_ends = [
                        jax_starts[i + 1] if i + 1 < len(jax_starts) else n_subset
                        for i in range(len(jax_starts))
                    ]

                    t_rot_start = time.perf_counter()
                    UtG_np, actual_jax_len = _prepare_jax_chunk(
                        jax_starts[0], jax_ends[0], chunk
                    )
                    t_rot_end = time.perf_counter()
                    rot_dur = t_rot_end - t_rot_start
                    t_rotation_total += rot_dur
                    t_rotation_exposed_total += exposed_rotation_time(
                        rot_dur, t_rot_end, prev_compute_end
                    )
                    UtG_jax = jax.device_put(UtG_np, placement.snp)
                    del UtG_np

                    for i, _jax_start in enumerate(jax_starts):
                        current_actual_len = actual_jax_len
                        current_UtG = UtG_jax

                        future = None
                        if i + 1 < len(jax_starts):
                            future = executor.submit(
                                _prepare_jax_chunk_timed,
                                jax_starts[i + 1],
                                jax_ends[i + 1],
                                chunk,
                            )

                        t_jax_start = time.perf_counter()

                        try:
                            with jax.profiler.TraceAnnotation("jax_optimization"):
                                Uab_batch = batch_compute_uab(
                                    n_cvt, UtW_jax, Uty_jax, current_UtG
                                )

                                chunk_result = _compute_lmm_chunk(
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
                                block_chunk_result(chunk_result, lmm_mode)
                        except Exception as e:
                            if future is not None:
                                future.cancel()
                            log_jax_error(
                                e,
                                chunk_label=f"streaming {i + 1}",
                                chunk_snps=current_actual_len,
                                n_samples=n_samples,
                                n_cvt=n_cvt,
                            )
                            raise

                        t_jax_end = time.perf_counter()
                        t_jax_compute_total += t_jax_end - t_jax_start
                        prev_compute_end = t_jax_end

                        if future is not None:
                            try:
                                UtG_np, actual_jax_len, rot_dur = future.result()
                            except MemoryError:
                                raise
                            except Exception as exc:
                                processed = (
                                    writer.count
                                    if writer is not None
                                    else len(all_results)
                                )
                                raise RuntimeError(
                                    f"Background rotation failed for streaming "
                                    f"sub-chunk starting at index "
                                    f"{jax_starts[i + 1]}. "
                                    f"Processed ~{processed} results "
                                    f"before failure."
                                ) from exc
                            t_rot_end = time.perf_counter()
                            t_rotation_total += rot_dur
                            t_rotation_exposed_total += min(
                                rot_dur, max(0.0, t_rot_end - t_jax_end)
                            )
                            UtG_jax = jax.device_put(UtG_np, placement.snp)
                            del UtG_np  # Safe: JAX holds internal ref

                        subchunk_filtered_start = filt_start + jax_starts[i]
                        subchunk_filtered_end = (
                            subchunk_filtered_start + current_actual_len
                        )

                        t_write_start = time.perf_counter()
                        with jax.profiler.TraceAnnotation("result_write"):
                            arrays = _chunk_result_to_numpy(
                                chunk_result,
                                _ACCUM_KEYS[lmm_mode],
                                current_actual_len,
                            )

                            chunk_lmin, chunk_lmax = count_lambda_boundary_hits(
                                lmm_mode, arrays, l_min, l_max
                            )
                            n_at_lmin += chunk_lmin
                            n_at_lmax += chunk_lmax

                            if writer is not None:
                                writer.write_arrays_batch(
                                    lmm_mode,
                                    snp_indices[
                                        subchunk_filtered_start:subchunk_filtered_end
                                    ],
                                    snp_info,
                                    filtered_afs[
                                        subchunk_filtered_start:subchunk_filtered_end
                                    ],
                                    filtered_miss[
                                        subchunk_filtered_start:subchunk_filtered_end
                                    ],
                                    arrays,
                                )
                            else:
                                chunk_results = list(
                                    _yield_chunk_results(
                                        lmm_mode,
                                        np.arange(
                                            subchunk_filtered_start,
                                            subchunk_filtered_end,
                                        ),
                                        snp_indices,
                                        filtered_afs,
                                        filtered_miss,
                                        snp_info,
                                        arrays,
                                    )
                                )
                                all_results.extend(chunk_results)

                        del arrays, chunk_result, Uab_batch, current_UtG
                        if future is None:
                            UtG_jax = None
                        t_write_end = time.perf_counter()
                        t_result_write_total += t_write_end - t_write_start

            if show_progress:
                log_rss_memory("lmm_streaming", "after_association")

                elapsed = time.perf_counter() - start_time
                t_io = t_io_end - t_io_start
                t_snp = t_snp_end - t_snp_start
                t_eigen = t_eigen_end - t_eigen_start
                accounted = (
                    t_io
                    + t_snp
                    + t_eigen
                    + t_rotation_total
                    + t_jax_compute_total
                    + t_result_write_total
                )
                logger.info("Timing breakdown:")
                logger.info(f"  I/O read (pass 1):   {t_io:.2f}s")
                logger.info(f"  SNP statistics:      {t_snp:.2f}s")
                logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
                logger.info(f"  UT@G rotation:       {t_rotation_total:.2f}s")
                logger.info(f"  UT@G exposed:        {t_rotation_exposed_total:.2f}s")
                logger.info(f"  JAX compute:         {t_jax_compute_total:.2f}s")
                logger.info(f"  Result write:        {t_result_write_total:.2f}s")
                logger.info("  ----")
                logger.info(f"  Accounted:           {accounted:.2f}s")
                logger.info(f"  Total:               {elapsed:.2f}s")

            if writer is not None and show_progress:
                logger.info(f"Wrote {writer.count:,} results to {output_path}")

            log_lambda_boundary_warning(n_at_lmin, n_at_lmax, l_min, l_max)

            if show_progress:
                elapsed = time.perf_counter() - start_time
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

            n_tested = writer.count if writer is not None else len(all_results)
            return (
                LmmRunResult(
                    associations=[] if output_path is not None else all_results,
                    pve=pve,
                    pve_se=pve_se,
                    n_tested=n_tested if output_path is not None else None,
                ),
                n_tested,
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
