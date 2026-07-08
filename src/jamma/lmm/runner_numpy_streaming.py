"""Disk-streaming NumPy LMM association runner.

Two-pass disk streaming using C extension compute path:
  Pass 1: SNP statistics for filtering (float32, lightweight).
  Pass 2: Association per chunk via C workspace (float64, compute-heavy).

Never allocates the full genotype matrix. Uses jlinalg.dgemm for rotation
and the _lmm_accel C workspace for golden-section-optimized REML/MLE.
"""

from __future__ import annotations

import contextlib
import gc
import time
from pathlib import Path
from typing import cast

import numpy as np
from loguru import logger

from jamma.core.snp_stats import (
    SnpFilterSpec,
    collect_streamed_snp_stats,
    filter_snp_stats,
)
from jamma.core.threading import blas_threads, get_physical_core_count
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.lmm.compute_numpy import LmmMode
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    _compute_null_model_common,
    _eigendecompose_or_reuse,
    compute_and_log_pve,
    validate_runner_inputs,
)
from jamma.lmm.results import _build_results
from jamma.lmm.runner_numpy import RawLmmChunk, run_lmm_chunk_source_numpy
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

    Safe to call from any thread -- returns an independent dict.
    """
    return dict(last_run_timing)


def run_lmm_association_numpy_streaming(
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
    check_memory: bool = True,
    show_progress: bool = True,
    output_path: Path | None = None,
    lmm_mode: int = 1,
    snps_indices: np.ndarray | None = None,
    hwe_threshold: float = 0.0,
    validate_genotypes: bool = True,
    config: LmmConfig | None = None,
) -> tuple[LmmRunResult, int]:
    """Run LMM association tests by streaming genotypes from disk (NumPy/C path).

    Two-pass disk streaming: pass 1 computes SNP statistics for filtering,
    pass 2 runs C extension compute per chunk. Uses jlinalg.dgemm for
    eigenrotation and _lmm_accel C workspace for golden-section-optimized REML/MLE.

    Args:
        bed_path: PLINK file prefix (without .bed/.bim/.fam extension).
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples), or None when
            pre-computed eigenvalues and eigenvectors are provided.
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
        check_memory: Check available memory before workflow.
        show_progress: Show progress bars and GEMMA-style logging.
        output_path: Path for incremental result writing, or None for in-memory.
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        snps_indices: Pre-resolved column indices for -snps restriction, or None.
        hwe_threshold: HWE p-value threshold; SNPs with p < threshold are
            removed. 0.0 disables HWE filtering (default).
        validate_genotypes: Check for unexpected genotype values during pass-1.
        config: LmmConfig instance. When provided, overrides individual
            threshold/mode kwargs above.

    Returns:
        Tuple of (LmmRunResult, n_tested) where LmmRunResult contains
        associations (empty if output_path is set -- results on disk) and
        PVE from null model. n_tested is the number of SNPs that passed
        filtering and were tested.
    """
    # Unpack config if provided (config takes precedence over individual kwargs).
    if config is not None:
        kw = config.as_kwargs()
        maf_threshold = kw["maf_threshold"]
        miss_threshold = kw["miss_threshold"]
        l_min, l_max = kw["l_min"], kw["l_max"]
        n_grid, n_refine = kw["n_grid"], kw["n_refine"]
        check_memory = kw["check_memory"]
        show_progress, lmm_mode = kw["show_progress"], kw["lmm_mode"]

    start_time = time.perf_counter()

    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps = meta["n_snps"]

    if snp_info is None:
        snp_info = _LazySnpMeta(meta)

    # Validate inputs and apply sample filtering
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
    lmm_mode = cast(LmmMode, lmm_mode)

    n_samples = phenotypes.shape[0]

    if show_progress:
        logger.info("Performing LMM Association Test (NumPy streaming)")
        logger.info(f"  Total individuals: {n_samples_total:,}")
        logger.info(f"  Analyzed individuals: {n_valid:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.info(f"  Lambda range: [{l_min:.2e}, {l_max:.2e}]")

    needs_sample_filter = not np.all(valid_mask)

    # === PASS 1: SNP statistics (single-pass C kernel) ===
    t_io_start = time.perf_counter()
    stats_sample_indices = np.where(valid_mask)[0] if needs_sample_filter else None
    stats = collect_streamed_snp_stats(
        bed_path,
        n_snps=n_snps,
        n_samples=n_samples_total,
        chunk_size=chunk_size,
        sample_indices=stats_sample_indices,
        include_hwe=hwe_threshold > 0,
        validate_genotypes=validate_genotypes,
        show_progress=show_progress,
        progress_label="Computing SNP statistics",
        dtype=np.float32,
        sample_scope="valid_samples" if needs_sample_filter else "all_samples",
    )
    if validate_genotypes and stats.n_unexpected > 0:
        logger.warning(
            f"Genotype validation: {stats.n_unexpected} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    t_io_end = time.perf_counter()

    # === SNP statistics: filtering + stats construction ===
    t_snp_start = time.perf_counter()
    snp_selection = filter_snp_stats(
        stats,
        SnpFilterSpec(
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            restrict_indices=snps_indices,
            hwe_threshold=hwe_threshold,
            restrict_label="SNP list filter",
        ),
    )
    snp_indices = snp_selection.indices
    n_filtered = len(snp_indices)

    if show_progress:
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")

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

    filtered_afs = snp_selection.filtered_afs
    filtered_miss = snp_selection.filtered_miss
    filtered_means = snp_selection.filtered_means
    del stats, snp_selection

    t_snp_end = time.perf_counter()

    # === Eigendecomp + rotation + null model ===
    t_eigen_start = time.perf_counter()

    eigenvalues_np, U = _eigendecompose_or_reuse(
        kinship,
        eigenvalues,
        eigenvectors,
        show_progress,
        "lmm_numpy_streaming",
        check_memory=check_memory,
    )
    if kinship is not None:
        del kinship
    gc.collect()

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    # Use all physical cores for BLAS rotation
    rotation_threads = get_physical_core_count()

    with blas_threads(rotation_threads):
        UtW = U.T @ W
        Uty = U.T @ phenotypes

    # Null model for Score/LRT/All
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

    pve, pve_se = compute_and_log_pve(eigenvalues_np, UtW, Uty, n_cvt, l_min, l_max)

    # === PASS 2: Compute per chunk (float64) ===
    last_run_timing.clear()
    all_results: list[AssocResult] = []

    def _make_stream_source(source_chunk_size: int):
        chunk_iter = iter(
            stream_genotype_chunks(
                bed_path,
                chunk_size=source_chunk_size,
                dtype=np.float64,
                show_progress=False,
                snp_indices=snp_indices,
            )
        )

        def _next_chunk() -> RawLmmChunk | None:
            try:
                chunk, filt_start, filt_end = next(chunk_iter)
            except StopIteration:
                return None

            if needs_sample_filter:
                chunk = chunk[valid_mask, :]

            return RawLmmChunk(np.ascontiguousarray(chunk), filt_start, filt_end)

        return _next_chunk

    with contextlib.ExitStack() as stack:
        writer = None
        if output_path is not None:
            writer = stack.enter_context(
                IncrementalAssocWriter(output_path, test_type=_TEST_TYPE_MAP[lmm_mode])
            )

        def _sink(
            chunk_arrays: dict[str, np.ndarray], filtered_start: int, filtered_end: int
        ) -> None:
            if writer is not None:
                writer.write_arrays_batch(
                    lmm_mode,
                    snp_indices[filtered_start:filtered_end],
                    snp_info,
                    filtered_afs[filtered_start:filtered_end],
                    filtered_miss[filtered_start:filtered_end],
                    chunk_arrays,
                )
            else:
                chunk_results = _build_results(
                    lmm_mode,
                    snp_indices[filtered_start:filtered_end],
                    filtered_afs[filtered_start:filtered_end],
                    filtered_miss[filtered_start:filtered_end],
                    snp_info,
                    chunk_arrays,
                )
                all_results.extend(chunk_results)

        auto_scaled_chunk_size = chunk_size == 10_000
        requested_chunk_size = None if auto_scaled_chunk_size else chunk_size
        chunk_stats = run_lmm_chunk_source_numpy(
            raw_chunk_source_factory=_make_stream_source,
            chunk_sink=_sink,
            U=U,
            eigenvalues_np=eigenvalues_np,
            UtW=UtW,
            Uty=Uty,
            Hi_eval_null=Hi_eval_null,
            logl_H0=logl_H0,
            n_samples=n_samples,
            n_filtered=n_filtered,
            n_cvt=n_cvt,
            lmm_mode=lmm_mode,
            filtered_means=filtered_means,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_refine=n_refine,
            requested_chunk_size=requested_chunk_size,
            auto_scale_chunk_size=auto_scaled_chunk_size,
            show_progress=show_progress,
            progress_label="LMM association (streaming)",
            log_dispatch_choices=False,
        )

        # === Post-loop diagnostics ===
        if show_progress:
            log_rss_memory("lmm_numpy_streaming", "after_association")

        if show_progress:
            elapsed = time.perf_counter() - start_time
            t_io = t_io_end - t_io_start
            t_snp = t_snp_end - t_snp_start
            t_eigen = t_eigen_end - t_eigen_start
            accounted = (
                t_io
                + t_snp
                + t_eigen
                + chunk_stats.rotation_s
                + chunk_stats.compute_s
                + chunk_stats.result_write_s
            )
            logger.info("Timing breakdown:")
            logger.info(f"  I/O read (pass 1):   {t_io:.2f}s")
            logger.info(f"  SNP statistics:      {t_snp:.2f}s")
            logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
            logger.info(f"  UT@G rotation:       {chunk_stats.rotation_s:.2f}s")
            logger.info(f"  NumPy compute:       {chunk_stats.compute_s:.2f}s")
            logger.info(f"  Result write:        {chunk_stats.result_write_s:.2f}s")
            logger.info("  ----")
            logger.info(f"  Accounted:           {accounted:.2f}s")
            logger.info(f"  Total:               {elapsed:.2f}s")

        if writer is not None and show_progress:
            logger.info(f"Wrote {writer.count:,} results to {output_path}")

        if show_progress:
            elapsed = time.perf_counter() - start_time
            logger.info(f"LMM Association completed in {elapsed:.2f}s")

        last_run_timing.clear()
        last_run_timing.update(
            {
                "rotation_s": chunk_stats.rotation_s,
                "numpy_compute_s": chunk_stats.compute_s,
                "result_write_s": chunk_stats.result_write_s,
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
