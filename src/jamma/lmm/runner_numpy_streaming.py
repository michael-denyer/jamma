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

import numpy as np
from loguru import logger

from jamma.core.snp_stats import (
    SnpFilterSpec,
    collect_streamed_snp_stats,
    filter_snp_stats,
)
from jamma.io.plink import get_plink_metadata, stream_genotype_chunks
from jamma.lmm.chunk_runner_numpy import RawLmmChunk, run_lmm_chunk_source_numpy
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,
    prepare_lmm_run,
    validate_runner_inputs,
)
from jamma.lmm.results import make_result_list_sink, make_writer_sink
from jamma.lmm.schema import (
    DEFAULT_LMM_CONFIG,
    LmmConfig,
    LmmRunResult,
    RunnerTiming,
    SnpMeta,
)
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.lmm.stats import AssocResult
from jamma.utils.logging import log_rss_memory

# SNPs per block in the statistics pass when the caller names no chunk size.
# Pass 1 reads the .bed and accumulates per-SNP counts, so its footprint is one
# block of genotypes rather than the rotation and grid buffers the association
# pass carries; it needs no RAM-budgeted sizing of its own.
_DEFAULT_STATS_CHUNK = 10_000


def run_lmm_association_numpy_streaming(
    bed_path: Path,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None = None,
    snp_info: list | None = None,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    chunk_size: int | None = None,
    output_path: Path | None = None,
    snps_indices: np.ndarray | None = None,
    hwe_threshold: float = 0.0,
    validate_genotypes: bool = True,
    config: LmmConfig = DEFAULT_LMM_CONFIG,
) -> LmmRunResult:
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
        n_refine: Golden section iterations for lambda refinement (clamped to
            min 20 internally for ~1e-5 tolerance).
        chunk_size: Cap on SNPs per chunk, for both the statistics pass and
            the association pass. None (default) reads statistics in
            _DEFAULT_STATS_CHUNK blocks and lets the chunk engine size the
            association chunks against the RAM budget.
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
        LmmRunResult with associations (empty if output_path is set --
        results on disk), PVE from the null model, n_tested counting the
        SNPs that passed filtering and were tested, and the run's timing
        breakdown.
    """
    # One source for every knob. The runner reads locals rather than config.x
    # at forty-odd sites; the dual surface this replaced let a caller pass both
    # a config and a contradicting keyword.
    maf_threshold = config.maf_threshold
    miss_threshold = config.miss_threshold
    l_min, l_max = config.l_min, config.l_max
    n_grid, n_refine = config.n_grid, config.n_refine
    check_memory = config.check_memory
    show_progress = config.show_progress
    lmm_mode = config.lmm_mode

    # Checked here, not in the chunk runner, because the statistics pass reads
    # the whole .bed first: a bad value must fail before that, not after it.
    if chunk_size is not None and chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1 or None, got {chunk_size}")

    start_time = time.perf_counter()

    meta = get_plink_metadata(bed_path)
    n_samples_total = meta["n_samples"]
    n_snps = meta["n_snps"]

    # Caller-supplied list, or the PLINK metadata, parsed once into columns.
    snp_meta = (
        SnpMeta.from_plink_meta(meta)
        if snp_info is None
        else SnpMeta.from_dicts(snp_info)
    )

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
        chunk_size=_DEFAULT_STATS_CHUNK if chunk_size is None else chunk_size,
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
        return LmmRunResult(associations=[], n_tested=0)

    filtered_afs = snp_selection.filtered_afs
    filtered_miss = snp_selection.filtered_miss
    filtered_means = snp_selection.filtered_means
    del stats, snp_selection

    t_snp_end = time.perf_counter()

    # === Eigendecomp + rotation + null model ===
    t_eigen_start = time.perf_counter()

    W, n_cvt = _build_covariate_matrix(covariates, n_samples)

    prepared = prepare_lmm_run(
        kinship=kinship,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        phenotypes=phenotypes,
        W=W,
        n_cvt=n_cvt,
        lmm_mode=lmm_mode,
        l_min=l_min,
        l_max=l_max,
        show_progress=show_progress,
        check_memory=check_memory,
        label="lmm_numpy_streaming",
    )
    if kinship is not None:
        del kinship
    gc.collect()

    t_eigen_end = time.perf_counter()

    # === PASS 2: Compute per chunk (float64) ===
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

        if writer is not None:
            _sink = make_writer_sink(
                writer, lmm_mode, snp_meta, snp_indices, filtered_afs, filtered_miss
            )
        else:
            _sink = make_result_list_sink(
                all_results,
                lmm_mode,
                snp_meta,
                snp_indices,
                filtered_afs,
                filtered_miss,
            )

        chunk_stats = run_lmm_chunk_source_numpy(
            raw_chunk_source_factory=_make_stream_source,
            chunk_sink=_sink,
            U=prepared.U,
            eigenvalues_np=prepared.eigenvalues,
            UtW=prepared.UtW,
            Uty=prepared.Uty,
            Hi_eval_null=prepared.Hi_eval_null,
            logl_H0=prepared.logl_H0,
            n_samples=n_samples,
            n_filtered=n_filtered,
            n_cvt=n_cvt,
            lmm_mode=lmm_mode,
            filtered_means=filtered_means,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_refine=n_refine,
            max_chunk_size=chunk_size,
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

        n_tested = writer.count if writer is not None else len(all_results)
        return LmmRunResult(
            associations=[] if output_path is not None else all_results,
            n_tested=n_tested,
            pve=prepared.pve,
            pve_se=prepared.pve_se,
            timing=RunnerTiming(
                rotation_s=chunk_stats.rotation_s,
                numpy_compute_s=chunk_stats.compute_s,
                result_write_s=chunk_stats.result_write_s,
            ),
        )
