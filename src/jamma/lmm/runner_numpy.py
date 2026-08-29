"""Shared NumPy LMM run body and the in-memory batch runner.

One body, many sources: ``_run_numpy_lmm`` drives per-SNP statistics,
filtering, preparation, and the chunk engine over a ``GenotypeSource``.
The batch and streaming runners are thin wrappers that build a source;
LOCO builds one per chromosome.
"""

from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from jamma.core.memory import estimate_lmm_memory
from jamma.core.memory_snapshot import log_memory_snapshot
from jamma.core.snp_filter import _SNP_STATS_CHUNK_SIZE
from jamma.core.snp_stats import (
    SnpFilterSpec,
    SnpStats,
    collect_snp_stats_from_chunks,
    filter_snp_stats,
)
from jamma.lmm.chunk_runner_numpy import RawLmmChunk, run_lmm_chunk_source_numpy
from jamma.lmm.chunk_sizing import plan_lmm_chunks
from jamma.lmm.compute_numpy import select_current_dispatch_path
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


class GenotypeSource(Protocol):
    """Where a run's genotypes come from.

    The run body asks for per-SNP statistics once, then for a chunk stream
    over the filtered SNP set. Sample filtering is the source's job: both
    methods receive the valid-sample mask and must compute over exactly
    those rows. Each source also owns its statistics dtype, so a source
    that must match a historical accumulation (float32 streaming pass,
    float64 LOCO reads) keeps it without a knob on the body.
    """

    @property
    def n_snps(self) -> int:
        """Total SNP columns before filtering."""
        ...

    def snp_stats(self, valid_mask: np.ndarray, *, include_hwe: bool) -> SnpStats:
        """Per-SNP statistics over valid rows, with ``global_indices`` set."""
        ...

    def chunks(
        self, chunk_size: int, snp_indices: np.ndarray, valid_mask: np.ndarray
    ) -> Callable[[], RawLmmChunk | None]:
        """A next-chunk callable over the filtered SNP columns."""
        ...


class MatrixSource:
    """An in-memory genotype matrix as a :class:`GenotypeSource`."""

    def __init__(self, genotypes: np.ndarray) -> None:
        self._genotypes = genotypes
        self._rows: np.ndarray | None = None

    @property
    def n_snps(self) -> int:
        return self._genotypes.shape[1]

    def _valid_rows(self, valid_mask: np.ndarray) -> np.ndarray:
        # Row-filter once and share between the statistics pass and the chunk
        # stream; a second fancy-index would hold two filtered copies at once.
        if self._rows is None:
            self._rows = (
                self._genotypes
                if bool(np.all(valid_mask))
                else self._genotypes[valid_mask, :]
            )
        return self._rows

    def snp_stats(self, valid_mask: np.ndarray, *, include_hwe: bool) -> SnpStats:
        rows = self._valid_rows(valid_mask)
        n_samples, n_snps = rows.shape

        def _stat_chunks():
            for start in range(0, n_snps, _SNP_STATS_CHUNK_SIZE):
                end = min(start + _SNP_STATS_CHUNK_SIZE, n_snps)
                yield np.ascontiguousarray(rows[:, start:end]), start, end

        return collect_snp_stats_from_chunks(
            _stat_chunks(),
            n_snps=n_snps,
            n_samples=n_samples,
            global_indices=np.arange(n_snps, dtype=np.intp),
            include_hwe=include_hwe,
            sample_scope=(
                "all_samples" if bool(np.all(valid_mask)) else "valid_samples"
            ),
        )

    def chunks(
        self, chunk_size: int, snp_indices: np.ndarray, valid_mask: np.ndarray
    ) -> Callable[[], RawLmmChunk | None]:
        rows = self._valid_rows(valid_mask)
        n_filtered = len(snp_indices)
        chunk_starts = iter(range(0, n_filtered, chunk_size))
        geno_buf = np.empty((rows.shape[0], chunk_size), dtype=np.float64)

        def _next_chunk() -> RawLmmChunk | None:
            try:
                chunk_start = next(chunk_starts)
            except StopIteration:
                return None

            chunk_end = min(chunk_start + chunk_size, n_filtered)
            actual_len = chunk_end - chunk_start
            geno_chunk = geno_buf[:, :actual_len]
            geno_chunk[:] = rows[:, snp_indices[chunk_start:chunk_end]]
            return RawLmmChunk(geno_chunk, chunk_start, chunk_end)

        return _next_chunk


def _run_numpy_lmm(
    source: GenotypeSource,
    *,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    covariates: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    snp_meta: SnpMeta,
    config: LmmConfig,
    output_path: Path | None,
    banner: str,
    label: str,
    writer: IncrementalAssocWriter | None = None,
    snps_indices: np.ndarray | None = None,
    hwe_threshold: float = 0.0,
    max_chunk_size: int | None = None,
    compute_pve: bool = True,
    progress_label: str = "LMM association",
    lambda_warning_prefix: str = "",
    log_dispatch_choices: bool = True,
) -> LmmRunResult:
    """Run one NumPy LMM association over any genotype source.

    Statistics, filtering, eigen preparation, the chunk loop, and result
    routing are identical for every runner; only where genotypes come from
    differs, and that lives in ``source``.

    Args:
        source: Genotype provider; owns sample-row filtering and stats dtype.
        phenotypes: Phenotype vector (n_samples_total,), NaN for missing.
        kinship: Kinship matrix, or None when eigenpairs are supplied.
        covariates: Covariate matrix or None for intercept-only.
        eigenvalues: Pre-computed eigenvalues (ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        snp_meta: SNP metadata columns indexed by global SNP index.
        config: Thresholds, lambda bounds and grid, test type, progress.
        output_path: Stream results to this file, or None for in-memory.
        banner: Runner name for the progress banner ("NumPy batch", ...).
        label: Memory-log label identifying the calling runner.
        writer: A caller-owned writer to append results to, instead of
            output_path. LOCO shares one writer across its chromosome
            loop; the body neither opens nor closes it.
        snps_indices: Global indices restricting the tested SNP set, or None.
        hwe_threshold: HWE p-value threshold; 0.0 disables the filter.
        max_chunk_size: Cap on association-pass chunk width, or None to let
            the engine size chunks against the RAM budget.
        compute_pve: Whether to run the null-REML PVE estimate.
        progress_label: Chunk-loop progress bar label.
        lambda_warning_prefix: Prefix for lambda-boundary warnings.
        log_dispatch_choices: Whether the chunk engine logs its dispatch.

    Returns:
        LmmRunResult with associations (empty when output_path routed them
        to disk), n_tested, PVE, and the run's timing breakdown.
    """
    if output_path is not None and writer is not None:
        raise ValueError("pass output_path or writer, not both")

    maf_threshold = config.maf_threshold
    miss_threshold = config.miss_threshold
    l_min, l_max = config.l_min, config.l_max
    n_grid, n_refine = config.n_grid, config.n_refine
    check_memory = config.check_memory
    show_progress = config.show_progress
    lmm_mode = config.lmm_mode

    start_time = time.perf_counter()
    n_samples_total = phenotypes.shape[0]
    n_snps = source.n_snps

    setup = validate_runner_inputs(
        phenotypes, kinship, covariates, eigenvalues, eigenvectors, lmm_mode
    )
    phenotypes = setup.phenotypes
    kinship = setup.kinship
    covariates = setup.covariates
    eigenvalues = setup.eigenvalues
    eigenvectors = setup.eigenvectors
    valid_mask = setup.valid_mask
    n_samples = phenotypes.shape[0]

    if show_progress:
        logger.info(f"Performing LMM Association Test ({banner})")
        logger.info(f"  Total individuals: {n_samples_total:,}")
        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.info(f"  Lambda range: [{l_min:.2e}, {l_max:.2e}]")

    # === PASS 1: per-SNP statistics + filtering ===
    t_stats_start = time.perf_counter()
    stats = source.snp_stats(valid_mask, include_hwe=hwe_threshold > 0)
    if stats.n_unexpected > 0:
        logger.warning(
            f"Genotype validation: {stats.n_unexpected} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    selection = filter_snp_stats(
        stats,
        SnpFilterSpec(
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            restrict_indices=snps_indices,
            hwe_threshold=hwe_threshold,
            restrict_label="SNP list filter",
        ),
    )
    snp_indices = selection.indices
    n_filtered = len(snp_indices)

    if show_progress:
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")

    if n_filtered == 0:
        logger.warning(
            f"All {n_snps} SNPs filtered out (MAF>{maf_threshold}, "
            f"miss<{miss_threshold}). No association tests to run. "
            f"Consider relaxing --maf or --miss thresholds."
        )
        if output_path is not None:
            with IncrementalAssocWriter(
                output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
            ):
                pass  # Context manager writes the header, no data rows
        return LmmRunResult(associations=[], n_tested=0)

    filtered_afs = selection.filtered_afs
    filtered_miss = selection.filtered_miss
    filtered_means = selection.filtered_means
    del stats, selection
    t_stats_end = time.perf_counter()

    # === Eigendecomp + rotation + null model + PVE ===
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
        label=label,
        compute_pve=compute_pve,
    )
    del kinship
    gc.collect()
    t_eigen_end = time.perf_counter()

    # === PASS 2: association per chunk ===
    all_results: list[AssocResult] = []
    with contextlib.ExitStack() as stack:
        if output_path is not None:
            writer = stack.enter_context(
                IncrementalAssocWriter(output_path, test_type=_TEST_TYPE_MAP[lmm_mode])
            )

        sink_args = (lmm_mode, snp_meta, snp_indices, filtered_afs, filtered_miss)
        chunk_sink = (
            make_writer_sink(writer, *sink_args)
            if writer is not None
            else make_result_list_sink(all_results, *sink_args)
        )

        chunk_stats = run_lmm_chunk_source_numpy(
            raw_chunk_source_factory=lambda size: source.chunks(
                size, snp_indices, valid_mask
            ),
            chunk_sink=chunk_sink,
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
            max_chunk_size=max_chunk_size,
            show_progress=show_progress,
            progress_label=progress_label,
            lambda_warning_prefix=lambda_warning_prefix,
            log_dispatch_choices=log_dispatch_choices,
        )

        if show_progress:
            log_memory_snapshot(f"{label}:after_association")

            elapsed = time.perf_counter() - start_time
            t_stats = t_stats_end - t_stats_start
            t_eigen = t_eigen_end - t_eigen_start
            accounted = (
                t_stats
                + t_eigen
                + chunk_stats.rotation_s
                + chunk_stats.compute_s
                + chunk_stats.result_write_s
            )
            logger.info("Timing breakdown:")
            logger.info(f"  SNP statistics:      {t_stats:.2f}s")
            logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
            logger.info(f"  UT@G rotation:       {chunk_stats.rotation_s:.2f}s")
            logger.info(f"  NumPy compute:       {chunk_stats.compute_s:.2f}s")
            logger.info(f"  Result write:        {chunk_stats.result_write_s:.2f}s")
            logger.info("  ----")
            logger.info(f"  Accounted:           {accounted:.2f}s")
            logger.info(f"  Total:               {elapsed:.2f}s")

            if output_path is not None and writer is not None:
                logger.info(f"Wrote {writer.count:,} results to {output_path}")
            logger.info(f"LMM Association completed in {elapsed:.2f}s")

        return LmmRunResult(
            associations=all_results if writer is None else [],
            n_tested=chunk_stats.processed,
            pve=prepared.pve,
            pve_se=prepared.pve_se,
            timing=RunnerTiming(
                rotation_s=chunk_stats.rotation_s,
                numpy_compute_s=chunk_stats.compute_s,
                result_write_s=chunk_stats.result_write_s,
            ),
        )


def run_lmm_association_numpy(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    snp_info: list | SnpMeta,
    covariates: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    config: LmmConfig = DEFAULT_LMM_CONFIG,
    output_path: Path | None = None,
    hwe_threshold: float = 0.0,
) -> LmmRunResult:
    """Run LMM association tests using pure-NumPy batch processing.

    Processes SNPs in memory-bounded chunks using BLAS-backed NumPy
    operations. Input genotypes must fit in memory; for disk streaming
    use run_lmm_association_numpy_streaming.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2.
        phenotypes: Phenotype vector (n_samples,).
        kinship: Kinship matrix (n_samples, n_samples) or None when
            pre-computed eigenvalues/eigenvectors are provided. WARNING: may
            be overwritten in-place during eigendecomposition (buffer reused
            for eigenvectors). Treat as consumed; pass kinship.copy() if you
            need the original matrix after this call.
        snp_info: SnpMeta, or a list of dicts with keys chr, rs, pos, a1, a0
            for the public batch API.
        covariates: Covariate matrix (n_samples, n_cvt) or None for
            intercept-only.
        eigenvalues: Pre-computed eigenvalues (sorted ascending) or None.
        eigenvectors: Pre-computed eigenvectors or None.
        config: LmmConfig with thresholds, lambda bounds, test type,
            memory check and progress settings.
        output_path: Path for per-chunk disk streaming. When set, results
            are written incrementally and the returned LmmRunResult has
            empty associations, with n_tested counting the SNPs written.
        hwe_threshold: HWE p-value threshold; SNPs with p < threshold are
            removed. 0.0 disables HWE filtering (default).

    Returns:
        LmmRunResult with per-SNP associations, n_tested, and PVE from the
        null model. When output_path is set, associations is empty and the
        results are on disk.

    Raises:
        MemoryError: If check_memory=True and insufficient memory.
        ValueError: If only one of eigenvalues/eigenvectors is provided,
            or if no valid samples remain after filtering.
    """
    n_samples, n_snps = genotypes.shape
    # Intercept column counts as a covariate, so minimum is 1 when no user
    # covariates are passed.
    n_cvt = covariates.shape[1] if covariates is not None else 1

    # Plan the chunk once, from the same sizer the engine allocates from, and
    # give both the memory gate and the engine that one number. Otherwise the
    # gate prices a chunk it never actually allocates: this call has no
    # MemoryPlan from a pipeline preflight to inherit, so without this it
    # priced estimate_lmm_memory's lmm_batch_size=20_000 default while the
    # engine sized its own chunk from the real RAM budget and dispatch path.
    dispatch = select_current_dispatch_path(n_cvt, config.lmm_mode, log_choices=False)
    chunk_plan = plan_lmm_chunks(n_samples, n_snps, n_cvt, dispatch)

    if config.check_memory:
        # Propagate n_cvt so the preflight correctly sizes Uab/Iab for
        # multi-covariate runs. Otherwise the estimator silently uses its
        # n_cvt=1 default and can let a multi-covariate run pass preflight
        # before OOMing at the real allocation.
        est = estimate_lmm_memory(
            n_samples,
            n_snps,
            lmm_batch_size=chunk_plan.chunk_size,
            n_cvt=n_cvt,
            n_buffers=chunk_plan.n_buffers,
        )
        logger.info(
            f"LMM memory: estimated {est.total_gb:.1f}GB, "
            f"available {est.available_gb:.1f}GB"
        )
        if not est.sufficient:
            raise MemoryError(
                f"Insufficient memory for LMM workflow with {n_samples:,} samples × "
                f"{n_snps:,} SNPs.\n"
                f"Need: {est.total_gb:.1f}GB, Available: {est.available_gb:.1f}GB\n"
                f"Breakdown: kinship={est.kinship_gb:.1f}GB, "
                f"eigenvectors={est.eigenvectors_gb:.1f}GB, "
                f"genotypes={est.genotypes_gb:.1f}GB"
            )

    snp_meta = (
        snp_info if isinstance(snp_info, SnpMeta) else SnpMeta.from_dicts(snp_info)
    )
    return _run_numpy_lmm(
        MatrixSource(genotypes),
        phenotypes=phenotypes,
        kinship=kinship,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        snp_meta=snp_meta,
        config=config,
        output_path=output_path,
        hwe_threshold=hwe_threshold,
        max_chunk_size=chunk_plan.chunk_size,
        banner="NumPy batch",
        label="lmm_numpy",
    )
