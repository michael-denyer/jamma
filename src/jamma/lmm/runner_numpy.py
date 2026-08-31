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
from pathlib import Path

import numpy as np
from loguru import logger

from jamma.core import memory
from jamma.core.memory_snapshot import log_memory_snapshot
from jamma.core.snp_filter import _SNP_STATS_CHUNK_SIZE
from jamma.core.snp_stats import (
    SnpFilterSpec,
    collect_snp_stats_from_chunks,
)
from jamma.lmm.association_plan import (
    ExecutableAssociationPlan,
    plan_association,
)
from jamma.lmm.chunk_runner_numpy import (
    ChunkRunOptions,
    RawLmmChunk,
    run_lmm_chunk_source_numpy,
)
from jamma.lmm.genotype_source import (
    GenotypeSource,
    PreparedGenotypes,
    SampleBasis,
    bind_prepared_genotypes,
)
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


class MatrixSource:
    """An in-memory genotype matrix as a :class:`GenotypeSource`."""

    def __init__(self, genotypes: np.ndarray, snp_meta: SnpMeta) -> None:
        if genotypes.ndim != 2:
            raise ValueError(f"genotypes must be 2-D, got ndim={genotypes.ndim}")
        if genotypes.shape[1] != len(snp_meta):
            raise ValueError(
                "genotype columns must match paired SnpMeta: "
                f"got {genotypes.shape[1]} columns and {len(snp_meta)} metadata rows"
            )
        self._genotypes = genotypes
        self._snp_meta = snp_meta

    @property
    def n_snps(self) -> int:
        return self._genotypes.shape[1]

    def prepare(
        self, samples: SampleBasis, filters: SnpFilterSpec
    ) -> PreparedGenotypes:
        if samples.source_row_count != self._genotypes.shape[0]:
            raise ValueError(
                "sample basis row count must match genotype rows: "
                f"got {samples.source_row_count} and {self._genotypes.shape[0]}"
            )
        rows = (
            self._genotypes
            if samples.is_all_samples
            else self._genotypes[samples.positions, :]
        )
        n_samples, n_snps = rows.shape

        def _stat_chunks():
            for start in range(0, n_snps, _SNP_STATS_CHUNK_SIZE):
                end = min(start + _SNP_STATS_CHUNK_SIZE, n_snps)
                yield np.ascontiguousarray(rows[:, start:end]), start, end

        stats = collect_snp_stats_from_chunks(
            _stat_chunks(),
            n_snps=n_snps,
            n_samples=n_samples,
            global_indices=np.arange(n_snps, dtype=np.intp),
            include_hwe=filters.hwe_threshold > 0,
            sample_scope="all_samples" if samples.is_all_samples else "valid_samples",
        )

        def _bind_chunks(selection):
            selected_columns = selection.local_indices

            def _chunks(chunk_size: int):
                n_filtered = len(selected_columns)
                chunk_starts = iter(range(0, n_filtered, chunk_size))
                geno_buf = np.empty((rows.shape[0], chunk_size), dtype=np.float64)

                def _next_chunk() -> RawLmmChunk | None:
                    try:
                        chunk_start = next(chunk_starts)
                    except StopIteration:
                        return None

                    chunk_end = min(chunk_start + chunk_size, n_filtered)
                    actual_len = chunk_end - chunk_start
                    geno_chunk = (
                        geno_buf
                        if actual_len == chunk_size
                        else np.empty((rows.shape[0], actual_len), dtype=np.float64)
                    )
                    geno_chunk[:] = rows[:, selected_columns[chunk_start:chunk_end]]
                    return RawLmmChunk(geno_chunk, chunk_start, chunk_end)

                return _next_chunk

            return _chunks

        return bind_prepared_genotypes(
            snp_meta=self._snp_meta,
            stats=stats,
            filters=filters,
            chunk_factory=_bind_chunks,
        )


def _run_numpy_lmm(
    source: GenotypeSource,
    *,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    covariates: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    config: LmmConfig,
    output_path: Path | None,
    banner: str,
    label: str,
    writer: IncrementalAssocWriter | None = None,
    snps_indices: np.ndarray | None = None,
    hwe_threshold: float = 0.0,
    execution: ExecutableAssociationPlan,
    compute_pve: bool = True,
    progress_label: str = "LMM association",
    lambda_warning_prefix: str = "",
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
        config: Thresholds, lambda bounds and grid, test type, progress.
        output_path: Stream results to this file, or None for in-memory.
        banner: Runner name for the progress banner ("NumPy batch", ...).
        label: Memory-log label identifying the calling runner.
        writer: A caller-owned writer to append results to, instead of
            output_path. LOCO shares one writer across its chromosome
            loop; the body neither opens nor closes it.
        snps_indices: Global indices restricting the tested SNP set, or None.
        hwe_threshold: HWE p-value threshold; 0.0 disables the filter.
        execution: Mode, dispatch, and conservative association geometry.
        compute_pve: Whether to run the null-REML PVE estimate.
        progress_label: Chunk-loop progress bar label.
        lambda_warning_prefix: Prefix for lambda-boundary warnings.

    Returns:
        LmmRunResult with associations (empty when output_path routed them
        to disk), n_tested, PVE, and the run's timing breakdown.
    """
    if output_path is not None and writer is not None:
        raise ValueError("pass output_path or writer, not both")

    maf_threshold = config.maf_threshold
    miss_threshold = config.miss_threshold
    l_min, l_max = config.l_min, config.l_max
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

    # === PASS 1: bind sample rows, SNP statistics, filtering, and chunks ===
    t_stats_start = time.perf_counter()
    genotypes = source.prepare(
        SampleBasis.from_mask(valid_mask),
        SnpFilterSpec(
            maf_threshold=maf_threshold,
            miss_threshold=miss_threshold,
            restrict_indices=snps_indices,
            hwe_threshold=hwe_threshold,
            restrict_label="SNP list filter",
        ),
    )
    if genotypes.n_unexpected > 0:
        logger.warning(
            f"Genotype validation: {genotypes.n_unexpected} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    n_filtered = genotypes.n_filtered
    association_execution = execution.tighten_after_filter(n_filtered)

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

        chunk_sink = (
            make_writer_sink(writer, lmm_mode, genotypes)
            if writer is not None
            else make_result_list_sink(all_results, lmm_mode, genotypes)
        )

        chunk_stats = run_lmm_chunk_source_numpy(
            genotypes=genotypes,
            chunk_sink=chunk_sink,
            execution=association_execution,
            prepared=prepared,
            config=config,
            options=ChunkRunOptions(
                progress_label=progress_label,
                lambda_warning_prefix=lambda_warning_prefix,
            ),
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
    max_chunk_size: int | None = None,
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
        max_chunk_size: Optional cap on association-pass chunk width.

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
    n_cvt = covariates.shape[1] if covariates is not None else 1
    execution = plan_association(
        n_samples,
        n_snps,
        requested="numpy",
        n_cvt=n_cvt,
        lmm_mode=config.lmm_mode,
        mem_budget=config.mem_budget,
        max_chunk_size=max_chunk_size,
        log_dispatch_choices=True,
    )
    return _run_lmm_association_numpy_planned(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        config=config,
        output_path=output_path,
        hwe_threshold=hwe_threshold,
        execution=execution,
        check_association_memory=config.check_memory and max_chunk_size is None,
    )


def _run_lmm_association_numpy_planned(
    *,
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    snp_info: list | SnpMeta,
    covariates: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    config: LmmConfig,
    output_path: Path | None,
    hwe_threshold: float,
    execution: ExecutableAssociationPlan,
    check_association_memory: bool,
) -> LmmRunResult:
    """Run the batch boundary with policy supplied by the pipeline."""
    if check_association_memory:
        quote = execution.price()
        logger.info(
            f"LMM memory: estimated {quote.total_peak_gb:.1f}GB, "
            f"available {quote.available_gb:.1f}GB"
        )
        memory.require(
            quote.total_peak_gb,
            quote.available_gb,
            f"LMM workflow with {execution.n_samples:,} samples x "
            f"{execution.n_snps_before_filter:,} SNPs",
            budget_gb=execution.mem_budget_gb,
        )

    snp_meta = (
        snp_info if isinstance(snp_info, SnpMeta) else SnpMeta.from_dicts(snp_info)
    )
    return _run_numpy_lmm(
        MatrixSource(genotypes, snp_meta),
        phenotypes=phenotypes,
        kinship=kinship,
        covariates=covariates,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        config=config,
        output_path=output_path,
        hwe_threshold=hwe_threshold,
        execution=execution,
        banner="NumPy batch",
        label="lmm_numpy",
    )
