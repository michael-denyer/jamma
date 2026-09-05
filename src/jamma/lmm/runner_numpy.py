"""Shared NumPy LMM run body and the in-memory batch runner.

One body, many sources: ``run_lmm_association`` drives per-SNP statistics,
filtering, preparation, and the chunk engine over a ``GenotypeSource`` under
one ``LmmRunSpec``. The batch and streaming entries build a source and a
spec from their public arguments; the pipeline builds one source per run;
LOCO builds one per chromosome.
"""

from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from loguru import logger

from jamma.core import memory
from jamma.core.memory_snapshot import log_memory_snapshot
from jamma.core.snp_filter import _SNP_STATS_CHUNK_SIZE
from jamma.core.snp_stats import (
    SnpFilterSpec,
    SnpSelection,
    collect_snp_stats_from_chunks,
)
from jamma.lmm.association_plan import (
    ExecutableAssociationPlan,
    plan_association,
)
from jamma.lmm.chunk_runner_numpy import (
    PhenotypeChunkJob,
    RawLmmChunk,
    run_lmm_chunk_source_numpy,
    run_lmm_chunk_source_numpy_group,
)
from jamma.lmm.genotype_source import (
    GenotypeSource,
    PreparedGenotypes,
    SampleBasis,
    bind_prepared_genotypes,
)
from jamma.lmm.io import IncrementalAssocWriter
from jamma.lmm.prepare_common import (
    EigenInput,
    EigenPairs,
    PreparedCovariates,
    _build_covariate_matrix,
    compute_valid_mask,
    parse_eigen_input,
    prepare_lmm_run,
    validate_runner_inputs,
)
from jamma.lmm.results import make_result_list_sink, make_writer_sink
from jamma.lmm.schema import (
    DEFAULT_LMM_CONFIG,
    LmmConfig,
    LmmRunResult,
    SnpInfoRecord,
    SnpMeta,
)
from jamma.lmm.schema import TEST_TYPE_MAP as _TEST_TYPE_MAP
from jamma.lmm.stats import AssocResult


@dataclass(frozen=True, slots=True)
class RunLabels:
    """How one runner names itself in the log and the progress bar."""

    banner: str
    label: str
    progress_label: str = "LMM association"
    lambda_warning_prefix: str = ""


BATCH_LABELS = RunLabels(banner="NumPy batch", label="lmm_numpy")
STREAMING_LABELS = RunLabels(
    banner="NumPy streaming",
    label="lmm_numpy_streaming",
    progress_label="LMM association (streaming)",
)
LOCO_LABELS = RunLabels(
    banner="NumPy LOCO", label="lmm_loco", lambda_warning_prefix="LOCO "
)


@dataclass(frozen=True, slots=True)
class LmmRunSpec:
    """Everything one association run decides before it reads a genotype.

    Attributes:
        config: Thresholds, lambda bounds and grid, test type, progress.
        execution: Mode, dispatch, and conservative association geometry.
        snps_indices: Global indices restricting the tested SNP set, or
            None. Joins the MAF, missingness and HWE filters in the body,
            so every source applies the restriction at the same layer.
        hwe_threshold: HWE p-value threshold; 0.0 disables the filter.
        compute_pve: Whether to run the null-REML PVE estimate.
        labels: The runner's banner and progress-bar wording.
    """

    config: LmmConfig
    execution: ExecutableAssociationPlan
    snps_indices: np.ndarray | None = None
    hwe_threshold: float = 0.0
    compute_pve: bool = True
    labels: RunLabels = BATCH_LABELS


@dataclass(frozen=True, slots=True)
class PreparedPhenotypeSpec:
    """One already sample-filtered phenotype and its output destination."""

    phenotypes: np.ndarray
    output_path: Path


class GroupedLmmRunResult(NamedTuple):
    """Phenotype results plus the measured shared genotype rotation time."""

    results: tuple[LmmRunResult, ...]
    rotation_s: float


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

        def _iter_chunks(
            selection: SnpSelection, chunk_size: int
        ) -> Iterator[RawLmmChunk]:
            selected_columns = selection.local_indices
            n_filtered = len(selected_columns)
            geno_buf = np.empty((rows.shape[0], chunk_size), dtype=np.float64)
            for chunk_start in range(0, n_filtered, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_filtered)
                actual_len = chunk_end - chunk_start
                geno_chunk = (
                    geno_buf
                    if actual_len == chunk_size
                    else np.empty((rows.shape[0], actual_len), dtype=np.float64)
                )
                geno_chunk[:] = rows[:, selected_columns[chunk_start:chunk_end]]
                yield RawLmmChunk(geno_chunk, chunk_start, chunk_end)

        return bind_prepared_genotypes(
            snp_meta=self._snp_meta,
            stats=stats,
            filters=filters,
            sample_basis=samples,
            chunk_source=_iter_chunks,
        )


def run_lmm_association(
    source: GenotypeSource,
    spec: LmmRunSpec,
    *,
    phenotypes: np.ndarray,
    eigen_input: EigenInput,
    covariates: np.ndarray | None,
    output_path: Path | None = None,
    writer: IncrementalAssocWriter | None = None,
) -> LmmRunResult:
    """Run one NumPy LMM association over any genotype source.

    Statistics, filtering, eigen preparation, the chunk loop, and result
    routing are identical for every runner; only where genotypes come from
    differs, and that lives in ``source``, and what policy the run follows,
    which lives in ``spec``.

    Args:
        source: Genotype provider; owns sample-row filtering and stats dtype.
        spec: The run's policy: config, execution plan, SNP restriction,
            HWE threshold, PVE choice, and labels.
        phenotypes: Phenotype vector (n_samples_total,), NaN for missing.
        eigen_input: Kinship matrix or complete pre-computed eigenpairs.
        covariates: Covariate matrix or None for intercept-only.
        output_path: Stream results to this file, or None for in-memory.
        writer: A caller-owned writer to append results to, instead of
            output_path. LOCO shares one writer across its chromosome
            loop; the body neither opens nor closes it.

    Returns:
        LmmRunResult with associations (empty when output_path routed them
        to disk), n_tested, PVE, and the run's timing breakdown.
    """
    return _run_lmm_association(
        source,
        spec,
        phenotypes=phenotypes,
        eigen_input=eigen_input,
        covariates=covariates,
        output_path=output_path,
        writer=writer,
        prepared_genotypes=None,
    )


def run_lmm_association_prepared(
    genotypes: PreparedGenotypes,
    spec: LmmRunSpec,
    *,
    phenotypes: np.ndarray,
    eigen_input: EigenInput,
    covariates: np.ndarray | None,
    output_path: Path | None = None,
    writer: IncrementalAssocWriter | None = None,
    prepared_covariates: PreparedCovariates | None = None,
) -> LmmRunResult:
    """Run one phenotype over an already prepared genotype selection.

    The prepared object's exact sample basis must match the valid-sample mask
    derived from this phenotype and its covariates. This keeps shared
    preparation safe when two masks have the same number of samples at
    different source positions.
    """
    return _run_lmm_association(
        None,
        spec,
        phenotypes=phenotypes,
        eigen_input=eigen_input,
        covariates=covariates,
        output_path=output_path,
        writer=writer,
        prepared_genotypes=genotypes,
        prepared_covariates=prepared_covariates,
    )


def prepare_genotypes(
    source: GenotypeSource, spec: LmmRunSpec, sample_basis: SampleBasis
) -> PreparedGenotypes:
    """Collect and filter phenotype-independent genotype data once."""
    config = spec.config
    return source.prepare(
        sample_basis,
        SnpFilterSpec(
            maf_threshold=config.maf_threshold,
            miss_threshold=config.miss_threshold,
            restrict_indices=spec.snps_indices,
            hwe_threshold=spec.hwe_threshold,
            restrict_label="SNP list filter",
        ),
    )


def run_lmm_association_group_prepared(
    genotypes: PreparedGenotypes,
    spec: LmmRunSpec,
    runs: tuple[PreparedPhenotypeSpec, ...],
    *,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    prepared_covariates: PreparedCovariates,
) -> GroupedLmmRunResult:
    """Run a bounded phenotype group through one genotype chunk stream."""
    if not runs:
        raise ValueError("at least one prepared phenotype run is required")
    if len(runs) > spec.execution.phenotype_group_size:
        raise ValueError(
            "phenotype group exceeds the execution plan's priced capacity: "
            f"got {len(runs)}, limit {spec.execution.phenotype_group_size}"
        )

    config = spec.config
    eigen_input = EigenPairs(eigenvalues, eigenvectors)
    lmm_mode = config.lmm_mode
    if genotypes.n_filtered == 0:
        for run in runs:
            with IncrementalAssocWriter(
                run.output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
            ):
                pass
        return GroupedLmmRunResult(
            tuple(LmmRunResult([], 0) for _run in runs), rotation_s=0.0
        )

    prepared_runs = []
    for run in runs:
        if run.phenotypes.shape != (genotypes.analyzed_sample_count,):
            raise ValueError(
                "prepared phenotype length does not match genotype sample basis: "
                f"got {run.phenotypes.shape}, expected "
                f"({genotypes.analyzed_sample_count},)"
            )
        if not np.all(np.isfinite(run.phenotypes)):
            raise ValueError("prepared phenotypes must contain only finite values")
        prepared_runs.append(
            prepare_lmm_run(
                eigen_input=eigen_input,
                phenotypes=run.phenotypes,
                W=prepared_covariates.W,
                n_cvt=prepared_covariates.n_cvt,
                l_min=config.l_min,
                l_max=config.l_max,
                show_progress=config.show_progress,
                check_memory=config.check_memory,
                label=spec.labels.label,
                compute_pve=spec.compute_pve,
                rotated_covariates=prepared_covariates.UtW,
            )
        )

    chunks = spec.execution.conservative_chunks.narrow(genotypes.n_filtered)
    with contextlib.ExitStack() as stack:
        writers = tuple(
            stack.enter_context(
                IncrementalAssocWriter(
                    run.output_path, test_type=_TEST_TYPE_MAP[lmm_mode]
                )
            )
            for run in runs
        )
        jobs = tuple(
            PhenotypeChunkJob(
                prepared=prepared,
                chunk_sink=make_writer_sink(writer, lmm_mode, genotypes),
                config=config,
                lambda_warning_prefix=spec.labels.lambda_warning_prefix,
            )
            for prepared, writer in zip(prepared_runs, writers, strict=True)
        )
        grouped = run_lmm_chunk_source_numpy_group(
            genotypes=genotypes,
            jobs=jobs,
            dispatch=spec.execution.dispatch,
            chunks=chunks,
            workspace=spec.execution.workspace,
            progress_label=spec.labels.progress_label,
        )

    results = tuple(
        LmmRunResult(
            associations=[],
            n_tested=timing.processed,
            pve=prepared.pve,
            pve_se=prepared.pve_se,
            timing=timing,
        )
        for prepared, timing in zip(prepared_runs, grouped.phenotypes, strict=True)
    )
    return GroupedLmmRunResult(results=results, rotation_s=grouped.rotation_s)


def _run_lmm_association(
    source: GenotypeSource | None,
    spec: LmmRunSpec,
    *,
    phenotypes: np.ndarray,
    eigen_input: EigenInput,
    covariates: np.ndarray | None,
    output_path: Path | None,
    writer: IncrementalAssocWriter | None,
    prepared_genotypes: PreparedGenotypes | None,
    prepared_covariates: PreparedCovariates | None = None,
) -> LmmRunResult:
    """Shared implementation for source-owned and caller-prepared runs."""
    if output_path is not None and writer is not None:
        raise ValueError("pass output_path or writer, not both")

    config = spec.config
    execution = spec.execution
    labels = spec.labels
    maf_threshold = config.maf_threshold
    miss_threshold = config.miss_threshold
    l_min, l_max = config.l_min, config.l_max
    check_memory = config.check_memory
    show_progress = config.show_progress
    lmm_mode = config.lmm_mode

    start_time = time.perf_counter()
    n_samples_total = phenotypes.shape[0]
    if source is not None:
        n_snps = source.n_snps
    elif prepared_genotypes is not None:
        n_snps = len(prepared_genotypes.snp_meta)
    else:
        raise RuntimeError("source or prepared genotypes are required")

    setup = validate_runner_inputs(phenotypes, eigen_input, covariates)
    phenotypes = setup.phenotypes
    eigen_input = setup.eigen_input
    covariates = setup.covariates
    valid_mask = setup.valid_mask
    n_samples = phenotypes.shape[0]

    if show_progress:
        logger.info(f"Performing LMM Association Test ({labels.banner})")
        logger.info(f"  Total individuals: {n_samples_total:,}")
        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.info(f"  Lambda range: [{l_min:.2e}, {l_max:.2e}]")

    # === PASS 1: bind sample rows, SNP statistics, filtering, and chunks ===
    t_stats_start = time.perf_counter()
    sample_basis = SampleBasis.from_mask(valid_mask)
    if prepared_genotypes is None:
        if source is None:
            raise RuntimeError("source is required when genotypes are not prepared")
        genotypes = prepare_genotypes(source, spec, sample_basis)
    else:
        genotypes = prepared_genotypes
        prepared_basis = genotypes.sample_basis
        if (
            prepared_basis.source_row_count != sample_basis.source_row_count
            or not np.array_equal(prepared_basis.positions, sample_basis.positions)
        ):
            raise ValueError(
                "prepared genotype sample basis does not match phenotype and "
                "covariate valid-sample mask"
            )
    if genotypes.n_unexpected > 0:
        logger.warning(
            f"Genotype validation: {genotypes.n_unexpected} values outside "
            f"expected range {{0, 1, 2, NaN}}"
        )

    n_filtered = genotypes.n_filtered
    tightened_chunks = execution.conservative_chunks.narrow(n_filtered)

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
    if prepared_covariates is None:
        W, n_cvt = _build_covariate_matrix(covariates, n_samples)
        rotated_covariates = None
    else:
        W = prepared_covariates.W
        n_cvt = prepared_covariates.n_cvt
        rotated_covariates = prepared_covariates.UtW
        if W.shape[0] != n_samples:
            raise ValueError(
                "prepared covariate sample count does not match phenotype: "
                f"got {W.shape[0]} and {n_samples}"
            )
    prepared = prepare_lmm_run(
        eigen_input=eigen_input,
        phenotypes=phenotypes,
        W=W,
        n_cvt=n_cvt,
        l_min=l_min,
        l_max=l_max,
        show_progress=show_progress,
        check_memory=check_memory,
        label=labels.label,
        compute_pve=spec.compute_pve,
        rotated_covariates=rotated_covariates,
    )
    del eigen_input
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
            dispatch=execution.dispatch,
            chunks=tightened_chunks,
            workspace=execution.workspace,
            prepared=prepared,
            config=config,
            progress_label=labels.progress_label,
            lambda_warning_prefix=labels.lambda_warning_prefix,
        )

        if show_progress:
            log_memory_snapshot(f"{labels.label}:after_association")

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
            timing=chunk_stats,
        )


def run_lmm_association_numpy(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray | None,
    snp_info: Sequence[SnpInfoRecord] | SnpMeta,
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
    n_input_samples, n_snps = genotypes.shape
    valid_mask = compute_valid_mask(phenotypes, covariates)
    n_samples = int(np.count_nonzero(valid_mask))
    n_cvt = covariates.shape[1] if covariates is not None else 1
    execution = plan_association(
        n_samples,
        n_snps,
        n_input_samples=n_input_samples,
        requested="numpy",
        n_cvt=n_cvt,
        lmm_mode=config.lmm_mode,
        n_grid=config.n_grid,
        n_refine=config.n_refine,
        mem_budget=config.mem_budget,
        max_chunk_size=max_chunk_size,
        log_dispatch_choices=True,
    )
    if config.check_memory and max_chunk_size is None:
        quote = execution.price()
        available_gb = memory.available_ram_gb()
        logger.info(
            f"LMM memory: estimated {quote.total_peak_gb:.1f}GB, "
            f"available {available_gb:.1f}GB"
        )
        memory.require(
            quote.total_peak_gb,
            available_gb,
            f"LMM workflow with {n_samples:,} samples x {n_snps:,} SNPs",
            budget_gb=execution.mem_budget_gb,
        )

    snp_meta = (
        snp_info if isinstance(snp_info, SnpMeta) else SnpMeta.from_dicts(snp_info)
    )
    return run_lmm_association(
        MatrixSource(genotypes, snp_meta),
        LmmRunSpec(config=config, execution=execution, hwe_threshold=hwe_threshold),
        phenotypes=phenotypes,
        eigen_input=parse_eigen_input(kinship, eigenvalues, eigenvectors),
        covariates=covariates,
        output_path=output_path,
    )
