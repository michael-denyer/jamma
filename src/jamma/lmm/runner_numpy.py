"""Shared NumPy LMM run body and the in-memory batch runner.

One body: ``run_association`` runs a bounded phenotype group over prepared
genotypes and one rotated basis. The pipeline calls it once per group.
``run_single`` runs one phenotype as a group of one: it prepares genotypes,
decomposes the kinship when needed, and hands the body one run. The batch and
streaming entries and every LOCO chromosome go through ``run_single``.
"""

from __future__ import annotations

import contextlib
import gc
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
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
from jamma.lmm.assoc_output import (
    AssocResult,
    ChunkSink,
    IncrementalAssocWriter,
    make_result_list_sink,
    make_writer_sink,
)
from jamma.lmm.association_plan import (
    ExecutableAssociationPlan,
    plan_association,
)
from jamma.lmm.chunk_runner_numpy import (
    PhenotypeChunkJob,
    RawLmmChunk,
    run_lmm_chunk_source_numpy_group,
)
from jamma.lmm.genotype_source import (
    GenotypeSource,
    PreparedGenotypes,
    SampleBasis,
    bind_prepared_genotypes,
)
from jamma.lmm.prepare_common import (
    AnalysedPhenotype,
    EigenInput,
    RotatedBasis,
    _build_covariate_matrix,
    _eigendecompose_or_reuse,
    fit_null,
    parse_eigen_input,
    restrict_eigen_input,
    rotate_basis,
)
from jamma.lmm.schema import (
    DEFAULT_LMM_CONFIG,
    MODE_SPECS,
    LmmConfig,
    LmmRunResult,
    ModeSpec,
    SnpInfoRecord,
    SnpMeta,
)


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


AssocDestination = Path | IncrementalAssocWriter | list[AssocResult]
"""Where one phenotype's rows go: a file this run opens, a caller-owned
writer it appends to, or a list it collects in memory."""


@dataclass(frozen=True, slots=True)
class PhenotypeRun:
    """One phenotype over the analysed samples and its result destination."""

    phenotypes: np.ndarray
    destination: AssocDestination


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
                yield rows[:, start:end], start, end

        stats = collect_snp_stats_from_chunks(
            _stat_chunks(),
            n_snps=n_snps,
            n_samples=n_samples,
            global_indices=np.arange(n_snps, dtype=np.intp),
            include_hwe=filters.hwe_threshold > 0,
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


def prepare_genotypes(
    source: GenotypeSource, spec: LmmRunSpec, sample_basis: SampleBasis
) -> PreparedGenotypes:
    """Collect and filter phenotype-independent genotype data once."""
    config = spec.config
    genotypes = source.prepare(
        sample_basis,
        SnpFilterSpec(
            maf_threshold=config.maf_threshold,
            miss_threshold=config.miss_threshold,
            restrict_indices=spec.snps_indices,
            hwe_threshold=spec.hwe_threshold,
            restrict_label="SNP list filter",
        ),
    )
    if genotypes.n_unexpected > 0:
        logger.warning(
            f"Genotype validation: {genotypes.n_unexpected} values outside "
            "expected range {0, 1, 2, NaN}"
        )
    return genotypes


def _publish_empty(destination: AssocDestination, mode: ModeSpec) -> LmmRunResult:
    """Give a file destination its header row when no SNP passed filtering."""
    if isinstance(destination, Path):
        with IncrementalAssocWriter(destination, mode):
            pass
    return LmmRunResult(associations=[], n_tested=0)


def _chunk_sink(
    destination: AssocDestination,
    stack: contextlib.ExitStack,
    mode: ModeSpec,
    genotypes: PreparedGenotypes,
) -> ChunkSink:
    if isinstance(destination, list):
        return make_result_list_sink(destination, mode, genotypes)
    if isinstance(destination, Path):
        destination = stack.enter_context(IncrementalAssocWriter(destination, mode))
    return make_writer_sink(destination, genotypes)


def run_association(
    genotypes: PreparedGenotypes,
    spec: LmmRunSpec,
    basis: RotatedBasis,
    runs: Sequence[PhenotypeRun],
) -> GroupedLmmRunResult:
    """Run a bounded phenotype group through one genotype chunk stream.

    Every run shares ``basis``. A ``Path`` destination receives a header even
    when no SNP passed filtering; a caller-owned writer or list is left alone.

    Returns:
        One result per run, in order. ``associations`` is the destination list
        for an in-memory run and empty otherwise.
    """
    if not runs:
        raise ValueError("at least one prepared phenotype run is required")
    if len(runs) > spec.execution.phenotype_group_size:
        raise ValueError(
            "phenotype group exceeds the execution plan's priced capacity: "
            f"got {len(runs)}, limit {spec.execution.phenotype_group_size}"
        )

    config = spec.config
    mode = MODE_SPECS[config.lmm_mode]
    if genotypes.n_filtered == 0:
        return GroupedLmmRunResult(
            tuple(_publish_empty(run.destination, mode) for run in runs),
            rotation_s=0.0,
        )

    fits = tuple(
        fit_null(basis, run.phenotypes, config, compute_pve=spec.compute_pve)
        for run in runs
    )
    with contextlib.ExitStack() as stack:
        jobs = tuple(
            PhenotypeChunkJob(fit, _chunk_sink(run.destination, stack, mode, genotypes))
            for fit, run in zip(fits, runs, strict=True)
        )
        grouped = run_lmm_chunk_source_numpy_group(
            genotypes=genotypes,
            basis=basis,
            jobs=jobs,
            config=config,
            dispatch=spec.execution.dispatch,
            chunks=spec.execution.conservative_chunks.narrow(genotypes.n_filtered),
            workspace=spec.execution.workspace,
            progress_label=spec.labels.progress_label,
            lambda_warning_prefix=spec.labels.lambda_warning_prefix,
        )

    results = tuple(
        LmmRunResult(
            associations=run.destination if isinstance(run.destination, list) else [],
            n_tested=timing.processed,
            pve=fit.pve,
            pve_se=fit.pve_se,
            timing=timing,
        )
        for run, fit, timing in zip(runs, fits, grouped.phenotypes, strict=True)
    )
    return GroupedLmmRunResult(results=results, rotation_s=grouped.rotation_s)


def run_single(
    source: GenotypeSource,
    spec: LmmRunSpec,
    samples: AnalysedPhenotype,
    eigen_input: EigenInput,
    destination: AssocDestination,
) -> LmmRunResult:
    """Run one phenotype as a group of one.

    SNP statistics run before the eigendecomposition, and the decomposition is
    skipped when every SNP is filtered out.

    Args:
        source: Genotype provider over the rows ``samples.valid_mask`` indexes.
        spec: The run's policy.
        samples: The phenotype and covariates over the analysed samples.
        eigen_input: Kinship or eigenpairs over the analysed samples. A kinship
            is consumed by the eigendecomposition.
        destination: Output file, caller-owned writer, or in-memory list.

    Returns:
        LmmRunResult whose timing carries the whole genotype rotation time.
    """
    config = spec.config
    labels = spec.labels
    show_progress = config.show_progress
    start_time = time.perf_counter()
    n_snps = source.n_snps

    if show_progress:
        logger.info(f"Performing LMM Association Test ({labels.banner})")
        logger.info(f"  Total individuals: {samples.valid_mask.shape[0]:,}")
        logger.info(f"  Analyzed individuals: {samples.n_samples:,}")
        logger.info(f"  Total SNPs: {n_snps:,}")
        logger.info(f"  Lambda range: [{config.l_min:.2e}, {config.l_max:.2e}]")

    t_stats_start = time.perf_counter()
    genotypes = prepare_genotypes(
        source, spec, SampleBasis.from_mask(samples.valid_mask)
    )
    if show_progress:
        logger.info(f"  Analyzed SNPs: {genotypes.n_filtered:,}")

    if genotypes.n_filtered == 0:
        logger.warning(
            f"All {n_snps} SNPs filtered out (MAF>{config.maf_threshold}, "
            f"miss<{config.miss_threshold}). No association tests to run. "
            f"Consider relaxing --maf or --miss thresholds."
        )
        return _publish_empty(destination, MODE_SPECS[config.lmm_mode])
    t_stats_end = time.perf_counter()

    t_eigen_start = time.perf_counter()
    W, _n_cvt = _build_covariate_matrix(samples.covariates, samples.n_samples)
    eigenvalues, U = _eigendecompose_or_reuse(
        eigen_input, show_progress, labels.label, check_memory=config.check_memory
    )
    del eigen_input
    basis = rotate_basis(eigenvalues, U, W)
    gc.collect()
    t_eigen_end = time.perf_counter()

    grouped = run_association(
        genotypes, spec, basis, (PhenotypeRun(samples.phenotypes, destination),)
    )
    result = grouped.results[0]
    timing = replace(result.timing, rotation_s=grouped.rotation_s)

    if show_progress:
        log_memory_snapshot(f"{labels.label}:after_association")
        elapsed = time.perf_counter() - start_time
        t_stats = t_stats_end - t_stats_start
        t_eigen = t_eigen_end - t_eigen_start
        accounted = (
            t_stats
            + t_eigen
            + timing.rotation_s
            + timing.compute_s
            + timing.result_write_s
        )
        logger.info("Timing breakdown:")
        logger.info(f"  SNP statistics:      {t_stats:.2f}s")
        logger.info(f"  Setup (eigen+null):  {t_eigen:.2f}s")
        logger.info(f"  UT@G rotation:       {timing.rotation_s:.2f}s")
        logger.info(f"  NumPy compute:       {timing.compute_s:.2f}s")
        logger.info(f"  Result write:        {timing.result_write_s:.2f}s")
        logger.info("  ----")
        logger.info(f"  Accounted:           {accounted:.2f}s")
        logger.info(f"  Total:               {elapsed:.2f}s")
        if isinstance(destination, Path):
            logger.info(f"Wrote {result.n_tested:,} results to {destination}")
        logger.info(f"LMM Association completed in {elapsed:.2f}s")

    return replace(result, timing=timing)


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
            pre-computed eigenvalues/eigenvectors are provided. Consumed:
            centred in place, then overwritten by the eigendecomposition
            (zeroed on the NumPy fallback). Must be writeable; pass
            kinship.copy() to keep the original matrix.
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
        ValueError: If a phenotype is infinite, only one of eigenvalues and
            eigenvectors is provided, or no valid samples remain after filtering.
    """
    n_input_samples, n_snps = genotypes.shape
    samples = AnalysedPhenotype.from_inputs(phenotypes, covariates)
    n_samples = samples.n_samples
    execution = plan_association(
        n_samples,
        n_snps,
        config=config,
        backend="numpy",
        n_cvt=samples.n_cvt,
        n_input_samples=n_input_samples,
        max_chunk_size=max_chunk_size,
    )
    if config.check_memory and max_chunk_size is None:
        quote = execution.price(eigen=None)
        available_gb = memory.available_ram_gb()
        logger.info(
            f"LMM memory: estimated {quote.association_gb:.1f}GB, "
            f"available {available_gb:.1f}GB"
        )
        memory.require(
            quote.association_gb,
            available_gb,
            f"LMM workflow with {n_samples:,} samples x {n_snps:,} SNPs",
            budget_gb=execution.mem_budget_gb,
        )

    snp_meta = (
        snp_info if isinstance(snp_info, SnpMeta) else SnpMeta.from_dicts(snp_info)
    )
    return run_single(
        MatrixSource(genotypes, snp_meta),
        LmmRunSpec(config=config, execution=execution, hwe_threshold=hwe_threshold),
        samples,
        restrict_eigen_input(
            parse_eigen_input(kinship, eigenvalues, eigenvectors), samples.valid_mask
        ),
        output_path if output_path is not None else [],
    )
