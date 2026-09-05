"""Shared NumPy LMM chunk engine (orchestrator).

Owns the per-run chunk loop that the batch, disk-streaming, and LOCO NumPy
runners share. It consumes prepared genotypes, final dispatch, and chunk
geometry, drives the optional rotate/compute pipeline, imputes missing genotypes,
rotates via
``jlinalg.dgemm``, prepares Uab inputs, dispatches compute, and accumulates
per-chunk diagnostics. Callers provide one bound genotype session and a result
sink; everything after that boundary is owned here.

The kernel and the state it needs live in ``chunk_kernel``, chunk sizing in
``chunk_sizing``, and the overlapped driver in ``chunk_pipeline``; this module
wires them together.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from loguru import logger

from jamma import jlinalg
from jamma.core.estimates import estimate_lmm_seconds
from jamma.core.progress import progress_iterator
from jamma.core.threading import (
    blas_threads,
    get_c_extension_thread_count,
)
from jamma.lmm import accel
from jamma.lmm.chunk_kernel import Kernel, RunInvariants, make_kernel
from jamma.lmm.chunk_pipeline import _drive_pipeline, plan_thread_budget
from jamma.lmm.chunk_sizing import LmmChunkPlan
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.genotype_source import PreparedGenotypes
from jamma.lmm.impute import impute_missing_inplace
from jamma.lmm.likelihood import reset_p_yy_warned
from jamma.lmm.prepare_common import PreparedLmmRun
from jamma.lmm.results import (
    ChunkSink,
    count_lambda_boundary_hits,
    log_lambda_boundary_warning,
)
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import ChunkRunStats, LmmConfig
from jamma.lmm.uab import batch_compute_uab_numpy
from jamma.lmm.workspace import WorkspaceSpec


class LmmChunkRange(NamedTuple):
    """Contiguous half-open range in the filtered SNP coordinate space."""

    filtered_start: int
    filtered_end: int

    @property
    def length(self) -> int:
        return self.filtered_end - self.filtered_start

    def validate_next(self, expected_start: int, n_filtered: int) -> None:
        if self.filtered_start != expected_start:
            raise RuntimeError(
                "raw LMM chunks must be contiguous in filtered SNP order: "
                f"expected start {expected_start}, got {self.filtered_start}"
            )
        if self.filtered_end <= self.filtered_start:
            raise RuntimeError(
                "raw LMM chunks must be non-empty after empty chunks are skipped: "
                f"got [{self.filtered_start}, {self.filtered_end})"
            )
        if self.filtered_end > n_filtered:
            raise RuntimeError(
                "raw LMM chunk extends past filtered SNP count: "
                f"end {self.filtered_end}, n_filtered {n_filtered}"
            )


class RawLmmChunk(NamedTuple):
    """Raw genotype chunk handed to the shared NumPy LMM chunk runner.

    ``genotypes`` must be a mutable float64 array with shape
    ``(n_samples, filtered_end - filtered_start)``. Chunk ranges are in the
    filtered SNP coordinate space and must arrive contiguously.
    """

    genotypes: np.ndarray
    filtered_start: int
    filtered_end: int

    @property
    def filtered_range(self) -> LmmChunkRange:
        return LmmChunkRange(self.filtered_start, self.filtered_end)


class _PreparedLmmChunk(NamedTuple):
    """A rotated chunk ready for compute, tagged with its filtered SNP range.

    ``data`` is raw ``utg_t`` with shape ``(n_snps, n_samples)``. Every
    phenotype consumer sees this same rotation. A fallback consumer derives
    its phenotype-dependent Uab immediately before its sequential compute.
    """

    data: np.ndarray
    filtered_range: LmmChunkRange


@dataclass(slots=True)
class _PhenotypeConsumer:
    """One phenotype's kernel, sink, diagnostics, and timing."""

    inv: RunInvariants
    kernel: Kernel
    chunk_sink: ChunkSink
    processed: int = 0
    compute_s: float = 0.0
    result_write_s: float = 0.0
    nan_counts: dict[str, int] = field(default_factory=dict)
    n_at_lmin: int = 0
    n_at_lmax: int = 0

    def kernel_input(self, utg_t: np.ndarray) -> np.ndarray:
        if self.inv.dispatch.feeds_raw_utg:
            return utg_t
        return batch_compute_uab_numpy(
            self.inv.n_cvt, self.inv.UtW, self.inv.Uty, utg_t
        )

    def compute_and_write(self, prepared: _PreparedLmmChunk, omp_threads: int) -> None:
        chunk_range = prepared.filtered_range
        filtered_start = chunk_range.filtered_start
        actual_len = chunk_range.length
        if filtered_start != self.processed:
            raise RuntimeError(
                "prepared LMM chunks reached compute out of order: "
                f"expected start {self.processed}, got {filtered_start}"
            )

        t_compute_start = time.perf_counter()
        cr = self.kernel.compute_chunk(
            self.kernel_input(prepared.data), omp_threads, self.processed
        )
        self.compute_s += time.perf_counter() - t_compute_start

        t_write_start = time.perf_counter()
        chunk_arrays = {
            key: cr[key][:actual_len] for key in _RESULT_FIELDS[self.inv.lmm_mode]
        }
        chunk_lmin, chunk_lmax = count_lambda_boundary_hits(
            self.inv.lmm_mode, chunk_arrays, self.inv.l_min, self.inv.l_max
        )
        self.n_at_lmin += chunk_lmin
        self.n_at_lmax += chunk_lmax
        for key, arr in chunk_arrays.items():
            if arr.dtype.kind != "f":
                continue
            n_nan = int(np.count_nonzero(np.isnan(arr)))
            if n_nan > 0:
                self.nan_counts[key] = self.nan_counts.get(key, 0) + n_nan

        self.chunk_sink(chunk_arrays, filtered_start, chunk_range.filtered_end)
        self.processed += actual_len
        self.result_write_s += time.perf_counter() - t_write_start


class GroupedChunkRunStats(NamedTuple):
    """Per-phenotype compute/write timing plus one shared rotation timing."""

    phenotypes: tuple[ChunkRunStats, ...]
    rotation_s: float


@dataclass(frozen=True, slots=True)
class PhenotypeChunkJob:
    """Prepared phenotype state and destination for a shared chunk pass."""

    prepared: PreparedLmmRun
    chunk_sink: ChunkSink
    config: LmmConfig
    lambda_warning_prefix: str = ""


class _ChunkEngine:
    """The chunk loop's state: its buffers, its thread split, its counters.

    ``prepare`` rotates each chunk and ``compute_and_write`` consumes it. Both
    share the buffers and counters here. The OpenMP allocation is fixed by the
    per-run thread plan before the pipeline starts.
    """

    def __init__(
        self,
        *,
        consumers: tuple[_PhenotypeConsumer, ...],
        U: np.ndarray,
        filtered_means: np.ndarray,
        raw_chunks: Iterator[RawLmmChunk],
        chunk_size: int,
        n_buffers: int,
        omp_threads: int,
    ) -> None:
        if not consumers:
            raise ValueError("chunk engine requires at least one phenotype consumer")
        self.consumers = consumers
        self.inv = consumers[0].inv
        self.kernel = consumers[0].kernel
        self.U = U
        self.filtered_means = filtered_means
        self.raw_chunks = raw_chunks
        self.chunk_size = chunk_size

        # Fixed by plan_thread_budget before the engine is created.
        self.omp_threads = omp_threads

        n_samples = self.inv.n_samples
        self.utg_bufs = [
            np.empty((chunk_size, n_samples), dtype=np.float64)
            for _ in range(n_buffers)
        ]

        self.chunk_counter = 0
        self.next_expected_start = 0

    def prepare(self) -> _PreparedLmmChunk | None:
        """Pull the next raw chunk, impute it, rotate it, and shape it for the kernel.

        Returns None once the source is exhausted. Runs on the background thread
        during a pipelined run, so it touches only its own buffers and counters.
        """
        raw = self._next_non_empty_chunk()
        if raw is None:
            return None

        chunk_range = raw.filtered_range
        chunk_range.validate_next(self.next_expected_start, self.inv.n_filtered)
        actual_len = chunk_range.length
        if raw.genotypes.shape != (self.inv.n_samples, actual_len):
            raise ValueError(
                "raw LMM chunk shape mismatch: expected "
                f"({self.inv.n_samples}, {actual_len}), got {raw.genotypes.shape}"
            )
        self.next_expected_start = chunk_range.filtered_end

        buf_idx = self.chunk_counter % len(self.utg_bufs)
        self.chunk_counter += 1

        impute_missing_inplace(
            raw.genotypes,
            self.filtered_means[chunk_range.filtered_start : chunk_range.filtered_end],
        )

        utg_out = self.utg_bufs[buf_idx][:actual_len, :]
        utg_t = jlinalg.dgemm(raw.genotypes, self.U, transa="T", out=utg_out)

        return _PreparedLmmChunk(utg_t, chunk_range)

    def _next_non_empty_chunk(self) -> RawLmmChunk | None:
        """Skip zero-length chunks, checking each keeps the contiguity contract."""
        raw = next(self.raw_chunks, None)
        while raw is not None and raw.filtered_end <= raw.filtered_start:
            empty_range = raw.filtered_range
            if empty_range.filtered_start != empty_range.filtered_end:
                raise RuntimeError(
                    "raw LMM chunk has an invalid empty range: "
                    f"[{empty_range.filtered_start}, {empty_range.filtered_end})"
                )
            if empty_range.filtered_start != self.next_expected_start:
                raise RuntimeError(
                    "empty raw LMM chunks must preserve contiguous order: "
                    f"expected {self.next_expected_start}, "
                    f"got {empty_range.filtered_start}"
                )
            raw = next(self.raw_chunks, None)
        return raw

    def compute_and_write(self, prepared: _PreparedLmmChunk) -> None:
        """Run one shared rotation through every phenotype consumer."""
        for consumer in self.consumers:
            consumer.compute_and_write(prepared, self.omp_threads)

    @property
    def processed(self) -> int:
        return self.consumers[0].processed

    @property
    def compute_s(self) -> float:
        return self.consumers[0].compute_s

    @property
    def result_write_s(self) -> float:
        return self.consumers[0].result_write_s

    @property
    def nan_counts(self) -> dict[str, int]:
        return self.consumers[0].nan_counts

    @property
    def n_at_lmin(self) -> int:
        return self.consumers[0].n_at_lmin

    @property
    def n_at_lmax(self) -> int:
        return self.consumers[0].n_at_lmax


def run_lmm_chunk_source_numpy_group(
    *,
    genotypes: PreparedGenotypes,
    jobs: tuple[PhenotypeChunkJob, ...],
    dispatch: DispatchPath,
    chunks: LmmChunkPlan,
    workspace: WorkspaceSpec,
    progress_label: str = "LMM association",
) -> GroupedChunkRunStats:
    """Rotate each raw genotype chunk once for a bounded phenotype group.

    The prepared source owns aligned imputation means and raw chunks. This
    function consumes it with the selected dispatch and chunk geometry
    (``chunks``, already tightened to the filtered SNP count), then owns
    pipeline driving, eigen-rotation, Uab preparation, C/Python compute dispatch,
    diagnostics, and timing. Batch and LOCO use this path so their chunk compute
    behavior cannot drift.

    A result field that is all NaN (e.g. a non-PSD kinship matrix, or a phenotype
    made degenerate by collinear covariates) is surfaced via ``logger.warning``
    per field, not raised: an all-NaN result is a legitimate GEMMA-equivalent
    outcome for degenerate inputs, and a fatal abort would be both wrong there and
    sensitive to platform floating-point (the NaN fraction can differ by BLAS).
    """
    if not jobs:
        raise ValueError("at least one phenotype chunk job is required")
    reset_p_yy_warned()

    first_job = jobs[0]
    prepared = first_job.prepared
    config = first_job.config
    n_samples = prepared.n_samples
    n_filtered = genotypes.n_filtered
    l_min = config.l_min
    l_max = config.l_max
    show_progress = config.show_progress

    if genotypes.analyzed_sample_count != n_samples:
        raise ValueError(
            "prepared genotype sample count does not match prepared LMM run: "
            f"got {genotypes.analyzed_sample_count} and {n_samples}"
        )

    for job in jobs[1:]:
        if job.prepared.n_samples != n_samples:
            raise ValueError("all phenotype jobs must use the same sample count")
        if job.prepared.U is not prepared.U:
            raise ValueError("all phenotype jobs must share one eigenvector matrix")
        if job.config != config:
            raise ValueError("all phenotype jobs must use one LMM configuration")

    if n_filtered == 0:
        return GroupedChunkRunStats(
            tuple(ChunkRunStats() for _job in jobs), rotation_s=0.0
        )

    chunk_size = chunks.chunk_size
    n_chunks = chunks.n_chunks
    use_pipeline = chunks.use_pipeline

    if show_progress:
        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"  Processing in {n_chunks} chunks ({chunk_size:,} SNPs/chunk)"
            )

    threads = plan_thread_budget(
        n_samples=n_samples,
        omp_threads=get_c_extension_thread_count(accel.available(), accel.HAS_OPENMP),
        max_omp_threads=workspace.max_threads,
        use_pipeline=use_pipeline,
    )
    consumer_list = []
    for job in jobs:
        invariants = RunInvariants.build(dispatch, job.prepared, job.config, n_filtered)
        consumer_list.append(
            _PhenotypeConsumer(
                invariants,
                make_kernel(invariants, workspace),
                job.chunk_sink,
            )
        )
    consumers = tuple(consumer_list)

    engine = _ChunkEngine(
        consumers=consumers,
        U=prepared.U,
        filtered_means=genotypes.imputation_means,
        raw_chunks=genotypes.chunks(chunk_size),
        chunk_size=chunk_size,
        n_buffers=chunks.n_buffers,
        omp_threads=threads.omp,
    )

    rotation_s = 0.0
    if use_pipeline:
        rotation_s += _drive_pipeline(
            engine,
            n_chunks=n_chunks,
            rotation_threads=threads.rotation,
            n_samples=n_samples,
            n_filtered=n_filtered,
            show_progress=show_progress,
            progress_label=progress_label,
        )
    else:
        if show_progress and n_chunks > 1:
            chunk_iterator = progress_iterator(
                iter(range(n_chunks)),
                total=n_chunks,
                desc=progress_label,
                initial_eta_seconds=estimate_lmm_seconds(n_samples, n_filtered),
            )
        else:
            chunk_iterator = range(n_chunks)

        for _chunk_idx in chunk_iterator:
            t_rot_start = time.perf_counter()
            with blas_threads(threads.rotation):
                prepared_chunk = engine.prepare()
            rotation_s += time.perf_counter() - t_rot_start
            if prepared_chunk is None:
                break
            engine.compute_and_write(prepared_chunk)

    phenotype_stats = []
    for consumer, job in zip(engine.consumers, jobs, strict=True):
        if consumer.processed != n_filtered:
            raise RuntimeError(
                "Pre-allocated array size mismatch: wrote "
                f"{consumer.processed} results, expected {n_filtered}. "
                "This is an internal error; please report this issue with "
                "your dataset dimensions."
            )
        for key, n_nan in consumer.nan_counts.items():
            logger.warning(
                f"{n_nan}/{n_filtered} SNPs have NaN {key}; check for "
                "degenerate genotypes and kinship matrix quality"
            )
        log_lambda_boundary_warning(
            consumer.n_at_lmin,
            consumer.n_at_lmax,
            l_min,
            l_max,
            prefix=job.lambda_warning_prefix,
        )
        phenotype_stats.append(
            ChunkRunStats(
                processed=consumer.processed,
                compute_s=consumer.compute_s,
                result_write_s=consumer.result_write_s,
            )
        )

    return GroupedChunkRunStats(
        phenotypes=tuple(phenotype_stats),
        rotation_s=rotation_s,
    )


def run_lmm_chunk_source_numpy(
    *,
    genotypes: PreparedGenotypes,
    chunk_sink: ChunkSink,
    dispatch: DispatchPath,
    chunks: LmmChunkPlan,
    workspace: WorkspaceSpec,
    prepared: PreparedLmmRun,
    config: LmmConfig,
    progress_label: str = "LMM association",
    lambda_warning_prefix: str = "",
) -> ChunkRunStats:
    """Run the shared chunk engine for one phenotype."""
    grouped = run_lmm_chunk_source_numpy_group(
        genotypes=genotypes,
        jobs=(
            PhenotypeChunkJob(
                prepared=prepared,
                chunk_sink=chunk_sink,
                config=config,
                lambda_warning_prefix=lambda_warning_prefix,
            ),
        ),
        dispatch=dispatch,
        chunks=chunks,
        workspace=workspace,
        progress_label=progress_label,
    )
    stats = grouped.phenotypes[0]
    return ChunkRunStats(
        processed=stats.processed,
        rotation_s=grouped.rotation_s,
        compute_s=stats.compute_s,
        result_write_s=stats.result_write_s,
    )
