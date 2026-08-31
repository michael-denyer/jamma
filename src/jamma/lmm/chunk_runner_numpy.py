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
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
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
from jamma.lmm.association_plan import AssociationExecution
from jamma.lmm.chunk_kernel import Kernel, RunInvariants, make_kernel
from jamma.lmm.chunk_pipeline import _drive_pipeline, plan_thread_budget
from jamma.lmm.genotype_source import PreparedGenotypes
from jamma.lmm.impute import impute_missing_inplace
from jamma.lmm.likelihood import reset_p_yy_warned
from jamma.lmm.prepare_common import PreparedLmmRun
from jamma.lmm.results import count_lambda_boundary_hits, log_lambda_boundary_warning
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import LmmConfig
from jamma.lmm.uab import batch_compute_uab_numpy


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

    ``data`` is polymorphic by dispatch path: 2-D ``utg_t`` of shape
    ``(n_snps, n_samples)`` when ``dispatch.feeds_raw_utg`` (every C path),
    otherwise the full Uab batch the NumPy fallback consumes.
    ``_compute_and_write`` selects the matching kernel via the same
    ``dispatch`` flags that produced the data, so the two never disagree.
    """

    data: np.ndarray
    filtered_range: LmmChunkRange


class LmmChunkRunStats(NamedTuple):
    """What the chunk runner hands back: how much it did, and how long it took.

    Six further counters used to ride along here (nan_counts, the two lambda
    boundary tallies, chunk_size, n_chunks, used_pipeline). No caller read any
    of them; the runner already logs the diagnostics they carried.
    """

    processed: int
    rotation_s: float
    compute_s: float
    result_write_s: float


ChunkSink = Callable[[dict[str, np.ndarray], int, int], None]


@dataclass(frozen=True, slots=True)
class ChunkRunOptions:
    """Presentation choices for one shared chunk run."""

    progress_label: str = "LMM association"
    lambda_warning_prefix: str = ""


_DEFAULT_CHUNK_RUN_OPTIONS = ChunkRunOptions()


class _ChunkEngine:
    """The chunk loop's state: its buffers, its thread split, its counters.

    ``prepare`` and ``compute_and_write`` were closures in the runner body over
    seven ``nonlocal`` counters, and the pipeline driver reached the live thread
    split through a separate mutable object because a bare-int ``nonlocal``
    cannot cross a module boundary. Both are ordinary fields here, so the driver
    takes one typed argument and rebinds ``rot_threads``/``omp_threads``
    directly.
    """

    def __init__(
        self,
        *,
        invariants: RunInvariants,
        kernel: Kernel,
        U: np.ndarray,
        filtered_means: np.ndarray,
        raw_chunks: Iterator[RawLmmChunk],
        chunk_sink: Callable[[dict[str, np.ndarray], int, int], None],
        chunk_size: int,
        n_buffers: int,
        rot_threads: int,
        omp_threads: int,
    ) -> None:
        self.inv = invariants
        self.kernel = kernel
        self.U = U
        self.filtered_means = filtered_means
        self.raw_chunks = raw_chunks
        self.chunk_sink = chunk_sink
        self.chunk_size = chunk_size

        # Rebound by _drive_pipeline once it has profiled the first chunk.
        self.rot_threads = rot_threads
        self.omp_threads = omp_threads

        n_samples = invariants.n_samples
        self.utg_bufs = [
            np.empty((chunk_size, n_samples), dtype=np.float64)
            for _ in range(n_buffers)
        ]

        self.chunk_counter = 0
        self.next_expected_start = 0
        self.processed = 0
        self.compute_s = 0.0
        self.result_write_s = 0.0
        self.nan_counts: dict[str, int] = {}
        self.n_at_lmin = 0
        self.n_at_lmax = 0

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

        return _PreparedLmmChunk(self._kernel_input(utg_t), chunk_range)

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

    def _kernel_input(self, utg_t: np.ndarray) -> np.ndarray:
        """Shape the rotated chunk into whatever this run's kernel consumes."""
        if self.inv.dispatch.feeds_raw_utg:
            return utg_t

        return batch_compute_uab_numpy(
            self.inv.n_cvt, self.inv.UtW, self.inv.Uty, utg_t
        )

    def compute_and_write(self, prepared: _PreparedLmmChunk) -> None:
        """Run one prepared chunk through the kernel and hand results to the sink."""
        chunk_range = prepared.filtered_range
        filtered_start = chunk_range.filtered_start
        actual_len = chunk_range.length
        if filtered_start != self.processed:
            raise RuntimeError(
                "prepared LMM chunks reached compute out of order: "
                f"expected start {self.processed}, got {filtered_start}"
            )

        t_compute_start = time.perf_counter()
        blas_ctx = blas_threads(1) if self.kernel.uses_c else nullcontext()
        with blas_ctx:
            cr = self.kernel.compute_chunk(
                prepared.data, self.omp_threads, self.processed
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


def run_lmm_chunk_source_numpy(
    *,
    genotypes: PreparedGenotypes,
    chunk_sink: ChunkSink,
    execution: AssociationExecution,
    prepared: PreparedLmmRun,
    config: LmmConfig,
    options: ChunkRunOptions = _DEFAULT_CHUNK_RUN_OPTIONS,
) -> LmmChunkRunStats:
    """Run LMM association over caller-provided raw genotype chunks.

    The prepared source owns aligned imputation means and raw chunks. This
    function consumes it with the selected dispatch and chunk geometry, then owns
    pipeline driving, eigen-rotation, Uab preparation, C/Python compute dispatch,
    diagnostics, and timing. Batch and LOCO use this path so their chunk compute
    behavior cannot drift.

    A result field that is all NaN (e.g. a non-PSD kinship matrix, or a phenotype
    made degenerate by collinear covariates) is surfaced via ``logger.warning``
    per field, not raised: an all-NaN result is a legitimate GEMMA-equivalent
    outcome for degenerate inputs, and a fatal abort would be both wrong there and
    sensitive to platform floating-point (the NaN fraction can differ by BLAS).
    """
    # Every runner reaches this entry, so the per-run diagnostic reset lives
    # here rather than in one of them.
    reset_p_yy_warned()

    n_samples = prepared.n_samples
    n_cvt = prepared.n_cvt
    n_filtered = genotypes.n_filtered
    lmm_mode = config.lmm_mode
    l_min = config.l_min
    l_max = config.l_max
    show_progress = config.show_progress

    if genotypes.analyzed_sample_count != n_samples:
        raise ValueError(
            "prepared genotype sample count does not match prepared LMM run: "
            f"got {genotypes.analyzed_sample_count} and {n_samples}"
        )

    if n_filtered == 0:
        return LmmChunkRunStats(
            processed=0, rotation_s=0.0, compute_s=0.0, result_write_s=0.0
        )

    dispatch = execution.dispatch
    chunk_plan = execution.chunks
    chunk_size = chunk_plan.chunk_size
    n_chunks = chunk_plan.n_chunks
    use_pipeline = chunk_plan.use_pipeline

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
        use_pipeline=use_pipeline,
    )
    n_refine = max(config.n_refine, 20)

    invariants = RunInvariants.build(
        dispatch=dispatch,
        lmm_mode=lmm_mode,
        n_cvt=n_cvt,
        n_samples=n_samples,
        n_filtered=n_filtered,
        eigenvalues=prepared.eigenvalues,
        UtW=prepared.UtW,
        Uty=prepared.Uty,
        Hi_eval_null=prepared.Hi_eval_null,
        logl_H0=prepared.logl_H0,
        l_min=l_min,
        l_max=l_max,
        n_grid=config.n_grid,
        n_refine=n_refine,
    )
    kernel = make_kernel(invariants, threads.omp)

    engine = _ChunkEngine(
        invariants=invariants,
        kernel=kernel,
        U=prepared.U,
        filtered_means=genotypes.imputation_means,
        raw_chunks=genotypes.chunks(chunk_size),
        chunk_sink=chunk_sink,
        chunk_size=chunk_size,
        n_buffers=chunk_plan.n_buffers,
        rot_threads=threads.rot,
        omp_threads=threads.omp,
    )

    rotation_s = 0.0
    if use_pipeline:
        rotation_s += _drive_pipeline(
            engine,
            n_chunks=n_chunks,
            total_cores=threads.total_cores,
            n_samples=n_samples,
            n_filtered=n_filtered,
            show_progress=show_progress,
            progress_label=options.progress_label,
        )
    else:
        if show_progress and n_chunks > 1:
            chunk_iterator = progress_iterator(
                iter(range(n_chunks)),
                total=n_chunks,
                desc=options.progress_label,
                initial_eta_seconds=estimate_lmm_seconds(n_samples, n_filtered),
            )
        else:
            chunk_iterator = range(n_chunks)

        for _chunk_idx in chunk_iterator:
            t_rot_start = time.perf_counter()
            prepared_chunk = engine.prepare()
            rotation_s += time.perf_counter() - t_rot_start
            if prepared_chunk is None:
                break
            engine.compute_and_write(prepared_chunk)

    if engine.processed != n_filtered:
        raise RuntimeError(
            f"Pre-allocated array size mismatch: wrote {engine.processed} results, "
            f"expected {n_filtered}. This is an internal error — please report "
            f"this issue with your dataset dimensions."
        )

    for key, n_nan in engine.nan_counts.items():
        logger.warning(
            f"{n_nan}/{n_filtered} SNPs have NaN {key} — "
            "check for degenerate (constant) genotypes and kinship matrix quality"
        )
    log_lambda_boundary_warning(
        engine.n_at_lmin,
        engine.n_at_lmax,
        l_min,
        l_max,
        prefix=options.lambda_warning_prefix,
    )

    return LmmChunkRunStats(
        processed=engine.processed,
        rotation_s=rotation_s,
        compute_s=engine.compute_s,
        result_write_s=engine.result_write_s,
    )
