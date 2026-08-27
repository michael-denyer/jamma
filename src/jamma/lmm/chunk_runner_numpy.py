"""Shared NumPy LMM chunk engine (orchestrator).

Owns the per-run chunk loop that the batch, disk-streaming, and LOCO NumPy
runners share: it selects the C/Python dispatch path, sizes chunks, drives the
optional rotate/compute pipeline, imputes missing genotypes, rotates via
``jlinalg.dgemm``, prepares Uab inputs, dispatches compute, and accumulates
per-chunk diagnostics. Callers provide raw genotype chunks
(``raw_chunk_source_factory``) and a result sink (``chunk_sink``); everything
after that boundary is owned here.

The kernel and the state it needs live in ``chunk_kernel``, chunk sizing in
``chunk_sizing``, and the overlapped driver in ``chunk_pipeline``; this module
wires them together.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import NamedTuple

import numpy as np
from loguru import logger

from jamma import jlinalg
from jamma.core.estimates import estimate_lmm_seconds
from jamma.core.progress import progress_iterator
from jamma.core.threading import (
    blas_threads,
    get_c_extension_thread_count,
    jlinalg_threads,
)
from jamma.lmm import compute_numpy
from jamma.lmm.chunk_kernel import Kernel, RunInvariants, make_kernel
from jamma.lmm.chunk_pipeline import _drive_pipeline, plan_thread_budget
from jamma.lmm.chunk_sizing import plan_lmm_chunks
from jamma.lmm.compute_numpy import select_current_dispatch_path
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.impute import impute_missing_inplace
from jamma.lmm.likelihood import classify_uab_columns, reset_p_yy_warned
from jamma.lmm.results import count_lambda_boundary_hits, log_lambda_boundary_warning
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS
from jamma.lmm.schema import LmmMode
from jamma.lmm.uab import batch_compute_uab_numpy, batch_compute_uab_varying_soa_numpy


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
    ``(n_snps, n_samples)`` when ``dispatch.feeds_raw_utg`` (the fused family),
    otherwise the 3-D varying-Uab SoA of shape ``(n_snps, n_var, n_samples)`` for
    the SoA-split path. ``_compute_and_write`` selects the matching kernel via the
    same ``dispatch`` flags that produced the data, so the two never disagree.
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
        raw_chunk_source: Callable[[], RawLmmChunk | None],
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
        self.raw_chunk_source = raw_chunk_source
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
        if invariants.dispatch is DispatchPath.SOA_SPLIT:
            n_var = len(classify_uab_columns(invariants.n_cvt)[1])
            self.uab_var_bufs: list[np.ndarray] | None = [
                np.empty((chunk_size, n_var, n_samples), dtype=np.float64)
                for _ in range(n_buffers)
            ]
        else:
            self.uab_var_bufs = None

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
        with jlinalg_threads(self.rot_threads):
            utg_t = jlinalg.dgemm(raw.genotypes, self.U, transa="T", out=utg_out)

        return _PreparedLmmChunk(
            self._kernel_input(utg_t, buf_idx, actual_len), chunk_range
        )

    def _next_non_empty_chunk(self) -> RawLmmChunk | None:
        """Skip zero-length chunks, checking each keeps the contiguity contract."""
        raw = self.raw_chunk_source()
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
            raw = self.raw_chunk_source()
        return raw

    def _kernel_input(
        self, utg_t: np.ndarray, buf_idx: int, actual_len: int
    ) -> np.ndarray:
        """Shape the rotated chunk into whatever this run's kernel consumes."""
        if self.inv.dispatch.feeds_raw_utg:
            return utg_t

        if self.inv.dispatch is DispatchPath.SOA_SPLIT:
            out_var = (
                self.uab_var_bufs[buf_idx][:actual_len, :, :]
                if self.uab_var_bufs is not None and actual_len == self.chunk_size
                else None
            )
            return batch_compute_uab_varying_soa_numpy(
                self.inv.n_cvt, self.inv.UtW, self.inv.Uty, utg_t, out=out_var
            )

        return batch_compute_uab_numpy(
            self.inv.n_cvt, self.inv.UtW, self.inv.Uty, utg_t.T
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
        blas_ctx = (
            blas_threads(1) if compute_numpy._accel is not None else nullcontext()
        )
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
    raw_chunk_source_factory: Callable[[int], Callable[[], RawLmmChunk | None]],
    chunk_sink: Callable[[dict[str, np.ndarray], int, int], None],
    U: np.ndarray,
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    Hi_eval_null: np.ndarray | None,
    logl_H0: float | None,
    n_samples: int,
    n_filtered: int,
    n_cvt: int,
    lmm_mode: LmmMode,
    filtered_means: np.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    max_chunk_size: int | None = None,
    show_progress: bool = True,
    progress_label: str = "LMM association",
    lambda_warning_prefix: str = "",
    log_dispatch_choices: bool = True,
) -> LmmChunkRunStats:
    """Run LMM association over caller-provided raw genotype chunks.

    The caller owns where raw genotype chunks come from and where result chunks
    go. This function owns the canonical NumPy LMM chunk machinery: dispatch
    selection, chunk sizing, optional pipeline driving, missing-value imputation,
    eigen-rotation, Uab preparation, C/Python compute dispatch, diagnostics, and
    timing. Batch and LOCO use this path so their chunk compute behavior cannot
    drift.

    A result field that is all NaN (e.g. a non-PSD kinship matrix, or a phenotype
    made degenerate by collinear covariates) is surfaced via ``logger.warning``
    per field, not raised: an all-NaN result is a legitimate GEMMA-equivalent
    outcome for degenerate inputs, and a fatal abort would be both wrong there and
    sensitive to platform floating-point (the NaN fraction can differ by BLAS).
    """
    # Every runner reaches this entry, so the per-run diagnostic reset lives
    # here rather than in one of them.
    reset_p_yy_warned()

    if n_filtered == 0:
        return LmmChunkRunStats(
            processed=0, rotation_s=0.0, compute_s=0.0, result_write_s=0.0
        )

    if max_chunk_size is not None and max_chunk_size < 1:
        raise ValueError(f"max_chunk_size must be >= 1, got {max_chunk_size}")
    if len(filtered_means) != n_filtered:
        raise ValueError(
            f"filtered_means length ({len(filtered_means)}) does not match "
            f"n_filtered ({n_filtered})"
        )
    if lmm_mode in (3, 4) and Hi_eval_null is None:
        raise RuntimeError("LMM Score/All mode requires Hi_eval_null")
    if lmm_mode in (2, 4) and logl_H0 is None:
        raise RuntimeError("LMM LRT/All mode requires logl_H0")

    hi_eval_for_compute = (
        np.empty(0, dtype=np.float64) if Hi_eval_null is None else Hi_eval_null
    )
    logl_H0_for_compute = float("nan") if logl_H0 is None else logl_H0

    dispatch = select_current_dispatch_path(
        n_cvt, lmm_mode, log_choices=log_dispatch_choices
    )

    chunk_plan = plan_lmm_chunks(
        n_samples, n_filtered, n_cvt, dispatch, max_chunk_size=max_chunk_size
    )
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
        omp_threads=get_c_extension_thread_count(
            compute_numpy._accel is not None, compute_numpy._C_HAS_OPENMP
        ),
        use_pipeline=use_pipeline,
    )
    n_refine = max(n_refine, 20)

    invariants = RunInvariants.build(
        dispatch=dispatch,
        lmm_mode=lmm_mode,
        n_cvt=n_cvt,
        n_samples=n_samples,
        n_filtered=n_filtered,
        eigenvalues=eigenvalues_np,
        UtW=UtW,
        Uty=Uty,
        Hi_eval_null=hi_eval_for_compute,
        logl_H0=logl_H0_for_compute,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_refine=n_refine,
    )
    kernel = make_kernel(invariants, threads.omp)

    engine = _ChunkEngine(
        invariants=invariants,
        kernel=kernel,
        U=U,
        filtered_means=filtered_means,
        raw_chunk_source=raw_chunk_source_factory(chunk_size),
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
            prepared = engine.prepare()
            rotation_s += time.perf_counter() - t_rot_start
            if prepared is None:
                break
            engine.compute_and_write(prepared)

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
        engine.n_at_lmin, engine.n_at_lmax, l_min, l_max, prefix=lambda_warning_prefix
    )

    return LmmChunkRunStats(
        processed=engine.processed,
        rotation_s=rotation_s,
        compute_s=engine.compute_s,
        result_write_s=engine.result_write_s,
    )
