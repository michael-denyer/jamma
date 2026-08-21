"""Shared NumPy LMM chunk engine (orchestrator).

Owns the per-run chunk loop that the batch, disk-streaming, and LOCO NumPy
runners share: it selects the C/Python dispatch path, sizes chunks, drives the
optional rotate/compute pipeline, imputes missing genotypes, rotates via
``jlinalg.dgemm``, prepares Uab inputs, dispatches compute, and accumulates
per-chunk diagnostics. Callers provide raw genotype chunks
(``raw_chunk_source_factory``) and a result sink (``chunk_sink``); everything
after that boundary is owned here.

The workspace lifecycle, kernel-dispatch ladder, chunk sizing, and pipeline
driver live in sibling modules (``chunk_workspaces``, ``chunk_dispatch``,
``chunk_sizing``, ``chunk_pipeline``); this module wires them together.
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
    get_physical_core_count,
    jlinalg_threads,
)
from jamma.lmm import compute_numpy
from jamma.lmm.chunk_dispatch import (
    _ComputeContext,
    _dispatch_compute,
    _guarded_compute,
)
from jamma.lmm.chunk_pipeline import (
    _drive_pipeline,
    _ThreadBudget,
    compute_pipeline_core_split,
)
from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
from jamma.lmm.chunk_workspaces import _create_workspaces
from jamma.lmm.compute_numpy import (
    LmmMode,
    compute_lmm_chunk_numpy,
    select_current_dispatch_path,
)
from jamma.lmm.impute import impute_missing_inplace
from jamma.lmm.likelihood_numpy import (
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
    reset_p_yy_warned,
)
from jamma.lmm.results import count_lambda_boundary_hits, log_lambda_boundary_warning
from jamma.lmm.schema import RESULT_FIELDS as _RESULT_FIELDS

# Minimum number of chunks before pipelined execution is worthwhile.
_MIN_PIPELINE_CHUNKS = 8


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
    """Timing and diagnostic counters from the shared chunk runner."""

    processed: int
    rotation_s: float
    compute_s: float
    result_write_s: float
    nan_counts: dict[str, int]
    n_at_lmin: int
    n_at_lmax: int
    chunk_size: int
    n_chunks: int
    used_pipeline: bool


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
    requested_chunk_size: int | None = None,
    auto_scale_chunk_size: bool = True,
    chunk_sizer: Callable[..., int] = compute_chunk_size_numpy,
    min_pipeline_chunks: int = _MIN_PIPELINE_CHUNKS,
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
            processed=0,
            rotation_s=0.0,
            compute_s=0.0,
            result_write_s=0.0,
            nan_counts={},
            n_at_lmin=0,
            n_at_lmax=0,
            chunk_size=0,
            n_chunks=0,
            used_pipeline=False,
        )

    if requested_chunk_size is not None and requested_chunk_size < 1:
        raise ValueError(
            f"requested_chunk_size must be >= 1, got {requested_chunk_size}"
        )
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
    use_split = dispatch.use_split
    use_fused_general = dispatch.use_fused_general

    def _compute_engine_chunk_size(*, pipeline_buffers: int = 1) -> int:
        chunk = chunk_sizer(
            n_samples,
            n_filtered,
            n_cvt,
            use_split=use_split,
            lmm_mode=lmm_mode,
            use_fused_general=use_fused_general,
            pipeline_buffers=pipeline_buffers,
        )
        if requested_chunk_size is not None:
            chunk = min(chunk, requested_chunk_size)
        return max(1, chunk)

    if requested_chunk_size is None or auto_scale_chunk_size:
        chunk_size = _compute_engine_chunk_size()
    else:
        chunk_size = requested_chunk_size

    n_chunks = (n_filtered + chunk_size - 1) // chunk_size
    use_pipeline = use_split and n_chunks >= min_pipeline_chunks

    if use_pipeline:
        if requested_chunk_size is None or auto_scale_chunk_size:
            chunk_size = _compute_engine_chunk_size(pipeline_buffers=2)
        else:
            chunk_size = max(1, chunk_size // 2)
        n_chunks = (n_filtered + chunk_size - 1) // chunk_size
        use_pipeline = use_split and n_chunks >= min_pipeline_chunks

    if show_progress:
        logger.info(f"  Analyzed individuals: {n_samples:,}")
        logger.info(f"  Analyzed SNPs: {n_filtered:,}")
        if chunk_size < n_filtered:
            logger.info(
                f"  Processing in {n_chunks} chunks ({chunk_size:,} SNPs/chunk)"
            )

    omp_threads = get_c_extension_thread_count(
        compute_numpy._C_ACCEL_AVAILABLE, compute_numpy._C_HAS_OPENMP
    )

    if use_pipeline:
        logger.debug(f"Pipeline mode: overlapping rotation/compute ({n_chunks} chunks)")
        total_cores = get_physical_core_count()
        if omp_threads == 1:
            pipeline_rot_threads = total_cores
            pipeline_omp_threads = 1
        else:
            rot_threads, compute_threads = compute_pipeline_core_split(
                n_samples, total_cores
            )
            pipeline_omp_threads = min(compute_threads, omp_threads)
            pipeline_rot_threads = max(1, total_cores - pipeline_omp_threads)
            logger.debug(
                f"Pipeline core split: {pipeline_rot_threads} rotation, "
                f"{pipeline_omp_threads} compute (n_samples={n_samples:,})"
            )
    else:
        total_cores = get_physical_core_count()
        pipeline_omp_threads = omp_threads
        pipeline_rot_threads = total_cores

    budget = _ThreadBudget(pipeline_rot_threads, pipeline_omp_threads)
    n_refine = max(n_refine, 20)

    uab_invariant_soa = (
        compute_uab_invariant_soa(UtW, Uty, n_cvt) if use_split else None
    )
    w = UtW[:, 0].copy() if dispatch.needs_null_w else None

    lmm_workspace, score_fused_workspace, lrt_fused_workspace = _create_workspaces(
        dispatch,
        lmm_mode,
        n_cvt,
        eigenvalues_np,
        uab_invariant_soa,
        UtW,
        Uty,
        w,
        hi_eval_for_compute,
        logl_H0_for_compute,
        n_samples,
        l_min,
        l_max,
        n_grid,
        n_refine,
        pipeline_omp_threads,
    )
    compute_ctx = _ComputeContext(
        dispatch=dispatch,
        lmm_mode=lmm_mode,
        n_cvt=n_cvt,
        lmm_workspace=lmm_workspace,
        score_fused_workspace=score_fused_workspace,
        lrt_fused_workspace=lrt_fused_workspace,
        w=w,
        Uty=Uty,
        Hi_eval_null=hi_eval_for_compute,
        uab_invariant_soa=uab_invariant_soa,
        eigenvalues_np=eigenvalues_np,
        n_samples=n_samples,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_refine=n_refine,
        logl_H0=logl_H0_for_compute,
        n_filtered=n_filtered,
    )

    raw_chunk_source = raw_chunk_source_factory(chunk_size)

    if use_pipeline:
        utg_bufs = [
            np.empty((chunk_size, n_samples), dtype=np.float64),
            np.empty((chunk_size, n_samples), dtype=np.float64),
        ]
    else:
        utg_bufs = [np.empty((chunk_size, n_samples), dtype=np.float64)]

    if use_split and not dispatch.feeds_raw_utg:
        from jamma.lmm.likelihood import classify_uab_columns

        _inv_cols, var_cols = classify_uab_columns(n_cvt)
        n_var = len(var_cols)
        if use_pipeline:
            uab_var_bufs = [
                np.empty((chunk_size, n_var, n_samples), dtype=np.float64),
                np.empty((chunk_size, n_var, n_samples), dtype=np.float64),
            ]
        else:
            uab_var_bufs = [np.empty((chunk_size, n_var, n_samples), dtype=np.float64)]
    else:
        uab_var_bufs = None

    chunk_counter = 0
    next_expected_start = 0
    processed = 0
    rotation_s = 0.0
    compute_s = 0.0
    result_write_s = 0.0
    nan_counts: dict[str, int] = {}
    n_at_lmin = 0
    n_at_lmax = 0

    def _prepare_chunk() -> _PreparedLmmChunk | None:
        nonlocal chunk_counter, next_expected_start

        raw = raw_chunk_source()
        while raw is not None and raw.filtered_end <= raw.filtered_start:
            empty_range = raw.filtered_range
            if empty_range.filtered_start != empty_range.filtered_end:
                raise RuntimeError(
                    "raw LMM chunk has an invalid empty range: "
                    f"[{empty_range.filtered_start}, {empty_range.filtered_end})"
                )
            if empty_range.filtered_start != next_expected_start:
                raise RuntimeError(
                    "empty raw LMM chunks must preserve contiguous order: "
                    f"expected {next_expected_start}, "
                    f"got {empty_range.filtered_start}"
                )
            raw = raw_chunk_source()
        if raw is None:
            return None

        chunk_range = raw.filtered_range
        chunk_range.validate_next(next_expected_start, n_filtered)
        actual_len = chunk_range.length
        if raw.genotypes.shape != (n_samples, actual_len):
            raise ValueError(
                "raw LMM chunk shape mismatch: expected "
                f"({n_samples}, {actual_len}), got {raw.genotypes.shape}"
            )
        next_expected_start = chunk_range.filtered_end

        buf_idx = chunk_counter % len(utg_bufs)
        chunk_counter += 1

        impute_missing_inplace(
            raw.genotypes,
            filtered_means[chunk_range.filtered_start : chunk_range.filtered_end],
        )

        utg_out = utg_bufs[buf_idx][:actual_len, :]
        with jlinalg_threads(budget.rot):
            utg_t = jlinalg.dgemm(raw.genotypes, U, transa="T", out=utg_out)

        if dispatch.feeds_raw_utg:
            return _PreparedLmmChunk(utg_t, chunk_range)

        if use_split:
            out_var = (
                uab_var_bufs[buf_idx][:actual_len, :, :]
                if uab_var_bufs is not None and actual_len == chunk_size
                else None
            )
            uab_var_soa = batch_compute_uab_varying_soa_numpy(
                n_cvt, UtW, Uty, utg_t, out=out_var
            )
            return _PreparedLmmChunk(uab_var_soa, chunk_range)

        uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, utg_t.T)
        return _PreparedLmmChunk(uab_batch, chunk_range)

    def _compute_and_write(prepared: _PreparedLmmChunk) -> None:
        nonlocal processed, compute_s, result_write_s, n_at_lmin, n_at_lmax

        chunk_data = prepared.data
        chunk_range = prepared.filtered_range
        filtered_start = chunk_range.filtered_start
        filtered_end = chunk_range.filtered_end
        actual_len = chunk_range.length
        if filtered_start != processed:
            raise RuntimeError(
                "prepared LMM chunks reached compute out of order: "
                f"expected start {processed}, got {filtered_start}"
            )

        t_compute_start = time.perf_counter()
        blas_ctx = (
            blas_threads(1) if compute_numpy._C_ACCEL_AVAILABLE else nullcontext()
        )
        with blas_ctx:
            if use_split:
                cr = _dispatch_compute(compute_ctx, chunk_data, budget.omp, processed)
            else:
                cr = _guarded_compute(
                    compute_lmm_chunk_numpy,
                    lmm_mode,
                    n_cvt,
                    eigenvalues_np,
                    chunk_data,
                    n_samples,
                    l_min=l_min,
                    l_max=l_max,
                    n_grid=n_grid,
                    n_refine=n_refine,
                    Hi_eval_null=hi_eval_for_compute,
                    logl_H0=logl_H0_for_compute,
                    n_threads=budget.omp,
                    operation="LMM chunk compute",
                    write_offset=processed,
                    n_filtered=n_filtered,
                )
        compute_s += time.perf_counter() - t_compute_start

        t_write_start = time.perf_counter()
        chunk_arrays = {key: cr[key][:actual_len] for key in _RESULT_FIELDS[lmm_mode]}

        chunk_lmin, chunk_lmax = count_lambda_boundary_hits(
            lmm_mode, chunk_arrays, l_min, l_max
        )
        n_at_lmin += chunk_lmin
        n_at_lmax += chunk_lmax

        for key, arr in chunk_arrays.items():
            if arr.dtype.kind != "f":
                continue
            n_nan = int(np.count_nonzero(np.isnan(arr)))
            if n_nan > 0:
                nan_counts[key] = nan_counts.get(key, 0) + n_nan

        chunk_sink(chunk_arrays, filtered_start, filtered_end)
        processed += actual_len
        result_write_s += time.perf_counter() - t_write_start

    if use_pipeline:
        rotation_s += _drive_pipeline(
            _prepare_chunk,
            _compute_and_write,
            budget,
            n_chunks=n_chunks,
            total_cores=total_cores,
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
            prepared = _prepare_chunk()
            rotation_s += time.perf_counter() - t_rot_start
            if prepared is None:
                break
            _compute_and_write(prepared)

    if processed != n_filtered:
        raise RuntimeError(
            f"Pre-allocated array size mismatch: wrote {processed} results, "
            f"expected {n_filtered}. This is an internal error — please report "
            f"this issue with your dataset dimensions."
        )

    for key, n_nan in nan_counts.items():
        logger.warning(
            f"{n_nan}/{n_filtered} SNPs have NaN {key} — "
            "check for degenerate (constant) genotypes and kinship matrix quality"
        )
    log_lambda_boundary_warning(
        n_at_lmin, n_at_lmax, l_min, l_max, prefix=lambda_warning_prefix
    )

    return LmmChunkRunStats(
        processed=processed,
        rotation_s=rotation_s,
        compute_s=compute_s,
        result_write_s=result_write_s,
        nan_counts=nan_counts,
        n_at_lmin=n_at_lmin,
        n_at_lmax=n_at_lmax,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        used_pipeline=use_pipeline,
    )
