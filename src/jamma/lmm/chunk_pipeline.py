"""Rotation/compute thread split and the overlapped chunk pipeline driver.

Owns the core-split heuristics (static, adaptive, and the per-run plan) and
``_drive_pipeline``, which overlaps background rotation of chunk N+1 with
foreground C compute of chunk N. Split out from ``chunk_runner_numpy`` so the
concurrency machinery is isolated from chunk sizing and kernel dispatch.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

from jamma.core.estimates import estimate_lmm_seconds
from jamma.core.progress import create_progress_bar
from jamma.core.threading import get_physical_core_count, is_blas_controllable

if TYPE_CHECKING:
    from jamma.lmm.chunk_runner_numpy import _ChunkEngine


def compute_pipeline_core_split(n_samples: int, total_cores: int) -> tuple[int, int]:
    """Compute rotation/compute thread split for the pipeline path.

    DGEMM rotation scales with n_samples^2 * chunk_size while per-SNP
    compute scales with chunk_size * (n_grid + n_refine). For large
    n_samples rotation dominates; for small n_samples compute dominates.

    Args:
        n_samples: Number of samples in the dataset.
        total_cores: Physical core count available.

    Returns:
        (rotation_threads, compute_threads) tuple. Both >= 1.
    """
    if n_samples > 10_000:
        rot = max(1, total_cores // 2)
    elif n_samples > 1_000:
        rot = max(1, total_cores // 3)
    else:
        rot = max(1, total_cores // 4)
    return rot, max(1, total_cores - rot)


def compute_adaptive_core_split(
    rot_time: float,
    compute_time: float,
    total_cores: int,
    *,
    n_samples: int = 0,
) -> tuple[int, int]:
    """Compute rotation/compute thread split from measured first-chunk times.

    Allocates threads proportionally to observed rotation vs compute wall time.
    Falls back to static heuristic when profiling data is degenerate (both
    times near zero, which happens on small datasets where profiling overhead
    dominates).

    Args:
        rot_time: Wall time for first-chunk rotation (UT@G DGEMM), seconds.
        compute_time: Wall time for first-chunk compute (C extension), seconds.
        total_cores: Physical core count available.
        n_samples: Sample count for static fallback (only used when times are
            degenerate).

    Returns:
        (rotation_threads, compute_threads) tuple. Both >= 1.
    """
    total_time = rot_time + compute_time
    if total_time < 0.01:  # < 10ms: profiling not meaningful, use static
        return compute_pipeline_core_split(n_samples, total_cores)

    rot_fraction = rot_time / total_time
    rot_threads = max(1, min(total_cores - 1, round(total_cores * rot_fraction)))
    compute_threads = max(1, total_cores - rot_threads)
    return rot_threads, compute_threads


class ThreadPlan(NamedTuple):
    """The compute (OpenMP) thread budget for a run, before any profiling.

    Rotation runs under whatever BLAS thread count the process holds; only
    the compute side is a budget the chunk engine reads.
    """

    omp: int
    total_cores: int


def plan_thread_budget(
    *, n_samples: int, omp_threads: int, use_pipeline: bool
) -> ThreadPlan:
    """Divide the physical cores between rotation and compute for this run.

    A sequential run never rotates and computes at once, so rotation gets every
    core and compute keeps its own OpenMP budget. A pipelined run overlaps the
    two, so the cores are split, with rotation taking whatever compute does not.
    ``_drive_pipeline`` re-derives the split from the profiled first chunk.
    """
    total_cores = get_physical_core_count()
    if not use_pipeline:
        return ThreadPlan(omp=omp_threads, total_cores=total_cores)

    logger.debug("Pipeline mode: overlapping rotation/compute")
    if omp_threads == 1:
        return ThreadPlan(omp=1, total_cores=total_cores)

    _rot, compute_threads = compute_pipeline_core_split(n_samples, total_cores)
    omp = min(compute_threads, omp_threads)
    rot = max(1, total_cores - omp)
    logger.debug(
        f"Pipeline core split: {rot} rotation, {omp} compute (n_samples={n_samples:,})"
    )
    return ThreadPlan(omp=omp, total_cores=total_cores)


def _drive_pipeline(
    engine: _ChunkEngine,
    *,
    n_chunks: int,
    total_cores: int,
    n_samples: int,
    n_filtered: int,
    show_progress: bool,
    progress_label: str,
) -> float:
    """Drive the overlapped chunk pipeline shared by every NumPy runner.

    Profiles the first chunk, re-derives the rotation/compute core split from
    its measured stage durations, then overlaps rotation of chunk N+1 (a
    background ``engine.prepare``) with C compute of chunk N (a foreground
    ``engine.compute_and_write``) via a single-worker executor. Both stages
    release the GIL, so they run concurrently.

    The engine owns the chunk source, the sink, the buffers, and the live core
    split, so the driver takes one typed argument rather than a pair of opaque
    callbacks plus a shared mutable budget object. The split it writes back is
    what the engine reads on every subsequent chunk.

    Args:
        engine: The chunk engine to drive. Its ``omp_threads`` is rebound
            here from the profiled first chunk.
        n_chunks: Expected chunk count (progress total; adaptive-split guard).
        total_cores: Physical core count for the adaptive split.
        n_samples: Sample count (adaptive-split static fallback; ETA estimate).
        n_filtered: Filtered SNP count (ETA estimate; error diagnostics).
        show_progress: Whether to render a progress bar.
        progress_label: Progress-bar label.

    Returns:
        Total rotation wall-time (seconds) measured around the prepare calls,
        for the caller's timing breakdown. Compute and write time is
        accumulated by the engine itself.
    """
    rotation_s = 0.0

    # Profile the first chunk: prepare (rotation) then compute, timing each
    # stage so the adaptive split below uses empirically measured durations.
    t = time.perf_counter()
    first = engine.prepare()
    t_first_rot = time.perf_counter() - t
    rotation_s += t_first_rot
    if first is None:
        return rotation_s

    t = time.perf_counter()
    engine.compute_and_write(first)
    t_first_compute = time.perf_counter() - t
    del first

    # Re-derive the core split from measured times (only when chunks remain and
    # BLAS is controllable). The engine reads these fields on every subsequent
    # chunk, so writing them here is what makes the new split take effect.
    if n_chunks > 2 and is_blas_controllable():
        old_omp = engine.omp_threads
        _rot, engine.omp_threads = compute_adaptive_core_split(
            t_first_rot, t_first_compute, total_cores, n_samples=n_samples
        )
        if engine.omp_threads != old_omp:
            logger.debug(
                f"Adaptive compute split: {old_omp} -> {engine.omp_threads} "
                f"of {total_cores} cores "
                f"(rot={t_first_rot:.3f}s, compute={t_first_compute:.3f}s)"
            )

    # Seed the pipeline with the next chunk (uses the updated split).
    t = time.perf_counter()
    current = engine.prepare()
    rotation_s += time.perf_counter() - t

    # Progress: profiled chunk + seeded chunk already accounted, so start at 2.
    bar = (
        create_progress_bar(
            n_chunks,
            progress_label,
            initial_eta_seconds=estimate_lmm_seconds(n_samples, n_filtered),
        )
        if show_progress and n_chunks > 1
        else None
    )
    if bar is not None:
        bar.update(2)

    i = 2
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            while current is not None:
                # Prepare chunk N+1 in the background while computing chunk N;
                # both release the GIL, so rotation and compute overlap.
                future = executor.submit(engine.prepare)
                engine.compute_and_write(current)

                t = time.perf_counter()
                try:
                    current = future.result()
                except (MemoryError, ValueError, TypeError, OverflowError, OSError):
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"Pipeline chunk preparation failed at chunk {i} of "
                        f"{n_chunks} during overlapped rotation "
                        f"({n_filtered} SNPs total)."
                    ) from exc
                rotation_s += time.perf_counter() - t

                i += 1
                if bar is not None:
                    bar.update(i)
    finally:
        if bar is not None:
            with suppress(Exception):
                bar.update(n_chunks)
                bar.finish()

    return rotation_s
