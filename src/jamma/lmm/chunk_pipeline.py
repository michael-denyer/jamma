"""Rotation/compute thread split and the overlapped chunk pipeline driver.

Owns the core-split heuristics and the per-run plan, plus
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
from jamma.core.threading import blas_threads, get_physical_core_count

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


class ThreadPlan(NamedTuple):
    """Fixed rotation and compute budgets for one association run.

    ``rotation`` is a process-wide BLAS limit. ``omp`` is the active compute
    count and may not exceed the native workspace capacity priced by the
    association plan.
    """

    rotation: int
    omp: int
    total_cores: int


def plan_thread_budget(
    *,
    n_samples: int,
    omp_threads: int,
    max_omp_threads: int,
    use_pipeline: bool,
) -> ThreadPlan:
    """Divide the physical cores between rotation and compute for this run.

    A sequential run never rotates and computes at once, so rotation gets every
    core and compute keeps its own OpenMP budget. A pipelined run overlaps the
    two, so the cores are split. The split stays fixed while work overlaps
    because the BLAS controller changes process-wide state.
    """
    total_cores = get_physical_core_count()
    omp_capacity = min(omp_threads, max_omp_threads)
    if not use_pipeline:
        return ThreadPlan(
            rotation=total_cores, omp=omp_capacity, total_cores=total_cores
        )

    logger.debug("Pipeline mode: overlapping rotation/compute")
    if omp_threads == 1:
        return ThreadPlan(rotation=total_cores, omp=1, total_cores=total_cores)

    _rot, compute_threads = compute_pipeline_core_split(n_samples, total_cores)
    omp = min(compute_threads, omp_capacity)
    rot = max(1, total_cores - omp)
    logger.debug(
        f"Pipeline core split: {rot} rotation, {omp} compute (n_samples={n_samples:,})"
    )
    return ThreadPlan(rotation=rot, omp=omp, total_cores=total_cores)


def _drive_pipeline(
    engine: _ChunkEngine,
    *,
    n_chunks: int,
    rotation_threads: int,
    n_samples: int,
    n_filtered: int,
    show_progress: bool,
    progress_label: str,
) -> float:
    """Drive the overlapped chunk pipeline shared by every NumPy runner.

    Prepares and computes the first chunk, then overlaps rotation of chunk N+1 (a
    background ``engine.prepare``) with C compute of chunk N (a foreground
    ``engine.compute_and_write``) via a single-worker executor. Both stages
    release the GIL, so they run concurrently.

    The engine owns the chunk source, the sink, the buffers, and the live core
    split, so the driver takes one typed argument rather than a pair of opaque
    callbacks plus a shared mutable budget object.

    Args:
        engine: The chunk engine to drive.
        n_chunks: Expected chunk count (progress total).
        rotation_threads: Process-wide BLAS limit held for the whole drive.
        n_samples: Sample count (ETA estimate).
        n_filtered: Filtered SNP count (ETA estimate; error diagnostics).
        show_progress: Whether to render a progress bar.
        progress_label: Progress-bar label.

    Returns:
        Total rotation wall-time (seconds) measured around the prepare calls,
        for the caller's timing breakdown. Compute and write time is
        accumulated by the engine itself.
    """
    with blas_threads(rotation_threads):
        rotation_s = 0.0

        # The first chunk is sequential. The fixed run plan already accounts
        # for maximum compute capacity and avoids changing global BLAS state.
        t = time.perf_counter()
        first = engine.prepare()
        rotation_s += time.perf_counter() - t
        if first is None:
            return rotation_s

        engine.compute_and_write(first)
        del first

        t = time.perf_counter()
        current = engine.prepare()
        rotation_s += time.perf_counter() - t

        # The first two prepared chunks are already accounted for.
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
                    # Both calls release the GIL, so rotation and compute overlap.
                    future = executor.submit(engine.prepare)
                    engine.compute_and_write(current)

                    t = time.perf_counter()
                    try:
                        current = future.result()
                    except (
                        MemoryError,
                        ValueError,
                        TypeError,
                        OverflowError,
                        OSError,
                    ):
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
