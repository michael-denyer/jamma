"""Rotation/compute thread split and the overlapped chunk pipeline driver.

Owns the core-split heuristics and the per-run plan, plus the pair that overlaps
background rotation of chunk N+1 with foreground C compute of chunk N:
``_overlapped_chunks`` yields the prepared chunks and ``_drive_pipeline``
computes them. Split out from ``chunk_runner_numpy`` so the concurrency
machinery is isolated from chunk sizing and kernel dispatch.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

from jamma.core.estimates import estimate_lmm_seconds
from jamma.core.progress import progress_iterator
from jamma.core.threading import blas_threads, get_physical_core_count

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from jamma.lmm.chunk_runner_numpy import _ChunkEngine, _PreparedLmmChunk


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


def _overlapped_chunks(
    engine: _ChunkEngine,
    executor: ThreadPoolExecutor,
    rotation_s: list[float],
) -> Iterator[_PreparedLmmChunk]:
    """Yield each prepared chunk while the next one rotates in the background.

    Submitting the successor before yielding is what overlaps the two stages.
    The caller computes chunk N on the foreground thread while ``engine.prepare``
    rotates chunk N+1 on the executor, and both release the GIL.

    Args:
        engine: The chunk engine to pull from.
        executor: Single-worker executor that owns the background rotation.
        rotation_s: Single-element accumulator for foreground rotation seconds.

    Yields:
        Prepared chunks in source order, until the engine is exhausted.
    """

    def awaited(
        next_chunk: Callable[[], _PreparedLmmChunk | None],
    ) -> _PreparedLmmChunk | None:
        t = time.perf_counter()
        prepared = next_chunk()
        rotation_s[0] += time.perf_counter() - t
        return prepared

    current = awaited(engine.prepare)
    while current is not None:
        future = executor.submit(engine.prepare)
        yield current
        current = awaited(future.result)


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

    Computes every chunk :func:`_overlapped_chunks` yields, so rotation of chunk
    N+1 runs on the executor while C compute of chunk N runs here. The executor
    is owned here rather than by the generator, so a failing compute unwinds
    through ``ThreadPoolExecutor.__exit__`` and waits for the in-flight rotation
    instead of leaving that cleanup to garbage collection.

    The engine owns the chunk source, the sink, the buffers, and the live core
    split, so the driver takes one typed argument rather than a pair of opaque
    callbacks plus a shared mutable budget object.

    Args:
        engine: The chunk engine to drive.
        n_chunks: Expected chunk count (progress total).
        rotation_threads: Process-wide BLAS limit held for the whole drive.
        n_samples: Sample count (ETA estimate).
        n_filtered: Filtered SNP count (ETA estimate).
        show_progress: Whether to render a progress bar.
        progress_label: Progress-bar label.

    Returns:
        Total rotation wall-time (seconds) measured around the prepare calls,
        for the caller's timing breakdown. Compute and write time is
        accumulated by the engine itself.
    """
    rotation_s = [0.0]
    with blas_threads(rotation_threads), ThreadPoolExecutor(max_workers=1) as executor:
        chunks: Iterator[_PreparedLmmChunk] = _overlapped_chunks(
            engine, executor, rotation_s
        )
        if show_progress and n_chunks > 1:
            chunks = progress_iterator(
                chunks,
                total=n_chunks,
                desc=progress_label,
                initial_eta_seconds=estimate_lmm_seconds(n_samples, n_filtered),
            )
        for chunk in chunks:
            engine.compute_and_write(chunk)
    return rotation_s[0]
