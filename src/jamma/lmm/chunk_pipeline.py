"""Rotation/compute thread split and the overlapped chunk pipeline driver.

Owns the core-split heuristics (static and adaptive) and ``_drive_pipeline``,
which overlaps background rotation of chunk N+1 with foreground C compute of
chunk N. Split out from ``chunk_runner_numpy`` so the concurrency machinery is
isolated from chunk sizing, workspace allocation, and per-chunk dispatch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any

from loguru import logger

from jamma.core.estimates import estimate_lmm_seconds
from jamma.core.progress import create_progress_bar
from jamma.core.threading import is_blas_controllable


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


class _ThreadBudget:
    """Mutable rotation/compute core split shared with the pipeline callbacks.

    The pipeline driver re-derives the split from the profiled first chunk and
    rebinds these fields. Because the prepare/compute callbacks live in the
    runner's scope while the driver lives here, a bare-int ``nonlocal`` cannot
    carry the update across the boundary — the callbacks would keep reading the
    pre-profile values. A shared mutable object does: the callbacks read
    ``budget.rot`` / ``budget.omp`` and the driver mutates them in place.
    """

    __slots__ = ("omp", "rot")

    def __init__(self, rot: int, omp: int) -> None:
        self.rot = rot
        self.omp = omp


def _drive_pipeline(
    prepare: Callable[[], Any | None],
    compute: Callable[[Any], None],
    budget: _ThreadBudget,
    *,
    n_chunks: int,
    total_cores: int,
    n_samples: int,
    n_filtered: int,
    show_progress: bool,
    progress_label: str,
) -> float:
    """Drive the overlapped chunk pipeline shared by both NumPy runners.

    Profiles the first chunk, re-derives the rotation/compute core split from
    its measured stage durations, then overlaps rotation of chunk N+1 (a
    background ``prepare``) with C compute of chunk N (a foreground ``compute``)
    via a single-worker executor. Both stages release the GIL, so they run
    concurrently.

    Only the chunk source (in-memory fancy-index vs. disk stream) and the result
    sink differ between runners; those are supplied as ``prepare`` and
    ``compute`` callbacks. ``prepare`` returns an opaque prepared-chunk object,
    or None at exhaustion; the driver passes it straight to ``compute`` and
    never inspects it. Both callbacks read the live core split from ``budget``,
    which this function mutates after profiling.

    Args:
        prepare: Zero-arg callback that prepares the next chunk (slice/impute/
            rotate), returning an opaque object or None when no chunks remain.
        compute: Callback that runs C compute on a prepared chunk and writes its
            results. Owns its own compute/write timing and diagnostics.
        budget: Shared mutable core split; rebound from the profiled first chunk.
        n_chunks: Expected chunk count (progress total; adaptive-split guard).
        total_cores: Physical core count for the adaptive split.
        n_samples: Sample count (adaptive-split static fallback; ETA estimate).
        n_filtered: Filtered SNP count (ETA estimate; error diagnostics).
        show_progress: Whether to render a progress bar.
        progress_label: Progress-bar label.

    Returns:
        Total rotation wall-time (seconds) measured around the prepare calls,
        for the caller's timing breakdown. Compute/write time is accumulated by
        the ``compute`` callback itself.
    """
    rotation_s = 0.0

    # Profile the first chunk: prepare (rotation) then compute, timing each
    # stage so the adaptive split below uses empirically measured durations.
    t = time.perf_counter()
    first = prepare()
    t_first_rot = time.perf_counter() - t
    rotation_s += t_first_rot
    if first is None:
        return rotation_s

    t = time.perf_counter()
    compute(first)
    t_first_compute = time.perf_counter() - t
    del first

    # Re-derive the core split from measured times (only when chunks remain and
    # BLAS is controllable). Mutating the shared budget rebinds what the
    # prepare/compute callbacks read on every subsequent call.
    if n_chunks > 2 and is_blas_controllable():
        old_rot, old_omp = budget.rot, budget.omp
        budget.rot, budget.omp = compute_adaptive_core_split(
            t_first_rot, t_first_compute, total_cores, n_samples=n_samples
        )
        if (budget.rot, budget.omp) != (old_rot, old_omp):
            logger.debug(
                f"Adaptive core split: {old_rot}/{old_omp} -> "
                f"{budget.rot}/{budget.omp} "
                f"(rot={t_first_rot:.3f}s, compute={t_first_compute:.3f}s)"
            )

    # Seed the pipeline with the next chunk (uses the updated split).
    t = time.perf_counter()
    current = prepare()
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
                future = executor.submit(prepare)
                compute(current)

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
