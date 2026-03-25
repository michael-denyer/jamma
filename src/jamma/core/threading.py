"""BLAS thread management for numpy operations.

This module provides explicit thread control for numpy BLAS operations
(eigendecomp, UT@G rotation) via threadpool_limits.

On macOS with Apple Accelerate, threadpoolctl cannot control the BLAS thread
count (Accelerate has no public thread-count API and ignores VECLIB_MAXIMUM_THREADS
after library init). In this case blas_threads() is a no-op and
is_blas_controllable() returns False so callers can adjust OpenMP thread counts
to avoid oversubscription.
"""

from __future__ import annotations

import functools
import os
import threading as _py_threading
from collections.abc import Generator
from contextlib import contextmanager

import psutil
from loguru import logger
from threadpoolctl import threadpool_info, threadpool_limits

_JLINALG_THREAD_LOCK = _py_threading.RLock()


def get_blas_backend() -> str:
    """Return the BLAS backend name from threadpool_info.

    Iterates over threadpool entries, returns the internal_api of the first
    entry with user_api == "blas". Returns "unknown" if none found.

    Returns:
        BLAS backend name (e.g., "mkl", "openblas", "accelerate") or "unknown".
    """
    for entry in threadpool_info():
        if entry.get("user_api") == "blas":
            return entry.get("internal_api", "unknown")
    return "unknown"


def get_physical_core_count() -> int:
    """Return the number of physical CPU cores.

    Use this for BLAS operations (e.g. eigendecomp, U.T @ G rotation).

    Returns:
        Physical core count, falling back to os.cpu_count() if psutil
        can't determine it.
    """
    return psutil.cpu_count(logical=False) or (os.cpu_count() or 1)


@functools.cache
def is_blas_controllable() -> bool:
    """Check if threadpoolctl can control the active BLAS library.

    Returns False on macOS with Apple Accelerate (threadpoolctl can't detect it)
    and in environments with no BLAS library loaded. Returns True for MKL and
    OpenBLAS.

    The result is cached — the BLAS library doesn't change mid-process.
    """
    return any(entry.get("user_api") == "blas" for entry in threadpool_info())


def get_blas_thread_count() -> int:
    """Determine the number of BLAS threads to use for numpy operations.

    Priority:
    1. JAMMA_BLAS_THREADS env var (explicit override for benchmarking)
    2. Physical core count via psutil (avoids hyperthreading oversubscription)

    Returns:
        Positive integer thread count. Capped at os.cpu_count().
    """
    max_threads = os.cpu_count() or 64

    env_override = os.environ.get("JAMMA_BLAS_THREADS")
    if env_override is not None:
        try:
            n = int(env_override)
        except ValueError:
            logger.warning(
                f"JAMMA_BLAS_THREADS={env_override!r} is not a valid integer, "
                "falling back to physical core count"
            )
        else:
            n = max(1, min(n, max_threads))
            logger.debug(f"BLAS threads from JAMMA_BLAS_THREADS: {n}")
            return n

    physical_cores = psutil.cpu_count(logical=False) or max_threads
    n = max(1, min(physical_cores, max_threads))
    logger.debug(f"BLAS threads: {n} (physical cores)")
    return n


def get_loco_worker_count() -> int:
    """Return configured LOCO worker count.

    Controls how many chromosomes are processed in parallel during
    LOCO analysis. Default is 1 (sequential), matching current behavior.
    Increase with caution — each parallel worker holds a full K_loco
    matrix (n_samples^2 * 8 bytes) in memory.

    Priority:
    1. JAMMA_LOCO_WORKERS env var
    2. Default: 1 (sequential)

    Returns:
        Positive integer worker count.
    """
    env_override = os.environ.get("JAMMA_LOCO_WORKERS")
    if env_override is not None:
        try:
            n = int(env_override)
        except ValueError:
            logger.warning(
                f"JAMMA_LOCO_WORKERS={env_override!r} is not a valid integer, "
                "falling back to 1 (sequential)"
            )
            return 1
        if n < 1:
            logger.warning(
                f"JAMMA_LOCO_WORKERS={n} is not a positive integer, "
                "clamping to 1 (sequential)"
            )
            return 1
        logger.debug(f"LOCO workers from JAMMA_LOCO_WORKERS: {n}")
        return n
    return 1


def get_c_extension_thread_count(
    c_accel_available: bool,
    c_has_openmp: bool,
) -> int:
    """Return the thread count for `_lmm_accel` compute kernels.

    The LMM C extension only runs in parallel when it was compiled with
    OpenMP support. When the extension is missing or single-threaded, callers
    must pass ``1`` so logs and pipeline heuristics do not pretend a serial
    kernel is running with many worker threads.

    Args:
        c_accel_available: Whether `_lmm_accel` imported successfully.
        c_has_openmp: Whether `_lmm_accel` was compiled with OpenMP.

    Returns:
        Thread count to pass to `_lmm_accel`.
    """
    if not c_accel_available or not c_has_openmp:
        return 1

    cores = get_physical_core_count()
    return max(1, cores // 2) if not is_blas_controllable() else cores


@contextmanager
def blas_threads(n_threads: int | None = None) -> Generator[None, None, None]:
    """Context manager for scoped BLAS thread control.

    Wraps threadpool_limits to centralise default thread count logic.
    Use around numpy BLAS operations (eigendecomp, matmul).

    Args:
        n_threads: Number of BLAS threads. None uses get_blas_thread_count().

    Example:
        >>> with blas_threads(8):
        ...     eigenvalues, eigenvectors = np.linalg.eigh(K)
    """
    if n_threads is None:
        n_threads = get_blas_thread_count()

    if not is_blas_controllable():
        # Accelerate or no BLAS detected — threadpool_limits is a no-op.
        # Log once so the user knows thread control isn't active.
        _warn_uncontrollable_blas()
        yield
        return

    with threadpool_limits(limits=n_threads, user_api="blas"):
        yield


@contextmanager
def jlinalg_threads(n_threads: int | None = None) -> Generator[None, None, None]:
    """Temporarily set jlinalg's internal thread count.

    jlinalg's snp_stats pthread pool does not use threadpoolctl; it is
    controlled by ``jlinalg.set_n_threads()``. The setting is process-global,
    so callers must scope changes carefully. A module-level lock serialises
    the set + compute window so concurrent pipeline workers cannot race
    thread-count changes.

    Args:
        n_threads: jlinalg thread count. None uses all physical cores.
    """
    if n_threads is None:
        n_threads = get_physical_core_count()
    n_threads = max(1, int(n_threads))

    try:
        from jamma import jlinalg
    except ImportError:
        yield
        return

    with _JLINALG_THREAD_LOCK:
        old_threads: int | None = None
        try:
            old_threads = jlinalg.set_n_threads(n_threads)
        except AttributeError:
            logger.warning(
                "jlinalg.set_n_threads() not available — build may be stale. "
                "Thread control disabled; run: python -m jamma.jlinalg._compile_jlinalg"
            )
            yield
            return
        except ValueError as exc:
            logger.warning(
                f"jlinalg.set_n_threads({n_threads}) failed: {exc}. "
                "Running with default thread count."
            )
            yield
            return

        try:
            yield
        finally:
            if old_threads is not None:
                jlinalg.set_n_threads(old_threads)


@functools.cache
def _warn_uncontrollable_blas() -> None:
    """Log a one-time warning that BLAS thread control is not active."""
    logger.warning(
        "BLAS thread control is not active — threadpoolctl found no "
        "controllable BLAS library. On macOS with Apple Accelerate this "
        "is expected; Accelerate manages its own threads internally. "
        "OpenMP thread counts are reduced automatically to compensate."
    )
