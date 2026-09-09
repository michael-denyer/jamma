"""BLAS thread management for numpy operations.

This module provides explicit thread control for numpy BLAS operations
(eigendecomp, UT@G rotation) via threadpool_limits.

On macOS with Apple Accelerate, threadpoolctl cannot control the BLAS thread
count (Accelerate has no public thread-count API and ignores VECLIB_MAXIMUM_THREADS
after library init). In this case blas_threads() is a no-op and
is_blas_controllable() returns False, which the pipeline driver reads to skip
its adaptive rotation/compute core split.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import psutil
from loguru import logger
from threadpoolctl import threadpool_info, threadpool_limits

from jamma.core.constants import Env

_BLAS_DISPLAY: dict[str, str] = {
    "mkl": "MKL",
    "openblas": "OpenBLAS",
    "accelerate": "Accelerate",
}


def blas_display_name(backend: str) -> str:
    """Return the log spelling of a lowercase BLAS backend name."""
    return _BLAS_DISPLAY.get(backend, backend.title())


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

    env_override = Env.current().blas_threads_raw
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
    env_override = Env.current().loco_workers_raw
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

    return get_physical_core_count()


@dataclass(frozen=True, slots=True)
class RunThreads:
    """Every thread decision for one run, read from the machine once.

    The kernels take their counts from here and the logs print the same
    fields, so a banner cannot show a number the run never used.
    """

    blas: int
    """Requested BLAS thread count (``get_blas_thread_count``)."""
    blas_controllable: bool
    """Whether ``blas_threads`` can enforce ``blas``; False on Accelerate."""
    blas_backend: str
    """Lowercase BLAS name: threadpoolctl's, else derived from jlinalg's."""
    c_ext: int
    """OpenMP thread count for ``_lmm_accel``; 1 when serial or absent."""
    c_ext_openmp: bool
    """Whether ``_lmm_accel`` was compiled with OpenMP."""
    c_ext_available: bool
    """Whether ``_lmm_accel`` loaded."""
    loco_workers: int
    """Parallel chromosome workers for LOCO (``get_loco_worker_count``)."""

    @property
    def blas_display(self) -> str:
        return blas_display_name(self.blas_backend)

    def blas_label(self) -> str:
        """``"18"`` when the BLAS honours the request, else what it does instead."""
        if self.blas_controllable:
            return str(self.blas)
        return f"uncontrolled ({self.blas_display})"

    def describe(self) -> str:
        """One log line naming every thread count the run will use."""
        blas = f"BLAS={self.blas} ({self.blas_display}"
        blas += ")" if self.blas_controllable else ", uncontrolled)"
        if not self.c_ext_available:
            c_ext = "C-ext=none"
        elif self.c_ext_openmp:
            c_ext = f"C-ext={self.c_ext} (OpenMP)"
        else:
            c_ext = f"C-ext={self.c_ext} (no OpenMP)"
        return f"Threads: {blas} | {c_ext} | LOCO workers={self.loco_workers}"


def _jlinalg_backend_key(name: str) -> str:
    """Map jlinalg's ``blas_backend`` ("Accelerate-ILP64") to threadpoolctl's key."""
    if name.startswith("numpy-fallback"):
        return "unknown"
    key = name.lower()
    for suffix in ("-ilp64", "-lp64"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def run_threads() -> RunThreads:
    """Read every thread decision for this run from the machine.

    threadpoolctl cannot see Apple Accelerate, so when it reports no BLAS the
    backend name comes from ``jlinalg.blas_backend``, which found the library
    by ``dlopen``. The import is deferred because ``jamma.lmm.accel`` loads
    the C extension and depends on this module.
    """
    from jamma import jlinalg
    from jamma.lmm import accel

    blas_backend = get_blas_backend()
    if blas_backend == "unknown":
        blas_backend = _jlinalg_backend_key(jlinalg.blas_backend)
    c_ext_available = accel.available()
    return RunThreads(
        blas=get_blas_thread_count(),
        blas_controllable=is_blas_controllable(),
        blas_backend=blas_backend,
        c_ext=get_c_extension_thread_count(c_ext_available, accel.HAS_OPENMP),
        c_ext_openmp=accel.HAS_OPENMP,
        c_ext_available=c_ext_available,
        loco_workers=get_loco_worker_count(),
    )


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


@functools.cache
def _warn_uncontrollable_blas() -> None:
    """Log a one-time warning that BLAS thread control is not active."""
    logger.warning(
        "BLAS thread control is not active — threadpoolctl found no "
        "controllable BLAS library. On macOS with Apple Accelerate this "
        "is expected; Accelerate manages its own threads internally."
    )
