"""BLAS thread management for numpy operations.

JAMMA has two separate BLAS paths:
- Numpy operations (eigendecomp, UT@G rotation): use system BLAS (MKL on Databricks).
  Controlled by threadpool_limits.
- JAX JIT operations (batch_compute_uab, optimize_lambda): use XLA's bundled Eigen.
  NOT affected by threadpool_limits.

This module provides explicit thread control for the numpy path only.

On macOS with Apple Accelerate, threadpoolctl cannot control the BLAS thread
count (Accelerate has no public thread-count API and ignores VECLIB_MAXIMUM_THREADS
after library init). In this case blas_threads() is a no-op and
is_blas_controllable() returns False so callers can adjust OpenMP thread counts
to avoid oversubscription.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Generator
from contextlib import contextmanager

import psutil
from loguru import logger
from threadpoolctl import threadpool_info, threadpool_limits


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

    Use this for BLAS operations that run without JAX contention
    (e.g. eigendecomp, U.T @ G rotation when JAX isn't computing).
    Unlike get_blas_thread_count(), this does NOT divide by n_jax_devices.

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
    2. JAX device-aware reduction: physical_cores // n_jax_devices
    3. Physical core count via psutil (avoids hyperthreading oversubscription)

    When JAX is configured with multiple virtual CPU devices, MKL threads are
    reduced proportionally to avoid oversubscription. Each JAX device manages
    its own XLA thread pool, so BLAS gets ``physical_cores // n_devices``
    threads, leaving the rest for XLA.

    Returns:
        Positive integer thread count. Capped at os.cpu_count() for the
        env var and single-device paths; proportionally reduced for multi-device.
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

    # Lazy import — calling jax.devices() before configure_jax() would
    # permanently freeze the backend at 1 device.
    try:
        import jax
    except ImportError:
        # NumPy-only install — no JAX available.
        n = max(1, min(physical_cores, max_threads))
        logger.debug(f"BLAS threads: {n} (JAX not installed)")
        return n

    # JAX is importable — jax_config must also be importable (it's part of
    # jamma, not an optional dependency).  Let ImportError propagate if broken.
    from jamma.core.jax_config import is_jax_configured

    if not is_jax_configured():
        # Don't query jax.devices() — it would freeze the backend at 1 device.
        # Expected during kinship and eigendecomp phases (JAX deferred until LMM).
        n = max(1, min(physical_cores, max_threads))
        logger.debug(f"BLAS threads: {n} (JAX not yet initialized)")
        return n

    n_jax_devices = len(jax.devices("cpu"))
    if n_jax_devices > 1:
        n = max(1, physical_cores // n_jax_devices)
        logger.debug(
            f"BLAS threads: {n} ({physical_cores} cores / {n_jax_devices} JAX devices)"
        )
    else:
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


@contextmanager
def blas_threads(n_threads: int | None = None) -> Generator[None, None, None]:
    """Context manager for scoped BLAS thread control.

    Wraps threadpool_limits to centralise default thread count logic.
    Use around numpy BLAS operations (eigendecomp, matmul) -- NOT around
    JAX JIT calls (which use XLA's own thread pool).

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
        "is expected; Accelerate manages its own threads internally. "
        "OpenMP thread counts are reduced automatically to compensate."
    )
