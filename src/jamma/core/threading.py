"""BLAS thread management for numpy operations.

JAMMA has two separate BLAS paths:
- Numpy operations (eigendecomp, UT@G rotation): use system BLAS (MKL on Databricks).
  Controlled by threadpool_limits.
- JAX JIT operations (batch_compute_uab, optimize_lambda): use XLA's bundled Eigen.
  NOT affected by threadpool_limits.

This module provides explicit thread control for the numpy path only.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

import psutil
from loguru import logger
from threadpoolctl import threadpool_limits


def get_blas_thread_count() -> int:
    """Determine the number of BLAS threads to use for numpy operations.

    Priority:
    1. JAMMA_BLAS_THREADS env var (explicit override for benchmarking)
    2. Physical core count via psutil (avoids hyperthreading oversubscription)

    Returns:
        Positive integer thread count, capped at os.cpu_count().
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

    n = psutil.cpu_count(logical=False) or max_threads
    n = max(1, min(n, max_threads))
    logger.debug(f"BLAS threads from physical core count: {n}")
    return n


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

    with threadpool_limits(limits=n_threads, user_api="blas"):
        yield
