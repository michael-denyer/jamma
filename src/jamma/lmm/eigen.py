"""Eigendecomposition of kinship matrix.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.

Uses numpy.linalg.eigh (LAPACK) for eigendecomposition. Thread control is
handled by jamma.core.threading via threadpool_limits.

Note: Uses numpy (LAPACK) instead of JAX because JAX's int32 buffer indexing
overflows at ~46k x 46k matrices. With ILP64 numpy (MKL), matrices up to
200k+ are supported.
"""

import gc
import time
import warnings

import numpy as np
from loguru import logger
from threadpoolctl import threadpool_info

from jamma.core.memory import (
    check_memory_available,
    estimate_eigendecomp_memory,
    log_memory_snapshot,
)
from jamma.core.threading import blas_threads, get_blas_thread_count


def eigendecompose_kinship(
    K: np.ndarray, threshold: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose kinship matrix, zeroing small eigenvalues.

    GEMMA behavior from EigenDecomp_Zeroed:
    - Eigenvalues < 1e-10 are set to 0
    - Warning if >1 zero eigenvalue
    - Warning if negative eigenvalues remain after thresholding

    Note: Uses numpy (LAPACK) instead of JAX to support matrices larger than
    46k x 46k samples (JAX hits int32 overflow at ~2.1B elements). With ILP64
    numpy (MKL), matrices up to 200k+ are supported.

    Args:
        K: Symmetric kinship matrix (n_samples, n_samples).
        threshold: Eigenvalues below this are zeroed (default: 1e-10)

    Returns:
        Tuple of (eigenvalues, eigenvectors) where:
        - eigenvalues: (n_samples,) sorted ascending
        - eigenvectors: (n_samples, n_samples) columns are eigenvectors

    Raises:
        ValueError: If kinship matrix is not square or has invalid shape.
        MemoryError: If matrix is too large to decompose.
    """
    n_samples = K.shape[0]
    n_elements = n_samples * n_samples

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kinship matrix must be square, got shape {K.shape}")

    logger.info(f"Eigendecomposing kinship matrix ({n_samples:,} x {n_samples:,})")
    logger.debug(
        f"Matrix elements: {n_elements:,}, memory: ~{n_elements * 8 / 1e9:.1f} GB"
    )

    # Fail fast before LAPACK allocates -- prevents silent SIGKILL from OOM killer
    required_gb = estimate_eigendecomp_memory(n_samples)
    check_memory_available(
        required_gb,
        safety_margin=0.1,
        operation=f"eigendecomposition of {n_samples:,}x{n_samples:,} kinship matrix",
    )

    log_memory_snapshot(f"before_eigendecomp_{n_samples}samples")

    n_threads = get_blas_thread_count()
    blas_libs = [lib for lib in threadpool_info() if lib.get("user_api") == "blas"]
    for lib in blas_libs:
        logger.debug(
            f"BLAS: {lib.get('internal_api')}, "
            f"current={lib.get('num_threads')}, target={n_threads}"
        )

    logger.info(f"Eigendecomp: numpy.linalg.eigh, threads={n_threads}")

    start_time = time.perf_counter()
    try:
        with blas_threads(n_threads):
            eigenvalues, eigenvectors = np.linalg.eigh(K)
    except MemoryError:
        mem_gb = estimate_eigendecomp_memory(n_samples)
        logger.error(
            f"MemoryError during eigendecomposition of {n_samples:,}x{n_samples:,} "
            f"matrix. Estimated memory: ~{mem_gb:.1f} GB. "
            f"Consider using a machine with more RAM or reducing sample size."
        )
        raise
    except Exception as e:
        logger.error(f"Eigendecomposition failed: {type(e).__name__}: {e}")
        raise

    del K
    gc.collect()

    elapsed = time.perf_counter() - start_time
    logger.info(f"Eigendecomposition completed in {elapsed:.2f} seconds")
    log_memory_snapshot(f"after_eigendecomp_{n_samples}samples")

    n_negative = np.sum(eigenvalues < -threshold)
    if n_negative > 0:
        warnings.warn(
            f"Kinship matrix has {n_negative} negative eigenvalue(s). "
            "Matrix may not be positive semi-definite.",
            stacklevel=2,
        )

    eigenvalues = np.where(np.abs(eigenvalues) < threshold, 0.0, eigenvalues)

    n_zero = np.sum(eigenvalues == 0.0)
    if n_zero > 1:
        warnings.warn(
            f"Kinship matrix has {n_zero} eigenvalues close to zero. "
            "Matrix may be rank-deficient.",
            stacklevel=2,
        )

    return eigenvalues, eigenvectors
