"""Eigendecomposition of kinship matrix.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.

Uses scipy.linalg.eigh (LAPACK) with explicit driver selection and in-place
operation. Thread control is handled by jamma.core.threading via threadpool_limits.

Driver selection:
- dsyevd (driver='evd'): Divide-and-conquer, fastest but O(n^2) workspace.
- dsyevr (driver='evr'): Relatively robust representations, O(n) workspace fallback.

overwrite_a=True avoids an internal copy of the kinship matrix, saving ~65 GB
at 90k samples.
"""

import gc
import time
import warnings

import numpy as np
import psutil
import scipy.linalg
from loguru import logger
from threadpoolctl import threadpool_info

from jamma.core.memory import (
    _dsyevd_workspace_gb,
    check_memory_available,
    estimate_eigendecomp_memory,
    log_memory_snapshot,
)
from jamma.core.threading import blas_threads, get_blas_thread_count


def _select_eigendecomp_driver(n_samples: int) -> str:
    """Select LAPACK driver based on available memory.

    dsyevd (divide-and-conquer): O(n^2) workspace, fastest.
    dsyevr (relatively robust representations): O(n) workspace, slower.

    With overwrite_a=True, K is already allocated, so we only need space for
    eigenvectors (output) and dsyevd workspace.

    Args:
        n_samples: Number of samples (matrix dimension).

    Returns:
        'evd' if dsyevd workspace fits in available memory, 'evr' otherwise.
    """
    available_gb = psutil.virtual_memory().available / 1e9
    dsyevd_workspace_gb = _dsyevd_workspace_gb(n_samples)
    eigenvectors_gb = n_samples**2 * 8 / 1e9
    total_needed = eigenvectors_gb + dsyevd_workspace_gb

    if total_needed * 1.1 < available_gb:
        return "evd"

    logger.warning(
        f"dsyevd workspace ({dsyevd_workspace_gb:.1f}GB) too large for "
        f"available memory ({available_gb:.1f}GB), "
        f"falling back to dsyevr (slower but O(n) workspace)"
    )
    return "evr"


def eigendecompose_kinship(
    K: np.ndarray, threshold: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose kinship matrix, zeroing small eigenvalues.

    GEMMA behavior from EigenDecomp_Zeroed:
    - Eigenvalues < 1e-10 are set to 0
    - Warning if >1 zero eigenvalue
    - Warning if negative eigenvalues remain after thresholding

    Uses scipy.linalg.eigh with explicit LAPACK driver selection instead of
    numpy.linalg.eigh. This enables:
    - overwrite_a=True: destroys input K to avoid internal copy (~65 GB at 90k)
    - Adaptive driver: dsyevd (fast, O(n^2) workspace) with dsyevr fallback
      (slower, O(n) workspace) when available memory is insufficient

    Note: Uses scipy (LAPACK) instead of JAX to support matrices larger than
    46k x 46k samples (JAX hits int32 overflow at ~2.1B elements).

    Args:
        K: Symmetric kinship matrix (n_samples, n_samples). Contents are
            destroyed by overwrite_a=True -- caller should del their reference.
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

    driver = _select_eigendecomp_driver(n_samples)
    logger.info(f"Eigendecomp: driver={driver}, threads={n_threads}, overwrite_a=True")

    # Ensure Fortran order for true in-place operation with overwrite_a
    if not K.flags["F_CONTIGUOUS"]:
        logger.debug("Converting K to Fortran order for in-place eigendecomp")
        K = np.asfortranarray(K)

    start_time = time.perf_counter()
    try:
        with blas_threads(n_threads):
            eigenvalues, eigenvectors = scipy.linalg.eigh(
                K,
                driver=driver,
                overwrite_a=True,
                check_finite=False,
            )
    except MemoryError:
        mem_gb = estimate_eigendecomp_memory(n_samples, driver=driver)
        logger.error(
            f"MemoryError during eigendecomposition of {n_samples:,}x{n_samples:,} "
            f"matrix (driver={driver}). Estimated memory: ~{mem_gb:.1f} GB. "
            f"Consider using a machine with more RAM or reducing sample size."
        )
        raise
    except Exception as e:
        logger.error(f"Eigendecomposition failed: {type(e).__name__}: {e}")
        raise

    del K  # Destroyed by overwrite_a=True
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
