"""Eigendecomposition of kinship matrix.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.
Uses scipy.linalg.eigh (LAPACK) to support large matrices (200k+ samples) that
exceed JAX's int32 buffer limits.

Note on threading: OpenBLAS can segfault with multi-threaded eigendecomposition
on large matrices (>50k) due to memory allocation races. We use threadpoolctl
to limit BLAS threads to 1 for matrices above a size threshold.
See: https://github.com/scipy/scipy/issues/8741
"""

import time
import warnings

import numpy as np
from loguru import logger
from scipy import linalg

try:
    from threadpoolctl import threadpool_limits

    HAVE_THREADPOOLCTL = True
except ImportError:
    HAVE_THREADPOOLCTL = False

from jamma.core.memory import (
    check_memory_available,
    estimate_eigendecomp_memory,
    log_memory_snapshot,
)

# Threshold above which we limit BLAS threads to prevent SIGSEGV
# 50k samples = 2.5B elements, empirically safe with threading
# 100k samples = 10B elements, crashes with OpenBLAS multi-threading
SINGLE_THREAD_THRESHOLD = 50_000


def eigendecompose_kinship(
    K: np.ndarray, threshold: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose kinship matrix, zeroing small eigenvalues.

    GEMMA behavior from EigenDecomp_Zeroed:
    - Eigenvalues < 1e-10 are set to 0
    - Warning if >1 zero eigenvalue
    - Warning if negative eigenvalues remain after thresholding

    Note: Uses scipy.linalg.eigh (LAPACK) instead of JAX to support matrices
    larger than 46k x 46k samples (JAX hits int32 overflow at ~2.1B elements).

    Args:
        K: Symmetric kinship matrix (n_samples, n_samples)
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

    # Validate input
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kinship matrix must be square, got shape {K.shape}")

    logger.info(f"## Eigendecomposing kinship matrix ({n_samples:,} x {n_samples:,})")
    logger.debug(
        f"Matrix elements: {n_elements:,}, memory: ~{n_elements * 8 / 1e9:.1f} GB"
    )

    # Pre-flight memory check - fail fast before scipy allocates
    # This prevents silent OOM crashes on Linux where OOM killer sends SIGKILL
    required_gb = estimate_eigendecomp_memory(n_samples)
    check_memory_available(
        required_gb,
        safety_margin=0.1,
        operation=f"eigendecomposition of {n_samples:,}x{n_samples:,} kinship matrix",
    )

    # Use scipy.linalg.eigh which uses LAPACK with int64 indexing
    # This supports matrices up to sqrt(int64_max) ≈ 3 billion rows
    # JAX's jnp.linalg.eigh hits int32 overflow at ~46k x 46k (2.1B elements)
    #
    # Log memory state right before allocation for debugging OOM crashes
    log_memory_snapshot(f"before_eigendecomp_{n_samples}samples")

    # For large matrices, limit BLAS threads to 1 to prevent SIGSEGV
    # OpenBLAS has threading bugs with large eigendecompositions
    # See: https://github.com/scipy/scipy/issues/8741
    use_single_thread = n_samples >= SINGLE_THREAD_THRESHOLD

    if use_single_thread:
        if HAVE_THREADPOOLCTL:
            logger.info(
                f"Using single-threaded BLAS for {n_samples:,}x{n_samples:,} matrix "
                "(prevents OpenBLAS SIGSEGV)"
            )
        else:
            logger.warning(
                f"threadpoolctl not installed - cannot limit BLAS threads for "
                f"{n_samples:,}x{n_samples:,} matrix. "
                "Install with: pip install threadpoolctl"
            )

    start_time = time.perf_counter()
    try:
        if use_single_thread and HAVE_THREADPOOLCTL:
            # Limit all BLAS/LAPACK libraries to single thread
            with threadpool_limits(limits=1, user_api="blas"):
                eigenvalues, eigenvectors = linalg.eigh(K)
        else:
            eigenvalues, eigenvectors = linalg.eigh(K)
    except MemoryError:
        mem_gb = n_elements * 8 * 3 / 1e9
        logger.error(
            f"MemoryError during eigendecomposition of {n_samples:,}x{n_samples:,} "
            f"matrix. Required memory: ~{mem_gb:.1f} GB. "
            f"Consider using a machine with more RAM or reducing sample size."
        )
        raise
    except Exception as e:
        logger.error(f"Eigendecomposition failed: {type(e).__name__}: {e}")
        raise

    elapsed = time.perf_counter() - start_time
    logger.info(f"Eigendecomposition completed in {elapsed:.2f} seconds")

    # Log memory after eigendecomp - shows peak allocation
    log_memory_snapshot(f"after_eigendecomp_{n_samples}samples")

    # Count negative eigenvalues before thresholding
    n_negative = np.sum(eigenvalues < -threshold)
    if n_negative > 0:
        warnings.warn(
            f"Kinship matrix has {n_negative} negative eigenvalue(s). "
            "Matrix may not be positive semi-definite.",
            stacklevel=2,
        )

    # Zero small eigenvalues (GEMMA's behavior)
    eigenvalues = np.where(np.abs(eigenvalues) < threshold, 0.0, eigenvalues)

    # Count zero eigenvalues after thresholding
    n_zero = np.sum(eigenvalues == 0.0)
    if n_zero > 1:
        warnings.warn(
            f"Kinship matrix has {n_zero} eigenvalues close to zero. "
            "Matrix may be rank-deficient.",
            stacklevel=2,
        )

    return eigenvalues, eigenvectors
