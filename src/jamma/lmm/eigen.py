"""Eigendecomposition of kinship matrix.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.
Uses scipy.linalg.eigh (LAPACK) to support large matrices (200k+ samples) that
exceed JAX's int32 buffer limits.
"""

import time
import warnings

import numpy as np
from loguru import logger
from scipy import linalg


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

    # Use scipy.linalg.eigh which uses LAPACK with int64 indexing
    # This supports matrices up to sqrt(int64_max) ≈ 3 billion rows
    # JAX's jnp.linalg.eigh hits int32 overflow at ~46k x 46k (2.1B elements)
    start_time = time.perf_counter()
    try:
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
