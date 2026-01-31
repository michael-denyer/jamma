"""Eigendecomposition of kinship matrix using JAX.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.
Uses JAX's XLA-compiled eigh for performance (6-7x faster than SciPy/LAPACK).
"""

import warnings

import jax.numpy as jnp
import numpy as np
from jax import config

# Ensure 64-bit precision for numerical equivalence
config.update("jax_enable_x64", True)


def eigendecompose_kinship(
    K: np.ndarray, threshold: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose kinship matrix, zeroing small eigenvalues.

    GEMMA behavior from EigenDecomp_Zeroed:
    - Eigenvalues < 1e-10 are set to 0
    - Warning if >1 zero eigenvalue
    - Warning if negative eigenvalues remain after thresholding

    Args:
        K: Symmetric kinship matrix (n_samples, n_samples)
        threshold: Eigenvalues below this are zeroed (default: 1e-10)

    Returns:
        Tuple of (eigenvalues, eigenvectors) where:
        - eigenvalues: (n_samples,) sorted ascending
        - eigenvectors: (n_samples, n_samples) columns are eigenvectors
    """
    # Convert to JAX array for computation
    K_jax = jnp.array(K, dtype=jnp.float64)

    # JAX eigh returns eigenvalues in ascending order (same as LAPACK)
    eigenvalues, eigenvectors = jnp.linalg.eigh(K_jax)

    # Convert back to numpy for thresholding with warnings
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors)

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
