"""JAX kinship matrix computation.

This module provides the main kinship matrix computation function,
implementing GEMMA's centered relatedness matrix algorithm (-gk 1 mode).

The kinship matrix K is computed as:
    K = (1/p) * X_c @ X_c.T

where X_c is the centered genotype matrix with missing values imputed
to per-SNP mean, and p is the number of SNPs.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import jit

from jamma.core import configure_jax
from jamma.kinship.missing import impute_and_center

# Ensure 64-bit precision for GEMMA equivalence
configure_jax()


@jit
def _accumulate_kinship(K: jnp.ndarray, X_centered: jnp.ndarray) -> jnp.ndarray:
    """Accumulate kinship contribution from centered SNP batch.

    Args:
        K: Current kinship matrix accumulator (n_samples, n_samples)
        X_centered: Centered genotype batch (n_samples, batch_snps)

    Returns:
        Updated kinship matrix with batch contribution added.
    """
    return K + jnp.matmul(X_centered, X_centered.T)


def compute_centered_kinship(
    genotypes: np.ndarray,
    batch_size: int = 10000,
) -> np.ndarray:
    """Compute centered relatedness matrix (GEMMA -gk 1).

    Implements: K = (1/p) * X_c @ X_c.T
    where X_c is centered with missing values imputed to SNP mean.

    GEMMA's PlinkKin algorithm:
    1. For each SNP batch: impute missing to mean, center
    2. Accumulate K += X_batch @ X_batch.T
    3. Scale K /= n_snps

    Args:
        genotypes: Genotype matrix (n_samples, n_snps), NaN for missing.
            Values are typically 0, 1, or 2 representing minor allele counts.
        batch_size: SNPs per batch (default 10000, matches GEMMA).
            Batching prevents memory issues with large SNP counts.

    Returns:
        Kinship matrix (n_samples, n_samples), symmetric, scaled by n_snps.

    Example:
        >>> import numpy as np
        >>> X = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 0]], dtype=np.float64)
        >>> K = compute_centered_kinship(X)
        >>> K.shape
        (3, 3)
        >>> np.allclose(K, K.T)  # Symmetric
        True
    """
    n_samples, n_snps = genotypes.shape

    # Convert to JAX array
    X = jnp.array(genotypes, dtype=jnp.float64)

    # Initialize kinship accumulator
    K = jnp.zeros((n_samples, n_samples), dtype=jnp.float64)

    # Process SNPs in batches
    for start in range(0, n_snps, batch_size):
        end = min(start + batch_size, n_snps)
        X_batch = X[:, start:end]

        # Impute and center the batch
        X_centered = impute_and_center(X_batch)

        # Accumulate kinship contribution
        K = _accumulate_kinship(K, X_centered)

    # Scale by number of SNPs
    K = K / n_snps

    # Return as numpy array for downstream compatibility
    return np.array(K)
