"""Missing data imputation for kinship computation.

This module implements GEMMA's missing data algorithm for kinship
matrix computation. Missing genotypes (NaN values) are imputed to
the per-SNP mean before centering.

GEMMA Algorithm (from PlinkKin):
1. Compute mean per SNP excluding missing (NaN) values
2. Replace missing values with the computed mean
3. Center by subtracting the mean

This approach ensures that missing data has minimal impact on the
kinship matrix while maintaining numerical equivalence with GEMMA.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import jit


@jit
def impute_and_center(X: jnp.ndarray) -> jnp.ndarray:
    """Impute missing values to SNP mean and center.

    Implements GEMMA's PlinkKin algorithm for handling missing data:
    1. Compute mean per SNP excluding missing (NaN)
    2. Replace missing with mean
    3. Center: x = x - mean

    Args:
        X: Genotype matrix (n_samples, n_snps), NaN for missing values.
            Values are typically 0, 1, or 2 representing minor allele counts.

    Returns:
        Centered genotype matrix with missing values imputed to SNP mean.
        Shape is (n_samples, n_snps), dtype matches input (typically float64).

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[0.0, 1.0], [jnp.nan, 2.0], [2.0, 1.0]])
        >>> X_centered = impute_and_center(X)
        >>> # Mean of column 0 is (0+2)/2 = 1.0 (excluding NaN)
        >>> # NaN is replaced with 1.0, then column is centered
    """
    # Compute per-SNP mean excluding NaN values
    # nanmean ignores NaN when computing the mean
    snp_means = jnp.nanmean(X, axis=0, keepdims=True)

    # Replace NaN with SNP mean
    # where(condition, x, y) returns x where condition is True, else y
    X_imputed = jnp.where(jnp.isnan(X), snp_means, X)

    # Center by subtracting mean
    X_centered = X_imputed - snp_means

    return X_centered
