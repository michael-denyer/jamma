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

    # Handle all-missing columns: nanmean returns NaN, replace with 0
    # This ensures such SNPs contribute nothing to kinship (centered = 0)
    snp_means = jnp.nan_to_num(snp_means, nan=0.0)

    # Replace NaN with SNP mean (0 for all-missing columns)
    # where(condition, x, y) returns x where condition is True, else y
    X_imputed = jnp.where(jnp.isnan(X), snp_means, X)

    # Center by subtracting mean
    X_centered = X_imputed - snp_means

    return X_centered


@jit
def impute_center_and_standardize(X: jnp.ndarray) -> jnp.ndarray:
    """Impute missing values, center, and standardize by per-SNP standard deviation.

    Implements GEMMA's standardized kinship preprocessing (-gk 2):
    1. Compute mean per SNP excluding missing (NaN)
    2. Replace missing with mean
    3. Center: x = x - mean
    4. Compute variance on imputed data: var = E[X^2] - E[X]^2
    5. Standardize: z = centered / sqrt(var), with zero-variance SNPs set to 0

    GEMMA computes variance over all samples including imputed values.
    The impute-to-mean step makes missing values equal to the mean, so they
    contribute zero to centered values but DO affect the variance denominator
    (sample count is n_samples, not n_observed).

    Args:
        X: Genotype matrix (n_samples, n_snps), NaN for missing values.
            Values are typically 0, 1, or 2 representing minor allele counts.

    Returns:
        Standardized genotype matrix with missing values imputed to SNP mean,
        centered, and divided by per-SNP standard deviation.
        Shape is (n_samples, n_snps), dtype matches input (typically float64).
        Zero-variance SNPs contribute zero (matching GEMMA's geno_var != 0 check).

    Example:
        >>> import jax.numpy as jnp
        >>> X = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
        >>> Z = impute_center_and_standardize(X)
        >>> # Each column is centered and divided by its standard deviation
    """
    # Compute per-SNP mean excluding NaN values
    snp_means = jnp.nanmean(X, axis=0, keepdims=True)

    # Handle all-missing columns: nanmean returns NaN, replace with 0
    snp_means = jnp.nan_to_num(snp_means, nan=0.0)

    # Replace NaN with SNP mean
    X_imputed = jnp.where(jnp.isnan(X), snp_means, X)

    # Center by subtracting mean
    X_centered = X_imputed - snp_means

    # Compute variance AFTER imputation (matching GEMMA):
    # geno_var = E[X^2] - E[X]^2 where sums include imputed values
    snp_var = jnp.mean(X_imputed**2, axis=0, keepdims=True) - snp_means**2

    # Standard deviation
    snp_sd = jnp.sqrt(snp_var)

    # Standardize, guarding against zero variance (monomorphic SNPs)
    # Zero-variance SNPs contribute nothing to kinship
    X_standardized = jnp.where(snp_sd > 0, X_centered / snp_sd, 0.0)

    return X_standardized
