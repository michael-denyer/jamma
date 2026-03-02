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

import numpy as np


def impute_and_center(X: np.ndarray) -> np.ndarray:
    """Impute missing values to SNP mean and center.

    When X is a writable NumPy array, operates in-place for zero-copy
    performance. Falls back to a copy-based path for non-writable or
    non-NumPy arrays (e.g., JAX arrays in streaming kinship).

    Implements GEMMA's PlinkKin algorithm for handling missing data:
    1. Compute mean per SNP excluding missing (NaN)
    2. Replace missing with mean
    3. Center: x -= mean

    Args:
        X: Genotype matrix (n_samples, n_snps), NaN for missing values.
            For in-place operation, must be a writable NumPy float array.
            If X is a view into a larger array, the underlying data will
            be mutated.

    Returns:
        Centered array with missing values imputed to SNP mean.
        Same object as X when in-place path is taken; new array otherwise.

    Example:
        >>> import numpy as np
        >>> X = np.array([[0.0, 1.0], [np.nan, 2.0], [2.0, 1.0]])
        >>> X_centered = impute_and_center(X)
        >>> # Mean of column 0 is (0+2)/2 = 1.0 (excluding NaN)
        >>> # NaN is replaced with 1.0, then column is centered
    """
    # Compute per-SNP mean excluding NaN values; shape (n_snps,)
    snp_means = np.nanmean(X, axis=0)

    # Handle all-missing columns: nanmean returns NaN, replace with 0
    # This ensures such SNPs contribute nothing to kinship (centered = 0)
    snp_means = np.nan_to_num(snp_means, nan=0.0)

    # In-place path: writable numpy arrays avoid an O(N*M) copy
    if isinstance(X, np.ndarray) and X.flags.writeable:
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = np.take(snp_means, np.where(nan_mask)[1])
        X -= snp_means
        return X

    # Copy-based path for immutable arrays (e.g., JAX arrays in streaming kinship)
    X_imputed = np.where(np.isnan(X), snp_means, X)
    return X_imputed - snp_means


def impute_center_and_standardize(X: np.ndarray) -> np.ndarray:
    """Impute missing values, center, and standardize by per-SNP standard deviation.

    Implements GEMMA's standardized kinship preprocessing (-gk 2):
    1. Compute mean per SNP excluding missing (NaN)
    2. Replace missing with mean
    3. Center: x = x - mean
    4. Compute variance from centered data: var = mean((X - mean)^2)
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
        >>> import numpy as np
        >>> X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
        >>> Z = impute_center_and_standardize(X)
        >>> # Each column is centered and divided by its standard deviation
    """
    # Compute per-SNP mean excluding NaN values
    snp_means = np.nanmean(X, axis=0, keepdims=True)

    # Handle all-missing columns: nanmean returns NaN, replace with 0
    snp_means = np.nan_to_num(snp_means, nan=0.0)

    # Replace NaN with SNP mean
    X_imputed = np.where(np.isnan(X), snp_means, X)

    # Center by subtracting mean
    X_centered = X_imputed - snp_means

    # Compute variance AFTER imputation (matching GEMMA):
    # var(X) = mean((X - mu)^2), computed via einsum to avoid O(N*M) X**2 allocation
    # einsum('ij,ij->j') computes sum of squared elements per column without
    # materializing the full squared matrix intermediate.
    n_samples = X_centered.shape[0]
    snp_var = np.einsum("ij,ij->j", X_centered, X_centered, optimize=True) / n_samples
    snp_var = snp_var[np.newaxis, :]  # shape (1, n_snps) to broadcast with X_centered

    # Standard deviation
    snp_sd = np.sqrt(snp_var)

    # Standardize, guarding against zero variance (monomorphic SNPs)
    # Zero-variance SNPs contribute nothing to kinship
    X_standardized = np.where(snp_sd > 0, X_centered / snp_sd, 0.0)

    return X_standardized
