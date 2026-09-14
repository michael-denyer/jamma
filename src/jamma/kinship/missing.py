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
    non-NumPy arrays.

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
    # nanmean materializes a float copy of the entire input. Reduce through
    # boolean masks instead, keeping scratch to two bytes per genotype.
    nan_mask = np.isnan(X)
    observed = ~nan_mask
    counts = observed.sum(axis=0)
    sums = np.sum(X, axis=0, where=observed)
    mean_dtype = sums.dtype if np.issubdtype(sums.dtype, np.inexact) else np.float64
    snp_means = np.divide(
        sums, counts, out=np.zeros_like(sums, dtype=mean_dtype), where=counts > 0
    )
    del observed

    # All-missing columns have mean zero and contribute nothing after centering.
    snp_means = np.nan_to_num(snp_means, nan=0.0)

    # In-place path: writable numpy arrays avoid an O(N*M) copy
    if isinstance(X, np.ndarray) and X.flags.writeable:
        np.copyto(X, snp_means, where=nan_mask)
        X -= snp_means
        return X

    # Copy-based path for immutable or non-writable arrays
    X_imputed = np.where(nan_mask, snp_means, X)
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
    # Preserve this helper's non-mutating contract, including integer inputs
    # whose means and standardized values require floating-point storage.
    values = np.asarray(X)
    dtype = values.dtype if np.issubdtype(values.dtype, np.inexact) else np.float64
    X_centered = impute_and_center(np.array(values, dtype=dtype, copy=True))

    # Compute variance AFTER imputation (matching GEMMA):
    # var(X) = mean((X - mu)^2), computed via einsum to avoid O(N*M) X**2 allocation
    # einsum('ij,ij->j') computes sum of squared elements per column without
    # materializing the full squared matrix intermediate.
    n_samples = X_centered.shape[0]
    snp_var = np.einsum("ij,ij->j", X_centered, X_centered, optimize=True) / n_samples
    snp_var = snp_var[np.newaxis, :]  # shape (1, n_snps) to broadcast with X_centered

    # Standard deviation
    snp_sd = np.sqrt(snp_var)

    # Normalize the existing buffer; no full division result or where copy.
    nonzero = snp_sd > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(X_centered, snp_sd, out=X_centered, where=nonzero)
    X_centered[:, ~nonzero[0]] = 0.0

    return X_centered
