"""In-place mean imputation for genotype chunks."""

import numpy as np


def impute_missing_inplace(geno_chunk: np.ndarray, col_means: np.ndarray) -> None:
    """Replace NaN entries in geno_chunk with per-column means, in-place.

    Skips work entirely when no values are missing (common for clean data).
    Only touches NaN positions — non-missing values are untouched.

    Args:
        geno_chunk: Genotype matrix (n_samples, n_snps), modified in-place.
        col_means: Pre-computed column means (n_snps,).
    """
    missing = np.isnan(geno_chunk)
    if missing.any():
        cols = np.where(missing)[1]
        geno_chunk[missing] = col_means[cols]
