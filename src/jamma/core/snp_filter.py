"""Shared SNP filtering utilities for JAMMA.

Provides reusable functions for computing per-SNP statistics and
applying quality control filters (MAF, missing rate, monomorphism).
Used by both kinship computation and LMM association runners.
"""

import numpy as np


def compute_snp_stats(
    genotypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-SNP mean, missing count, and variance.

    Handles all-NaN columns gracefully by replacing NaN outputs with 0.0.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with NaN for missing.

    Returns:
        Tuple of (col_means, miss_counts, col_vars) where each is a
        1-D array of length n_snps.
    """
    miss_counts = np.sum(np.isnan(genotypes), axis=0)
    with np.errstate(invalid="ignore"):
        col_means = np.nanmean(genotypes, axis=0)
        col_vars = np.nanvar(genotypes, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    col_vars = np.nan_to_num(col_vars, nan=0.0)
    return col_means, miss_counts, col_vars


def compute_snp_filter_mask(
    col_means: np.ndarray,
    miss_counts: np.ndarray,
    col_vars: np.ndarray,
    n_samples: int,
    maf_threshold: float,
    miss_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute boolean mask for SNPs passing QC filters.

    Applies three filters matching GEMMA behavior:
    1. Minor allele frequency >= maf_threshold
    2. Missing rate <= miss_threshold
    3. Polymorphic (variance > 0)

    Args:
        col_means: Per-SNP genotype means from compute_snp_stats.
        miss_counts: Per-SNP missing counts from compute_snp_stats.
        col_vars: Per-SNP genotype variances from compute_snp_stats.
        n_samples: Total sample count for computing missing rates.
        maf_threshold: Minimum MAF for inclusion.
        miss_threshold: Maximum missing rate for inclusion.

    Returns:
        Tuple of (snp_mask, allele_freqs, mafs) where snp_mask is a
        boolean array indicating which SNPs pass all filters.
    """
    miss_rates = miss_counts / n_samples
    allele_freqs = col_means / 2.0
    mafs = np.minimum(allele_freqs, 1.0 - allele_freqs)
    is_polymorphic = col_vars > 0
    snp_mask = (mafs >= maf_threshold) & (miss_rates <= miss_threshold) & is_polymorphic
    return snp_mask, allele_freqs, mafs
