"""Shared SNP filtering utilities for JAMMA.

Provides reusable functions for computing per-SNP statistics and
applying quality control filters (MAF, missing rate, monomorphism, HWE).
Used by both kinship computation and LMM association runners.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def apply_snp_list_mask(
    snp_mask: np.ndarray, indices: np.ndarray, n_snps: int, label: str
) -> None:
    """Apply SNP list restriction to mask in-place with bounds validation.

    Args:
        snp_mask: Boolean mask to modify in-place (AND with list mask).
        indices: Array of SNP indices to include.
        n_snps: Total number of SNPs (for bounds checking).
        label: Human-readable label for log messages (e.g. "Kinship SNP list").
    """
    if len(indices) > 0 and indices.max() >= n_snps:
        raise ValueError(
            f"{label} index {indices.max()} out of range for {n_snps} SNPs"
        )
    list_mask = np.zeros(n_snps, dtype=bool)
    list_mask[indices] = True
    snp_mask &= list_mask
    logger.info(f"{label}: restricting to {len(indices)} requested SNPs")


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


def compute_hwe_pvalues(
    n_aa: np.ndarray, n_ab: np.ndarray, n_bb: np.ndarray
) -> np.ndarray:
    """Compute Hardy-Weinberg equilibrium chi-squared p-values.

    Computes the chi-squared goodness-of-fit test for HWE for each SNP.
    Under HWE, genotype frequencies are p^2 (AA), 2pq (AB), q^2 (BB)
    where p = freq(A), q = 1-p.

    This uses the chi-squared approximation (df=1). Note this differs
    from GEMMA's exact test (Wigginton et al. 2005) but is standard
    for large-sample QC filtering.

    Degenerate SNPs (all same genotype, zero total, or monomorphic)
    return p-value = 1.0 by convention (they pass HWE trivially).

    Args:
        n_aa: Count of homozygous reference genotypes per SNP.
        n_ab: Count of heterozygous genotypes per SNP.
        n_bb: Count of homozygous alternate genotypes per SNP.

    Returns:
        Array of p-values, one per SNP. Values >= threshold pass HWE.
    """
    import jax.numpy as jnp
    import jax.scipy.stats as jax_stats

    n_aa = np.asarray(n_aa, dtype=np.float64)
    n_ab = np.asarray(n_ab, dtype=np.float64)
    n_bb = np.asarray(n_bb, dtype=np.float64)

    n = n_aa + n_ab + n_bb

    with np.errstate(invalid="ignore", divide="ignore"):
        p = (2 * n_aa + n_ab) / (2 * n)
        q = 1.0 - p

        e_aa = n * p**2
        e_ab = 2 * n * p * q
        e_bb = n * q**2

        chi_sq = (
            (n_aa - e_aa) ** 2 / e_aa
            + (n_ab - e_ab) ** 2 / e_ab
            + (n_bb - e_bb) ** 2 / e_bb
        )

    # Degenerate SNPs (monomorphic, zero count) produce NaN chi_sq.
    # Replace with 0.0 so they get p-value = 1.0 (pass HWE by convention).
    chi_sq = np.where(np.isnan(chi_sq), 0.0, chi_sq)

    # Use JAX chi2.sf for p-value computation (avoids scipy runtime dep)
    chi_sq_jax = jnp.asarray(chi_sq)
    pvalues_jax = jax_stats.chi2.sf(chi_sq_jax, df=1)

    # np.asarray is an implicit JAX sync point (per MEMORY.md)
    return np.asarray(pvalues_jax)
