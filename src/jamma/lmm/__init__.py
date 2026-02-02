"""Linear Mixed Model (LMM) association testing.

This module implements GEMMA-compatible LMM association tests using JAX
for accelerated numerical computation. The core algorithm follows
Zhou & Stephens (2012) Nature Genetics.

Key components:
- eigendecompose_kinship: Eigendecomposition with GEMMA-compatible thresholding
- reml_log_likelihood: REML log-likelihood for variance component estimation
- optimize_lambda: Brent's method optimization for variance ratio
- calc_wald_test: Wald test statistics (beta, SE, p-value)
- run_lmm_association: Full LMM workflow for association testing
- run_lmm_association_jax: JAX-optimized batch processing (faster for large datasets)
"""

from pathlib import Path

import numpy as np
from loguru import logger

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.io import IncrementalAssocWriter, write_assoc_results
from jamma.lmm.likelihood import compute_Uab
from jamma.lmm.optimize import optimize_lambda_for_snp
from jamma.lmm.runner_jax import (
    auto_tune_chunk_size,
    run_lmm_association_jax,
    run_lmm_association_streaming,
)
from jamma.lmm.stats import AssocResult, calc_wald_test_from_uab

__all__ = [
    "auto_tune_chunk_size",
    "run_lmm_association",
    "run_lmm_association_jax",
    "run_lmm_association_streaming",
    "AssocResult",
    "eigendecompose_kinship",
    "write_assoc_results",
]


def _compute_snp_stats(
    genotypes: np.ndarray, snp_idx: int
) -> tuple[float, float, int, bool]:
    """Compute MAF, missing rate, missing count, and polymorphism for a SNP.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps)
        snp_idx: SNP index

    Returns:
        Tuple of (maf, miss_rate, n_miss, is_polymorphic)
    """
    x = genotypes[:, snp_idx]
    n_samples = len(x)

    # Count missing
    missing_mask = np.isnan(x)
    n_miss = int(np.sum(missing_mask))
    miss_rate = n_miss / n_samples

    # Compute allele frequency and variance on non-missing samples
    valid_x = x[~missing_mask]
    if len(valid_x) > 0:
        af = np.mean(valid_x) / 2.0  # Divide by 2 because genotypes are 0, 1, 2
        maf = min(af, 1.0 - af)  # Minor allele frequency
        variance = np.var(valid_x)
        is_polymorphic = variance > 0
    else:
        maf = 0.0
        is_polymorphic = False

    return maf, miss_rate, n_miss, is_polymorphic


def run_lmm_association(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list,
    covariates: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    output_path: Path | None = None,
) -> list[AssocResult]:
    """Run LMM association tests for all SNPs.

    Full LMM workflow:
    1. Filter samples with missing phenotypes (-9 or NaN)
    2. Filter SNPs based on MAF and missing rate thresholds
    3. Eigendecompose kinship matrix
    4. Rotate phenotype and covariates
    5. For each SNP: optimize lambda, compute Wald statistics

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2
        phenotypes: Phenotype vector (n_samples,)
        kinship: Kinship matrix (n_samples, n_samples)
        snp_info: List of dicts with keys: chr, rs, pos, a1, a0
        covariates: Optional covariate matrix (n_samples, n_covariates)
        maf_threshold: Minimum MAF for SNP inclusion (default: 0.01)
        miss_threshold: Maximum missing rate for SNP inclusion (default: 0.05)
        output_path: If provided, write results incrementally to this file.
            Returns empty list when output_path is set (results are on disk).
            If None (default), accumulate results in memory and return list.

    Returns:
        List of AssocResult for each SNP that passes filtering.
        Empty list if output_path was provided (results written to disk).
    """
    # Step 0: Filter samples with missing phenotypes (-9 or NaN) or missing covariates
    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9.0)

    # Also filter samples with any missing covariate value
    if covariates is not None:
        # Sample is invalid if ANY covariate is NaN
        valid_covariate = np.all(~np.isnan(covariates), axis=1)
        valid_mask = valid_mask & valid_covariate

    # Apply filtering if any samples are invalid
    if not np.all(valid_mask):
        genotypes = genotypes[valid_mask, :]
        phenotypes = phenotypes[valid_mask]
        kinship = kinship[np.ix_(valid_mask, valid_mask)]
        if covariates is not None:
            covariates = covariates[valid_mask, :]

    n_samples, n_snps = genotypes.shape

    # Step 1: Pre-compute SNP stats and filter
    snp_stats = []
    snp_indices = []
    for i in range(n_snps):
        maf, miss_rate, n_miss, is_polymorphic = _compute_snp_stats(genotypes, i)

        # Apply GEMMA-style filtering: MAF, missing rate, and monomorphism
        if maf >= maf_threshold and miss_rate <= miss_threshold and is_polymorphic:
            snp_indices.append(i)
            snp_stats.append((maf, n_miss))

    # Step a: Eigendecompose kinship
    eigenvalues, U = eigendecompose_kinship(kinship)

    # Step b: Rotate phenotype
    Uty = U.T @ phenotypes

    # Step c: Rotate covariates (intercept if None)
    # CRITICAL: Match GEMMA behavior - do NOT auto-prepend intercept
    # User must include intercept column if desired (GEMMA -c flag behavior)
    if covariates is None:
        W = np.ones((n_samples, 1))
    else:
        W = covariates.astype(np.float64)
        # Warn if first column is not all 1s (missing intercept)
        first_col = W[:, 0]
        if not np.allclose(first_col, 1.0):
            logger.warning(
                "Covariate matrix does not have intercept column "
                "(first column is not all 1s). "
                "Model will NOT include an intercept term."
            )
    UtW = U.T @ W
    n_cvt = W.shape[1]

    results = []
    writer = None
    if output_path is not None:
        writer = IncrementalAssocWriter(output_path)
        writer.__enter__()

    try:
        for j, snp_idx in enumerate(snp_indices):
            # Get pre-computed stats
            maf, n_miss = snp_stats[j]

            # Step d: Rotate genotype (with imputation of missing to mean)
            # For all-missing SNPs (shouldn't happen after filtering), impute to 0.0
            # to match JAX path behavior and avoid NaN propagation
            x = genotypes[:, snp_idx].copy()
            missing = np.isnan(x)
            if np.any(missing):
                mean_val = np.nanmean(x)
                # Handle all-missing case: nanmean returns NaN, use 0.0 instead
                if np.isnan(mean_val):
                    mean_val = 0.0
                x[missing] = mean_val
            Utx = U.T @ x

            # Compute Uab with SNP genotype
            Uab = compute_Uab(UtW, Uty, Utx)

            # Optimize lambda via REML (using Brent minimization for speed)
            lambda_opt, logl_H1 = optimize_lambda_for_snp(eigenvalues, Uab, n_cvt)

            # Compute Wald statistics using new function
            beta, se, p_wald = calc_wald_test_from_uab(
                lambda_opt, eigenvalues, Uab, n_cvt, n_samples
            )

            # Build result
            info = snp_info[snp_idx]
            result = AssocResult(
                chr=info["chr"],
                rs=info["rs"],
                ps=info.get("pos", info.get("ps", 0)),
                n_miss=n_miss,
                allele1=info.get("a1", info.get("allele1", "")),
                allele0=info.get("a0", info.get("allele0", "")),
                af=maf,
                beta=beta,
                se=se,
                logl_H1=logl_H1,
                l_remle=lambda_opt,
                p_wald=p_wald,
            )

            if writer is not None:
                writer.write(result)
            else:
                results.append(result)
    finally:
        if writer is not None:
            writer.__exit__(None, None, None)
            logger.info(f"Wrote {writer.count} results to {output_path}")

    if output_path is not None:
        return []
    return results
