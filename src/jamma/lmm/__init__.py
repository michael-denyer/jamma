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
"""

import numpy as np

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.io import write_assoc_results
from jamma.lmm.likelihood import compute_Uab
from jamma.lmm.optimize import optimize_lambda_for_snp
from jamma.lmm.stats import AssocResult, calc_wald_test_from_uab

__all__ = [
    "run_lmm_association",
    "AssocResult",
    "eigendecompose_kinship",
    "write_assoc_results",
]


def _compute_snp_stats(genotypes: np.ndarray, snp_idx: int) -> tuple[float, float, int]:
    """Compute MAF, missing rate, and missing count for a SNP.

    Args:
        genotypes: Genotype matrix (n_samples, n_snps)
        snp_idx: SNP index

    Returns:
        Tuple of (maf, miss_rate, n_miss)
    """
    x = genotypes[:, snp_idx]
    n_samples = len(x)

    # Count missing
    missing_mask = np.isnan(x)
    n_miss = int(np.sum(missing_mask))
    miss_rate = n_miss / n_samples

    # Compute allele frequency on non-missing samples
    valid_x = x[~missing_mask]
    if len(valid_x) > 0:
        af = np.mean(valid_x) / 2.0  # Divide by 2 because genotypes are 0, 1, 2
        maf = min(af, 1.0 - af)  # Minor allele frequency
    else:
        maf = 0.0

    return maf, miss_rate, n_miss


def run_lmm_association(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list,
    covariates: np.ndarray | None = None,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
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

    Returns:
        List of AssocResult for each SNP that passes filtering
    """
    # Step 0: Filter samples with missing phenotypes (-9 or NaN)
    valid_mask = ~np.isnan(phenotypes) & (phenotypes != -9.0)
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
        maf, miss_rate, n_miss = _compute_snp_stats(genotypes, i)

        # Apply GEMMA-style filtering
        if maf >= maf_threshold and miss_rate <= miss_threshold:
            snp_indices.append(i)
            snp_stats.append((maf, n_miss))

    # Step a: Eigendecompose kinship
    eigenvalues, U = eigendecompose_kinship(kinship)

    # Step b: Rotate phenotype
    Uty = U.T @ phenotypes

    # Step c: Rotate covariates (intercept if None)
    if covariates is None:
        W = np.ones((n_samples, 1))
    else:
        W = np.column_stack([np.ones(n_samples), covariates])
    UtW = U.T @ W
    n_cvt = W.shape[1]

    results = []
    for j, snp_idx in enumerate(snp_indices):
        # Get pre-computed stats
        maf, n_miss = snp_stats[j]

        # Step d: Rotate genotype (with imputation of missing to mean)
        x = genotypes[:, snp_idx].copy()
        missing = np.isnan(x)
        if np.any(missing):
            x[missing] = np.nanmean(x)
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
        results.append(result)

    return results
