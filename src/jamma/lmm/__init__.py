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
from jamma.lmm.likelihood import compute_Uab, compute_pab
from jamma.lmm.optimize import optimize_lambda_for_snp
from jamma.lmm.stats import AssocResult, calc_wald_test, get_pab_index

__all__ = [
    "run_lmm_association",
    "AssocResult",
    "eigendecompose_kinship",
    "write_assoc_results",
]


def run_lmm_association(
    genotypes: np.ndarray,
    phenotypes: np.ndarray,
    kinship: np.ndarray,
    snp_info: list,
    covariates: np.ndarray | None = None,
) -> list[AssocResult]:
    """Run LMM association tests for all SNPs.

    Full LMM workflow:
    1. Eigendecompose kinship matrix
    2. Rotate phenotype and covariates
    3. For each SNP: optimize lambda, compute Wald statistics

    Args:
        genotypes: Genotype matrix (n_samples, n_snps) with values 0, 1, 2
        phenotypes: Phenotype vector (n_samples,)
        kinship: Kinship matrix (n_samples, n_samples)
        snp_info: List of dicts with keys: chr, rs, pos, a1, a0, maf, n_miss
        covariates: Optional covariate matrix (n_samples, n_covariates)

    Returns:
        List of AssocResult for each SNP
    """
    n_samples, n_snps = genotypes.shape

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
    for i in range(n_snps):
        # Step d: Rotate genotype
        x = genotypes[:, i]
        Utx = U.T @ x

        # Compute Uab with SNP genotype
        Uab = compute_Uab(UtW, Uty, Utx)

        # Optimize lambda via REML
        lambda_opt, logl_H1 = optimize_lambda_for_snp(eigenvalues, Uab, n_cvt)

        # Compute Pab for Wald test
        Hi_eval = 1.0 / (lambda_opt * eigenvalues + 1.0)
        Pab = compute_pab(Hi_eval, Uab, get_pab_index)

        # Compute Wald statistics
        beta, se, p_wald = calc_wald_test(lambda_opt, Pab, n_cvt, n_samples)

        # Build result
        info = snp_info[i]
        result = AssocResult(
            chr=info["chr"],
            rs=info["rs"],
            ps=info["pos"],
            n_miss=info["n_miss"],
            allele1=info["a1"],
            allele0=info["a0"],
            af=info["maf"],
            beta=beta,
            se=se,
            logl_H1=logl_H1,
            l_remle=lambda_opt,
            p_wald=p_wald,
        )
        results.append(result)

    return results
