"""Wald test statistics computation for LMM association.

Implements the Wald test formula from GEMMA's CalcRLWald function.
Uses scipy for F-distribution survival function (CPU-native, no JAX overhead).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import betainc

from jamma.lmm.likelihood import calc_pab, get_ab_index


@dataclass
class AssocResult:
    """Association test result for a single SNP.

    Matches GEMMA's output format for -lmm 1 (Wald test).
    """

    chr: str
    rs: str
    ps: int  # base position
    n_miss: int  # missing count for this SNP
    allele1: str  # minor allele
    allele0: str  # major allele
    af: float  # allele frequency
    beta: float
    se: float
    logl_H1: float
    l_remle: float  # lambda REMLE
    p_wald: float


def get_pab_index(a: int, b: int) -> int:
    """Legacy index function - wrapper for get_ab_index with n_cvt=1.

    For backwards compatibility with code that assumes n_cvt=1.
    New code should use get_ab_index directly.

    Args:
        a: First index (0-based, will be converted to 1-based)
        b: Second index (0-based, will be converted to 1-based)

    Returns:
        Linear index into packed storage
    """
    # Convert from 0-based to 1-based for GEMMA convention
    return get_ab_index(a + 1, b + 1, n_cvt=1)


def f_sf(x: float, df1: float, df2: float) -> float:
    """F-distribution survival function using JAX.

    Computes P(F > x) for F-distributed random variable with df1 and df2
    degrees of freedom. Uses the regularized incomplete beta function
    for numerical stability with small p-values.

    The relationship is:
    SF(x) = 1 - CDF(x) = I_{df2/(df2 + df1*x)}(df2/2, df1/2)

    where I_x(a, b) is the regularized incomplete beta function.

    Args:
        x: F statistic value
        df1: Numerator degrees of freedom
        df2: Denominator degrees of freedom

    Returns:
        Survival function value (p-value for F-test)
    """
    # Handle edge cases
    if x <= 0:
        return 1.0
    if not np.isfinite(x):
        return 0.0

    # Compute beta function argument
    # For F with df1, df2: SF = I_{df2/(df2 + df1*x)}(df2/2, df1/2)
    z = df2 / (df2 + df1 * x)

    # Use JAX betainc (regularized incomplete beta function)
    # betainc(a, b, x) = I_x(a, b)
    result = betainc(df2 / 2.0, df1 / 2.0, z)

    return float(result)


def calc_wald_test(
    lambda_val: float,
    Pab: np.ndarray,
    n_cvt: int,
    ni_test: int,
) -> tuple[float, float, float]:
    """Compute Wald test statistics following GEMMA's CalcRLWald exactly.

    From GEMMA lmm.cpp CalcRLWald:
    - P_yy = Pab[n_cvt, index_yy]      (y'Py after projecting out covariates)
    - P_xx = Pab[n_cvt, index_xx]      (x'Px after projecting out covariates)
    - P_xy = Pab[n_cvt, index_xy]      (x'Py after projecting out covariates)
    - Px_yy = Pab[n_cvt+1, index_yy]   (y'Py after projecting out covariates AND X)
    - beta = P_xy / P_xx
    - tau = df / Px_yy
    - se = sqrt(1 / (tau * P_xx))
    - p_wald = F-distribution survival function((P_yy - Px_yy) * tau, 1, df)

    Args:
        lambda_val: Optimized variance ratio (unused here, kept for API compat)
        Pab: Pab matrix from calc_pab (n_cvt+2, n_index)
        n_cvt: Number of covariates
        ni_test: Number of samples

    Returns:
        Tuple of (beta, se, p_wald)
    """
    df = ni_test - n_cvt - 1

    # GEMMA indexing (1-based):
    # - Covariates are indices 1..n_cvt
    # - Genotype is index n_cvt+1
    # - Phenotype is index n_cvt+2
    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    index_xx = get_ab_index(n_cvt + 1, n_cvt + 1, n_cvt)
    index_xy = get_ab_index(n_cvt + 2, n_cvt + 1, n_cvt)

    # Extract Pab values at the appropriate projection level
    # After projecting out n_cvt covariates (row index = n_cvt, 0-based)
    P_yy = Pab[n_cvt, index_yy]
    P_xx = Pab[n_cvt, index_xx]
    P_xy = Pab[n_cvt, index_xy]

    # After projecting out covariates AND genotype (row index = n_cvt+1, 0-based)
    Px_yy = Pab[n_cvt + 1, index_yy]

    # Compute effect size and standard error
    beta = P_xy / P_xx
    tau = float(df) / Px_yy
    se = np.sqrt(1.0 / (tau * P_xx))

    # Compute F-statistic and p-value
    # F = (SSR_reduced - SSR_full) / (df_reduced - df_full) / (SSR_full / df_full)
    # For single SNP: F = (P_yy - Px_yy) * tau
    f_stat = (P_yy - Px_yy) * tau
    p_wald = f_sf(f_stat, 1.0, float(df))

    return beta, se, p_wald


def calc_wald_test_from_uab(
    lambda_val: float,
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
    ni_test: int,
) -> tuple[float, float, float]:
    """Compute Wald test from Uab matrix directly.

    This combines calc_pab and calc_wald_test for convenience.

    Args:
        lambda_val: Optimized variance ratio
        eigenvalues: Eigenvalues of kinship matrix
        Uab: Matrix products from compute_Uab
        n_cvt: Number of covariates
        ni_test: Number of samples

    Returns:
        Tuple of (beta, se, p_wald)
    """
    # Compute Hi_eval
    Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)

    # Compute Pab
    Pab = calc_pab(n_cvt, Hi_eval, Uab)

    # Compute Wald test
    return calc_wald_test(lambda_val, Pab, n_cvt, ni_test)
