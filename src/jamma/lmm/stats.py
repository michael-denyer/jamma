"""Wald test statistics computation for LMM association.

Implements the Wald test formula from GEMMA's CalcRLWald function.
Uses JAX for F-distribution survival function computation.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import config
from jax.scipy.special import betainc

# Ensure 64-bit precision
config.update("jax_enable_x64", True)


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
    """Compute index into Pab array using GEMMA's GetabIndex formula.

    For symmetric matrix storage, uses lower triangular packing:
    index = a*(a+1)/2 + b for a >= b

    Args:
        a: First index
        b: Second index

    Returns:
        Linear index into packed storage
    """
    if a >= b:
        return a * (a + 1) // 2 + b
    return b * (b + 1) // 2 + a


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
    """Compute Wald test statistics from optimized lambda and Pab matrix.

    Implements GEMMA's CalcRLWald formula:
    - beta = P_xy / P_xx
    - tau = df / Px_yy
    - se = sqrt(1 / (tau * P_xx))
    - f_stat = (P_yy - Px_yy) * tau
    - p_wald = F-distribution survival function

    Args:
        lambda_val: Optimized variance ratio
        Pab: H_inv weighted products from compute_pab
        n_cvt: Number of covariates
        ni_test: Number of samples

    Returns:
        Tuple of (beta, se, p_wald)
    """
    df = ni_test - n_cvt - 1

    # Extract Pab values using GEMMA's indexing convention
    # n_cvt is 0-indexed position of intercept
    # n_cvt + 1 is phenotype (y)
    # n_cvt + 2 is genotype (x) - but in our 0-indexed scheme:
    #   y index = n_cvt (since we have n_cvt covariates, y is next)
    #   x index = n_cvt + 1 (x is after y)
    # Pab indices:
    P_yy = Pab[get_pab_index(n_cvt, n_cvt)]  # Phenotype-phenotype
    P_xx = Pab[get_pab_index(n_cvt + 1, n_cvt + 1)]  # Genotype-genotype
    P_xy = Pab[get_pab_index(n_cvt + 1, n_cvt)]  # Genotype-phenotype

    # Compute residual variance component
    # Px_yy = P_yy - (P_xy^2 / P_xx)
    Px_yy = P_yy - (P_xy * P_xy) / P_xx

    # Compute effect size and standard error
    beta = P_xy / P_xx
    tau = df / Px_yy
    se = np.sqrt(1.0 / (tau * P_xx))

    # Compute F-statistic and p-value
    f_stat = (P_yy - Px_yy) * tau
    p_wald = f_sf(f_stat, 1.0, float(df))

    return beta, se, p_wald
