"""Wald test statistics computation for LMM association.

Implements the Wald test formula from GEMMA's CalcRLWald function.
Uses JAX's betainc for the F-distribution survival function.
"""

from dataclasses import dataclass

import numpy as np
from jax.scipy.special import betainc

from jamma.lmm.likelihood import calc_pab, get_ab_index


def _safe_sqrt(d: float) -> float:
    """Safe square root following GEMMA's safe_sqrt behavior.

    GEMMA's safe_sqrt (mathfunc.cpp:122-131):
    - If |d| < 0.001, use abs(d) to tolerate small negative values from rounding
    - If d < 0 after that check, return NaN
    - Otherwise return sqrt(d)

    This handles numerical edge cases where Px_yy becomes slightly negative
    due to floating-point errors in the projection computation.

    Args:
        d: Value to take square root of

    Returns:
        sqrt(d) or sqrt(abs(d)) for small negatives, NaN for large negatives
    """
    if abs(d) < 0.001:
        d = abs(d)
    if d < 0.0:
        return float("nan")
    return np.sqrt(d)


@dataclass
class AssocResult:
    """Association test result for a single SNP.

    Matches GEMMA's output format. Fields present depend on test type:
    - Wald (-lmm 1): logl_H1, l_remle, p_wald
    - LRT (-lmm 2): l_mle, p_lrt (no beta/se in GEMMA output, but kept for consistency)
    - Score (-lmm 3): p_score only (no per-SNP logl_H1/l_remle)
    - All (-lmm 4): All fields
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
    logl_H1: float | None = None  # Not present for Score-only
    l_remle: float | None = None  # Not present for Score-only
    p_wald: float | None = None  # Only for Wald/-lmm 1
    p_score: float | None = None  # Only for Score/-lmm 3
    l_mle: float | None = None  # MLE lambda (for LRT/-lmm 2)
    p_lrt: float | None = None  # LRT p-value (for LRT/-lmm 2)


def f_sf(x: float, df1: float, df2: float) -> float:
    """F-distribution survival function using scipy.

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
    if x <= 0:
        return 1.0
    if not np.isfinite(x):
        return 0.0

    z = df2 / (df2 + df1 * x)
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

    # Guard against degenerate cases (matches JAX path behavior)
    # P_xx <= 0 means SNP has no variance after projection
    # Px_yy <= 0 means residual variance is zero or negative (numerical issue)
    if P_xx <= 0.0:
        return float("nan"), float("nan"), float("nan")

    # Clamp Px_yy like JAX path does for P_yy (GEMMA lmm.cpp:854)
    # Only clamp if >= 0 and < 1e-8; leave negative values to produce NaN
    if Px_yy >= 0.0 and Px_yy < 1e-8:
        Px_yy = 1e-8

    # Compute effect size and standard error
    # Use safe_sqrt to handle edge cases where 1/(tau*P_xx) could be slightly negative
    # due to numerical issues (matches GEMMA's safe_sqrt behavior)
    beta = P_xy / P_xx
    tau = float(df) / Px_yy
    se = _safe_sqrt(1.0 / (tau * P_xx))

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


def calc_lrt_test(
    logl_H1: float,
    logl_H0: float,
) -> float:
    """Compute LRT p-value using chi-squared distribution.

    LRT statistic: 2 * (logl_H1 - logl_H0)
    Under H0, follows chi-squared with df=1.

    Args:
        logl_H1: MLE log-likelihood under alternative (SNP has effect)
        logl_H0: MLE log-likelihood under null (no SNP effect)

    Returns:
        p_lrt: LRT p-value from chi2.sf(stat, df=1)
    """
    lrt_stat = 2.0 * (logl_H1 - logl_H0)

    # Guard against negative statistic (numerical artifact)
    if lrt_stat < 0:
        return 1.0

    # Chi-squared survival function with df=1
    from jax.scipy.stats import chi2

    p_lrt = chi2.sf(lrt_stat, df=1)

    return float(p_lrt)


def calc_score_test(
    lambda_null: float,
    Pab: np.ndarray,
    n_cvt: int,
    ni_test: int,
) -> tuple[float, float, float]:
    """Compute Score test statistics following GEMMA's CalcRLScore.

    The Score test uses fixed null model lambda (computed once, reused for all SNPs)
    rather than per-SNP optimization. This makes it faster than Wald test.

    Key difference from Wald: extracts P_xx, P_xy, P_yy at projection level n_cvt
    (after covariates only), not n_cvt+1 (after covariates AND genotype).

    Args:
        lambda_null: Null model lambda (fixed, same for all SNPs)
        Pab: Pab matrix from calc_pab (n_cvt+2, n_index)
        n_cvt: Number of covariates
        ni_test: Number of samples

    Returns:
        Tuple of (beta, se, p_score) where beta/se are informational only
        (computed under null model, not used in hypothesis testing)
    """
    df = ni_test - n_cvt - 1

    # GEMMA indexing (1-based):
    # - Covariates are indices 1..n_cvt
    # - Genotype is index n_cvt+1
    # - Phenotype is index n_cvt+2
    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    index_xx = get_ab_index(n_cvt + 1, n_cvt + 1, n_cvt)
    index_xy = get_ab_index(n_cvt + 2, n_cvt + 1, n_cvt)

    # KEY DIFFERENCE FROM WALD: Extract at projection level n_cvt (NOT n_cvt+1)
    # Score test extracts values BEFORE projecting out genotype
    # This is the fundamental difference between Score and Wald tests
    P_yy = Pab[n_cvt, index_yy]  # y'Py after projecting out covariates only
    P_xx = Pab[n_cvt, index_xx]  # x'Px after projecting out covariates only
    P_xy = Pab[n_cvt, index_xy]  # x'Py after projecting out covariates only

    # Px_yy for beta/se computation (after projecting out covariates AND genotype)
    Px_yy = Pab[n_cvt + 1, index_yy]

    # Guard against degenerate cases
    # P_xx <= 0 means SNP has no variance after projection (constant genotype)
    if P_xx <= 0.0:
        return float("nan"), float("nan"), float("nan")

    # Clamp Px_yy like Wald test does (GEMMA lmm.cpp:854)
    if Px_yy >= 0.0 and Px_yy < 1e-8:
        Px_yy = 1e-8

    # Compute beta and se (informational only for Score test)
    beta = P_xy / P_xx
    tau = float(df) / Px_yy
    se = _safe_sqrt(1.0 / (tau * P_xx))

    # Score test statistic: F = n * P_xy^2 / (P_yy * P_xx)
    # This is derived from the Score statistic: U^2 / Var(U)
    # where U = x'(y - Xb_0) is the score under null hypothesis
    f_stat = float(ni_test) * (P_xy * P_xy) / (P_yy * P_xx)
    p_score = f_sf(f_stat, 1.0, float(df))

    return beta, se, p_score
