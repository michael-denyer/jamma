"""Scalar ports of GEMMA's CalcRLWald, CalcRLScore and CalcLRT.

Production runs ``jamma.lmm.likelihood_numpy``'s batch statistics and the C
kernels; these one-SNP functions keep GEMMA's formulas in scalar form.
"""

from __future__ import annotations

import numpy as np

from jamma.lmm.likelihood import _P_YY_MIN, get_ab_index
from jamma.lmm.special import betainc, chi2_sf


def safe_sqrt(d: float) -> float:
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


def f_sf(x: float, df1: float, df2: float) -> float:
    """F-distribution survival function using regularized incomplete beta.

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
    complement_z = df1 * x / (df2 + df1 * x)
    result = betainc(df2 / 2.0, df1 / 2.0, z, complement_z=complement_z)

    return float(result)


def calc_wald_test(
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

    # Guard against degenerate cases (matches GEMMA behavior)
    # P_xx <= 0 means SNP has no variance after projection
    # Px_yy <= 0 means residual variance is zero or negative (numerical issue)
    if P_xx <= 0.0:
        return float("nan"), float("nan"), float("nan")

    # Clamp Px_yy to prevent negative variance (GEMMA lmm.cpp:854)
    # Only clamp if >= 0 and < _P_YY_MIN; leave negative values to produce NaN
    if Px_yy >= 0.0 and Px_yy < _P_YY_MIN:
        Px_yy = _P_YY_MIN

    # Compute effect size and standard error
    # Use safe_sqrt to handle edge cases where 1/(tau*P_xx) could be slightly negative
    # due to numerical issues (matches GEMMA's safe_sqrt behavior)
    beta = P_xy / P_xx
    tau = float(df) / Px_yy
    se = safe_sqrt(1.0 / (tau * P_xx))

    # Compute F-statistic and p-value
    # F = (SSR_reduced - SSR_full) / (df_reduced - df_full) / (SSR_full / df_full)
    # For single SNP: F = (P_yy - Px_yy) * tau
    f_stat = (P_yy - Px_yy) * tau
    p_wald = f_sf(f_stat, 1.0, float(df))

    return beta, se, p_wald


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
    p_lrt = chi2_sf(lrt_stat)

    return float(p_lrt)


def calc_score_test(
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
    if Px_yy >= 0.0 and Px_yy < _P_YY_MIN:
        Px_yy = _P_YY_MIN

    # Compute beta and se (informational only for Score test)
    beta = P_xy / P_xx
    tau = float(df) / Px_yy
    se = safe_sqrt(1.0 / (tau * P_xx))

    # Score test statistic: F = n * P_xy^2 / (P_yy * P_xx)
    # This is derived from the Score statistic: U^2 / Var(U)
    # where U = x'(y - Xb_0) is the score under null hypothesis
    f_stat = float(ni_test) * (P_xy * P_xy) / (P_yy * P_xx)
    p_score = f_sf(f_stat, 1.0, float(df))

    return beta, se, p_score
