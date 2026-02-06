"""REML/MLE log-likelihood computation following GEMMA's exact algorithm.

Implements restricted maximum likelihood (REML) and maximum likelihood (MLE)
functions for variance component estimation in LMM. This closely follows
GEMMA's lmm.cpp CalcPab, LogRL_f, LogL_f, and CalcRLWald functions.

Also provides null model optimization via golden section search for Score
and LRT tests.

Key data structures:
- Uab: 2D matrix (n_samples x n_index) storing element-wise products of rotated vectors
- Pab: 2D matrix (n_cvt+2 x n_index) storing H-inv weighted projections
- Hi_eval: 1/(lambda * eigenvalues + 1) weighting vector

Reference: Zhou & Stephens (2012) Nature Genetics, Supplementary Information
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def get_ab_index(a: int, b: int, n_cvt: int) -> int:
    """Compute index for accessing Uab/Pab elements using GEMMA's GetabIndex.

    GEMMA uses upper triangular storage with 1-based indices:
    index = (2 * cols - a1 + 2) * (a1 - 1) / 2 + b1 - a1

    where cols = n_cvt + 2, and a1 <= b1 (swapped if necessary).

    Args:
        a: First index (1-based in GEMMA convention)
        b: Second index (1-based in GEMMA convention)
        n_cvt: Number of covariates

    Returns:
        Linear index into packed storage
    """
    cols = n_cvt + 2
    a1, b1 = (a, b) if a <= b else (b, a)
    return (2 * cols - a1 + 2) * (a1 - 1) // 2 + b1 - a1


def compute_Uab(
    UtW: np.ndarray, Uty: np.ndarray, Utx: np.ndarray | None = None
) -> np.ndarray:
    """Compute Uab matrix following GEMMA's CalcUab exactly.

    Uab is a 2D matrix (n_samples × n_index) storing element-wise products
    of rotated vectors. Each column stores the product u_a * u_b for indices
    a and b, where:
    - Columns 1..n_cvt are the rotated covariates (UtW)
    - Column n_cvt+1 is the rotated genotype (Utx) - if provided
    - Column n_cvt+2 is the rotated phenotype (Uty)

    The indexing follows GEMMA's GetabIndex formula.

    Args:
        UtW: Rotated covariates (n_samples, n_cvt)
        Uty: Rotated phenotype (n_samples,)
        Utx: Rotated genotype for current SNP (n_samples,) - optional

    Returns:
        Uab matrix (n_samples, n_index)
    """
    n = len(Uty)
    n_cvt = UtW.shape[1] if UtW.ndim > 1 else 1
    UtW = UtW.reshape(n, -1) if UtW.ndim == 1 else UtW

    # Fast path for n_cvt=1 (most common case)
    if n_cvt == 1:
        return _compute_Uab_ncvt1(UtW, Uty, Utx)

    # General case for n_cvt > 1
    return _compute_Uab_general(UtW, Uty, Utx, n_cvt)


def _compute_Uab_ncvt1(
    UtW: np.ndarray, Uty: np.ndarray, Utx: np.ndarray | None
) -> np.ndarray:
    """Optimized Uab computation for n_cvt=1 (intercept only).

    For n_cvt=1, indices are:
    - (1,1)=0: WW, (1,2)=1: WX, (1,3)=2: WY
    - (2,2)=3: XX, (2,3)=4: XY
    - (3,3)=5: YY

    This vectorized implementation avoids nested loops entirely.
    """
    n = len(Uty)
    w = UtW[:, 0]  # Intercept column

    # Pre-allocate output
    Uab = np.zeros((n, 6), dtype=np.float64)

    # Covariate and phenotype products (always computed)
    Uab[:, 0] = w * w  # (1,1): WW
    Uab[:, 2] = w * Uty  # (1,3): WY
    Uab[:, 5] = Uty * Uty  # (3,3): YY

    # Genotype products (only if Utx provided)
    if Utx is not None:
        Uab[:, 1] = w * Utx  # (1,2): WX
        Uab[:, 3] = Utx * Utx  # (2,2): XX
        Uab[:, 4] = Utx * Uty  # (2,3): XY

    return Uab


def _compute_Uab_general(
    UtW: np.ndarray, Uty: np.ndarray, Utx: np.ndarray | None, n_cvt: int
) -> np.ndarray:
    """General Uab computation for arbitrary n_cvt.

    Uses pre-computed index mapping to avoid repeated get_ab_index calls.
    """
    n = len(Uty)
    n_index = (n_cvt + 2 + 1) * (n_cvt + 2) // 2
    Uab = np.zeros((n, n_index), dtype=np.float64)

    # Build combined vector matrix: [W1, W2, ..., W_ncvt, X, Y]
    # where X is genotype (placeholder if None) and Y is phenotype
    if Utx is not None:
        vectors = np.column_stack([UtW, Utx, Uty])  # (n, n_cvt+2)
    else:
        vectors = np.column_stack([UtW, np.zeros(n), Uty])  # Placeholder for X

    # Pre-compute all index pairs (a, b) and their linear indices
    # Using 1-based indexing as per GEMMA convention
    for a in range(1, n_cvt + 3):
        for b in range(a, n_cvt + 3):
            # Skip genotype if not provided
            if Utx is None and (a == n_cvt + 1 or b == n_cvt + 1):
                continue

            idx = get_ab_index(a, b, n_cvt)
            # vectors column a-1 corresponds to index a (1-based)
            Uab[:, idx] = vectors[:, a - 1] * vectors[:, b - 1]

    return Uab


def calc_pab(
    n_cvt: int,
    Hi_eval: np.ndarray,
    Uab: np.ndarray,
) -> np.ndarray:
    """Compute Pab matrix following GEMMA's CalcPab exactly.

    Pab stores v_a P_p v_b quantities where P_p is the projection matrix.
    The computation uses a recursive formula:

    For p=0 (row 0):
        Pab[0, index_ab] = dot(Hi_eval, Uab[:, index_ab])

    For p>0 (rows 1..n_cvt+1):
        Pab[p, index_ab] = Pab[p-1, index_ab] -
                           Pab[p-1, index_aw] * Pab[p-1, index_bw] / Pab[p-1, index_ww]

    where w = p (the covariate being projected out).

    GEMMA indexing (1-based):
    - p from 0 to n_cvt+1 (projection levels)
    - a from p+1 to n_cvt+2 (vector indices)
    - b from a to n_cvt+2 (symmetric)

    Args:
        n_cvt: Number of covariates
        Hi_eval: 1 / (lambda * eigenvalues + 1) vector (n_samples,)
        Uab: Matrix products from compute_Uab (n_samples, n_index)

    Returns:
        Pab matrix (n_cvt+2, n_index)
    """
    return _calc_pab_general(n_cvt, Hi_eval, Uab)


def _calc_pab_general(n_cvt: int, Hi_eval: np.ndarray, Uab: np.ndarray) -> np.ndarray:
    """General Pab computation for arbitrary n_cvt.

    Row 0 is vectorized; subsequent rows use the recursive formula.
    """
    n_index = (n_cvt + 2 + 1) * (n_cvt + 2) // 2
    Pab = np.zeros((n_cvt + 2, n_index), dtype=np.float64)

    # Row 0: Vectorized weighted dot products
    Pab[0, :] = Hi_eval @ Uab

    # Rows 1 to n_cvt+1: Recursive projection
    for p in range(1, n_cvt + 2):
        for a in range(p + 1, n_cvt + 3):
            for b in range(a, n_cvt + 3):
                index_ab = get_ab_index(a, b, n_cvt)
                index_aw = get_ab_index(a, p, n_cvt)
                index_bw = get_ab_index(b, p, n_cvt)
                index_ww = get_ab_index(p, p, n_cvt)

                ps_ab = Pab[p - 1, index_ab]
                ps_aw = Pab[p - 1, index_aw]
                ps_bw = Pab[p - 1, index_bw]
                ps_ww = Pab[p - 1, index_ww]

                if ps_ww != 0:
                    Pab[p, index_ab] = ps_ab - ps_aw * ps_bw / ps_ww
                else:
                    Pab[p, index_ab] = ps_ab

    return Pab


def calc_iab(
    n_cvt: int,
    Uab: np.ndarray,
) -> np.ndarray:
    """Compute Iab matrix (identity-weighted Pab for logdet_hiw).

    This is the same as calc_pab but with Hi_eval = all ones.
    Used for computing |WHiW| - |WW| in REML.

    Args:
        n_cvt: Number of covariates
        Uab: Matrix products from compute_Uab (n_samples, n_index)

    Returns:
        Iab matrix (n_cvt+2, n_index)
    """
    n_samples = Uab.shape[0]
    ones = np.ones(n_samples, dtype=np.float64)
    return calc_pab(n_cvt, ones, Uab)


def reml_log_likelihood(
    lambda_val: float, eigenvalues: np.ndarray, Uab: np.ndarray, n_cvt: int
) -> float:
    """Compute REML log-likelihood following GEMMA's LogRL_f exactly.

    The REML log-likelihood is:
    f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * log(P_yy)

    where:
    - c = 0.5 * df * (log(df) - log(2*pi) - 1)
    - logdet_h = sum(log(lambda * eval + 1))
    - logdet_hiw = sum(log(Pab[i,ww])) - sum(log(Iab[i,ww])) for covariates
    - P_yy = Pab[nc_total, index_yy]
    - df = n - n_cvt - 1

    Args:
        lambda_val: Variance component ratio (sigma_g^2 / sigma_e^2)
        eigenvalues: Eigenvalues of kinship matrix (n_samples,)
        Uab: Matrix products from compute_Uab (n_samples, n_index)
        n_cvt: Number of covariates

    Returns:
        Log-likelihood value (positive for maximization)
    """
    n = len(eigenvalues)
    nc_total = n_cvt + 1
    df = n - n_cvt - 1

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)))

    Pab = calc_pab(n_cvt, Hi_eval, Uab)
    Iab = calc_iab(n_cvt, Uab)

    logdet_hiw = 0.0
    for i in range(nc_total):
        index_ww = get_ab_index(i + 1, i + 1, n_cvt)
        d_pab = Pab[i, index_ww]
        d_iab = Iab[i, index_ww]
        if d_pab > 0:
            logdet_hiw += np.log(d_pab)
        if d_iab > 0:
            logdet_hiw -= np.log(d_iab)

    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    P_yy = Pab[nc_total, index_yy]

    P_YY_MIN = 1e-8
    if P_yy >= 0.0 and P_yy < P_YY_MIN:
        P_yy = P_YY_MIN

    c = 0.5 * df * (np.log(df) - np.log(2 * np.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)

    return f


def reml_log_likelihood_null(
    lambda_val: float, eigenvalues: np.ndarray, Uab: np.ndarray, n_cvt: int
) -> float:
    """Compute REML log-likelihood for NULL model (no genotype effect).

    Key differences from alternative model (reml_log_likelihood):
    - nc_total = n_cvt (not n_cvt + 1) - no genotype column
    - df = n - n_cvt (not n - n_cvt - 1) - different degrees of freedom
    - logdet_hiw loop iterates to n_cvt (not n_cvt + 1)

    This matches GEMMA's LogRL_f with calc_null=true.

    Args:
        lambda_val: Variance component ratio (sigma_g^2 / sigma_e^2)
        eigenvalues: Eigenvalues of kinship matrix (n_samples,)
        Uab: Matrix products from compute_Uab (n_samples, n_index)
        n_cvt: Number of covariates

    Returns:
        Log-likelihood value (positive for maximization)
    """
    n = len(eigenvalues)
    nc_total = n_cvt  # NULL MODEL: no genotype column
    df = n - n_cvt  # NULL MODEL: different degrees of freedom

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)))

    Pab = calc_pab(n_cvt, Hi_eval, Uab)
    Iab = calc_iab(n_cvt, Uab)

    # logdet_hiw: iterate to nc_total (n_cvt for null, not n_cvt+1)
    logdet_hiw = 0.0
    for i in range(nc_total):
        index_ww = get_ab_index(i + 1, i + 1, n_cvt)
        d_pab = Pab[i, index_ww]
        d_iab = Iab[i, index_ww]
        if d_pab > 0:
            logdet_hiw += np.log(d_pab)
        if d_iab > 0:
            logdet_hiw -= np.log(d_iab)

    # P_yy at level nc_total (n_cvt for null model)
    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    P_yy = Pab[nc_total, index_yy]

    P_YY_MIN = 1e-8
    if P_yy >= 0.0 and P_yy < P_YY_MIN:
        P_yy = P_YY_MIN

    c = 0.5 * df * (np.log(df) - np.log(2 * np.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)

    return f


def mle_log_likelihood_null(
    lambda_val: float, eigenvalues: np.ndarray, Uab: np.ndarray, n_cvt: int
) -> float:
    """Compute MLE log-likelihood for NULL model (no genotype effect).

    Key differences from alternative model (mle_log_likelihood):
    - nc_total = n_cvt (not n_cvt + 1) - no genotype column
    - P_yy extracted at Pab[n_cvt, index_yy]

    No logdet_hiw term (same as alternative MLE).

    This matches GEMMA's LogL_f with calc_null=true.

    Args:
        lambda_val: Variance component ratio (sigma_g^2 / sigma_e^2)
        eigenvalues: Eigenvalues of kinship matrix (n_samples,)
        Uab: Matrix products from compute_Uab (n_samples, n_index)
        n_cvt: Number of covariates

    Returns:
        Log-likelihood value (positive for maximization)
    """
    n = len(eigenvalues)
    nc_total = n_cvt  # NULL MODEL: no genotype column

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)))

    Pab = calc_pab(n_cvt, Hi_eval, Uab)

    # P_yy at level nc_total (n_cvt for null model)
    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    P_yy = Pab[nc_total, index_yy]

    P_YY_MIN = 1e-8
    if P_yy >= 0.0 and P_yy < P_YY_MIN:
        P_yy = P_YY_MIN

    # MLE formula (uses n, not df; no logdet_hiw)
    c = 0.5 * n * (np.log(n) - np.log(2 * np.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * n * np.log(P_yy)

    return f


def _golden_section_minimize(
    func: Callable[[float], float],
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[float, float]:
    """Minimize a scalar function over [l_min, l_max] using golden section search.

    Pure Python implementation matching the JAX batch optimizer algorithm:
    1. Log-spaced grid search to bracket the minimum
    2. Golden section refinement within the bracket

    Operates in log-lambda space for numerical stability across the wide
    lambda range (1e-5 to 1e5). After 20 iterations, achieves relative
    tolerance of 0.618^20 ~ 6.6e-5, comparable to Brent's method with
    tol=1e-5.

    Args:
        func: Scalar function to minimize (negative log-likelihood).
        l_min: Lower bound for lambda search.
        l_max: Upper bound for lambda search.
        n_grid: Number of coarse grid points.
        n_iter: Golden section refinement iterations.

    Returns:
        (optimal_lambda, positive_logl) where positive_logl = -func(optimal_lambda).
    """
    import math

    phi = 0.6180339887498949  # Golden ratio - 1

    # Stage 1: Coarse grid search on log scale
    log_l_min = math.log(l_min)
    log_l_max = math.log(l_max)
    step = (log_l_max - log_l_min) / (n_grid - 1)
    log_lambdas = [log_l_min + i * step for i in range(n_grid)]

    # Evaluate func at each grid point, find minimum
    best_idx = 0
    best_val = func(math.exp(log_lambdas[0]))
    for i in range(1, n_grid):
        val = func(math.exp(log_lambdas[i]))
        if val < best_val:
            best_val = val
            best_idx = i

    # Bracket around best grid point
    idx_low = max(best_idx - 1, 0)
    idx_high = min(best_idx + 1, n_grid - 1)
    a = log_lambdas[idx_low]
    b = log_lambdas[idx_high]

    # Stage 2: Golden section refinement in log space
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = func(math.exp(c))
    fd = func(math.exp(d))

    for _ in range(n_iter):
        if fc < fd:
            # Minimum is in [a, d]
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = func(math.exp(c))
        else:
            # Minimum is in [c, b]
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = func(math.exp(d))

    log_opt = (a + b) / 2.0
    opt_lambda = math.exp(log_opt)
    opt_val = func(opt_lambda)

    return opt_lambda, -opt_val


def compute_null_model_lambda(
    eigenvalues: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> tuple[float, float]:
    """Compute lambda under null model (no genotype effect).

    Used by Score test (-lmm 3) which reuses null model lambda for all SNPs
    instead of re-optimizing per SNP (as Wald does).

    Uses reml_log_likelihood_null() which implements GEMMA's LogRL_f with
    calc_null=true (nc_total = n_cvt, df = n - n_cvt).

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,)
        UtW: Rotated covariates (n_samples, n_cvt)
        Uty: Rotated phenotype (n_samples,)
        n_cvt: Number of covariates
        l_min, l_max: Lambda bounds for optimization

    Returns:
        (lambda_null, logl_null) - Null model lambda and log-likelihood
    """
    # Compute Uab without genotype (Utx=None)
    # This sets genotype-related columns to zero via placeholder
    Uab = compute_Uab(UtW, Uty, Utx=None)

    # Create closure for null model REML optimization
    def neg_reml_null(lam: float) -> float:
        return -reml_log_likelihood_null(lam, eigenvalues, Uab, n_cvt)

    # Optimize lambda under the null model using golden section search
    lambda_null, logl_null = _golden_section_minimize(neg_reml_null, l_min, l_max)

    return lambda_null, logl_null


def mle_log_likelihood(
    lambda_val: float, eigenvalues: np.ndarray, Uab: np.ndarray, n_cvt: int
) -> float:
    """Compute MLE log-likelihood (NOT REML) for LRT.

    Key differences from REML:
    - Uses n (sample size) instead of df = n - n_cvt - 1
    - Does NOT include logdet_hiw term
    - MLE constant: c = 0.5 * n * (log(n) - log(2*pi) - 1)

    The MLE log-likelihood is:
    f = c - 0.5 * logdet_h - 0.5 * n * log(P_yy)

    where:
    - c = 0.5 * n * (log(n) - log(2*pi) - 1)
    - logdet_h = sum(log(lambda * eval + 1))
    - P_yy = Pab[nc_total, index_yy]

    Used by LRT (-lmm 2) which requires MLE likelihood.

    Args:
        lambda_val: Variance component ratio (sigma_g^2 / sigma_e^2)
        eigenvalues: Eigenvalues of kinship matrix (n_samples,)
        Uab: Matrix products from compute_Uab (n_samples, n_index)
        n_cvt: Number of covariates

    Returns:
        Log-likelihood value (positive for maximization)
    """
    n = len(eigenvalues)
    nc_total = n_cvt + 1

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)))

    Pab = calc_pab(n_cvt, Hi_eval, Uab)

    # NO logdet_hiw computation for MLE (key difference from REML)

    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    P_yy = Pab[nc_total, index_yy]

    P_YY_MIN = 1e-8
    if P_yy >= 0.0 and P_yy < P_YY_MIN:
        P_yy = P_YY_MIN

    # MLE formula (uses n, not df; no logdet_hiw)
    c = 0.5 * n * (np.log(n) - np.log(2 * np.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * n * np.log(P_yy)

    return f


def compute_null_model_mle(
    eigenvalues: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> tuple[float, float]:
    """Compute MLE lambda under null model (no genotype effect).

    Used by LRT (-lmm 2) which requires MLE (not REML) likelihood.
    The null model MLE is computed once and reused for all SNPs.

    Uses mle_log_likelihood_null() which implements GEMMA's LogL_f with
    calc_null=true (nc_total = n_cvt).

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,)
        UtW: Rotated covariates (n_samples, n_cvt)
        Uty: Rotated phenotype (n_samples,)
        n_cvt: Number of covariates
        l_min, l_max: Lambda bounds for optimization

    Returns:
        (lambda_null_mle, logl_H0) - Null model MLE lambda and log-likelihood
    """
    # Compute Uab without genotype (Utx=None)
    Uab = compute_Uab(UtW, Uty, Utx=None)

    # Create closure for null model MLE optimization
    def neg_mle_null(lam: float) -> float:
        return -mle_log_likelihood_null(lam, eigenvalues, Uab, n_cvt)

    # Optimize lambda under the null model using golden section search
    lambda_null_mle, logl_H0 = _golden_section_minimize(neg_mle_null, l_min, l_max)

    return lambda_null_mle, logl_H0
