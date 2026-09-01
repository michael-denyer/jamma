"""REML/MLE log-likelihood computation following GEMMA's exact algorithm.

Implements restricted maximum likelihood (REML) and maximum likelihood (MLE)
functions for variance component estimation in LMM. This closely follows
GEMMA's lmm.cpp CalcPab, LogRL_f and LogL_f. One function serves the null
and the alternative model of each likelihood: ``nc_total`` is the number of
columns projected out, ``n_cvt`` for the null model and ``n_cvt + 1`` once
the genotype joins. The scalar ports GEMMA keeps alongside (CalcPPab,
CalcPPPab, LogRL_dev2, CalcRLWald, CalcRLScore) live in ``tests/reference``.

Also provides null model optimization via golden section search for Score
and LRT tests.

Key data structures:
- Uab: 2D matrix (n_samples x n_index) storing element-wise products of rotated vectors
- Pab: 2D matrix (n_cvt+2 x n_index) storing H-inv weighted projections
- Hi_eval: 1/(lambda * eigenvalues + 1) weighting vector

Key functions for C extension support:
- classify_uab_columns: splits Uab indices into SNP-invariant and SNP-varying
- build_pab_table_for_c: flattens recursion data into C-friendly int32 arrays
- PabIndexTable / PabCTable: the two tables above, as typed NamedTuples

Reference: Zhou & Stephens (2012) Nature Genetics, Supplementary Information
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
from loguru import logger

from jamma.lmm.pab import (
    _NCVT1,
    _P_YY_MIN,
    PabCTable,
    PabIndexTable,
    build_index_table,
    build_pab_table_for_c,
    calc_iab,
    calc_pab,
    classify_uab_columns,
    compute_Uab,
    get_ab_index,
    n_index,
)

__all__ = [
    "_NCVT1",
    "_P_YY_MIN",
    "PabCTable",
    "PabIndexTable",
    "build_index_table",
    "build_pab_table_for_c",
    "calc_iab",
    "calc_pab",
    "classify_uab_columns",
    "compute_Uab",
    "compute_null_model_lambda",
    "compute_null_model_mle",
    "finite_difference_dev2",
    "get_ab_index",
    "mle_log_likelihood",
    "n_index",
    "reml_log_likelihood",
    "reset_p_yy_warned",
    "warn_p_yy_once",
]

# One thread-local flag deduplicates the negative-P_yy warning across the
# scalar (_clamp_p_yy) and batch (likelihood_numpy._guard_P_yy) guards within
# a single LMM run. The batch guard fires hundreds of times per run (once per
# grid eval + golden section iter per chunk), which buried the meaningful first
# warning under identical log spam. Reset at run start via reset_p_yy_warned().
_p_yy_state = threading.local()


def reset_p_yy_warned() -> None:
    """Reset the P_yy warning flag so each LMM run gets its own warning."""
    _p_yy_state.warned = False


def warn_p_yy_once(message: str) -> None:
    """Log ``message`` at WARNING level once per run, then stay silent."""
    if getattr(_p_yy_state, "warned", False):
        return
    logger.warning(message)
    _p_yy_state.warned = True


def _clamp_p_yy(P_yy: float, lambda_val: float) -> float:
    """Clamp P_yy to prevent log(0) or log(negative) in log-likelihood.

    Returns NaN for negative P_yy (propagates through np.log as NaN;
    optimizer avoids NaN regions) and clamps near-zero positive values
    to _P_YY_MIN.

    Warning deduplication: only logs the first negative P_yy per run.
    Call reset_p_yy_warned() at the start of each LMM run.

    Args:
        P_yy: Projected residual variance from Pab.
        lambda_val: Current lambda value (for diagnostic logging).

    Returns:
        Clamped P_yy, or NaN for negative values (signals invalid region).
    """
    if P_yy < 0:
        warn_p_yy_once(
            f"Negative P_yy ({P_yy:.6e}) at lambda={lambda_val:.6e} — "
            "numerical breakdown (subsequent occurrences suppressed). "
            "The kinship matrix may not be positive semi-definite."
        )
        return float("nan")  # np.log(nan) = nan, optimizer avoids
    if P_yy < _P_YY_MIN:
        return _P_YY_MIN
    return P_yy


def finite_difference_dev2(
    lambda_val: float,
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> float:
    """Compute REML log-likelihood second derivative via central finite differences.

    Full numerical second derivative using a central stencil on the null
    model REML log-likelihood. This is the production path for se(pve);
    ``tests/reference/likelihood.py`` holds the analytical LogRL_dev2 port it
    is checked against. Uses h ~ O(eps^{1/4}) * lambda for optimal
    second-derivative accuracy. Falls back to a one-sided stencil when lambda
    is near the optimiser bounds.

    Args:
        lambda_val: REML-optimal lambda (null model).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab: Null model Uab (n_samples, n_index).
        n_cvt: Number of covariates.
        l_min: Lower bound of lambda search range.
        l_max: Upper bound of lambda search range.

    Returns:
        Second derivative d² ℓ / d λ² (negative at maximum).
    """
    h = max(lambda_val * 1e-4, 1e-8)

    can_go_left = (lambda_val - h) > l_min
    can_go_right = (lambda_val + h) < l_max

    def f(lam: float) -> float:
        return reml_log_likelihood(lam, eigenvalues, Uab, n_cvt, nc_total=n_cvt)

    if can_go_left and can_go_right:
        # Central stencil
        return (f(lambda_val + h) - 2.0 * f(lambda_val) + f(lambda_val - h)) / (h * h)
    elif can_go_right:
        # Forward stencil (near l_min)
        return (f(lambda_val + 2 * h) - 2.0 * f(lambda_val + h) + f(lambda_val)) / (
            h * h
        )
    elif can_go_left:
        # Backward stencil (near l_max)
        return (f(lambda_val) - 2.0 * f(lambda_val - h) + f(lambda_val - 2 * h)) / (
            h * h
        )
    else:
        logger.warning("lambda range too narrow for finite-difference stencil")
        return np.nan


def reml_log_likelihood(
    lambda_val: float,
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
    *,
    nc_total: int,
) -> float:
    """Compute REML log-likelihood following GEMMA's LogRL_f exactly.

    The REML log-likelihood is:
    f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * log(P_yy)

    where:
    - c = 0.5 * df * (log(df) - log(2*pi) - 1)
    - logdet_h = sum(log(lambda * eval + 1))
    - logdet_hiw = sum(log(Pab[i,ww])) - sum(log(Iab[i,ww])) over nc_total rows
    - P_yy = Pab[nc_total, index_yy]
    - df = n - nc_total

    Args:
        lambda_val: Variance component ratio (sigma_g^2 / sigma_e^2)
        eigenvalues: Eigenvalues of kinship matrix (n_samples,)
        Uab: Matrix products from compute_Uab (n_samples, n_index)
        n_cvt: Number of covariates
        nc_total: Columns projected out. ``n_cvt`` for the null model
            (GEMMA ``calc_null=true``), ``n_cvt + 1`` for the alternative.

    Returns:
        Log-likelihood value (positive for maximization)
    """
    n = len(eigenvalues)
    df = n - nc_total

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)))

    table = build_index_table(n_cvt)
    Pab = calc_pab(n_cvt, Hi_eval, Uab)
    Iab = calc_iab(n_cvt, Uab)

    logdet_hiw = 0.0
    for i, index_ww in table.logdet_diag_indices[:nc_total]:
        d_pab = Pab[i, index_ww]
        d_iab = Iab[i, index_ww]
        if d_pab > 0:
            logdet_hiw += np.log(d_pab)
        if d_iab > 0:
            logdet_hiw -= np.log(d_iab)

    P_yy = _clamp_p_yy(Pab[nc_total, table.idx_yy], lambda_val)

    c = 0.5 * df * (np.log(df) - np.log(2 * np.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)

    return f


def _mle_p_yy_scalar_ncvt1(Hi_eval: np.ndarray, Uab: np.ndarray) -> float:
    """Compute MLE P_yy via scalar Schur complements for n_cvt=1.

    Avoids allocating full (3, 6) Pab matrix — computes only the 6 dot products
    and 2 Schur complement steps needed for P_yy = Pab[2][5].

    For n_cvt=1, nc_total=2, the trace is:
      Row 0: s_ww, s_wx, s_wy, s_xx, s_xy, s_yy = Hi_eval @ Uab[:, 0..5]
      Row 1: p1_xx = s_xx - s_wx^2/s_ww
              p1_xy = s_xy - s_wx*s_wy/s_ww
              p1_yy = s_yy - s_wy^2/s_ww
      Row 2: P_yy = p1_yy - p1_xy^2/p1_xx

    Args:
        Hi_eval: 1/(lambda*eigenvalues + 1) vector (n_samples,).
        Uab: Matrix products (n_samples, 6) for n_cvt=1.

    Returns:
        P_yy scalar (the projected phenotype variance).
    """
    s_ww = Hi_eval @ Uab[:, _NCVT1.ww]
    s_wx = Hi_eval @ Uab[:, _NCVT1.wx]
    s_wy = Hi_eval @ Uab[:, _NCVT1.wy]
    s_xx = Hi_eval @ Uab[:, _NCVT1.xx]
    s_xy = Hi_eval @ Uab[:, _NCVT1.xy]
    s_yy = Hi_eval @ Uab[:, _NCVT1.yy]

    # Row 1: project out W (Schur complement)
    if s_ww <= 0:
        if s_ww < 0:
            logger.warning(
                f"Negative s_ww ({s_ww:.6e}) in scalar MLE P_yy — "
                "eigendecomposition may be degenerate."
            )
        return float(s_yy)  # degenerate
    inv_ww = 1.0 / s_ww
    p1_xx = s_xx - s_wx * s_wx * inv_ww
    p1_xy = s_xy - s_wx * s_wy * inv_ww
    p1_yy = s_yy - s_wy * s_wy * inv_ww

    # Row 2: project out X (Schur complement)
    if p1_xx == 0:
        return float(p1_yy)  # degenerate
    P_yy = p1_yy - p1_xy * p1_xy / p1_xx
    return float(P_yy)


def _mle_p_yy_scalar_null_ncvt1(Hi_eval: np.ndarray, Uab: np.ndarray) -> float:
    """Compute null-model MLE P_yy for n_cvt=1.

    Null model: nc_total=n_cvt=1, so P_yy = Pab[1][5] = p1_yy.
    Only row 0 and row 1 Schur complement needed.

    Args:
        Hi_eval: 1/(lambda*eigenvalues + 1) vector (n_samples,).
        Uab: Matrix products (n_samples, 6) for n_cvt=1 null model.

    Returns:
        P_yy scalar for the null model.
    """
    s_ww = Hi_eval @ Uab[:, _NCVT1.ww]
    s_wy = Hi_eval @ Uab[:, _NCVT1.wy]
    s_yy = Hi_eval @ Uab[:, _NCVT1.yy]

    if s_ww <= 0:
        if s_ww < 0:
            logger.warning(
                f"Negative s_ww ({s_ww:.6e}) in scalar null MLE P_yy — "
                "eigendecomposition may be degenerate."
            )
        return float(s_yy)
    p1_yy = s_yy - s_wy * s_wy / s_ww
    return float(p1_yy)


def _golden_section_minimize(
    func: Callable[[float], float],
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[float, float]:
    """Minimize a scalar function over [l_min, l_max] using golden section search.

    Pure Python implementation using grid search + golden section refinement:
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
    best_val = float("inf")
    for i in range(n_grid):
        val = func(math.exp(log_lambdas[i]))
        if not math.isnan(val) and val < best_val:
            best_val = val
            best_idx = i

    if math.isinf(best_val):
        logger.warning(
            "All grid points returned NaN log-likelihood — "
            "kinship matrix may be degenerate. Returning boundary lambda."
        )
        return l_min, float("nan")

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

    Uses reml_log_likelihood() with nc_total = n_cvt, GEMMA's LogRL_f with
    calc_null=true.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,)
        UtW: Rotated covariates (n_samples, n_cvt)
        Uty: Rotated phenotype (n_samples,)
        n_cvt: Number of covariates
        l_min, l_max: Lambda bounds for optimization

    Returns:
        (lambda_null, logl_null) - Null model lambda and log-likelihood
    """
    reset_p_yy_warned()

    # Compute Uab without genotype (Utx=None)
    # This sets genotype-related columns to zero via placeholder
    Uab = compute_Uab(UtW, Uty, Utx=None)

    # Create closure for null model REML optimization
    def neg_reml_null(lam: float) -> float:
        return -reml_log_likelihood(lam, eigenvalues, Uab, n_cvt, nc_total=n_cvt)

    # Optimize lambda under the null model using golden section search
    lambda_null, logl_null = _golden_section_minimize(neg_reml_null, l_min, l_max)

    return lambda_null, logl_null


def mle_log_likelihood(
    lambda_val: float,
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
    *,
    nc_total: int,
) -> float:
    """Compute MLE log-likelihood (NOT REML) following GEMMA's LogL_f.

    Key differences from REML:
    - Uses n (sample size) instead of df
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
        nc_total: Columns projected out. ``n_cvt`` for the null model
            (GEMMA ``calc_null=true``), ``n_cvt + 1`` for the alternative.

    Returns:
        Log-likelihood value (positive for maximization)
    """
    n = len(eigenvalues)

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)))

    # Scalar path for n_cvt=1: skip full Pab allocation
    if n_cvt == nc_total == 1:
        P_yy_raw = _mle_p_yy_scalar_null_ncvt1(Hi_eval, Uab)
    elif n_cvt == 1:
        P_yy_raw = _mle_p_yy_scalar_ncvt1(Hi_eval, Uab)
    else:
        Pab = calc_pab(n_cvt, Hi_eval, Uab)
        P_yy_raw = Pab[nc_total, build_index_table(n_cvt).idx_yy]

    P_yy = _clamp_p_yy(P_yy_raw, lambda_val)

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

    Uses mle_log_likelihood() with nc_total = n_cvt, GEMMA's LogL_f with
    calc_null=true.

    Args:
        eigenvalues: Kinship eigenvalues (n_samples,)
        UtW: Rotated covariates (n_samples, n_cvt)
        Uty: Rotated phenotype (n_samples,)
        n_cvt: Number of covariates
        l_min, l_max: Lambda bounds for optimization

    Returns:
        (lambda_null_mle, logl_H0) - Null model MLE lambda and log-likelihood
    """
    reset_p_yy_warned()

    # Compute Uab without genotype (Utx=None)
    Uab = compute_Uab(UtW, Uty, Utx=None)

    # Create closure for null model MLE optimization
    def neg_mle_null(lam: float) -> float:
        return -mle_log_likelihood(lam, eigenvalues, Uab, n_cvt, nc_total=n_cvt)

    # Optimize lambda under the null model using golden section search
    lambda_null_mle, logl_H0 = _golden_section_minimize(neg_mle_null, l_min, l_max)

    return lambda_null_mle, logl_H0
