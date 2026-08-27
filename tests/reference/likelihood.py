"""Scalar ports of GEMMA's CalcPPab, CalcPPPab and LogRL_dev2.

Production computes se(pve) by finite differences on the null REML
log-likelihood (``jamma.lmm.likelihood.finite_difference_dev2``). The
analytical second derivative here is what that stencil is checked against.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from jamma.lmm.likelihood import _P_YY_MIN, calc_iab, calc_pab, get_ab_index, n_index


def calc_ppab(
    n_cvt: int,
    HiHi_eval: np.ndarray,
    Uab: np.ndarray,
    Pab: np.ndarray,
) -> np.ndarray:
    """Compute PPab (second-order projected Pab) for REML second derivative.

    PPab stores v_a P^2 v_b quantities. Row 0 uses HiHi_eval weighted
    dot products; subsequent rows use Schur complement recursion involving
    both Pab and PPab from the previous level.

    Port of GEMMA v0.98.5 CalcPPab (e_mode=0 path).

    Args:
        n_cvt: Number of covariates.
        HiHi_eval: 1/(lambda*eval + 1)^2 vector (n_samples,).
        Uab: Matrix products (n_samples, n_index).
        Pab: First-order projection from calc_pab (n_cvt+2, n_index).

    Returns:
        PPab matrix (n_cvt+2, n_index).
    """
    PPab = np.zeros((n_cvt + 2, n_index(n_cvt)), dtype=np.float64)

    # Row 0: weighted dot products with HiHi_eval
    PPab[0, :] = HiHi_eval @ Uab

    # Rows 1..n_cvt+1: recursive projection
    for p in range(1, n_cvt + 2):
        for a in range(p + 1, n_cvt + 3):
            for b in range(a, n_cvt + 3):
                index_ab = get_ab_index(a, b, n_cvt)
                index_aw = get_ab_index(a, p, n_cvt)
                index_bw = get_ab_index(b, p, n_cvt)
                index_ww = get_ab_index(p, p, n_cvt)

                ps2_ab = PPab[p - 1, index_ab]
                ps_aw = Pab[p - 1, index_aw]
                ps_bw = Pab[p - 1, index_bw]
                ps_ww = Pab[p - 1, index_ww]
                ps2_aw = PPab[p - 1, index_aw]
                ps2_bw = PPab[p - 1, index_bw]
                ps2_ww = PPab[p - 1, index_ww]

                if ps_ww != 0:
                    p2_ab = ps2_ab + ps_aw * ps_bw * ps2_ww / (ps_ww * ps_ww)
                    p2_ab -= (ps_aw * ps2_bw + ps_bw * ps2_aw) / ps_ww
                else:
                    p2_ab = ps2_ab

                PPab[p, index_ab] = p2_ab

    return PPab


def calc_pppab(
    n_cvt: int,
    HiHiHi_eval: np.ndarray,
    Uab: np.ndarray,
    Pab: np.ndarray,
    PPab: np.ndarray,
) -> np.ndarray:
    """Compute PPPab (third-order projected Pab) for REML second derivative.

    PPPab stores v_a P^3 v_b quantities. Row 0 uses HiHiHi_eval weighted
    dot products; subsequent rows use Schur complement recursion involving
    Pab, PPab, and PPPab from the previous level.

    Port of GEMMA v0.98.5 CalcPPPab (e_mode=0 path).

    Args:
        n_cvt: Number of covariates.
        HiHiHi_eval: 1/(lambda*eval + 1)^3 vector (n_samples,).
        Uab: Matrix products (n_samples, n_index).
        Pab: First-order projection from calc_pab (n_cvt+2, n_index).
        PPab: Second-order projection from calc_ppab (n_cvt+2, n_index).

    Returns:
        PPPab matrix (n_cvt+2, n_index).
    """
    PPPab = np.zeros((n_cvt + 2, n_index(n_cvt)), dtype=np.float64)

    # Row 0: weighted dot products with HiHiHi_eval
    PPPab[0, :] = HiHiHi_eval @ Uab

    # Rows 1..n_cvt+1: recursive projection
    for p in range(1, n_cvt + 2):
        for a in range(p + 1, n_cvt + 3):
            for b in range(a, n_cvt + 3):
                index_ab = get_ab_index(a, b, n_cvt)
                index_aw = get_ab_index(a, p, n_cvt)
                index_bw = get_ab_index(b, p, n_cvt)
                index_ww = get_ab_index(p, p, n_cvt)

                ps3_ab = PPPab[p - 1, index_ab]
                ps_aw = Pab[p - 1, index_aw]
                ps_bw = Pab[p - 1, index_bw]
                ps_ww = Pab[p - 1, index_ww]
                ps2_aw = PPab[p - 1, index_aw]
                ps2_bw = PPab[p - 1, index_bw]
                ps2_ww = PPab[p - 1, index_ww]
                ps3_aw = PPPab[p - 1, index_aw]
                ps3_bw = PPPab[p - 1, index_bw]
                ps3_ww = PPPab[p - 1, index_ww]

                if ps_ww != 0:
                    ps_ww2 = ps_ww * ps_ww
                    ps_ww3 = ps_ww2 * ps_ww

                    p3_ab = ps3_ab
                    # Term: -aw*bw*ps2_ww^2 / ps_ww^3
                    p3_ab -= ps_aw * ps_bw * ps2_ww * ps2_ww / ps_ww3
                    # Term: -(aw*ps3_bw + bw*ps3_aw + ps2_aw*ps2_bw) / ps_ww
                    p3_ab -= (ps_aw * ps3_bw + ps_bw * ps3_aw + ps2_aw * ps2_bw) / ps_ww
                    # Term: +(aw*ps2_bw*ps2_ww + bw*ps2_aw*ps2_ww
                    #         + aw*bw*ps3_ww) / ps_ww^2
                    p3_ab += (
                        ps_aw * ps2_bw * ps2_ww
                        + ps_bw * ps2_aw * ps2_ww
                        + ps_aw * ps_bw * ps3_ww
                    ) / ps_ww2
                else:
                    p3_ab = ps3_ab

                PPPab[p, index_ab] = p3_ab

    return PPPab


def logdet_hiw_null(
    lambda_val: float,
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
) -> float:
    """Compute logdet(W'HiW) - logdet(W'W) for null model.

    Helper for computing the logdet_hiw second derivative via finite
    differences on this isolated term. Much cheaper than finite-
    differencing the full log-likelihood (only calc_pab + calc_iab,
    no log/constant overhead).

    Args:
        lambda_val: Variance ratio.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab: Null model Uab (n_samples, n_index).
        n_cvt: Number of covariates.

    Returns:
        logdet_hiw scalar.
    """
    Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
    Pab = calc_pab(n_cvt, Hi_eval, Uab)
    Iab = calc_iab(n_cvt, Uab)

    nc_total = n_cvt  # null model
    logdet_hiw = 0.0
    for i in range(nc_total):
        index_ww = get_ab_index(i + 1, i + 1, n_cvt)
        d_pab = Pab[i, index_ww]
        d_iab = Iab[i, index_ww]
        if d_pab > 0:
            logdet_hiw += np.log(d_pab)
        if d_iab > 0:
            logdet_hiw -= np.log(d_iab)

    return logdet_hiw


def reml_log_likelihood_dev2(
    lambda_val: float,
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
) -> float:
    """Compute REML log-likelihood second derivative for null model.

    Used by the delta method to compute se(pve). The second derivative
    at the REML optimum determines the precision of the lambda estimate.

    Port of GEMMA v0.98.5 LogRL_dev2 (e_mode=0, calc_null=True).
    The trace and yPKPy terms are computed analytically from Pab/PPab/PPPab.
    The logdet_hiw second derivative uses a local finite-difference stencil
    on the logdet_hiw function alone (3 lightweight calc_pab + calc_iab
    evaluations, not 3 full likelihood evaluations).

    Args:
        lambda_val: REML-optimal lambda (null model).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab: Null model Uab (n_samples, n_index).
        n_cvt: Number of covariates.

    Returns:
        Second derivative d^2 ell / d lambda^2 (negative at maximum).
    """
    if lambda_val <= 0:
        logger.warning(f"lambda_val={lambda_val:.6e} is non-positive in dev2")
        return np.nan

    n = len(eigenvalues)
    nc_total = n_cvt  # null model
    df = n - n_cvt

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    HiHi_eval = Hi_eval * Hi_eval
    HiHiHi_eval = HiHi_eval * Hi_eval

    trace_Hi = np.sum(Hi_eval)
    trace_HiHi = np.sum(HiHi_eval)
    trace_HiKHiK = (n + trace_HiHi - 2.0 * trace_Hi) / (lambda_val * lambda_val)

    Pab = calc_pab(n_cvt, Hi_eval, Uab)
    PPab_mat = calc_ppab(n_cvt, HiHi_eval, Uab, Pab)
    PPPab_mat = calc_pppab(n_cvt, HiHiHi_eval, Uab, Pab, PPab_mat)

    idx_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    P_yy = Pab[nc_total, idx_yy]
    if P_yy < _P_YY_MIN:
        logger.warning(
            f"P_yy={P_yy:.6e} below floor {_P_YY_MIN} in dev2 "
            f"— phenotype may be degenerate after projection"
        )
        return np.nan
    PP_yy = PPab_mat[nc_total, idx_yy]
    PPP_yy = PPPab_mat[nc_total, idx_yy]

    yPKPy = (P_yy - PP_yy) / lambda_val
    yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (lambda_val * lambda_val)

    # d^2(logdet_hiw)/dlambda^2 via central finite differences on
    # the isolated logdet_hiw(lambda) function. Step size h ~ eps^{1/4}
    # * lambda for optimal second-derivative accuracy.
    h = max(lambda_val * 1e-4, 1e-8)
    logdet_p = logdet_hiw_null(lambda_val + h, eigenvalues, Uab, n_cvt)
    logdet_c = logdet_hiw_null(lambda_val, eigenvalues, Uab, n_cvt)
    logdet_m = logdet_hiw_null(lambda_val - h, eigenvalues, Uab, n_cvt)
    d2_logdet_hiw = (logdet_p - 2.0 * logdet_c + logdet_m) / (h * h)

    dev2 = (
        0.5 * trace_HiKHiK
        - 0.5 * d2_logdet_hiw
        - 0.5 * df * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) / (P_yy * P_yy)
    )

    return dev2
