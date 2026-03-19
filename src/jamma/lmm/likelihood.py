"""REML/MLE log-likelihood computation following GEMMA's exact algorithm.

Implements restricted maximum likelihood (REML) and maximum likelihood (MLE)
functions for variance component estimation in LMM. This closely follows
GEMMA's lmm.cpp CalcPab, CalcPPab, CalcPPPab, LogRL_f, LogRL_dev2, LogL_f,
and CalcRLWald functions.

Also provides null model optimization via golden section search for Score
and LRT tests.

Key data structures:
- Uab: 2D matrix (n_samples x n_index) storing element-wise products of rotated vectors
- Pab: 2D matrix (n_cvt+2 x n_index) storing H-inv weighted projections
- Hi_eval: 1/(lambda * eigenvalues + 1) weighting vector

Key functions for C extension support:
- classify_uab_columns: splits Uab indices into SNP-invariant and SNP-varying
- build_pab_table_for_c: flattens recursion data into C-friendly int32 arrays

Reference: Zhou & Stephens (2012) Nature Genetics, Supplementary Information
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable

import numpy as np
from loguru import logger

_P_YY_MIN = 1e-8

# Thread-local flag — deduplicates _clamp_p_yy warning across all calls
# within a single LMM run. Reset at run start via reset_scalar_p_yy_warned().
_scalar_p_yy_state = threading.local()


def reset_scalar_p_yy_warned() -> None:
    """Reset the scalar P_yy warning flag so each LMM run gets its own warning."""
    _scalar_p_yy_state.warned = False


def _clamp_p_yy(P_yy: float, lambda_val: float) -> float:
    """Clamp P_yy to prevent log(0) or log(negative) in log-likelihood.

    Returns NaN for negative P_yy (propagates through np.log as NaN;
    optimizer avoids NaN regions) and clamps near-zero positive values
    to _P_YY_MIN.

    Warning deduplication: only logs the first negative P_yy per run.
    Call reset_scalar_p_yy_warned() at the start of each LMM run.

    Args:
        P_yy: Projected residual variance from Pab.
        lambda_val: Current lambda value (for diagnostic logging).

    Returns:
        Clamped P_yy, or NaN for negative values (signals invalid region).
    """
    if P_yy < 0:
        if not getattr(_scalar_p_yy_state, "warned", False):
            logger.warning(
                f"Negative P_yy ({P_yy:.6e}) at lambda={lambda_val:.6e} — "
                "numerical breakdown (subsequent occurrences suppressed). "
                "The kinship matrix may not be positive semi-definite."
            )
            _scalar_p_yy_state.warned = True
        return float("nan")  # np.log(nan) = nan, optimizer avoids
    if P_yy < _P_YY_MIN:
        return _P_YY_MIN
    return P_yy


@functools.lru_cache(maxsize=8)
def build_index_table(n_cvt: int) -> dict:
    """Precompute all index mappings for a given n_cvt.

    Pure Python function with zero JAX dependency. Runs at Python level.
    When called inside a JIT function with n_cvt as a static argument,
    it executes at trace time, producing compile-time constants.

    GEMMA convention (1-based):
      Columns 1..n_cvt = covariates (W)
      Column n_cvt+1 = genotype (X)
      Column n_cvt+2 = phenotype (Y)

    Args:
        n_cvt: Number of covariates.

    Returns:
        Dict with precomputed index mappings:
        - n_index: total (a,b) pairs = (n_cvt+3)*(n_cvt+2)//2
        - idx_yy: index for phenotype-phenotype
        - idx_xx: index for genotype-genotype
        - idx_xy: index for genotype-phenotype
        - uab_pairs: list of (a_col, b_col, index) for Uab construction
        - pab_recursion: per-level recursion tuples for Pab
        - logdet_diag_indices: (row, col) pairs for logdet_hiw diagonal
    """
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2

    idx_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    idx_xx = get_ab_index(n_cvt + 1, n_cvt + 1, n_cvt)
    idx_xy = get_ab_index(n_cvt + 2, n_cvt + 1, n_cvt)

    # Uab column pairs: (0-based col_a, 0-based col_b, linear index)
    # Vectors array is [W1,...,W_ncvt, X, Y] with 0-based columns
    uab_pairs = []
    for a in range(1, n_cvt + 3):
        for b in range(a, n_cvt + 3):
            idx = get_ab_index(a, b, n_cvt)
            uab_pairs.append((a - 1, b - 1, idx))

    # Pab recursion: for each projection level p (1..n_cvt+1),
    # build list of (a, b, index_ab, index_aw, index_bw, index_ww)
    # using GEMMA 1-based indexing
    pab_recursion = {}
    for p in range(1, n_cvt + 2):
        entries = []
        for a in range(p + 1, n_cvt + 3):
            for b in range(a, n_cvt + 3):
                index_ab = get_ab_index(a, b, n_cvt)
                index_aw = get_ab_index(a, p, n_cvt)
                index_bw = get_ab_index(b, p, n_cvt)
                index_ww = get_ab_index(p, p, n_cvt)
                entries.append((a, b, index_ab, index_aw, index_bw, index_ww))
        pab_recursion[p] = entries

    # logdet_hiw diagonal: for i=0..n_cvt, the diagonal element is
    # Pab[i, get_ab_index(i+1, i+1, n_cvt)]
    logdet_diag_indices = []
    for i in range(n_cvt + 1):
        col = get_ab_index(i + 1, i + 1, n_cvt)
        logdet_diag_indices.append((i, col))

    return {
        "n_index": n_index,
        "idx_yy": idx_yy,
        "idx_xx": idx_xx,
        "idx_xy": idx_xy,
        "uab_pairs": uab_pairs,
        "pab_recursion": pab_recursion,
        "logdet_diag_indices": logdet_diag_indices,
    }


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
    n_index = (n_cvt + 2 + 1) * (n_cvt + 2) // 2
    PPab = np.zeros((n_cvt + 2, n_index), dtype=np.float64)

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
    n_index = (n_cvt + 2 + 1) * (n_cvt + 2) // 2
    PPPab = np.zeros((n_cvt + 2, n_index), dtype=np.float64)

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


def _logdet_hiw_null(
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
    on the logdet_hiw function alone (2 extra calc_pab calls, not 3 full
    likelihood evaluations).

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
    raw_P_yy = Pab[nc_total, idx_yy]
    if raw_P_yy < _P_YY_MIN:
        logger.warning(
            f"P_yy={raw_P_yy:.6e} below floor {_P_YY_MIN} in dev2 "
            f"— phenotype may be degenerate after projection"
        )
        return np.nan
    P_yy = raw_P_yy
    PP_yy = PPab_mat[nc_total, idx_yy]
    PPP_yy = PPPab_mat[nc_total, idx_yy]

    yPKPy = (P_yy - PP_yy) / lambda_val
    yPKPKPy = (P_yy + PPP_yy - 2.0 * PP_yy) / (lambda_val * lambda_val)

    # d^2(logdet_hiw)/dlambda^2 via central finite differences on
    # the isolated logdet_hiw(lambda) function. Step size h ~ eps^{1/4}
    # * lambda for optimal second-derivative accuracy.
    h = max(lambda_val * 1e-4, 1e-8)
    logdet_p = _logdet_hiw_null(lambda_val + h, eigenvalues, Uab, n_cvt)
    logdet_c = _logdet_hiw_null(lambda_val, eigenvalues, Uab, n_cvt)
    logdet_m = _logdet_hiw_null(lambda_val - h, eigenvalues, Uab, n_cvt)
    d2_logdet_hiw = (logdet_p - 2.0 * logdet_c + logdet_m) / (h * h)

    dev2 = (
        0.5 * trace_HiKHiK
        - 0.5 * d2_logdet_hiw
        - 0.5 * df * (2.0 * yPKPKPy * P_yy - yPKPy * yPKPy) / (P_yy * P_yy)
    )

    return dev2


def finite_difference_dev2(
    lambda_val: float,
    eigenvalues: np.ndarray,
    Uab: np.ndarray,
    n_cvt: int,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> float:
    """Compute REML log-likelihood second derivative via central finite differences.

    Full numerical second derivative using a central stencil on
    reml_log_likelihood_null. Retained as a test oracle for validating
    the analytical reml_log_likelihood_dev2. Uses h ~ O(eps^{1/4}) *
    lambda for optimal second-derivative accuracy. Falls back to a
    one-sided stencil when lambda is near the optimiser bounds.

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
        return reml_log_likelihood_null(lam, eigenvalues, Uab, n_cvt)

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
    P_yy = _clamp_p_yy(Pab[nc_total, index_yy], lambda_val)

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
    P_yy = _clamp_p_yy(Pab[nc_total, index_yy], lambda_val)

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
    s_ww = Hi_eval @ Uab[:, 0]
    s_wx = Hi_eval @ Uab[:, 1]
    s_wy = Hi_eval @ Uab[:, 2]
    s_xx = Hi_eval @ Uab[:, 3]
    s_xy = Hi_eval @ Uab[:, 4]
    s_yy = Hi_eval @ Uab[:, 5]

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
    s_ww = Hi_eval @ Uab[:, 0]
    s_wy = Hi_eval @ Uab[:, 2]
    s_yy = Hi_eval @ Uab[:, 5]

    if s_ww <= 0:
        if s_ww < 0:
            logger.warning(
                f"Negative s_ww ({s_ww:.6e}) in scalar null MLE P_yy — "
                "eigendecomposition may be degenerate."
            )
        return float(s_yy)
    p1_yy = s_yy - s_wy * s_wy / s_ww
    return float(p1_yy)


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

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)))

    # Scalar path for n_cvt=1: skip full Pab allocation
    if n_cvt == 1:
        P_yy_raw = _mle_p_yy_scalar_null_ncvt1(Hi_eval, Uab)
    else:
        nc_total = n_cvt  # NULL MODEL: no genotype column
        Pab = calc_pab(n_cvt, Hi_eval, Uab)
        index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
        P_yy_raw = Pab[nc_total, index_yy]

    P_yy = _clamp_p_yy(P_yy_raw, lambda_val)

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


@functools.lru_cache(maxsize=8)
def classify_uab_columns(n_cvt: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Classify Uab columns as invariant or SNP-varying.

    A column is SNP-varying if its (a_col, b_col) pair involves the genotype
    (0-based index = n_cvt). Otherwise it is invariant across SNPs.

    Pure Python, lru_cached. No JAX dependency.

    Args:
        n_cvt: Number of covariates.

    Returns:
        (invariant_indices, varying_indices) as tuples of linear column indices.
    """
    table = build_index_table(n_cvt)
    genotype_col = n_cvt  # 0-based index of X in vectors array
    invariant = []
    varying = []
    for a_col, b_col, linear_idx in table["uab_pairs"]:
        if a_col == genotype_col or b_col == genotype_col:
            varying.append(linear_idx)
        else:
            invariant.append(linear_idx)
    return tuple(invariant), tuple(varying)


@functools.lru_cache(maxsize=8)
def build_pab_table_for_c(n_cvt: int) -> dict:
    """Build flat C-friendly arrays from build_index_table recursion data.

    Converts the nested Python dicts from build_index_table() and the
    invariant/varying classification from classify_uab_columns() into
    flat int32 numpy arrays that the C extension can consume via
    PyArray_DATA().

    The entries array is stride-4: each entry is
    [index_ab, index_aw, index_bw, index_ww].

    Level 0 has no entries (row 0 comes from dot products).
    Levels 1..n_cvt+1 have recursion entries.

    Args:
        n_cvt: Number of covariates.

    Returns:
        Dict with scalar and array fields for C extension consumption.
    """
    table = build_index_table(n_cvt)
    inv_indices, var_indices = classify_uab_columns(n_cvt)

    # Flatten pab_recursion into contiguous entries array
    # Level 0 has no entries; levels 1..n_cvt+1 have entries
    n_levels = n_cvt + 2  # levels 0..n_cvt+1
    level_counts_list = [0] * n_levels  # level 0 has 0 entries
    all_entries = []

    for level in range(1, n_cvt + 2):
        level_entries = table["pab_recursion"][level]
        level_counts_list[level] = len(level_entries)
        for _, _, idx_ab, idx_aw, idx_bw, idx_ww in level_entries:
            all_entries.extend([idx_ab, idx_aw, idx_bw, idx_ww])

    # Build level_offsets (cumulative)
    level_offsets_list = []
    running = 0
    for count in level_counts_list:
        level_offsets_list.append(running)
        running += count

    # Extract logdet_diag_indices
    diag_rows = [r for r, _ in table["logdet_diag_indices"]]
    diag_cols = [c for _, c in table["logdet_diag_indices"]]

    def _frozen(data: list[int]) -> np.ndarray:
        arr = np.array(data, dtype=np.int32)
        arr.flags.writeable = False
        return arr

    return {
        "n_cvt": n_cvt,
        "n_index": table["n_index"],
        "n_rows": n_cvt + 2,
        "n_inv": len(inv_indices),
        "n_var": len(var_indices),
        "idx_xx": table["idx_xx"],
        "idx_xy": table["idx_xy"],
        "idx_yy": table["idx_yy"],
        "invariant_indices": _frozen(inv_indices),
        "varying_indices": _frozen(var_indices),
        "logdet_diag_rows": _frozen(diag_rows),
        "logdet_diag_cols": _frozen(diag_cols),
        "level_offsets": _frozen(level_offsets_list),
        "level_counts": _frozen(level_counts_list),
        "entries": _frozen(all_entries),
    }


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
    reset_scalar_p_yy_warned()

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

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)))

    # Scalar path for n_cvt=1: skip full Pab allocation
    if n_cvt == 1:
        P_yy_raw = _mle_p_yy_scalar_ncvt1(Hi_eval, Uab)
    else:
        nc_total = n_cvt + 1
        Pab = calc_pab(n_cvt, Hi_eval, Uab)
        index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
        P_yy_raw = Pab[nc_total, index_yy]

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
    reset_scalar_p_yy_warned()

    # Compute Uab without genotype (Utx=None)
    Uab = compute_Uab(UtW, Uty, Utx=None)

    # Create closure for null model MLE optimization
    def neg_mle_null(lam: float) -> float:
        return -mle_log_likelihood_null(lam, eigenvalues, Uab, n_cvt)

    # Optimize lambda under the null model using golden section search
    lambda_null_mle, logl_H0 = _golden_section_minimize(neg_mle_null, l_min, l_max)

    return lambda_null_mle, logl_H0
