"""REML log-likelihood computation following GEMMA's exact algorithm.

Implements the restricted maximum likelihood (REML) function for
variance component estimation in LMM. This closely follows GEMMA's
lmm.cpp CalcPab, LogRL_f, and CalcRLWald functions.

Key data structures:
- Uab: 2D matrix (n_samples × n_index) storing element-wise products of rotated vectors
- Pab: 2D matrix (n_cvt+2 × n_index) storing H-inv weighted projections
- Hi_eval: 1/(lambda * eigenvalues + 1) weighting vector

Reference: Zhou & Stephens (2012) Nature Genetics, Supplementary Information
"""

import numpy as np
from jax import config

# Ensure 64-bit precision
config.update("jax_enable_x64", True)


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
    # Fast path for n_cvt=1 (most common case)
    if n_cvt == 1:
        return _calc_pab_ncvt1(Hi_eval, Uab)

    # General case
    return _calc_pab_general(n_cvt, Hi_eval, Uab)


def _calc_pab_ncvt1(Hi_eval: np.ndarray, Uab: np.ndarray) -> np.ndarray:
    """Optimized Pab computation for n_cvt=1.

    For n_cvt=1, we have 3 projection levels (p=0,1,2) and 6 Uab columns.
    Index mapping:
    - (1,1)=0: WW, (1,2)=1: WX, (1,3)=2: WY
    - (2,2)=3: XX, (2,3)=4: XY
    - (3,3)=5: YY

    This vectorized version computes row 0 with a single matrix-vector product,
    then uses direct formulas for rows 1 and 2.
    """
    # Row 0: Base case - weighted dot products (vectorized)
    Pab_0 = Hi_eval @ Uab  # Shape: (6,)

    # Extract values for clarity
    P0_WW = Pab_0[0]  # (1,1)
    P0_WX = Pab_0[1]  # (1,2)
    P0_WY = Pab_0[2]  # (1,3)
    P0_XX = Pab_0[3]  # (2,2)
    P0_XY = Pab_0[4]  # (2,3)
    P0_YY = Pab_0[5]  # (3,3)

    # Row 1: Project out W (covariate 1)
    # P1[ab] = P0[ab] - P0[aW] * P0[bW] / P0[WW]
    inv_P0_WW = 1.0 / P0_WW if P0_WW != 0 else 0.0
    P1_XX = P0_XX - P0_WX * P0_WX * inv_P0_WW
    P1_XY = P0_XY - P0_WX * P0_WY * inv_P0_WW
    P1_YY = P0_YY - P0_WY * P0_WY * inv_P0_WW

    # Row 2: Project out X (genotype)
    # P2[YY] = P1[YY] - P1[XY] * P1[XY] / P1[XX]
    inv_P1_XX = 1.0 / P1_XX if P1_XX != 0 else 0.0
    P2_YY = P1_YY - P1_XY * P1_XY * inv_P1_XX

    # Build output matrix (3 rows, 6 columns)
    Pab = np.zeros((3, 6), dtype=np.float64)
    Pab[0, :] = Pab_0
    Pab[1, 3] = P1_XX
    Pab[1, 4] = P1_XY
    Pab[1, 5] = P1_YY
    Pab[2, 5] = P2_YY

    return Pab


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

    # Degrees of freedom for alternative model (with SNP)
    nc_total = n_cvt + 1  # Covariates + genotype
    df = n - n_cvt - 1

    # H_inv = 1 / (lambda * eigenvalues + 1)
    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp

    # Log determinant of H
    logdet_h = np.sum(np.log(np.abs(v_temp)))

    # Compute Pab and Iab
    Pab = calc_pab(n_cvt, Hi_eval, Uab)
    Iab = calc_iab(n_cvt, Uab)

    # Calculate logdet_hiw = log|WHiW| - log|WW|
    logdet_hiw = 0.0
    for i in range(nc_total):
        index_ww = get_ab_index(i + 1, i + 1, n_cvt)
        d_pab = Pab[i, index_ww]
        d_iab = Iab[i, index_ww]
        if d_pab > 0:
            logdet_hiw += np.log(d_pab)
        if d_iab > 0:
            logdet_hiw -= np.log(d_iab)

    # Get P_yy (phenotype-phenotype after projecting out covariates and genotype)
    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    P_yy = Pab[nc_total, index_yy]

    # Minimum P_yy to avoid numerical issues (matches GEMMA's P_YY_MIN)
    P_YY_MIN = 1e-8
    if P_yy >= 0.0 and P_yy < P_YY_MIN:
        P_yy = P_YY_MIN

    # REML log-likelihood
    c = 0.5 * df * (np.log(df) - np.log(2 * np.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)

    return f
