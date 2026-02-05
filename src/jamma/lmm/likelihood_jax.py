"""JAX-optimized REML log-likelihood computation.

This module provides JIT-compiled, vectorizable implementations of the
REML likelihood functions. Designed for efficient execution on both
CPU (via XLA compilation) and GPU (via JAX's device abstraction).

Key optimizations:
- All functions are JIT-compiled for fast repeated evaluation
- Batch operations use vmap for automatic vectorization
- Pure JAX arrays avoid NumPy/JAX conversion overhead
- Static shapes enable aggressive compiler optimizations

Usage:
    # For single SNP (falls back to NumPy version for CPU efficiency)
    from jamma.lmm.likelihood import reml_log_likelihood

    # For batch processing (uses JAX for GPU acceleration)
    from jamma.lmm.likelihood_jax import batch_reml_log_likelihood

Type annotations use jaxtyping for shape documentation:
    n = n_samples, p = n_snps, g = n_grid
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, vmap

from jamma.lmm.likelihood import get_ab_index

if TYPE_CHECKING:
    from jaxtyping import Array, Float

# DEPRECATED: only used by legacy callers if any. Use build_index_table instead.
_IDX_WW = 0  # covariate-covariate
_IDX_WX = 1  # covariate-genotype
_IDX_WY = 2  # covariate-phenotype
_IDX_XX = 3  # genotype-genotype
_IDX_XY = 4  # genotype-phenotype
_IDX_YY = 5  # phenotype-phenotype


def build_index_table(n_cvt: int) -> dict:
    """Precompute all index mappings for a given n_cvt.

    This function runs at Python level (not JIT-compiled). When called
    inside a JIT function with n_cvt as a static argument, it executes
    at trace time, producing compile-time constants.

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


@partial(jit, static_argnums=(0,))
def compute_uab_jax(
    n_cvt: int,
    UtW: Float[Array, "n nc"],
    Uty: Float[Array, " n"],
    Utx: Float[Array, " n"],
) -> Float[Array, "n ni"]:
    """Compute Uab matrix for a single SNP using JAX.

    Generalized for arbitrary n_cvt. Since n_cvt is static, JIT produces
    specialized code for each covariate count.

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        Utx: Rotated genotype (n_samples,).

    Returns:
        Uab matrix (n_samples, n_index) where n_index = (n_cvt+3)*(n_cvt+2)//2.
    """
    table = build_index_table(n_cvt)
    n = Uty.shape[0]
    n_index = table["n_index"]

    # Build vectors array: [W1,...,W_ncvt, X, Y] shape (n, n_cvt+2)
    vectors = jnp.column_stack([UtW, Utx[:, None], Uty[:, None]])

    # Fill Uab using precomputed index pairs
    Uab = jnp.zeros((n, n_index), dtype=jnp.float64)
    for a_col, b_col, idx in table["uab_pairs"]:
        Uab = Uab.at[:, idx].set(vectors[:, a_col] * vectors[:, b_col])

    return Uab


@jit
def calc_pab_jax(
    Hi_eval: Float[Array, " n"], Uab: Float[Array, "n 6"]
) -> Float[Array, "3 6"]:
    """Compute Pab matrix using JAX (optimized for n_cvt=1).

    This is the core projection computation. For n_cvt=1:
    - Row 0: weighted dot products (base case)
    - Row 1: project out covariate (intercept)
    - Row 2: project out genotype

    Args:
        Hi_eval: 1 / (lambda * eigenvalues + 1) vector (n_samples,)
        Uab: Matrix products (n_samples, 6)

    Returns:
        Pab matrix (3, 6) for n_cvt=1
    """
    # Base case (p=0): weighted dot products for all (a,b) pairs
    Pab_0 = jnp.dot(Hi_eval, Uab)  # Shape: (6,)

    # p=1: project out covariate W (index 1)
    # Formula: Pab[1,ab] = Pab[0,ab] - Pab[0,aW]*Pab[0,bW]/Pab[0,WW]
    P0_WW = Pab_0[_IDX_WW]
    P0_WX = Pab_0[_IDX_WX]
    P0_WY = Pab_0[_IDX_WY]
    P0_XX = Pab_0[_IDX_XX]
    P0_XY = Pab_0[_IDX_XY]
    P0_YY = Pab_0[_IDX_YY]

    # Projection: P1[ab] = P0[ab] - P0[aW] * P0[bW] / P0[WW]
    inv_P0_WW = jnp.where(P0_WW != 0, 1.0 / P0_WW, 0.0)
    P1_XX = P0_XX - P0_WX * P0_WX * inv_P0_WW
    P1_XY = P0_XY - P0_WX * P0_WY * inv_P0_WW
    P1_YY = P0_YY - P0_WY * P0_WY * inv_P0_WW

    # p=2: project out genotype X (index 2)
    # P2[YY] = P1[YY] - P1[XY] * P1[XY] / P1[XX]
    inv_P1_XX = jnp.where(P1_XX != 0, 1.0 / P1_XX, 0.0)
    P2_YY = P1_YY - P1_XY * P1_XY * inv_P1_XX

    # Build Pab matrix
    Pab = jnp.zeros((3, 6), dtype=jnp.float64)
    Pab = Pab.at[0, :].set(Pab_0)
    Pab = Pab.at[1, _IDX_XX].set(P1_XX)
    Pab = Pab.at[1, _IDX_XY].set(P1_XY)
    Pab = Pab.at[1, _IDX_YY].set(P1_YY)
    Pab = Pab.at[2, _IDX_YY].set(P2_YY)

    return Pab


@jit
def mle_log_likelihood_jax(
    lambda_val: Float[Array, ""],
    eigenvalues: Float[Array, " n"],
    Uab: Float[Array, "n 6"],
) -> Float[Array, ""]:
    """MLE log-likelihood (not REML) for n_cvt=1.

    Key difference from REML: no logdet_hiw term, uses n instead of df.

    Args:
        lambda_val: Variance ratio to evaluate
        eigenvalues: Kinship eigenvalues
        Uab: Pre-computed Uab matrix

    Returns:
        MLE log-likelihood value
    """
    n = eigenvalues.shape[0]

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = jnp.sum(jnp.log(jnp.abs(v_temp)))

    # Compute Pab using existing calc_pab_jax
    Pab = calc_pab_jax(Hi_eval, Uab)

    # P_yy after projecting out covariates and genotype
    # For n_cvt=1: nc_total = 2, so Pab[2, _IDX_YY]
    P_yy = Pab[2, _IDX_YY]
    P_yy = jnp.where((P_yy >= 0.0) & (P_yy < 1e-8), 1e-8, P_yy)

    # MLE formula (NO logdet_hiw, uses n not df)
    c = 0.5 * n * (jnp.log(n) - jnp.log(2 * jnp.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * n * jnp.log(P_yy)

    return f


@jit
def calc_lrt_pvalue_jax(
    logl_H1: Float[Array, ""],
    logl_H0: Float[Array, ""],
) -> Float[Array, ""]:
    """Compute LRT p-value with numerical guards.

    LRT statistic: 2 * (logl_H1 - logl_H0)
    Under H0, follows chi-squared with df=1.

    Args:
        logl_H1: MLE log-likelihood under alternative
        logl_H0: MLE log-likelihood under null

    Returns:
        LRT p-value from chi2.sf(stat, df=1)
    """
    lrt_stat = 2.0 * (logl_H1 - logl_H0)
    lrt_stat = jnp.maximum(lrt_stat, 0.0)
    p_lrt = jax.scipy.stats.chi2.sf(lrt_stat, df=1)
    return p_lrt


@jit
def reml_log_likelihood_jax(
    lambda_val: Float[Array, ""],
    eigenvalues: Float[Array, " n"],
    Uab: Float[Array, "n 6"],
) -> Float[Array, ""]:
    """Compute REML log-likelihood using JAX (optimized for n_cvt=1).

    JIT-compiled version for efficient repeated evaluation during optimization.

    Args:
        lambda_val: Variance component ratio (scalar)
        eigenvalues: Eigenvalues of kinship matrix (n_samples,)
        Uab: Matrix products (n_samples, 6)

    Returns:
        Log-likelihood value (scalar)
    """
    # Compute Iab inline (identity weighting for logdet correction)
    ones = jnp.ones(eigenvalues.shape[0], dtype=jnp.float64)
    Iab = calc_pab_jax(ones, Uab)
    return _reml_with_precomputed_iab(lambda_val, eigenvalues, Uab, Iab)


@jit
def _reml_with_precomputed_iab(
    lambda_val: Float[Array, ""],
    eigenvalues: Float[Array, " n"],
    Uab: Float[Array, "n 6"],
    Iab: Float[Array, "3 6"],
) -> Float[Array, ""]:
    """REML log-likelihood with precomputed Iab (avoids redundant computation).

    This is the optimized inner loop - Iab can be computed once per SNP
    and reused across all lambda evaluations during optimization.

    Args:
        lambda_val: Variance component ratio (scalar)
        eigenvalues: Eigenvalues of kinship matrix (n_samples,)
        Uab: Matrix products (n_samples, 6)
        Iab: Precomputed identity-weighted Pab (3, 6) - constant for given Uab

    Returns:
        Log-likelihood value (scalar)
    """
    n = eigenvalues.shape[0]
    n_cvt = 1
    nc_total = n_cvt + 1  # = 2
    df = n - n_cvt - 1

    # H_inv weights
    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp

    # Log determinant of H
    logdet_h = jnp.sum(jnp.log(jnp.abs(v_temp)))

    # Compute Pab with H-inverse weighting
    Pab = calc_pab_jax(Hi_eval, Uab)

    # logdet_hiw = log|WHiW| - log|WW|
    # For n_cvt=1: sum over i=0,1 (W and X)
    logdet_hiw = 0.0
    # i=0: WW index
    d_pab_0 = Pab[0, _IDX_WW]
    d_iab_0 = Iab[0, _IDX_WW]
    logdet_hiw = logdet_hiw + jnp.where(d_pab_0 > 0, jnp.log(d_pab_0), 0.0)
    logdet_hiw = logdet_hiw - jnp.where(d_iab_0 > 0, jnp.log(d_iab_0), 0.0)
    # i=1: XX index (after projecting out W)
    d_pab_1 = Pab[1, _IDX_XX]
    d_iab_1 = Iab[1, _IDX_XX]
    logdet_hiw = logdet_hiw + jnp.where(d_pab_1 > 0, jnp.log(d_pab_1), 0.0)
    logdet_hiw = logdet_hiw - jnp.where(d_iab_1 > 0, jnp.log(d_iab_1), 0.0)

    # P_yy after projecting out covariates and genotype
    # Use GEMMA's conditional clamping: only clamp if P_yy >= 0 and P_yy < 1e-8
    # Matches lmm.cpp:854: if (P_yy >= 0.0 && P_yy < P_YY_MIN) P_yy = P_YY_MIN
    P_yy = Pab[nc_total, _IDX_YY]
    P_yy = jnp.where((P_yy >= 0.0) & (P_yy < 1e-8), 1e-8, P_yy)

    # REML log-likelihood
    c = 0.5 * df * (jnp.log(df) - jnp.log(2 * jnp.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * jnp.log(P_yy)

    return f


def batch_compute_uab(
    UtW: Float[Array, "n 1"],
    Uty: Float[Array, " n"],
    UtG: Float[Array, "n p"],
) -> Float[Array, "p n 6"]:
    """Compute Uab matrices for all SNPs at once.

    Args:
        UtW: Rotated covariates (n_samples, 1)
        Uty: Rotated phenotype (n_samples,)
        UtG: Rotated genotypes for all SNPs (n_samples, n_snps)

    Returns:
        Uab matrices (n_snps, n_samples, 6)
    """
    n_samples, n_snps = UtG.shape
    w = UtW[:, 0]
    UtG_T = UtG.T  # (n_snps, n_samples)

    # Pre-compute all element-wise products
    ww = w * w
    wy = w * Uty
    yy = Uty * Uty
    wx = w[None, :] * UtG_T
    xx = UtG_T * UtG_T
    xy = UtG_T * Uty[None, :]

    # Stack into output array using jnp.stack for efficiency
    return jnp.stack(
        [
            jnp.broadcast_to(ww, (n_snps, n_samples)),  # WW
            wx,  # WX
            jnp.broadcast_to(wy, (n_snps, n_samples)),  # WY
            xx,  # XX
            xy,  # XY
            jnp.broadcast_to(yy, (n_snps, n_samples)),  # YY
        ],
        axis=-1,
    )


@jit
def batch_compute_iab(
    Uab_batch: Float[Array, "p n 6"],
) -> Float[Array, "p 3 6"]:
    """Precompute identity-weighted Iab for all SNPs (lambda-independent).

    Iab is used in the logdet correction term of REML and only depends on Uab,
    not lambda. By precomputing it once per chunk, we avoid redundant
    computation during lambda optimization (~70 evaluations per SNP).

    Args:
        Uab_batch: Uab matrices (n_snps, n_samples, 6)

    Returns:
        Iab matrices (n_snps, 3, 6) - identity-weighted projections
    """
    n_samples = Uab_batch.shape[1]
    ones = jnp.ones(n_samples, dtype=jnp.float64)
    return vmap(lambda Uab: calc_pab_jax(ones, Uab))(Uab_batch)


@partial(jit, static_argnums=(3, 4, 5, 6), donate_argnums=(1,))
def golden_section_optimize_lambda(
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n 6"],
    Iab_batch: Float[Array, "p 3 6"],
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[Float[Array, " p"], Float[Array, " p"]]:
    """Optimize lambda using grid search + golden section refinement.

    This hybrid approach:
    1. Grid search to find approximate region (vectorized across SNPs)
    2. Golden section for precise convergence (vectorized across SNPs)

    Performance Optimization:
    ========================
    Iab (identity-weighted projection) is precomputed once per chunk and passed
    in, avoiding ~70 redundant calc_pab_jax calls per SNP during optimization.

    Mathematical Equivalence to Brent's Method:
    ============================================
    Both find the maximum of a unimodal function. Golden section achieves
    convergence rate O(0.618^n) per iteration. After grid search brackets
    the optimum to ±1 grid cell, 20 iterations reduce uncertainty by
    0.618^20 ≈ 6.6e-5, giving relative tolerance < 1e-5 for typical lambda.

    Performance:
    - Grid search: O(n_grid) REML evaluations (shared across SNPs)
    - Golden section: O(n_iter) REML evaluations per SNP (vectorized)
    - Total: ~70 REML evaluations vs ~50 for Brent (similar cost)
    - All computations stay on device (no host/device sync in loops)
    - Uses lax.fori_loop for golden section to avoid Python loop retracing
    - donate_argnums=(1,) hints XLA to reuse Uab_batch memory (when possible)

    Args:
        eigenvalues: Eigenvalues (n_samples,)
        Uab_batch: Uab matrices (n_snps, n_samples, 6)
        Iab_batch: Precomputed identity-weighted Pab (n_snps, 3, 6)
        l_min, l_max: Lambda bounds
        n_grid: Coarse grid points
        n_iter: Golden section iterations (20 gives ~1e-5 tolerance)

    Returns:
        (optimal_lambdas, optimal_logls) for each SNP
    """
    phi = 0.6180339887498949  # Golden ratio - 1

    # Stage 1: Coarse grid search on log scale (fully on device)
    log_l_min = jnp.log(l_min)
    log_l_max = jnp.log(l_max)
    log_lambdas = jnp.linspace(log_l_min, log_l_max, n_grid)
    lambdas = jnp.exp(log_lambdas)

    # Evaluate all SNPs at all grid points using vmap (stays on device)
    all_logls = _batch_grid_reml_with_iab(lambdas, eigenvalues, Uab_batch, Iab_batch)

    # Find best index per SNP
    best_idx = jnp.argmax(all_logls, axis=0)

    # Set up bounds for golden section (one grid cell on each side)
    idx_low = jnp.maximum(best_idx - 1, 0)
    idx_high = jnp.minimum(best_idx + 1, n_grid - 1)

    # Initial bounds (log scale)
    a = log_lambdas[idx_low]
    b = log_lambdas[idx_high]

    # Helper to compute REML at log-scale lambdas (batch) - uses precomputed Iab
    def compute_reml_batch(log_lams):
        lams = jnp.exp(log_lams)
        return vmap(
            lambda lam, Uab, Iab: _reml_with_precomputed_iab(
                lam, eigenvalues, Uab, Iab
            ),
            in_axes=(0, 0, 0),
        )(lams, Uab_batch, Iab_batch)

    # Initial probe points (golden ratio positions)
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = compute_reml_batch(c)
    fd = compute_reml_batch(d)

    # Stage 2: Golden section iterations using lax.fori_loop (stays on device)
    def golden_step(_, state):
        a, b, c, d, fc, fd = state

        # Where fc > fd, maximum is in [a, d], otherwise in [c, b]
        keep_left = fc > fd

        # Update bounds
        new_a = jnp.where(keep_left, a, c)
        new_b = jnp.where(keep_left, d, b)

        # Compute new interior points
        new_c = new_b - phi * (new_b - new_a)
        new_d = new_a + phi * (new_b - new_a)

        # Compute REML at the new position only
        new_logl = compute_reml_batch(jnp.where(keep_left, new_c, new_d))

        # Reuse function values where possible
        new_fc = jnp.where(keep_left, new_logl, fd)
        new_fd = jnp.where(keep_left, fc, new_logl)

        return (new_a, new_b, new_c, new_d, new_fc, new_fd)

    init_state = (a, b, c, d, fc, fd)
    final_state = jax.lax.fori_loop(0, n_iter, golden_step, init_state)
    a, b, c, d, fc, fd = final_state

    # Return midpoint as optimal (log scale)
    log_opt = (a + b) / 2
    best_lambdas = jnp.exp(log_opt)

    # Final likelihood at optimal point
    best_logls = compute_reml_batch(log_opt)

    return best_lambdas, best_logls


def _batch_grid_reml(
    lambdas: Float[Array, " g"],
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n 6"],
) -> Float[Array, "g p"]:
    """Compute REML at all grid points for all SNPs (fully on device).

    Uses vmap over lambda values to avoid Python loops and host/device sync.

    Args:
        lambdas: Grid of lambda values (n_grid,)
        eigenvalues: Eigenvalues (n_samples,)
        Uab_batch: Uab matrices (n_snps, n_samples, 6)

    Returns:
        Log-likelihoods (n_grid, n_snps)
    """

    # vmap over lambda values, then vmap over SNPs
    def reml_for_lambda(lam):
        return vmap(
            lambda Uab: reml_log_likelihood_jax(lam, eigenvalues, Uab), in_axes=0
        )(Uab_batch)

    return vmap(reml_for_lambda)(lambdas)


def _batch_grid_reml_with_iab(
    lambdas: Float[Array, " g"],
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n 6"],
    Iab_batch: Float[Array, "p 3 6"],
) -> Float[Array, "g p"]:
    """Compute REML at all grid points using precomputed Iab (optimized).

    Uses vmap over lambda values to avoid Python loops and host/device sync.
    Iab is precomputed once per chunk, avoiding redundant identity-weighted
    projection computations (~n_grid fewer calc_pab_jax calls per SNP).

    Args:
        lambdas: Grid of lambda values (n_grid,)
        eigenvalues: Eigenvalues (n_samples,)
        Uab_batch: Uab matrices (n_snps, n_samples, 6)
        Iab_batch: Precomputed identity-weighted Pab (n_snps, 3, 6)

    Returns:
        Log-likelihoods (n_grid, n_snps)
    """

    # vmap over lambda values, then vmap over SNPs
    def reml_for_lambda(lam):
        return vmap(
            lambda Uab, Iab: _reml_with_precomputed_iab(lam, eigenvalues, Uab, Iab),
            in_axes=(0, 0),
        )(Uab_batch, Iab_batch)

    return vmap(reml_for_lambda)(lambdas)


@jit
def calc_wald_stats_jax(
    lambda_val: Float[Array, ""],
    eigenvalues: Float[Array, " n"],
    Uab: Float[Array, "n 6"],
    n_samples: int,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Compute Wald test statistics using JAX (n_cvt=1).

    Args:
        lambda_val: Optimized variance ratio (scalar)
        eigenvalues: Eigenvalues (n_samples,)
        Uab: Matrix products (n_samples, 6)
        n_samples: Number of samples

    Returns:
        Tuple of (beta, se, p_wald) - all scalars
    """
    n_cvt = 1
    df = n_samples - n_cvt - 1

    # Compute Pab
    Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
    Pab = calc_pab_jax(Hi_eval, Uab)

    # Extract values (using n_cvt=1 indices)
    P_XX = Pab[n_cvt, _IDX_XX]  # After projecting out W
    P_XY = Pab[n_cvt, _IDX_XY]
    P_YY = Pab[n_cvt, _IDX_YY]
    Px_YY = Pab[n_cvt + 1, _IDX_YY]  # After projecting out W and X

    # Clamp Px_YY like NumPy path (GEMMA lmm.cpp:854)
    # Only clamp if >= 0 and < 1e-8; leave negative values to produce NaN
    Px_YY = jnp.where((Px_YY >= 0.0) & (Px_YY < 1e-8), 1e-8, Px_YY)

    # Effect size and standard error
    # Guard P_XX <= 0: SNP has no variance, return NaN for all stats
    # GEMMA safe_sqrt: if |d| < 0.001, use abs(d) to tolerate small negatives
    # This matches GEMMA mathfunc.cpp:122-131
    is_valid = P_XX > 0

    # Safe division avoiding divide-by-zero
    beta = jnp.where(is_valid, P_XY / jnp.where(is_valid, P_XX, 1.0), jnp.nan)
    tau = df / Px_YY
    variance_beta = jnp.where(
        is_valid, 1.0 / (tau * jnp.where(is_valid, P_XX, 1.0)), jnp.nan
    )
    # Apply safe_sqrt: for small negatives (|v| < 0.001), use abs; otherwise use as-is
    # In JAX, we handle this with jnp.where for the small negative case
    variance_safe = jnp.where(
        jnp.abs(variance_beta) < 0.001,
        jnp.abs(variance_beta),
        variance_beta,
    )
    # For large negatives, sqrt will produce NaN (matching GEMMA behavior)
    se = jnp.where(is_valid, jnp.sqrt(variance_safe), jnp.nan)

    # F-statistic and p-value
    f_stat = (P_YY - Px_YY) * tau

    # Guard: if f_stat <= 0, p-value = 1.0 (no evidence against null)
    # Clamp z to [0, 1] to ensure betainc is well-defined
    z = df / (df + jnp.maximum(f_stat, 1e-10))
    z = jnp.clip(z, 0.0, 1.0)
    p_wald = jax.scipy.special.betainc(df / 2.0, 0.5, z)
    # If f_stat was non-positive or P_XX invalid, return p=NaN
    p_wald = jnp.where(f_stat <= 0, 1.0, p_wald)
    p_wald = jnp.where(is_valid, p_wald, jnp.nan)

    return beta, se, p_wald


# Vectorized version for batch processing
batch_calc_wald_stats = vmap(
    calc_wald_stats_jax,
    in_axes=(0, None, 0, None),  # lambda per SNP, shared eigenvalues, Uab per SNP
    out_axes=(0, 0, 0),  # beta, se, p per SNP
)
