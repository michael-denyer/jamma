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

    # For batch processing (JAX GPU acceleration)
    from jamma.lmm.likelihood_jax import (
        batch_compute_uab, golden_section_optimize_lambda)

Type annotations use jaxtyping for shape documentation:
    n = n_samples, p = n_snps, g = n_grid
"""

from __future__ import annotations

import functools
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, vmap

from jamma.lmm.likelihood import (
    _P_YY_MIN,
    build_index_table,
    get_ab_index,  # noqa: F401
)


@functools.lru_cache(maxsize=8)
def classify_uab_columns(n_cvt: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Classify Uab columns as invariant or SNP-varying.

    A column is SNP-varying if its (a_col, b_col) pair involves the genotype
    (0-based index = n_cvt). Otherwise it is invariant across SNPs.

    Pure Python, lru_cached. When called inside JIT with static n_cvt,
    executes at trace time producing compile-time constants.

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


if TYPE_CHECKING:
    from jaxtyping import Array, Float


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
def calc_pab_ncvt1_jax(
    s_ww: Float[Array, ""],
    s_wx: Float[Array, ""],
    s_wy: Float[Array, ""],
    s_xx: Float[Array, ""],
    s_xy: Float[Array, ""],
    s_yy: Float[Array, ""],
) -> Float[Array, "3 6"]:
    """Direct Pab for n_cvt=1 — no recursive loop, no index table.

    Mirrors calc_pab_ncvt1_split from _lmm_accel.c:674-697.
    Takes 6 scalar dot-product sums and returns Pab as a (3, 6) array
    using direct Schur complement arithmetic.

    Column ordering matches the n_cvt=1 Uab layout:
        col 0: ww, col 1: wx, col 2: wy, col 3: xx, col 4: xy, col 5: yy

    Args:
        s_ww: Hi-weighted dot product of w with w.
        s_wx: Hi-weighted dot product of w with x (SNP-varying).
        s_wy: Hi-weighted dot product of w with y.
        s_xx: Hi-weighted dot product of x with x (SNP-varying).
        s_xy: Hi-weighted dot product of x with y (SNP-varying).
        s_yy: Hi-weighted dot product of y with y.

    Returns:
        Pab matrix (3, 6).
    """
    pab = jnp.zeros((3, 6), dtype=jnp.float64)
    # Row 0: raw sums
    pab = pab.at[0, 0].set(s_ww)
    pab = pab.at[0, 1].set(s_wx)
    pab = pab.at[0, 2].set(s_wy)
    pab = pab.at[0, 3].set(s_xx)
    pab = pab.at[0, 4].set(s_xy)
    pab = pab.at[0, 5].set(s_yy)
    # Row 1: project out W (Schur complement)
    inv_ww = jnp.where(s_ww != 0, 1.0 / s_ww, 0.0)
    pab = pab.at[1, 3].set(s_xx - s_wx * s_wx * inv_ww)
    pab = pab.at[1, 4].set(s_xy - s_wx * s_wy * inv_ww)
    pab = pab.at[1, 5].set(s_yy - s_wy * s_wy * inv_ww)
    # Row 2: project out X (Schur complement)
    ps_xx = pab[1, 3]
    inv_xx = jnp.where(ps_xx != 0, 1.0 / ps_xx, 0.0)
    pab = pab.at[2, 5].set(pab[1, 5] - pab[1, 4] * pab[1, 4] * inv_xx)
    return pab


@partial(jit, static_argnums=(0,))
def calc_pab_jax(
    n_cvt: int,
    Hi_eval: Float[Array, " n"],
    Uab: Float[Array, "n ni"],
) -> Float[Array, "nr ni"]:
    """Compute Pab matrix using JAX for arbitrary n_cvt.

    Recursive projection computation matching GEMMA's CalcPab exactly.
    Since n_cvt is static, all loops are fully unrolled by JIT.

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        Hi_eval: 1 / (lambda * eigenvalues + 1) vector (n_samples,).
        Uab: Matrix products (n_samples, n_index).

    Returns:
        Pab matrix (n_cvt+2, n_index).
    """
    table = build_index_table(n_cvt)
    n_index = table["n_index"]

    Pab = jnp.zeros((n_cvt + 2, n_index), dtype=jnp.float64)

    # Row 0: weighted dot products for all (a,b) pairs
    Pab = Pab.at[0, :].set(jnp.dot(Hi_eval, Uab))

    # Rows 1..n_cvt+1: recursive projection (fully unrolled since n_cvt is static)
    for p in range(1, n_cvt + 2):
        for _a, _b, index_ab, index_aw, index_bw, index_ww in table["pab_recursion"][p]:
            ps_ww = Pab[p - 1, index_ww]
            inv_ps_ww = jnp.where(ps_ww != 0, 1.0 / ps_ww, 0.0)
            val = (
                Pab[p - 1, index_ab]
                - Pab[p - 1, index_aw] * Pab[p - 1, index_bw] * inv_ps_ww
            )
            Pab = Pab.at[p, index_ab].set(val)

    return Pab


@partial(jit, static_argnums=(0,))
def mle_log_likelihood_jax(
    n_cvt: int,
    lambda_val: Float[Array, ""],
    eigenvalues: Float[Array, " n"],
    Uab: Float[Array, "n ni"],
) -> Float[Array, ""]:
    """MLE log-likelihood (not REML) for arbitrary n_cvt.

    Key difference from REML: no logdet_hiw term, uses n instead of df.

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        lambda_val: Variance ratio to evaluate.
        eigenvalues: Kinship eigenvalues.
        Uab: Pre-computed Uab matrix (n_samples, n_index).

    Returns:
        MLE log-likelihood value.
    """
    n = eigenvalues.shape[0]
    nc_total = n_cvt + 1
    table = build_index_table(n_cvt)
    idx_yy = table["idx_yy"]

    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp
    logdet_h = jnp.sum(jnp.log(jnp.abs(v_temp)))

    Pab = calc_pab_jax(n_cvt, Hi_eval, Uab)

    # P_yy after projecting out covariates and genotype
    # Negative P_yy → NaN (numerical breakdown); near-zero → clamp to avoid log(0)
    P_yy = Pab[nc_total, idx_yy]
    P_yy = jnp.where(P_yy < 0.0, jnp.nan, P_yy)
    P_yy = jnp.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)

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


@partial(jit, static_argnums=(0,))
def reml_log_likelihood_jax(
    n_cvt: int,
    lambda_val: Float[Array, ""],
    eigenvalues: Float[Array, " n"],
    Uab: Float[Array, "n ni"],
) -> Float[Array, ""]:
    """Compute REML log-likelihood using JAX for arbitrary n_cvt.

    JIT-compiled version for efficient repeated evaluation during optimization.

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        lambda_val: Variance component ratio (scalar).
        eigenvalues: Eigenvalues of kinship matrix (n_samples,).
        Uab: Matrix products (n_samples, n_index).

    Returns:
        Log-likelihood value (scalar).
    """
    # Compute Iab inline (identity weighting for logdet correction)
    ones = jnp.ones(eigenvalues.shape[0], dtype=jnp.float64)
    Iab = calc_pab_jax(n_cvt, ones, Uab)
    return _reml_with_precomputed_iab(n_cvt, lambda_val, eigenvalues, Uab, Iab)


@partial(jit, static_argnums=(0,))
def _reml_with_precomputed_iab(
    n_cvt: int,
    lambda_val: Float[Array, ""],
    eigenvalues: Float[Array, " n"],
    Uab: Float[Array, "n ni"],
    Iab: Float[Array, "nr ni"],
) -> Float[Array, ""]:
    """REML log-likelihood with precomputed Iab for arbitrary n_cvt.

    This is the optimized inner loop - Iab can be computed once per SNP
    and reused across all lambda evaluations during optimization.

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        lambda_val: Variance component ratio (scalar).
        eigenvalues: Eigenvalues of kinship matrix (n_samples,).
        Uab: Matrix products (n_samples, n_index).
        Iab: Precomputed identity-weighted Pab (n_cvt+2, n_index).

    Returns:
        Log-likelihood value (scalar).
    """
    n = eigenvalues.shape[0]
    nc_total = n_cvt + 1
    df = n - n_cvt - 1
    table = build_index_table(n_cvt)
    idx_yy = table["idx_yy"]

    # H_inv weights
    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp

    # Log determinant of H
    logdet_h = jnp.sum(jnp.log(jnp.abs(v_temp)))

    # Compute Pab with H-inverse weighting
    Pab = calc_pab_jax(n_cvt, Hi_eval, Uab)

    # logdet_hiw = log|WHiW| - log|WW|
    # For each diagonal element i=0..n_cvt, accumulate
    # log(Pab[i, diag_col]) - log(Iab[i, diag_col])
    logdet_hiw = 0.0
    for row, col in table["logdet_diag_indices"]:
        d_pab = Pab[row, col]
        d_iab = Iab[row, col]
        logdet_hiw = logdet_hiw + jnp.where(d_pab > 0, jnp.log(d_pab), 0.0)
        logdet_hiw = logdet_hiw - jnp.where(d_iab > 0, jnp.log(d_iab), 0.0)

    # P_yy after projecting out covariates and genotype
    # Matches lmm.cpp:854: negative P_yy → NaN (numerical breakdown),
    # near-zero P_yy → clamp to P_YY_MIN to avoid log(0)
    P_yy = Pab[nc_total, idx_yy]
    P_yy = jnp.where(P_yy < 0.0, jnp.nan, P_yy)
    P_yy = jnp.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)

    # REML log-likelihood (NaN P_yy propagates → optimizer avoids this region)
    c = 0.5 * df * (jnp.log(df) - jnp.log(2 * jnp.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * jnp.log(P_yy)

    return f


@jit
def _reml_ncvt1_split(
    Hi_eval: Float[Array, " n"],
    uab_varying: Float[Array, "n 3"],
    s_ww: Float[Array, ""],
    s_wy: Float[Array, ""],
    log_s_ww: Float[Array, ""],
    pab1_5: Float[Array, ""],
    logdet_iab: Float[Array, ""],
    logdet_h: Float[Array, ""],
    df: int,
) -> Float[Array, ""]:
    """REML log-likelihood for n_cvt=1 with split invariant/varying sums.

    Mirrors reml_logl_ncvt1_cached_split from _lmm_accel.c:712-766.
    Computes only 3 SNP-varying dot products; invariant sums come from
    precomputed arguments. This halves DRAM reads vs the general path.

    Args:
        Hi_eval: 1/(lambda * eigenvalues + 1) vector (n_samples,).
        uab_varying: SNP-varying Uab columns (n_samples, 3) — [wx, xx, xy].
        s_ww: Precomputed invariant sum: dot(Hi_eval, ww).
        s_wy: Precomputed invariant sum: dot(Hi_eval, wy).
        log_s_ww: log(s_ww) for logdet computation (precomputed).
        pab1_5: s_yy - s_wy^2/s_ww (completely SNP-invariant per lambda).
        logdet_iab: log|WW| contribution (precomputed, lambda-independent).
        logdet_h: log|H| (precomputed for this lambda).
        df: Degrees of freedom = n_samples - n_cvt - 1 = n_samples - 2.

    Returns:
        REML log-likelihood scalar.
    """
    # 3 SNP-varying dot products only
    s_wx = jnp.dot(Hi_eval, uab_varying[:, 0])
    s_xx = jnp.dot(Hi_eval, uab_varying[:, 1])
    s_xy = jnp.dot(Hi_eval, uab_varying[:, 2])

    # Pab row 1 (project out W)
    inv_ww = jnp.where(s_ww != 0, 1.0 / s_ww, 0.0)
    p1_xx = s_xx - s_wx * s_wx * inv_ww
    p1_xy = s_xy - s_wx * s_wy * inv_ww
    # p1_yy is completely invariant (precomputed as pab1_5)

    # Pab row 2 (project out X)
    inv_xx = jnp.where(p1_xx != 0, 1.0 / p1_xx, 0.0)
    P_yy = pab1_5 - p1_xy * p1_xy * inv_xx

    # logdet_hiw = log|WHiW| - log|WW|
    # For n_cvt=1: diagonal entries are s_ww (row 0) and p1_xx (row 1)
    logdet_pab = log_s_ww + jnp.where(p1_xx > 0, jnp.log(p1_xx), 0.0)
    logdet_hiw = logdet_pab - logdet_iab

    # Guard P_yy: negative -> NaN, near-zero -> clamp
    P_yy = jnp.where(P_yy < 0.0, jnp.nan, P_yy)
    P_yy = jnp.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)

    c = 0.5 * df * (jnp.log(df) - jnp.log(2 * jnp.pi) - 1.0)
    return c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * jnp.log(P_yy)


@partial(jit, static_argnums=(0,))
def batch_compute_uab(
    n_cvt: int,
    UtW: Float[Array, "n nc"],
    Uty: Float[Array, " n"],
    UtG: Float[Array, "n p"],
) -> Float[Array, "p n ni"]:
    """Compute Uab matrices for all SNPs at once.

    Generalized for arbitrary n_cvt. Uses vmap over SNPs to produce
    one Uab matrix per genotype column.

    For n_cvt=1, keeps the explicit broadcasting fast path to avoid
    vmap overhead (the n_cvt==1 branch is resolved at trace time
    since n_cvt is static).

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        UtG: Rotated genotypes for all SNPs (n_samples, n_snps).

    Returns:
        Uab matrices (n_snps, n_samples, n_index).
    """
    if n_cvt == 1:
        # Fast path: explicit broadcasting avoids vmap overhead
        n_samples, n_snps = UtG.shape
        w = UtW[:, 0]
        UtG_T = UtG.T  # (n_snps, n_samples)

        ww = w * w
        wy = w * Uty
        yy = Uty * Uty
        wx = w[None, :] * UtG_T
        xx = UtG_T * UtG_T
        xy = UtG_T * Uty[None, :]

        return jnp.stack(
            [
                jnp.broadcast_to(ww, (n_snps, n_samples)),
                wx,
                jnp.broadcast_to(wy, (n_snps, n_samples)),
                xx,
                xy,
                jnp.broadcast_to(yy, (n_snps, n_samples)),
            ],
            axis=-1,
        )

    # General path: vmap over SNPs
    return vmap(lambda utx: compute_uab_jax(n_cvt, UtW, Uty, utx))(UtG.T)


@partial(jit, static_argnums=(0,))
def batch_compute_iab(
    n_cvt: int,
    Uab_batch: Float[Array, "p n ni"],
) -> Float[Array, "p nr ni"]:
    """Precompute identity-weighted Iab for all SNPs (lambda-independent).

    Iab is used in the logdet correction term of REML and only depends on Uab,
    not lambda. By precomputing it once per chunk, we avoid redundant
    computation during lambda optimization (~70 evaluations per SNP).

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Iab matrices (n_snps, n_cvt+2, n_index) - identity-weighted projections.
    """
    n_samples = Uab_batch.shape[1]
    ones = jnp.ones(n_samples, dtype=jnp.float64)
    return vmap(lambda Uab: calc_pab_jax(n_cvt, ones, Uab))(Uab_batch)


def _golden_section_refine(
    compute_batch,
    all_logls: Float[Array, "g p"],
    log_lambdas: Float[Array, " g"],
    n_grid: int,
    n_iter: int,
) -> tuple[Float[Array, " p"], Float[Array, " p"]]:
    """Grid-to-golden-section refinement for lambda optimization.

    Given grid log-likelihoods, brackets the optimum per SNP and runs
    golden section iterations using lax.fori_loop (stays on device).

    This is the shared core of both REML and MLE lambda optimizers.
    The only difference is the compute_batch function:
    - REML: evaluates _reml_with_precomputed_iab
    - MLE: evaluates mle_log_likelihood_jax

    Convergence: 0.618^20 ~ 6.6e-5 relative tolerance after 20 iterations.

    Args:
        compute_batch: Function (log_lambdas_per_snp,) -> (logls_per_snp,).
            Evaluates the likelihood at one log-lambda value per SNP.
        all_logls: Grid log-likelihoods (n_grid, n_snps).
        log_lambdas: Log-scale grid points (n_grid,).
        n_grid: Number of grid points.
        n_iter: Golden section iterations.

    Returns:
        (optimal_lambdas, optimal_logls) for each SNP.
    """
    phi = 0.6180339887498949  # Golden ratio - 1

    # Find best grid point per SNP and bracket
    safe_logls = jnp.where(jnp.isnan(all_logls), -jnp.inf, all_logls)
    best_idx = jnp.argmax(safe_logls, axis=0)
    idx_low = jnp.maximum(best_idx - 1, 0)
    idx_high = jnp.minimum(best_idx + 1, n_grid - 1)

    a = log_lambdas[idx_low]
    b = log_lambdas[idx_high]

    # Initial probe points
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = compute_batch(c)
    fd = compute_batch(d)

    # Golden section iterations via lax.fori_loop (stays on device)
    def golden_step(_, state):
        a, b, c, d, fc, fd = state
        keep_left = fc > fd

        new_a = jnp.where(keep_left, a, c)
        new_b = jnp.where(keep_left, d, b)
        new_c = new_b - phi * (new_b - new_a)
        new_d = new_a + phi * (new_b - new_a)

        new_logl = compute_batch(jnp.where(keep_left, new_c, new_d))
        new_fc = jnp.where(keep_left, new_logl, fd)
        new_fd = jnp.where(keep_left, fc, new_logl)

        return (new_a, new_b, new_c, new_d, new_fc, new_fd)

    final_state = jax.lax.fori_loop(0, n_iter, golden_step, (a, b, c, d, fc, fd))
    a, b = final_state[0], final_state[1]

    log_opt = (a + b) / 2
    best_lambdas = jnp.exp(log_opt)
    best_logls = compute_batch(log_opt)

    return best_lambdas, best_logls


@partial(jit, static_argnums=(0, 4, 5, 6, 7))
def golden_section_optimize_lambda(
    n_cvt: int,
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n ni"],
    Iab_batch: Float[Array, "p nr ni"],
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[Float[Array, " p"], Float[Array, " p"]]:
    """Optimize REML lambda using grid search + golden section refinement.

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
    the optimum to +/-1 grid cell, 20 iterations reduce uncertainty by
    0.618^20 ~ 6.6e-5, giving relative tolerance < 1e-5 for typical lambda.

    Performance:
    - Grid search: O(n_grid) likelihood evaluations (shared across SNPs)
    - Golden section: O(n_iter) likelihood evaluations per SNP (vectorized)
    - Total: ~70 evaluations vs ~50 for Brent (similar cost)
    - All computations stay on device (no host/device sync in loops)

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        eigenvalues: Eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index).
        l_min, l_max: Lambda bounds.
        n_grid: Coarse grid points.
        n_iter: Golden section iterations (20 gives ~1e-5 tolerance).

    Returns:
        (optimal_lambdas, optimal_logls) for each SNP.
    """
    # Stage 1: Coarse grid search on log scale
    log_l_min = jnp.log(l_min)
    log_l_max = jnp.log(l_max)
    log_lambdas = jnp.linspace(log_l_min, log_l_max, n_grid)
    lambdas = jnp.exp(log_lambdas)

    all_logls = _batch_grid_reml_with_iab(
        n_cvt, lambdas, eigenvalues, Uab_batch, Iab_batch
    )

    if n_cvt == 1:
        # n_cvt=1 fast path: extract invariant columns once, reuse across SNPs.
        # During golden section, each SNP has its own lambda, so invariant sums
        # must be computed per-SNP lambda. The invariant Uab columns (ww, wy, yy)
        # are still shared — extract once and close over them.
        ww = Uab_batch[0, :, 0]  # same for all SNPs
        wy = Uab_batch[0, :, 2]
        yy = Uab_batch[0, :, 5]
        uab_varying = Uab_batch[:, :, [1, 3, 4]]  # (n_snps, n_samples, 3)
        n_samples = eigenvalues.shape[0]
        df = n_samples - 2

        # logdet_iab: lambda-independent, but SNP-varying (xx, wx differ per SNP).
        # logdet_diag_indices = [(0,0), (1,3)] for n_cvt=1:
        #   Iab[0, 0] = sum(ww) — invariant; Iab[1, 3] = projected xx — SNP-varying
        iab_s_ww_all = Iab_batch[:, 0, 0]  # (n_snps,) — actually all equal for n_cvt=1
        iab_p1_xx_all = Iab_batch[:, 1, 3]  # (n_snps,) — SNP-varying
        logdet_iab_all = jnp.where(
            iab_s_ww_all > 0, jnp.log(iab_s_ww_all), 0.0
        ) + jnp.where(iab_p1_xx_all > 0, jnp.log(iab_p1_xx_all), 0.0)  # (n_snps,)

        def compute_reml_batch(log_lams):
            lams = jnp.exp(log_lams)

            def reml_for_snp(lam, uab_snp, logdet_iab):
                v_temp = lam * eigenvalues + 1.0
                Hi_eval = 1.0 / v_temp
                logdet_h = jnp.sum(jnp.log(jnp.abs(v_temp)))
                s_ww = jnp.dot(Hi_eval, ww)
                s_wy = jnp.dot(Hi_eval, wy)
                s_yy_val = jnp.dot(Hi_eval, yy)
                log_s_ww = jnp.where(s_ww > 0, jnp.log(s_ww), 0.0)
                inv_ww = jnp.where(s_ww != 0, 1.0 / s_ww, 0.0)
                pab1_5 = s_yy_val - s_wy * s_wy * inv_ww
                return _reml_ncvt1_split(
                    Hi_eval,
                    uab_snp,
                    s_ww,
                    s_wy,
                    log_s_ww,
                    pab1_5,
                    logdet_iab,
                    logdet_h,
                    df,
                )

            return vmap(reml_for_snp, in_axes=(0, 0, 0))(
                lams, uab_varying, logdet_iab_all
            )

    else:
        # General n_cvt path: use split invariant/varying optimization
        invariant_indices, varying_indices = classify_uab_columns(n_cvt)
        inv_idx_arr = jnp.array(list(invariant_indices))
        var_idx_arr = jnp.array(list(varying_indices))

        # (n_samples, n_inv)
        uab_invariant = Uab_batch[0, :, :][:, inv_idx_arr]
        # (n_snps, n_samples, n_var)
        uab_varying_batch_gs = Uab_batch[:, :, :][:, :, var_idx_arr]

        n_samples = eigenvalues.shape[0]
        df_gs = n_samples - n_cvt - 1
        table_gs = build_index_table(n_cvt)

        # Per-SNP logdet_iab (lambda-independent)
        logdet_iab_gs = jnp.zeros(Uab_batch.shape[0], dtype=jnp.float64)
        for row, col in table_gs["logdet_diag_indices"]:
            d_iab = Iab_batch[:, row, col]
            logdet_iab_gs = logdet_iab_gs + jnp.where(d_iab > 0, jnp.log(d_iab), 0.0)

        def compute_reml_batch(log_lams):
            lams = jnp.exp(log_lams)

            def reml_for_snp(lam, uab_var_snp, logdet_iab_snp):
                v_temp = lam * eigenvalues + 1.0
                Hi_eval = 1.0 / v_temp
                logdet_h = jnp.sum(jnp.log(jnp.abs(v_temp)))
                inv_sums = jnp.dot(Hi_eval, uab_invariant)
                return _reml_general_split(
                    n_cvt,
                    Hi_eval,
                    uab_var_snp,
                    inv_sums,
                    logdet_iab_snp,
                    logdet_h,
                    df_gs,
                )

            return vmap(reml_for_snp, in_axes=(0, 0, 0))(
                lams, uab_varying_batch_gs, logdet_iab_gs
            )

    # Stage 2: Golden section refinement
    return _golden_section_refine(
        compute_reml_batch, all_logls, log_lambdas, n_grid, n_iter
    )


@partial(jit, static_argnums=(0,))
def _reml_general_split(
    n_cvt: int,
    Hi_eval: Float[Array, " n"],
    uab_varying: Float[Array, "n nv"],
    invariant_sums: Float[Array, " ni_inv"],
    logdet_iab: Float[Array, ""],
    logdet_h: Float[Array, ""],
    df: int,
) -> Float[Array, ""]:
    """REML log-likelihood for general n_cvt with split invariant/varying sums.

    Computes only the SNP-varying dot products; invariant sums come from
    precomputed arguments. This reduces DRAM bandwidth by ~70% for n_cvt=4
    (15 of 21 columns are invariant).

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        Hi_eval: 1/(lambda * eigenvalues + 1) vector (n_samples,).
        uab_varying: SNP-varying Uab columns (n_samples, n_varying).
        invariant_sums: Precomputed dot(Hi_eval, uab_invariant) per invariant column.
        logdet_iab: log|WW| contribution (precomputed, lambda-independent).
        logdet_h: log|H| (precomputed for this lambda).
        df: Degrees of freedom = n_samples - n_cvt - 1.

    Returns:
        REML log-likelihood scalar.
    """
    table = build_index_table(n_cvt)
    n_index = table["n_index"]
    idx_yy = table["idx_yy"]
    nc_total = n_cvt + 1
    invariant_indices = classify_uab_columns(n_cvt)[0]
    varying_indices = classify_uab_columns(n_cvt)[1]

    # Compute varying sums
    varying_sums = jnp.dot(Hi_eval, uab_varying)  # (n_varying,)

    # Build full Pab row 0 from invariant + varying sums
    row0 = jnp.zeros(n_index, dtype=jnp.float64)
    for i, idx in enumerate(invariant_indices):
        row0 = row0.at[idx].set(invariant_sums[i])
    for i, idx in enumerate(varying_indices):
        row0 = row0.at[idx].set(varying_sums[i])

    # Build full Pab via recursion (rows 1..n_cvt+1)
    Pab = jnp.zeros((n_cvt + 2, n_index), dtype=jnp.float64)
    Pab = Pab.at[0, :].set(row0)

    for p in range(1, n_cvt + 2):
        for _a, _b, index_ab, index_aw, index_bw, index_ww in table["pab_recursion"][p]:
            ps_ww = Pab[p - 1, index_ww]
            inv_ps_ww = jnp.where(ps_ww != 0, 1.0 / ps_ww, 0.0)
            val = (
                Pab[p - 1, index_ab]
                - Pab[p - 1, index_aw] * Pab[p - 1, index_bw] * inv_ps_ww
            )
            Pab = Pab.at[p, index_ab].set(val)

    # logdet_hiw = log|WHiW| - log|WW|
    logdet_hiw = 0.0
    for row, col in table["logdet_diag_indices"]:
        d_pab = Pab[row, col]
        logdet_hiw = logdet_hiw + jnp.where(d_pab > 0, jnp.log(d_pab), 0.0)
    logdet_hiw = logdet_hiw - logdet_iab

    # P_yy guard
    P_yy = Pab[nc_total, idx_yy]
    P_yy = jnp.where(P_yy < 0.0, jnp.nan, P_yy)
    P_yy = jnp.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)

    c = 0.5 * df * (jnp.log(df) - jnp.log(2 * jnp.pi) - 1.0)
    return c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * jnp.log(P_yy)


def _batch_grid_reml_general(
    n_cvt: int,
    lambdas: Float[Array, " g"],
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n ni"],
    Iab_batch: Float[Array, "p nr ni"],
) -> Float[Array, "g p"]:
    """General n_cvt split path for grid REML: precompute invariant sums per lambda.

    Mirrors _batch_grid_reml_ncvt1 but for arbitrary n_cvt. Splits Uab columns
    into invariant (shared across SNPs) and varying (per-SNP), precomputes
    invariant sums once per lambda, then vmaps over SNPs.

    Args:
        n_cvt: Number of covariates (>1).
        lambdas: Grid of lambda values (n_grid,).
        eigenvalues: Eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Identity-weighted Pab (n_snps, n_cvt+2, n_index).

    Returns:
        Log-likelihoods (n_grid, n_snps).
    """
    n_samples = eigenvalues.shape[0]
    df = n_samples - n_cvt - 1
    table = build_index_table(n_cvt)
    invariant_indices, varying_indices = classify_uab_columns(n_cvt)

    # Extract invariant Uab columns from SNP 0 (identical across all SNPs)
    inv_idx_list = list(invariant_indices)
    var_idx_list = list(varying_indices)
    # (n_samples, n_inv)
    uab_invariant = Uab_batch[0, :, :][:, jnp.array(inv_idx_list)]

    # Extract varying Uab columns per SNP: (n_snps, n_samples, n_var)
    uab_varying_batch = Uab_batch[:, :, :][:, :, jnp.array(var_idx_list)]

    # Precompute per-SNP logdet_iab (lambda-independent)
    logdet_iab_all = jnp.zeros(Uab_batch.shape[0], dtype=jnp.float64)
    for row, col in table["logdet_diag_indices"]:
        d_iab = Iab_batch[:, row, col]  # (n_snps,)
        logdet_iab_all = logdet_iab_all + jnp.where(d_iab > 0, jnp.log(d_iab), 0.0)

    # Precompute per-lambda invariants
    def compute_lambda_invariants(lam):
        v_temp = lam * eigenvalues + 1.0
        Hi_eval = 1.0 / v_temp
        logdet_h = jnp.sum(jnp.log(jnp.abs(v_temp)))
        inv_sums = jnp.dot(Hi_eval, uab_invariant)  # (n_inv,)
        return Hi_eval, inv_sums, logdet_h

    Hi_eval_grid, inv_sums_grid, logdet_h_grid = vmap(compute_lambda_invariants)(
        lambdas
    )
    # Hi_eval_grid: (n_grid, n_samples)
    # inv_sums_grid: (n_grid, n_inv)
    # logdet_h_grid: (n_grid,)

    # Inner function: for one SNP, evaluate REML across all lambdas
    def reml_for_snp(uab_var_snp, logdet_iab_snp):
        def reml_at_lambda(Hi_eval, inv_sums, logdet_h):
            return _reml_general_split(
                n_cvt,
                Hi_eval,
                uab_var_snp,
                inv_sums,
                logdet_iab_snp,
                logdet_h,
                df,
            )

        return vmap(reml_at_lambda)(Hi_eval_grid, inv_sums_grid, logdet_h_grid)

    # vmap over SNPs
    all_logls = vmap(reml_for_snp)(uab_varying_batch, logdet_iab_all)
    return all_logls.T  # (n_grid, n_snps)


def _batch_grid_reml_with_iab(
    n_cvt: int,
    lambdas: Float[Array, " g"],
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n ni"],
    Iab_batch: Float[Array, "p nr ni"],
) -> Float[Array, "g p"]:
    """Compute REML at all grid points using precomputed Iab (optimized).

    For n_cvt=1: uses a fast path that precomputes invariant sums once per
    lambda and vmaps SNPs outer / lambdas inner, keeping per-SNP data in
    cache across all lambda evaluations. This halves DRAM reads (3 varying
    columns vs 6 full columns per SNP per grid eval).

    For n_cvt>1: falls through to the general lambda-outer vmap.

    Args:
        n_cvt: Number of covariates (passed through to _reml_with_precomputed_iab).
        lambdas: Grid of lambda values (n_grid,).
        eigenvalues: Eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index).

    Returns:
        Log-likelihoods (n_grid, n_snps).
    """
    if n_cvt == 1:
        return _batch_grid_reml_ncvt1(lambdas, eigenvalues, Uab_batch, Iab_batch)

    # General path: use split invariant/varying optimization
    return _batch_grid_reml_general(n_cvt, lambdas, eigenvalues, Uab_batch, Iab_batch)


def _batch_grid_reml_ncvt1(
    lambdas: Float[Array, " g"],
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n 6"],
    Iab_batch: Float[Array, "p 3 6"],
) -> Float[Array, "g p"]:
    """n_cvt=1 fast path for grid REML: SNPs-outer, lambdas-inner vmap.

    Precomputes per-lambda invariant sums (s_ww, s_wy, s_yy, logdet_h)
    once for all SNPs, then evaluates each SNP across all lambdas keeping
    the 3 SNP-varying columns resident in cache.

    Args:
        lambdas: Grid of lambda values (n_grid,).
        eigenvalues: Eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, 6).
        Iab_batch: Identity-weighted Pab (n_snps, 3, 6).

    Returns:
        Log-likelihoods (n_grid, n_snps).
    """
    n_samples = eigenvalues.shape[0]
    df = n_samples - 2  # n_cvt=1 → df = n - 1 - 1

    # Extract invariant columns from Uab (identical across all SNPs for n_cvt=1)
    # Uab column layout for n_cvt=1: [ww, wx, wy, xx, xy, yy]
    #   col 0: ww, col 2: wy, col 5: yy
    ww = Uab_batch[0, :, 0]  # (n_samples,) — same for all SNPs
    wy = Uab_batch[0, :, 2]
    yy = Uab_batch[0, :, 5]

    # Extract SNP-varying columns: wx=col1, xx=col3, xy=col4
    # uab_varying: (n_snps, n_samples, 3)
    uab_varying = Uab_batch[:, :, [1, 3, 4]]

    # Precompute per-SNP logdet_iab (lambda-independent, but SNP-varying).
    # For n_cvt=1: logdet_diag_indices = [(0,0), (1,3)]
    #   Iab[0, 0] = sum(ww)        — SNP-invariant (ww is shared)
    #   Iab[1, 3] = sum(xx) - sum(wx)^2/sum(ww) — SNP-varying (xx, wx differ per SNP)
    # We use Iab_batch[:, 0, 0] (same for all SNPs) and Iab_batch[:, 1, 3] (per-SNP).
    iab_s_ww_all = Iab_batch[
        :, 0, 0
    ]  # (n_snps,) — actually all equal, but extract per-SNP
    iab_p1_xx_all = Iab_batch[:, 1, 3]  # (n_snps,) — SNP-varying
    log_iab_s_ww_all = jnp.where(iab_s_ww_all > 0, jnp.log(iab_s_ww_all), 0.0)
    log_iab_p1_xx_all = jnp.where(iab_p1_xx_all > 0, jnp.log(iab_p1_xx_all), 0.0)
    logdet_iab_all = log_iab_s_ww_all + log_iab_p1_xx_all  # (n_snps,)

    # Precompute per-lambda invariant sums: vmap over lambdas
    # Each lambda produces: Hi_eval, s_ww, s_wy, log_s_ww, pab1_5, logdet_h
    def compute_lambda_invariants(lam):
        v_temp = lam * eigenvalues + 1.0
        Hi_eval = 1.0 / v_temp
        logdet_h = jnp.sum(jnp.log(jnp.abs(v_temp)))
        s_ww = jnp.dot(Hi_eval, ww)
        s_wy = jnp.dot(Hi_eval, wy)
        s_yy = jnp.dot(Hi_eval, yy)
        log_s_ww = jnp.where(s_ww > 0, jnp.log(s_ww), 0.0)
        inv_ww = jnp.where(s_ww != 0, 1.0 / s_ww, 0.0)
        pab1_5 = s_yy - s_wy * s_wy * inv_ww  # completely SNP-invariant per lambda
        return Hi_eval, s_ww, s_wy, log_s_ww, pab1_5, logdet_h

    # Vectorize over lambda grid: (n_grid, ...)
    (
        Hi_eval_grid,  # (n_grid, n_samples)
        s_ww_grid,  # (n_grid,)
        s_wy_grid,  # (n_grid,)
        log_s_ww_grid,  # (n_grid,)
        pab1_5_grid,  # (n_grid,)
        logdet_h_grid,  # (n_grid,)
    ) = vmap(compute_lambda_invariants)(lambdas)

    # Inner function: for one SNP, evaluate REML across all lambdas (lambdas-inner)
    # uab_snp: (n_samples, 3) — [wx, xx, xy]
    # logdet_iab_snp: scalar, pre-computed for this SNP
    def reml_for_snp(uab_snp, logdet_iab_snp):
        def reml_at_lambda(Hi_eval, s_ww, s_wy, log_s_ww, pab1_5, logdet_h):
            return _reml_ncvt1_split(
                Hi_eval,
                uab_snp,
                s_ww,
                s_wy,
                log_s_ww,
                pab1_5,
                logdet_iab_snp,
                logdet_h,
                df,
            )

        return vmap(reml_at_lambda)(
            Hi_eval_grid,
            s_ww_grid,
            s_wy_grid,
            log_s_ww_grid,
            pab1_5_grid,
            logdet_h_grid,
        )  # (n_grid,)

    # vmap over SNPs (outer): (n_snps, n_grid)
    all_logls = vmap(reml_for_snp)(uab_varying, logdet_iab_all)
    return all_logls.T  # (n_grid, n_snps)


@partial(jit, static_argnums=(0,))
def calc_wald_stats_jax(
    n_cvt: int,
    lambda_val: Float[Array, ""],
    eigenvalues: Float[Array, " n"],
    Uab: Float[Array, "n ni"],
    n_samples: int,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Compute Wald test statistics using JAX for arbitrary n_cvt.

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        lambda_val: Optimized variance ratio (scalar).
        eigenvalues: Eigenvalues (n_samples,).
        Uab: Matrix products (n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Tuple of (beta, se, p_wald) - all scalars.
    """
    df = n_samples - n_cvt - 1
    table = build_index_table(n_cvt)
    idx_xx = table["idx_xx"]
    idx_xy = table["idx_xy"]
    idx_yy = table["idx_yy"]

    # Compute Pab
    Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
    Pab = calc_pab_jax(n_cvt, Hi_eval, Uab)

    # Extract values using precomputed indices
    P_XX = Pab[n_cvt, idx_xx]  # After projecting out covariates
    P_XY = Pab[n_cvt, idx_xy]
    P_YY = Pab[n_cvt, idx_yy]
    Px_YY = Pab[n_cvt + 1, idx_yy]  # After projecting out covariates AND genotype

    # Clamp Px_YY like NumPy path (GEMMA lmm.cpp:854)
    # Only clamp if >= 0 and < _P_YY_MIN; leave negative values to produce NaN
    Px_YY = jnp.where((Px_YY >= 0.0) & (Px_YY < _P_YY_MIN), _P_YY_MIN, Px_YY)

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


def batch_calc_wald_stats(
    n_cvt: int,
    lambdas: Float[Array, " p"],
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n ni"],
    n_samples: int,
) -> tuple[Float[Array, " p"], Float[Array, " p"], Float[Array, " p"]]:
    """Vectorized Wald test statistics across SNPs.

    Since n_cvt is static, it cannot be vmapped over. Instead we create
    a lambda that closes over n_cvt and vmap over the remaining args.

    Args:
        n_cvt: Number of covariates (static).
        lambdas: Optimized lambda per SNP (n_snps,).
        eigenvalues: Shared eigenvalues (n_samples,).
        Uab_batch: Uab matrices per SNP (n_snps, n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Tuple of (betas, ses, p_walds) - each (n_snps,).
    """
    return vmap(
        lambda lam, uab: calc_wald_stats_jax(n_cvt, lam, eigenvalues, uab, n_samples),
        in_axes=(0, 0),
    )(lambdas, Uab_batch)


@partial(jit, static_argnums=(0,))
def calc_score_stats_jax(
    n_cvt: int,
    Pab: Float[Array, "nr ni"],
    n_samples: int,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Compute Score test statistics from Pab using JAX.

    Follows stats.py:calc_score_test EXACTLY. Key difference from Wald:
    P_xx, P_xy, P_yy are extracted at Pab level n_cvt (after projecting
    out covariates only), NOT n_cvt+1 (after projecting out genotype).

    The Score F-statistic uses n_samples (not df) in the numerator:
        F = n_samples * P_xy^2 / (P_yy * P_xx)

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        Pab: Pab matrix (n_cvt+2, n_index).
        n_samples: Number of samples.

    Returns:
        Tuple of (beta, se, p_score) - all scalars.
    """
    df = n_samples - n_cvt - 1
    table = build_index_table(n_cvt)
    idx_xx = table["idx_xx"]
    idx_xy = table["idx_xy"]
    idx_yy = table["idx_yy"]

    # Score test: extract at level n_cvt (covariates only, NOT genotype)
    P_yy = Pab[n_cvt, idx_yy]
    P_xx = Pab[n_cvt, idx_xx]
    P_xy = Pab[n_cvt, idx_xy]

    # Px_yy for beta/se computation (after projecting out covariates AND genotype)
    Px_yy = Pab[n_cvt + 1, idx_yy]
    Px_yy = jnp.where((Px_yy >= 0.0) & (Px_yy < _P_YY_MIN), _P_YY_MIN, Px_yy)

    # Guard degenerate SNPs
    is_valid = P_xx > 0

    # Beta and SE (informational only for Score test)
    beta = jnp.where(is_valid, P_xy / jnp.where(is_valid, P_xx, 1.0), jnp.nan)
    tau = df / Px_yy
    variance_beta = jnp.where(
        is_valid, 1.0 / (tau * jnp.where(is_valid, P_xx, 1.0)), jnp.nan
    )
    variance_safe = jnp.where(
        jnp.abs(variance_beta) < 0.001,
        jnp.abs(variance_beta),
        variance_beta,
    )
    se = jnp.where(is_valid, jnp.sqrt(variance_safe), jnp.nan)

    # Score F-statistic: F = n * P_xy^2 / (P_yy * P_xx)
    # NOTE: uses n_samples (not df), and P_yy * P_xx (not Px_yy)
    f_stat = n_samples * (P_xy * P_xy) / (P_yy * jnp.where(is_valid, P_xx, 1.0))

    # p_score via betainc (F-distribution survival function)
    z = df / (df + jnp.maximum(f_stat, 1e-10))
    z = jnp.clip(z, 0.0, 1.0)
    p_score = jax.scipy.special.betainc(df / 2.0, 0.5, z)

    # Guard f_stat <= 0 and invalid SNPs
    p_score = jnp.where(f_stat <= 0, 1.0, p_score)
    p_score = jnp.where(is_valid, p_score, jnp.nan)

    return beta, se, p_score


def batch_calc_score_stats(
    n_cvt: int,
    Hi_eval_null: Float[Array, " n"],
    Uab_batch: Float[Array, "p n ni"],
    n_samples: int,
) -> tuple[Float[Array, " p"], Float[Array, " p"], Float[Array, " p"]]:
    """Batch Score test: compute Pab with fixed null Hi_eval, extract stats.

    Score test uses a single null-model lambda for all SNPs, so Hi_eval
    is constant across the batch. This is cheaper than Wald because no
    per-SNP lambda optimization is needed.

    Args:
        n_cvt: Number of covariates (static).
        Hi_eval_null: 1 / (lambda_null * eigenvalues + 1) vector (n_samples,).
        Uab_batch: Uab matrices per SNP (n_snps, n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Tuple of (betas, ses, p_scores) - each (n_snps,).
    """
    Pab_batch = vmap(lambda Uab: calc_pab_jax(n_cvt, Hi_eval_null, Uab))(Uab_batch)
    return vmap(lambda Pab: calc_score_stats_jax(n_cvt, Pab, n_samples))(Pab_batch)


def _batch_grid_mle(
    n_cvt: int,
    lambdas: Float[Array, " g"],
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n ni"],
) -> Float[Array, "g p"]:
    """Compute MLE log-likelihood at all grid points for all SNPs.

    MLE counterpart of _batch_grid_reml_with_iab. Key difference: no Iab
    argument needed because MLE has no logdet_hiw term.

    Args:
        n_cvt: Number of covariates (passed through to mle_log_likelihood_jax).
        lambdas: Grid of lambda values (n_grid,).
        eigenvalues: Eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Log-likelihoods (n_grid, n_snps).
    """

    def mle_for_lambda(lam):
        return vmap(lambda Uab: mle_log_likelihood_jax(n_cvt, lam, eigenvalues, Uab))(
            Uab_batch
        )

    return vmap(mle_for_lambda)(lambdas)


@partial(jit, static_argnums=(0, 3, 4, 5, 6))
def golden_section_optimize_lambda_mle(
    n_cvt: int,
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n ni"],
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[Float[Array, " p"], Float[Array, " p"]]:
    """Optimize MLE lambda using grid search + golden section refinement.

    MLE counterpart of golden_section_optimize_lambda (REML). Used by
    LRT (-lmm 2) which requires per-SNP MLE lambda optimization.

    Key differences from the REML optimizer:
    - Uses mle_log_likelihood_jax instead of _reml_with_precomputed_iab
    - No Iab_batch argument (MLE has no logdet_hiw term)
    - Returns MLE log-likelihoods (not REML)

    The grid search covers boundaries (l_min and l_max are linspace
    endpoints), handling MLE boundary optima naturally.

    Args:
        n_cvt: Number of covariates (static, triggers recompilation).
        eigenvalues: Eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        l_min, l_max: Lambda bounds.
        n_grid: Coarse grid points.
        n_iter: Golden section iterations (20 gives ~1e-5 tolerance).

    Returns:
        (optimal_lambdas, optimal_logls) - MLE log-likelihoods for each SNP.
    """
    # Stage 1: Coarse grid search on log scale
    log_l_min = jnp.log(l_min)
    log_l_max = jnp.log(l_max)
    log_lambdas = jnp.linspace(log_l_min, log_l_max, n_grid)
    lambdas = jnp.exp(log_lambdas)

    all_logls = _batch_grid_mle(n_cvt, lambdas, eigenvalues, Uab_batch)

    # MLE batch evaluator (no Iab needed)
    def compute_mle_batch(log_lams):
        lams = jnp.exp(log_lams)
        return vmap(
            lambda lam, Uab: mle_log_likelihood_jax(n_cvt, lam, eigenvalues, Uab),
            in_axes=(0, 0),
        )(lams, Uab_batch)

    # Stage 2: Golden section refinement
    return _golden_section_refine(
        compute_mle_batch, all_logls, log_lambdas, n_grid, n_iter
    )
