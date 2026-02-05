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

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, vmap

from jamma.lmm.likelihood import get_ab_index

if TYPE_CHECKING:
    from jaxtyping import Array, Float


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
    P_yy = Pab[nc_total, idx_yy]
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
    # Use GEMMA's conditional clamping: only clamp if P_yy >= 0 and P_yy < 1e-8
    # Matches lmm.cpp:854: if (P_yy >= 0.0 && P_yy < P_YY_MIN) P_yy = P_YY_MIN
    P_yy = Pab[nc_total, idx_yy]
    P_yy = jnp.where((P_yy >= 0.0) & (P_yy < 1e-8), 1e-8, P_yy)

    # REML log-likelihood
    c = 0.5 * df * (jnp.log(df) - jnp.log(2 * jnp.pi) - 1.0)
    f = c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * jnp.log(P_yy)

    return f


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


@partial(jit, static_argnums=(0, 4, 5, 6, 7), donate_argnums=(2,))
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
    the optimum to +/-1 grid cell, 20 iterations reduce uncertainty by
    0.618^20 ~ 6.6e-5, giving relative tolerance < 1e-5 for typical lambda.

    Performance:
    - Grid search: O(n_grid) REML evaluations (shared across SNPs)
    - Golden section: O(n_iter) REML evaluations per SNP (vectorized)
    - Total: ~70 REML evaluations vs ~50 for Brent (similar cost)
    - All computations stay on device (no host/device sync in loops)
    - Uses lax.fori_loop for golden section to avoid Python loop retracing
    - donate_argnums=(2,) hints XLA to reuse Uab_batch memory (when possible)

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
    phi = 0.6180339887498949  # Golden ratio - 1

    # Stage 1: Coarse grid search on log scale (fully on device)
    log_l_min = jnp.log(l_min)
    log_l_max = jnp.log(l_max)
    log_lambdas = jnp.linspace(log_l_min, log_l_max, n_grid)
    lambdas = jnp.exp(log_lambdas)

    # Evaluate all SNPs at all grid points using vmap (stays on device)
    all_logls = _batch_grid_reml_with_iab(
        n_cvt, lambdas, eigenvalues, Uab_batch, Iab_batch
    )

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
                n_cvt, lam, eigenvalues, Uab, Iab
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


def _batch_grid_reml_with_iab(
    n_cvt: int,
    lambdas: Float[Array, " g"],
    eigenvalues: Float[Array, " n"],
    Uab_batch: Float[Array, "p n ni"],
    Iab_batch: Float[Array, "p nr ni"],
) -> Float[Array, "g p"]:
    """Compute REML at all grid points using precomputed Iab (optimized).

    Uses vmap over lambda values to avoid Python loops and host/device sync.
    Iab is precomputed once per chunk, avoiding redundant identity-weighted
    projection computations (~n_grid fewer calc_pab_jax calls per SNP).

    Args:
        n_cvt: Number of covariates (passed through to _reml_with_precomputed_iab).
        lambdas: Grid of lambda values (n_grid,).
        eigenvalues: Eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index).

    Returns:
        Log-likelihoods (n_grid, n_snps).
    """

    # vmap over lambda values, then vmap over SNPs
    def reml_for_lambda(lam):
        return vmap(
            lambda Uab, Iab: _reml_with_precomputed_iab(
                n_cvt, lam, eigenvalues, Uab, Iab
            ),
            in_axes=(0, 0),
        )(Uab_batch, Iab_batch)

    return vmap(reml_for_lambda)(lambdas)


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
    Px_yy = jnp.where((Px_yy >= 0.0) & (Px_yy < 1e-8), 1e-8, Px_yy)

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
