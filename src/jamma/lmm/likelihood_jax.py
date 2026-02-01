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
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

# Pre-compute index mappings for n_cvt=1 (most common case)
# For n_cvt=1: indices are (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
# Mapping: 0=WW, 1=WX, 2=WY, 3=XX, 4=XY, 5=YY
_IDX_WW = 0  # covariate-covariate
_IDX_WX = 1  # covariate-genotype
_IDX_WY = 2  # covariate-phenotype
_IDX_XX = 3  # genotype-genotype
_IDX_XY = 4  # genotype-phenotype
_IDX_YY = 5  # phenotype-phenotype


@jit
def compute_uab_jax(
    UtW: jnp.ndarray, Uty: jnp.ndarray, Utx: jnp.ndarray
) -> jnp.ndarray:
    """Compute Uab matrix for a single SNP using JAX.

    Optimized for n_cvt=1 (intercept only), which is the most common case.

    Args:
        UtW: Rotated covariates (n_samples, 1) - typically just intercept
        Uty: Rotated phenotype (n_samples,)
        Utx: Rotated genotype (n_samples,)

    Returns:
        Uab matrix (n_samples, 6) for n_cvt=1
    """
    n = Uty.shape[0]
    w = UtW[:, 0]  # Intercept column

    # Build Uab with fixed index layout for n_cvt=1
    Uab = jnp.zeros((n, 6), dtype=jnp.float64)
    Uab = Uab.at[:, _IDX_WW].set(w * w)  # (1,1): W*W
    Uab = Uab.at[:, _IDX_WX].set(w * Utx)  # (1,2): W*X
    Uab = Uab.at[:, _IDX_WY].set(w * Uty)  # (1,3): W*Y
    Uab = Uab.at[:, _IDX_XX].set(Utx * Utx)  # (2,2): X*X
    Uab = Uab.at[:, _IDX_XY].set(Utx * Uty)  # (2,3): X*Y
    Uab = Uab.at[:, _IDX_YY].set(Uty * Uty)  # (3,3): Y*Y

    return Uab


@jit
def calc_pab_jax(Hi_eval: jnp.ndarray, Uab: jnp.ndarray) -> jnp.ndarray:
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
def reml_log_likelihood_jax(
    lambda_val: float,
    eigenvalues: jnp.ndarray,
    Uab: jnp.ndarray,
) -> float:
    """Compute REML log-likelihood using JAX (optimized for n_cvt=1).

    JIT-compiled version for efficient repeated evaluation during optimization.

    Args:
        lambda_val: Variance component ratio
        eigenvalues: Eigenvalues of kinship matrix (n_samples,)
        Uab: Matrix products (n_samples, 6)

    Returns:
        Log-likelihood value
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

    # Compute Iab (identity weighting) for logdet correction
    ones = jnp.ones(n, dtype=jnp.float64)
    Iab = calc_pab_jax(ones, Uab)

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
    UtW: jnp.ndarray,
    Uty: jnp.ndarray,
    UtG: jnp.ndarray,
) -> jnp.ndarray:
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


@partial(jit, static_argnums=(2, 3, 4, 5))
def golden_section_optimize_lambda(
    eigenvalues: jnp.ndarray,
    Uab_batch: jnp.ndarray,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Optimize lambda using grid search + golden section refinement.

    This hybrid approach:
    1. Grid search to find approximate region (vectorized across SNPs)
    2. Golden section for precise convergence (vectorized across SNPs)

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

    Args:
        eigenvalues: Eigenvalues (n_samples,)
        Uab_batch: Uab matrices (n_snps, n_samples, 6)
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
    all_logls = _batch_grid_reml(lambdas, eigenvalues, Uab_batch)

    # Find best index per SNP
    best_idx = jnp.argmax(all_logls, axis=0)

    # Set up bounds for golden section (one grid cell on each side)
    idx_low = jnp.maximum(best_idx - 1, 0)
    idx_high = jnp.minimum(best_idx + 1, n_grid - 1)

    # Initial bounds (log scale)
    a = log_lambdas[idx_low]
    b = log_lambdas[idx_high]

    # Helper to compute REML at log-scale lambdas (batch) - stays on device
    def compute_reml_batch(log_lams):
        lams = jnp.exp(log_lams)
        return vmap(
            lambda lam, Uab: reml_log_likelihood_jax(lam, eigenvalues, Uab),
            in_axes=(0, 0),
        )(lams, Uab_batch)

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
    lambdas: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    Uab_batch: jnp.ndarray,
) -> jnp.ndarray:
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


@jit
def calc_wald_stats_jax(
    lambda_val: float,
    eigenvalues: jnp.ndarray,
    Uab: jnp.ndarray,
    n_samples: int,
) -> tuple[float, float, float]:
    """Compute Wald test statistics using JAX (n_cvt=1).

    Args:
        lambda_val: Optimized variance ratio
        eigenvalues: Eigenvalues (n_samples,)
        Uab: Matrix products (n_samples, 6)
        n_samples: Number of samples

    Returns:
        Tuple of (beta, se, p_wald)
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

    # Effect size and standard error
    # GEMMA safe_sqrt: if |d| < 0.001, use abs(d) to tolerate small negatives
    # This matches GEMMA mathfunc.cpp:122-131
    beta = P_XY / P_XX
    tau = df / Px_YY
    variance_beta = 1.0 / (tau * P_XX)
    # Apply safe_sqrt: for small negatives (|v| < 0.001), use abs; otherwise use as-is
    # In JAX, we handle this with jnp.where for the small negative case
    variance_safe = jnp.where(
        jnp.abs(variance_beta) < 0.001,
        jnp.abs(variance_beta),
        variance_beta,
    )
    # For large negatives, sqrt will produce NaN (matching GEMMA behavior)
    se = jnp.sqrt(variance_safe)

    # F-statistic and p-value
    f_stat = (P_YY - Px_YY) * tau

    # Guard: if f_stat <= 0, p-value = 1.0 (no evidence against null)
    # Clamp z to [0, 1] to ensure betainc is well-defined
    z = df / (df + jnp.maximum(f_stat, 1e-10))
    z = jnp.clip(z, 0.0, 1.0)
    p_wald = jax.scipy.special.betainc(df / 2.0, 0.5, z)
    # If f_stat was non-positive, return p=1.0
    p_wald = jnp.where(f_stat <= 0, 1.0, p_wald)

    return beta, se, p_wald


# Vectorized version for batch processing
batch_calc_wald_stats = vmap(
    calc_wald_stats_jax,
    in_axes=(0, None, 0, None),  # lambda per SNP, shared eigenvalues, Uab per SNP
    out_axes=(0, 0, 0),  # beta, se, p per SNP
)
