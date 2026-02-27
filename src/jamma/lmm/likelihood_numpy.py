"""Pure-NumPy batch LMM likelihood computation.

Replaces JAX vmap/lax.fori_loop with numpy broadcasting and Python loops.
All functions process a chunk (batch) of SNPs at once without JAX.

Design:
- batch_compute_uab_numpy: vectorized Uab for n_snps SNPs at once
- batch_compute_pab_numpy / _batch_compute_pab_varying_numpy: Pab for a batch
- batch_compute_iab_numpy: Iab (identity-weighted Pab)
- golden_section_optimize_lambda_numpy / _mle: batch lambda optimization
- batch_calc_wald_stats_numpy / score / lrt: batch test statistics

No JAX imports anywhere in this module. Compatible with JAX-free environments.

Reference: likelihood_jax.py (ported to NumPy in this module).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from jamma.lmm.likelihood import _P_YY_MIN, build_index_table
from jamma.lmm.special import betainc, chi2_sf

# Module-level vectorized wrappers (created once, reused across calls)
_betainc_vec = np.vectorize(betainc)
_chi2_sf_vec = np.vectorize(chi2_sf)


# ---------------------------------------------------------------------------
# Uab batch computation
# ---------------------------------------------------------------------------


def batch_compute_uab_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    UtG: np.ndarray,
) -> np.ndarray:
    """Compute Uab matrices for all SNPs in a chunk.

    Direct port of likelihood_jax.py::batch_compute_uab with jnp replaced
    by np. Shape: (n_snps, n_samples, n_index).

    Args:
        n_cvt: Number of covariates. If 1, uses explicit fast-path broadcasting.
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        UtG: Rotated genotypes for all SNPs in this chunk (n_samples, n_snps).

    Returns:
        Uab matrices (n_snps, n_samples, n_index).
    """
    if n_cvt == 1:
        return _batch_compute_uab_ncvt1_numpy(UtW, Uty, UtG)
    return _batch_compute_uab_general_numpy(n_cvt, UtW, Uty, UtG)


def _batch_compute_uab_ncvt1_numpy(
    UtW: np.ndarray,
    Uty: np.ndarray,
    UtG: np.ndarray,
) -> np.ndarray:
    """Fast path batch Uab for n_cvt=1 (intercept only).

    Indices for n_cvt=1:
      0: ww  (1,1), 1: wx  (1,2), 2: wy  (1,3)
      3: xx  (2,2), 4: xy  (2,3), 5: yy  (3,3)

    Args:
        UtW: Rotated covariates (n_samples, 1).
        Uty: Rotated phenotype (n_samples,).
        UtG: Rotated genotypes (n_samples, n_snps).

    Returns:
        Uab batch (n_snps, n_samples, 6).
    """
    n_samples, n_snps = UtG.shape
    w = UtW[:, 0]  # (n_samples,)
    UtG_T = UtG.T  # (n_snps, n_samples)

    # Broadcast scalar fields across all SNPs
    ww = np.broadcast_to((w * w)[None, :], (n_snps, n_samples))
    wy = np.broadcast_to((w * Uty)[None, :], (n_snps, n_samples))
    yy = np.broadcast_to((Uty * Uty)[None, :], (n_snps, n_samples))

    # SNP-varying fields
    wx = w[None, :] * UtG_T  # (n_snps, n_samples)
    xx = UtG_T * UtG_T  # (n_snps, n_samples)
    xy = UtG_T * Uty[None, :]  # (n_snps, n_samples)

    return np.stack([ww, wx, wy, xx, xy, yy], axis=-1)


def _batch_compute_uab_general_numpy(
    n_cvt: int,
    UtW: np.ndarray,
    Uty: np.ndarray,
    UtG: np.ndarray,
) -> np.ndarray:
    """General batch Uab for arbitrary n_cvt.

    Loops over SNPs (rare case — n_cvt=1 covers >99% of GWAS runs).

    Args:
        n_cvt: Number of covariates (> 1).
        UtW: Rotated covariates (n_samples, n_cvt).
        Uty: Rotated phenotype (n_samples,).
        UtG: Rotated genotypes (n_samples, n_snps).

    Returns:
        Uab batch (n_snps, n_samples, n_index).
    """
    table = build_index_table(n_cvt)
    n_samples, n_snps = UtG.shape
    n_index = table["n_index"]
    Uab_batch = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)

    # Precompute the covariate/phenotype part (shared across all SNPs)
    # vectors_no_x: (n_samples, n_cvt+2) with X column zeroed out
    # (X index is n_cvt, 0-based)
    vectors_base = np.column_stack(
        [UtW, np.zeros(n_samples), Uty]
    )  # (n_samples, n_cvt+2)

    for snp_idx in range(n_snps):
        vectors = vectors_base.copy()
        vectors[:, n_cvt] = UtG[:, snp_idx]  # fill genotype column
        for a_col, b_col, idx in table["uab_pairs"]:
            Uab_batch[snp_idx, :, idx] = vectors[:, a_col] * vectors[:, b_col]

    return Uab_batch


# ---------------------------------------------------------------------------
# Pab batch computation
# ---------------------------------------------------------------------------


def batch_compute_pab_numpy(
    n_cvt: int,
    Hi_eval: np.ndarray,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Compute Pab for all SNPs using a shared Hi_eval vector.

    Used for Score test (fixed null lambda for all SNPs) and Iab computation.

    Row-0: tensordot(Hi_eval[n], Uab_batch[p,n,i]) -> (p, i) via BLAS.
    Rows 1..n_cvt+1: recursive projection vectorized over all SNPs.

    Args:
        n_cvt: Number of covariates.
        Hi_eval: Shared 1/(lambda*eval+1) vector (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Pab batch (n_snps, n_cvt+2, n_index).
    """
    table = build_index_table(n_cvt)
    n_snps, n_samples, n_index = Uab_batch.shape
    Pab_batch = np.zeros((n_snps, n_cvt + 2, n_index), dtype=np.float64)

    # Row 0: batched weighted dot product
    # tensordot contracts Hi_eval (axis 0) with Uab_batch (axis 1)
    # Result shape: (n_snps, n_index)
    Pab_batch[:, 0, :] = np.tensordot(Hi_eval, Uab_batch, axes=([0], [1]))

    # Rows 1..n_cvt+1: recursive projection (same for all SNPs)
    _fill_pab_recursion(Pab_batch, table, n_cvt)

    return Pab_batch


def _batch_compute_pab_varying_numpy(
    n_cvt: int,
    Hi_eval_batch: np.ndarray,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Compute Pab for all SNPs with per-SNP Hi_eval vectors.

    Used during lambda optimization (each SNP has its own lambda -> own Hi_eval).

    Row-0: einsum('pn,pni->pi', Hi_eval_batch, Uab_batch).
    Rows 1..n_cvt+1: same recursive projection as batch_compute_pab_numpy.

    Args:
        n_cvt: Number of covariates.
        Hi_eval_batch: Per-SNP Hi_eval (n_snps, n_samples).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Pab batch (n_snps, n_cvt+2, n_index).
    """
    table = build_index_table(n_cvt)
    n_snps, n_samples, n_index = Uab_batch.shape
    Pab_batch = np.zeros((n_snps, n_cvt + 2, n_index), dtype=np.float64)

    # Row 0: per-SNP einsum
    # Hi_eval_batch: (n_snps, n_samples)
    # Uab_batch: (n_snps, n_samples, n_index)
    # -> (n_snps, n_index)
    Pab_batch[:, 0, :] = np.einsum("pn,pni->pi", Hi_eval_batch, Uab_batch)

    # Rows 1..n_cvt+1: recursive projection
    _fill_pab_recursion(Pab_batch, table, n_cvt)

    return Pab_batch


def _fill_pab_recursion(
    Pab_batch: np.ndarray,
    table: dict,
    n_cvt: int,
) -> None:
    """Fill rows 1..n_cvt+1 of Pab_batch using GEMMA's recursive formula.

    Mutates Pab_batch in-place. Row 0 must already be filled.
    All SNPs are processed simultaneously (vectorized over axis 0).

    Args:
        Pab_batch: (n_snps, n_cvt+2, n_index) array, row 0 pre-filled.
        table: Index table from build_index_table.
        n_cvt: Number of covariates.
    """
    n_degenerate = 0
    for p in range(1, n_cvt + 2):
        for _a, _b, index_ab, index_aw, index_bw, index_ww in table["pab_recursion"][p]:
            ps_ww = Pab_batch[:, p - 1, index_ww]  # (n_snps,)
            # Guard: ps_ww=0 means degenerate covariate projection. Use 0
            # to prevent NaN/Inf propagation to valid SNPs in the same batch.
            # Degenerate SNPs are caught downstream by P_XX > 0 checks.
            n_degenerate += int(np.sum(ps_ww == 0))
            with np.errstate(divide="ignore"):
                safe_inv = np.where(ps_ww != 0, 1.0 / ps_ww, 0.0)
            Pab_batch[:, p, index_ab] = (
                Pab_batch[:, p - 1, index_ab]
                - Pab_batch[:, p - 1, index_aw]
                * Pab_batch[:, p - 1, index_bw]
                * safe_inv
            )
    if n_degenerate > 0:
        logger.debug(
            f"Pab recursion: {n_degenerate} degenerate ps_ww=0 entries guarded"
        )


def batch_compute_iab_numpy(
    n_cvt: int,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Compute identity-weighted Pab (Iab) for all SNPs.

    Iab = Pab with Hi_eval = ones. Used for logdet_hiw term in REML.
    Lambda-independent: precompute once per chunk.

    Args:
        n_cvt: Number of covariates.
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Iab batch (n_snps, n_cvt+2, n_index).
    """
    n_samples = Uab_batch.shape[1]
    ones = np.ones(n_samples, dtype=np.float64)
    return batch_compute_pab_numpy(n_cvt, ones, Uab_batch)


# ---------------------------------------------------------------------------
# Batch REML / MLE log-likelihood evaluation
# ---------------------------------------------------------------------------


def _batch_reml_at_lambda_numpy(
    n_cvt: int,
    lambda_vals: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Iab_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate REML log-likelihood for each SNP at its own lambda value.

    Args:
        n_cvt: Number of covariates.
        lambda_vals: Per-SNP lambda values (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index).

    Returns:
        REML log-likelihoods (n_snps,).
    """
    table = build_index_table(n_cvt)
    n_snps = Uab_batch.shape[0]
    n = eigenvalues.shape[0]
    nc_total = n_cvt + 1
    df = n - n_cvt - 1

    # Per-SNP H-inv weights: (n_snps, n_samples)
    v_temp = lambda_vals[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_batch = 1.0 / v_temp  # (n_snps, n_samples)

    # Log determinant of H per SNP: (n_snps,)
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)

    # Pab with per-SNP Hi_eval
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, Hi_eval_batch, Uab_batch)

    # logdet_hiw per SNP: sum over diagonal indices
    # Guard: non-positive diagonal Pab/Iab entries (degenerate SNPs) use 0.0
    # instead of log to prevent NaN/Inf from corrupting the batch. Degenerate
    # SNPs are caught downstream by the P_yy < 0 → NaN guard.
    logdet_hiw = np.zeros(n_snps, dtype=np.float64)
    for row, col in table["logdet_diag_indices"]:
        d_pab = Pab_batch[:, row, col]  # (n_snps,)
        d_iab = Iab_batch[:, row, col]  # (n_snps,)
        with np.errstate(divide="ignore", invalid="ignore"):
            logdet_hiw += np.where(d_pab > 0, np.log(d_pab), 0.0)
            logdet_hiw -= np.where(d_iab > 0, np.log(d_iab), 0.0)

    # P_yy per SNP with guards
    P_yy = Pab_batch[:, nc_total, table["idx_yy"]]  # (n_snps,)
    P_yy = np.where(P_yy < 0.0, np.nan, P_yy)
    P_yy = np.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)

    # REML log-likelihood per SNP
    c = 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)
    return c - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)


def _batch_mle_at_lambda_numpy(
    n_cvt: int,
    lambda_vals: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate MLE log-likelihood for each SNP at its own lambda value.

    Key differences from REML: no logdet_hiw term, uses n instead of df.

    Args:
        n_cvt: Number of covariates.
        lambda_vals: Per-SNP lambda values (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        MLE log-likelihoods (n_snps,).
    """
    table = build_index_table(n_cvt)
    n = eigenvalues.shape[0]
    nc_total = n_cvt + 1

    # Per-SNP H-inv weights: (n_snps, n_samples)
    v_temp = lambda_vals[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_batch = 1.0 / v_temp

    # Log determinant of H per SNP: (n_snps,)
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)

    # Pab with per-SNP Hi_eval
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, Hi_eval_batch, Uab_batch)

    # P_yy per SNP with guards
    P_yy = Pab_batch[:, nc_total, table["idx_yy"]]
    P_yy = np.where(P_yy < 0.0, np.nan, P_yy)
    P_yy = np.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)

    # MLE log-likelihood per SNP (no logdet_hiw, uses n not df)
    c = 0.5 * n * (np.log(n) - np.log(2.0 * np.pi) - 1.0)
    return c - 0.5 * logdet_h - 0.5 * n * np.log(P_yy)


# ---------------------------------------------------------------------------
# Grid evaluations
# ---------------------------------------------------------------------------


def _batch_grid_reml_numpy(
    n_cvt: int,
    lambdas_grid: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Iab_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate REML at all grid lambda values for all SNPs.

    Args:
        n_cvt: Number of covariates.
        lambdas_grid: Grid of lambda values (n_grid,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed Iab (n_snps, n_cvt+2, n_index).

    Returns:
        Log-likelihoods (n_grid, n_snps).
    """
    n_snps = Uab_batch.shape[0]
    n_grid = len(lambdas_grid)
    all_logls = np.zeros((n_grid, n_snps), dtype=np.float64)

    for i, lam in enumerate(lambdas_grid):
        lam_batch = np.full(n_snps, lam, dtype=np.float64)
        all_logls[i, :] = _batch_reml_at_lambda_numpy(
            n_cvt, lam_batch, eigenvalues, Uab_batch, Iab_batch
        )

    return all_logls


def _batch_grid_mle_numpy(
    n_cvt: int,
    lambdas_grid: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> np.ndarray:
    """Evaluate MLE at all grid lambda values for all SNPs.

    Args:
        n_cvt: Number of covariates.
        lambdas_grid: Grid of lambda values (n_grid,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Log-likelihoods (n_grid, n_snps).
    """
    n_snps = Uab_batch.shape[0]
    n_grid = len(lambdas_grid)
    all_logls = np.zeros((n_grid, n_snps), dtype=np.float64)

    for i, lam in enumerate(lambdas_grid):
        lam_batch = np.full(n_snps, lam, dtype=np.float64)
        all_logls[i, :] = _batch_mle_at_lambda_numpy(
            n_cvt, lam_batch, eigenvalues, Uab_batch
        )

    return all_logls


# ---------------------------------------------------------------------------
# Golden section optimizer
# ---------------------------------------------------------------------------


def _batch_golden_section_numpy(
    compute_batch_fn,
    grid_logls: np.ndarray,
    log_lambdas: np.ndarray,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-to-golden-section refinement for lambda optimization.

    Direct translation of likelihood_jax.py::_golden_section_refine
    with jnp -> np and lax.fori_loop -> Python for loop.

    All operations are vectorized over SNPs (axis 0).
    After 20 iterations: 0.618^20 ~ 6.6e-5 relative tolerance.

    Args:
        compute_batch_fn: callable(log_lambdas_per_snp: (n_snps,)) -> (n_snps,).
        grid_logls: Grid log-likelihoods (n_grid, n_snps).
        log_lambdas: Log-scale grid points (n_grid,).
        n_iter: Golden section iterations (should be >= 20).

    Returns:
        (optimal_lambdas, optimal_logls) both shape (n_snps,).
    """
    phi = 0.6180339887498949  # golden ratio - 1

    # Find best grid point per SNP and bracket
    best_idx = np.argmax(grid_logls, axis=0)  # (n_snps,)
    idx_low = np.maximum(best_idx - 1, 0)
    idx_high = np.minimum(best_idx + 1, len(log_lambdas) - 1)

    a = log_lambdas[idx_low]  # (n_snps,)
    b = log_lambdas[idx_high]  # (n_snps,)

    # Initial probe points
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = compute_batch_fn(c)
    fd = compute_batch_fn(d)

    # Golden section iterations (Python for loop, vectorized over SNPs)
    for _ in range(n_iter):
        keep_left = fc > fd  # (n_snps,) boolean

        new_a = np.where(keep_left, a, c)
        new_b = np.where(keep_left, d, b)
        new_c = new_b - phi * (new_b - new_a)
        new_d = new_a + phi * (new_b - new_a)

        new_logl = compute_batch_fn(np.where(keep_left, new_c, new_d))
        new_fc = np.where(keep_left, new_logl, fd)
        new_fd = np.where(keep_left, fc, new_logl)

        a, b, c, d, fc, fd = new_a, new_b, new_c, new_d, new_fc, new_fd

    log_opt = (a + b) / 2.0
    return np.exp(log_opt), compute_batch_fn(log_opt)


def golden_section_optimize_lambda_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Iab_batch: np.ndarray,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimize REML lambda using grid search + golden section refinement.

    Port of likelihood_jax.py::golden_section_optimize_lambda. Replaces
    jnp/vmap with np broadcasting, and lax.fori_loop with Python for loop.

    Enforces minimum of 20 golden section iterations to guarantee
    lambda relative tolerance < 1e-5 (matching GEMMA Brent tolerance).

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index).
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Coarse grid points.
        n_iter: Golden section iterations (minimum 20 enforced).

    Returns:
        (optimal_lambdas, optimal_logls) both shape (n_snps,).
    """
    n_iter = max(n_iter, 20)

    log_l_min = np.log(l_min)
    log_l_max = np.log(l_max)
    log_lambdas = np.linspace(log_l_min, log_l_max, n_grid)
    lambdas_grid = np.exp(log_lambdas)

    # Stage 1: Coarse grid search
    grid_logls = _batch_grid_reml_numpy(
        n_cvt, lambdas_grid, eigenvalues, Uab_batch, Iab_batch
    )

    # REML batch evaluator closure (over precomputed Iab)
    def compute_reml_batch(log_lams: np.ndarray) -> np.ndarray:
        lams = np.exp(log_lams)
        return _batch_reml_at_lambda_numpy(
            n_cvt, lams, eigenvalues, Uab_batch, Iab_batch
        )

    # Stage 2: Golden section refinement
    return _batch_golden_section_numpy(
        compute_reml_batch, grid_logls, log_lambdas, n_iter
    )


def golden_section_optimize_lambda_mle_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimize MLE lambda using grid search + golden section refinement.

    Port of likelihood_jax.py::golden_section_optimize_lambda_mle.
    No Iab argument needed (MLE has no logdet_hiw term).

    Enforces minimum of 20 golden section iterations.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Coarse grid points.
        n_iter: Golden section iterations (minimum 20 enforced).

    Returns:
        (optimal_lambdas, optimal_logls_mle) both shape (n_snps,).
    """
    n_iter = max(n_iter, 20)

    log_l_min = np.log(l_min)
    log_l_max = np.log(l_max)
    log_lambdas = np.linspace(log_l_min, log_l_max, n_grid)
    lambdas_grid = np.exp(log_lambdas)

    # Stage 1: Coarse grid search
    grid_logls = _batch_grid_mle_numpy(n_cvt, lambdas_grid, eigenvalues, Uab_batch)

    # MLE batch evaluator closure
    def compute_mle_batch(log_lams: np.ndarray) -> np.ndarray:
        lams = np.exp(log_lams)
        return _batch_mle_at_lambda_numpy(n_cvt, lams, eigenvalues, Uab_batch)

    # Stage 2: Golden section refinement
    return _batch_golden_section_numpy(
        compute_mle_batch, grid_logls, log_lambdas, n_iter
    )


# ---------------------------------------------------------------------------
# Shared helpers for batch test statistics
# ---------------------------------------------------------------------------


def _beta_se_from_pab(
    P_XX: np.ndarray,
    P_XY: np.ndarray,
    Px_YY: np.ndarray,
    df: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute beta, SE, and validity mask from Pab projections.

    Shared by Wald and Score tests. Handles degenerate SNPs (P_XX <= 0)
    by setting beta/SE to NaN and GEMMA-compatible safe_sqrt for tiny
    variance values.

    Args:
        P_XX: Projected genotype variance per SNP (n_snps,).
        P_XY: Projected genotype-phenotype covariance per SNP (n_snps,).
        Px_YY: Projected phenotype variance at n_cvt+1 level (n_snps,).
        df: Degrees of freedom (n_samples - n_cvt - 1).

    Returns:
        Tuple of (beta, se, is_valid) each shape (n_snps,).
    """
    is_valid = P_XX > 0
    safe_P_XX = np.where(is_valid, P_XX, 1.0)

    beta = np.where(is_valid, P_XY / safe_P_XX, np.nan)
    tau = df / Px_YY
    variance_beta = np.where(is_valid, 1.0 / (tau * safe_P_XX), np.nan)
    # safe_sqrt: for |v| < 0.001, use abs(v) to avoid sqrt of tiny negative
    # values from FP rounding (matches GEMMA lmm.cpp safe_sqrt behaviour)
    variance_safe = np.where(
        np.abs(variance_beta) < 0.001,
        np.abs(variance_beta),
        variance_beta,
    )
    se = np.where(is_valid, np.sqrt(variance_safe), np.nan)

    return beta, se, is_valid


def _f_to_pvalue(f_stat: np.ndarray, df: int, is_valid: np.ndarray) -> np.ndarray:
    """Convert F-statistics to p-values via regularized incomplete beta.

    Shared by Wald and Score test computations. Uses algebraically exact
    complement_z = f_safe / (df + f_safe) to avoid cancellation near z = 1.

    Args:
        f_stat: F-statistics per SNP (n_snps,).
        df: Degrees of freedom (n_samples - n_cvt - 1).
        is_valid: Boolean mask of valid (non-degenerate) SNPs.

    Returns:
        P-values (n_snps,), NaN for invalid SNPs.
    """
    f_safe = np.maximum(f_stat, 1e-10)
    denom = df + f_safe
    z = np.clip(df / denom, 0.0, 1.0)
    complement_z = f_safe / denom  # algebraically exact 1-z, avoids cancellation
    p_val = _betainc_vec(df / 2.0, 0.5, z, complement_z)
    p_val = np.where(f_stat <= 0, 1.0, p_val)
    return np.where(is_valid, p_val, np.nan)


# ---------------------------------------------------------------------------
# Batch test statistics
# ---------------------------------------------------------------------------


def batch_calc_wald_stats_numpy(
    n_cvt: int,
    lambdas: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Wald test statistics for a batch of SNPs.

    Port of likelihood_jax.py::batch_calc_wald_stats. Computes per-SNP
    Hi_eval from optimized lambdas, then calls _batch_compute_pab_varying_numpy.

    p_wald uses Cephes betainc via np.vectorize (more accurate than JAX XLA
    betainc for large a).

    Args:
        n_cvt: Number of covariates.
        lambdas: Optimized REML lambda per SNP (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Tuple of (betas, ses, p_walds) each shape (n_snps,).
    """
    table = build_index_table(n_cvt)
    idx_xx = table["idx_xx"]
    idx_xy = table["idx_xy"]
    idx_yy = table["idx_yy"]
    df = n_samples - n_cvt - 1

    # Per-SNP Hi_eval
    Hi_eval_batch = 1.0 / (lambdas[:, None] * eigenvalues[None, :] + 1.0)

    # Pab batch with per-SNP Hi_eval
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, Hi_eval_batch, Uab_batch)

    P_XX = Pab_batch[:, n_cvt, idx_xx]
    P_XY = Pab_batch[:, n_cvt, idx_xy]
    P_YY = Pab_batch[:, n_cvt, idx_yy]
    Px_YY = Pab_batch[:, n_cvt + 1, idx_yy]

    # Clamp Px_YY
    Px_YY = np.where((Px_YY >= 0.0) & (Px_YY < _P_YY_MIN), _P_YY_MIN, Px_YY)

    beta, se, is_valid = _beta_se_from_pab(P_XX, P_XY, Px_YY, df)

    # F-statistic and p-value via Cephes betainc
    tau = df / Px_YY
    f_stat = (P_YY - Px_YY) * tau
    p_wald = _f_to_pvalue(f_stat, df, is_valid)

    return beta, se, p_wald


def batch_calc_score_stats_numpy(
    n_cvt: int,
    Hi_eval_null: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Score test statistics for a batch of SNPs.

    Port of likelihood_jax.py::batch_calc_score_stats. Uses fixed null-model
    Hi_eval shared across all SNPs (cheaper than Wald — no per-SNP optimization).

    Score F-statistic uses n_samples (not df) in numerator and P_yy*P_xx
    denominator (not Px_yy). Matches GEMMA CalcRLScore exactly.

    Args:
        n_cvt: Number of covariates.
        Hi_eval_null: Null-model 1/(lambda_null*eval+1) vector (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Tuple of (betas, ses, p_scores) each shape (n_snps,).
    """
    table = build_index_table(n_cvt)
    idx_xx = table["idx_xx"]
    idx_xy = table["idx_xy"]
    idx_yy = table["idx_yy"]
    df = n_samples - n_cvt - 1

    # Batch Pab with shared null Hi_eval
    Pab_batch = batch_compute_pab_numpy(n_cvt, Hi_eval_null, Uab_batch)

    # Score test: extract at level n_cvt (covariates only, NOT genotype)
    P_yy = Pab_batch[:, n_cvt, idx_yy]
    P_yy = np.where((P_yy >= 0.0) & (P_yy < _P_YY_MIN), _P_YY_MIN, P_yy)
    P_xx = Pab_batch[:, n_cvt, idx_xx]
    P_xy = Pab_batch[:, n_cvt, idx_xy]

    # Px_yy for beta/se computation
    Px_yy = Pab_batch[:, n_cvt + 1, idx_yy]
    Px_yy = np.where((Px_yy >= 0.0) & (Px_yy < _P_YY_MIN), _P_YY_MIN, Px_yy)

    beta, se, is_valid = _beta_se_from_pab(P_xx, P_xy, Px_yy, df)

    # Score F-statistic: F = n * P_xy^2 / (P_yy * P_xx)
    safe_P_xx = np.where(is_valid, P_xx, 1.0)
    f_stat = n_samples * (P_xy * P_xy) / (P_yy * safe_P_xx)

    # p_score via Cephes betainc
    p_score = _f_to_pvalue(f_stat, df, is_valid)

    return beta, se, p_score


def _batch_lrt_pvalues_numpy(
    logls_mle: np.ndarray,
    logl_H0: float,
) -> np.ndarray:
    """Compute LRT p-values for a batch of SNPs.

    Port of likelihood_jax.py::calc_lrt_pvalue_jax for batch use.
    LRT statistic = 2 * (logl_H1 - logl_H0), chi2 with df=1.

    Uses special.chi2_sf (stdlib-only) via np.vectorize.

    Args:
        logls_mle: Per-SNP MLE log-likelihoods under alternative (n_snps,).
        logl_H0: Null model MLE log-likelihood (scalar).

    Returns:
        LRT p-values (n_snps,).
    """
    lrt_stats = 2.0 * (logls_mle - logl_H0)
    lrt_stats = np.maximum(lrt_stats, 0.0)
    p_lrts = _chi2_sf_vec(lrt_stats)
    # Propagate NaN from MLE optimization failures (degenerate SNPs)
    p_lrts = np.where(np.isnan(logls_mle), np.nan, p_lrts)
    return p_lrts
