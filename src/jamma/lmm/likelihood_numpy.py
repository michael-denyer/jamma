"""Pure-NumPy batch REML/MLE evaluation and lambda optimisation.

The fallback the chunk engine runs when the C accelerator is unavailable.
Uab, Pab and Iab batches come from ``jamma.lmm.uab``; the Wald, Score and
LRT statistics that consume the optimised lambdas live in ``jamma.lmm.stats``.

Design:
- _compute_reml_const: the per-run REML normalising constant
- _batch_grid_*_numpy / _batch_*_at_lambda_numpy: grid and per-SNP evaluation
- golden_section_optimize_lambda_numpy / _mle: batch lambda optimization

All operations are vectorized over SNPs using NumPy broadcasting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from jamma.lmm.pab import build_index_table, guard_p_yy
from jamma.lmm.reml_score import (
    _batch_reml_score_log_lambda_numpy,
    _refine_reml_optima,
)
from jamma.lmm.uab import _batch_compute_pab_varying_numpy, _fill_pab_recursion

# The objective handed to the golden-section refinement: per-SNP log-lambdas
# (n_snps,) in, per-SNP log-likelihoods (n_snps,) out.
_BatchLoglFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Precomputed REML constant
# ---------------------------------------------------------------------------


def _compute_reml_const(df: int) -> float:
    """Precompute REML normalizing constant: 0.5 * df * (log(df) - log(2*pi) - 1).

    Constant across all SNPs and lambda values — compute once per run.

    Args:
        df: Degrees of freedom (n_samples - n_cvt - 1).

    Returns:
        REML normalizing constant.
    """
    return 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)


# ---------------------------------------------------------------------------
# Batch REML / MLE log-likelihood evaluation
# ---------------------------------------------------------------------------


def _batch_pab_at_lambda_numpy(
    n_cvt: int,
    lambda_vals: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-SNP Pab, logdet(H), and guarded P_yy at each SNP's lambda.

    The per-SNP mirror of ``_batch_grid_pab_numpy``: shared by the REML and
    MLE evaluators, which differ only in the finisher they apply.

    Args:
        n_cvt: Number of covariates.
        lambda_vals: Per-SNP lambda values (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Tuple of (Pab, logdet_h, P_yy):
            Pab: (n_snps, n_cvt+2, n_index)
            logdet_h: (n_snps,)
            P_yy: (n_snps,) -- guarded
    """
    table = build_index_table(n_cvt)
    nc_total = n_cvt + 1

    # Per-SNP H-inv weights: (n_snps, n_samples)
    v_temp = lambda_vals[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_batch = 1.0 / v_temp  # (n_snps, n_samples)

    # Log determinant of H per SNP: (n_snps,)
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)

    # Pab with per-SNP Hi_eval
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, Hi_eval_batch, Uab_batch)

    P_yy = guard_p_yy(Pab_batch[:, nc_total, table.idx_yy])
    return Pab_batch, logdet_h, P_yy


def _batch_reml_at_lambda_numpy(
    n_cvt: int,
    lambda_vals: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Iab_batch: np.ndarray,
    reml_const: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate REML log-likelihood for each SNP at its own lambda value.

    Args:
        n_cvt: Number of covariates.
        lambda_vals: Per-SNP lambda values (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index).
        reml_const: Precomputed 0.5*df*(log(df)-log(2*pi)-1).

    Returns:
        ``(log-likelihoods (n_snps,), Pab_batch (n_snps, n_cvt+2, n_index))``.
        Pab falls out of the log-likelihood computation, so it is always
        returned rather than gated on a flag; the refinement loop discards it
        and the final evaluation feeds it to the Wald statistics.
    """
    table = build_index_table(n_cvt)
    n_snps = Uab_batch.shape[0]
    n = eigenvalues.shape[0]
    df = n - n_cvt - 1

    Pab_batch, logdet_h, P_yy = _batch_pab_at_lambda_numpy(
        n_cvt, lambda_vals, eigenvalues, Uab_batch
    )

    # logdet_hiw per SNP: sum over diagonal indices
    # Guard: non-positive diagonal Pab/Iab entries (degenerate SNPs) use 0.0
    # instead of log to prevent NaN/Inf from corrupting the batch. Degenerate
    # SNPs are caught downstream by the P_yy < 0 → NaN guard.
    logdet_hiw = np.zeros(n_snps, dtype=np.float64)
    for row, col in table.logdet_diag_indices:
        d_pab = Pab_batch[:, row, col]  # (n_snps,)
        d_iab = Iab_batch[:, row, col]  # (n_snps,)
        with np.errstate(divide="ignore", invalid="ignore"):
            logdet_hiw += np.where(d_pab > 0, np.log(d_pab), 0.0)
            logdet_hiw -= np.where(d_iab > 0, np.log(d_iab), 0.0)

    # REML log-likelihood per SNP
    logl = reml_const - 0.5 * logdet_h - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)
    return logl, Pab_batch


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
    n = eigenvalues.shape[0]

    _Pab, logdet_h, P_yy = _batch_pab_at_lambda_numpy(
        n_cvt, lambda_vals, eigenvalues, Uab_batch
    )

    # MLE log-likelihood per SNP (no logdet_hiw, uses n not df)
    c = 0.5 * n * (np.log(n) - np.log(2.0 * np.pi) - 1.0)
    return c - 0.5 * logdet_h - 0.5 * n * np.log(P_yy)


# ---------------------------------------------------------------------------
# Grid evaluations
# ---------------------------------------------------------------------------


def _batch_grid_pab_numpy(
    n_cvt: int,
    lambdas_grid: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute grid-level Pab, logdet(H), and guarded P_yy.

    Shared computation for both REML and MLE grid evaluation. Since all SNPs
    share the same lambda at each grid point, Hi_eval is (n_grid, n_samples)
    not (n_snps, n_samples) -- eliminates the dominant memory allocation at
    scale.

    Args:
        n_cvt: Number of covariates.
        lambdas_grid: Grid of lambda values (n_grid,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        Tuple of (Pab, logdet_h, P_yy):
            Pab: (n_grid, n_snps, n_cvt+2, n_index)
            logdet_h: (n_grid,)
            P_yy: (n_grid, n_snps) -- guarded
    """
    table = build_index_table(n_cvt)
    n_snps, _n_samples, n_index = Uab_batch.shape
    n_grid = len(lambdas_grid)
    nc_total = n_cvt + 1

    # All grid lambdas at once: (n_grid, n_samples)
    v_temp = lambdas_grid[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_grid = 1.0 / v_temp

    # logdet(H) per grid lambda -- shared across SNPs: (n_grid,)
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)

    # Pab row 0 via single tensordot (BLAS gemm):
    # Hi_eval_grid (n_grid, n_samples) x Uab_batch (n_snps, n_samples, n_index)
    # contracts n_samples -> (n_grid, n_snps, n_index)
    Pab = np.zeros((n_grid, n_snps, n_cvt + 2, n_index), dtype=np.float64)
    Pab[:, :, 0, :] = np.tensordot(Hi_eval_grid, Uab_batch, axes=([1], [1]))

    # Recursive rows 1..n_cvt+1 (uses ... indexing for 4D)
    _fill_pab_recursion(Pab, table, n_cvt)

    P_yy = guard_p_yy(Pab[:, :, nc_total, table.idx_yy])

    return Pab, logdet_h, P_yy


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
    table = build_index_table(n_cvt)
    n_snps = Uab_batch.shape[0]
    n_samples = eigenvalues.shape[0]
    df = n_samples - n_cvt - 1

    Pab, logdet_h, P_yy = _batch_grid_pab_numpy(
        n_cvt, lambdas_grid, eigenvalues, Uab_batch
    )
    n_grid = len(lambdas_grid)

    # logdet_hiw: Iab part is lambda-independent, precompute once
    logdet_iab = np.zeros(n_snps, dtype=np.float64)
    logdet_pab = np.zeros((n_grid, n_snps), dtype=np.float64)
    for row, col in table.logdet_diag_indices:
        d_pab = Pab[:, :, row, col]  # (n_grid, n_snps)
        d_iab = Iab_batch[:, row, col]  # (n_snps,)
        with np.errstate(divide="ignore", invalid="ignore"):
            logdet_pab += np.where(d_pab > 0, np.log(d_pab), 0.0)
            logdet_iab += np.where(d_iab > 0, np.log(d_iab), 0.0)
    logdet_hiw = logdet_pab - logdet_iab[None, :]

    # REML log-likelihood: (n_grid, n_snps)
    c = 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)
    return c - 0.5 * logdet_h[:, None] - 0.5 * logdet_hiw - 0.5 * df * np.log(P_yy)


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
    n_samples = eigenvalues.shape[0]

    _Pab, logdet_h, P_yy = _batch_grid_pab_numpy(
        n_cvt, lambdas_grid, eigenvalues, Uab_batch
    )

    # MLE log-likelihood: (n_grid, n_snps) -- no logdet_hiw, uses n not df
    c = 0.5 * n_samples * (np.log(n_samples) - np.log(2.0 * np.pi) - 1.0)
    return c - 0.5 * logdet_h[:, None] - 0.5 * n_samples * np.log(P_yy)


# ---------------------------------------------------------------------------
# Golden section optimizer
# ---------------------------------------------------------------------------


def _batch_golden_section_bracket_numpy(
    compute_batch_fn: _BatchLoglFn,
    grid_logls: np.ndarray,
    log_lambdas: np.ndarray,
    n_iter: int,
    *,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine each SNP's bracket and return the optimal log-lambda per SNP.

    Grid-to-golden-section refinement using NumPy broadcasting over SNPs.

    All operations are vectorized over SNPs (axis 0). Twenty iterations shrink
    the initial bracket by 0.618^20; interior REML peaks receive a subsequent
    analytic-score correction because bracket width alone does not bound the
    lambda error when likelihood comparisons are at floating-point noise.

    Stops at the optimal log-lambda rather than evaluating there, because each
    caller wants a different final evaluation: the REML optimizers need the Pab
    batch that falls out of it, the MLE optimizer has no Pab. Every caller then
    evaluates at the returned midpoint, so its (lambda, logl) pair comes from a
    single point.

    Args:
        compute_batch_fn: callable(log_lambdas_per_snp: (n_snps,)) -> (n_snps,).
        grid_logls: Grid log-likelihoods (n_grid, n_snps).
        log_lambdas: Log-scale grid points (n_grid,).
        n_iter: Golden section iterations (should be >= 20).

    Returns:
        Optimal log-lambda per SNP (n_snps,).
    """
    phi = 0.6180339887498949  # golden ratio - 1

    # Find best grid point per SNP and bracket
    safe_logls = np.where(np.isnan(grid_logls), -np.inf, grid_logls)
    best_idx = np.argmax(safe_logls, axis=0)  # (n_snps,)
    idx_low = np.maximum(best_idx - 1, 0)
    idx_high = np.minimum(best_idx + 1, len(log_lambdas) - 1)

    a = log_lambdas[idx_low]  # (n_snps,)
    b = log_lambdas[idx_high]  # (n_snps,)
    coarse_a = a.copy()
    coarse_b = b.copy()

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

    # A bracket that still touches a coarse endpoint can be monotone.
    # Refine every enclosed peak, independent of likelihood rounding.
    interior = (a > coarse_a) & (b < coarse_b)
    log_opt = (a + b) / 2.0
    if return_details:
        return log_opt, coarse_a, coarse_b, interior
    return log_opt


def golden_section_optimize_lambda_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    Iab_batch: np.ndarray,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimize REML lambda using grid search + golden section refinement.

    Optimize REML lambda using grid search + golden section refinement with
    NumPy broadcasting over the SNP batch.

    Uses 20 golden-section iterations, then refines interior peaks with one
    safeguarded analytic-score Newton step.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index).
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Coarse grid points.
        n_iter: Golden section iterations (runner-level code requires at least 20).

    Returns:
        ``(optimal_lambdas, optimal_logls, Pab_final)`` where the first two are
        (n_snps,) and Pab_final is (n_snps, n_cvt+2, n_index). Pab comes from
        the final evaluation, so the Wald stats step reuses it instead of
        reconstructing Hi_eval and Pab.
    """
    log_l_min = np.log(l_min)
    log_l_max = np.log(l_max)
    log_lambdas = np.linspace(log_l_min, log_l_max, n_grid)
    lambdas_grid = np.exp(log_lambdas)

    n = eigenvalues.shape[0]
    df = n - n_cvt - 1
    reml_const = _compute_reml_const(df)

    # Stage 1: Coarse grid search
    grid_logls = _batch_grid_reml_numpy(
        n_cvt, lambdas_grid, eigenvalues, Uab_batch, Iab_batch
    )

    def reml_at(log_lams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _batch_reml_at_lambda_numpy(
            n_cvt,
            np.exp(log_lams),
            eigenvalues,
            Uab_batch,
            Iab_batch,
            reml_const=reml_const,
        )

    # Stage 2: Golden section refinement, then one evaluation at the optimum.
    log_opt, coarse_a, coarse_b, interior = _batch_golden_section_bracket_numpy(
        lambda log_lams: reml_at(log_lams)[0],
        grid_logls,
        log_lambdas,
        n_iter,
        return_details=True,
    )
    log_opt = _refine_reml_optima(
        log_opt,
        coarse_a,
        coarse_b,
        interior,
        lambda values, indices: _batch_reml_score_log_lambda_numpy(
            n_cvt, values, eigenvalues, Uab_batch[indices]
        ),
    )
    opt_logls, Pab_final = reml_at(log_opt)
    return np.exp(log_opt), opt_logls, Pab_final


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

    Optimize MLE lambda using grid search + golden section refinement.
    No Iab argument needed (MLE has no logdet_hiw term).

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).
        l_min: Minimum lambda.
        l_max: Maximum lambda.
        n_grid: Coarse grid points.
        n_iter: Golden section iterations (should be >= 20 for 1e-5 tolerance;
            runner-level code enforces the minimum).

    Returns:
        (optimal_lambdas, optimal_logls_mle) both shape (n_snps,).
    """
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

    # Stage 2: Golden section refinement, then one evaluation at the optimum.
    log_opt, _coarse_a, _coarse_b, _interior = _batch_golden_section_bracket_numpy(
        compute_mle_batch,
        grid_logls,
        log_lambdas,
        n_iter,
        return_details=True,
    )
    return np.exp(log_opt), compute_mle_batch(log_opt)
