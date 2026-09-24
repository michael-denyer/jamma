"""Pure-NumPy batch REML/MLE evaluation and lambda optimisation.

The fallback the chunk engine runs when the C accelerator is unavailable.
Uab, Pab and Iab batches come from ``jamma.lmm.uab``; the Wald, Score and
LRT statistics that consume the optimised lambdas live in ``jamma.lmm.stats``.

Design:
- _reml_logl / _mle_logl: one finisher per likelihood, for any batch shape
- _batch_grid_pab_numpy / _batch_pab_at_lambda_numpy: grid and per-SNP Pab
- golden_section_optimize_lambda_numpy / _mle: batch lambda optimization

All operations are vectorized over SNPs using NumPy broadcasting.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from jamma.lmm.pab import build_index_table, guard_p_yy
from jamma.lmm.reml_score import (
    _batch_mle_score_log_lambda_numpy,
    _batch_reml_score_log_lambda_numpy,
    _refine_reml_optima,
)
from jamma.lmm.uab import _batch_compute_pab_varying_numpy, _fill_pab_recursion

# The objective handed to the golden-section refinement: per-SNP log-lambdas
# (n_snps,) in, per-SNP log-likelihoods (n_snps,) out.
_BatchLoglFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Likelihood finishers
# ---------------------------------------------------------------------------


def _logl_const(m: int) -> float:
    """Return the likelihood normalizing constant 0.5 * m * (log(m) - log(2*pi) - 1).

    REML passes ``m = df``, MLE passes ``m = n_samples``. Constant across all
    SNPs and lambda values.

    Args:
        m: Degrees of freedom (REML) or sample count (MLE).

    Returns:
        The normalizing constant.
    """
    return 0.5 * m * (np.log(m) - np.log(2.0 * np.pi) - 1.0)


def _logdet_diag(M: np.ndarray) -> np.ndarray:
    """Sum the logs of the logdet diagonal of Pab-shaped ``M[..., n_cvt+2, n_index]``.

    A non-positive entry (a constant SNP, or one collinear with a covariate)
    makes that SNP's sum NaN, the rule ``logdet_diag_term`` in ``_lmm_types.h``
    applies in the C accelerator and GEMMA's ``LogRL_f`` applies by taking an
    unguarded ``log``. Other SNPs in the batch are unaffected.

    Args:
        M: A Pab or Iab batch with any leading shape.

    Returns:
        Log-determinant sums with the leading shape of ``M``.
    """
    table = build_index_table(M.shape[-2] - 2)
    total = np.zeros(M.shape[:-2], dtype=np.float64)
    for row, col in table.logdet_diag_indices:
        d = M[..., row, col]
        with np.errstate(divide="ignore", invalid="ignore"):
            total += np.where(d > 0, np.log(d), np.nan)
    return total


def _guarded_p_yy(Pab: np.ndarray) -> np.ndarray:
    table = build_index_table(Pab.shape[-2] - 2)
    return guard_p_yy(Pab[..., -1, table.idx_yy])


def _reml_logl(
    Pab: np.ndarray, logdet_h: np.ndarray, logdet_iab: np.ndarray, df: int
) -> np.ndarray:
    """Finish the REML log-likelihood from Pab, for grid or per-SNP batches.

    Args:
        Pab: ``(..., n_snps, n_cvt+2, n_index)``.
        logdet_h: logdet(H), broadcastable to ``Pab.shape[:-2]``.
        logdet_iab: ``_logdet_diag(Iab_batch)``, shape (n_snps,); it does not
            depend on lambda, so callers compute it once per chunk.
        df: Degrees of freedom (n_samples - n_cvt - 1).

    Returns:
        Log-likelihoods with shape ``Pab.shape[:-2]``.
    """
    logdet_hiw = _logdet_diag(Pab) - logdet_iab
    return (
        _logl_const(df)
        - 0.5 * logdet_h
        - 0.5 * logdet_hiw
        - 0.5 * df * np.log(_guarded_p_yy(Pab))
    )


def _mle_logl(Pab: np.ndarray, logdet_h: np.ndarray, n: int) -> np.ndarray:
    """Finish the MLE log-likelihood: no logdet_hiw term, n in place of df.

    Args:
        Pab: ``(..., n_snps, n_cvt+2, n_index)``.
        logdet_h: logdet(H), broadcastable to ``Pab.shape[:-2]``.
        n: Number of samples.

    Returns:
        Log-likelihoods with shape ``Pab.shape[:-2]``.
    """
    return _logl_const(n) - 0.5 * logdet_h - 0.5 * n * np.log(_guarded_p_yy(Pab))


# ---------------------------------------------------------------------------
# Pab at per-SNP lambdas and on the shared grid
# ---------------------------------------------------------------------------


def _batch_pab_at_lambda_numpy(
    n_cvt: int,
    lambda_vals: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-SNP Pab and logdet(H) at each SNP's own lambda.

    Args:
        n_cvt: Number of covariates.
        lambda_vals: Per-SNP lambda values (n_snps,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        ``(Pab (n_snps, n_cvt+2, n_index), logdet_h (n_snps,))``.
    """
    v_temp = lambda_vals[:, None] * eigenvalues[None, :] + 1.0
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, 1.0 / v_temp, Uab_batch)
    return Pab_batch, logdet_h


def _batch_grid_pab_numpy(
    n_cvt: int,
    lambdas_grid: np.ndarray,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Pab and logdet(H) at every grid lambda for every SNP.

    All SNPs share each grid lambda, so Hi_eval is (n_grid, n_samples) rather
    than (n_snps, n_samples), which removes the dominant allocation at scale.

    Args:
        n_cvt: Number of covariates.
        lambdas_grid: Grid of lambda values (n_grid,).
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Uab matrices (n_snps, n_samples, n_index).

    Returns:
        ``(Pab (n_grid, n_snps, n_cvt+2, n_index), logdet_h (n_grid, 1))``.
    """
    table = build_index_table(n_cvt)
    n_snps, _n_samples, n_index = Uab_batch.shape
    n_grid = len(lambdas_grid)

    v_temp = lambdas_grid[:, None] * eigenvalues[None, :] + 1.0
    Hi_eval_grid = 1.0 / v_temp
    logdet_h = np.sum(np.log(np.abs(v_temp)), axis=1)

    # Row 0 as one BLAS gemm: contract n_samples -> (n_grid, n_snps, n_index).
    Pab = np.zeros((n_grid, n_snps, n_cvt + 2, n_index), dtype=np.float64)
    Pab[:, :, 0, :] = np.tensordot(Hi_eval_grid, Uab_batch, axes=([1], [1]))
    _fill_pab_recursion(Pab, table, n_cvt)
    return Pab, logdet_h[:, None]


# ---------------------------------------------------------------------------
# Golden section optimizer
# ---------------------------------------------------------------------------


def _batch_golden_section_bracket_numpy(
    compute_batch_fn: _BatchLoglFn,
    grid_logls: np.ndarray,
    log_lambdas: np.ndarray,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        ``(log_opt, coarse_a, coarse_b, interior)``, each (n_snps,): the
        optimal log-lambda, the coarse grid bracket it was refined from, and
        whether the refined bracket left both coarse endpoints.
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
    return (a + b) / 2.0, coarse_a, coarse_b, interior


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

    Uses 20 golden-section iterations, then refines interior peaks with up to
    three safeguarded analytic-score Newton steps.

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
    log_lambdas = np.linspace(np.log(l_min), np.log(l_max), n_grid)
    df = eigenvalues.shape[0] - n_cvt - 1
    logdet_iab = _logdet_diag(Iab_batch)

    # Stage 1: Coarse grid search
    grid_logls = _reml_logl(
        *_batch_grid_pab_numpy(n_cvt, np.exp(log_lambdas), eigenvalues, Uab_batch),
        logdet_iab,
        df,
    )

    def reml_at(log_lams: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Pab, logdet_h = _batch_pab_at_lambda_numpy(
            n_cvt, np.exp(log_lams), eigenvalues, Uab_batch
        )
        return _reml_logl(Pab, logdet_h, logdet_iab, df), Pab

    # Stage 2: Golden section refinement, then one evaluation at the optimum.
    log_opt, coarse_a, coarse_b, interior = _batch_golden_section_bracket_numpy(
        lambda log_lams: reml_at(log_lams)[0], grid_logls, log_lambdas, n_iter
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
    # A SNP whose likelihood is NaN at every grid point reports l_min, as the
    # C refiners do for a fully degenerate SNP.
    log_opt = np.where(np.all(np.isnan(grid_logls), axis=0), log_lambdas[0], log_opt)
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
        n_iter: Golden section iterations (should be >= 20; runner-level code
            enforces the minimum). Twenty place the golden-section estimate
            within about 3.1e-5 of the optimum in log lambda; the score
            refinement supplies the rest of the interior accuracy.

    Returns:
        (optimal_lambdas, optimal_logls_mle) both shape (n_snps,).
    """
    log_lambdas = np.linspace(np.log(l_min), np.log(l_max), n_grid)
    n = eigenvalues.shape[0]

    # Stage 1: Coarse grid search
    grid_logls = _mle_logl(
        *_batch_grid_pab_numpy(n_cvt, np.exp(log_lambdas), eigenvalues, Uab_batch),
        n,
    )

    def mle_at(log_lams: np.ndarray) -> np.ndarray:
        return _mle_logl(
            *_batch_pab_at_lambda_numpy(
                n_cvt, np.exp(log_lams), eigenvalues, Uab_batch
            ),
            n,
        )

    # Stage 2: Golden section refinement, then one evaluation at the optimum.
    log_opt, coarse_a, coarse_b, interior = _batch_golden_section_bracket_numpy(
        mle_at, grid_logls, log_lambdas, n_iter
    )
    log_opt = _refine_reml_optima(
        log_opt,
        coarse_a,
        coarse_b,
        interior,
        lambda values, indices: _batch_mle_score_log_lambda_numpy(
            n_cvt, values, eigenvalues, Uab_batch[indices]
        ),
    )
    return np.exp(log_opt), mle_at(log_opt)
