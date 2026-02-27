"""NumPy mode dispatch for LMM chunk computation.

Mirrors compute.py for the NumPy backend. All JAX replaced with NumPy batch
functions from likelihood_numpy.py. No JAX imports.

The caller is responsible for:
- Computing Uab_batch via batch_compute_uab_numpy (before calling this)
- There is no async dispatch in the NumPy backend — results are immediately
  available after the call returns.
"""

from __future__ import annotations

import numpy as np

from jamma.lmm.likelihood_numpy import (
    _batch_lrt_pvalues_numpy,
    batch_calc_score_stats_numpy,
    batch_calc_wald_stats_numpy,
    batch_compute_iab_numpy,
    golden_section_optimize_lambda_mle_numpy,
    golden_section_optimize_lambda_numpy,
)


def _compute_wald_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    Iab_batch: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute REML-optimized Wald test statistics.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (minimum 20 enforced internally).
        Iab_batch: Pre-computed identity-weighted Pab. If None, computed internally.

    Returns:
        Dict with keys: lambdas, logls, betas, ses, pwalds.
    """
    if Iab_batch is None:
        Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
    lambdas, logls = golden_section_optimize_lambda_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        Iab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        n_cvt, lambdas, eigenvalues, Uab_batch, n_samples
    )
    return {
        "lambdas": lambdas,
        "logls": logls,
        "betas": betas,
        "ses": ses,
        "pwalds": pwalds,
    }


def _compute_lrt_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
    logl_H0: float,
) -> dict[str, np.ndarray]:
    """Compute MLE-optimized LRT statistics.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (minimum 20 enforced internally).
        logl_H0: Null model MLE log-likelihood (scalar).

    Returns:
        Dict with keys: lambdas_mle, p_lrts.
    """
    lambdas_mle, logls_mle = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,  # callee enforces min 20
    )
    p_lrts = _batch_lrt_pvalues_numpy(logls_mle, logl_H0)
    return {"lambdas_mle": lambdas_mle, "p_lrts": p_lrts}


def _compute_score_numpy(
    n_cvt: int,
    Hi_eval_null: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Compute Score test statistics (no optimization needed).

    Args:
        n_cvt: Number of covariates.
        Hi_eval_null: Pre-computed null-model Hi_eval (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Dict with keys: betas, ses, p_scores.
    """
    betas, ses, p_scores = batch_calc_score_stats_numpy(
        n_cvt, Hi_eval_null, Uab_batch, n_samples
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def _compute_lmm_chunk_numpy(
    lmm_mode: int,
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    *,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = 10,
    Hi_eval_null: np.ndarray | None = None,
    logl_H0: float | None = None,
) -> dict[str, np.ndarray | None]:
    """Compute LMM statistics for a chunk of SNPs (NumPy backend).

    Mirrors _compute_lmm_chunk in compute.py but uses NumPy batch functions
    instead of JAX. No async dispatch — results are immediately available.

    Args:
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations (minimum 20 enforced).
        Hi_eval_null: Pre-computed 1/(lambda_null*eval+1) for Score test.
        logl_H0: Null model MLE log-likelihood for LRT.

    Returns:
        Dict with keys: lambdas, logls, betas, ses, pwalds,
        lambdas_mle, p_lrts, p_scores. Keys not relevant to the
        mode are set to None.
    """
    if lmm_mode in (2, 4) and logl_H0 is None:
        raise ValueError("logl_H0 is required for LRT (mode 2) and All (mode 4)")
    if lmm_mode in (3, 4) and Hi_eval_null is None:
        raise ValueError("Hi_eval_null is required for Score (mode 3) and All (mode 4)")

    result: dict[str, np.ndarray | None] = {
        "lambdas": None,
        "logls": None,
        "betas": None,
        "ses": None,
        "pwalds": None,
        "lambdas_mle": None,
        "p_lrts": None,
        "p_scores": None,
    }

    if lmm_mode == 1:
        result.update(
            _compute_wald_numpy(
                n_cvt, eigenvalues, Uab_batch, n_samples, l_min, l_max, n_grid, n_refine
            )
        )

    elif lmm_mode == 2:
        result.update(
            _compute_lrt_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
                logl_H0,
            )
        )

    elif lmm_mode == 3:
        result.update(_compute_score_numpy(n_cvt, Hi_eval_null, Uab_batch, n_samples))

    elif lmm_mode == 4:
        # Compose all three tests; Score is cheapest so runs first
        result.update(_compute_score_numpy(n_cvt, Hi_eval_null, Uab_batch, n_samples))
        result.update(
            _compute_lrt_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                l_min,
                l_max,
                n_grid,
                n_refine,
                logl_H0,
            )
        )
        # Pre-compute Iab once for Wald (lambda-independent)
        Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
        # Wald overwrites betas/ses from Score (REML-optimized values)
        result.update(
            _compute_wald_numpy(
                n_cvt,
                eigenvalues,
                Uab_batch,
                n_samples,
                l_min,
                l_max,
                n_grid,
                n_refine,
                Iab_batch=Iab_batch,
            )
        )

    else:
        raise ValueError(
            f"lmm_mode must be 1 (Wald), 2 (LRT), 3 (Score), or 4 (All), got {lmm_mode}"
        )

    return result
