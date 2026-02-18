"""Shared LMM chunk computation.

Encapsulates the mode-specific dispatch logic (modes 1-4) that is
shared across runner_jax.py, runner_streaming.py, and loco.py.

The caller is responsible for:
- Computing Uab_batch via batch_compute_uab (before calling this)
- Calling block_until_ready() on results (after calling this)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jamma.lmm.likelihood_jax import (
    batch_calc_score_stats,
    batch_calc_wald_stats,
    batch_compute_iab,
    calc_lrt_pvalue_jax,
    golden_section_optimize_lambda_mle,
)
from jamma.lmm.prepare import _grid_optimize_lambda_batched


def _compute_lmm_chunk(
    lmm_mode: int,
    n_cvt: int,
    eigenvalues: jax.Array,
    Uab_batch: jax.Array,
    n_samples: int,
    *,
    # Wald/LRT optimization params (modes 1, 2, 4)
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 100,
    n_refine: int = 5,
    # Score test params (modes 3, 4)
    Hi_eval_null: jax.Array | None = None,
    # LRT params (modes 2, 4)
    logl_H0: float | None = None,
) -> dict[str, jax.Array | None]:
    """Compute LMM statistics for a chunk of SNPs.

    Dispatches to mode-specific computation (Wald, LRT, Score, or All)
    without calling batch_compute_uab or block_until_ready -- those are
    the caller's responsibility.

    Args:
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues on device.
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations.
        Hi_eval_null: Pre-computed 1/(lambda_null*eval+1) for Score test.
        logl_H0: Null model MLE log-likelihood for LRT.

    Returns:
        Dict with keys: best_lambdas, best_logls, betas, ses, p_walds,
        best_lambdas_mle, p_lrts, p_scores. Keys not relevant to the
        mode are set to None.
    """
    result: dict[str, jax.Array | None] = {
        "best_lambdas": None,
        "best_logls": None,
        "betas": None,
        "ses": None,
        "p_walds": None,
        "best_lambdas_mle": None,
        "p_lrts": None,
        "p_scores": None,
    }

    if lmm_mode == 1:  # Wald
        Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
        best_lambdas, best_logls = _grid_optimize_lambda_batched(
            n_cvt,
            eigenvalues,
            Uab_batch,
            Iab_batch,
            l_min,
            l_max,
            n_grid,
            n_refine,
        )
        betas, ses, p_walds = batch_calc_wald_stats(
            n_cvt, best_lambdas, eigenvalues, Uab_batch, n_samples
        )
        result["best_lambdas"] = best_lambdas
        result["best_logls"] = best_logls
        result["betas"] = betas
        result["ses"] = ses
        result["p_walds"] = p_walds

    elif lmm_mode == 3:  # Score
        betas, ses, p_scores = batch_calc_score_stats(
            n_cvt, Hi_eval_null, Uab_batch, n_samples
        )
        result["betas"] = betas
        result["ses"] = ses
        result["p_scores"] = p_scores

    elif lmm_mode == 2:  # LRT
        best_lambdas_mle, best_logls_mle = golden_section_optimize_lambda_mle(
            n_cvt,
            eigenvalues,
            Uab_batch,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_iter=max(n_refine, 20),
        )
        p_lrts = jax.vmap(calc_lrt_pvalue_jax)(
            best_logls_mle, jnp.full_like(best_logls_mle, logl_H0)
        )
        result["best_lambdas_mle"] = best_lambdas_mle
        result["p_lrts"] = p_lrts

    elif lmm_mode == 4:  # All tests
        # Score test (cheapest, no optimization, reads Uab_batch)
        _, _, p_scores = batch_calc_score_stats(
            n_cvt, Hi_eval_null, Uab_batch, n_samples
        )

        # MLE optimization for LRT
        best_lambdas_mle, best_logls_mle = golden_section_optimize_lambda_mle(
            n_cvt,
            eigenvalues,
            Uab_batch,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_iter=max(n_refine, 20),
        )
        p_lrts = jax.vmap(calc_lrt_pvalue_jax)(
            best_logls_mle, jnp.full_like(best_logls_mle, logl_H0)
        )

        # REML optimization for Wald
        Iab_batch = batch_compute_iab(n_cvt, Uab_batch)
        best_lambdas, best_logls = _grid_optimize_lambda_batched(
            n_cvt,
            eigenvalues,
            Uab_batch,
            Iab_batch,
            l_min,
            l_max,
            n_grid,
            n_refine,
        )
        betas, ses, p_walds = batch_calc_wald_stats(
            n_cvt, best_lambdas, eigenvalues, Uab_batch, n_samples
        )

        result["best_lambdas"] = best_lambdas
        result["best_logls"] = best_logls
        result["betas"] = betas
        result["ses"] = ses
        result["p_walds"] = p_walds
        result["best_lambdas_mle"] = best_lambdas_mle
        result["p_lrts"] = p_lrts
        result["p_scores"] = p_scores

    return result
