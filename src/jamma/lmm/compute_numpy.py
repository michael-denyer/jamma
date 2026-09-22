"""NumPy mode dispatch for LMM chunk computation.

The full-Uab helpers here (``_compute_wald_numpy`` and its LRT and Score
siblings) are pure NumPy, reached only through ``compute_lmm_chunk_numpy``,
which the runner calls only on ``DispatchPath.NUMPY_FALLBACK``. That path is
selected only when the extension is absent.

``compute_wald_split_numpy`` is the exception. It is the ``NUMPY_WALD`` kernel
body, which ``chunk_kernel`` calls directly on the split Uab rows.

The caller computes ``Uab_batch`` (n_snps, n_samples, n_index) for chunk
dispatch. Every call is synchronous; results are available when it returns.
"""

from __future__ import annotations

import numpy as np

from jamma.lmm.likelihood_numpy import (
    golden_section_optimize_lambda_mle_numpy,
    golden_section_optimize_lambda_numpy,
    golden_section_optimize_lambda_split_ncvt1_numpy,
)
from jamma.lmm.schema import MIN_N_REFINE, LmmMode, LmmTest, get_spec
from jamma.lmm.stats import (
    _batch_lrt_pvalues_numpy,
    batch_calc_score_stats_numpy,
    batch_calc_wald_stats_from_pab_numpy,
)
from jamma.lmm.uab import batch_compute_iab_numpy, compute_iab_invariant_scalars_ncvt1

MAX_C_N_CVT = 100  # Must match MAX_N_CVT in _lmm_types.h


def compute_wald_split_numpy(
    eigenvalues: np.ndarray,
    uab_varying_soa: np.ndarray,
    uab_invariant_soa: np.ndarray,
    iab_scalars: tuple[float, float, float, float],
    n_samples: int,
    *,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
) -> dict[str, np.ndarray]:
    """Intercept-only Wald from three varying rows and shared invariant products."""
    iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = iab_scalars
    lambdas, logls, Pab_final = golden_section_optimize_lambda_split_ncvt1_numpy(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_s_ww,
        iab_s_wy,
        iab_s_yy,
        iab_logdet,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    betas, ses, pwalds = batch_calc_wald_stats_from_pab_numpy(1, Pab_final, n_samples)
    return {
        "lambdas": lambdas,
        "logls": logls,
        "betas": betas,
        "ses": ses,
        "pwalds": pwalds,
    }


def _compute_wald_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
) -> dict[str, np.ndarray]:
    """Compute REML-optimized Wald test statistics.

    Pure NumPy. The runner reaches this only on ``DispatchPath.NUMPY_FALLBACK``,
    which is selected only when the extension is absent, so there is no C branch
    to take: n_cvt=1 uses the split-Uab optimizer, n_cvt>1 the generic one.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (should be >= 20 for 1e-5 tolerance;
            C extension requires >= 1). Runner-level code enforces the minimum.

    Returns:
        Dict with keys: lambdas, logls, betas, ses, pwalds.
    """

    if n_cvt == 1:
        # Python split path for n_cvt=1: separate invariant (ww, wy, yy)
        # and varying (wx, xx, xy) Uab columns to reduce per-SNP computation.
        # Column layout: 0=ww, 1=wx, 2=wy, 3=xx, 4=xy, 5=yy.
        # Invariant columns (ww, wy, yy) are identical across SNPs — use SNP 0.
        uab_varying_soa = np.stack(
            [Uab_batch[:, :, 1], Uab_batch[:, :, 3], Uab_batch[:, :, 4]], axis=1
        )  # (n_snps, 3, n_samples): rows [wx, xx, xy]
        uab_invariant_soa = np.stack(
            [Uab_batch[0, :, 0], Uab_batch[0, :, 2], Uab_batch[0, :, 5]], axis=0
        )  # (3, n_samples): rows [ww, wy, yy]

        return compute_wald_split_numpy(
            eigenvalues,
            uab_varying_soa,
            uab_invariant_soa,
            compute_iab_invariant_scalars_ncvt1(uab_invariant_soa),
            n_samples,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_refine=n_refine,
        )
    else:
        # Generic Python path for n_cvt > 1
        Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
        lambdas, logls, Pab_final = golden_section_optimize_lambda_numpy(
            n_cvt,
            eigenvalues,
            Uab_batch,
            Iab_batch,
            l_min=l_min,
            l_max=l_max,
            n_grid=n_grid,
            n_iter=n_refine,
        )

    betas, ses, pwalds = batch_calc_wald_stats_from_pab_numpy(
        n_cvt, Pab_final, n_samples
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

    Pure NumPy, via the golden section MLE optimizer. The runner reaches this
    only on ``DispatchPath.NUMPY_FALLBACK``, which is selected only when the
    extension is absent.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution.
        n_refine: Golden section iterations (should be >= 20 for 1e-5 tolerance).
        logl_H0: Null model MLE log-likelihood (scalar).

    Returns:
        Dict with ``logls``, ``lambdas_mle``, and ``p_lrts``. GEMMA reports the
        alternative-model MLE likelihood as ``logl_H1`` in modes 2 and 4.
    """
    lambdas_mle, logls_mle = golden_section_optimize_lambda_mle_numpy(
        n_cvt,
        eigenvalues,
        Uab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=n_refine,
    )
    p_lrts = _batch_lrt_pvalues_numpy(logls_mle, logl_H0)
    return {"logls": logls_mle, "lambdas_mle": lambdas_mle, "p_lrts": p_lrts}


def _compute_score_numpy(
    n_cvt: int,
    eigenvalues: np.ndarray,
    Hi_eval_null: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """Compute Score test statistics (no optimization needed).

    Pure NumPy. The runner reaches this only on ``DispatchPath.NUMPY_FALLBACK``,
    which is selected only when the extension is absent.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,). Used by C path for validation.
        Hi_eval_null: Pre-computed null-model Hi_eval (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.

    Returns:
        Dict with keys: betas, ses, p_scores.
    """
    if not np.all(np.isfinite(Hi_eval_null)):
        bad_idx = np.where(~np.isfinite(Hi_eval_null))[0]
        raise ValueError(
            f"Hi_eval_null has {len(bad_idx)} non-finite value(s) at indices "
            f"{bad_idx[:5].tolist()}. Null model optimization may have failed."
        )
    if np.any(Hi_eval_null <= 0):
        bad_idx = np.where(Hi_eval_null <= 0)[0]
        raise ValueError(
            f"Hi_eval_null has {len(bad_idx)} non-positive value(s) at indices "
            f"{bad_idx[:5].tolist()}. Check kinship matrix conditioning."
        )

    betas, ses, p_scores = batch_calc_score_stats_numpy(
        n_cvt, Hi_eval_null, Uab_batch, n_samples
    )
    return {"betas": betas, "ses": ses, "p_scores": p_scores}


def compute_lmm_chunk_numpy(
    lmm_mode: LmmMode,
    n_cvt: int,
    eigenvalues: np.ndarray,
    Uab_batch: np.ndarray,
    n_samples: int,
    *,
    Hi_eval_null: np.ndarray,
    logl_H0: float,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    n_grid: int = 50,
    n_refine: int = MIN_N_REFINE,
) -> dict[str, np.ndarray]:
    """Compute LMM statistics for a chunk of SNPs (NumPy backend).

    Runs the mode's tests in the order Score, Wald, LRT, so in mode 4 Wald's
    REML beta and se replace Score's, and LRT's MLE likelihood replaces
    Wald's REML one: GEMMA writes the alternative-model MLE likelihood into
    ``logl_H1`` whenever LRT runs.

    Args:
        lmm_mode: Test type: 1=Wald, 2=LRT, 3=Score, 4=All.
        n_cvt: Number of covariates.
        eigenvalues: Kinship eigenvalues (n_samples,).
        Uab_batch: Pre-computed Uab matrices (n_snps, n_samples, n_index).
        n_samples: Number of samples.
        Hi_eval_null: Pre-computed 1/(lambda_null*eval+1), read by Score.
        logl_H0: Null model MLE log-likelihood, read by LRT.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        n_grid: Grid search resolution for lambda bracketing.
        n_refine: Golden section iterations. ``LmmConfig`` raises this to
            ``MIN_N_REFINE`` for every runner; a direct caller passes it.

    Returns:
        Dict keyed by the mode's ``stat_columns`` array keys.
    """
    tests = get_spec(lmm_mode).tests
    result: dict[str, np.ndarray] = {}
    if LmmTest.SCORE in tests:
        result.update(
            _compute_score_numpy(n_cvt, eigenvalues, Hi_eval_null, Uab_batch, n_samples)
        )
    if LmmTest.WALD in tests:
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
            )
        )
    if LmmTest.LRT in tests:
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
    return result
