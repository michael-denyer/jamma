"""Pure-NumPy setup utilities shared by JAX and NumPy LMM runners.

Provides covariate matrix construction, eigendecomposition handling,
and null model computation without any JAX dependency. Both the JAX
runner (via prepare.py) and the NumPy runner (runner_numpy.py) import
from this module.
"""

from __future__ import annotations

import gc

import numpy as np
from loguru import logger

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import compute_null_model_mle
from jamma.utils.logging import log_rss_memory


def _build_covariate_matrix(
    covariates: np.ndarray | None, n_samples: int
) -> tuple[np.ndarray, int]:
    """Construct covariate matrix W and return (W, n_cvt).

    If covariates is None, uses intercept-only model. Warns if provided
    covariates lack an intercept column.

    Args:
        covariates: Optional covariate matrix (n_samples, n_covariates).
        n_samples: Number of samples (for intercept construction).

    Returns:
        Tuple of (W, n_cvt) where W is the covariate matrix.
    """
    if covariates is None:
        W = np.ones((n_samples, 1))
    else:
        W = covariates.astype(np.float64)
        if not np.allclose(W[:, 0], 1.0):
            logger.warning(
                "Covariate matrix does not have intercept column "
                "(first column is not all 1s). "
                "Model will NOT include an intercept term."
            )
    n_cvt = W.shape[1]
    # df = n_samples - n_cvt - 1 must be positive for valid REML
    if n_samples <= n_cvt + 1:
        raise ValueError(
            f"Over-parameterized model: {n_samples} samples with {n_cvt} "
            f"covariates leaves df={n_samples - n_cvt - 1} "
            f"(need at least {n_cvt + 2} samples)"
        )
    # Rank-deficient covariates cause singular Pab → cryptic LAPACK errors
    rank = np.linalg.matrix_rank(W)
    if rank < n_cvt:
        raise ValueError(
            f"Covariate matrix is rank-deficient: rank={rank} but "
            f"n_cvt={n_cvt}. Check for linearly dependent columns."
        )
    return W, n_cvt


def _eigendecompose_or_reuse(
    kinship: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    show_progress: bool,
    label: str,
    *,
    check_memory: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigendecomposition, computing it if not provided.

    Args:
        kinship: Kinship matrix (n_samples, n_samples), or None when
            pre-computed eigenvalues and eigenvectors are provided.
        eigenvalues: Pre-computed eigenvalues or None.
        eigenvectors: Pre-computed eigenvectors or None.
        show_progress: Whether to log memory usage.
        label: Label for memory logging (e.g. "lmm_jax", "lmm_streaming").
        check_memory: If True (default), check available memory before
            eigendecomposition.

    Returns:
        Tuple of (eigenvalues, eigenvectors).
    """
    if eigenvalues is not None and eigenvectors is not None:
        if show_progress:
            logger.debug("Using pre-computed eigendecomposition")
        return eigenvalues, eigenvectors

    if show_progress:
        log_rss_memory(label, "before_eigendecomp")
    eigenvalues_np, U = eigendecompose_kinship(kinship, check_memory=check_memory)
    # Release LAPACK DSYEVD workspace before LMM phase
    gc.collect()
    if show_progress:
        log_rss_memory(label, "after_eigendecomp")
    return eigenvalues_np, U


def _compute_null_model_common(
    lmm_mode: int,
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    show_progress: bool,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> tuple[float | None, float | None, np.ndarray | None]:
    """Compute null model MLE for Score, LRT, and All-tests modes.

    Pure-NumPy version of the null model computation. Returns Hi_eval_null
    as a plain NumPy array (no JAX device placement). Callers that need a
    JAX array should wrap the result with jax.device_put.

    GEMMA computes both REML and MLE null lambdas in CalcLambda, but uses
    MLE lambda for Hi_eval in the Score test:
    Hi_eval_null = 1 / (lambda_null_mle * eigenvalues + 1).

    Wald (mode 1) skips this entirely; LRT (mode 2) needs only logl_H0;
    Score/All (modes 3, 4) precompute Hi_eval at the null lambda.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        eigenvalues_np: Kinship eigenvalues as numpy array.
        UtW: Rotated covariates.
        Uty: Rotated phenotype.
        n_cvt: Number of covariates.
        show_progress: Whether to log results.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.

    Returns:
        Tuple of (logl_H0, lambda_null_mle, Hi_eval_null_np).
        All None for Wald (mode 1). For LRT (mode 2), Hi_eval_null_np
        is None. For Score/All (modes 3, 4), all three are populated.
    """
    if lmm_mode not in (2, 3, 4):
        return None, None, None

    lambda_null_mle, logl_H0 = compute_null_model_mle(
        eigenvalues_np, UtW, Uty, n_cvt, l_min=l_min, l_max=l_max
    )
    if show_progress:
        logger.info(
            f"Null model MLE: lambda={lambda_null_mle:.6f}, logl_H0={logl_H0:.6f}"
        )

    Hi_eval_null_np = None
    if lmm_mode in (3, 4):
        Hi_eval_null_np = 1.0 / (lambda_null_mle * eigenvalues_np + 1.0)

    return logl_H0, lambda_null_mle, Hi_eval_null_np
