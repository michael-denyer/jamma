"""Shared setup utilities for LMM association runners.

Provides device selection, covariate matrix construction,
eigendecomposition handling, null model computation, and
batch lambda optimization used by both the batch and streaming runners.
"""

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import compute_null_model_mle
from jamma.lmm.likelihood_jax import golden_section_optimize_lambda
from jamma.utils.logging import log_rss_memory


def _select_jax_device(use_gpu: bool) -> jax.Device:
    """Select JAX compute device with safe GPU detection.

    Falls back to CPU if GPU backend is unavailable.

    Args:
        use_gpu: Whether to attempt GPU selection.

    Returns:
        JAX device to use for computation.
    """
    device = jax.devices("cpu")[0]
    if use_gpu:
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                device = gpu_devices[0]
        except RuntimeError:
            pass
    return device


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
    return W, W.shape[1]


def _eigendecompose_or_reuse(
    kinship: np.ndarray | None,
    eigenvalues: np.ndarray | None,
    eigenvectors: np.ndarray | None,
    show_progress: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigendecomposition, computing it if not provided.

    Args:
        kinship: Kinship matrix (n_samples, n_samples), or None when
            pre-computed eigenvalues and eigenvectors are provided.
        eigenvalues: Pre-computed eigenvalues or None.
        eigenvectors: Pre-computed eigenvectors or None.
        show_progress: Whether to log memory usage.
        label: Label for memory logging (e.g. "lmm_jax", "lmm_streaming").

    Returns:
        Tuple of (eigenvalues, eigenvectors).
    """
    if eigenvalues is not None and eigenvectors is not None:
        if show_progress:
            logger.debug("Using pre-computed eigendecomposition")
        return eigenvalues, eigenvectors

    if show_progress:
        log_rss_memory(label, "before_eigendecomp")
    eigenvalues_np, U = eigendecompose_kinship(kinship)
    if show_progress:
        log_rss_memory(label, "after_eigendecomp")
    return eigenvalues_np, U


def _compute_null_model(
    lmm_mode: int,
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    device: jax.Device,
    show_progress: bool,
) -> tuple[float | None, float | None, jnp.ndarray | None]:
    """Compute null model MLE for Score, LRT, and All-tests modes.

    Score test (mode 3) and All-tests (mode 4) additionally precompute Hi_eval
    at the null lambda. Wald (mode 1) skips this entirely.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        eigenvalues_np: Kinship eigenvalues as numpy array.
        UtW: Rotated covariates.
        Uty: Rotated phenotype.
        n_cvt: Number of covariates.
        device: JAX device for Hi_eval placement.
        show_progress: Whether to log results.

    Returns:
        Tuple of (logl_H0, lambda_null_mle, Hi_eval_null_jax).
        All None for Wald mode.
    """
    if lmm_mode not in (2, 3, 4):
        return None, None, None

    lambda_null_mle, logl_H0 = compute_null_model_mle(eigenvalues_np, UtW, Uty, n_cvt)
    if show_progress:
        logger.info(
            f"Null model MLE: lambda={lambda_null_mle:.6f}, logl_H0={logl_H0:.6f}"
        )

    Hi_eval_null_jax = None
    if lmm_mode in (3, 4):
        Hi_eval_null = 1.0 / (lambda_null_mle * eigenvalues_np + 1.0)
        Hi_eval_null_jax = jax.device_put(Hi_eval_null, device)

    return logl_H0, lambda_null_mle, Hi_eval_null_jax


def _grid_optimize_lambda_batched(
    n_cvt: int,
    eigenvalues: jnp.ndarray,
    Uab_batch: jnp.ndarray,
    Iab_batch: jnp.ndarray,
    l_min: float,
    l_max: float,
    n_grid: int,
    n_refine: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Batch lambda optimization using grid search + golden section refinement.

    Delegates to golden_section_optimize_lambda with precomputed Iab and at
    least 20 iterations to achieve ~1e-5 relative tolerance.

    Args:
        n_cvt: Number of covariates.
        eigenvalues: Eigenvalues (n_samples,)
        Uab_batch: Uab matrices (n_snps, n_samples, n_index)
        Iab_batch: Precomputed identity-weighted Pab (n_snps, n_cvt+2, n_index)
        l_min, l_max: Lambda bounds
        n_grid: Coarse grid points
        n_refine: Golden section iterations
    """
    return golden_section_optimize_lambda(
        n_cvt,
        eigenvalues,
        Uab_batch,
        Iab_batch,
        l_min=l_min,
        l_max=l_max,
        n_grid=n_grid,
        n_iter=max(n_refine, 20),
    )
