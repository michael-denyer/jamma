"""Shared setup utilities for LMM association runners.

Provides device selection, covariate matrix construction,
eigendecomposition handling, null model computation, and
batch lambda optimization used by both the batch and streaming runners.
"""

import gc

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from loguru import logger

from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import compute_null_model_mle
from jamma.lmm.likelihood_jax import golden_section_optimize_lambda
from jamma.utils.logging import log_rss_memory


def _setup_cpu_sharding() -> tuple[NamedSharding | None, NamedSharding | None]:
    """Create NamedSharding specs for SNP parallelism across CPU devices.

    Returns (snp_spec, rep_spec) or (None, None) if only 1 CPU device
    or if sharding setup fails.

    Returns:
        snp_spec: Sharding for UtG (n_samples, n_snps) — shard on SNP axis.
        rep_spec: Sharding for eigenvalues, UtW, Uty — replicated on all devices.
    """
    cpu_devices = jax.devices("cpu")
    if len(cpu_devices) <= 1:
        return None, None

    try:
        mesh = Mesh(np.array(cpu_devices), ("snps",))
        snp_spec = NamedSharding(mesh, P(None, "snps"))
        rep_spec = NamedSharding(mesh, P())
        return snp_spec, rep_spec
    except Exception as e:
        logger.warning(
            f"Failed to create CPU sharding mesh with {len(cpu_devices)} devices: "
            f"{type(e).__name__}: {e}. Falling back to single-device mode. "
            "Set JAMMA_JAX_DEVICES=1 to suppress this warning."
        )
        return None, None


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
        except RuntimeError as e:
            logger.warning(f"GPU requested but not available, falling back to CPU: {e}")
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
    n_cvt = W.shape[1]
    # df = n_samples - n_cvt - 1 must be positive for valid REML
    if n_samples <= n_cvt + 1:
        raise ValueError(
            f"Over-parameterized model: {n_samples} samples with {n_cvt} "
            f"covariates leaves df={n_samples - n_cvt - 1} "
            f"(need at least {n_cvt + 2} samples)"
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


def _compute_null_model(
    lmm_mode: int,
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    device: jax.Device,
    show_progress: bool,
    l_min: float = 1e-5,
    l_max: float = 1e5,
    rep_spec: NamedSharding | None = None,
) -> tuple[float | None, float | None, jnp.ndarray | None]:
    """Compute null model MLE for Score, LRT, and All-tests modes.

    GEMMA computes both REML and MLE null lambdas in CalcLambda, but uses
    MLE lambda for Hi_eval in the Score test. This matches GEMMA's behavior:
    Hi_eval_null = 1 / (lambda_null_mle * eigenvalues + 1).

    Score test (mode 3) and All-tests (mode 4) additionally precompute Hi_eval
    at the null lambda. Wald (mode 1) skips this entirely.

    Args:
        lmm_mode: Test type (1=Wald, 2=LRT, 3=Score, 4=All).
        eigenvalues_np: Kinship eigenvalues as numpy array.
        UtW: Rotated covariates.
        Uty: Rotated phenotype.
        n_cvt: Number of covariates.
        device: JAX device for Hi_eval placement (used when rep_spec is None).
        show_progress: Whether to log results.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.
        rep_spec: NamedSharding replication spec for multi-device placement.
            When provided, Hi_eval_null is replicated across all devices.
            When None, Hi_eval_null is placed on the single device.

    Returns:
        Tuple of (logl_H0, lambda_null_mle, Hi_eval_null_jax).
        All None for Wald mode.
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

    Hi_eval_null_jax = None
    if lmm_mode in (3, 4):
        Hi_eval_null = 1.0 / (lambda_null_mle * eigenvalues_np + 1.0)
        if rep_spec is not None:
            Hi_eval_null_jax = jax.device_put(Hi_eval_null, rep_spec)
        else:
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
