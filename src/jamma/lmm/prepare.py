"""Shared setup utilities for LMM association runners.

Provides device selection, covariate matrix construction,
eigendecomposition handling, null model computation, device placement,
and batch lambda optimization used by both the batch and streaming runners.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from loguru import logger

from jamma.core.threading import blas_threads
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
        rep_spec: Sharding for eigenvalues, UtW, Uty, Hi_eval_null — replicated
            on all devices.
    """
    cpu_devices = jax.devices("cpu")
    if len(cpu_devices) <= 1:
        return None, None

    try:
        mesh = Mesh(np.array(cpu_devices), ("snps",))
        snp_spec = NamedSharding(mesh, P(None, "snps"))
        rep_spec = NamedSharding(mesh, P())
        return snp_spec, rep_spec
    except (RuntimeError, ValueError, TypeError) as e:
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


def _compute_null_model(
    lmm_mode: int,
    eigenvalues_np: np.ndarray,
    UtW: np.ndarray,
    Uty: np.ndarray,
    n_cvt: int,
    rep_placement: NamedSharding | jax.Device,
    show_progress: bool,
    l_min: float = 1e-5,
    l_max: float = 1e5,
) -> tuple[float | None, float | None, jnp.ndarray | None]:
    """Compute null model MLE for Score, LRT, and All-tests modes.

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
        rep_placement: JAX device or NamedSharding for Hi_eval placement.
        show_progress: Whether to log results.
        l_min: Minimum lambda for optimization.
        l_max: Maximum lambda for optimization.

    Returns:
        Tuple of (logl_H0, lambda_null_mle, Hi_eval_null_jax).
        All None for Wald (mode 1). For LRT (mode 2), Hi_eval_null_jax
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

    Hi_eval_null_jax = None
    if lmm_mode in (3, 4):
        Hi_eval_null = 1.0 / (lambda_null_mle * eigenvalues_np + 1.0)
        Hi_eval_null_jax = jax.device_put(Hi_eval_null, rep_placement)

    return logl_H0, lambda_null_mle, Hi_eval_null_jax


@dataclass(frozen=True)
class DevicePlacement:
    """Resolved device/sharding placement targets for JAX arrays.

    Encapsulates the common pattern of choosing between NamedSharding
    (multi-device) and single device placement, used identically by
    both the batch, streaming, and LOCO runners.

    Invariants (enforced by __post_init__):
    - snp and rep are always the same type (both NamedSharding or both Device)
    - n_devices > 1 iff placement uses NamedSharding
    - n_devices >= 1
    """

    snp: NamedSharding | jax.Device
    rep: NamedSharding | jax.Device
    n_devices: int

    def __post_init__(self) -> None:
        if self.n_devices < 1:
            raise ValueError(f"n_devices must be >= 1, got {self.n_devices}")
        if type(self.snp) is not type(self.rep):
            raise ValueError(
                f"snp and rep must be the same type, got "
                f"{type(self.snp).__name__} and {type(self.rep).__name__}"
            )
        if isinstance(self.snp, NamedSharding) and self.n_devices == 1:
            raise ValueError("NamedSharding placement requires n_devices > 1")
        if not isinstance(self.snp, NamedSharding) and self.n_devices > 1:
            raise ValueError(
                f"Single-device placement inconsistent with n_devices={self.n_devices}"
            )


def resolve_device_placement(use_gpu: bool) -> DevicePlacement:
    """Set up JAX device selection and sharding placement.

    Combines _select_jax_device and _setup_cpu_sharding into a single call
    that returns resolved placement targets ready for jax.device_put.

    Args:
        use_gpu: Whether to attempt GPU selection.

    Returns:
        DevicePlacement with resolved sharding/device targets.
    """
    device = _select_jax_device(use_gpu)
    snp_spec, rep_spec = _setup_cpu_sharding()
    # When sharding fails, n_devices must be 1 to match the single-device
    # fallback — otherwise padding/alignment logic acts as if multi-device
    # is active while placement targets a single device.
    n_devices = len(jax.devices("cpu")) if snp_spec is not None else 1
    return DevicePlacement(
        snp=snp_spec if snp_spec is not None else device,
        rep=rep_spec if rep_spec is not None else device,
        n_devices=n_devices,
    )


def prepare_utg_chunk(
    geno_chunk: np.ndarray,
    U: np.ndarray,
    chunk_size: int,
    placement: DevicePlacement,
    rotation_threads: int,
) -> tuple[np.ndarray, int]:
    """Impute, pad, rotate, and device-align a genotype chunk for JAX.

    Shared by both the batch and streaming runners. The caller is
    responsible for mean-imputation of missing values before calling this.

    Steps:
    1. Pad to chunk_size if this is a tail chunk (fewer SNPs than chunk_size).
    2. Rotate: UtG = U.T @ geno_chunk (BLAS matmul).
    3. Pad to device-count multiple for even NamedSharding distribution.

    Args:
        geno_chunk: Mean-imputed genotype chunk (n_samples, n_snps_actual).
        U: Eigenvector matrix for rotation (n_samples, n_samples).
        chunk_size: Target chunk width (for tail-chunk padding).
        placement: Resolved device placement (for device-alignment padding).
        rotation_threads: BLAS thread count for U.T @ G rotation.

    Returns:
        Tuple of (UtG_chunk_np, actual_len) where UtG_chunk_np is ready for
        jax.device_put and actual_len is the number of real (non-padded) SNPs.
    """
    actual_len = geno_chunk.shape[1]

    if actual_len < chunk_size:
        pad_width = chunk_size - actual_len
        geno_chunk = np.pad(geno_chunk, ((0, 0), (0, pad_width)), mode="constant")

    with blas_threads(rotation_threads):
        with jax.profiler.TraceAnnotation("dgemm_rotation"):
            UtG_chunk = np.ascontiguousarray(U.T @ geno_chunk)

    # Pad to device-count multiple for even NamedSharding distribution
    n_devices = placement.n_devices
    if n_devices > 1 and UtG_chunk.shape[1] % n_devices != 0:
        dev_pad = n_devices - (UtG_chunk.shape[1] % n_devices)
        UtG_chunk = np.pad(UtG_chunk, ((0, 0), (0, dev_pad)), mode="constant")

    return UtG_chunk, actual_len


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
