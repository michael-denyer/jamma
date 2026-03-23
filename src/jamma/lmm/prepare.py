"""JAX-specific setup utilities for LMM association runners.

Provides device selection, CPU sharding, device placement, and JAX null
model wrapper. Shared backend-agnostic logic (covariate construction,
eigendecomposition, null model core) lives in prepare_common.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from loguru import logger

from jamma import jlinalg
from jamma.core.threading import jlinalg_threads
from jamma.lmm.likelihood_jax import golden_section_optimize_lambda
from jamma.lmm.prepare_common import (
    _build_covariate_matrix,  # noqa: F401 — re-exported for existing callers
    _compute_null_model_common,
    _eigendecompose_or_reuse,  # noqa: F401 — re-exported for existing callers
)


def _setup_cpu_sharding() -> tuple[NamedSharding | None, NamedSharding | None]:
    """Create NamedSharding specs for SNP parallelism across CPU devices.

    Returns (snp_spec, rep_spec) or (None, None) if only 1 CPU device
    or if sharding setup fails.

    Returns:
        snp_spec: Sharding for utg_t (n_snps, n_samples) — shard on SNP axis.
        rep_spec: Sharding for eigenvalues, UtW, Uty, Hi_eval_null — replicated
            on all devices.
    """
    cpu_devices = jax.devices("cpu")
    if len(cpu_devices) <= 1:
        return None, None

    try:
        mesh = Mesh(np.array(cpu_devices), ("snps",))
        snp_spec = NamedSharding(mesh, P("snps", None))
        rep_spec = NamedSharding(mesh, P())
        return snp_spec, rep_spec
    except (RuntimeError, TypeError, ValueError) as e:
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

    Thin JAX wrapper around _compute_null_model_common. Calls the pure-NumPy
    computation and places Hi_eval_null on the specified JAX device.

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
    logl_H0, lambda_null_mle, Hi_eval_null_np = _compute_null_model_common(
        lmm_mode, eigenvalues_np, UtW, Uty, n_cvt, show_progress, l_min, l_max
    )
    if Hi_eval_null_np is not None:
        Hi_eval_null_jax = jax.device_put(Hi_eval_null_np, rep_placement)
    else:
        Hi_eval_null_jax = None
    return logl_H0, lambda_null_mle, Hi_eval_null_jax


@dataclass(frozen=True)
class DevicePlacement:
    """Resolved device/sharding placement targets for JAX arrays.

    Encapsulates the common pattern of choosing between NamedSharding
    (multi-device) and single device placement, used identically by
    the batch, streaming, and LOCO runners.

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
    placement: DevicePlacement,
    rotation_threads: int,
) -> tuple[np.ndarray, int]:
    """Rotate and device-align a genotype chunk for JAX.

    Shared by both the batch and streaming runners. The caller is
    responsible for mean-imputation of missing values before calling this.

    Steps:
    1. Rotate: utg_t = geno_chunk.T @ U via jlinalg.dgemm(geno_chunk, U, transa='T').
    2. Pad to device-count multiple for even NamedSharding distribution.

    Tail chunks (fewer SNPs than a full chunk) pass their actual width
    directly — no padding to chunk_size. JAX JIT traces once for the tail
    shape; the recompilation cost is negligible vs wasted BLAS on zeros.

    Args:
        geno_chunk: Mean-imputed genotype chunk (n_samples, n_snps_actual).
        U: Eigenvector matrix for rotation (n_samples, n_samples).
        placement: Resolved device placement (for device-alignment padding).
        rotation_threads: jlinalg thread count for rotation.

    Returns:
        Tuple of (utg_t_chunk, actual_len) where utg_t_chunk is
        (n_snps, n_samples) C-contiguous and actual_len is the number
        of real (non-padded) SNPs.
    """
    actual_len = geno_chunk.shape[1]

    # RUN-02: Tail chunks process actual SNP count — no padding to chunk_size.
    # JAX JIT traces once for the tail shape; cost is negligible vs saved BLAS.
    # Device-count alignment (below) still pads if needed for even shard distribution.

    with jlinalg_threads(rotation_threads):
        with jax.profiler.TraceAnnotation("dgemm_rotation"):
            utg_t_chunk = jlinalg.dgemm(geno_chunk, U, transa="T")

    # Pad SNP axis (axis 0) to device-count multiple for even NamedSharding distribution
    n_devices = placement.n_devices
    if n_devices > 1 and utg_t_chunk.shape[0] % n_devices != 0:
        dev_pad = n_devices - (utg_t_chunk.shape[0] % n_devices)
        utg_t_chunk = np.pad(utg_t_chunk, ((0, dev_pad), (0, 0)), mode="constant")

    return utg_t_chunk, actual_len


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
