"""Eigendecomposition of kinship matrix.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.
Uses jlinalg.eigh which dispatches to vendor DSYEVD/DSYEVR, falling back
to np.linalg.eigh when no vendor LAPACK is available.

Thread control is handled by jamma.core.threading via threadpool_limits.

Note: Uses jlinalg (backed by vendor LAPACK) for eigendecomposition because
large kinship matrices (46k+) require ILP64 LAPACK.
With ILP64 vendor BLAS, matrices up to 200k+ are supported.
"""

import time
import warnings

import numpy as np
from loguru import logger
from threadpoolctl import threadpool_info

from jamma import jlinalg
from jamma.core.eigen_plan import (
    _memory_margin_gb,
    forced_numpy_fallback,
    plan_eigen_driver,
    square_matrix_gb,
)
from jamma.core.memory import check_memory_available
from jamma.core.memory_snapshot import log_memory_snapshot
from jamma.core.progress import timed_progress
from jamma.core.threading import blas_threads, get_blas_thread_count

# For matrices >= this size, use sampled symmetry check instead of full np.allclose.
# Full check allocates an n*n temporary; at 100k samples that is ~80GB.
_SAMPLED_SYMMETRY_THRESHOLD = 5_000

# Symmetry check tolerance; matches LAPACK precision expectations.
_SYMMETRY_ATOL = 1e-11


def _check_symmetry_sampled(
    K: np.ndarray, n: int, *, atol: float = _SYMMETRY_ATOL
) -> None:
    """Check kinship symmetry via deterministic strided row sampling.

    Samples every sqrt(n)-th row and compares it against the
    corresponding column. Total work: O(n*sqrt(n)), no n*n temporary.
    Every column is covered at least once, so systematic asymmetry
    (e.g. from non-deterministic BLAS accumulation order) is caught.

    jlinalg.eigh reads only the lower triangle, so asymmetry is
    harmless — this check is purely diagnostic.

    Args:
        K: Square matrix (n, n).
        n: Matrix dimension (== K.shape[0]).
        atol: Absolute tolerance for element-wise comparison.
    """
    stride = max(1, int(np.sqrt(n)))
    # K[::stride, :] selects sampled rows; K[:, ::stride].T gives corresponding
    # columns transposed to the same shape. Temporary is (n/stride, n) — ~0.5 GB
    # at 100k vs 80 GB for full K - K.T.
    max_asym = float(np.max(np.abs(K[::stride, :] - K[:, ::stride].T)))

    if max_asym > atol:
        logger.warning(
            "Kinship matrix is not symmetric (sampled max asymmetry: %.2e). "
            "jlinalg.eigh will use lower triangle only.",
            max_asym,
        )


def eigendecompose_kinship(
    K: np.ndarray, threshold: float = 1e-10, *, check_memory: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose kinship matrix, zeroing small eigenvalues.

    GEMMA behavior from EigenDecomp_Zeroed:
    - Eigenvalues with |value| < 1e-10 are set to 0
    - Warning if >1 zero eigenvalue
    - Warning if negative eigenvalues remain after thresholding

    Uses jlinalg.eigh (vendor DSYEVD/DSYEVR dispatch, np.linalg.eigh fallback).
    K is consumed (overwritten as scratch) — callers must not reuse K.

    Args:
        K: Symmetric kinship matrix (n_samples, n_samples). Overwritten on exit.
        threshold: Eigenvalues below this are zeroed (default: 1e-10)
        check_memory: If True (default), check available memory before
            eigendecomposition. Set False to skip (e.g., when already checked).

    Returns:
        Tuple of (eigenvalues, eigenvectors) where:
        - eigenvalues: (n_samples,) sorted ascending
        - eigenvectors: (n_samples, n_samples) columns are eigenvectors

    Raises:
        ValueError: If kinship matrix is not square or has invalid shape.
        MemoryError: If matrix is too large to decompose.
        numpy.linalg.LinAlgError: If eigendecomposition fails to converge.
        RuntimeError: If eigendecomposition fails internally or inplace
            mode is unavailable (no vendor DSYEVD/DSYEVR).
    """
    n_samples = K.shape[0]
    n_elements = n_samples * n_samples

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kinship matrix must be square, got shape {K.shape}")

    # Symmetry check
    if n_samples < _SAMPLED_SYMMETRY_THRESHOLD:
        if not np.allclose(K, K.T, atol=_SYMMETRY_ATOL, rtol=0):
            logger.warning(
                "Kinship matrix is not symmetric (max asymmetry: %.2e). "
                "jlinalg.eigh will use lower triangle only.",
                np.max(np.abs(K - K.T)),
            )
    else:
        _check_symmetry_sampled(K, n_samples, atol=_SYMMETRY_ATOL)

    logger.info(f"Eigendecomposing kinship matrix ({n_samples:,} x {n_samples:,})")
    logger.debug(
        f"Matrix elements: {n_elements:,}, "
        f"memory: ~{square_matrix_gb(n_samples):.1f} GB"
    )

    # Memory pre-flight
    available_gb = log_memory_snapshot(
        f"before_eigendecomp_{n_samples}samples"
    ).available_gb

    # Decide eigendecomp driver: inplace DSYEVD > DSYEVD > DSYEVR.
    # Inplace requires vendor DSYEVD and a C-contiguous writeable float64 K
    # (otherwise PyArray_FROM_OTF copies, defeating memory savings).
    # JLINALG_NO_VENDOR_LAPACK forces np.linalg.eigh instead of vendor LAPACK.
    no_vendor_env = forced_numpy_fallback()
    inplace_eligible = (
        K.dtype == np.float64 and K.flags["C_CONTIGUOUS"] and K.flags["WRITEABLE"]
    )
    plan = plan_eigen_driver(
        n_samples,
        available_gb,
        has_dsyevd=bool(jlinalg.blas_has_dsyevd),
        has_dsyevr=bool(jlinalg.blas_has_dsyevr),
        no_vendor=no_vendor_env,
        inplace_eligible=inplace_eligible,
    )
    no_vendor = plan.no_vendor
    if no_vendor and not no_vendor_env:
        logger.info("No vendor LAPACK (DSYEVD/DSYEVR) — using np.linalg.eigh")

    required_gb = plan.required_gb

    # Warn when the chosen DSYEVD peak may exceed available memory and no DSYEVR
    # fallback exists (potential OOM at the real allocation).
    if (
        not no_vendor
        and not plan.use_dsyevr
        and required_gb + _memory_margin_gb(required_gb) > available_gb
    ):
        logger.warning(
            f"DSYEVD peak ({required_gb:.1f}GB) may exceed available memory "
            f"({available_gb:.1f}GB) and DSYEVR is not available. "
            f"Proceeding with {plan.driver}."
        )

    # Only the DSYEVD-not-inplace line names a K-derived reason. Reaching this
    # branch means vendor DSYEVD exists (no_vendor and DSYEVR-only both route
    # elsewhere in plan_eigen_driver), so the K flags are the only reason left.
    inplace_reason = ""
    if not plan.use_inplace and not plan.use_dsyevr and not no_vendor:
        if K.dtype != np.float64:
            inplace_reason = f"K dtype is {K.dtype}, not float64"
        elif not K.flags["C_CONTIGUOUS"]:
            inplace_reason = "K is not C-contiguous"
        else:
            inplace_reason = "kinship not writeable, cannot use inplace"
    logger.info(plan.describe(available_gb, inplace_reason))

    if check_memory:
        check_memory_available(
            required_gb,
            operation=(
                f"eigendecomposition of {n_samples:,}x{n_samples:,} kinship matrix"
            ),
        )

    # Use all physical cores for BLAS
    n_threads = get_blas_thread_count()
    blas_libs = [lib for lib in threadpool_info() if lib.get("user_api") == "blas"]
    if blas_libs:
        active = jlinalg.blas_backend or "numpy-fallback"
        detected = ", ".join(
            f"{lib.get('internal_api')}({lib.get('num_threads')}t)" for lib in blas_libs
        )
        logger.debug(
            f"BLAS active={active}, detected in process: {detected}, "
            f"target={n_threads}t"
        )

    from jamma.core.estimates import (
        estimate_eigendecomp_seconds,
        estimate_eigendecomp_time,
    )

    est_seconds = estimate_eigendecomp_seconds(n_samples, n_threads)
    logger.info(f"Eigendecomp: {plan.driver}, threads={n_threads}")
    logger.info(
        f"  Estimated time: "
        f"{estimate_eigendecomp_time(n_samples, n_threads, use_dsyevr=plan.use_dsyevr)}"
    )

    start_time = time.perf_counter()
    # jlinalg.eigh dispatches to vendor DSYEVD/DSYEVR or the NumPy fallback, and
    # honours JLINALG_NO_VENDOR_LAPACK itself, so one call covers every driver.
    # blas_threads sets the process-global thread count (not thread-local) that
    # governs both vendor and NumPy BLAS, and timed_progress blocks until done.
    try:
        with blas_threads(n_threads):
            eigenvalues, eigenvectors = timed_progress(
                lambda: jlinalg.eigh(K, inplace=plan.use_inplace),
                estimated_seconds=est_seconds,
                desc=f"Eigendecomp {n_samples:,}x{n_samples:,}",
            )
    except MemoryError:
        logger.error(
            f"MemoryError during eigendecomposition of "
            f"{n_samples:,}x{n_samples:,} matrix. "
            f"Estimated memory: ~{required_gb:.1f} GB. "
            f"Consider using a machine with more RAM or reducing sample size."
        )
        raise
    except np.linalg.LinAlgError as e:
        logger.error(
            f"Eigendecomposition convergence failure: {e}. "
            f"Kinship matrix may not be positive semi-definite."
        )
        raise
    except RuntimeError as e:
        logger.error(
            f"Eigendecomposition failed with internal error: {e}. "
            f"This may indicate a jlinalg bug — please report it."
        )
        raise

    elapsed = time.perf_counter() - start_time
    logger.info(f"Eigendecomposition completed in {elapsed:.2f} seconds")
    log_memory_snapshot(f"after_eigendecomp_{n_samples}samples")

    # Threshold small eigenvalues (GEMMA EigenDecomp_Zeroed behavior)
    abs_evals = np.abs(eigenvalues)
    n_negative = int(np.sum(eigenvalues < -threshold))
    if n_negative > 0:
        warnings.warn(
            f"Kinship matrix has {n_negative} negative eigenvalue(s). "
            "Zeroing them (matrix not positive semi-definite).",
            stacklevel=2,
        )
        eigenvalues[eigenvalues < -threshold] = 0.0

    small_mask = abs_evals < threshold
    n_zero = int(np.sum(small_mask))
    eigenvalues[small_mask] = 0.0

    if n_zero > 1:
        warnings.warn(
            f"Kinship matrix has {n_zero} eigenvalues close to zero. "
            "Matrix may be rank-deficient.",
            stacklevel=2,
        )

    return eigenvalues, eigenvectors
