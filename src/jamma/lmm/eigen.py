"""Eigendecomposition of kinship matrix.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.
Uses jlinalg.eigh which dispatches to vendor DSYEVD/DSYEVR or the jlinalg D&C
pipeline depending on available BLAS backends.

Thread control is handled by jamma.core.threading via threadpool_limits.

Note: Uses jlinalg (backed by vendor LAPACK or jlinalg D&C) instead of JAX because
eigendecomposition of large kinship matrices (46k+) requires ILP64 LAPACK.
With ILP64 vendor BLAS, matrices up to 200k+ are supported.
"""

import os
import time
import warnings

import numpy as np
from loguru import logger
from threadpoolctl import threadpool_info

from jamma import jlinalg
from jamma.core.memory import (
    _dsyevd_inplace_peak_gb,
    _dsyevd_peak_gb,
    _dsyevr_peak_gb,
    _memory_margin_gb,
    check_memory_available,
    log_memory_snapshot,
)
from jamma.core.threading import blas_threads, get_physical_core_count

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

    Uses jlinalg.eigh (vendor DSYEVD/DSYEVR dispatch with jlinalg D&C fallback).
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
        f"Matrix elements: {n_elements:,}, memory: ~{n_elements * 8 / 1e9:.1f} GB"
    )

    # Memory pre-flight
    available_gb = log_memory_snapshot(
        f"before_eigendecomp_{n_samples}samples"
    ).available_gb

    # Decide eigendecomp driver: inplace DSYEVD > DSYEVD > DSYEVR.
    # Inplace requires vendor DSYEVD and a C-contiguous writeable float64 K
    # (otherwise PyArray_FROM_OTF copies, defeating memory savings).
    # JLINALG_NO_VENDOR_LAPACK forces the D&C pipeline which cannot do inplace
    # (dsytrd overwrites K with Householder vectors that dormtr needs later).
    dsyevd_peak = _dsyevd_peak_gb(n_samples)
    no_vendor = os.environ.get("JLINALG_NO_VENDOR_LAPACK", "").strip() not in ("", "0")
    use_inplace = (
        not no_vendor
        and bool(jlinalg.blas_has_dsyevd)
        and K.dtype == np.float64
        and K.flags["C_CONTIGUOUS"]
        and K.flags["WRITEABLE"]
    )
    required_gb = _dsyevd_inplace_peak_gb(n_samples) if use_inplace else dsyevd_peak
    use_dsyevr = False

    margin = _memory_margin_gb(required_gb)
    if required_gb + margin > available_gb:
        if jlinalg.blas_has_dsyevr and not no_vendor:
            dsyevd_req = required_gb  # capture before overwrite
            required_gb = _dsyevr_peak_gb(n_samples)
            use_inplace = False
            use_dsyevr = True
            logger.info(
                f"DSYEVD peak ({dsyevd_req:.1f}GB) exceeds available memory "
                f"({available_gb:.1f}GB); using DSYEVR ({required_gb:.1f}GB)"
            )
        else:
            driver = "inplace DSYEVD" if use_inplace else "DSYEVD"
            logger.warning(
                f"DSYEVD peak ({required_gb:.1f}GB) may exceed available memory "
                f"({available_gb:.1f}GB) and DSYEVR is not available. "
                f"Proceeding with {driver}."
            )

    driver = "DSYEVR" if use_dsyevr else ("DSYEVD-inplace" if use_inplace else "DSYEVD")
    dsyevr_gb = _dsyevr_peak_gb(n_samples)
    inplace_gb = _dsyevd_inplace_peak_gb(n_samples)
    if use_inplace:
        logger.info(
            f"Eigendecomp memory (DSYEVD-inplace): estimated {inplace_gb:.1f}GB, "
            f"available {available_gb:.1f}GB "
            f"(kinship in memory, overwriting in place; "
            f"DSYEVR fallback={dsyevr_gb:.1f}GB)"
        )
    elif use_dsyevr:
        logger.info(
            f"Eigendecomp memory (DSYEVR): estimated {dsyevr_gb:.1f}GB, "
            f"available {available_gb:.1f}GB "
            f"(DSYEVD-inplace={inplace_gb:.1f}GB would not fit)"
        )
    else:
        logger.info(
            f"Eigendecomp memory (DSYEVD): estimated {required_gb:.1f}GB, "
            f"available {available_gb:.1f}GB "
            f"(kinship not writeable, cannot use inplace; "
            f"DSYEVR fallback={dsyevr_gb:.1f}GB)"
        )

    if check_memory:
        check_memory_available(
            required_gb,
            safety_margin=0.1,
            operation=(
                f"eigendecomposition of {n_samples:,}x{n_samples:,} kinship matrix"
            ),
        )

    # Use all physical cores for BLAS
    n_threads = get_physical_core_count()
    for lib in threadpool_info():
        if lib.get("user_api") == "blas":
            logger.debug(
                f"BLAS: {lib.get('internal_api')}, "
                f"current={lib.get('num_threads')}, target={n_threads}"
            )

    from jamma.core.estimates import estimate_eigendecomp_time

    logger.info(f"Eigendecomp: {driver}, threads={n_threads}")
    logger.info(
        f"  Estimated time: "
        f"{estimate_eigendecomp_time(n_samples, n_threads, use_dsyevr=use_dsyevr)}"
    )

    start_time = time.perf_counter()
    try:
        with blas_threads(n_threads):
            eigenvalues, eigenvectors = jlinalg.eigh(K, inplace=use_inplace)
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
