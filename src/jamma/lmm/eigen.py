"""Eigendecomposition of kinship matrix.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.
Uses scipy.linalg.eigh (LAPACK) to support large matrices (200k+ samples) that
exceed JAX's int32 buffer limits.

Note on threading: OpenBLAS can segfault with multi-threaded eigendecomposition
on large matrices (>50k) due to memory allocation races. We detect the BLAS
backend at runtime and only limit threads for OpenBLAS (MKL is stable).
See: https://github.com/scipy/scipy/issues/8741
"""

import time
import warnings

import numpy as np
from loguru import logger
from scipy import linalg

from jamma.core.backend import get_compute_backend

try:
    from threadpoolctl import threadpool_info, threadpool_limits

    HAVE_THREADPOOLCTL = True
except ImportError:
    HAVE_THREADPOOLCTL = False

from jamma.core.memory import (
    check_memory_available,
    estimate_eigendecomp_memory,
    log_memory_snapshot,
)

# Threshold above which we limit BLAS threads for OpenBLAS to prevent SIGSEGV
# 50k samples = 2.5B elements, empirically safe with threading
# 100k samples = 10B elements, crashes with OpenBLAS multi-threading
THREAD_LIMIT_THRESHOLD = 50_000

# Thread cap for large matrices with OpenBLAS (not 1, which kills performance)
# 4-8 threads is a reasonable balance between speed and stability
OPENBLAS_THREAD_CAP = 4


def _get_blas_backend() -> str | None:
    """Detect the BLAS backend in use (openblas, mkl, blis, etc).

    Returns:
        Backend name in lowercase, or None if detection fails.
    """
    if not HAVE_THREADPOOLCTL:
        return None

    try:
        info = threadpool_info()
        for lib in info:
            # Look for BLAS libraries
            if lib.get("user_api") == "blas":
                # internal_api gives the backend name
                backend = lib.get("internal_api", "").lower()
                if backend:
                    return backend
        return None
    except Exception:
        return None


def _should_limit_threads(n_samples: int, backend: str | None) -> tuple[bool, int]:
    """Determine if we should limit BLAS threads and to what value.

    Args:
        n_samples: Number of samples in the matrix.
        backend: BLAS backend name (openblas, mkl, etc).

    Returns:
        Tuple of (should_limit, thread_limit).
    """
    if n_samples < THREAD_LIMIT_THRESHOLD:
        return False, 0

    # MKL and BLIS are stable with multi-threading
    if backend in ("mkl", "blis"):
        return False, 0

    # OpenBLAS needs thread limiting for large matrices
    # Use a cap (not 1) to preserve some parallelism
    if backend == "openblas":
        return True, OPENBLAS_THREAD_CAP

    # Unknown backend - be conservative and limit threads
    # This includes None (detection failed) and any other backend
    return True, OPENBLAS_THREAD_CAP


def eigendecompose_kinship(
    K: np.ndarray, threshold: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose kinship matrix, zeroing small eigenvalues.

    GEMMA behavior from EigenDecomp_Zeroed:
    - Eigenvalues < 1e-10 are set to 0
    - Warning if >1 zero eigenvalue
    - Warning if negative eigenvalues remain after thresholding

    Note: Uses scipy.linalg.eigh (LAPACK) instead of JAX to support matrices
    larger than 46k x 46k samples (JAX hits int32 overflow at ~2.1B elements).

    Args:
        K: Symmetric kinship matrix (n_samples, n_samples)
        threshold: Eigenvalues below this are zeroed (default: 1e-10)

    Returns:
        Tuple of (eigenvalues, eigenvectors) where:
        - eigenvalues: (n_samples,) sorted ascending
        - eigenvectors: (n_samples, n_samples) columns are eigenvectors

    Raises:
        ValueError: If kinship matrix is not square or has invalid shape.
        MemoryError: If matrix is too large to decompose.
    """
    # Backend dispatch: Rust backend doesn't have pre-flight memory check
    # or thread limiting (faer handles this internally)
    backend = get_compute_backend()
    if backend == "rust":
        try:
            from jamma_core import eigendecompose_kinship as rust_eigendecompose

            n = K.shape[0]
            logger.info(
                f"## Eigendecomposing kinship matrix ({n:,} x {n:,}) [Rust/faer]"
            )
            start_time = time.perf_counter()
            eigenvalues, eigenvectors = rust_eigendecompose(K, threshold)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Eigendecomposition completed in {elapsed:.2f} seconds")
            return eigenvalues, eigenvectors
        except ImportError:
            logger.warning(
                "Rust backend selected but jamma_core not installed, "
                "falling back to scipy"
            )
            # Fall through to scipy path

    n_samples = K.shape[0]
    n_elements = n_samples * n_samples

    # Validate input
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kinship matrix must be square, got shape {K.shape}")

    logger.info(f"## Eigendecomposing kinship matrix ({n_samples:,} x {n_samples:,})")
    logger.debug(
        f"Matrix elements: {n_elements:,}, memory: ~{n_elements * 8 / 1e9:.1f} GB"
    )

    # Pre-flight memory check - fail fast before scipy allocates
    # This prevents silent OOM crashes on Linux where OOM killer sends SIGKILL
    required_gb = estimate_eigendecomp_memory(n_samples)
    check_memory_available(
        required_gb,
        safety_margin=0.1,
        operation=f"eigendecomposition of {n_samples:,}x{n_samples:,} kinship matrix",
    )

    # Use scipy.linalg.eigh which uses LAPACK with int64 indexing
    # This supports matrices up to sqrt(int64_max) ≈ 3 billion rows
    # JAX's jnp.linalg.eigh hits int32 overflow at ~46k x 46k (2.1B elements)
    #
    # Log memory state right before allocation for debugging OOM crashes
    log_memory_snapshot(f"before_eigendecomp_{n_samples}samples")

    # Detect BLAS backend and determine threading strategy
    # MKL is stable; OpenBLAS needs thread limiting for large matrices
    backend = _get_blas_backend()
    should_limit, thread_cap = _should_limit_threads(n_samples, backend)

    if backend:
        logger.debug(f"BLAS backend: {backend}")

    if should_limit:
        if HAVE_THREADPOOLCTL:
            logger.info(
                f"Limiting BLAS to {thread_cap} threads for "
                f"{n_samples:,}x{n_samples:,} matrix (backend={backend})"
            )
        else:
            logger.warning(
                f"threadpoolctl not installed - cannot limit BLAS threads for "
                f"{n_samples:,}x{n_samples:,} matrix. "
                "Install with: pip install threadpoolctl"
            )

    start_time = time.perf_counter()
    try:
        # Performance options:
        # - overwrite_a=True: reuse K's memory (saves ~80GB at 100k samples)
        # - check_finite=False: skip NaN/Inf check (we validated input)
        eigh_kwargs = {"overwrite_a": True, "check_finite": False}

        if should_limit and HAVE_THREADPOOLCTL:
            with threadpool_limits(limits=thread_cap, user_api="blas"):
                eigenvalues, eigenvectors = linalg.eigh(K, **eigh_kwargs)
        else:
            eigenvalues, eigenvectors = linalg.eigh(K, **eigh_kwargs)
    except MemoryError:
        mem_gb = n_elements * 8 * 3 / 1e9
        logger.error(
            f"MemoryError during eigendecomposition of {n_samples:,}x{n_samples:,} "
            f"matrix. Required memory: ~{mem_gb:.1f} GB. "
            f"Consider using a machine with more RAM or reducing sample size."
        )
        raise
    except Exception as e:
        logger.error(f"Eigendecomposition failed: {type(e).__name__}: {e}")
        raise

    elapsed = time.perf_counter() - start_time
    logger.info(f"Eigendecomposition completed in {elapsed:.2f} seconds")

    # Log memory after eigendecomp - shows peak allocation
    log_memory_snapshot(f"after_eigendecomp_{n_samples}samples")

    # Count negative eigenvalues before thresholding
    n_negative = np.sum(eigenvalues < -threshold)
    if n_negative > 0:
        warnings.warn(
            f"Kinship matrix has {n_negative} negative eigenvalue(s). "
            "Matrix may not be positive semi-definite.",
            stacklevel=2,
        )

    # Zero small eigenvalues (GEMMA's behavior)
    eigenvalues = np.where(np.abs(eigenvalues) < threshold, 0.0, eigenvalues)

    # Count zero eigenvalues after thresholding
    n_zero = np.sum(eigenvalues == 0.0)
    if n_zero > 1:
        warnings.warn(
            f"Kinship matrix has {n_zero} eigenvalues close to zero. "
            "Matrix may be rank-deficient.",
            stacklevel=2,
        )

    return eigenvalues, eigenvectors
