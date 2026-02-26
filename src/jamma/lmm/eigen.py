"""Eigendecomposition of kinship matrix.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.
Uses the LAPACK DSYEVD gufunc with in-place buffer reuse (eigenvectors
overwrite K) to save one n²×8 allocation. Falls back to numpy.linalg.eigh
if the internal gufunc is unavailable.

Thread control is handled by jamma.core.threading via threadpool_limits.

Note: Uses numpy (LAPACK) instead of JAX because JAX's int32 buffer indexing
overflows at ~46k x 46k matrices. With ILP64 numpy (MKL), matrices up to
200k+ are supported.
"""

import time
import warnings

import numpy as np
from loguru import logger
from threadpoolctl import threadpool_info

from jamma.core.memory import (
    check_memory_available,
    estimate_eigendecomp_memory,
    log_memory_snapshot,
)
from jamma.core.threading import blas_threads, get_physical_core_count

try:
    from numpy.linalg import _umath_linalg as _np_linalg_gufuncs

    _eigh_lo = _np_linalg_gufuncs.eigh_lo
except (ImportError, AttributeError):
    _eigh_lo = None
    logger.info(
        "In-place eigendecomp unavailable (numpy %s lacks _umath_linalg.eigh_lo); "
        "will use np.linalg.eigh with separate eigenvector allocation",
        np.__version__,
    )

# Whether in-place eigendecomp is available. Used by memory estimators to
# decide whether to include a separate eigenvector allocation in peak estimates.
INPLACE_EIGEN_AVAILABLE: bool = _eigh_lo is not None

# For matrices >= this size, use sampled symmetry check instead of full np.allclose.
# Full check allocates an n*n temporary; at 100k samples that is ~80GB.
_SAMPLED_SYMMETRY_THRESHOLD = 10_000


def _check_symmetry_sampled(K: np.ndarray, n: int, *, atol: float = 1e-10) -> None:
    """Check kinship symmetry via deterministic strided row sampling.

    Samples every sqrt(n)-th row and compares it against the
    corresponding column. Total work: O(n*sqrt(n)), no n*n temporary.
    Every column is covered at least once, so systematic asymmetry
    (e.g. from non-deterministic BLAS accumulation order) is caught.

    np.linalg.eigh reads only the lower triangle, so asymmetry is
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
            "np.linalg.eigh will use lower triangle only.",
            max_asym,
        )


def _eigh_inplace(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose K in-place, reusing K's buffer for eigenvectors.

    Uses numpy's internal LAPACK gufunc with out= parameter to write
    eigenvectors directly into K's memory buffer. Saves one n²×8 allocation
    (~142GB at 133k samples). Falls back to np.linalg.eigh if the private
    gufunc is unavailable.

    Args:
        K: Symmetric float64 matrix (n, n). OVERWRITTEN with eigenvectors
            on return.

    Returns:
        Tuple of (eigenvalues, eigenvectors) where eigenvectors shares K's buffer.

    Raises:
        TypeError: If K is not float64.
        ValueError: If K is read-only.
    """
    if K.dtype != np.float64:
        raise TypeError(
            f"_eigh_inplace requires float64 input, got {K.dtype}. "
            "Cast with K.astype(np.float64) before eigendecomposition."
        )
    if not K.flags.writeable:
        raise ValueError(
            "K must be writeable for in-place eigendecomposition. "
            "If K is a memory-mapped or read-only array, copy it first: K = K.copy()"
        )

    if _eigh_lo is not None:
        n = K.shape[0]
        eigenvalues = np.empty(n, dtype=np.float64)
        _, eigenvectors = _eigh_lo(K, signature="d->dd", out=(eigenvalues, K))
        if eigenvectors.ctypes.data != K.ctypes.data:
            logger.warning(
                "eigh_lo did not reuse K's buffer; "
                "in-place optimization inactive (numpy %s). "
                "Memory estimates may undercount by %.1fGB.",
                np.__version__,
                n**2 * 8 / 1e9,
            )
        return eigenvalues, eigenvectors
    else:
        logger.warning(
            "_umath_linalg.eigh_lo unavailable (numpy %s); "
            "falling back to np.linalg.eigh — allocates an additional "
            "%.1fGB for eigenvector buffer",
            np.__version__,
            K.shape[0] ** 2 * 8 / 1e9,
        )
        return np.linalg.eigh(K)


def eigendecompose_kinship(
    K: np.ndarray, threshold: float = 1e-10, *, check_memory: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose kinship matrix, zeroing small eigenvalues.

    GEMMA behavior from EigenDecomp_Zeroed:
    - Eigenvalues with |value| < 1e-10 are set to 0
    - Warning if >1 zero eigenvalue
    - Warning if negative eigenvalues remain after thresholding

    Note: Uses numpy (LAPACK) instead of JAX to support matrices larger than
    46k x 46k samples (JAX hits int32 overflow at ~2.1B elements). With ILP64
    numpy (MKL), matrices up to 200k+ are supported.

    Args:
        K: Symmetric kinship matrix (n_samples, n_samples). Overwritten
            with eigenvectors on return (buffer reused to save memory).
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
    """
    n_samples = K.shape[0]
    n_elements = n_samples * n_samples

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kinship matrix must be square, got shape {K.shape}")

    if n_samples < _SAMPLED_SYMMETRY_THRESHOLD:
        # Full check: O(n^2), fine for small matrices (<1s)
        if not np.allclose(K, K.T, atol=1e-10):
            logger.warning(
                "Kinship matrix is not symmetric (max asymmetry: %.2e). "
                "np.linalg.eigh will use lower triangle only.",
                np.max(np.abs(K - K.T)),
            )
    else:
        # Sampled check: O(n*sqrt(n)), avoids n*n temporary allocation.
        _check_symmetry_sampled(K, n_samples, atol=1e-10)

    logger.info(f"Eigendecomposing kinship matrix ({n_samples:,} x {n_samples:,})")
    logger.debug(
        f"Matrix elements: {n_elements:,}, memory: ~{n_elements * 8 / 1e9:.1f} GB"
    )

    # Always log estimated memory requirement (useful even without hard check)
    required_gb = estimate_eigendecomp_memory(n_samples)
    available_gb = log_memory_snapshot(
        f"before_eigendecomp_{n_samples}samples"
    ).available_gb
    logger.info(
        f"Eigendecomp memory: estimated {required_gb:.1f}GB, "
        f"available {available_gb:.1f}GB"
    )

    # Fail fast before LAPACK allocates -- prevents silent SIGKILL from OOM killer
    if check_memory:
        check_memory_available(
            required_gb,
            safety_margin=0.1,
            operation=(
                f"eigendecomposition of {n_samples:,}x{n_samples:,} kinship matrix"
            ),
        )

    # Use all physical cores for BLAS (no JAX contention during eigh)
    n_threads = get_physical_core_count()
    for lib in threadpool_info():
        if lib.get("user_api") == "blas":
            logger.debug(
                f"BLAS: {lib.get('internal_api')}, "
                f"current={lib.get('num_threads')}, target={n_threads}"
            )

    from jamma.core.estimates import estimate_eigendecomp_time

    logger.info(f"Eigendecomp: numpy.linalg.eigh, threads={n_threads}")
    logger.info(f"  Estimated time: {estimate_eigendecomp_time(n_samples, n_threads)}")

    start_time = time.perf_counter()
    try:
        with blas_threads(n_threads):
            eigenvalues, eigenvectors = _eigh_inplace(K)
    except MemoryError:
        logger.error(
            f"MemoryError during eigendecomposition of {n_samples:,}x{n_samples:,} "
            f"matrix. Estimated memory: ~{required_gb:.1f} GB. "
            f"Consider using a machine with more RAM or reducing sample size."
        )
        raise
    except np.linalg.LinAlgError as e:
        logger.error(
            f"Eigendecomposition failed: {e}. "
            f"Kinship matrix may not be positive semi-definite."
        )
        raise
    except Exception as e:
        logger.error(f"Eigendecomposition failed: {type(e).__name__}: {e}")
        raise

    elapsed = time.perf_counter() - start_time
    logger.info(f"Eigendecomposition completed in {elapsed:.2f} seconds")
    log_memory_snapshot(f"after_eigendecomp_{n_samples}samples")

    abs_evals = np.abs(eigenvalues)
    n_negative = int(np.sum(eigenvalues < -threshold))
    if n_negative > 0:
        warnings.warn(
            f"Kinship matrix has {n_negative} negative eigenvalue(s). "
            "Matrix may not be positive semi-definite.",
            stacklevel=2,
        )

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
