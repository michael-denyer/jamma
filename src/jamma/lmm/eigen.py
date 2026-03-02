"""Eigendecomposition of kinship matrix.

Provides GEMMA-compatible eigendecomposition with small eigenvalue thresholding.
Preferred path: DSYEVR C extension (_eigen_accel) for O(N) workspace (MRRR algorithm).
Fallback: numpy's internal ``eigh_lo`` gufunc (backed by LAPACK DSYEVD) with
in-place buffer reuse (eigenvectors overwrite K) to save one n²×8 allocation.
Final fallback: ``numpy.linalg.eigh`` if the internal gufunc is unavailable.

Thread control is handled by jamma.core.threading via threadpool_limits.

Note: Uses numpy (LAPACK) instead of JAX because JAX's int32 buffer indexing
overflows at ~46k x 46k matrices. With ILP64 numpy (MKL), matrices up to
200k+ are supported.
"""

import sys
import time
import warnings

import numpy as np
from loguru import logger
from threadpoolctl import threadpool_info

from jamma.core.memory import (
    _dsyevd_peak_gb,
    _dsyevr_peak_gb,
    check_memory_available,
    log_memory_snapshot,
)
from jamma.core.threading import blas_threads, get_physical_core_count

try:
    from numpy.linalg import _umath_linalg as _np_linalg_gufuncs

    _eigh_lo = _np_linalg_gufuncs.eigh_lo
except (ImportError, AttributeError):
    _eigh_lo = None
    logger.warning(
        "In-place eigendecomp unavailable (numpy %s lacks _umath_linalg.eigh_lo); "
        "will use np.linalg.eigh with separate eigenvector allocation",
        np.__version__,
    )

# Whether in-place eigendecomp is available. Used by memory estimators to
# decide whether to include a separate eigenvector allocation in peak estimates.
INPLACE_EIGEN_AVAILABLE: bool = _eigh_lo is not None

_EXPECTED_EIGEN_ABI = 2  # Must match ABI_VERSION in _eigen_accel.c


def _try_import_dsyevr() -> tuple[bool, object | None]:
    """Attempt to import the DSYEVR C extension and validate ABI."""
    try:
        from jamma.lmm._eigen_accel import ABI_VERSION as abi
        from jamma.lmm._eigen_accel import eigh_dsyevr
    except ImportError as e:
        logger.debug(
            f"_eigen_accel not importable: {e}. "
            "Expected if the extension has not been compiled."
        )
        return False, None
    except AttributeError as e:
        logger.warning(
            f"_eigen_accel loaded but missing attribute: {e}. "
            "Stale .so may need recompilation."
        )
        return False, None

    if abi != _EXPECTED_EIGEN_ABI:
        logger.warning(
            f"_eigen_accel ABI mismatch: extension has ABI_VERSION={abi}, "
            f"expected {_EXPECTED_EIGEN_ABI}. Extension will not be used. "
            f"Recompile with: python -m jamma.lmm._compile_eigen"
        )
        return False, None

    return True, eigh_dsyevr


def _auto_recompile_eigen() -> bool:
    """Auto-recompile _eigen_accel and reimport."""
    try:
        from jamma.lmm._compile_eigen import compile_extension
    except ImportError:
        logger.debug(
            "_compile_eigen module not available — auto-recompilation not possible"
        )
        return False

    logger.info(
        "_eigen_accel needs recompilation (ABI mismatch or missing). Compiling..."
    )

    if not compile_extension(verbose=False):
        logger.warning(
            "Auto-recompilation of _eigen_accel failed. "
            "Falling back to DSYEVD. "
            "To diagnose: python -m jamma.lmm._compile_eigen"
        )
        return False

    sys.modules.pop("jamma.lmm._eigen_accel", None)
    logger.info("_eigen_accel recompiled successfully.")
    return True


# Import-time probe: fast, read-only — no compilation side effects.
_DSYEVR_AVAILABLE, _eigh_dsyevr_func = _try_import_dsyevr()
_DSYEVR_RECOMPILE_ATTEMPTED = False

if not _DSYEVR_AVAILABLE:
    logger.debug(
        "DSYEVR C extension unavailable — using DSYEVD (numpy.linalg.eigh). "
        "DSYEVR saves ~250GB workspace at 125k samples. "
        "To compile: python -m jamma.lmm._compile_eigen"
    )


def _lazy_init_dsyevr() -> None:
    """One-shot lazy init: attempt auto-recompile if import-time probe failed.

    Called on first eigendecompose_kinship() invocation only.
    Deferred from module import to avoid surprising subprocess/compiler
    side effects during import.
    """
    global _DSYEVR_AVAILABLE, _eigh_dsyevr_func, _DSYEVR_RECOMPILE_ATTEMPTED
    if _DSYEVR_AVAILABLE or _DSYEVR_RECOMPILE_ATTEMPTED:
        return
    _DSYEVR_RECOMPILE_ATTEMPTED = True
    if _auto_recompile_eigen():
        _DSYEVR_AVAILABLE, _eigh_dsyevr_func = _try_import_dsyevr()


# Whether DSYEVR path is available. Used by eigendecompose_kinship()
# to decide between DSYEVD and DSYEVR at runtime based on available memory.

# For matrices >= this size, use sampled symmetry check instead of full np.allclose.
# Full check allocates an n*n temporary; at 100k samples that is ~80GB.
_SAMPLED_SYMMETRY_THRESHOLD = 10_000

# Symmetry check tolerance; matches LAPACK precision expectations.
_SYMMETRY_ATOL = 1e-11


def _check_symmetry_sampled(K: np.ndarray, n: int, *, atol: float = 1e-11) -> None:
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


def _eigh_dsyevr(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose K via DSYEVR (MRRR algorithm, O(N) workspace).

    Requires _eigen_accel C extension. Input K is destroyed on exit.

    Args:
        K: Symmetric float64 matrix (n, n). Overwritten on exit.

    Returns:
        Tuple of (eigenvalues, eigenvectors) with eigenvalues ascending.
        Eigenvectors are F-order (LAPACK native).
    """
    if _eigh_dsyevr_func is None:
        raise RuntimeError(
            "DSYEVR C extension (_eigen_accel) is not available. "
            "This is a bug — _eigh_dsyevr should not be called when "
            "_DSYEVR_AVAILABLE is False. "
            "To compile: python -m jamma.lmm._compile_eigen"
        )
    return _eigh_dsyevr_func(K)


def _eigh_inplace(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose K, reusing K's buffer for eigenvectors when possible.

    When the ``eigh_lo`` gufunc is available, writes eigenvectors directly
    into K's memory buffer via the ``out=`` parameter, saving one n²×8
    allocation (~142GB at 133k samples). Falls back to ``np.linalg.eigh``
    (separate allocation) if the gufunc is unavailable.

    Callers should treat K as consumed after this call regardless of path.

    Args:
        K: Symmetric float64 matrix (n, n). May be overwritten with
            eigenvectors on return (when gufunc is available).

    Returns:
        Tuple of (eigenvalues, eigenvectors). Eigenvectors shares K's
        buffer when in-place path succeeds; separate allocation on fallback.

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
            "K must be writeable for eigendecomposition. "
            "If K is a memory-mapped or read-only array, copy it first: K = K.copy()"
        )

    if _eigh_lo is None:
        logger.warning(
            "_umath_linalg.eigh_lo unavailable (numpy %s); "
            "falling back to np.linalg.eigh — allocates an additional "
            "%.1fGB for eigenvector buffer",
            np.__version__,
            K.shape[0] ** 2 * 8 / 1e9,
        )
        return np.linalg.eigh(K)

    # One-way latch (True -> False only): once a buffer mismatch is detected,
    # conservatively assume all subsequent calls will also mismatch.
    # Importers must not cache this value — _inplace_eigen_available() reads the
    # module attribute each call to see runtime updates.
    global INPLACE_EIGEN_AVAILABLE

    n = K.shape[0]
    eigenvalues = np.empty(n, dtype=np.float64)
    _, eigenvectors = _eigh_lo(K, signature="d->dd", out=(eigenvalues, K))
    if eigenvectors.ctypes.data != K.ctypes.data and INPLACE_EIGEN_AVAILABLE:
        # No lock needed: False is the conservative direction, so races
        # between LOCO calls are benign. Guard on INPLACE_EIGEN_AVAILABLE
        # so only the first mismatch warns.
        INPLACE_EIGEN_AVAILABLE = False
        logger.warning(
            "eigh_lo did not reuse K's buffer; "
            "in-place optimization inactive (numpy %s). "
            "Memory estimates corrected to include separate "
            "eigenvector allocation (%.1fGB).",
            np.__version__,
            n**2 * 8 / 1e9,
        )
    return eigenvalues, eigenvectors


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

    Under memory pressure, may use DSYEVR (O(N) workspace) instead of
    DSYEVD. In this case, K is destroyed but eigenvectors are in a
    separate allocation (not reused from K's buffer).

    Args:
        K: Symmetric kinship matrix (n_samples, n_samples). May be
            overwritten with eigenvectors on return (buffer reused when
            gufunc available). Treat K as consumed after this call.
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
    # Lazy auto-recompile: deferred from module import to first use to avoid
    # surprising subprocess/compiler side effects during import.
    _lazy_init_dsyevr()

    n_samples = K.shape[0]
    n_elements = n_samples * n_samples

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kinship matrix must be square, got shape {K.shape}")

    if n_samples < _SAMPLED_SYMMETRY_THRESHOLD:
        # Full check: O(n^2), fine for small matrices (<1s).
        # rtol=0 ensures only atol governs the check, consistent with
        # _check_symmetry_sampled which uses max_asym > atol directly.
        if not np.allclose(K, K.T, atol=_SYMMETRY_ATOL, rtol=0):
            logger.warning(
                "Kinship matrix is not symmetric (max asymmetry: %.2e). "
                "np.linalg.eigh will use lower triangle only.",
                np.max(np.abs(K - K.T)),
            )
    else:
        # Sampled check: O(n*sqrt(n)), avoids n*n temporary allocation.
        _check_symmetry_sampled(K, n_samples, atol=_SYMMETRY_ATOL)

    logger.info(f"Eigendecomposing kinship matrix ({n_samples:,} x {n_samples:,})")
    logger.debug(
        f"Matrix elements: {n_elements:,}, memory: ~{n_elements * 8 / 1e9:.1f} GB"
    )

    # Memory-aware driver selection: prefer DSYEVD (faster), fall back to
    # DSYEVR (O(N) workspace) only when DSYEVD won't fit in available memory.
    # This runs regardless of check_memory — check_memory only gates the
    # MemoryError raise, not the driver choice.
    available_gb = log_memory_snapshot(
        f"before_eigendecomp_{n_samples}samples"
    ).available_gb

    use_dsyevr = False
    dsyevd_peak = _dsyevd_peak_gb(n_samples)
    required_gb = dsyevd_peak

    if _DSYEVR_AVAILABLE:
        margin_gb = min(dsyevd_peak * 0.1, 10.0)
        if dsyevd_peak + margin_gb > available_gb:
            # DSYEVD won't fit — check DSYEVR
            dsyevr_peak = _dsyevr_peak_gb(n_samples)
            dsyevr_margin = min(dsyevr_peak * 0.1, 10.0)
            if dsyevr_peak + dsyevr_margin <= available_gb:
                use_dsyevr = True
                required_gb = dsyevr_peak
                logger.info(
                    f"DSYEVD peak ({dsyevd_peak:.1f}GB) exceeds "
                    f"available memory ({available_gb:.1f}GB). "
                    f"Using DSYEVR ({dsyevr_peak:.1f}GB)."
                )
            else:
                # Neither fits — use DSYEVR anyway (smaller peak)
                use_dsyevr = True
                required_gb = dsyevr_peak
                logger.warning(
                    f"Neither DSYEVD ({dsyevd_peak:.1f}GB) nor DSYEVR "
                    f"({dsyevr_peak:.1f}GB) fits in available memory "
                    f"({available_gb:.1f}GB). Using DSYEVR (smaller peak). "
                    f"OOM risk is high."
                )
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

    driver = "DSYEVR via _eigen_accel" if use_dsyevr else "DSYEVD via numpy.linalg.eigh"
    logger.info(f"Eigendecomp: {driver}, threads={n_threads}")
    logger.info(f"  Estimated time: {estimate_eigendecomp_time(n_samples, n_threads)}")

    start_time = time.perf_counter()
    try:
        with blas_threads(n_threads):
            if use_dsyevr:
                eigenvalues, eigenvectors = _eigh_dsyevr(K)
            else:
                eigenvalues, eigenvectors = _eigh_inplace(K)
    except MemoryError:
        if not use_dsyevr and _DSYEVR_AVAILABLE:
            # DSYEVD workspace allocation failed before computation started.
            # K is still valid: MemoryError from malloc precedes LAPACK modifying K.
            # Attempt DSYEVR which has O(N) workspace instead of O(N^2).
            logger.warning(
                f"DSYEVD MemoryError for {n_samples:,}x{n_samples:,} matrix "
                f"(estimated {required_gb:.1f}GB). "
                f"Attempting DSYEVR fallback (O(N) workspace)."
            )
            use_dsyevr = True
            driver = "DSYEVR via _eigen_accel (fallback from DSYEVD MemoryError)"
            try:
                with blas_threads(n_threads):
                    eigenvalues, eigenvectors = _eigh_dsyevr(K)
            except (MemoryError, Exception) as fallback_err:
                logger.error(
                    f"DSYEVR fallback also failed: "
                    f"{type(fallback_err).__name__}: {fallback_err}. "
                    f"Consider using a machine with more RAM or reducing sample size."
                )
                raise fallback_err from None
        else:
            logger.error(
                f"MemoryError during eigendecomposition of "
                f"{n_samples:,}x{n_samples:,} matrix. "
                f"Estimated memory: ~{required_gb:.1f} GB. "
                f"Consider using a machine with more RAM or reducing sample size."
            )
            raise
    except np.linalg.LinAlgError as e:
        logger.error(
            f"Eigendecomposition failed: {e}. "
            f"Kinship matrix may not be positive semi-definite."
        )
        raise
    except (ValueError, RuntimeError) as e:
        msg = f"Eigendecomposition failed ({driver}): {e}."
        if use_dsyevr:
            msg += (
                " If this persists, remove the C extension"
                " .so to force DSYEVD fallback."
            )
        logger.error(msg)
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
