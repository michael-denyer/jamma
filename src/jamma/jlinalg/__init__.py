"""jlinalg: JAMMA's vendor BLAS/LAPACK dispatch layer with NumPy fallback.

The C extension provides vendor BLAS dispatch (dgemm, dsyrk via system
BLAS/LAPACK), eigendecomposition (eigh via vendor DSYEVD/DSYEVR), and
single-pass SNP statistics (compute_snp_stats_chunk).

Level 1/2 BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv) and dsyr2k
always use NumPy implementations -- they were removed from the C extension
in ABI version 12 as vendor dispatch provides no benefit for these operations.

Exports:
    ddot: Inner product of two double vectors (NumPy).
    dnrm2: Euclidean norm of a double vector (NumPy).
    daxpy: In-place y += alpha * x (NumPy).
    dscal: In-place x *= alpha (NumPy).
    dgemv: Matrix-vector product A @ x (NumPy).
    dgemm: Matrix-matrix product op(A) @ op(B) via vendor BLAS or NumPy.
    dsyrk: Symmetric rank-k update K = X @ X.T via vendor BLAS or NumPy.
    dsyr2k: Symmetric rank-2k update C -= A @ B.T + B @ A.T (NumPy).
    eigh: Eigenvalues and eigenvectors of a symmetric matrix via vendor LAPACK.
    blas_has_dgemm: True if ILP64 vendor DGEMM is available. When 0, dgemm is
        the NumPy implementation -- the C one raises RuntimeError.
    blas_has_dsyevd: True if ILP64 vendor DSYEVD is available.
    blas_has_dsyevr: True if ILP64 vendor DSYEVR is available.
    blas_has_dsyrk: True if ILP64 vendor DSYRK is available.
    blas_is_ilp64: True if the active BLAS backend is ILP64.
    jlinalg_isa: String identifying the active ISA ("AVX2", "NEON", "generic",
        or "numpy-fallback").
    ABI_VERSION: Integer ABI version (0 when using NumPy fallback).
    HAS_C_EXTENSION: True if the compiled C extension is loaded.
    HAS_OPENMP: True if the C extension was compiled with OpenMP support.
    compute_snp_stats_chunk: Single-pass per-SNP statistics (mean, variance,
        miss count, optional HWE genotype counts).

Env vars:
    JAMMA_FORCE_NUMPY_FALLBACK: when truthy (anything other than "" or "0"),
        skips the _jlinalg.so import entirely. blas_backend is set to
        "numpy-fallback-forced" so the sanitizer workflow can confirm the
        gate engaged (distinguishable from the natural "numpy-fallback"
        value used when the .so genuinely failed to import).
    JLINALG_NO_VENDOR_DGEMM: when truthy, dispatch leaves vendor dgemm
        unwired, so blas_has_dgemm reports 0 with the extension loaded. That
        is the permanent state of an LP64-only host, which CI never reaches.
"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as _np

from jamma._build_support.compile_and_link import JLINALG_SPEC
from jamma.core.constants import env_flag
from jamma.core.recompile import _load_c_module

_so_exists = importlib.util.find_spec("jamma.jlinalg._jlinalg") is not None
HAS_C_EXTENSION: bool = False

_EXPECTED_JLINALG_ABI = 15  # Must match JLINALG_ABI_VERSION in include/jlinalg.h


def _validate_dsyrk(X: _np.ndarray, out: _np.ndarray | None, beta: float) -> None:
    """Validate the public dsyrk contract before backend dispatch."""
    if X.ndim != 2:
        raise ValueError(f"dsyrk: X must be a 2-D array, got {X.ndim}-D")
    if out is None:
        if beta != 0.0:
            raise ValueError("dsyrk: beta requires out")
    else:
        if not isinstance(out, _np.ndarray):
            raise TypeError("dsyrk: out must be a numpy array")
        if out.dtype != _np.float64:
            raise ValueError(f"dsyrk: out must be float64, got {out.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("dsyrk: out must be C-contiguous")
        if not out.flags["ALIGNED"]:
            raise ValueError("dsyrk: out must be aligned")
        if not out.flags["WRITEABLE"]:
            raise ValueError("dsyrk: out must be writeable")
        if out.ndim != 2:
            raise ValueError(f"dsyrk: out must be 2-D, got {out.ndim}-D")
        expected = (X.shape[0], X.shape[0])
        if out.shape != expected:
            raise ValueError(
                f"dsyrk: out shape {out.shape} doesn't match result shape {expected}"
            )


_DSYRK_BLOCK_BYTES = 64 << 20  # Absolute cap on the fallback's scratch buffer
_DSYRK_BLOCK_FRACTION = 8  # Scratch is also held to 1/8 of the output
_DSYRK_MIRROR_BLOCK = 512  # Mirror tile edge; its scratch is O(edge^2), not O(n)

# What one mirror tile holds: tril_indices' two int64 index arrays plus the
# gathered float64 values (each edge*(edge-1)/2 long), and one edge-by-edge
# transposed tile. Tiling both mirror axes keeps this independent of n, so
# dsyrk_scratch_bytes can bound the fallback with a constant rather than a term
# that grows with the matrix.
_MIRROR_SCRATCH_BYTES = 8 * (
    3 * (_DSYRK_MIRROR_BLOCK * (_DSYRK_MIRROR_BLOCK - 1) // 2)
    + _DSYRK_MIRROR_BLOCK * _DSYRK_MIRROR_BLOCK
)


def _dsyrk_row_block(n: int) -> int:
    """Rows per accumulation block.

    Held to a fraction of ``n`` as well as an absolute byte cap: the scratch is
    block-by-n, so a byte cap alone leaves it comparable to the whole output at
    modest ``n``.
    """
    by_bytes = _DSYRK_BLOCK_BYTES // max(1, n * 8)
    by_fraction = max(1, n // _DSYRK_BLOCK_FRACTION)
    return max(1, min(n, by_fraction, by_bytes))


def dsyrk_scratch_bytes(n: int) -> int:
    """Upper bound on what one ``dsyrk`` call holds beyond its n-by-n output.

    Zero on the native backend, which accumulates in place. The NumPy fallback
    holds one block-by-n float64 product during accumulation and a fixed set of
    tiles during the mirror, whichever is larger. A memory pre-flight budgeting
    only the accumulator under-counts a fallback run, so ``jamma.core.memory``
    adds this to the kinship phase peak.
    """
    if _dsyrk_backend is not _dsyrk_numpy_impl:
        return 0
    return max(_dsyrk_row_block(n) * n * 8, _MIRROR_SCRATCH_BYTES)


def _mirror_lower_to_upper(result: _np.ndarray) -> None:
    """Copy the strictly lower triangle onto the upper, tile by tile.

    ``np.tril_indices_from`` allocates two n^2/2 index arrays plus an n^2/2
    value gather — together more than the matrix being symmetrised. Tiling both
    axes caps every temporary at the tile edge, so the scratch is constant in
    ``n`` and ``_MIRROR_SCRATCH_BYTES`` can state it exactly.
    """
    n = result.shape[0]
    block = min(_DSYRK_MIRROR_BLOCK, n) or 1
    full_il = _np.tril_indices(block, -1)
    for i in range(0, n, block):
        j = min(i + block, n)
        for c in range(0, i, block):
            # Rows i:j, columns c:c+block lie entirely below the diagonal. Every
            # such tile is full width: i is a multiple of block, so the partial
            # tile can only ever be the diagonal one handled below.
            result[c : c + block, i:j] = result[i:j, c : c + block].T
        diag = result[i:j, i:j]
        edge = j - i
        il = full_il if edge == block else _np.tril_indices(edge, -1)
        diag.T[il] = diag[il]


def _dsyrk_numpy_impl(
    X: _np.ndarray, *, out: _np.ndarray | None = None, beta: float = 0.0
) -> _np.ndarray:
    """Unchecked NumPy implementation of C = X @ X.T + beta*C.

    The accumulating path walks row blocks and touches only the lower triangle,
    which the closing mirror copies up. ``result += X @ X.T`` would materialise
    a second full n-by-n product, so the fallback's peak was several times the
    output it writes — memory the kinship pre-flight does not budget for.
    ``dsyrk_scratch_bytes`` reports what one block costs so it can.
    """
    X64 = _np.ascontiguousarray(X, dtype=_np.float64)
    n = X64.shape[0]
    if out is None:
        result = _np.dot(X64, X64.T)
    else:
        result = out
        if beta == 0.0:
            _np.dot(X64, X64.T, out=result)
        else:
            if beta != 1.0:
                result *= beta
            block = _dsyrk_row_block(n)
            for i in range(0, n, block):
                j = min(i + block, n)
                result[i:j, :j] += X64[i:j] @ X64[:j].T
    _mirror_lower_to_upper(result)
    return result


def _dsyrk_numpy(
    X: _np.ndarray, *, out: _np.ndarray | None = None, beta: float = 0.0
) -> _np.ndarray:
    """Validated NumPy implementation, exposed for backend-specific tests."""
    _validate_dsyrk(X, out, beta)
    return _dsyrk_numpy_impl(X, out=out, beta=beta)


# Default backend; a usable native implementation replaces it during import.
_dsyrk_backend = _dsyrk_numpy_impl


def _dgemm_numpy(
    A: _np.ndarray,
    B: _np.ndarray,
    transa: str = "N",
    transb: str = "N",
    out: _np.ndarray | None = None,
) -> _np.ndarray:
    """Compute matrix-matrix product op(A) @ op(B).

    Args:
        A: Left matrix, float64, C-contiguous.
        B: Right matrix, float64, C-contiguous.
        transa: 'N' (no transpose) or 'T' (transpose A).
        transb: 'N' (no transpose) or 'T' (transpose B).
        out: Optional preallocated output array. If provided, the result
            is stored in this buffer and the same array is returned.
            Must be 2-D float64, C-contiguous, with shape (M, N)
            matching the result dimensions.

    Returns:
        Result matrix C = op(A) @ op(B), float64.

    Raises:
        ValueError: If A or B is not 2-D, inner dimensions don't match,
            or out has wrong shape/dtype/layout.
    """
    if A.ndim != 2:
        raise ValueError(f"dgemm: A must be a 2-D array, got {A.ndim}-D")
    if B.ndim != 2:
        raise ValueError(f"dgemm: B must be a 2-D array, got {B.ndim}-D")
    if not isinstance(transa, str):
        raise TypeError(f"dgemm: transa must be a string, got {type(transa).__name__}")
    if not isinstance(transb, str):
        raise TypeError(f"dgemm: transb must be a string, got {type(transb).__name__}")
    if transa.upper() not in ("N", "T"):
        raise ValueError(f"dgemm: transa must be 'N' or 'T', got '{transa}'")
    if transb.upper() not in ("N", "T"):
        raise ValueError(f"dgemm: transb must be 'N' or 'T', got '{transb}'")
    _A = A.T if transa.upper() == "T" else A
    _B = B.T if transb.upper() == "T" else B
    if _A.shape[1] != _B.shape[0]:
        raise ValueError(
            f"dgemm: op(A) columns ({_A.shape[1]}) must match "
            f"op(B) rows ({_B.shape[0]})"
        )
    if out is not None:
        expected = (_A.shape[0], _B.shape[1])
        if out.ndim != 2 or out.shape != expected:
            raise ValueError(
                f"dgemm: out shape {out.shape} doesn't match result shape {expected}"
            )
        if out.dtype != _np.float64:
            raise ValueError(f"dgemm: out must be float64, got {out.dtype}")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("dgemm: out must be C-contiguous")
        if not out.flags["WRITEABLE"]:
            raise ValueError("dgemm: out must be writeable")
        _np.matmul(
            _A.astype(_np.float64, copy=False),
            _B.astype(_np.float64, copy=False),
            out=out,
        )
        return out
    return _np.asarray(
        _np.matmul(
            _A.astype(_np.float64, copy=False),
            _B.astype(_np.float64, copy=False),
        ),
        dtype=_np.float64,
    )


# ASAN/UBSAN sanitizer workflow needs a way to skip the
# _jlinalg.so import entirely. RESEARCH §"Pitfall 4" — ASAN + dlopen(...,
# RTLD_LAZY) inside blas_dispatch.c can produce false-positive
# heap-buffer-overflow reports. Forcing the import to be skipped (rather
# than disabling downstream calls) means dlopen never runs, so ASAN's
# interceptors never see the unowned BLAS-internal pointers. Truthy
# values: anything other than "", "0".
_FORCE_NUMPY = env_flag("JAMMA_FORCE_NUMPY_FALLBACK")

# One shared seam does the import, ABI check, and recompile-then-retry that this
# module used to hand-write twice (initial try plus post-recompile retry). It
# returns the validated extension module or None. Under JAMMA_FORCE_NUMPY_FALLBACK
# it returns None without importing, so the sanitizer workflow's dlopen never
# runs; required-symbol presence is checked against JLINALG_SPEC.required_attrs.
_mod = _load_c_module(JLINALG_SPEC, _EXPECTED_JLINALG_ABI)

if _mod is not None:
    ABI_VERSION = _mod.ABI_VERSION
    HAS_OPENMP = _mod.HAS_OPENMP
    blas_backend = _mod.blas_backend
    blas_has_dgemm = _mod.blas_has_dgemm
    blas_has_dsyevd = _mod.blas_has_dsyevd
    blas_has_dsyevr = _mod.blas_has_dsyevr
    blas_has_dsyrk = _mod.blas_has_dsyrk
    blas_has_lapacke_dsyevd = _mod.blas_has_lapacke_dsyevd
    blas_is_ilp64 = _mod.blas_is_ilp64
    compute_snp_stats_chunk = _mod.compute_snp_stats_chunk
    eigh = _mod.eigh
    get_n_threads = _mod.get_n_threads
    jlinalg_isa = _mod.jlinalg_isa
    set_n_threads = _mod.set_n_threads
    _dgemm_native = _mod.dgemm
    _dsyrk_native = _mod.dsyrk

    HAS_C_EXTENSION: bool = True

    # C extension loaded, but vendor BLAS/LAPACK may not be available.
    dgemm = _dgemm_native if blas_has_dgemm else _dgemm_numpy
    _dsyrk_backend = _dsyrk_native if blas_has_dsyrk else _dsyrk_numpy_impl

    if not blas_has_dsyevd and not blas_has_dsyevr:

        def eigh(  # type: ignore[misc]
            K: _np.ndarray, inplace: bool = False
        ) -> tuple[_np.ndarray, _np.ndarray]:
            """NumPy fallback: eigendecomposition of symmetric matrix."""
            if K.ndim != 2:
                raise ValueError(f"eigh: K must be a 2-D array, got {K.ndim}-D")
            if K.shape[0] != K.shape[1]:
                raise ValueError(f"eigh: K must be square, got shape {K.shape}")
            K64 = _np.asarray(K, dtype=_np.float64)
            w, v = _np.linalg.eigh(K64)
            if inplace:
                K[:] = v
                return w, K
            return w, v

elif _FORCE_NUMPY:
    # Skip the _jlinalg.so import entirely. HAS_C_EXTENSION stays False; the
    # `if not HAS_C_EXTENSION:` block below defines the rest of the fallback
    # state. The "numpy-fallback-forced" value is the discoverable telemetry
    # signal the sanitizer workflow log greps for.
    blas_backend = "numpy-fallback-forced"
    blas_is_ilp64 = 0

if not HAS_C_EXTENSION:
    if not _FORCE_NUMPY:
        # Only warn on natural fallback — forced fallback is a deliberate
        # choice (sanitizer workflow, dependency-free environments), not a
        # degradation. The "numpy-fallback-forced" telemetry value below is
        # the discoverable signal for forced runs.
        if _so_exists:
            warnings.warn(
                "jlinalg C extension found but failed to load; "
                "this usually indicates an ABI mismatch or missing shared library. "
                "Falling back to NumPy (slower). "
                "Reinstall jamma or run 'python -m jamma.jlinalg._compile_jlinalg'.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                "jlinalg C extension not compiled; "
                "using NumPy fallback (slower). "
                "Run 'python -m jamma.jlinalg._compile_jlinalg' to compile.",
                stacklevel=2,
            )
    # C extension not available -- use NumPy-backed fallback with identical signatures.
    ABI_VERSION: int = 0
    HAS_C_EXTENSION = False
    HAS_OPENMP: bool = False
    jlinalg_isa: str = "numpy-fallback"
    # Use globals().setdefault so a previously-set "numpy-fallback-forced" value
    # (from the _FORCE_NUMPY branch above) is preserved. This is the ONLY
    # acceptable form — a `dir()` check is order-dependent and unreliable.
    globals().setdefault("blas_backend", "numpy-fallback")
    globals().setdefault("blas_is_ilp64", 0)
    blas_has_dgemm: int = 0
    blas_has_dsyrk: int = 0
    blas_has_dsyevd: int = 0
    blas_has_dsyevr: int = 0
    blas_has_lapacke_dsyevd: int = 0

    import warnings as _warnings

    def compute_snp_stats_chunk(
        data: _np.ndarray,
        means: _np.ndarray,
        miss_counts: _np.ndarray,
        variances: _np.ndarray,
        n_aa: _np.ndarray | None = None,
        n_ab: _np.ndarray | None = None,
        n_bb: _np.ndarray | None = None,
    ) -> None:
        """NumPy fallback for per-SNP statistics into pre-allocated output arrays.

        Computes mean, variance (population), and missing count per column.
        Optionally counts genotype values (0, 1, 2) for HWE testing.

        Note: this fallback uses multiple NumPy passes; the C kernel is single-pass.

        Args:
            data: Genotype matrix (n_samples, n_snps), float32 or float64, C-contiguous.
            means: Output array (n_snps,), float64.
            miss_counts: Output array (n_snps,), intp.
            variances: Output array (n_snps,), float64.
            n_aa: Output array (n_snps,), int64 (None to skip HWE).
            n_ab: Output array (n_snps,), int64 (None to skip HWE).
            n_bb: Output array (n_snps,), int64 (None to skip HWE).
        """
        is_nan = _np.isnan(data)
        mc = _np.sum(is_nan, axis=0)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            m = _np.nanmean(data, axis=0)
            v = _np.nanvar(data, axis=0)
        m = _np.nan_to_num(m, nan=0.0)
        v = _np.nan_to_num(v, nan=0.0)
        means[:] = m
        miss_counts[:] = mc
        variances[:] = v
        if n_aa is not None and n_ab is not None and n_bb is not None:
            valid = ~is_nan
            n_aa[:] = _np.sum((data == 0) & valid, axis=0)
            n_ab[:] = _np.sum((data == 1) & valid, axis=0)
            n_bb[:] = _np.sum((data == 2) & valid, axis=0)

    dgemm = _dgemm_numpy

    _dsyrk_backend = _dsyrk_numpy_impl

    def eigh(K: _np.ndarray, inplace: bool = False) -> tuple[_np.ndarray, _np.ndarray]:
        """Compute eigenvalues and eigenvectors of a symmetric matrix.

        Args:
            K: Symmetric matrix, shape (N, N), float64.
            inplace: If True, overwrite K with eigenvectors in-place.

        Returns:
            Tuple of (eigenvalues, eigenvectors).

        Raises:
            ValueError: If K is not 2-D square float64.
        """
        if K.ndim != 2:
            raise ValueError(f"eigh: K must be a 2-D array, got {K.ndim}-D")
        if K.shape[0] != K.shape[1]:
            raise ValueError(f"eigh: K must be square, got shape {K.shape}")
        if inplace:
            if K.dtype != _np.float64:
                raise ValueError(f"eigh: inplace=True requires float64, got {K.dtype}")
            if not K.flags["C_CONTIGUOUS"]:
                raise ValueError("eigh: inplace=True requires a C-contiguous array")
            if not K.flags["WRITEABLE"]:
                raise ValueError("eigh: inplace=True requires a writeable array")
        K64 = _np.asarray(K, dtype=_np.float64)
        w, v = _np.linalg.eigh(K64)
        if inplace:
            K[:] = v
            return w, K
        else:
            if K.dtype == _np.float64 and K.flags["WRITEABLE"]:
                K[:] = 0.0
            return w, v

    import os as _os

    _fallback_thread_state = [_os.cpu_count() or 1]

    def get_n_threads() -> int:
        """Get current jlinalg thread count (fallback: os.cpu_count())."""
        return _fallback_thread_state[0]

    def set_n_threads(n: int) -> int:
        """Set jlinalg thread count (fallback: clamped to os.cpu_count()).

        Args:
            n: Desired thread count (must be >= 1).

        Returns:
            Previous thread count.

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError("set_n_threads: n must be >= 1")
        old = _fallback_thread_state[0]
        max_threads = _os.cpu_count() or 1
        _fallback_thread_state[0] = min(n, max_threads)
        return old


def dsyrk(
    X: _np.ndarray, *, out: _np.ndarray | None = None, beta: float = 0.0
) -> _np.ndarray:
    """Compute ``X @ X.T + beta * out`` with a shared backend contract."""
    _validate_dsyrk(X, out, beta)
    return _dsyrk_backend(X, out=out, beta=beta)


# ---------------------------------------------------------------------------
# NumPy-only BLAS functions (always use NumPy, never from C extension)
# ---------------------------------------------------------------------------
# These were removed from the C extension in ABI 12 because vendor dispatch
# provides no benefit for Level 1/2 operations -- NumPy's BLAS binding is
# equally fast and simpler to maintain.


def ddot(x: _np.ndarray, y: _np.ndarray) -> float:
    """Compute inner product of two double vectors.

    Args:
        x: First vector, float64, C-contiguous.
        y: Second vector, float64, C-contiguous, same length as x.

    Returns:
        Scalar dot product as Python float.

    Raises:
        ValueError: If x or y is not 1-D, or lengths differ.
    """
    if x.ndim != 1:
        raise ValueError(f"ddot: x must be a 1-D array, got {x.ndim}-D")
    if y.ndim != 1:
        raise ValueError(f"ddot: y must be a 1-D array, got {y.ndim}-D")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"ddot: x and y must have the same length, "
            f"got {x.shape[0]} and {y.shape[0]}"
        )
    return float(
        _np.dot(
            x.astype(_np.float64, copy=False),
            y.astype(_np.float64, copy=False),
        )
    )


def dnrm2(x: _np.ndarray) -> float:
    """Compute Euclidean norm of a double vector.

    Args:
        x: Input vector, float64, C-contiguous.

    Returns:
        Scalar norm as Python float.

    Raises:
        ValueError: If x is not 1-D.
    """
    if x.ndim != 1:
        raise ValueError(f"dnrm2: x must be a 1-D array, got {x.ndim}-D")
    return float(_np.linalg.norm(x.astype(_np.float64, copy=False)))


def daxpy(alpha: float, x: _np.ndarray, y: _np.ndarray) -> None:
    """Compute y += alpha * x in-place.

    Args:
        alpha: Scalar multiplier.
        x: Input vector, float64, C-contiguous.
        y: In/out vector, float64, C-contiguous, same length as x.
            Modified in-place.

    Raises:
        ValueError: If x or y is not 1-D, or lengths differ.
    """
    if x.ndim != 1:
        raise ValueError(f"daxpy: x must be a 1-D array, got {x.ndim}-D")
    if y.ndim != 1:
        raise ValueError(f"daxpy: y must be a 1-D array, got {y.ndim}-D")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"daxpy: x and y must have the same length, "
            f"got {x.shape[0]} and {y.shape[0]}"
        )
    if y.dtype != _np.float64:
        raise TypeError(f"daxpy: y must be float64, got {y.dtype}")
    y += alpha * _np.asarray(x, dtype=_np.float64)


def dscal(alpha: float, x: _np.ndarray) -> None:
    """Compute x *= alpha in-place.

    Args:
        alpha: Scalar multiplier.
        x: In/out vector, float64, C-contiguous. Modified in-place.

    Raises:
        ValueError: If x is not 1-D.
    """
    if x.ndim != 1:
        raise ValueError(f"dscal: x must be a 1-D array, got {x.ndim}-D")
    if x.dtype != _np.float64:
        raise TypeError(f"dscal: x must be float64, got {x.dtype}")
    if alpha == 0.0:
        x[:] = 0.0  # Match reference BLAS: NaN/Inf -> +0.0
    else:
        x *= alpha


def dgemv(A: _np.ndarray, x: _np.ndarray) -> _np.ndarray:
    """Compute matrix-vector product A @ x.

    Args:
        A: Input matrix, shape (m, n), float64, C-contiguous.
        x: Input vector, shape (n,), float64, C-contiguous.

    Returns:
        Result vector, shape (m,), float64.

    Raises:
        ValueError: If A is not 2-D, x is not 1-D, or shapes don't match.
    """
    if A.ndim != 2:
        raise ValueError(f"dgemv: A must be a 2-D array, got {A.ndim}-D")
    if x.ndim != 1:
        raise ValueError(f"dgemv: x must be a 1-D array, got {x.ndim}-D")
    if A.shape[1] != x.shape[0]:
        raise ValueError(
            f"dgemv: A columns ({A.shape[1]}) must match x length ({x.shape[0]})"
        )
    return _np.asarray(
        A.astype(_np.float64, copy=False) @ x.astype(_np.float64, copy=False),
        dtype=_np.float64,
    )


def dsyr2k(C: _np.ndarray, A: _np.ndarray, B: _np.ndarray) -> _np.ndarray:
    """Compute symmetric rank-2k update: result = C - A @ B.T - B @ A.T.

    Returns a new array; the input C is not modified.

    Args:
        C: Symmetric matrix, shape (N, N), float64.
        A: First factor, shape (N, K), float64.
        B: Second factor, shape (N, K), float64.

    Returns:
        Updated result (new array), shape (N, N), float64.

    Raises:
        ValueError: If C is not 2-D square, or A/B are not 2-D, or
            dimensions are inconsistent.
    """
    if C.ndim != 2:
        raise ValueError(f"dsyr2k: C must be a 2-D array, got {C.ndim}-D")
    if C.shape[0] != C.shape[1]:
        raise ValueError(f"dsyr2k: C must be square, got shape {C.shape}")
    if A.ndim != 2:
        raise ValueError(f"dsyr2k: A must be a 2-D array, got {A.ndim}-D")
    if B.ndim != 2:
        raise ValueError(f"dsyr2k: B must be a 2-D array, got {B.ndim}-D")
    N = C.shape[0]
    if A.shape[0] != N:
        raise ValueError(f"dsyr2k: A rows ({A.shape[0]}) must match C dimension ({N})")
    if B.shape[0] != N:
        raise ValueError(f"dsyr2k: B rows ({B.shape[0]}) must match C dimension ({N})")
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"dsyr2k: A columns ({A.shape[1]}) must match B columns ({B.shape[1]})"
        )
    C64 = _np.asarray(C, dtype=_np.float64).copy()
    A64 = _np.asarray(A, dtype=_np.float64)
    B64 = _np.asarray(B, dtype=_np.float64)
    C64 -= A64 @ B64.T
    C64 -= B64 @ A64.T
    return C64


__all__ = [
    "ABI_VERSION",
    "HAS_C_EXTENSION",
    "HAS_OPENMP",
    "blas_backend",
    "blas_has_dgemm",
    "blas_has_dsyevd",
    "blas_has_dsyevr",
    "blas_has_dsyrk",
    "blas_has_lapacke_dsyevd",
    "blas_is_ilp64",
    "compute_snp_stats_chunk",
    "daxpy",
    "ddot",
    "dgemm",
    "dgemv",
    "dnrm2",
    "dscal",
    "dsyr2k",
    "dsyrk",
    "dsyrk_scratch_bytes",
    "eigh",
    "get_n_threads",
    "jlinalg_isa",
    "set_n_threads",
]
