"""jblas: JAMMA's self-contained BLAS compute layer.

Provides Level 1/2 BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv) and
Level 3 BLAS (dgemm — C implementation with AVX2/NEON microkernels, dsyrk and
dsyr2k for symmetric rank-k updates) plus LAPACK eigendecomposition (eigh) via
a C extension when available, falling back to NumPy when the C extension has
not been compiled.

Exports:
    ddot: Inner product of two double vectors.
    dnrm2: Euclidean norm of a double vector.
    daxpy: In-place y += alpha * x.
    dscal: In-place x *= alpha.
    dgemv: Matrix-vector product A @ x.
    dgemm: Matrix-matrix product op(A) @ op(B) with optional transpose support.
    dsyrk: Symmetric rank-k update K = X @ X.T.
    dsyr2k: Symmetric rank-2k update C -= A @ B.T + B @ A.T.
    eigh: Eigenvalues and eigenvectors of a symmetric matrix (DSYTRD + DSTEDC + DORMTR).
    jblas_isa: String identifying the active ISA ("AVX2", "NEON", "generic",
        or "numpy-fallback").
    ABI_VERSION: Integer ABI version (0 when using NumPy fallback).
    HAS_C_EXTENSION: True if the compiled C extension is loaded.
    HAS_OPENMP: True if the C extension was compiled with OpenMP support.
    JBLAS_MR: Microkernel row tile size (ISA-specific; generic fallback: 4).
    JBLAS_NR: Microkernel column tile size (ISA-specific; generic fallback: 4).
    JBLAS_KC: KC blocking depth (ISA-specific; generic fallback: 128).
    JBLAS_MC: MC row panel size (ISA-specific; generic fallback: 32).
    JBLAS_NC: NC column panel size (ISA-specific; generic fallback: 1024).
"""

from __future__ import annotations

import importlib.util
import warnings

_so_exists = importlib.util.find_spec("jamma.jblas._jblas") is not None

try:
    from jamma.jblas._jblas import (  # noqa: F401
        ABI_VERSION,
        HAS_OPENMP,
        JBLAS_KC,
        JBLAS_MC,
        JBLAS_MR,
        JBLAS_NC,
        JBLAS_NR,
        blas_backend,
        daxpy,
        ddot,
        dgemm,
        dgemv,
        dnrm2,
        dscal,
        dsyr2k,
        dsyrk,
        eigh,
        get_n_threads,
        jblas_isa,
        set_n_threads,
    )

    HAS_C_EXTENSION: bool = True

except ImportError as _exc:
    if _so_exists:
        # .so exists but failed to load — ABI mismatch, missing shared lib, etc.
        warnings.warn(
            f"jblas C extension found but failed to load ({_exc}); "
            "this usually indicates an ABI mismatch or missing shared library. "
            "Falling back to NumPy (slower). "
            "Reinstall jamma or run 'python -m jamma.jblas._compile_jblas'.",
            stacklevel=2,
        )
    else:
        warnings.warn(
            f"jblas C extension not compiled ({_exc}); "
            "using NumPy fallback (slower). "
            "Run 'python -m jamma.jblas._compile_jblas' to compile.",
            stacklevel=2,
        )
    # C extension not available — use NumPy-backed fallback with identical signatures.
    ABI_VERSION: int = 0
    HAS_C_EXTENSION = False
    HAS_OPENMP: bool = False
    jblas_isa: str = "numpy-fallback"
    blas_backend: str = "numpy-fallback"

    # Blocking parameters: generic defaults (matches jblas generic ISA).
    # Tests that import these should guard on HAS_C_EXTENSION.
    JBLAS_MR: int = 4
    JBLAS_NR: int = 4
    JBLAS_KC: int = 128
    JBLAS_MC: int = 32
    JBLAS_NC: int = 1024

    import numpy as _np

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
            x[:] = 0.0  # Match reference BLAS: NaN/Inf → +0.0
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

    def dgemm(
        A: _np.ndarray,
        B: _np.ndarray,
        transa: str = "N",
        transb: str = "N",
    ) -> _np.ndarray:
        """Compute matrix-matrix product op(A) @ op(B).

        Args:
            A: Left matrix, float64, C-contiguous.
            B: Right matrix, float64, C-contiguous.
            transa: 'N' (no transpose) or 'T' (transpose A).
            transb: 'N' (no transpose) or 'T' (transpose B).

        Returns:
            Result matrix C = op(A) @ op(B), float64.

        Raises:
            ValueError: If A or B is not 2-D, or inner dimensions don't match.
        """
        if A.ndim != 2:
            raise ValueError(f"dgemm: A must be a 2-D array, got {A.ndim}-D")
        if B.ndim != 2:
            raise ValueError(f"dgemm: B must be a 2-D array, got {B.ndim}-D")
        if not isinstance(transa, str):
            raise TypeError(
                f"dgemm: transa must be a string, got {type(transa).__name__}"
            )
        if not isinstance(transb, str):
            raise TypeError(
                f"dgemm: transb must be a string, got {type(transb).__name__}"
            )
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
        return _np.asarray(
            _np.matmul(
                _A.astype(_np.float64, copy=False),
                _B.astype(_np.float64, copy=False),
            ),
            dtype=_np.float64,
        )

    def dsyrk(X: _np.ndarray) -> _np.ndarray:
        """Compute symmetric rank-k update: K = X @ X.T.

        Computes the full symmetric matrix K = X @ X.T, filling both
        lower and upper triangles.  Uses the lower-triangle-only tile
        computation path in the C extension for efficiency (saves ~50%
        tile iterations vs dgemm).  This NumPy fallback computes the
        full product directly.

        Args:
            X: Input matrix, shape (N, K), float64.

        Returns:
            Symmetric result matrix K, shape (N, N), float64.

        Raises:
            ValueError: If X is not 2-D.
        """
        if X.ndim != 2:
            raise ValueError(f"dsyrk: X must be a 2-D array, got {X.ndim}-D")
        X64 = _np.ascontiguousarray(X, dtype=_np.float64)
        result = _np.dot(X64, X64.T)
        # Mirror lower to upper to guarantee bitwise symmetry,
        # matching the C extension contract.
        il = _np.tril_indices_from(result, -1)
        result.T[il] = result[il]
        return result

    def dsyr2k(C: _np.ndarray, A: _np.ndarray, B: _np.ndarray) -> _np.ndarray:
        """Compute symmetric rank-2k update: result = C - A @ B.T - B @ A.T.

        Returns a new array; the input C is not modified.  Both the C
        extension and this NumPy fallback update the full matrix.

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
            raise ValueError(
                f"dsyr2k: A rows ({A.shape[0]}) must match C dimension ({N})"
            )
        if B.shape[0] != N:
            raise ValueError(
                f"dsyr2k: B rows ({B.shape[0]}) must match C dimension ({N})"
            )
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

    def eigh(K: _np.ndarray) -> tuple[_np.ndarray, _np.ndarray]:
        """Compute eigenvalues and eigenvectors of a symmetric matrix.

        Args:
            K: Symmetric matrix, shape (N, N), float64. Overwritten as scratch.

        Returns:
            Tuple of (eigenvalues, eigenvectors) where eigenvalues is shape (N,)
            ascending, eigenvectors is shape (N, N) with columns as unit eigenvectors.

        Raises:
            ValueError: If K is not 2-D square float64.
            numpy.linalg.LinAlgError: If convergence fails (NumPy fallback).
            RuntimeError: If convergence fails (C extension).
            MemoryError: If workspace allocation fails (C extension).
        """
        if K.ndim != 2:
            raise ValueError(f"eigh: K must be a 2-D array, got {K.ndim}-D")
        if K.shape[0] != K.shape[1]:
            raise ValueError(f"eigh: K must be square, got shape {K.shape}")
        K64 = _np.asarray(K, dtype=_np.float64)
        w, v = _np.linalg.eigh(K64)
        # Match C extension contract: K is overwritten as scratch
        if K.dtype == _np.float64 and K.flags["WRITEABLE"]:
            K[:] = 0.0
        return w, v

    import os as _os

    # Mutable container for fallback thread state (closures can't rebind
    # names in enclosing except-block scope, but can mutate a list).
    _fallback_thread_state = [_os.cpu_count() or 1]

    def get_n_threads() -> int:
        """Get current jblas thread count (fallback: os.cpu_count())."""
        return _fallback_thread_state[0]

    def set_n_threads(n: int) -> int:
        """Set jblas thread count (fallback: clamped to os.cpu_count()).

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


__all__ = [
    "ABI_VERSION",
    "blas_backend",
    "ddot",
    "dnrm2",
    "daxpy",
    "dscal",
    "dgemv",
    "dgemm",
    "dsyrk",
    "dsyr2k",
    "eigh",
    "get_n_threads",
    "set_n_threads",
    "jblas_isa",
    "HAS_C_EXTENSION",
    "HAS_OPENMP",
    "JBLAS_MR",
    "JBLAS_NR",
    "JBLAS_KC",
    "JBLAS_MC",
    "JBLAS_NC",
]
