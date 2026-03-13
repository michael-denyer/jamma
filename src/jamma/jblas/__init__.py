"""jblas: JAMMA's self-contained BLAS/LAPACK compute layer.

Provides Level 1/2 BLAS primitives (ddot, dnrm2, daxpy, dscal, dgemv) and
Level 3 BLAS (dgemm) via a C extension when available, falling back to NumPy
when the C extension has not been compiled.

Exports:
    ddot: Inner product of two double vectors.
    dnrm2: Euclidean norm of a double vector.
    daxpy: In-place y += alpha * x.
    dscal: In-place x *= alpha.
    dgemv: Matrix-vector product A @ x.
    dgemm: Matrix-matrix product A @ B (Phase 78 will implement C version).
    jblas_isa: String identifying the active ISA ("AVX2", "NEON", "generic",
        or "numpy-fallback").
    HAS_C_EXTENSION: True if the compiled C extension is loaded.
    HAS_OPENMP: True if the C extension was compiled with OpenMP support.
"""

from __future__ import annotations

try:
    from jamma.jblas._jblas import (  # noqa: F401
        HAS_OPENMP,
        daxpy,
        ddot,
        dgemv,
        dnrm2,
        dscal,
        jblas_isa,
    )

    HAS_C_EXTENSION: bool = True

    # dgemm is implemented in Phase 78; expose it from C extension when present.
    try:
        from jamma.jblas._jblas import dgemm  # noqa: F401
    except ImportError:
        # C extension does not yet expose dgemm (pre-Phase-78 wheel); use fallback.
        import numpy as _np

        def dgemm(A: _np.ndarray, B: _np.ndarray) -> _np.ndarray:
            """Compute matrix-matrix product A @ B.

            Args:
                A: Left matrix, shape (m, k), float64, C-contiguous.
                B: Right matrix, shape (k, n), float64, C-contiguous.

            Returns:
                Result matrix, shape (m, n), float64.
            """
            return _np.matmul(A, B)

except ImportError:
    # C extension not available — use NumPy-backed fallback with identical signatures.
    HAS_C_EXTENSION = False
    HAS_OPENMP: bool = False
    jblas_isa: str = "numpy-fallback"

    import numpy as _np

    def ddot(x: _np.ndarray, y: _np.ndarray) -> float:
        """Compute inner product of two double vectors.

        Args:
            x: First vector, float64, C-contiguous.
            y: Second vector, float64, C-contiguous, same length as x.

        Returns:
            Scalar dot product as Python float.
        """
        return float(_np.dot(x, y))

    def dnrm2(x: _np.ndarray) -> float:
        """Compute Euclidean norm of a double vector.

        Args:
            x: Input vector, float64, C-contiguous.

        Returns:
            Scalar norm as Python float.
        """
        return float(_np.linalg.norm(x))

    def daxpy(alpha: float, x: _np.ndarray, y: _np.ndarray) -> None:
        """Compute y += alpha * x in-place.

        Args:
            alpha: Scalar multiplier.
            x: Input vector, float64, C-contiguous.
            y: In/out vector, float64, C-contiguous, same length as x.
                Modified in-place.
        """
        y += alpha * x

    def dscal(alpha: float, x: _np.ndarray) -> None:
        """Compute x *= alpha in-place.

        Args:
            alpha: Scalar multiplier.
            x: In/out vector, float64, C-contiguous. Modified in-place.
        """
        x *= alpha

    def dgemv(A: _np.ndarray, x: _np.ndarray) -> _np.ndarray:
        """Compute matrix-vector product A @ x.

        Args:
            A: Input matrix, shape (m, n), float64, C-contiguous.
            x: Input vector, shape (n,), float64, C-contiguous.

        Returns:
            Result vector, shape (m,), float64.
        """
        return A @ x

    def dgemm(A: _np.ndarray, B: _np.ndarray) -> _np.ndarray:
        """Compute matrix-matrix product A @ B.

        Args:
            A: Left matrix, shape (m, k), float64, C-contiguous.
            B: Right matrix, shape (k, n), float64, C-contiguous.

        Returns:
            Result matrix, shape (m, n), float64.
        """
        return _np.matmul(A, B)


__all__ = [
    "ddot",
    "dnrm2",
    "daxpy",
    "dscal",
    "dgemv",
    "dgemm",
    "jblas_isa",
    "HAS_C_EXTENSION",
    "HAS_OPENMP",
]
