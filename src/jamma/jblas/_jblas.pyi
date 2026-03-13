"""Type stub for the jblas C extension (_jblas).

Provides IDE completion and mypy checking for the compiled extension module.
When the C extension is not available, jamma.jblas.__init__ provides NumPy-backed
fallbacks with identical signatures.
"""

import numpy as np

# Module-level constants set at PyInit__jblas time.
ABI_VERSION: int
"""JBLAS ABI version number for compatibility checking."""

jblas_isa: str
"""Active ISA name: "AVX2", "NEON", or "generic"."""

HAS_OPENMP: bool
"""True if the extension was compiled with OpenMP support."""

def ddot(x: np.ndarray, y: np.ndarray) -> float:
    """Compute inner product of two double vectors.

    Args:
        x: First vector, float64, C-contiguous.
        y: Second vector, float64, C-contiguous, same length as x.

    Returns:
        Scalar dot product.
    """
    ...

def dnrm2(x: np.ndarray) -> float:
    """Compute Euclidean norm of a double vector.

    Uses the Blue (1978) three-accumulator algorithm to avoid overflow and
    underflow for extreme element values.

    Args:
        x: Input vector, float64, C-contiguous.

    Returns:
        Scalar Euclidean norm.
    """
    ...

def daxpy(alpha: float, x: np.ndarray, y: np.ndarray) -> None:
    """Compute y += alpha * x in-place.

    Args:
        alpha: Scalar multiplier.
        x: Input vector, float64, C-contiguous.
        y: In/out vector, float64, C-contiguous, same length as x.
            Modified in-place.
    """
    ...

def dscal(alpha: float, x: np.ndarray) -> None:
    """Compute x *= alpha in-place.

    Args:
        alpha: Scalar multiplier.
        x: In/out vector, float64, C-contiguous. Modified in-place.
    """
    ...

def dgemv(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute matrix-vector product A @ x.

    Args:
        A: Input matrix, shape (m, n), float64, C-contiguous.
        x: Input vector, shape (n,), float64, C-contiguous.

    Returns:
        Result vector, shape (m,), float64.
    """
    ...

def dgemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute matrix-matrix product A @ B.

    C implementation planned. Currently uses NumPy fallback via __init__.py.

    Args:
        A: Left matrix, shape (m, k), float64, C-contiguous.
        B: Right matrix, shape (k, n), float64, C-contiguous.

    Returns:
        Result matrix, shape (m, n), float64.
    """
    ...
