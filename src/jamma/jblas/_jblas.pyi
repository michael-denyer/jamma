"""Type stub for the jblas C extension (_jblas).

Provides IDE completion and mypy checking for the compiled extension module.
When the C extension is not available, jamma.jblas.__init__ provides NumPy-backed
fallbacks with identical signatures.
"""

from typing import Final, Literal

import numpy as np
import numpy.typing as npt

# Module-level constants set at PyInit__jblas time.
ABI_VERSION: Final[int]
"""JBLAS ABI version number for compatibility checking."""

jblas_isa: Final[Literal["AVX2", "NEON", "generic"]]
"""Active ISA name: "AVX2", "NEON", or "generic"."""

blas_backend: Final[str]
"""Active dgemm backend: "MKL-ILP64", "MKL-LP64", "OpenBLAS-ILP64",
"OpenBLAS-LP64", "Accelerate", "BLIS", "jblas-own", etc."""

HAS_OPENMP: Final[bool]
"""True if the extension was compiled with OpenMP support.

Level 1/2 kernels are single-threaded; Level 3 (dgemm, dsyrk, dsyr2k) uses
OpenMP parallel-for over the IC loop when OpenMP is available.
"""

# Blocking parameters (ISA-dependent, set by jblas_init()).
JBLAS_MR: Final[int]
JBLAS_NR: Final[int]
JBLAS_KC: Final[int]
JBLAS_MC: Final[int]
JBLAS_NC: Final[int]

def ddot(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Compute inner product of two double vectors.

    Args:
        x: First vector, float64, C-contiguous.
        y: Second vector, float64, C-contiguous, same length as x.

    Returns:
        Scalar dot product.
    """
    ...

def dnrm2(x: npt.NDArray[np.float64]) -> float:
    """Compute Euclidean norm of a double vector.

    Uses the Blue (1978) three-accumulator algorithm to avoid overflow and
    underflow for extreme element values.

    Args:
        x: Input vector, float64, C-contiguous.

    Returns:
        Scalar Euclidean norm.
    """
    ...

def daxpy(alpha: float, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
    """Compute y += alpha * x in-place.

    Args:
        alpha: Scalar multiplier.
        x: Input vector, float64, C-contiguous.
        y: In/out vector, float64, C-contiguous, same length as x.
            Modified in-place.
    """
    ...

def dscal(alpha: float, x: npt.NDArray[np.float64]) -> None:
    """Compute x *= alpha in-place.

    Args:
        alpha: Scalar multiplier.
        x: In/out vector, float64, C-contiguous. Modified in-place.
    """
    ...

def dgemv(
    A: npt.NDArray[np.float64], x: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute matrix-vector product A @ x.

    Args:
        A: Input matrix, shape (m, n), float64, C-contiguous.
        x: Input vector, shape (n,), float64, C-contiguous.

    Returns:
        Result vector, shape (m,), float64.
    """
    ...

def dgemm(
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    transa: str = ...,
    transb: str = ...,
) -> npt.NDArray[np.float64]:
    """Compute matrix-matrix product with optional transpose.

    Args:
        A: Left matrix, float64, C-contiguous.
        B: Right matrix, float64, C-contiguous.
        transa: 'N' (no transpose) or 'T' (transpose A).
        transb: 'N' (no transpose) or 'T' (transpose B).

    Returns:
        Result matrix C = op(A) @ op(B), float64.
    """
    ...

def dsyrk(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute symmetric rank-k update: C = X @ X.T.

    Args:
        X: Input matrix, shape (N, K), float64, C-contiguous.

    Returns:
        Symmetric result matrix, shape (N, N), float64.
    """
    ...

def dsyr2k(
    C: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute symmetric rank-2k update: result = C - A @ B.T - B @ A.T.

    Returns a new array; the input C is not modified.

    Args:
        C: Symmetric matrix, shape (N, N), float64, C-contiguous.
        A: First factor, shape (N, K), float64, C-contiguous.
        B: Second factor, shape (N, K), float64, C-contiguous.

    Returns:
        Updated result (new array), shape (N, N), float64.
    """
    ...
