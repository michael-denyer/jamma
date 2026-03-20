"""Type stub for the jlinalg C extension (_jlinalg).

Provides IDE completion and mypy checking for the compiled extension module.
When the C extension is not available, jamma.jlinalg.__init__ provides NumPy-backed
fallbacks with identical signatures.
"""

from typing import Final, Literal

import numpy as np
import numpy.typing as npt

# Module-level constants set at PyInit__jlinalg time.
ABI_VERSION: Final[int]
"""JLINALG ABI version number for compatibility checking."""

jlinalg_isa: Final[Literal["AVX2", "NEON", "generic"]]
"""Active ISA name: "AVX2", "NEON", or "generic"."""

blas_backend: Final[
    Literal[
        "MKL-ILP64",
        "MKL-LP64",
        "OpenBLAS-ILP64",
        "OpenBLAS-LP64",
        "Accelerate",
        "Accelerate-ILP64",
        "BLIS",
        "BLIS-ILP64",
        "jlinalg-own",
        "system-BLAS-ILP64",
        "system-BLAS-LP64",
    ]
]
"""Active dgemm backend. ILP64 > jlinalg-own > LP64 (LP64 not wired by default)."""

blas_is_ilp64: Final[int]
"""1 if the active dgemm backend uses ILP64 (64-bit int) parameters, 0 otherwise."""

HAS_OPENMP: Final[bool]
"""True if the extension was compiled with OpenMP support.

Level 1/2 kernels are single-threaded; Level 3 (dgemm, dsyrk, dsyr2k) uses
OpenMP parallel-for over the IC loop when OpenMP is available.
"""

# Blocking parameters (ISA-dependent, set by jlinalg_init()).
JLINALG_MR: Final[int]
JLINALG_NR: Final[int]
JLINALG_KC: Final[int]
JLINALG_MC: Final[int]
JLINALG_NC: Final[int]

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
    transa: Literal["N", "T", "n", "t"] = ...,
    transb: Literal["N", "T", "n", "t"] = ...,
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

def eigh(
    K: npt.NDArray[np.float64],
    inplace: bool = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute eigenvalues and eigenvectors of a symmetric matrix.

    K is overwritten as scratch (Householder vectors from dsytrd).

    Args:
        K: Symmetric matrix, shape (N, N), float64, C-contiguous.

    Returns:
        Tuple of (eigenvalues, eigenvectors) where eigenvalues is shape (N,)
        ascending, eigenvectors is shape (N, N) with columns as unit eigenvectors.

    Raises:
        ValueError: If K is not 2-D square float64.
        numpy.linalg.LinAlgError: If convergence fails.
        RuntimeError: If illegal argument detected (internal jlinalg bug).
        MemoryError: If workspace allocation fails.
    """
    ...

def eigh_factored(
    K: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Factored eigendecomposition: dsytrd + dstedc without dormtr.

    K is overwritten in-place with Householder vectors (lower triangle).

    Args:
        K: Symmetric matrix, shape (N, N), float64, C-contiguous.

    Returns:
        Tuple of (eigenvalues, tau, V) where:
        - eigenvalues: shape (N,), ascending
        - tau: shape (N-1,), Householder scalars
        - V: shape (N, N), tridiagonal eigenvectors (row-major)

    Raises:
        NotImplementedError: If D&C pipeline not available.
        numpy.linalg.LinAlgError: If convergence fails.
        MemoryError: If workspace allocation fails.
    """
    ...

def rotate_via_householder(
    K_householder: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute V.T @ (Q^T @ target) without forming U.

    Q is encoded in K_householder + tau from eigh_factored.

    Args:
        K_householder: (N, N) Householder vectors from eigh_factored (not modified).
        tau: (N-1,) Householder scalars from eigh_factored.
        V: (N, N) tridiagonal eigenvectors from eigh_factored.
        target: (N, M) columns to rotate, C-contiguous float64.

    Returns:
        Result array, shape (N, M), float64 = V.T @ (Q^T @ target).
    """
    ...

def get_n_threads() -> int:
    """Get the current jlinalg thread count for Level 3 operations."""
    ...

def set_n_threads(n: int) -> int:
    """Set the jlinalg thread count for Level 3 operations.

    Clamped to the init-time maximum (prevents packed_A out-of-bounds).

    Args:
        n: Desired thread count (must be >= 1).

    Returns:
        Previous thread count.

    Raises:
        ValueError: If n < 1.
    """
    ...
