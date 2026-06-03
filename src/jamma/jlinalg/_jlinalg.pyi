"""Type stub for the jlinalg C extension (_jlinalg).

Provides IDE completion and mypy checking for the compiled extension module.
When the C extension is not available, jamma.jlinalg.__init__ provides NumPy-backed
fallbacks with identical signatures.

ABI 12: Level 1/2 BLAS (ddot, dnrm2, daxpy, dscal, dgemv) and dsyr2k were
removed from the C extension. They are NumPy-only in __init__.py.
"""

from typing import Final, Literal, overload

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
        "system-BLAS-ILP64",
        "system-BLAS-LP64",
        "numpy-fallback",
    ]
]
"""Active dgemm backend. ILP64 vendor > numpy-fallback."""

blas_is_ilp64: Final[int]
"""1 if the active dgemm backend uses ILP64 (64-bit int) parameters, 0 otherwise."""

HAS_OPENMP: Final[bool]
"""True if the extension was compiled with OpenMP support."""

# BLAS capability flags (set during init based on vendor detection).
blas_has_dsyrk: Final[int]
blas_has_dsyevd: Final[int]
blas_has_dsyevr: Final[int]
blas_has_dgeqrf: Final[int]
blas_has_dgesvd: Final[int]
blas_has_lapacke_dsyevd: Final[int]

def dgemm(
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    transa: Literal["N", "T", "n", "t"] = ...,
    transb: Literal["N", "T", "n", "t"] = ...,
    out: npt.NDArray[np.float64] | None = ...,
) -> npt.NDArray[np.float64]:
    """Compute matrix-matrix product with optional transpose.

    Args:
        A: Left matrix, float64, C-contiguous.
        B: Right matrix, float64, C-contiguous.
        transa: 'N' (no transpose) or 'T' (transpose A).
        transb: 'N' (no transpose) or 'T' (transpose B).
        out: Optional preallocated (M, N) float64 C-contiguous buffer. When
            given, the result is written into it and the same array returned.

    Returns:
        Result matrix C = op(A) @ op(B), float64.
    """

def dsyrk(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute symmetric rank-k update: C = X @ X.T.

    Args:
        X: Input matrix, shape (N, K), float64, C-contiguous.

    Returns:
        Symmetric result matrix, shape (N, N), float64.
    """

def eigh(
    K: npt.NDArray[np.float64],
    inplace: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute eigenvalues and eigenvectors of a symmetric matrix.

    K is overwritten as scratch by vendor DSYEVD.

    Args:
        K: Symmetric matrix, shape (N, N), float64, C-contiguous.
        inplace: If True, overwrite K with eigenvectors (saves memory).

    Returns:
        Tuple of (eigenvalues, eigenvectors) where eigenvalues is shape (N,)
        ascending, eigenvectors is shape (N, N) with columns as unit eigenvectors.

    Raises:
        ValueError: If K is not 2-D square float64.
        numpy.linalg.LinAlgError: If convergence fails.
        RuntimeError: If illegal argument detected (internal jlinalg bug).
        MemoryError: If workspace allocation fails.
    """

def qr(
    A: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute QR decomposition of a matrix.

    Args:
        A: Input matrix, shape (M, N), float64, C-contiguous.

    Returns:
        Tuple of (Q, R).
    """

@overload
def svd(
    A: npt.NDArray[np.float64], compute_uv: Literal[True] = ...
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]: ...
@overload
def svd(
    A: npt.NDArray[np.float64], compute_uv: Literal[False]
) -> npt.NDArray[np.float64]: ...
def svd(
    A: npt.NDArray[np.float64], compute_uv: bool = ...
) -> (
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
    | npt.NDArray[np.float64]
):
    """Compute reduced SVD of a tall-skinny matrix (m >= n).

    Args:
        A: Input matrix, shape (M, N) with M >= N, float64, C-contiguous.
        compute_uv: If True (default), return (U, S, Vt). If False, return S only.

    Returns:
        (U, S, Vt) when compute_uv is True, else the singular values S.
    """

def compute_snp_stats_chunk(
    genotypes: npt.NDArray[np.float64],
    n_samples: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute per-SNP MAF, missingness, and allele frequencies.

    Args:
        genotypes: Genotype matrix chunk, shape (n_samples, n_snps), float64.
        n_samples: Number of samples.

    Returns:
        Tuple of (maf, miss_rate, af) arrays, each shape (n_snps,).
    """

def get_n_threads() -> int:
    """Get the current jlinalg thread count for Level 3 operations."""

def set_n_threads(n: int) -> int:
    """Set the jlinalg thread count for Level 3 operations.

    Args:
        n: Desired thread count (must be >= 1).

    Returns:
        Previous thread count.

    Raises:
        ValueError: If n < 1.
    """
