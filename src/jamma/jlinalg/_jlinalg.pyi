"""Type stub for the jlinalg C extension (_jlinalg).

Provides IDE completion and pyrefly checking for the compiled extension module.
When the C extension is not available, jamma.jlinalg.__init__ provides NumPy-backed
fallbacks with identical signatures.

ABI 12: Level 1/2 BLAS (ddot, dnrm2, daxpy, dscal, dgemv) and dsyr2k were
removed from the C extension. ABI 17 removed their NumPy fallback
implementations too; jamma.jlinalg still defines NumPy fallbacks for
dgemm, dsyrk, eigh, compute_snp_stats_chunk, and the thread-count helpers.
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
        "OpenBLAS-ILP64",
        "Accelerate-ILP64",
        "system-BLAS-ILP64",
        "numpy-fallback",
    ]
]
"""Resolved vendor BLAS/LAPACK library backing dsyrk and eigh.

Under JLINALG_NO_VENDOR_DGEMM this names the ILP64 vendor library while
dgemm itself is left unwired and falls back to NumPy; check
blas_has_dgemm for whether dgemm actually dispatches to this backend.
"""

blas_is_ilp64: Final[int]
"""1 if the active dgemm backend uses ILP64 (64-bit int) parameters, 0 otherwise."""

HAS_OPENMP: Final[bool]
"""True if the extension was compiled with OpenMP support."""

# BLAS capability flags (set during init based on vendor detection).
blas_has_dgemm: Final[int]
blas_has_dsyrk: Final[int]
blas_has_dsyevd: Final[int]
blas_has_dsyevr: Final[int]
blas_has_lapacke_dsyevd: Final[int]

def dgemm(
    A: npt.NDArray[np.floating],
    B: npt.NDArray[np.floating],
    transa: Literal["N", "T", "n", "t"] = ...,
    transb: Literal["N", "T", "n", "t"] = ...,
    out: npt.NDArray[np.float64] | None = ...,
) -> npt.NDArray[np.float64]:
    """Compute matrix-matrix product with optional transpose.

    Args:
        A: Left matrix, C-contiguous. Any float dtype; non-float64 input is
            coerced to float64 (a copy).
        B: Right matrix, C-contiguous. Coerced like A.
        transa: 'N' (no transpose) or 'T' (transpose A).
        transb: 'N' (no transpose) or 'T' (transpose B).
        out: Optional preallocated (M, N) float64 C-contiguous buffer. When
            given, the result is written into it and the same array returned.

    Returns:
        Result matrix C = op(A) @ op(B), float64.

    Raises:
        RuntimeError: If no vendor dgemm is wired (blas_has_dgemm == 0).
            jamma.jlinalg binds the NumPy dgemm in that case.
    """

def dsyrk(
    X: npt.NDArray[np.float64],
    *,
    out: npt.NDArray[np.float64] | None = None,
    beta: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Compute symmetric rank-k update: C = X @ X.T + beta*C.

    Args:
        X: Input matrix, shape (N, K), float64, C-contiguous.
        out: Optional writable, aligned, C-contiguous output buffer, shape
            (N, N), float64.
        beta: Scale applied to the existing output. Requires out when nonzero.

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

def compute_snp_stats_chunk(
    data: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    means: npt.NDArray[np.float64],
    miss_counts: npt.NDArray[np.intp],
    variances: npt.NDArray[np.float64],
    n_aa: npt.NDArray[np.int64] | None = ...,
    n_ab: npt.NDArray[np.int64] | None = ...,
    n_bb: npt.NDArray[np.int64] | None = ...,
) -> None:
    """Compute per-SNP mean, variance, and missing count into preallocated arrays.

    Single-pass per-column statistics. Optionally counts genotype values
    (0, 1, 2) for HWE testing when n_aa/n_ab/n_bb are all provided.

    Args:
        data: Genotype matrix (n_samples, n_snps), float32 or float64, C-contiguous.
        means: Output (n_snps,) float64 — per-SNP mean.
        miss_counts: Output (n_snps,) intp — per-SNP NaN count.
        variances: Output (n_snps,) float64 — per-SNP population variance.
        n_aa: Optional output (n_snps,) int64 — count of genotype 0 (None skips HWE).
        n_ab: Optional output (n_snps,) int64 — count of genotype 1 (None skips HWE).
        n_bb: Optional output (n_snps,) int64 — count of genotype 2 (None skips HWE).
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
