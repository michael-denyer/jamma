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

jblas_isa: Final[Literal["AVX2", "generic"]]
"""Active ISA name: "AVX2" or "generic"."""

HAS_OPENMP: Final[bool]
"""True if the extension was compiled with OpenMP support.

Currently informational only — Level 1/2 kernels are single-threaded.
OpenMP parallelism is planned for dgemm (Level 3) in a future phase.
"""

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

# NOTE: dgemm is not yet exported by the C extension. It is provided as a
# NumPy fallback via __init__.py.  Do not add a stub here until the C
# implementation is wired into pymodule.c.
