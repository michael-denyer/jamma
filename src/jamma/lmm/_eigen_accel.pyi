"""Type stubs for the _eigen_accel C extension.

WARNING: eigh_dsyevr overwrites the input matrix K.

LAPACK is resolved at runtime via dlopen — no compile-time dependency.
IS_ILP64 indicates whether ILP64 (dsyevr_64_) or LP64 (dsyevr_) was found.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt

ABI_VERSION: int
IS_ILP64: int

def eigh_dsyevr(
    K: npt.NDArray[np.float64],
    uplo: Literal["L", "U", "l", "u"] = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
