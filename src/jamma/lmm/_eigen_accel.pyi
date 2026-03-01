"""Type stubs for the _eigen_accel C extension.

WARNING: eigh_dsyevr overwrites the input matrix K.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt

ABI_VERSION: int

def eigh_dsyevr(
    K: npt.NDArray[np.float64],
    uplo: Literal["L", "U", "l", "u"] = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
