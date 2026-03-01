"""Type stubs for the _eigen_accel C extension."""

import numpy as np
import numpy.typing as npt

ABI_VERSION: int

def eigh_dsyevr(
    K: npt.NDArray[np.float64],
    uplo: str = ...,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
