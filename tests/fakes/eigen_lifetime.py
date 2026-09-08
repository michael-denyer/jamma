"""Resource observers that read and decompose real eigen matrices."""

from pathlib import Path
from weakref import ReferenceType, ref

import numpy as np

from jamma.jlinalg import EighStatus
from jamma.lmm.eigen_io import read_eigen_files
from tests.fakes.jlinalg import FakeJlinalg


class LifetimeCheckedJlinalg(FakeJlinalg):
    """Use real NumPy eigenvalues, refusing overlap with the previous output."""

    previous: ReferenceType[np.ndarray] | None = None

    def eigh(
        self, K: np.ndarray, inplace: bool = False, driver: str = "auto"
    ) -> tuple[np.ndarray, np.ndarray, EighStatus]:
        if self.previous is not None:
            assert self.previous() is None, "previous U is live before decomposition"
        result = super().eigh(K, inplace=inplace, driver=driver)
        self.previous = ref(result[1])
        return result


class LifetimeCheckedEigenReader:
    """Read actual files, refusing to load while the previous U remains live."""

    previous: ReferenceType[np.ndarray] | None = None

    def __call__(
        self,
        eigenD_path: Path,
        eigenU_path: Path,
        n_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.previous is not None:
            assert self.previous() is None, "previous U is live before cache read"
        result = read_eigen_files(eigenD_path, eigenU_path, n_samples=n_samples)
        self.previous = ref(result[1])
        return result
