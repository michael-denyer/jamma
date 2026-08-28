"""A stand-in for ``jamma.jlinalg`` at the ``jamma.lmm.eigen`` import site.

``eigen.py`` reads five names from jlinalg: the three detection flags, the
backend label, and ``eigh``. Vendor LAPACK cannot be made to raise
``MemoryError`` on a 20x20 matrix, or to report LP64 on this machine, so
tests of the error and warning paths substitute this object for the module.
Anything ``eigen.py`` reads that is not declared here raises
``AttributeError``, which is what distinguishes it from ``MagicMock``.
``tests/fakes/test_fakes.py`` checks the declared names against the real
module and against every ``jlinalg.<name>`` read in ``eigen.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from jamma.lmm import eigen


@dataclass
class FakeJlinalg:
    """Configurable ``jamma.jlinalg`` surface as seen from ``eigen.py``.

    ``eigh`` raises ``eigh_error`` when set, returns ``eigh_result`` when
    set, and otherwise computes the real ``np.linalg.eigh``.
    """

    blas_has_dsyevd: int = 1
    blas_has_dsyevr: int = 0
    blas_is_ilp64: int = 1
    blas_backend: str = "fake"
    eigh_result: tuple[np.ndarray, np.ndarray] | None = None
    eigh_error: BaseException | None = None

    def eigh(
        self, K: np.ndarray, inplace: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        del inplace
        if self.eigh_error is not None:
            raise self.eigh_error
        if self.eigh_result is not None:
            return self.eigh_result
        return np.linalg.eigh(K)


def use_fake_jlinalg(monkeypatch: pytest.MonkeyPatch, fake: FakeJlinalg) -> FakeJlinalg:
    """Install ``fake`` where ``eigen.py`` looks jlinalg up, for one test."""
    monkeypatch.setattr(
        eigen, "jlinalg", fake
    )  # allow-patch: the one seam a fake jlinalg enters by
    return fake
