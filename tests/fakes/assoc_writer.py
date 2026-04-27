"""In-memory fake for jamma.lmm.io.IncrementalAssocWriter."""

from __future__ import annotations

import numpy as np


class FakeAssocWriter:
    """In-memory stand-in for ``IncrementalAssocWriter``.

    Captures ``write_arrays_batch`` calls so tests can assert on call count
    and arguments without ``MagicMock``. Accessing an attribute that does
    not exist raises ``AttributeError`` — detecting interface drift the
    moment a method is renamed.
    """

    def __init__(self) -> None:
        self.batches: list[tuple] = []

    @property
    def call_count(self) -> int:
        return len(self.batches)

    def write_arrays_batch(
        self,
        lmm_mode: int,
        snp_indices: np.ndarray,
        snp_info: list,
        afs: np.ndarray,
        miss_counts: np.ndarray,
        arrays: dict[str, np.ndarray],
    ) -> None:
        self.batches.append((lmm_mode, snp_indices, snp_info, afs, miss_counts, arrays))
