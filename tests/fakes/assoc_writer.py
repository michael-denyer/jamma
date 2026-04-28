"""In-memory fake for jamma.lmm.io.IncrementalAssocWriter."""

from __future__ import annotations

import numpy as np


class FakeAssocWriter:
    """In-memory stand-in for ``IncrementalAssocWriter``.

    Captures ``write_arrays_batch`` arguments on ``self.batches`` so tests
    can assert on the recorded payload (per-batch tuple of all positional
    args). Accessing an attribute that does not exist raises
    ``AttributeError`` — detecting interface drift the moment a method is
    renamed. Tests should assert on ``len(writer.batches)`` and the tuple
    contents, not on a separate counter — the real
    ``IncrementalAssocWriter.count`` tracks rows written, not batches, so
    a ``call_count`` property here would shadow the production name with
    different semantics.
    """

    def __init__(self) -> None:
        self.batches: list[tuple] = []

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
