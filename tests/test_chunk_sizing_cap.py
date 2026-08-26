"""Pin _MAX_CHUNK so a test can tell the cap from the n_filtered bound.

test_runner_numpy.py::test_compute_chunk_size_large_dataset passes
n_filtered=200_000, the same value as _MAX_CHUNK, so it cannot distinguish
"capped by _MAX_CHUNK" from "capped by n_filtered". These tests set
n_filtered well above the cap and control the RAM budget directly, so each
assertion pins exactly one of the two bounds.
"""

from __future__ import annotations

import pytest

from jamma.core import memory
from jamma.lmm.chunk_sizing import _MAX_CHUNK, compute_chunk_size_numpy
from jamma.lmm.dispatch import DispatchPath

pytestmark = pytest.mark.tier0

_N_SAMPLES = 1000
_N_CVT = 1
_BYTES_PER_SNP = 48_000  # n_samples * n_index(n_cvt=1) * 8, NUMPY_FALLBACK


def test_chunk_size_capped_by_max_chunk(monkeypatch):
    """A generous RAM budget still caps the chunk at _MAX_CHUNK.

    n_filtered sits far above _MAX_CHUNK, and available RAM is set high
    enough that the budget-derived chunk would otherwise exceed it, so the
    cap is the only thing that can produce this result.
    """
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 1_000_000.0)

    chunk = compute_chunk_size_numpy(
        n_samples=_N_SAMPLES,
        n_filtered=_MAX_CHUNK * 3,
        n_cvt=_N_CVT,
        dispatch=DispatchPath.NUMPY_FALLBACK,
    )

    assert chunk == _MAX_CHUNK


def test_chunk_size_bound_by_ram_budget_below_cap(monkeypatch):
    """A tight RAM budget binds below _MAX_CHUNK, not at it.

    available_ram_gb is small enough that 15% of it, floored at the 2 GB
    minimum budget, yields a budget-derived chunk well under both
    _MAX_CHUNK and n_filtered.
    """
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 20.0)

    chunk = compute_chunk_size_numpy(
        n_samples=_N_SAMPLES,
        n_filtered=_MAX_CHUNK * 3,
        n_cvt=_N_CVT,
        dispatch=DispatchPath.NUMPY_FALLBACK,
    )

    mem_budget = max(2_000_000_000, min(int(20.0 * 1e9 * 0.15), 40_000_000_000))
    expected = mem_budget // _BYTES_PER_SNP

    assert chunk == expected
    assert chunk < _MAX_CHUNK
