"""Chunk-size policy for the association pass: ``compute_chunk_size_numpy``.

One seam, previously spread over three files. The sizer turns a RAM budget and
a dispatch path into a SNPs-per-chunk count, and these pin all three of its
inputs: the per-path column accounting, the _MAX_CHUNK ceiling against the
n_filtered bound, and its pipeline_buffers guards, beside the rotation buffers
the association quote prices per pipeline buffer.
"""

from __future__ import annotations

import pytest

from jamma.lmm.chunk_sizing import (
    _MAX_CHUNK,
    _MIN_CHUNK,
    chunk_budget_bytes,
    compute_chunk_size_numpy,
)
from jamma.lmm.dispatch import DispatchPath

pytestmark = pytest.mark.tier0

# ---------------------------------------------------------------------------
# Chunk size computation
# ---------------------------------------------------------------------------


def test_compute_chunk_size_small_dataset():
    """Small dataset: chunk size = n_filtered (everything in one chunk)."""
    chunk = compute_chunk_size_numpy(
        n_samples=100,
        n_filtered=500,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    assert chunk == 500, f"Expected 500, got {chunk}"


def test_compute_chunk_size_large_dataset():
    """Large dataset: chunk capped by memory budget or _MAX_CHUNK."""
    chunk = compute_chunk_size_numpy(
        n_samples=10_000,
        n_filtered=200_000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    assert 100 <= chunk <= 200_000, f"Chunk {chunk} outside expected bounds"


def test_compute_chunk_size_zero_bytes():
    """bytes_per_snp=0 (n_samples=0): returns n_filtered directly."""
    chunk = compute_chunk_size_numpy(
        n_samples=0,
        n_filtered=1000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    assert chunk == 1000, f"Expected 1000, got {chunk}"


@pytest.mark.xfail(
    strict=True,
    reason="the sizer plans chunks below the throughput floor instead of "
    "leaving a budget that cannot hold the floor to the preflight",
)
def test_compute_chunk_size_never_plans_below_the_throughput_floor():
    """A budget too tight for 100 SNPs still plans 100.

    Every chunk re-streams the eigenvector matrix through the rotation GEMM,
    so a chunk narrower than the floor trades memory nobody asked to save
    for a run that is slower by the same factor. Whether the floored chunk
    fits is the memory preflight's question, and it refuses the run; the
    sizer never answers it by planning 1-SNP chunks.
    """
    chunk = compute_chunk_size_numpy(
        n_samples=1_000_000,
        n_filtered=200,
        n_cvt=10,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    assert chunk == _MIN_CHUNK


def test_chunk_size_split_larger_than_full():
    """Split Uab accounting produces larger chunks than full Uab."""
    full = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(10e9),
    )
    split = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=int(10e9),
    )
    assert split > full, f"Split chunk ({split}) should exceed full ({full})"


def test_chunk_size_explicit_budget():
    """Explicit mem_budget_bytes overrides auto-scaling."""
    small_budget = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    large_budget = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(20e9),
    )
    assert large_budget > small_budget


def test_chunk_size_pipeline_halves_budget():
    """pipeline_buffers=2 produces roughly half the chunk size."""
    single = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=int(20e9),
    )
    double = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=int(20e9),
        pipeline_buffers=2,
    )
    # Double-buffering halves the budget, so chunk should be ~half
    assert double < single
    assert double >= single // 2 - 1  # allow rounding


def test_chunk_budget_auto_scales_with_memory():
    """Auto-scaled budget uses 15% of available RAM between 2-40 GB bounds."""
    # 400 GB available -> 15% = 60 GB (hits 40 GB ceiling)
    assert chunk_budget_bytes(None, available_bytes=int(400e9)) == 40_000_000_000
    # 10 GB available -> 15% = 1.5 GB (hits 2 GB floor)
    assert chunk_budget_bytes(None, available_bytes=int(10e9)) == 2_000_000_000
    # 100 GB available -> 15 GB, inside both bounds
    assert chunk_budget_bytes(None, available_bytes=int(100e9)) == 15_000_000_000
    # A user ceiling in GB wins outright, whatever the machine has
    assert chunk_budget_bytes(1.5, available_bytes=int(400e9)) == 1_500_000_000

    chunk_big = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=chunk_budget_bytes(None, available_bytes=int(400e9)),
    )
    chunk_small = compute_chunk_size_numpy(
        n_samples=50_000,
        n_filtered=100_000,
        n_cvt=1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=chunk_budget_bytes(None, available_bytes=int(10e9)),
    )
    assert chunk_big > chunk_small


def test_chunk_size_accounting_by_dispatch_path():
    """Each path's column count, named by path rather than by mode.

    The one C path, FUSED, hands ``utg_t`` straight to its kernel, so it sizes
    at one column per SNP. The NumPy fallback materialises the whole
    six-column table (at n_cvt=1).

    This replaced a test that called the sizer three times with identical
    arguments and asserted the three results matched. It could not fail, and
    its "4-col" claim had been wrong since the C-availability flags collapsed
    to one bit: every n_cvt=1 C path had already moved to one column.
    """
    n_samples = 10_000
    budget = int(5e9)

    def size(dispatch):
        return compute_chunk_size_numpy(
            n_samples=n_samples,
            n_filtered=500_000,
            n_cvt=1,
            dispatch=dispatch,
            mem_budget_bytes=budget,
        )

    fused = size(DispatchPath.FUSED)

    # 1 column vs 6 ((n_cvt+3)(n_cvt+2)/2 at n_cvt=1).
    # Floor division: the sizer truncates budget/bytes_per_snp.
    assert size(DispatchPath.NUMPY_FALLBACK) == fused // 6


# ---------------------------------------------------------------------------
# _MAX_CHUNK ceiling versus the n_filtered bound
#
# test_compute_chunk_size_large_dataset above passes n_filtered=200_000, the
# same value as _MAX_CHUNK, so it cannot distinguish "capped by _MAX_CHUNK"
# from "capped by n_filtered". These set n_filtered well above the cap and
# control the RAM budget directly, so each assertion isolates one bound.
# ---------------------------------------------------------------------------

_N_SAMPLES = 1000
_N_CVT = 1
_BYTES_PER_SNP = 48_000  # n_samples * n_index(n_cvt=1) * 8, NUMPY_FALLBACK


def test_chunk_size_capped_by_max_chunk():
    """A generous RAM budget still caps the chunk at _MAX_CHUNK.

    n_filtered sits far above _MAX_CHUNK, and available RAM is set high
    enough that the budget-derived chunk would otherwise exceed it, so the
    cap is the only thing that can produce this result. This asserts the cap
    exists as a code path distinct from n_filtered, not the cap's value.
    """
    chunk = compute_chunk_size_numpy(
        n_samples=_N_SAMPLES,
        n_filtered=_MAX_CHUNK * 3,
        n_cvt=_N_CVT,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=chunk_budget_bytes(None, available_bytes=int(1e15)),
    )

    assert chunk == _MAX_CHUNK


def test_chunk_size_bound_by_ram_budget_below_cap():
    """A tight RAM budget binds below _MAX_CHUNK, not at it.

    Available RAM is small enough that 15% of it, floored at the 2 GB
    minimum budget, yields a budget-derived chunk well under both
    _MAX_CHUNK and n_filtered.
    """
    mem_budget = chunk_budget_bytes(None, available_bytes=int(20e9))

    chunk = compute_chunk_size_numpy(
        n_samples=_N_SAMPLES,
        n_filtered=_MAX_CHUNK * 3,
        n_cvt=_N_CVT,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=mem_budget,
    )

    expected = mem_budget // _BYTES_PER_SNP

    assert chunk == expected
    assert chunk < _MAX_CHUNK


# ---------------------------------------------------------------------------
# pipeline_buffers invariants
# ---------------------------------------------------------------------------


class TestPipelineBuffers:
    """pipeline_buffers in the association quote and the chunk sizer."""

    def test_streaming_quote_gains_one_rotation_buffer_per_pipeline_buffer(self):
        """The association quote gains one more rotation buffer at n_buffers=2."""
        from jamma.core.memory import array_gb
        from tests.builders import association_price_plan

        n_samples, chunk_size = 1000, 10_000
        quote_1, quote_2 = (
            association_price_plan(
                "streaming",
                n_samples=n_samples,
                n_snps=100_000,
                chunk_size=chunk_size,
                n_buffers=n_buffers,
            ).price(eigen=None)
            for n_buffers in (1, 2)
        )

        assert quote_2.association_gb - quote_1.association_gb == pytest.approx(
            array_gb(n_samples, chunk_size), rel=1e-10
        )
        assert quote_2.statistics_gb == quote_1.statistics_gb

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_numpy_chunk_size_pipeline_buffers_invalid_raises(self, bad_value):
        """pipeline_buffers < 1 raises ValueError in NumPy chunk sizer."""
        from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
        from jamma.lmm.dispatch import DispatchPath

        with pytest.raises(ValueError, match="pipeline_buffers must be >= 1"):
            compute_chunk_size_numpy(
                n_samples=1000,
                n_filtered=50_000,
                dispatch=DispatchPath.FUSED,
                mem_budget_bytes=int(2e9),
                pipeline_buffers=bad_value,
            )
