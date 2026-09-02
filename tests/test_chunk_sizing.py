"""Chunk-size policy for the association pass: ``compute_chunk_size_numpy``.

One seam, previously spread over three files. The sizer turns a RAM budget and
a dispatch path into a SNPs-per-chunk count, and these pin all three of its
inputs: the per-path column accounting, the _MAX_CHUNK ceiling against the
n_filtered bound, and the pipeline_buffers guards it shares with the streaming
memory estimator.
"""

from __future__ import annotations

import pytest

from jamma.lmm.chunk_sizing import (
    _MAX_CHUNK,
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


def test_compute_chunk_size_minimum():
    """Chunk size never drops below 100."""
    # Huge n_samples to force small chunk_from_memory, tiny n_filtered to avoid cap
    chunk = compute_chunk_size_numpy(
        n_samples=1_000_000,
        n_filtered=200,
        n_cvt=10,
        dispatch=DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(2e9),
    )
    assert chunk >= 100, f"Chunk {chunk} below minimum 100"


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

    Every C path is in the fused family and hands ``utg_t`` straight to its
    kernel, so all four size identically at one column per SNP. The NumPy
    fallback materialises the whole six-column table (at n_cvt=1).

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

    fused = [
        size(DispatchPath.FUSED),
        size(DispatchPath.FUSED_GENERAL),
        size(DispatchPath.FUSED_SCORE_WS),
        size(DispatchPath.FUSED_LRT_WS),
    ]
    assert len(set(fused)) == 1, f"fused family must size alike, got {fused}"

    # 1 column vs 6 ((n_cvt+3)(n_cvt+2)/2 at n_cvt=1).
    # Floor division: the sizer truncates budget/bytes_per_snp.
    assert size(DispatchPath.NUMPY_FALLBACK) == fused[0] // 6


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
# pipeline_buffers invariants, shared with the streaming memory estimator
# ---------------------------------------------------------------------------


class TestStreamingMemoryPipelineBuffers:
    """Tests for pipeline_buffers parameter in streaming memory estimators."""

    def test_streaming_memory_double_buffer_rotation_doubles(self):
        """The LMM phase gains one more rotation buffer at pipeline_buffers=2."""
        from jamma.core.eigen_plan import array_gb
        from jamma.core.memory import estimate_streaming_memory

        n_samples, chunk_size = 1000, 10_000
        ledger_1 = estimate_streaming_memory(
            n_samples, chunk_size=chunk_size, pipeline_buffers=1
        )
        ledger_2 = estimate_streaming_memory(
            n_samples, chunk_size=chunk_size, pipeline_buffers=2
        )

        assert ledger_2.lmm_gb - ledger_1.lmm_gb == pytest.approx(
            array_gb(n_samples, chunk_size), rel=1e-10
        )
        assert ledger_2.kinship_gb == ledger_1.kinship_gb
        assert ledger_2.eigen_gb == ledger_1.eigen_gb

    def test_streaming_memory_default_matches_single_buffer(self):
        """Omitting pipeline_buffers gives the same ledger as pipeline_buffers=1."""
        from jamma.core.memory import estimate_streaming_memory

        assert estimate_streaming_memory(1000) == estimate_streaming_memory(
            1000, pipeline_buffers=1
        )

    @pytest.mark.parametrize("bad_value", [0, -1, -10])
    def test_streaming_memory_pipeline_buffers_invalid_raises(self, bad_value):
        """pipeline_buffers < 1 raises ValueError in memory estimators."""
        from jamma.core.memory import estimate_streaming_memory

        with pytest.raises(ValueError, match="pipeline_buffers must be >= 1"):
            estimate_streaming_memory(1000, pipeline_buffers=bad_value)

    @pytest.mark.parametrize("bad_value", [1.0, "2", None])
    def test_streaming_memory_pipeline_buffers_type_error(self, bad_value):
        """pipeline_buffers must be int in memory estimators."""
        from jamma.core.memory import estimate_streaming_memory

        with pytest.raises(TypeError, match="pipeline_buffers must be an int"):
            estimate_streaming_memory(1000, pipeline_buffers=bad_value)

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

    @pytest.mark.parametrize("bad_value", [1.0, "2", None])
    def test_numpy_chunk_size_pipeline_buffers_type_error(self, bad_value):
        """pipeline_buffers must be int in NumPy chunk sizer."""
        from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
        from jamma.lmm.dispatch import DispatchPath

        with pytest.raises(TypeError, match="pipeline_buffers must be an int"):
            compute_chunk_size_numpy(
                n_samples=1000,
                n_filtered=50_000,
                dispatch=DispatchPath.FUSED,
                mem_budget_bytes=int(2e9),
                pipeline_buffers=bad_value,
            )
