"""Regression tests verifying memory estimation uses computed chunk size."""

from unittest.mock import patch

from jamma.core.memory import check_memory_before_run, estimate_streaming_memory
from jamma.lmm.chunk import _compute_chunk_size


def test_chunk_size_varies_with_scale():
    """_compute_chunk_size returns different values for different scales."""
    small = _compute_chunk_size(1410, 12_000)
    large = _compute_chunk_size(100_000, 500_000)
    assert small != large, "Chunk size should vary with scale"


def test_memory_estimate_uses_computed_chunk():
    """Memory estimates differ when using computed chunk sizes at different scales."""
    small_chunk = _compute_chunk_size(1410, 12_000)
    large_chunk = _compute_chunk_size(100_000, 500_000)

    est_small = estimate_streaming_memory(1410, 12_000, chunk_size=small_chunk)
    est_large = estimate_streaming_memory(100_000, 500_000, chunk_size=large_chunk)

    assert est_small.total_peak_gb != est_large.total_peak_gb


def test_check_memory_before_run_uses_computed_chunk():
    """check_memory_before_run calls _compute_chunk_size internally."""
    with patch(
        "jamma.lmm.chunk._compute_chunk_size", wraps=_compute_chunk_size
    ) as mock:
        try:
            check_memory_before_run(1410, 12_000)
        except MemoryError:
            pass  # OK if memory insufficient on this machine
        mock.assert_called_once_with(1410, 12_000)
