"""Regression tests for searchsorted-based SNP chunk filtering.

Validates that np.searchsorted produces identical indices to naive linear scan
for all edge cases. This guards against off-by-one errors in the searchsorted
side parameter choice.
"""

import numpy as np
import pytest


def naive_chunk_filter(
    snp_indices: np.ndarray, file_start: int, file_end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation: linear scan for SNPs in [file_start, file_end).

    Args:
        snp_indices: Sorted array of SNP indices that passed filtering.
        file_start: Start index of the file chunk (inclusive).
        file_end: End index of the file chunk (exclusive).

    Returns:
        Tuple of (local_indices, col_indices) where:
        - local_indices: positions within snp_indices array
        - col_indices: positions within the chunk (snp_idx - file_start)
    """
    local_indices = []
    col_indices = []
    for i, snp_idx in enumerate(snp_indices):
        if file_start <= snp_idx < file_end:
            local_indices.append(i)
            col_indices.append(snp_idx - file_start)
    return np.array(local_indices, dtype=np.intp), np.array(col_indices, dtype=np.intp)


def searchsorted_chunk_filter(
    snp_indices: np.ndarray, file_start: int, file_end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Optimized implementation: binary search for SNPs in [file_start, file_end).

    Args:
        snp_indices: Sorted array of SNP indices that passed filtering.
        file_start: Start index of the file chunk (inclusive).
        file_end: End index of the file chunk (exclusive).

    Returns:
        Tuple of (local_indices, col_indices) where:
        - local_indices: positions within snp_indices array
        - col_indices: positions within the chunk (snp_idx - file_start)
    """
    left = np.searchsorted(snp_indices, file_start, side="left")
    right = np.searchsorted(snp_indices, file_end, side="left")
    local_indices = np.arange(left, right)
    col_indices = snp_indices[left:right] - file_start
    return local_indices, col_indices


def _assert_filters_equal(
    snp_indices: np.ndarray, file_start: int, file_end: int
) -> None:
    """Assert both implementations produce identical results."""
    naive_local, naive_col = naive_chunk_filter(snp_indices, file_start, file_end)
    fast_local, fast_col = searchsorted_chunk_filter(snp_indices, file_start, file_end)
    np.testing.assert_array_equal(fast_local, naive_local)
    np.testing.assert_array_equal(fast_col, naive_col)


@pytest.mark.tier0
class TestSearchsortedChunkFilter:
    """Validate searchsorted produces identical results to naive linear scan."""

    def test_basic_overlap(self):
        """SNPs partially overlap with chunk range."""
        snp_indices = np.array([2, 5, 8, 12, 15])
        _assert_filters_equal(snp_indices, file_start=4, file_end=10)
        # Expected: SNPs 5, 8 are in [4, 10)

        fast_local, fast_col = searchsorted_chunk_filter(snp_indices, 4, 10)
        np.testing.assert_array_equal(fast_local, [1, 2])  # positions of 5, 8
        np.testing.assert_array_equal(fast_col, [1, 4])  # 5-4=1, 8-4=4

    def test_no_overlap(self):
        """No SNPs fall within the chunk range."""
        snp_indices = np.array([2, 5, 8])
        _assert_filters_equal(snp_indices, file_start=10, file_end=20)

        fast_local, fast_col = searchsorted_chunk_filter(snp_indices, 10, 20)
        assert len(fast_local) == 0
        assert len(fast_col) == 0

    def test_full_overlap(self):
        """All SNPs fall within the chunk range."""
        snp_indices = np.array([0, 1, 2, 3, 4])
        _assert_filters_equal(snp_indices, file_start=0, file_end=5)

        fast_local, fast_col = searchsorted_chunk_filter(snp_indices, 0, 5)
        np.testing.assert_array_equal(fast_local, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(fast_col, [0, 1, 2, 3, 4])

    def test_edge_boundary_inclusive_start(self):
        """file_start is inclusive: SNP at file_start should be included."""
        snp_indices = np.array([5, 10])
        _assert_filters_equal(snp_indices, file_start=5, file_end=11)

        fast_local, fast_col = searchsorted_chunk_filter(snp_indices, 5, 11)
        np.testing.assert_array_equal(fast_local, [0, 1])
        np.testing.assert_array_equal(fast_col, [0, 5])

    def test_edge_boundary_exclusive_end(self):
        """file_end is exclusive: SNP at file_end should NOT be included."""
        snp_indices = np.array([5, 10])
        _assert_filters_equal(snp_indices, file_start=5, file_end=10)

        fast_local, fast_col = searchsorted_chunk_filter(snp_indices, 5, 10)
        np.testing.assert_array_equal(fast_local, [0])  # only SNP 5
        np.testing.assert_array_equal(fast_col, [0])

    def test_empty_snp_indices(self):
        """Empty snp_indices array should produce empty results."""
        snp_indices = np.array([], dtype=np.intp)
        _assert_filters_equal(snp_indices, file_start=0, file_end=100)

    def test_single_snp_in_range(self):
        """Single SNP exactly at file_start."""
        snp_indices = np.array([50])
        _assert_filters_equal(snp_indices, file_start=50, file_end=100)

        fast_local, fast_col = searchsorted_chunk_filter(snp_indices, 50, 100)
        np.testing.assert_array_equal(fast_local, [0])
        np.testing.assert_array_equal(fast_col, [0])

    def test_single_snp_at_end_excluded(self):
        """Single SNP exactly at file_end should be excluded."""
        snp_indices = np.array([100])
        _assert_filters_equal(snp_indices, file_start=50, file_end=100)

        fast_local, fast_col = searchsorted_chunk_filter(snp_indices, 50, 100)
        assert len(fast_local) == 0

    def test_large_sorted_array(self):
        """10,000 random sorted indices with random chunk ranges."""
        rng = np.random.default_rng(42)
        # Generate 10,000 unique sorted indices from range [0, 100_000)
        snp_indices = np.sort(rng.choice(100_000, size=10_000, replace=False))

        # Test with 20 random chunk ranges
        for _ in range(20):
            start = rng.integers(0, 90_000)
            end = start + rng.integers(1_000, 20_000)
            _assert_filters_equal(snp_indices, file_start=int(start), file_end=int(end))


@pytest.mark.tier0
@pytest.mark.parametrize(
    "file_start,file_end",
    [
        (0, 1000),  # First chunk
        (1000, 2000),  # Middle chunk
        (9000, 10000),  # Last chunk
        (0, 10000),  # Single chunk covering all
        (4999, 5001),  # Straddling a single SNP
        (500, 500),  # Zero-width range (empty)
        (10000, 20000),  # Beyond all SNPs
    ],
    ids=[
        "first_chunk",
        "middle_chunk",
        "last_chunk",
        "full_range",
        "straddle_single",
        "zero_width",
        "beyond_all",
    ],
)
def test_boundary_parametrized(file_start: int, file_end: int):
    """Parametrized boundary test across a large snp_indices array."""
    # Dense sorted indices: every 2nd index from 0 to 9999
    snp_indices = np.arange(0, 10000, 2)
    _assert_filters_equal(snp_indices, file_start=file_start, file_end=file_end)
