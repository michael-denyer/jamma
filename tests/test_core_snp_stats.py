"""Tests for shared streamed SNP statistics and filtering."""

import numpy as np
import pytest

from jamma.core.snp_filter import compute_snp_stats
from jamma.core.snp_stats import (
    HweCounts,
    SnpFilterSpec,
    SnpStats,
    collect_snp_stats_from_chunks,
    filter_snp_stats,
)


@pytest.mark.tier0
def test_collect_snp_stats_from_chunks_matches_shared_stats_kernel():
    genotypes = np.array(
        [
            [0.0, 1.0, 2.0, np.nan],
            [1.0, np.nan, 2.0, 0.0],
            [2.0, 1.0, np.nan, 1.0],
        ],
        dtype=np.float64,
    )
    chunks = [(genotypes[:, :2], 0, 2), (genotypes[:, 2:], 2, 4)]

    stats = collect_snp_stats_from_chunks(
        chunks,
        n_snps=4,
        n_samples=3,
        include_hwe=True,
        validate_genotypes=True,
    )
    expected_means, expected_miss, expected_vars = compute_snp_stats(genotypes)

    np.testing.assert_allclose(stats.col_means, expected_means)
    np.testing.assert_array_equal(stats.miss_counts, expected_miss)
    np.testing.assert_allclose(stats.col_vars, expected_vars)
    assert stats.n_samples == 3
    assert stats.n_unexpected == 0
    assert stats.hwe_counts is not None
    assert stats.hwe_counts.n_aa.shape == (4,)


@pytest.mark.tier0
def test_filter_snp_stats_uses_stats_sample_count_as_denominator():
    stats = SnpStats(
        col_means=np.array([1.0, 1.0]),
        miss_counts=np.array([1, 3]),
        col_vars=np.array([1.0, 1.0]),
        n_samples=4,
    )

    selection = filter_snp_stats(
        stats,
        SnpFilterSpec(maf_threshold=0.0, miss_threshold=0.5),
    )

    np.testing.assert_array_equal(selection.indices, [0])
    np.testing.assert_array_equal(selection.filtered_miss, [1])


@pytest.mark.tier0
def test_filter_snp_stats_restricts_subset_by_global_indices():
    stats = SnpStats(
        col_means=np.ones(4),
        miss_counts=np.zeros(4, dtype=np.intp),
        col_vars=np.ones(4),
        n_samples=10,
        global_indices=np.array([10, 12, 15, 18]),
    )

    selection = filter_snp_stats(
        stats,
        SnpFilterSpec(
            maf_threshold=0.0,
            miss_threshold=1.0,
            restrict_indices=np.array([12, 18]),
            restrict_label="test SNP list",
        ),
    )

    np.testing.assert_array_equal(selection.indices, [12, 18])
    np.testing.assert_array_equal(selection.local_indices, [1, 3])


@pytest.mark.tier0
def test_filter_snp_stats_applies_hwe_counts():
    stats = SnpStats(
        col_means=np.array([1.0, 1.0]),
        miss_counts=np.array([0, 0]),
        col_vars=np.array([1.0, 1.0]),
        n_samples=100,
        hwe_counts=HweCounts(
            n_aa=np.array([25, 50]),
            n_ab=np.array([50, 0]),
            n_bb=np.array([25, 50]),
        ),
    )

    selection = filter_snp_stats(
        stats,
        SnpFilterSpec(
            maf_threshold=0.0,
            miss_threshold=1.0,
            hwe_threshold=0.001,
        ),
    )

    np.testing.assert_array_equal(selection.indices, [0])
    assert selection.n_hwe_removed == 1
