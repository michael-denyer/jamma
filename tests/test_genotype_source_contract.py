"""Shared contract tests for prepared LMM genotype sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from jamma.core.snp_stats import (
    SnpFilterSpec,
    SnpStatsCache,
    collect_streamed_snp_stats,
)
from jamma.io import load_plink_binary
from jamma.io.plink import get_plink_metadata, partitions_from_metadata
from jamma.lmm.genotype_source import GenotypeSource, PreparedGenotypes, SampleBasis
from jamma.lmm.loco import _LocoChrSource
from jamma.lmm.runner_numpy import MatrixSource
from jamma.lmm.runner_numpy_streaming import BedSource
from jamma.lmm.schema import SnpMeta
from tests.conftest import require_fixture
from tests.fixture_paths import LOCO, SYNTHETIC

pytestmark = pytest.mark.tier1


@dataclass(frozen=True, slots=True)
class _SourceCase:
    source: GenotypeSource
    samples: SampleBasis
    filters: SnpFilterSpec
    expected_genotypes: np.ndarray
    expected_selection: np.ndarray
    expected_rs: np.ndarray
    source_matrix: np.ndarray | None = None


def _sample_basis(n_rows: int, positions: np.ndarray) -> SampleBasis:
    mask = np.zeros(n_rows, dtype=bool)
    mask[positions] = True
    return SampleBasis.from_mask(mask)


@pytest.fixture(params=("matrix", "bed", "loco"))
def source_case(request: pytest.FixtureRequest) -> _SourceCase:
    require_fixture(
        SYNTHETIC.bed,
        SYNTHETIC.bim,
        SYNTHETIC.fam,
        LOCO.bed,
        LOCO.bim,
        LOCO.fam,
    )

    if request.param == "matrix":
        plink = load_plink_binary(SYNTHETIC.bfile)
        source_columns = np.array([3, 7, 10, 15, 20], dtype=np.intp)
        source_matrix = plink.genotypes[:, source_columns].copy()
        source_matrix[5, 2] = np.nan
        positions = np.array([0, 2, 5, 9, 12, 20, 33, 50, 72, 99], dtype=np.intp)
        selected = np.array([0, 2, 4], dtype=np.intp)
        return _SourceCase(
            source=MatrixSource(
                source_matrix,
                SnpMeta.from_plink_meta(plink.meta, source_columns),
            ),
            samples=_sample_basis(plink.n_samples, positions),
            filters=SnpFilterSpec(
                maf_threshold=0.0,
                miss_threshold=1.0,
                restrict_indices=selected,
            ),
            expected_genotypes=source_matrix[np.ix_(positions, selected)],
            expected_selection=selected,
            expected_rs=np.array(["rs0003", "rs0010", "rs0020"]),
            source_matrix=source_matrix,
        )

    if request.param == "bed":
        plink = load_plink_binary(SYNTHETIC.bfile)
        positions = np.array([0, 2, 5, 9, 12, 20, 33, 50, 72, 99], dtype=np.intp)
        selected = np.array([3, 10, 20], dtype=np.intp)
        return _SourceCase(
            source=BedSource(
                SYNTHETIC.bfile,
                snp_meta=SnpMeta.from_plink_meta(plink.meta),
                n_samples=plink.n_samples,
                n_snps=plink.n_snps,
                stats_chunk_size=17,
                validate_genotypes=True,
                show_progress=False,
            ),
            samples=_sample_basis(plink.n_samples, positions),
            filters=SnpFilterSpec(
                maf_threshold=0.0,
                miss_threshold=1.0,
                restrict_indices=selected,
            ),
            expected_genotypes=plink.genotypes[np.ix_(positions, selected)],
            expected_selection=selected,
            expected_rs=np.array(["rs0003", "rs0010", "rs0020"]),
        )

    plink = load_plink_binary(LOCO.bfile)
    partitions = partitions_from_metadata(plink.meta)
    chromosome_indices = next(iter(partitions.values()))
    selected = chromosome_indices[np.array([3, 10, 21], dtype=np.intp)]
    source_rows = np.array([1, 4, 8, 10, 15, 22, 31, 47, 63, 88], dtype=np.intp)
    local_positions = np.array([0, 2, 3, 6, 8], dtype=np.intp)
    physical_rows = source_rows[local_positions]
    return _SourceCase(
        source=_LocoChrSource(
            LOCO.bfile,
            chromosome_indices,
            source_rows,
            snp_meta=SnpMeta.from_plink_meta(plink.meta),
            col_chunk_size=17,
            snp_stats_cache=None,
        ),
        samples=_sample_basis(len(source_rows), local_positions),
        filters=SnpFilterSpec(
            maf_threshold=0.0,
            miss_threshold=1.0,
            restrict_indices=selected,
        ),
        expected_genotypes=plink.genotypes[np.ix_(physical_rows, selected)],
        expected_selection=selected,
        expected_rs=np.array(["rs0003", "rs0010", "rs0021"]),
    )


def test_sample_basis_is_an_immutable_row_coordinate() -> None:
    mask = np.array([True, False, True, True], dtype=bool)

    samples = SampleBasis.from_mask(mask)

    np.testing.assert_array_equal(samples.positions, np.array([0, 2, 3]))
    assert samples.source_row_count == 4
    assert samples.analyzed_sample_count == 3
    assert not samples.is_all_samples
    assert not samples.positions.flags.writeable
    assert SampleBasis.from_mask(np.ones(4, dtype=bool)).is_all_samples


def test_prepared_source_binds_rows_statistics_identity_and_chunks(
    source_case: _SourceCase,
) -> None:
    original = (
        None if source_case.source_matrix is None else source_case.source_matrix.copy()
    )

    prepared = source_case.source.prepare(source_case.samples, source_case.filters)

    assert isinstance(prepared, PreparedGenotypes)
    assert prepared.analyzed_sample_count == source_case.expected_genotypes.shape[0]
    assert prepared.n_filtered == source_case.expected_genotypes.shape[1]
    assert prepared.n_unexpected == 0
    np.testing.assert_array_equal(
        prepared.selection.indices, source_case.expected_selection
    )

    expected_means = np.nanmean(source_case.expected_genotypes, axis=0)
    expected_missing = np.count_nonzero(
        np.isnan(source_case.expected_genotypes), axis=0
    )
    np.testing.assert_allclose(prepared.selection.filtered_means, expected_means)
    np.testing.assert_allclose(prepared.selection.filtered_afs, expected_means / 2.0)
    np.testing.assert_array_equal(prepared.selection.filtered_miss, expected_missing)

    actual_ids = prepared.snp_meta.rs[prepared.selection.indices]
    np.testing.assert_array_equal(actual_ids, source_case.expected_rs)

    chunks = prepared.chunks(2)
    expected_start = 0
    observed: list[np.ndarray] = []
    while (chunk := chunks()) is not None:
        assert chunk.filtered_start == expected_start
        assert chunk.filtered_end > chunk.filtered_start
        assert chunk.genotypes.dtype == np.float64
        assert chunk.genotypes.flags.c_contiguous
        assert chunk.genotypes.flags.writeable
        expected = source_case.expected_genotypes[
            :, chunk.filtered_start : chunk.filtered_end
        ]
        np.testing.assert_array_equal(chunk.genotypes, expected)
        observed.append(chunk.genotypes.copy())
        chunk.genotypes[0, 0] = -123.0
        expected_start = chunk.filtered_end

    assert expected_start == prepared.n_filtered
    np.testing.assert_array_equal(
        np.concatenate(observed, axis=1), source_case.expected_genotypes
    )
    if original is not None:
        np.testing.assert_array_equal(source_case.source_matrix, original)


def test_matrix_source_reuse_does_not_reuse_analyzed_rows() -> None:
    matrix = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
        ]
    )
    meta = SnpMeta(
        chr=np.array(["view", "view", "view"]),
        rs=np.array(["local-0", "local-1", "local-2"]),
        pos=np.array([10, 20, 30]),
        a1=np.array(["A", "A", "A"]),
        a0=np.array(["G", "G", "G"]),
    )
    source = MatrixSource(matrix, meta)
    filters = SnpFilterSpec(maf_threshold=0.0, miss_threshold=1.0)

    first = source.prepare(
        _sample_basis(4, np.array([0, 1, 2], dtype=np.intp)), filters
    )
    second = source.prepare(
        _sample_basis(4, np.array([1, 2, 3], dtype=np.intp)), filters
    )

    np.testing.assert_allclose(first.selection.filtered_means, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(second.selection.filtered_means, [5 / 3, 5 / 3, 2 / 3])
    second_chunk = second.chunks(3)()
    assert second_chunk is not None
    np.testing.assert_array_equal(
        second_chunk.genotypes,
        matrix[np.ix_(np.array([1, 2, 3], dtype=np.intp), second.selection.indices)],
    )


def test_bed_source_statistics_match_float32_streaming_pass() -> None:
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)
    meta = get_plink_metadata(SYNTHETIC.bfile)
    positions = np.array([0, 2, 5, 9, 12, 20, 33, 50, 72, 99], dtype=np.intp)
    samples = _sample_basis(meta.n_samples, positions)
    filters = SnpFilterSpec(maf_threshold=0.0, miss_threshold=1.0)
    source = BedSource(
        SYNTHETIC.bfile,
        snp_meta=SnpMeta.from_plink_meta(meta),
        n_samples=meta.n_samples,
        n_snps=meta.n_snps,
        stats_chunk_size=17,
        validate_genotypes=True,
        show_progress=False,
    )

    prepared = source.prepare(samples, filters)
    expected = collect_streamed_snp_stats(
        SYNTHETIC.bfile,
        n_snps=meta.n_snps,
        n_samples=meta.n_samples,
        chunk_size=17,
        sample_indices=positions,
        validate_genotypes=True,
        show_progress=False,
        dtype=np.float32,
        sample_scope="valid_samples",
    )

    np.testing.assert_array_equal(
        prepared.selection.filtered_means,
        expected.col_means[prepared.selection.local_indices],
    )


def test_loco_source_reuses_cache_only_for_the_full_physical_sample_basis() -> None:
    require_fixture(LOCO.bed, LOCO.bim, LOCO.fam)
    plink = load_plink_binary(LOCO.bfile)
    partitions = partitions_from_metadata(plink.meta)
    chromosome_indices = next(iter(partitions.values()))
    selected = chromosome_indices[np.array([3, 10, 21], dtype=np.intp)]
    distinctive_mean = 0.75
    cache = SnpStatsCache(
        col_means=np.full(plink.n_snps, distinctive_mean),
        miss_counts=np.zeros(plink.n_snps, dtype=np.intp),
        col_vars=np.ones(plink.n_snps),
        n_samples=plink.n_samples,
        global_indices=np.arange(plink.n_snps, dtype=np.intp),
        sample_scope="all_samples",
    )
    source = _LocoChrSource(
        LOCO.bfile,
        chromosome_indices,
        np.arange(plink.n_samples, dtype=np.intp),
        snp_meta=SnpMeta.from_plink_meta(plink.meta),
        col_chunk_size=17,
        snp_stats_cache=cache,
    )
    filters = SnpFilterSpec(
        maf_threshold=0.0,
        miss_threshold=1.0,
        restrict_indices=selected,
    )

    full = source.prepare(
        SampleBasis.from_mask(np.ones(plink.n_samples, dtype=bool)), filters
    )
    subset_positions = np.array([1, 8, 10, 31, 63], dtype=np.intp)
    subset = source.prepare(_sample_basis(plink.n_samples, subset_positions), filters)

    np.testing.assert_array_equal(
        full.selection.filtered_means,
        np.full(len(selected), distinctive_mean),
    )
    np.testing.assert_allclose(
        subset.selection.filtered_means,
        np.mean(plink.genotypes[np.ix_(subset_positions, selected)], axis=0),
    )
    assert np.all(subset.selection.filtered_means != distinctive_mean)
