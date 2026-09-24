"""Contract tests for ``prepare_genotypes`` over every dataset shape it serves.

``matrix`` is the in-memory batch path, ``plink`` the streaming path, and
``plink+stats`` the LOCO path, whose statistics come from an earlier pass over
one chromosome's columns. The oracle is bed-reader read directly.
"""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass

import numpy as np
import pytest

from jamma.genotype.dataset import GenotypeDataset
from jamma.genotype.snp_stats import (
    SnpFilterSpec,
    SnpStats,
    collect_snp_stats_from_chunks,
)
from jamma.genotype.variants import SnpMeta
from jamma.lmm.genotype_source import PreparedGenotypes, SampleBasis
from jamma.lmm.runner_numpy import prepare_genotypes
from tests.builders import read_plink_genotypes
from tests.fixture_paths import LOCO, SYNTHETIC
from tests.support import require_fixture

pytestmark = pytest.mark.tier1


@dataclass(frozen=True, slots=True)
class _PrepareCase:
    dataset: GenotypeDataset
    samples: SampleBasis
    filters: SnpFilterSpec
    stats: SnpStats | None
    expected_genotypes: np.ndarray
    expected_selection: np.ndarray
    expected_rs: np.ndarray
    source_matrix: np.ndarray | None = None


def _sample_basis(n_rows: int, positions: np.ndarray) -> SampleBasis:
    mask = np.zeros(n_rows, dtype=bool)
    mask[positions] = True
    return SampleBasis.from_mask(mask)


def _restrict(selected: np.ndarray) -> SnpFilterSpec:
    return SnpFilterSpec(
        maf_threshold=0.0, miss_threshold=1.0, restrict_indices=selected
    )


def _take_variants(variants: SnpMeta, columns: np.ndarray) -> SnpMeta:
    return SnpMeta(
        chr=variants.chr[columns],
        rs=variants.rs[columns],
        pos=variants.pos[columns],
        a1=variants.a1[columns],
        a0=variants.a0[columns],
    )


def _prepare(case: _PrepareCase) -> PreparedGenotypes:
    return prepare_genotypes(
        case.dataset, case.samples, case.filters, stats=case.stats, stats_block_size=17
    )


@pytest.fixture(params=("matrix", "plink", "plink+stats"))
def prepare_case(request: pytest.FixtureRequest) -> _PrepareCase:
    require_fixture(
        SYNTHETIC.bed,
        SYNTHETIC.bim,
        SYNTHETIC.fam,
        LOCO.bed,
        LOCO.bim,
        LOCO.fam,
    )
    positions = np.array([0, 2, 5, 9, 12, 20, 33, 50, 72, 99], dtype=np.intp)

    if request.param == "matrix":
        plink = GenotypeDataset.open_plink(SYNTHETIC.bfile)
        source_columns = np.array([3, 7, 10, 15, 20], dtype=np.intp)
        source_matrix = read_plink_genotypes(SYNTHETIC.bfile)[:, source_columns]
        source_matrix[5, 2] = np.nan
        selected = np.array([0, 2, 4], dtype=np.intp)
        return _PrepareCase(
            dataset=GenotypeDataset.from_matrix(
                source_matrix, _take_variants(plink.variants, source_columns)
            ),
            samples=_sample_basis(plink.n_samples, positions),
            filters=_restrict(selected),
            stats=None,
            expected_genotypes=source_matrix[np.ix_(positions, selected)],
            expected_selection=selected,
            expected_rs=np.array(["rs0003", "rs0010", "rs0020"]),
            source_matrix=source_matrix,
        )

    if request.param == "plink":
        dataset = GenotypeDataset.open_plink(SYNTHETIC.bfile)
        selected = np.array([3, 10, 20], dtype=np.intp)
        return _PrepareCase(
            dataset=dataset,
            samples=_sample_basis(dataset.n_samples, positions),
            filters=_restrict(selected),
            stats=None,
            expected_genotypes=read_plink_genotypes(SYNTHETIC.bfile)[
                np.ix_(positions, selected)
            ],
            expected_selection=selected,
            expected_rs=np.array(["rs0003", "rs0010", "rs0020"]),
        )

    dataset = GenotypeDataset.open_plink(LOCO.bfile)
    chromosome_indices = next(iter(dataset.partitions.values()))
    selected = chromosome_indices[np.array([3, 10, 21], dtype=np.intp)]
    physical_rows = np.array([1, 8, 10, 31, 63], dtype=np.intp)
    return _PrepareCase(
        dataset=dataset,
        samples=_sample_basis(dataset.n_samples, physical_rows),
        filters=_restrict(selected),
        stats=dataset.stats(physical_rows, columns=chromosome_indices, block_size=17),
        expected_genotypes=read_plink_genotypes(LOCO.bfile)[
            np.ix_(physical_rows, selected)
        ],
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


def test_prepared_genotypes_bind_rows_statistics_identity_and_chunks(
    prepare_case: _PrepareCase,
) -> None:
    original = (
        None
        if prepare_case.source_matrix is None
        else prepare_case.source_matrix.copy()
    )

    prepared = _prepare(prepare_case)

    assert prepared.analyzed_sample_count == prepare_case.expected_genotypes.shape[0]
    assert prepared.n_filtered == prepare_case.expected_genotypes.shape[1]
    assert prepared.n_unexpected == 0
    np.testing.assert_array_equal(
        prepared.selection.indices, prepare_case.expected_selection
    )

    expected_means = np.nanmean(prepare_case.expected_genotypes, axis=0)
    expected_missing = np.count_nonzero(
        np.isnan(prepare_case.expected_genotypes), axis=0
    )
    np.testing.assert_allclose(prepared.selection.filtered_means, expected_means)
    np.testing.assert_allclose(prepared.selection.filtered_afs, expected_means / 2.0)
    np.testing.assert_array_equal(prepared.selection.filtered_miss, expected_missing)

    actual_ids = prepared.snp_meta.rs[prepared.selection.indices]
    np.testing.assert_array_equal(actual_ids, prepare_case.expected_rs)

    expected_start = 0
    observed: list[np.ndarray] = []
    for chunk in prepared.chunks(2):
        assert chunk.filtered_start == expected_start
        assert chunk.filtered_end > chunk.filtered_start
        assert chunk.genotypes.dtype == np.float64
        assert chunk.genotypes.flags.c_contiguous
        assert chunk.genotypes.flags.writeable
        expected = prepare_case.expected_genotypes[
            :, chunk.filtered_start : chunk.filtered_end
        ]
        np.testing.assert_array_equal(chunk.genotypes, expected)
        observed.append(chunk.genotypes.copy())
        chunk.genotypes[0, 0] = -123.0
        expected_start = chunk.filtered_end

    assert expected_start == prepared.n_filtered
    np.testing.assert_array_equal(
        np.concatenate(observed, axis=1), prepare_case.expected_genotypes
    )
    if original is not None:
        np.testing.assert_array_equal(prepare_case.source_matrix, original)


def test_prepared_chunks_restream_the_same_values(
    prepare_case: _PrepareCase,
) -> None:
    """Each phenotype group re-reads the chunk stream; it must not be spent."""
    prepared = _prepare(prepare_case)

    first = [chunk.genotypes.copy() for chunk in prepared.chunks(2)]
    second = [chunk.genotypes.copy() for chunk in prepared.chunks(3)]

    np.testing.assert_array_equal(
        np.concatenate(first, axis=1), np.concatenate(second, axis=1)
    )


def _tiny_matrix_dataset(matrix: np.ndarray) -> GenotypeDataset:
    n = matrix.shape[1]
    return GenotypeDataset.from_matrix(
        matrix,
        SnpMeta(
            chr=np.array(["view"] * n),
            rs=np.array([f"local-{i}" for i in range(n)]),
            pos=np.arange(10, 10 * (n + 1), 10),
            a1=np.array(["A"] * n),
            a0=np.array(["G"] * n),
        ),
    )


def test_matrix_reuse_does_not_reuse_analyzed_rows() -> None:
    matrix = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
        ]
    )
    dataset = _tiny_matrix_dataset(matrix)
    filters = SnpFilterSpec(maf_threshold=0.0, miss_threshold=1.0)

    first = prepare_genotypes(
        dataset, _sample_basis(4, np.array([0, 1, 2], dtype=np.intp)), filters
    )
    second = prepare_genotypes(
        dataset, _sample_basis(4, np.array([1, 2, 3], dtype=np.intp)), filters
    )

    np.testing.assert_allclose(first.selection.filtered_means, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(second.selection.filtered_means, [5 / 3, 5 / 3, 2 / 3])
    second_chunk = next(second.chunks(3), None)
    assert second_chunk is not None
    np.testing.assert_array_equal(
        second_chunk.genotypes,
        matrix[np.ix_(np.array([1, 2, 3], dtype=np.intp), second.selection.indices)],
    )


def test_plink_statistics_equal_one_float32_block() -> None:
    """Blocks of 17 give the statistics of the whole float32 matrix, bit for bit."""
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)
    dataset = GenotypeDataset.open_plink(SYNTHETIC.bfile)
    positions = np.array([0, 2, 5, 9, 12, 20, 33, 50, 72, 99], dtype=np.intp)
    dense = read_plink_genotypes(SYNTHETIC.bfile)[positions, :]

    prepared = prepare_genotypes(
        dataset,
        _sample_basis(dataset.n_samples, positions),
        SnpFilterSpec(maf_threshold=0.0, miss_threshold=1.0),
        stats_block_size=17,
    )
    expected = collect_snp_stats_from_chunks(
        [(dense, 0, dataset.n_variants)],
        n_snps=dataset.n_variants,
        n_samples=len(positions),
    )

    np.testing.assert_array_equal(
        prepared.selection.filtered_means,
        expected.col_means[prepared.selection.local_indices],
        strict=True,
    )


def test_matrix_hard_calls_are_validated_without_changing_the_selection() -> None:
    """A from_matrix dataset is HARD_CALLS, so off-grid values are counted
    and logged; the statistics and selection are what they would be anyway."""
    matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.5, 1.0, 1.0],
            [2.0, 1.5, 0.0],
            [1.0, 0.0, 2.0],
        ]
    )
    filters = SnpFilterSpec(maf_threshold=0.0, miss_threshold=1.0)
    samples = SampleBasis.from_mask(np.ones(4, dtype=bool))

    prepared = prepare_genotypes(_tiny_matrix_dataset(matrix), samples, filters)

    assert prepared.n_unexpected == 2
    np.testing.assert_array_equal(
        prepared.selection.filtered_means, matrix.mean(axis=0), strict=True
    )
    np.testing.assert_array_equal(prepared.selection.indices, [0, 1, 2])


def test_row_count_must_match_the_dataset() -> None:
    dataset = _tiny_matrix_dataset(np.zeros((4, 2)))

    with pytest.raises(ValueError, match="must match the dataset rows"):
        prepare_genotypes(
            dataset,
            SampleBasis.from_mask(np.ones(5, dtype=bool)),
            SnpFilterSpec(maf_threshold=0.0, miss_threshold=1.0),
        )


def test_supplied_statistics_without_hwe_counts_refuse_an_hwe_filter() -> None:
    """LOCO statistics carry no HWE counts, so an HWE filter cannot apply."""
    matrix = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
    dataset = _tiny_matrix_dataset(matrix)
    stats = dataset.stats(None)

    with pytest.raises(ValueError, match="requires HWE counts"):
        prepare_genotypes(
            dataset,
            SampleBasis.from_mask(np.ones(3, dtype=bool)),
            SnpFilterSpec(maf_threshold=0.0, miss_threshold=1.0, hwe_threshold=1e-6),
            stats=stats,
        )


@pytest.mark.parametrize("subset", [False, True])
def test_chunk_stream_holds_nothing_between_chunks(subset: bool) -> None:
    """A consumed chunk is freed before the next is read, rows cut or not."""
    n, m = 1000, 500
    matrix = np.random.default_rng(13).integers(0, 3, (n, m)).astype(np.float32)
    variants = SnpMeta(
        np.full(m, "1"),
        np.arange(m).astype(str),
        np.arange(m),
        np.full(m, "A"),
        np.full(m, "G"),
    )
    positions = np.arange(0, n, 2) if subset else np.arange(n)
    prepared = prepare_genotypes(
        GenotypeDataset.from_matrix(matrix, variants),
        SampleBasis(positions, n),
        SnpFilterSpec(0.0, 1.0),
    )
    tracemalloc.start()
    try:
        chunks = prepared.chunks(m // 2)
        first = next(chunks)
        first_nbytes = first.genotypes.nbytes
        del first
        between = tracemalloc.get_traced_memory()[0]
        second = next(chunks)
    finally:
        tracemalloc.stop()
    assert between < first_nbytes // 10
    np.testing.assert_array_equal(second.genotypes, matrix[positions, m // 2 :])
    assert second.genotypes.flags.c_contiguous
