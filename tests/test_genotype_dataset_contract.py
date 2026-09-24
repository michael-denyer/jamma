"""Contract tests for ``GenotypeDataset`` over its matrix, PLINK and BGEN readers.

The oracle is bed-reader read directly and the jlinalg SNP-statistics kernel
run on the dense row subset, never the streaming helpers the dataset replaces.
The BGEN case encodes the same PLINK fileset as one-hot probabilities, so the
bed-reader oracle holds for it too.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from bed_reader import open_bed

from jamma.genotype.dataset import (
    GenotypeBlock,
    GenotypeDataset,
    GenotypeEncoding,
    SampleTable,
)
from jamma.genotype.variants import SnpMeta
from jamma.io.plink import validate_genotype_values
from jamma.jlinalg import compute_snp_stats_chunk
from tests.bgen_files import one_hot_bgen_from_plink
from tests.fixture_paths import LOCO, SYNTHETIC
from tests.support import require_fixture, requires_c

pytestmark = pytest.mark.tier0


@dataclass(frozen=True)
class _Case:
    dataset: GenotypeDataset
    bfile: Path
    dense64: np.ndarray
    # What dataset.stats() streams: float32 from a .bed, the matrix otherwise.
    stats_input: np.ndarray
    chromosome: np.ndarray


def _bed_oracle(bfile: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open_bed(Path(f"{bfile}.bed")) as bed:
        return (
            bed.read(dtype=np.float64),
            bed.read(dtype=np.float32),
            np.asarray(bed.chromosome),
        )


def _variants(bfile: Path) -> SnpMeta:
    with open_bed(Path(f"{bfile}.bed")) as bed:
        return SnpMeta(
            chr=np.asarray(bed.chromosome).astype(str),
            rs=bed.sid,
            pos=np.asarray(bed.bp_position, dtype=np.int64),
            a1=bed.allele_1,
            a0=bed.allele_2,
        )


@pytest.fixture(params=("synthetic", "loco", "asymmetric"))
def bfile(request: pytest.FixtureRequest) -> Path:
    if request.param == "synthetic":
        require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)
        return SYNTHETIC.bfile
    if request.param == "loco":
        require_fixture(LOCO.bed, LOCO.bim, LOCO.fam)
        return LOCO.bfile
    # Carries NaNs, which the committed fixtures do not.
    return request.getfixturevalue("asymmetric_plink")


@pytest.fixture(params=("matrix", "plink", pytest.param("bgen", marks=requires_c)))
def case(request: pytest.FixtureRequest, bfile: Path, tmp_path: Path) -> _Case:
    dense64, dense32, chromosome = _bed_oracle(bfile)
    if request.param == "matrix":
        dataset = GenotypeDataset.from_matrix(dense64.copy(), _variants(bfile))
        return _Case(dataset, bfile, dense64, dense64, chromosome)
    if request.param == "bgen":
        files = one_hot_bgen_from_plink(bfile, tmp_path / "onehot.bgen")
        dataset = GenotypeDataset.open_bgen(files.bgen, files.sample, files.bgi)
        return _Case(dataset, bfile, dense64, dense64, chromosome)
    dataset = GenotypeDataset.open_plink(bfile)
    return _Case(dataset, bfile, dense64, dense32, chromosome)


def _column_cases(n: int) -> dict[str, np.ndarray | None]:
    rng = np.random.default_rng(11)
    scattered = np.sort(rng.choice(n, size=n // 3, replace=False))
    scattered[0], scattered[-1] = 0, n - 1
    return {
        "unfiltered": None,
        "contiguous": np.arange(5, n - 3, dtype=np.intp),
        "scattered": np.unique(scattered).astype(np.intp),
    }


def _rows(n_samples: int) -> np.ndarray:
    rng = np.random.default_rng(5)
    return np.sort(rng.choice(n_samples, size=n_samples // 2 + 1, replace=False))


def _kernel_stats(dense: np.ndarray) -> tuple[np.ndarray, ...]:
    k = dense.shape[1]
    means, variances = np.zeros(k), np.zeros(k)
    miss = np.zeros(k, dtype=np.intp)
    n_aa, n_ab, n_bb = (np.zeros(k, dtype=np.int64) for _ in range(3))
    compute_snp_stats_chunk(dense, means, miss, variances, n_aa, n_ab, n_bb)
    return means, miss, variances, n_aa, n_ab, n_bb


def _assert_stats_equal(
    stats, dense: np.ndarray, columns: np.ndarray, *, hwe: bool = True
) -> None:
    """Match the kernel; HWE counts and hard-call validation only when ``hwe``.

    ``hwe`` is False for a PROBABILITIES dataset, which has neither.
    """
    means, miss, variances, n_aa, n_ab, n_bb = _kernel_stats(dense)
    np.testing.assert_array_equal(stats.col_means, means, strict=True)
    np.testing.assert_array_equal(stats.miss_counts, miss, strict=True)
    np.testing.assert_array_equal(stats.col_vars, variances, strict=True)
    if hwe:
        assert stats.hwe_counts is not None
        np.testing.assert_array_equal(stats.hwe_counts.n_aa, n_aa, strict=True)
        np.testing.assert_array_equal(stats.hwe_counts.n_ab, n_ab, strict=True)
        np.testing.assert_array_equal(stats.hwe_counts.n_bb, n_bb, strict=True)
        assert stats.n_unexpected == validate_genotype_values(dense)
    else:
        assert stats.hwe_counts is None
        assert stats.n_unexpected == 0
    assert stats.n_samples == dense.shape[0]
    np.testing.assert_array_equal(stats.global_indices, columns)


@pytest.mark.parametrize("columns_case", ["unfiltered", "contiguous", "scattered"])
@pytest.mark.parametrize("block_size", [1, 7, 64, 10_000])
def test_blocks_equal_bed_reader(case: _Case, columns_case: str, block_size: int):
    """Concatenated dosages equal a direct bed-reader read, NaNs included."""
    n = case.dataset.n_variants
    columns = _column_cases(n)[columns_case]
    expected_cols = np.arange(n) if columns is None else columns

    seen_cols, seen_values, next_start = [], [], 0
    for block in case.dataset.blocks(block_size, columns=columns):
        assert block.start == next_start
        assert block.end - block.start == len(block.columns) <= block_size
        next_start = block.end
        seen_cols.append(block.columns)
        values = block.dosages()
        assert values.dtype == np.float64
        seen_values.append(values)

    assert next_start == len(expected_cols)
    np.testing.assert_array_equal(np.concatenate(seen_cols), expected_cols)
    got = np.concatenate(seen_values, axis=1)
    expected = case.dense64[:, expected_cols]
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_array_equal(got, expected, strict=True)


def test_asymmetric_fixture_has_missing_values(asymmetric_plink: Path):
    """Guard: the NaN-pattern assertions above are vacuous without NaNs."""
    dense64, _, _ = _bed_oracle(asymmetric_plink)
    assert np.isnan(dense64).any()


def test_dosages_narrow_rows_and_columns(case: _Case):
    """dosages(rows, columns) equals the dense subset for each narrowing."""
    rows = _rows(case.dataset.n_samples)
    local = np.array([0, 2, 3], dtype=np.intp)
    blocks = case.dataset.blocks(
        5, columns=_column_cases(case.dataset.n_variants)["scattered"]
    )
    for narrowing, block in zip(("rows", "columns", "both"), blocks, strict=False):
        dense = case.dense64[:, block.columns]
        if narrowing == "rows":
            got, expected = block.dosages(rows=rows), dense[rows, :]
        elif narrowing == "columns":
            got, expected = block.dosages(columns=local), dense[:, local]
        else:
            got = block.dosages(rows, local)
            expected = dense[np.ix_(rows, local)]
        np.testing.assert_array_equal(got, expected, strict=True)


@pytest.mark.parametrize("columns_case", ["unfiltered", "scattered"])
def test_block_stats_equal_kernel_on_row_subset(case: _Case, columns_case: str):
    """block.stats(rows, hwe=...) equals the kernel on the dense subset."""
    rows = _rows(case.dataset.n_samples)
    columns = _column_cases(case.dataset.n_variants)[columns_case]
    hwe = case.dataset.encoding.supports_hwe
    for block in case.dataset.blocks(13, columns=columns):
        dense = case.dense64[np.ix_(rows, block.columns)]
        _assert_stats_equal(block.stats(rows, hwe=hwe), dense, block.columns, hwe=hwe)
        full = block.stats(hwe=hwe)
        dense = case.dense64[:, block.columns]
        _assert_stats_equal(full, dense, block.columns, hwe=hwe)


@pytest.mark.parametrize("columns_case", ["unfiltered", "contiguous", "scattered"])
def test_dataset_stats_equal_kernel_on_row_subset(case: _Case, columns_case: str):
    """dataset.stats streams the reader's stats dtype and matches the kernel."""
    rows = _rows(case.dataset.n_samples)
    n = case.dataset.n_variants
    columns = _column_cases(n)[columns_case]
    expected_cols = np.arange(n) if columns is None else columns
    hwe = case.dataset.encoding.supports_hwe

    stats = case.dataset.stats(rows, columns=columns, hwe=hwe, block_size=7)

    dense = case.stats_input[np.ix_(rows, expected_cols)]
    _assert_stats_equal(stats, dense, expected_cols, hwe=hwe)


def test_stats_count_values_outside_hard_calls():
    """n_unexpected counts non-{0,1,2} values; HWE classes skip them."""
    genotypes = np.array(
        [[0.0, 1.0, 2.0], [0.5, 1.0, np.nan], [2.0, 1.5, 0.0], [1.0, 0.0, 3.0]]
    )
    variants = SnpMeta(
        chr=np.array(["1", "1", "2"]),
        rs=np.array(["a", "b", "c"]),
        pos=np.array([1, 2, 3], dtype=np.int64),
        a1=np.array(["A", "A", "A"]),
        a0=np.array(["G", "G", "G"]),
    )
    dataset = GenotypeDataset.from_matrix(genotypes, variants)

    stats = dataset.stats(None, hwe=True, block_size=2)

    assert stats.n_unexpected == 3
    assert stats.hwe_counts is not None
    np.testing.assert_array_equal(stats.hwe_counts.n_aa, [1, 1, 1])
    np.testing.assert_array_equal(stats.hwe_counts.n_ab, [1, 2, 0])
    np.testing.assert_array_equal(stats.hwe_counts.n_bb, [1, 0, 1])


def test_partitions_match_per_chromosome_flatnonzero(case: _Case):
    """partitions equals np.flatnonzero per chromosome, first-appearance order."""
    expected: dict[str, np.ndarray] = {}
    for chrom in case.chromosome:
        expected.setdefault(str(chrom), np.flatnonzero(case.chromosome == chrom))

    partitions = case.dataset.partitions

    assert list(partitions) == list(expected)
    for chrom, indices in expected.items():
        np.testing.assert_array_equal(partitions[chrom], indices)


def test_partitions_follow_first_appearance_not_sort_order():
    """Chromosome order is file order, so '10' before '2' stays put."""
    chrom = np.array(["10", "10", "2", "1", "2"])
    variants = SnpMeta(
        chr=chrom,
        rs=np.array(["a", "b", "c", "d", "e"]),
        pos=np.arange(5, dtype=np.int64),
        a1=np.full(5, "A"),
        a0=np.full(5, "G"),
    )
    dataset = GenotypeDataset.from_matrix(np.zeros((2, 5)), variants)

    assert list(dataset.partitions) == ["10", "2", "1"]
    np.testing.assert_array_equal(dataset.partitions["2"], [2, 4])


def test_spent_block_raises(case: _Case):
    """dosages() spends the block; stats and a second dosages both raise."""
    block = next(case.dataset.blocks(4))
    block.stats()
    block.dosages()

    with pytest.raises(RuntimeError, match="spent"):
        block.stats()
    with pytest.raises(RuntimeError, match="spent"):
        block.dosages()


def test_plink_fingerprint_equals_eigen_cache_components(bfile: Path):
    """PLINK fingerprint() is exactly the eigen cache's file components.

    The expected values are built here the way master's eigen cache built
    them from the bed path, since the cache key now reads fingerprint().
    """
    dataset = GenotypeDataset.open_plink(bfile)
    bed = Path(f"{bfile}.bed")
    st = bed.stat()

    assert dataset.fingerprint() == {
        "bed_fingerprint": f"{bed.name}:{st.st_size}:{st.st_mtime_ns}",
        "bim_sha256": hashlib.sha256(Path(f"{bfile}.bim").read_bytes()).hexdigest(),
    }


def test_matrix_fingerprint_raises():
    variants = SnpMeta(
        chr=np.array(["1"]),
        rs=np.array(["a"]),
        pos=np.array([1], dtype=np.int64),
        a1=np.array(["A"]),
        a0=np.array(["G"]),
    )
    dataset = GenotypeDataset.from_matrix(np.zeros((2, 1)), variants)

    with pytest.raises(ValueError, match="no file fingerprint"):
        dataset.fingerprint()


def test_plink_identity_matches_bed_reader(bfile: Path):
    """open_plink carries bed-reader's samples and variants, HARD_CALLS."""
    dataset = GenotypeDataset.open_plink(bfile)

    with open_bed(Path(f"{bfile}.bed")) as bed:
        np.testing.assert_array_equal(dataset.samples.fid, bed.fid)
        np.testing.assert_array_equal(dataset.samples.iid, bed.iid)
        np.testing.assert_array_equal(dataset.variants.rs, bed.sid)
        np.testing.assert_array_equal(dataset.variants.a1, bed.allele_1)
        np.testing.assert_array_equal(dataset.variants.a0, bed.allele_2)
        np.testing.assert_array_equal(dataset.variants.pos, bed.bp_position)
        assert (dataset.n_samples, dataset.n_variants) == (
            bed.iid_count,
            bed.sid_count,
        )
    assert dataset.encoding is GenotypeEncoding.HARD_CALLS


def test_open_plink_rejects_truncated_bed(tmp_path: Path, asymmetric_plink: Path):
    """open_plink checks the .bed size against .fam and .bim counts."""
    for ext in (".bim", ".fam"):
        (tmp_path / f"cut{ext}").write_bytes(
            asymmetric_plink.with_suffix(ext).read_bytes()
        )
    (tmp_path / "cut.bed").write_bytes(
        asymmetric_plink.with_suffix(".bed").read_bytes()[:-1]
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        GenotypeDataset.open_plink(tmp_path / "cut")


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (np.array([3, 3]), "strictly ascending"),
        (np.array([4, 2]), "strictly ascending"),
        (np.array([-1, 2]), "out of bounds"),
        (np.array([0, 10**6]), "out of bounds"),
    ],
)
def test_blocks_reject_invalid_columns(case: _Case, columns: np.ndarray, message: str):
    with pytest.raises(ValueError, match=message):
        next(case.dataset.blocks(4, columns=columns))


def test_blocks_reject_block_size_below_one(case: _Case):
    with pytest.raises(ValueError, match="block_size"):
        next(case.dataset.blocks(0))


def test_encoding_capabilities():
    hard, probs = GenotypeEncoding.HARD_CALLS, GenotypeEncoding.PROBABILITIES
    assert (hard.supports_hwe, hard.supports_info, hard.validates_hard_calls) == (
        True,
        False,
        True,
    )
    assert (probs.supports_hwe, probs.supports_info, probs.validates_hard_calls) == (
        False,
        True,
        False,
    )


def test_probability_block_rejects_hwe_and_skips_validation():
    """A PROBABILITIES block refuses HWE counts and reports no unexpected values."""
    values = np.array([[0.25, 1.0], [1.5, np.nan], [2.0, 0.75]])
    block = GenotypeBlock(
        np.array([0, 1]), 0, 2, values, GenotypeEncoding.PROBABILITIES
    )

    with pytest.raises(ValueError, match="HWE"):
        block.stats(hwe=True)
    assert block.stats().n_unexpected == 0


def test_materialize_keeps_values_identity_and_float32_stats(bfile: Path):
    """materialize() reads once into memory; blocks and stats are unchanged."""
    dense64, dense32, _ = _bed_oracle(bfile)
    opened = GenotypeDataset.open_plink(bfile)

    dataset = opened.materialize()

    assert dataset.encoding is GenotypeEncoding.HARD_CALLS
    assert dataset.samples is opened.samples
    assert dataset.variants is opened.variants
    (block,) = dataset.blocks(dataset.n_variants)
    np.testing.assert_array_equal(block.dosages(), dense64, strict=True)
    rows = _rows(dataset.n_samples)
    _assert_stats_equal(
        dataset.stats(rows, hwe=True, block_size=7),
        dense32[rows, :],
        np.arange(dataset.n_variants),
    )
    with pytest.raises(ValueError, match="no file fingerprint"):
        dataset.fingerprint()


class _UnreadableReader:
    """A reader that fails the test if anything reads from it."""

    def read(self, columns, block_size, *, stats_only):
        raise AssertionError("materialize() must refuse before reading")

    def fingerprint(self) -> dict[str, str]:
        raise AssertionError("not called")


def test_materialize_rejects_probabilities():
    """float32 would round fractional dosages, so only hard calls materialize."""
    ids = np.array(["s0", "s1"])
    dataset = GenotypeDataset(
        _UnreadableReader(),
        GenotypeEncoding.PROBABILITIES,
        SampleTable(fid=ids, iid=ids),
        SnpMeta(
            chr=np.array(["1"]),
            rs=np.array(["a"]),
            pos=np.array([1], dtype=np.int64),
            a1=np.array(["A"]),
            a0=np.array(["G"]),
        ),
    )

    with pytest.raises(ValueError, match="only hard-call datasets"):
        dataset.materialize()
