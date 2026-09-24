"""Filtered streaming kinship reads the genotype file once and returns its accumulator.

Reads are observed at ``bed_reader.open_bed``, the boundary every genotype
stream crosses. A statistics pass that reaches the reader through some other
module binding still shows up here as a second sweep over the SNP columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.genotype.dataset import GenotypeDataset
from jamma.genotype.snp_filter import compute_snp_filter_mask, compute_snp_stats
from jamma.io import load_plink_binary, plink
from jamma.kinship import compute_kinship_streaming, stream
from tests.reference.kinship import (
    compute_centered_kinship,
    compute_standardized_kinship,
)

pytestmark = pytest.mark.tier0

MAF = 0.05
MISS = 0.1
FILTER_ROWS = np.arange(40)
ALL_ROWS = np.arange(80)
ONE_SWEEP_OF_61_SNPS_BY_7 = [
    (0, 7),
    (7, 14),
    (14, 21),
    (21, 28),
    (28, 35),
    (35, 42),
    (42, 49),
    (49, 56),
    (56, 61),
]


def _column_range(index) -> tuple[int, int]:
    columns = index[1]
    if isinstance(columns, slice):
        return (columns.start, columns.stop)
    return (int(columns[0]), int(columns[-1]) + 1)


class _RecordingBed:
    """Wrap an open bed-reader handle so each ``read`` logs its column range."""

    def __init__(self, bed, reads: list[tuple[int, int]]):
        self._bed = bed
        self._reads = reads

    def __enter__(self):
        self._bed.__enter__()
        return self

    def __exit__(self, *exc):
        return self._bed.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._bed, name)

    def read(self, index, **kwargs):
        self._reads.append(_column_range(index))
        return self._bed.read(index=index, **kwargs)


def _spy_genotype_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Record the SNP-column range of every genotype read; reading stays real."""
    reads: list[tuple[int, int]] = []
    real_open_bed = plink.open_bed
    monkeypatch.setattr(
        plink, "open_bed", lambda path: _RecordingBed(real_open_bed(path), reads)
    )
    return reads


def _genotypes(bfile: Path) -> np.ndarray:
    return load_plink_binary(bfile).genotypes.astype(np.float64)


def _kept_columns(genotypes: np.ndarray, rows: np.ndarray) -> np.ndarray:
    means, misses, variances = compute_snp_stats(genotypes[rows])
    keep, _afs, _mafs = compute_snp_filter_mask(
        means, misses, variances, len(rows), MAF, MISS
    )
    return keep


def _reference_kinship(
    mode: str, genotypes: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    reference = (
        compute_centered_kinship if mode == "centered" else compute_standardized_kinship
    )
    return reference(
        genotypes[:, keep], maf_threshold=0.0, miss_threshold=1.0, check_memory=False
    )


@pytest.mark.parametrize("mode", ["centered", "standardized"])
def test_subset_filtered_kinship_reads_the_bed_once(
    asymmetric_plink, monkeypatch, mode
):
    genotypes = _genotypes(asymmetric_plink)
    keep = _kept_columns(genotypes, FILTER_ROWS)
    keep_all = _kept_columns(genotypes, ALL_ROWS)
    assert np.flatnonzero(keep_all & ~keep).tolist() == [0, 57]
    reads = _spy_genotype_reads(monkeypatch)

    K = compute_kinship_streaming(
        GenotypeDataset.open_plink(asymmetric_plink),
        maf_threshold=MAF,
        miss_threshold=MISS,
        check_memory=False,
        show_progress=False,
        mode=mode,
        filter_sample_indices=FILTER_ROWS,
        chunk_size=7,
    )

    assert reads == ONE_SWEEP_OF_61_SNPS_BY_7
    expected = _reference_kinship(mode, genotypes, keep)
    np.testing.assert_allclose(K, expected, rtol=1e-12, atol=1e-14)


def test_ksnps_restriction_is_applied_inside_the_single_read(
    asymmetric_plink, monkeypatch
):
    genotypes = _genotypes(asymmetric_plink)
    keep = _kept_columns(genotypes, ALL_ROWS)
    keep[1::2] = False
    reads = _spy_genotype_reads(monkeypatch)

    K = compute_kinship_streaming(
        GenotypeDataset.open_plink(asymmetric_plink),
        maf_threshold=MAF,
        miss_threshold=MISS,
        check_memory=False,
        show_progress=False,
        ksnps_indices=np.arange(0, 61, 2),
        chunk_size=7,
    )

    assert reads == ONE_SWEEP_OF_61_SNPS_BY_7
    expected = _reference_kinship("centered", genotypes, keep)
    np.testing.assert_allclose(K, expected, rtol=1e-12, atol=1e-14)


def test_out_of_range_ksnps_index_is_rejected(asymmetric_plink):
    with pytest.raises(ValueError, match=r"-ksnps index 61 out of range for 61 SNPs"):
        compute_kinship_streaming(
            GenotypeDataset.open_plink(asymmetric_plink),
            ksnps_indices=np.array([0, 61]),
            check_memory=False,
            show_progress=False,
        )


@pytest.mark.parametrize("filtered", [False, True])
def test_streaming_kinship_returns_its_accumulator(
    asymmetric_plink, monkeypatch, filtered
):
    accumulators: list[np.ndarray] = []
    real_accumulate = stream.accumulate_kinship

    def recording(K: np.ndarray, X: np.ndarray) -> None:
        accumulators.append(K)
        real_accumulate(K, X)

    monkeypatch.setattr(stream, "accumulate_kinship", recording)

    K = compute_kinship_streaming(
        GenotypeDataset.open_plink(asymmetric_plink),
        maf_threshold=MAF if filtered else 0.0,
        miss_threshold=MISS if filtered else 1.0,
        filter_sample_indices=FILTER_ROWS if filtered else None,
        check_memory=False,
        show_progress=False,
    )

    assert np.shares_memory(K, accumulators[0])


def test_no_surviving_snp_raises_with_the_filter_in_the_message(asymmetric_plink):
    with pytest.raises(ValueError, match=r"No SNPs passed filtering \(maf>=0.6"):
        compute_kinship_streaming(
            GenotypeDataset.open_plink(asymmetric_plink),
            maf_threshold=0.6,
            check_memory=False,
            show_progress=False,
        )
