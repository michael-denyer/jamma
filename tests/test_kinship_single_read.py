"""Filtered streaming kinship reads the genotype file once."""

from __future__ import annotations

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship import compute_kinship_streaming, stream
from tests.conftest import require_fixture
from tests.fixture_paths import SYNTHETIC
from tests.reference.kinship import (
    compute_centered_kinship,
    compute_standardized_kinship,
)

pytestmark = pytest.mark.tier0


@pytest.fixture
def count_genotype_reads(monkeypatch):
    """Count BED streams opened by the kinship module; reading is untouched."""
    real = stream.stream_genotype_chunks
    opened: list[int] = []

    def counting(*args, **kwargs):
        opened.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(stream, "stream_genotype_chunks", counting)
    return opened


@pytest.mark.parametrize("mode", ["centered", "standardized"])
def test_filtered_kinship_reads_the_bed_once_and_matches_the_reference(
    mode, count_genotype_reads
):
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)
    data = load_plink_binary(SYNTHETIC.bfile)
    filter_rows = np.arange(0, data.n_samples, 2)

    K = compute_kinship_streaming(
        SYNTHETIC.bfile,
        maf_threshold=0.05,
        miss_threshold=0.05,
        check_memory=False,
        show_progress=False,
        mode=mode,
        filter_sample_indices=filter_rows,
        chunk_size=7,
    )

    assert count_genotype_reads == [1]
    reference = (
        compute_centered_kinship if mode == "centered" else compute_standardized_kinship
    )
    expected = reference(data.genotypes.copy(), maf_threshold=0.05, check_memory=False)
    np.testing.assert_allclose(K, expected, rtol=1e-12, atol=1e-14)


def test_ksnps_restriction_is_applied_inside_the_single_read(count_genotype_reads):
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)
    data = load_plink_binary(SYNTHETIC.bfile)
    ksnps = np.arange(1, data.n_snps, 3)

    K = compute_kinship_streaming(
        SYNTHETIC.bfile,
        maf_threshold=0.05,
        check_memory=False,
        show_progress=False,
        ksnps_indices=ksnps,
        chunk_size=7,
    )

    assert count_genotype_reads == [1]
    expected = compute_centered_kinship(
        data.genotypes[:, ksnps].copy(), maf_threshold=0.05, check_memory=False
    )
    np.testing.assert_allclose(K, expected, rtol=1e-12, atol=1e-14)


def test_no_surviving_snp_raises_with_the_filter_in_the_message():
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)
    with pytest.raises(ValueError, match=r"No SNPs passed filtering \(maf>=0.6"):
        compute_kinship_streaming(
            SYNTHETIC.bfile, maf_threshold=0.6, check_memory=False, show_progress=False
        )
