"""Regression cases from the genotype input review, using real data paths."""

import shutil
import tracemalloc

import numpy as np
import pytest

from jamma.genotype.dataset import GenotypeDataset
from jamma.genotype.snp_stats import SnpFilterSpec
from jamma.genotype.variants import SnpMeta
from jamma.gwas import gwas
from jamma.io.bgen import BgenFormatError
from jamma.lmm.genotype_source import SampleBasis
from jamma.lmm.runner_numpy import prepare_genotypes
from tests.bgen_files import write_bgen
from tests.support import requires_c

pytestmark = pytest.mark.tier0


@requires_c
@pytest.mark.parametrize("metadata", ["chromosome", "alleles", "rsid"])
def test_rejects_index_from_another_bgen(tmp_path, metadata):
    probabilities = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    source = write_bgen(tmp_path / "source.bgen", probabilities)
    other = write_bgen(
        tmp_path / "other.bgen",
        probabilities,
        chromosomes=["2"] if metadata == "chromosome" else None,
        alleles=[("G", "A")] if metadata == "alleles" else None,
        rsids=["rs1"] if metadata == "rsid" else None,
    )
    shutil.copyfile(other.bgi, source.bgi)
    dataset = GenotypeDataset.open_bgen(source.bgen, source.sample, source.bgi)
    with pytest.raises(BgenFormatError, match="index is stale"):
        next(dataset.blocks(1))


@pytest.mark.parametrize("subset", [False, True])
def test_batch_chunk_retains_only_its_owned_output(subset):
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
        chunks = prepared.chunks(m)
        chunk = next(chunks)
        retained = tracemalloc.get_traced_memory()[0]
        del chunks
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(chunk.genotypes, matrix[positions])
    assert chunk.genotypes.flags.c_contiguous
    assert retained < chunk.genotypes.nbytes + 100_000


@requires_c
def test_bgen_statistics_are_gated_by_the_real_buffer_cost(tmp_path):
    n, m = 1000, 12000
    one = np.zeros((n, 3))
    one[np.arange(n), np.arange(n) % 3] = 1
    probabilities = np.broadcast_to(one, (m, n, 3))
    probabilities.flags.writeable = True  # BgenWriter needs a writable buffer.
    files = write_bgen(tmp_path / "source.bgen", probabilities)
    np.save(tmp_path / "d.npy", np.ones(n))
    np.save(tmp_path / "u.npy", np.eye(n))
    np.savetxt(tmp_path / "pheno.txt", np.random.default_rng(3).normal(size=n))
    (tmp_path / "snps.txt").write_text("rs0\n")
    with pytest.raises(MemoryError, match="budget"):
        gwas(
            bgen=files.bgen,
            phenotype_file=tmp_path / "pheno.txt",
            eigenvalue_file=tmp_path / "d.npy",
            eigenvector_file=tmp_path / "u.npy",
            snps_file=tmp_path / "snps.txt",
            mem_budget=0.1,
            show_progress=False,
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out" / "result.assoc.txt").exists()

    # A sufficient budget must still run the same data through both passes.
    # Trace Python/NumPy allocations to catch retained previous decode blocks.
    tracemalloc.start()
    try:
        result = gwas(
            bgen=files.bgen,
            phenotype_file=tmp_path / "pheno.txt",
            eigenvalue_file=tmp_path / "d.npy",
            eigenvector_file=tmp_path / "u.npy",
            snps_file=tmp_path / "snps.txt",
            mem_budget=0.3,
            show_progress=True,
            output_dir=tmp_path / "out",
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result.n_snps_tested == 1
    # Decoding 10,000 columns needs 130 MB for dosages and probabilities;
    # 15 MB covers the bounded decode batch, statistics and Python metadata.
    assert peak < 145_000_000


@pytest.mark.parametrize("progress", [None, "Reading genotypes"])
def test_progress_does_not_retain_consumed_matrix(progress):
    n, m = 1000, 500
    matrix = np.ones((n, m), dtype=np.float32)
    variants = SnpMeta(
        np.full(m, "1"),
        np.arange(m).astype(str),
        np.arange(m),
        np.full(m, "A"),
        np.full(m, "G"),
    )
    dataset = GenotypeDataset.from_matrix(matrix, variants)
    tracemalloc.start()
    try:
        blocks = dataset.blocks(m, progress=progress)
        values = next(blocks).dosages(np.arange(0, n, 2))
        retained = tracemalloc.get_traced_memory()[0]
        del blocks
    finally:
        tracemalloc.stop()
    assert retained < values.nbytes + 100_000
