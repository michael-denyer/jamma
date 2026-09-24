"""Memory quotes bound the real allocations of genotype reads, for every encoding.

Each test traces one phase's Python and NumPy allocations and checks the peak
against the term that phase's quote reserves for it.
"""

import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import pytest
from bed_reader import to_bed

import jamma.genotype.snp_stats as snp_stats
from jamma.core.memory import array_gb
from jamma.genotype.dataset import GenotypeDataset
from jamma.genotype.snp_stats import SnpFilterSpec
from jamma.gwas import gwas
from jamma.jlinalg._snp_stats import compute_snp_stats_chunk as numpy_stats_kernel
from jamma.kinship.loco import compute_loco_kinship_streaming, loco_retained_set
from jamma.lmm.association_plan import plan_association
from jamma.lmm.genotype_source import SampleBasis
from jamma.lmm.runner_numpy import prepare_genotypes
from tests.bgen_files import BgenFiles, one_hot_bgen_from_plink, write_bgen
from tests.support import requires_c

pytestmark = pytest.mark.tier0

N_INPUT = 1000
N_VARIANTS = 1000
HALF = np.arange(0, N_INPUT, 2)
# Few analysed rows: the read, not the statistics kernel, sets the peak.
TENTH = np.arange(0, N_INPUT, 10)
# Interpreter and metadata allocations, far below one 8 MB block.
SLACK = 1_000_000

ENCODINGS = pytest.mark.parametrize(
    "kind", ["plink", pytest.param("bgen", marks=requires_c)]
)
ROWS = pytest.mark.parametrize(
    "rows", [None, HALF, TENTH], ids=["all_rows", "half_rows", "tenth_rows"]
)


class _Files(NamedTuple):
    bfile: Path
    bgen: BgenFiles


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> _Files:
    root = tmp_path_factory.mktemp("reads")
    rng = np.random.default_rng(470)
    values = rng.integers(0, 3, (N_INPUT, N_VARIANTS)).astype(np.float64)
    values[::11, ::5] = np.nan
    half = N_VARIANTS // 2
    to_bed(
        root / "g.bed", values, properties={"chromosome": ["1"] * half + ["2"] * half}
    )
    return _Files(root / "g", one_hot_bgen_from_plink(root / "g", root / "g.bgen"))


def _open(files: _Files, kind: str) -> GenotypeDataset:
    if kind == "plink":
        return GenotypeDataset.open_plink(files.bfile)
    b = files.bgen
    return GenotypeDataset.open_bgen(b.bgen, b.sample, b.bgi)


def _peak(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@ENCODINGS
@ROWS
def test_chunk_read_stays_within_the_block_and_its_overhead(files, kind, rows):
    dataset = _open(files, kind)
    positions = np.arange(N_INPUT) if rows is None else rows
    prepared = prepare_genotypes(
        dataset, SampleBasis(positions, N_INPUT), SnpFilterSpec(0.0, 1.0)
    )
    quote_gb = array_gb(N_INPUT, N_VARIANTS) + dataset.encoding.block_overhead_gb(
        N_INPUT, len(positions), N_VARIANTS
    )

    peak = _peak(lambda: next(prepared.chunks(N_VARIANTS)))

    assert peak <= quote_gb * 1e9 + SLACK


@ENCODINGS
@ROWS
@pytest.mark.parametrize("kernel", ["native", "numpy"])
def test_statistics_quote_covers_the_read_and_the_kernel(files, kind, rows, kernel):
    dataset = _open(files, kind)
    n = N_INPUT if rows is None else len(rows)
    plan = plan_association(
        n,
        N_VARIANTS,
        backend="numpy-streaming",
        n_input_samples=N_INPUT,
        genotype_encoding=dataset.encoding,
        stats_block_size=N_VARIANTS,
    )
    # U is priced in the same phase but not allocated by the read.
    quote_gb = plan.price(eigen=None).statistics_gb - array_gb(n, n)

    def stats():
        return dataset.stats(rows, block_size=N_VARIANTS)

    if kernel == "numpy":
        # allow-patch: the NumPy kernel runs when jlinalg is unavailable, and
        # forcing that fallback also removes the C extension BGEN needs.
        with patch.object(snp_stats, "compute_snp_stats_chunk", numpy_stats_kernel):
            peak = _peak(stats)
    else:
        peak = _peak(stats)

    assert peak <= quote_gb * 1e9 + SLACK


@ENCODINGS
def test_loco_quote_covers_the_kinship_pass(files, kind):
    dataset = _open(files, kind)
    retained = loco_retained_set(N_INPUT, N_INPUT, N_VARIANTS)

    def run():
        stream = compute_loco_kinship_streaming(
            dataset,
            chunk_size=N_VARIANTS,
            check_memory=False,
            show_progress=False,
            consumer_gb=0.0,
            _max_batch_chrs=1,
        )
        for _chr, _kinship in stream:
            pass

    # One chromosome per pass: S_full, K_loco_buf and one S_chr, plus the chunk.
    assert _peak(run) <= retained.while_consuming_gb * 1e9 + SLACK


@requires_c
def test_bgen_statistics_are_gated_before_the_read(tmp_path):
    n, m = 1000, 12000
    one = np.zeros((n, 3))
    one[np.arange(n), np.arange(n) % 3] = 1
    # Every variant shares one row of memory; write_bgen only reads it.
    probabilities = np.broadcast_to(one, (m, n, 3))
    probabilities.flags.writeable = True
    bgen = write_bgen(tmp_path / "source.bgen", probabilities)
    np.save(tmp_path / "d.npy", np.ones(n))
    np.save(tmp_path / "u.npy", np.eye(n))
    np.savetxt(tmp_path / "pheno.txt", np.random.default_rng(3).normal(size=n))
    (tmp_path / "snps.txt").write_text("rs0\n")

    def run(mem_budget: float):
        return gwas(
            bgen=bgen.bgen,
            phenotype_file=tmp_path / "pheno.txt",
            eigenvalue_file=tmp_path / "d.npy",
            eigenvector_file=tmp_path / "u.npy",
            snps_file=tmp_path / "snps.txt",
            mem_budget=mem_budget,
            show_progress=True,
            output_dir=tmp_path / "out",
        )

    # The 10,000-variant statistics read needs 80 MB of dosages and 50 MB of
    # probability buffers beside the 8 MB U.
    with pytest.raises(MemoryError, match="budget"):
        run(0.1)
    assert not (tmp_path / "out" / "result.assoc.txt").exists()
    assert run(0.3).n_snps_tested == 1
