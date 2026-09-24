"""INFO through ``GenotypeDataset``, the kinship pass and association selection.

Expected INFO comes from the GCTA oracle (``tests/reference/info.py``) run on
the reference decoder's quantised values (``tests/reference/bgen.py``), so no
production code sits on the expected side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from jamma.genotype.dataset import GenotypeDataset
from jamma.genotype.snp_stats import SnpFilterSpec
from jamma.kinship.loco import compute_loco_kinship_streaming
from jamma.kinship.stream import compute_kinship_streaming
from jamma.lmm.association_plan import plan_association
from jamma.lmm.genotype_source import SampleBasis
from jamma.lmm.prepare_common import (
    AnalysedPhenotype,
    parse_eigen_input,
    restrict_eigen_input,
)
from jamma.lmm.runner_numpy import LmmRunSpec, prepare_genotypes, run_single
from jamma.lmm.schema import LmmConfig
from tests.bgen_files import BgenFiles, random_probabilities, write_bgen
from tests.reference.bgen import decode_file
from tests.reference.info import gcta_info
from tests.support import requires_c

pytestmark = [pytest.mark.tier0, requires_c]

N_SAMPLES = 203
N_VARIANTS = 37
CHROMOSOMES = ["1"] * 12 + ["2"] * 13 + ["3"] * 12


def _write(tmp_path: Path, bit_depth: int = 8) -> BgenFiles:
    """Probabilities blending hard calls with noise, so INFO spans (<0, 1]."""
    rng = np.random.default_rng(bit_depth)
    noise = random_probabilities(rng, N_VARIANTS, N_SAMPLES, missing_rate=0.05)
    calls = rng.binomial(2, rng.uniform(0.1, 0.5, (N_VARIANTS, 1)), noise.shape[:2])
    certainty = np.linspace(0.0, 1.0, N_VARIANTS)[:, None, None]
    probs = certainty * np.eye(3)[calls] + (1 - certainty) * noise
    return write_bgen(
        tmp_path / f"info_b{bit_depth}.bgen",
        probs,
        bit_depth=bit_depth,
        chromosomes=CHROMOSOMES,
    )


def _open(files: BgenFiles) -> GenotypeDataset:
    return GenotypeDataset.open_bgen(files.bgen, files.sample, files.bgi)


def _oracle_info(files: BgenFiles, rows: np.ndarray | None) -> np.ndarray:
    decoded = decode_file(files.bgen)
    keep = range(N_SAMPLES) if rows is None else rows
    return np.array(
        [
            gcta_info(
                decoded.q11[:, j],
                decoded.q12[:, j],
                decoded.missing[:, j],
                int(decoded.bit_depth[j]),
                keep,
            )
            for j in range(N_VARIANTS)
        ]
    )


def _rows() -> np.ndarray:
    rng = np.random.default_rng(21)
    return np.sort(rng.choice(N_SAMPLES, 150, replace=False)).astype(np.intp)


def _threshold(expected: np.ndarray) -> float:
    """An INFO value some SNP has exactly, so the >= boundary is exercised."""
    t = float(np.sort(expected)[N_VARIANTS // 2])
    assert t > 0
    return t


def _kinship(dataset: GenotypeDataset, rows: np.ndarray, **kwargs) -> np.ndarray:
    return compute_kinship_streaming(
        dataset,
        chunk_size=10,
        check_memory=False,
        show_progress=False,
        filter_sample_indices=rows,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Dataset statistics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bit_depth", [1, 8, 16])
def test_dataset_stats_info_equals_oracle(tmp_path: Path, bit_depth: int):
    files = _write(tmp_path, bit_depth)
    dataset = _open(files)
    for rows in (None, _rows()):
        expected = _oracle_info(files, rows)
        np.testing.assert_array_equal(dataset.stats(rows, block_size=10).info, expected)
        columns = np.arange(3, N_VARIANTS, 4, dtype=np.intp)
        np.testing.assert_array_equal(
            dataset.stats(rows, columns=columns, block_size=3).info,
            expected[columns],
        )


def test_block_info_and_stats_agree_with_oracle(tmp_path: Path):
    files = _write(tmp_path)
    rows = _rows()
    expected = _oracle_info(files, rows)
    for block in _open(files).blocks(10):
        np.testing.assert_array_equal(block.info(rows), expected[block.columns])
        np.testing.assert_array_equal(block.stats(rows).info, expected[block.columns])
        block.dosages()
        with pytest.raises(RuntimeError, match="spent"):
            block.info(rows)


def test_plink_info_is_one(asymmetric_plink: Path):
    dataset = GenotypeDataset.open_plink(asymmetric_plink)
    np.testing.assert_array_equal(dataset.stats(None).info, np.ones(dataset.n_variants))
    for block in dataset.blocks(7):
        np.testing.assert_array_equal(block.info(), np.ones(len(block.columns)))
        np.testing.assert_array_equal(block.stats().info, np.ones(len(block.columns)))
        block.dosages()


# --------------------------------------------------------------------------
# Kinship pass
# --------------------------------------------------------------------------


def test_kinship_info_filter_keeps_exactly_oracle_survivors(tmp_path: Path):
    files = _write(tmp_path)
    dataset = _open(files)
    rows = _rows()
    expected = _oracle_info(files, rows)
    t = _threshold(expected)
    survivors = np.flatnonzero(expected >= t)
    assert 0 < len(survivors) < N_VARIANTS

    messages: list[str] = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        filtered = _kinship(dataset, rows, info_threshold=t)
    finally:
        logger.remove(sink)
    n_removed = N_VARIANTS - len(survivors)
    assert any(
        f"Kinship INFO filter: {n_removed} SNPs removed (INFO < {t})" in m
        for m in messages
    )
    restricted = _kinship(dataset, rows, ksnps_indices=survivors)
    np.testing.assert_array_equal(filtered, restricted)
    assert not np.array_equal(filtered, _kinship(dataset, rows))


def test_loco_first_pass_stats_carry_info(tmp_path: Path):
    files = _write(tmp_path)
    dataset = _open(files)
    rows = _rows()
    expected = _oracle_info(files, rows)
    t = _threshold(expected)
    survivors = np.flatnonzero(expected >= t)

    def loco(**kwargs):
        return compute_loco_kinship_streaming(
            dataset,
            chunk_size=10,
            check_memory=False,
            show_progress=False,
            filter_sample_indices=rows,
            consumer_gb=0.0,
            **kwargs,
        )

    stream = loco(info_threshold=t)
    matrices = stream.materialize()
    sink_stats = stream.snp_stats
    measured = dataset.stats(rows)
    np.testing.assert_array_equal(sink_stats.info, expected)
    np.testing.assert_array_equal(sink_stats.col_means, measured.col_means)
    np.testing.assert_array_equal(sink_stats.col_vars, measured.col_vars)
    assert sink_stats.n_unexpected == 0

    restricted = loco(ksnps_indices=survivors).materialize()
    assert matrices.keys() == restricted.keys()
    for chr_name, matrix in matrices.items():
        np.testing.assert_array_equal(matrix, restricted[chr_name])


def test_plink_kinship_rejects_info_threshold(asymmetric_plink: Path):
    dataset = GenotypeDataset.open_plink(asymmetric_plink)
    with pytest.raises(ValueError, match="INFO threshold"):
        compute_kinship_streaming(
            dataset, check_memory=False, show_progress=False, info_threshold=0.3
        )
    stream = compute_loco_kinship_streaming(
        dataset,
        check_memory=False,
        show_progress=False,
        consumer_gb=0.0,
        info_threshold=0.3,
    )
    with pytest.raises(ValueError, match="INFO threshold"):
        next(iter(stream))


def test_plink_loco_stats_info_is_one(asymmetric_plink: Path):
    dataset = GenotypeDataset.open_plink(asymmetric_plink)
    stream = compute_loco_kinship_streaming(
        dataset, check_memory=False, show_progress=False, consumer_gb=0.0
    )
    next(iter(stream))
    np.testing.assert_array_equal(stream.snp_stats.info, np.ones(dataset.n_variants))


# --------------------------------------------------------------------------
# Association selection
# --------------------------------------------------------------------------


def test_association_selection_keeps_exactly_oracle_survivors(tmp_path: Path):
    files = _write(tmp_path)
    dataset = _open(files)
    rows = _rows()
    mask = np.zeros(N_SAMPLES, dtype=bool)
    mask[rows] = True
    expected = _oracle_info(files, rows)
    t = _threshold(expected)
    prepared = prepare_genotypes(
        dataset, SampleBasis.from_mask(mask), SnpFilterSpec(0.0, 1.0, info_threshold=t)
    )
    np.testing.assert_array_equal(
        prepared.selection.indices, np.flatnonzero(expected >= t)
    )


def test_plink_association_rejects_info_threshold(asymmetric_plink: Path):
    dataset = GenotypeDataset.open_plink(asymmetric_plink)
    with pytest.raises(ValueError, match="INFO threshold"):
        prepare_genotypes(
            dataset,
            SampleBasis.from_mask(np.ones(dataset.n_samples, dtype=bool)),
            SnpFilterSpec(0.0, 1.0, info_threshold=0.3),
        )


@pytest.mark.filterwarnings("ignore:Kinship matrix has .* eigenvalues close to zero")
def test_end_to_end_kinship_and_association_apply_info(tmp_path: Path):
    """One INFO threshold filters both the kinship SNPs and the tested SNPs."""
    files = _write(tmp_path)
    dataset = _open(files)
    rng = np.random.default_rng(4)
    phenotypes = rng.normal(size=N_SAMPLES)
    expected = _oracle_info(files, None)
    t = _threshold(expected)
    survivors = np.flatnonzero(expected >= t)

    kinship = compute_kinship_streaming(
        dataset, check_memory=False, show_progress=False, info_threshold=t
    )
    config = LmmConfig(
        maf_threshold=0.0, miss_threshold=1.0, check_memory=False, show_progress=False
    )
    samples = AnalysedPhenotype.from_inputs(phenotypes, None)
    execution = plan_association(
        samples.n_samples,
        N_VARIANTS,
        config=config,
        backend="numpy",
        n_cvt=samples.n_cvt,
        n_input_samples=N_SAMPLES,
    )
    results: list = []
    run_single(
        dataset,
        LmmRunSpec(config=config, execution=execution, info_threshold=t),
        samples,
        restrict_eigen_input(
            parse_eigen_input(kinship, None, None), samples.valid_mask
        ),
        results,
    )
    assert [r.rs for r in results] == list(dataset.variants.rs[survivors])
