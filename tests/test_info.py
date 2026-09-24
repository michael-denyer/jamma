"""``info_from_quantised`` against the GCTA ``--info`` oracle.

The oracle, ``tests/reference/info.py``, transliterates GCTA's Geno.cpp INFO
loop in Python ints, so equality here is exact, not approximate.
"""

from __future__ import annotations

import numpy as np
import pytest
from loguru import logger

from jamma.genotype.info import info_from_quantised
from jamma.genotype.snp_stats import SnpFilterSpec, SnpStats, filter_snp_stats
from tests.reference.info import gcta_info

pytestmark = pytest.mark.tier0

N_SAMPLES = 301
N_VARIANTS = 40


def _quantised(
    rng: np.random.Generator, bits: np.ndarray, n: int, missing_rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random decoder-shaped ``(q11, q12, missing)``; missing samples store 0."""
    mask = (1 << bits.astype(np.int64)) - 1
    q11 = rng.integers(0, mask + 1, (n, len(bits)))
    q12 = (rng.random((n, len(bits))) * (mask - q11 + 1)).astype(np.int64)
    missing = rng.random((n, len(bits))) < missing_rate
    q11[missing] = 0
    q12[missing] = 0
    return (
        np.asfortranarray(q11.astype(np.uint16)),
        np.asfortranarray(q12.astype(np.uint16)),
        np.asfortranarray(missing),
    )


def _oracle(q11, q12, missing, bits, rows) -> np.ndarray:
    keep = range(q11.shape[0]) if rows is None else rows
    return np.array(
        [
            gcta_info(q11[:, j], q12[:, j], missing[:, j], int(bits[j]), keep)
            for j in range(q11.shape[1])
        ]
    )


def _row_cases(rng: np.random.Generator) -> list[np.ndarray | None]:
    return [
        None,
        np.sort(rng.choice(N_SAMPLES, N_SAMPLES // 2, replace=False)),
        np.arange(7, N_SAMPLES - 11),
    ]


@pytest.mark.parametrize("bits", [1, 8, 16])
def test_equals_gcta_oracle_exactly(bits: int):
    rng = np.random.default_rng(bits)
    bit_depth = np.full(N_VARIANTS, bits, dtype=np.uint8)
    q11, q12, missing = _quantised(rng, bit_depth, N_SAMPLES, missing_rate=0.1)
    for rows in _row_cases(rng):
        got = info_from_quantised(q11, q12, missing, bit_depth, rows)
        np.testing.assert_array_equal(got, _oracle(q11, q12, missing, bit_depth, rows))


def test_equals_gcta_oracle_with_mixed_bit_depths():
    """Each variant divides by its own 2**B - 1."""
    rng = np.random.default_rng(11)
    bit_depth = rng.choice(np.array([1, 3, 8, 10, 16], dtype=np.uint8), N_VARIANTS)
    q11, q12, missing = _quantised(rng, bit_depth, N_SAMPLES, missing_rate=0.05)
    got = info_from_quantised(q11, q12, missing, bit_depth)
    np.testing.assert_array_equal(got, _oracle(q11, q12, missing, bit_depth, None))


def test_random_probabilities_give_negative_info_unclamped():
    rng = np.random.default_rng(5)
    bit_depth = np.full(N_VARIANTS, 16, dtype=np.uint8)
    q11, q12, missing = _quantised(rng, bit_depth, N_SAMPLES, missing_rate=0.0)
    assert np.min(info_from_quantised(q11, q12, missing, bit_depth)) < 0


def test_uniform_uncertainty_is_minus_one_third():
    """P = (1/3, 1/3, 1/3) everywhere: 1 - (2/3) / (2 * 0.5 * 0.5) = -1/3."""
    q = np.full((50, 1), 85, dtype=np.uint16)  # 85 / 255 = 1/3
    info = info_from_quantised(
        q, q, np.zeros((50, 1), dtype=bool), np.array([8], dtype=np.uint8)
    )
    assert info[0] == pytest.approx(-1 / 3, rel=1e-15)


@pytest.mark.parametrize("q11_value", [0, 255])
def test_monomorphic_is_one(q11_value: int):
    """Every sample certain of the same genotype: theta is 0 or 1, INFO 1."""
    q11 = np.full((20, 1), q11_value, dtype=np.uint16)
    q12 = np.zeros((20, 1), dtype=np.uint16)
    info = info_from_quantised(
        q11, q12, np.zeros((20, 1), dtype=bool), np.array([8], dtype=np.uint8)
    )
    assert info[0] == 1.0


def test_all_missing_column_is_one():
    """GCTA divides 0 by 0 here (NaN); JAMMA reports 1, as for monomorphic."""
    q = np.zeros((20, 2), dtype=np.uint16)
    missing = np.zeros((20, 2), dtype=bool)
    missing[:, 0] = True
    missing[3:, 1] = True  # present only outside the requested rows
    info = info_from_quantised(
        q, q, missing, np.array([8, 8], dtype=np.uint8), np.arange(3, 20)
    )
    np.testing.assert_array_equal(info, [1.0, 1.0])
    assert np.isnan(gcta_info(q[:, 0], q[:, 0], missing[:, 0], 8, range(20)))


def test_missing_samples_are_excluded_from_n():
    """Missing rows, whatever they store, change nothing: N counts present rows."""
    rng = np.random.default_rng(8)
    bit_depth = np.full(N_VARIANTS, 8, dtype=np.uint8)
    q11, q12, _ = _quantised(rng, bit_depth, N_SAMPLES, missing_rate=0.0)
    extra = 40
    q11_pad = np.vstack([q11, rng.integers(0, 128, (extra, N_VARIANTS))])
    q12_pad = np.vstack([q12, rng.integers(0, 127, (extra, N_VARIANTS))])
    missing_pad = np.zeros(q11_pad.shape, dtype=bool)
    missing_pad[N_SAMPLES:] = True
    padded = info_from_quantised(
        q11_pad.astype(np.uint16), q12_pad.astype(np.uint16), missing_pad, bit_depth
    )
    no_missing = np.zeros((N_SAMPLES, N_VARIANTS), dtype=bool)
    plain = info_from_quantised(q11, q12, no_missing, bit_depth)
    np.testing.assert_array_equal(padded, plain)


def test_allele_flip_leaves_info_unchanged():
    """Counting the other allele swaps P11 and P22; INFO is symmetric.

    Not bit-identical: theta and 1 - theta round differently, so the two
    orientations agree to a few ULPs, in GCTA as here.
    """
    rng = np.random.default_rng(9)
    bit_depth = np.full(N_VARIANTS, 16, dtype=np.uint8)
    q11, q12, missing = _quantised(rng, bit_depth, N_SAMPLES, missing_rate=0.1)
    q22 = (65535 - q11.astype(np.int64) - q12).astype(np.uint16)
    q22[missing] = 0
    np.testing.assert_allclose(
        info_from_quantised(q22, q12, missing, bit_depth),
        info_from_quantised(q11, q12, missing, bit_depth),
        rtol=0,
        atol=1e-14,
    )


def _stats_with_info(info: list[float]) -> SnpStats:
    k = len(info)
    return SnpStats(
        col_means=np.ones(k),
        miss_counts=np.zeros(k, dtype=np.intp),
        col_vars=np.full(k, 0.5),
        n_samples=10,
        info=np.array(info),
    )


def test_filter_keeps_info_equal_to_threshold():
    stats = _stats_with_info([0.3, 0.5, 0.7, -0.1])
    messages: list[str] = []
    sink = logger.add(messages.append, level="INFO", format="{message}")
    try:
        selection = filter_snp_stats(stats, SnpFilterSpec(0.0, 1.0, info_threshold=0.5))
    finally:
        logger.remove(sink)
    np.testing.assert_array_equal(selection.indices, [1, 2])
    assert any("INFO filter: 2 SNPs removed (INFO < 0.5)" in m for m in messages)


def test_filter_off_keeps_negative_info():
    stats = _stats_with_info([0.3, 0.5, 0.7, -0.1])
    selection = filter_snp_stats(stats, SnpFilterSpec(0.0, 1.0))
    np.testing.assert_array_equal(selection.indices, [0, 1, 2, 3])


def test_filter_spec_rejects_negative_threshold():
    with pytest.raises(ValueError, match="info_threshold"):
        SnpFilterSpec(0.0, 1.0, info_threshold=-0.1)


def test_snp_stats_info_defaults_to_one_and_is_sliced_by_take():
    stats = SnpStats(
        col_means=np.ones(3),
        miss_counts=np.zeros(3, dtype=np.intp),
        col_vars=np.ones(3),
        n_samples=5,
    )
    np.testing.assert_array_equal(stats.info, [1.0, 1.0, 1.0])
    sliced = _stats_with_info([0.1, 0.2, 0.3]).take(np.array([2, 0]))
    np.testing.assert_array_equal(sliced.info, [0.3, 0.1])


def test_snp_stats_rejects_info_of_wrong_length():
    with pytest.raises(ValueError, match="info shape mismatch"):
        SnpStats(
            col_means=np.ones(3),
            miss_counts=np.zeros(3, dtype=np.intp),
            col_vars=np.ones(3),
            n_samples=5,
            info=np.ones(2),
        )
