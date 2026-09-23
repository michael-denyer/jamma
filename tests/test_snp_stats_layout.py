"""SNP statistics preserve input layout without changing accumulation order."""

import tracemalloc

import numpy as np
import pytest

from jamma import jlinalg
from jamma.genotype.snp_filter import compute_snp_stats

pytestmark = [
    pytest.mark.tier0,
    pytest.mark.skipif(
        not jlinalg.HAS_C_EXTENSION, reason="jlinalg C extension required"
    ),
]


def _stats(values, hwe):
    outputs = [
        np.empty(values.shape[1], dtype=dtype) for dtype in (float, np.intp, float)
    ]
    if hwe:
        outputs.extend(np.empty(values.shape[1], dtype=np.int64) for _ in range(3))
    jlinalg.compute_snp_stats_chunk(values, *outputs)
    return outputs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("hwe", [False, True])
def test_layouts_produce_identical_statistics(dtype, threads, hwe):
    rng = np.random.default_rng(194)
    values = rng.integers(0, 3, (33, 301)).astype(dtype)
    values[::7, ::5] = np.nan
    values[:, 0] = np.nan
    values[:, 1] = 2
    values[0, 2:5] = [0.125, 1.875, -0.25]
    old_threads = jlinalg.set_n_threads(threads)
    try:
        expected = _stats(values, hwe)
        unaligned = np.ndarray(
            values.shape, dtype=dtype, buffer=bytearray(values.nbytes + 1), offset=1
        )
        unaligned[:] = values
        padded = np.empty((66, 602), dtype=dtype)
        padded[::2, ::2] = values
        readonly = np.asfortranarray(values)
        readonly.flags.writeable = False
        swapped = values.astype(values.dtype.newbyteorder("S"))
        for candidate in (
            np.asfortranarray(values),
            readonly,
            padded[::2, ::2],
            unaligned,
            swapped,
        ):
            before = candidate.copy()
            for actual, reference in zip(_stats(candidate, hwe), expected, strict=True):
                np.testing.assert_array_equal(actual, reference)
            np.testing.assert_array_equal(candidate, before)
        for actual, reference in zip(
            _stats(values[::-1], hwe), _stats(values[::-1].copy(), hwe), strict=True
        ):
            np.testing.assert_array_equal(actual, reference)
    finally:
        jlinalg.set_n_threads(old_threads)


@pytest.mark.parametrize("through_filter", [False, True])
def test_fortran_statistics_do_not_copy_the_genotype_matrix(through_filter):
    values = np.ones((512, 1024), dtype=np.float64, order="F")
    tracemalloc.start()
    try:
        outputs = compute_snp_stats(values) if through_filter else _stats(values, True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(outputs[0], np.ones(values.shape[1]))
    assert peak < values.nbytes / 8, f"SNP stats allocated {peak:,} bytes"
