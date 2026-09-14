"""Association imputation preserves values with one boolean scratch matrix."""

import tracemalloc

import numpy as np
import pytest

from jamma.lmm.impute import impute_missing_inplace

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("missing_rate", [0.0, 0.01, 1.0])
def test_imputation_changes_only_missing_entries(order, dtype, missing_rate):
    rng = np.random.default_rng(204)
    values = np.array(rng.integers(0, 3, (40, 30)), dtype=dtype, order=order)
    missing = rng.random(values.shape) < missing_rate
    values[missing] = np.nan
    means = rng.uniform(0, 2, values.shape[1])
    expected = np.where(missing, means, values).astype(dtype)
    result = impute_missing_inplace(values, means)
    assert result is None
    np.testing.assert_array_equal(values, expected)


def test_imputation_does_not_allocate_missing_coordinates():
    values = np.ones((512, 1024))
    values[:, ::5] = np.nan
    means = np.ones(values.shape[1])
    tracemalloc.start()
    try:
        impute_missing_inplace(values, means)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(values, np.ones_like(values))
    assert peak < values.size * 2, f"imputation allocated {peak:,} bytes"
