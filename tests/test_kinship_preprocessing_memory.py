"""The kinship memory quote bounds real preprocessing allocations."""

import tracemalloc

import numpy as np
import pytest
from bed_reader import to_bed

from jamma.core.memory import estimate_streaming_memory
from jamma.kinship import (
    compute_kinship_streaming,
    impute_and_center,
    impute_center_and_standardize,
)
from tests.conftest import require_fixture, requires_c
from tests.fixture_paths import MOUSE

pytestmark = pytest.mark.tier0


@requires_c
def test_gwas_budget_accounts_for_kinship_preprocessing(tmp_path):
    from jamma import gwas

    require_fixture(MOUSE.bed, MOUSE.bim, MOUSE.fam)
    with pytest.raises(MemoryError, match="budget"):
        gwas(
            MOUSE.bfile,
            backend="numpy",
            mem_budget=0.25,
            output_dir=tmp_path,
            show_progress=False,
            no_telemetry=True,
        )


@pytest.mark.parametrize("order", ["C", "F"])
def test_inplace_centering_uses_only_boolean_matrix_scratch(order):
    rng = np.random.default_rng(104)
    values = np.array(rng.integers(0, 3, (512, 1024)), dtype=np.float64, order=order)
    values[::13, ::7] = np.nan
    means = np.nanmean(values, axis=0)
    expected = np.where(np.isnan(values), means, values) - means

    tracemalloc.start()
    try:
        actual = impute_and_center(values)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert actual is values
    np.testing.assert_array_equal(actual, expected)
    assert peak < values.nbytes / 2, f"centering scratch was {peak:,} bytes"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fractional_centering_preserves_precision(dtype):
    rng = np.random.default_rng(112)
    values = rng.normal(size=(100, 16)).astype(dtype)
    values[::7, ::3] = np.nan
    means = np.nanmean(values, axis=0)
    expected = np.where(np.isnan(values), means, values) - means
    actual = impute_and_center(values)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=8 * np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
@pytest.mark.parametrize("readonly", [False, True])
def test_standardizing_preserves_input_and_numeric_dtype(dtype, readonly):
    values = np.array([[1, 0, 2], [1, 1, 1], [1, 2, 0]], dtype=dtype)
    before = values.copy()
    values.flags.writeable = not readonly
    actual = impute_center_and_standardize(values)
    expected_dtype = np.float64 if dtype == np.int64 else dtype
    assert actual.dtype == expected_dtype
    np.testing.assert_array_equal(values, before)
    np.testing.assert_allclose(actual[:, 0], 0)
    np.testing.assert_allclose(actual[:, 1:].var(axis=0), 1, rtol=1e-6)


@pytest.mark.parametrize("mode", ["centered", "standardized"])
@pytest.mark.parametrize("subset", [False, True])
def test_kinship_quote_covers_preprocessing(tmp_path, mode, subset):
    rng = np.random.default_rng(108)
    values = rng.integers(0, 3, (800, 1000)).astype(np.float64)
    values[::11, ::5] = np.nan
    bfile = tmp_path / "memory"
    to_bed(bfile.with_suffix(".bed"), values)
    selected = np.arange(0, len(values), 2) if subset else None
    quote = estimate_streaming_memory(len(values), chunk_size=values.shape[1])

    tracemalloc.start()
    try:
        result = compute_kinship_streaming(
            bfile,
            chunk_size=values.shape[1],
            mode=mode,
            check_memory=False,
            show_progress=False,
            valid_indices=selected,
            filter_sample_indices=selected,
            ksnps_indices=np.arange(values.shape[1] - 1),
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.shape == ((400, 400) if subset else (800, 800))
    # Allow metadata and interpreter allocations, not an unpriced matrix.
    assert peak <= quote.kinship_gb * 1e9 + 1_000_000, (
        f"peak {peak:,} exceeds kinship quote {quote.kinship_gb * 1e9:,.0f}"
    )
