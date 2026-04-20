"""Regression tests verifying memory estimation uses computed chunk size."""

import contextlib
from unittest.mock import patch

import pytest

from jamma.core.memory import check_memory_before_run, estimate_streaming_memory
from jamma.lmm.chunk import _compute_chunk_size


@pytest.mark.tier0
def test_chunk_size_varies_with_scale():
    """_compute_chunk_size returns different values for different scales."""
    small = _compute_chunk_size(12_000)
    large = _compute_chunk_size(500_000)
    assert small != large, "Chunk size should vary with scale"


@pytest.mark.tier0
def test_memory_estimate_uses_computed_chunk():
    """Memory estimates differ when using computed chunk sizes at different scales."""
    small_chunk = _compute_chunk_size(12_000)
    large_chunk = _compute_chunk_size(500_000)

    est_small = estimate_streaming_memory(1410, chunk_size=small_chunk)
    est_large = estimate_streaming_memory(100_000, chunk_size=large_chunk)

    assert est_small.total_peak_gb != est_large.total_peak_gb


@pytest.mark.tier0
def test_check_memory_before_run_uses_computed_chunk():
    """check_memory_before_run calls _compute_chunk_size internally."""
    with patch(
        "jamma.core.chunk._compute_chunk_size", wraps=_compute_chunk_size
    ) as mock:
        # OK if memory insufficient on this machine — we only care that
        # _compute_chunk_size was called with the expected arguments.
        with contextlib.suppress(MemoryError):
            check_memory_before_run(1410, 12_000)
        mock.assert_called_once_with(12_000, n_samples=1410, pipeline_buffers=2)


@pytest.mark.tier0
def test_check_memory_before_run_threads_n_cvt():
    """Regression for jamma-ca6p: check_memory_before_run must thread n_cvt
    through both estimate_streaming_memory and the direct _uab_iab_gb call.

    Both call sites previously defaulted to n_cvt=1, so multi-covariate
    runs underestimated Uab_batch/Iab_batch and could OOM after passing
    preflight. Verifies the public helper accepts n_cvt and propagates it.
    """
    from jamma.core import memory as memory_mod

    real_estimate = memory_mod.estimate_streaming_memory
    real_uab = memory_mod._uab_iab_gb

    with (
        patch.object(
            memory_mod, "estimate_streaming_memory", wraps=real_estimate
        ) as mock_est,
        patch.object(memory_mod, "_uab_iab_gb", wraps=real_uab) as mock_uab,
    ):
        with contextlib.suppress(MemoryError):
            check_memory_before_run(1410, 12_000, n_cvt=5)

        est_kwargs = mock_est.call_args.kwargs
        assert est_kwargs.get("n_cvt") == 5, (
            f"estimate_streaming_memory must receive n_cvt=5, got {est_kwargs!r}"
        )

        # _uab_iab_gb is called multiple times (estimator internals + the
        # direct call at the peak_lmm site). Every call must use n_cvt=5.
        assert mock_uab.call_count >= 1
        for call in mock_uab.call_args_list:
            n_cvt_arg = call.kwargs.get("n_cvt")
            if n_cvt_arg is None and len(call.args) >= 3:
                n_cvt_arg = call.args[2]
            assert n_cvt_arg == 5, f"_uab_iab_gb call {call!r} did not receive n_cvt=5"
