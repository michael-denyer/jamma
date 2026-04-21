"""Regression tests verifying memory estimation threads chunk size and n_cvt.

These tests assert on observable outputs — estimated GB totals and whether
``check_memory_before_run`` raises MemoryError — rather than on internal
call counts of ``_compute_chunk_size`` / ``_uab_iab_gb``. This follows
CLAUDE.md: assert observable behavior, not delegation plumbing.
"""

from __future__ import annotations

import pytest

from jamma.core.memory import (
    _uab_iab_gb,
    check_memory_before_run,
    estimate_streaming_memory,
)
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
def test_uab_iab_gb_scales_with_n_cvt():
    """Observable invariant: Uab/Iab buffers grow with n_cvt.

    The runner's preflight underestimate bug (jamma-ca6p) was possible
    precisely because callers defaulted to n_cvt=1. If a future refactor
    silently re-introduces that default, this test will catch it via the
    observable GB output, not by inspecting call arguments.
    """
    chunk = 10_000
    gb_n1 = _uab_iab_gb(n_samples=1000, chunk_size=chunk, n_cvt=1, use_fused=False)
    gb_n5 = _uab_iab_gb(n_samples=1000, chunk_size=chunk, n_cvt=5, use_fused=False)
    gb_n10 = _uab_iab_gb(n_samples=1000, chunk_size=chunk, n_cvt=10, use_fused=False)

    assert gb_n1 < gb_n5 < gb_n10, (
        f"Uab/Iab must grow with n_cvt, got n_cvt=1:{gb_n1} < 5:{gb_n5} < 10:{gb_n10}"
    )


@pytest.mark.tier0
def test_estimate_streaming_memory_peak_scales_with_n_cvt():
    """estimate_streaming_memory's peak output differs for different n_cvt.

    Higher-level observable check of the same invariant: the preflight
    total reported to users must be larger for multi-covariate runs.
    """
    est_n1 = estimate_streaming_memory(n_samples=2000, n_cvt=1)
    est_n10 = estimate_streaming_memory(n_samples=2000, n_cvt=10)
    assert est_n10.total_peak_gb > est_n1.total_peak_gb, (
        f"Peak estimate must grow with n_cvt, got n_cvt=1:{est_n1.total_peak_gb} "
        f"n_cvt=10:{est_n10.total_peak_gb}"
    )


@pytest.mark.tier0
def test_check_memory_before_run_raises_when_n_cvt_inflates_past_available(
    monkeypatch,
):
    """Regression for jamma-ca6p: check_memory_before_run must thread n_cvt.

    We pick a sample count where n_cvt=1 fits comfortably in available
    memory but n_cvt=200 inflates Uab/Iab past the available budget. If
    the preflight silently defaults n_cvt back to 1, both calls will
    succeed — only a correctly threaded n_cvt produces the asymmetric
    pass/raise behavior this test asserts.
    """
    from jamma.core import memory as memory_mod

    # Pin available memory to a small fixed value to make the threshold
    # deterministic across machines. Snapshot only — no compute mocking.
    class _FakeSnap:
        rss_gb = 0.5
        available_gb = 2.0

    monkeypatch.setattr(memory_mod, "get_memory_snapshot", lambda: _FakeSnap())

    n_samples = 800
    n_snps = 10_000

    # With n_cvt=1, peak should be well under 2.0GB available.
    # With n_cvt=200, Uab/Iab alone inflates past the budget.
    assert check_memory_before_run(n_samples, n_snps, n_cvt=1) is True

    with pytest.raises(MemoryError):
        check_memory_before_run(n_samples, n_snps, n_cvt=200)


@pytest.mark.tier0
def test_check_memory_before_run_succeeds_at_low_n_cvt():
    """Sanity: the default n_cvt=1 call on a tiny dataset must pass on any
    machine with >1GB free. Exists to catch regressions where
    check_memory_before_run spuriously rejects small inputs.
    """
    # 100 samples x 1000 SNPs is negligible — must fit everywhere.
    assert check_memory_before_run(100, 1000, n_cvt=1) is True
