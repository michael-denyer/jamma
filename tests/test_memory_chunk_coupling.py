"""Regression tests verifying memory estimation threads chunk size and n_cvt.

These tests assert on observable outputs — estimated GB totals and whether
the live preflight gate (``memory_preflight``) raises MemoryError — rather
than on internal call counts of ``_compute_chunk_size`` / ``_uab_iab_gb``.
This follows CLAUDE.md: assert observable behavior, not delegation plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma.core.memory import _uab_iab_gb, estimate_streaming_memory
from jamma.lmm.chunk_sizing import compute_chunk_size_numpy
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.runner import ExecutionPlan
from jamma.pipeline_config import PipelineConfig
from jamma.pipeline_memory import memory_preflight


def _streaming_preflight(
    n_valid: int, n_snps: int, n_cvt: int, lmm_mode: int = 1
) -> None:
    """Drive the live preflight gate the way the pipeline does."""
    config = PipelineConfig(bfile=Path("unused"), lmm_mode=lmm_mode)
    memory_preflight(config, ExecutionPlan("streaming", "test"), n_valid, n_snps, n_cvt)


@pytest.mark.tier0
def test_chunk_size_varies_with_scale():
    """The sizer returns different values as the sample count scales."""
    small = compute_chunk_size_numpy(1_410, 12_000, dispatch=DispatchPath.FUSED)
    large = compute_chunk_size_numpy(100_000, 500_000, dispatch=DispatchPath.FUSED)
    assert small != large, "Chunk size should vary with scale"


@pytest.mark.tier0
def test_memory_estimate_uses_computed_chunk():
    """Memory estimates differ when using computed chunk sizes at different scales."""
    small_chunk = compute_chunk_size_numpy(1_410, 12_000, dispatch=DispatchPath.FUSED)
    large_chunk = compute_chunk_size_numpy(
        100_000, 500_000, dispatch=DispatchPath.FUSED
    )

    est_small = estimate_streaming_memory(1410, compute_chunk_size=small_chunk)
    est_large = estimate_streaming_memory(100_000, compute_chunk_size=large_chunk)

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
def test_preflight_raises_when_n_cvt_inflates_past_available(monkeypatch):
    """Regression for jamma-ca6p: the preflight gate must thread n_cvt.

    Mode 2 with n_cvt >= 2 dispatches to SOA_SPLIT, whose per-SNP varying
    Uab columns grow with n_cvt (fused paths hold no per-SNP batch arrays,
    so n_cvt=200 there is rejected by the kernel's own n_cvt cap, not by
    memory). n_cvt=1 fits comfortably in the pinned budget; n_cvt=90
    inflates the varying columns past it. If the preflight silently
    defaults n_cvt back to 1, both calls succeed — only a correctly
    threaded n_cvt produces the asymmetric pass/raise this asserts. The
    asymmetry holds under JAMMA_FORCE_NUMPY_FALLBACK too, where both legs
    price the full Uab batch instead.
    """
    from jamma.core import memory

    # Pin available memory to a small fixed value to make the threshold
    # deterministic across machines. Both the chunk sizer and the
    # sufficiency check read the one seam.
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 1.0)

    n_samples = 800
    n_snps = 10_000

    # With n_cvt=1, peak should be well under 1.0GB available.
    _streaming_preflight(n_samples, n_snps, n_cvt=1, lmm_mode=2)

    # With n_cvt=90, the split path's varying columns inflate past it.
    with pytest.raises(MemoryError):
        _streaming_preflight(n_samples, n_snps, n_cvt=90, lmm_mode=2)


@pytest.mark.tier0
def test_preflight_succeeds_at_low_n_cvt():
    """Sanity: the default n_cvt=1 call on a tiny dataset must pass on any
    machine with >1GB free. Exists to catch regressions where the preflight
    spuriously rejects small inputs.
    """
    # 100 samples x 1000 SNPs is negligible — must fit everywhere.
    _streaming_preflight(100, 1000, n_cvt=1)


@pytest.mark.tier0
def test_preflight_accepts_moderate_n_cvt(monkeypatch):
    """Regression (false-OOM): the preflight must size its compute chunk with the
    SAME n_cvt it estimates Uab with.

    The bug sized ``compute_chunk`` via ``_compute_chunk_size`` without n_cvt (so
    it defaulted to n_cvt=1 and capped at MAX_SAFE_CHUNK=50k), then estimated Uab
    at the real n_cvt. For 25 covariates that inflated the peak ~60x (~467GB) and
    raised MemoryError on a run the streaming runtime sizes down (chunk ~1.3k) and
    completes in ~13GB — e.g. a conditional analysis conditioning on a locus.

    The inflate-and-raise test above passes even with the bug, because the bug
    also over-estimates; only threading n_cvt into the chunk fixes this direction.
    """
    from jamma.core import memory

    # Pin available RAM at 100GB for both the chunk sizer and the gate's
    # sufficiency check, so the pass/raise boundary is deterministic. With an
    # n_cvt-aware chunk the peak is ~0.35x available (~35GB); the buggy
    # n_cvt-blind chunk estimates ~467GB and exceeds 100GB.
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 100.0)

    _streaming_preflight(3048, 88_268, n_cvt=25)
