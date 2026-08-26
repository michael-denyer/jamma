"""Regression tests verifying memory estimation threads chunk size and n_cvt.

These tests assert on observable outputs — estimated GB totals and whether
the live preflight gate (``memory_preflight``) raises MemoryError — rather
than on internal call counts of ``plan_lmm_chunks`` / ``_uab_iab_gb``. This
follows CLAUDE.md: assert observable behavior, not delegation plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma.core import memory
from jamma.core.memory import _uab_iab_gb, estimate_streaming_memory
from jamma.lmm.chunk_sizing import compute_chunk_size_numpy, plan_lmm_chunks
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.likelihood import classify_uab_columns
from jamma.lmm.runner import ExecutionPlan
from jamma.pipeline_config import PipelineConfig
from jamma.pipeline_memory import _compute_chunk, memory_preflight


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

    The bug sized ``compute_chunk`` via the chunk sizer without n_cvt (so
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


# n_cvt=1 selects FUSED/FUSED_SCORE_WS/FUSED_LRT_WS depending on lmm_mode;
# n_cvt>=2 selects FUSED_GENERAL or SOA_SPLIT. One representative lmm_mode per
# path, matching select_dispatch_path's own resolution table.
_DISPATCH_CASES = [
    pytest.param(1, 1, DispatchPath.FUSED, id="fused"),
    pytest.param(2, 1, DispatchPath.FUSED_GENERAL, id="fused_general"),
    pytest.param(1, 3, DispatchPath.FUSED_SCORE_WS, id="fused_score_ws"),
    pytest.param(1, 2, DispatchPath.FUSED_LRT_WS, id="fused_lrt_ws"),
    pytest.param(2, 2, DispatchPath.SOA_SPLIT, id="soa_split"),
]


def _engine_allocation_gb(
    n_samples: int, n_cvt: int, plan, dispatch: DispatchPath
) -> float:
    """The exact bytes ``_ChunkEngine.__init__`` allocates for this plan.

    Reproduces its buffer shapes directly (``utg_bufs``, and ``uab_var_bufs``
    for SOA_SPLIT) rather than calling back into the pricing helpers under
    test, so this is an independent check of the allocation, not a tautology.
    """
    utg_bytes = plan.chunk_size * n_samples * 8 * plan.n_buffers
    uab_var_bytes = 0
    if dispatch is DispatchPath.SOA_SPLIT:
        n_var = len(classify_uab_columns(n_cvt)[1])
        uab_var_bytes = plan.chunk_size * n_var * n_samples * 8 * plan.n_buffers
    # The raw genotype block the chunk source hands prepare(): one buffer
    # live at a time regardless of pipelining (chunk_runner_numpy.py
    # _drive_pipeline overlaps a rotated buffer with the next prepare() call,
    # never two raw reads at once).
    raw_block_bytes = plan.chunk_size * n_samples * 8
    return (utg_bytes + uab_var_bytes + raw_block_bytes) / 1e9


@pytest.mark.tier0
class TestChunkPlanMatchesEngine:
    """One LmmChunkPlan, computed once: the engine allocates from it, and the
    preflight prices from it. These pin that the two routes cannot diverge.
    """

    @pytest.mark.parametrize("n_cvt,lmm_mode,dispatch", _DISPATCH_CASES)
    def test_plan_chunk_size_matches_engine(
        self, monkeypatch, n_cvt, lmm_mode, dispatch
    ):
        """plan_lmm_chunks' chunk size is exactly what the engine sizes.

        chunk_runner_numpy.run_lmm_chunk_source_numpy calls plan_lmm_chunks
        with these same arguments to size chunk_size/n_chunks/n_buffers for
        the engine's _ChunkEngine, so calling it directly here with the same
        inputs reproduces the engine's own sizing decision.
        """
        from jamma.lmm.dispatch import select_dispatch_path
        from jamma.lmm.schema import parse_lmm_mode

        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)

        n_samples = 50_000
        n_filtered = 500_000

        # The parametrized dispatch must be what select_dispatch_path
        # actually derives for (n_cvt, lmm_mode) with the C extension
        # active, or this case is testing an unreachable combination.
        assert (
            select_dispatch_path(
                n_cvt, parse_lmm_mode(lmm_mode), accel=True, log_choices=False
            )
            is dispatch
        )

        plan = plan_lmm_chunks(n_samples, n_filtered, n_cvt, dispatch)

        assert plan.chunk_size >= 1
        assert plan.n_chunks == (n_filtered + plan.chunk_size - 1) // plan.chunk_size
        assert plan.n_buffers in (1, 2)
        if not dispatch.use_split:
            # NUMPY_FALLBACK never pipelines.
            assert plan.n_buffers == 1
            assert not plan.use_pipeline

    @pytest.mark.parametrize("n_cvt,lmm_mode,dispatch", _DISPATCH_CASES)
    def test_preflight_priced_bytes_match_engine_allocation(
        self, monkeypatch, n_cvt, lmm_mode, dispatch
    ):
        """The preflight's priced LMM-phase bytes equal what the engine holds.

        Regression for the P6 finding: pipeline_memory.py priced the Uab
        extra at one buffer's worth while chunk_runner_numpy.py allocated
        uab_var_bufs at n_buffers, and priced NUMPY_FALLBACK's chunk at
        pipeline_buffers=2 while the engine (which never pipelines that path)
        sized it at 1. Both are folded into plan_lmm_chunks now, so the
        preflight's rotation-buffer-plus-extra total must equal the engine's
        utg_bufs + uab_var_bufs + one live raw block, to the byte.
        """
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)

        n_samples = 50_000
        n_snps = 500_000

        plan = plan_lmm_chunks(n_samples, n_snps, n_cvt, dispatch)

        # What pipeline_memory._compute_chunk prices: rotation buffers
        # (n_buffers x chunk x n_samples x 8, i.e. utg_bufs) plus the
        # dispatch-specific extra (lmm_extra_bytes_per_snp), plus the one
        # live raw genotype block at the plan's chunk width.
        rotation_gb = plan.chunk_size * n_samples * 8 * plan.n_buffers / 1e9
        raw_block_gb = plan.chunk_size * n_samples * 8 / 1e9
        _, extra_gb = _compute_chunk(n_samples, n_snps, n_cvt, lmm_mode)
        priced_gb = rotation_gb + raw_block_gb + extra_gb

        allocated_gb = _engine_allocation_gb(n_samples, n_cvt, plan, dispatch)

        assert priced_gb == pytest.approx(allocated_gb, rel=1e-9), (
            f"{dispatch}: priced {priced_gb:.3f}GB != allocated {allocated_gb:.3f}GB"
        )

    def test_soa_split_pipelined_priced_extra_is_double_sequential(self, monkeypatch):
        """Direct regression for the measured 17.1GB vs 34.3GB gap (#finding 1).

        n=50000, n_cvt=4 dispatches to SOA_SPLIT under LRT (mode 2) and
        pipelines at this scale, so the engine allocates uab_var_bufs twice
        (n_buffers=2). The extra must price at exactly double the
        n_buffers=1 figure, not the same figure the pre-fix preflight priced
        regardless of whether the run pipelines.
        """
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 640.0)

        n_samples = 50_000
        n_snps = 500_000
        n_cvt = 4
        lmm_mode = 2  # LRT: n_cvt >= 2 dispatches to SOA_SPLIT

        plan = plan_lmm_chunks(n_samples, n_snps, n_cvt, DispatchPath.SOA_SPLIT)
        assert plan.use_pipeline, "this case must pipeline for the regression to bite"
        assert plan.n_buffers == 2

        _, extra_gb = _compute_chunk(n_samples, n_snps, n_cvt, lmm_mode)

        from jamma.lmm.chunk_sizing import lmm_extra_bytes_per_snp

        sequential_extra_gb = (
            plan.chunk_size
            * lmm_extra_bytes_per_snp(
                n_samples, n_cvt, DispatchPath.SOA_SPLIT, n_buffers=1
            )
            / 1e9
        )
        assert extra_gb == pytest.approx(2 * sequential_extra_gb, rel=1e-9)
