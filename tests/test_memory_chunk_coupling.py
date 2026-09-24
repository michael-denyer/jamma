"""Regression tests verifying memory estimation threads chunk size and n_cvt.

These tests assert on observable outputs — estimated GB totals and whether
the live preflight gate (``memory_preflight``) raises MemoryError — rather
than on internal call counts of ``LmmChunkPlan.plan`` /
``lmm_extra_bytes_per_snp``. This
follows CLAUDE.md: assert observable behavior, not delegation plumbing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jamma.core import memory
from jamma.core.memory import array_gb
from jamma.core.threading import is_blas_controllable
from jamma.genotype.dataset import GenotypeEncoding
from jamma.lmm.association_plan import plan_association
from jamma.lmm.chunk_sizing import (
    LmmChunkPlan,
    chunk_budget_bytes,
    compute_chunk_size_numpy,
    lmm_extra_bytes_per_snp,
)
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.pab import n_index
from jamma.lmm.schema import LmmConfig, LmmMode
from jamma.lmm.workspace import WorkspaceSpec
from jamma.pipeline_config import PipelineConfig
from tests.builders import association_price_plan, empty_workspace
from tests.fakes import use_fake_psutil
from tests.support import preflight, requires_c

pytestmark = pytest.mark.tier0


def _plan(
    n_samples: int,
    n_snps: int,
    n_cvt: int,
    dispatch: DispatchPath,
    *,
    mem_budget_bytes: int | None = None,
    blas_controllable: bool | None = None,
    max_chunk_size: int | None = None,
) -> LmmChunkPlan:
    """Plan a chunk geometry the way ``plan_association`` does.

    The planner is pure; ``plan_association`` resolves the budget from
    ``memory.available_ram_gb`` and the BLAS controllability once and passes
    them in. Tests that pin RAM through that seam get the same plan here.
    """
    budget = (
        mem_budget_bytes
        if mem_budget_bytes is not None
        else chunk_budget_bytes(
            None, available_bytes=int(memory.available_ram_gb() * 1e9)
        )
    )
    return LmmChunkPlan.plan(
        n_samples,
        n_snps,
        n_cvt,
        dispatch,
        budget_bytes=budget,
        blas_controllable=(
            is_blas_controllable() if blas_controllable is None else blas_controllable
        ),
        max_chunk_size=max_chunk_size,
    )


def _streaming_preflight(
    n_valid: int, n_snps: int, n_cvt: int, lmm_mode: LmmMode = 1
) -> None:
    """Drive the live preflight gate the way the pipeline does."""
    config = PipelineConfig(bfile=Path("unused"), lmm_mode=lmm_mode)
    plan = plan_association(
        n_valid,
        n_snps,
        config=LmmConfig(lmm_mode=lmm_mode),
        backend="numpy-streaming",
        n_cvt=n_cvt,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    )
    preflight(config, plan)


def test_chunk_size_varies_with_scale():
    """The sizer returns different values as the sample count scales."""
    budget = int(2e9)
    small = compute_chunk_size_numpy(
        1_410, 12_000, dispatch=DispatchPath.FUSED, mem_budget_bytes=budget
    )
    large = compute_chunk_size_numpy(
        100_000, 500_000, dispatch=DispatchPath.FUSED, mem_budget_bytes=budget
    )
    assert small != large, "Chunk size should vary with scale"


def test_fallback_uab_iab_price_scales_with_n_cvt():
    """Observable invariant: the fallback's Uab/Iab price grows with n_cvt.

    The runner's preflight underestimate bug (jamma-ca6p) was possible
    precisely because callers defaulted to n_cvt=1. If a future refactor
    silently re-introduces that default, this test will catch it via the
    observable GB output, not by inspecting call arguments.
    """
    chunk = 10_000

    def price(n_cvt: int) -> float:
        per_snp = lmm_extra_bytes_per_snp(1000, n_cvt, DispatchPath.NUMPY_FALLBACK)
        return chunk * per_snp / 1e9

    gb_n1, gb_n5, gb_n10 = price(1), price(5), price(10)

    assert gb_n1 < gb_n5 < gb_n10, (
        f"Uab/Iab must grow with n_cvt, got n_cvt=1:{gb_n1} < 5:{gb_n5} < 10:{gb_n10}"
    )


def test_streaming_quote_scales_with_n_cvt():
    """The streaming association quote grows with n_cvt on the NumPy fallback.

    Higher-level observable check of the same invariant: the preflight
    total reported to users must be larger for multi-covariate runs.
    """
    quote_n1, quote_n10 = (
        association_price_plan(
            "streaming", n_samples=2000, n_snps=100_000, chunk_size=10_000, n_cvt=n_cvt
        )
        .price(eigen=None)
        .association_gb
        for n_cvt in (1, 10)
    )
    assert quote_n10 > quote_n1, (
        f"Quote must grow with n_cvt, got n_cvt=1:{quote_n1} n_cvt=10:{quote_n10}"
    )


def test_preflight_narrows_when_n_cvt_inflates_toward_available(monkeypatch):
    """Regression for jamma-ca6p: the preflight gate must thread n_cvt.

    Every C dispatch path now holds no per-SNP batch array (the general
    workspace forms Uab on the fly), so n_cvt no longer inflates a C-path
    preflight. The NumPy fallback still materialises the full Uab batch,
    which grows with n_cvt, so this pins the bug there instead: n_cvt=1
    fits comfortably in the pinned budget; n_cvt=90 inflates the full Uab
    batch toward the ceiling. A feasible run must now narrow rather than fail.
    If the preflight silently defaults n_cvt back to 1, both plans retain the
    same width.
    """
    from jamma.core import memory
    from jamma.lmm import accel

    monkeypatch.setattr(accel, "_accel", None)  # force NUMPY_FALLBACK

    # Pin available memory to a small fixed value to make the threshold
    # deterministic across machines. Both the chunk sizer and the
    # sufficiency check read the one seam.
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 1.0)

    n_samples = 800
    n_snps = 10_000

    low = plan_association(
        n_samples,
        n_snps,
        config=LmmConfig(lmm_mode=2),
        backend="numpy-streaming",
        n_cvt=1,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    )
    high = plan_association(
        n_samples,
        n_snps,
        config=LmmConfig(lmm_mode=2),
        backend="numpy-streaming",
        n_cvt=90,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    )

    assert high.conservative_chunks.chunk_size < low.conservative_chunks.chunk_size
    _streaming_preflight(n_samples, n_snps, n_cvt=1, lmm_mode=2)
    _streaming_preflight(n_samples, n_snps, n_cvt=90, lmm_mode=2)


def test_preflight_succeeds_at_low_n_cvt():
    """Sanity: the default n_cvt=1 call on a tiny dataset must pass on any
    machine with >1GB free. Exists to catch regressions where the preflight
    spuriously rejects small inputs.
    """
    # 100 samples x 1000 SNPs is negligible — must fit everywhere.
    _streaming_preflight(100, 1000, n_cvt=1)


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


# With the extension loaded, every n_cvt and lmm_mode selects FUSED;
# accel=False always selects
# NUMPY_FALLBACK regardless of n_cvt/lmm_mode. One representative lmm_mode
# per path, matching select_dispatch_path's own resolution table.
_DISPATCH_CASES = [
    pytest.param(1, 1, True, DispatchPath.FUSED, id="fused"),
    pytest.param(2, 1, True, DispatchPath.FUSED, id="fused_general"),
    pytest.param(2, 2, True, DispatchPath.FUSED, id="fused_general_lrt"),
    pytest.param(1, 3, True, DispatchPath.FUSED, id="fused_score_ws"),
    pytest.param(1, 2, True, DispatchPath.FUSED, id="fused_lrt_ws"),
    pytest.param(4, 1, False, DispatchPath.NUMPY_FALLBACK, id="numpy_fallback"),
]


def _engine_allocation_gb(
    n_samples: int,
    n_cvt: int,
    plan,
    dispatch: DispatchPath,
    *,
    include_raw_block: bool,
) -> float:
    """The exact bytes the chunk engine's live buffers hold for this plan.

    Reproduces each dispatch path's real buffer shapes directly, rather than
    calling back into the pricing helpers under test, so this is an
    independent check of the allocation, not a tautology:

    - Every path: ``utg_bufs``, ``_ChunkEngine``'s rotation output buffer,
      shape ``(chunk_size, n_samples)`` per live buffer.
    - NUMPY_FALLBACK: the kernel holds ``Uab_batch`` (shape
      ``(chunk_size, n_samples, n_index)``, from
      ``uab.batch_compute_uab_numpy``'s documented return
      shape) and ``Iab_batch`` (shape ``(chunk_size, n_cvt + 2, n_index)``,
      from ``batch_compute_iab_numpy``) concurrently during compute. Derived
      from ``n_index`` directly, not from ``lmm_extra_bytes_per_snp``, which
      is the function this test exists to check.

    Args:
        include_raw_block: Whether to add the raw genotype block
            (``geno_buf`` / the streaming dataset's block) the chunk source hands
            ``prepare_genotypes()``. True for the streaming comparison. False for the
            batch comparison, which holds the whole genotype matrix instead.
    """
    utg_bytes = plan.chunk_size * n_samples * 8 * plan.n_buffers
    extra_bytes = 0
    if dispatch is DispatchPath.NUMPY_FALLBACK:
        idx = n_index(n_cvt)
        uab_batch_bytes = plan.chunk_size * n_samples * idx * 8
        iab_batch_bytes = plan.chunk_size * (n_cvt + 2) * idx * 8
        # Once, not per buffer: compute_and_write builds Uab on the calling
        # thread, one consumer at a time.
        extra_bytes = uab_batch_bytes + iab_batch_bytes
    raw_block_bytes = 0
    if include_raw_block:
        # The raw genotype block the chunk source hands prepare(): one
        # buffer live at a time regardless of pipelining
        # (chunk_runner_numpy.py's _drive_pipeline overlaps a rotated
        # buffer with the next prepare() call, never two raw reads at
        # once).
        raw_block_bytes = plan.chunk_size * n_samples * 8
    return (utg_bytes + extra_bytes + raw_block_bytes) / 1e9


def _priced_streaming_lmm_phase_gb(
    monkeypatch: pytest.MonkeyPatch,
    n_samples: int,
    n_snps: int,
    n_cvt: int,
    lmm_mode: LmmMode,
    accel: bool,
) -> tuple[float, LmmChunkPlan, WorkspaceSpec]:
    """The streaming preflight's real priced association-phase total.

    Plans through the real ``plan_association`` and reads the quote's own
    ``association_gb`` rather than recomputing the formula here, so a
    regression in either the planner or ``price()`` is visible. The planner
    derives its dispatch from the real loaded ``jamma.lmm.accel._accel``, so
    ``accel`` is pinned here to match the case under test rather than
    whatever extension state this test process happens to have loaded.
    """
    from jamma.lmm import accel as accel_module

    loaded_accel = accel_module.require() if accel else None
    monkeypatch.setattr(accel_module, "_accel", loaded_accel)
    execution = plan_association(
        n_samples,
        n_snps,
        config=LmmConfig(lmm_mode=lmm_mode),
        backend="numpy-streaming",
        n_cvt=n_cvt,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    )
    quote = execution.price(eigen=None)
    return (
        quote.association_gb,
        execution.conservative_chunks,
        execution.workspace,
    )


def _priced_batch_quote_gb(
    n_samples: int,
    n_snps: int,
    n_cvt: int,
    dispatch: DispatchPath,
    *,
    n_buffers: int | None = None,
) -> tuple[float, LmmChunkPlan]:
    """The batch quote ``price()`` gives the chunk ``LmmChunkPlan.plan`` picks.

    The plan carries an empty kernel workspace, so the quote holds U, the
    genotype matrix, and the chunk buffers alone. ``n_buffers`` overrides the
    planned buffer count at the same chunk width.
    """
    chunks = _plan(n_samples, n_snps, n_cvt, dispatch)
    if n_buffers is not None:
        chunks = replace(chunks, n_buffers=n_buffers, use_pipeline=n_buffers > 1)
    plan = replace(
        association_price_plan(
            "batch",
            n_samples=n_samples,
            n_snps=n_snps,
            chunk_size=chunks.chunk_size,
            n_cvt=n_cvt,
            dispatch=dispatch,
        ),
        conservative_chunks=chunks,
        workspace=empty_workspace(dispatch, n_samples, n_samples, n_cvt),
    )
    return plan.price(eigen=None).association_gb, chunks


class TestChunkPlanMatchesEngine:
    """One LmmChunkPlan, computed once: the engine allocates from it, and the
    preflight prices from it. These pin that the two routes cannot diverge.
    """

    @pytest.mark.parametrize("n_cvt,lmm_mode,accel,dispatch", _DISPATCH_CASES)
    def test_plan_chunk_size_matches_engine(
        self, monkeypatch, n_cvt, lmm_mode, accel, dispatch
    ):
        """LmmChunkPlan.plan' chunk size is exactly what the engine sizes.

        chunk_runner_numpy.run_lmm_chunk_source_numpy_group sizes the
        engine's _ChunkEngine from the chunk_size/n_chunks/n_buffers that
        LmmChunkPlan.plan returns for these same arguments, so calling it
        directly here reproduces the engine's own sizing decision.
        """
        from jamma.lmm.dispatch import select_dispatch_path

        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)

        n_samples = 50_000
        n_filtered = 500_000

        # The parametrized dispatch must be what select_dispatch_path derives
        # for accel, or this case is testing an unreachable combination.
        assert select_dispatch_path(accel=accel) is dispatch

        plan = _plan(n_samples, n_filtered, n_cvt, dispatch)

        assert plan.chunk_size >= 1
        assert plan.n_chunks == (n_filtered + plan.chunk_size - 1) // plan.chunk_size
        assert plan.n_buffers in (1, 2)
        if not dispatch.is_native:
            # NUMPY_FALLBACK never pipelines.
            assert plan.n_buffers == 1
            assert not plan.use_pipeline

    @pytest.mark.parametrize(
        "n_cvt,lmm_mode,accel,dispatch",
        [
            pytest.param(*case.values, id=case.id, marks=requires_c)
            if case.values[2]
            else case
            for case in _DISPATCH_CASES
        ],
    )
    def test_streaming_preflight_priced_bytes_match_engine_allocation(
        self, monkeypatch, n_cvt, lmm_mode, accel, dispatch
    ):
        """The streaming quote never under-prices the engine's real buffer
        allocation, across every dispatch path.

        Regression for the P6 finding (a per-SNP batch buffer priced at one
        buffer while the engine allocates n_buffers) and for Gap A
        (pipeline_buffers hardcoded to 2 regardless of whether the plan
        actually pipelines). Both would surface here because this drives the
        real planner and ``price()`` end to end rather than recomputing the
        formula.
        """
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)

        n_samples = 50_000
        n_snps = 500_000

        priced_gb, plan, workspace = _priced_streaming_lmm_phase_gb(
            monkeypatch, n_samples, n_snps, n_cvt, lmm_mode, accel
        )
        allocated_gb = _engine_allocation_gb(
            n_samples, n_cvt, plan, dispatch, include_raw_block=True
        ) + array_gb(n_samples, n_samples)
        allocated_gb += plan.chunk_size * workspace.bytes_per_snp / 1e9

        assert priced_gb >= allocated_gb - 1e-12, (
            f"{dispatch}: priced {priced_gb:.3f}GB < allocated {allocated_gb:.3f}GB"
        )

    @pytest.mark.parametrize("n_cvt,lmm_mode,accel,dispatch", _DISPATCH_CASES)
    def test_batch_gate_priced_bytes_are_at_least_engine_allocation(
        self, monkeypatch, n_cvt, lmm_mode, accel, dispatch
    ):
        """The batch quote never under-prices U, the genotype matrix, and the
        engine's real buffer allocation, across every dispatch path.

        Regression for Gap B: the batch quote once had no n_buffers concept,
        so a pipelined batch run (n_buffers=2) was priced at one buffer's
        worth.
        """
        del accel, lmm_mode  # dispatch alone determines pricing here
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)

        n_samples = 50_000
        n_snps = 500_000

        priced_gb, plan = _priced_batch_quote_gb(n_samples, n_snps, n_cvt, dispatch)
        allocated_gb = (
            _engine_allocation_gb(
                n_samples, n_cvt, plan, dispatch, include_raw_block=False
            )
            + array_gb(n_samples, n_samples)
            + array_gb(n_samples, n_snps)
        )

        assert priced_gb >= allocated_gb - 1e-9, (
            f"{dispatch}: priced {priced_gb:.3f}GB < allocated {allocated_gb:.3f}GB"
        )

    def test_batch_gate_priced_bytes_scale_with_pipelining(self, monkeypatch):
        """Direct regression for Gap B: the batch quote must change when the
        plan pipelines, not stay pinned to one buffer.

        Forces a pipelining case (n_chunks >= _MIN_PIPELINE_CHUNKS, a
        is_native dispatch) by pinning a small RAM budget so the sizer picks
        many small chunks, then compares the quote at one buffer against the
        plan's two: the second buffer adds exactly one rotation buffer.
        """
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 8.0)

        n_samples = 50_000
        n_snps = 500_000
        n_cvt = 2

        # n_cvt >= 2, mode 1 -> FUSED (is_native=True); dispatch is
        # passed directly below, so no lmm_mode is needed.
        dispatch = DispatchPath.FUSED
        plan = _plan(n_samples, n_snps, n_cvt, dispatch)
        assert plan.use_pipeline, "this case must pipeline for the regression to bite"
        assert plan.n_buffers == 2

        one_buffer_gb, _ = _priced_batch_quote_gb(
            n_samples, n_snps, n_cvt, dispatch, n_buffers=1
        )
        planned_gb, _ = _priced_batch_quote_gb(n_samples, n_snps, n_cvt, dispatch)

        assert planned_gb - one_buffer_gb == pytest.approx(
            array_gb(n_samples, plan.chunk_size), rel=1e-9
        )

    def test_plan_memory_priced_bytes_scale_with_non_pipelining(self, monkeypatch):
        """Direct regression for Gap A, through the real planner and price().

        The streaming quote once priced two rotation buffers unconditionally,
        regardless of whether the chunk plan actually pipelines.
        NUMPY_FALLBACK never pipelines (plan.n_buffers is always 1), so at
        parameters where the LMM chunk-loop term dominates the workflow peak
        (small n_samples keeps the O(n^2) terms negligible beside the Uab/Iab
        extra), the quote must equal what a real n_buffers=1 run allocates.
        """
        from jamma.lmm import accel

        monkeypatch.setattr(memory, "available_ram_gb", lambda: 8.0)
        monkeypatch.setattr(accel, "_accel", None)  # force NUMPY_FALLBACK

        n_samples = 2_000
        n_snps = 300_000
        n_cvt = 4
        lmm_mode = 1

        exec_plan = plan_association(
            n_samples,
            n_snps,
            config=LmmConfig(lmm_mode=lmm_mode),
            backend="numpy-streaming",
            n_cvt=n_cvt,
            genotype_encoding=GenotypeEncoding.HARD_CALLS,
        )
        dispatch = DispatchPath.NUMPY_FALLBACK
        plan = exec_plan.conservative_chunks
        assert not plan.use_pipeline, "this case must not pipeline (plan.n_buffers=1)"
        assert plan.n_buffers == 1
        mem_plan = exec_plan.price(eigen=None)

        allocated_gb = _engine_allocation_gb(
            n_samples, n_cvt, plan, dispatch, include_raw_block=True
        ) + array_gb(n_samples, n_samples)
        allocated_gb += (
            exec_plan.workspace.fixed_bytes
            + plan.chunk_size * exec_plan.workspace.bytes_per_snp
        ) / 1e9
        # The quote also reserves the chunk read's C-order copy. A run that
        # does not pipeline reads between computes, so it never holds that
        # copy beside Uab/Iab; the quote exceeds its live set by exactly it.
        allocated_gb += array_gb(n_samples, min(plan.chunk_size, n_snps))
        assert mem_plan.total_peak_gb == pytest.approx(allocated_gb, rel=1e-9), (
            f"quoted total {mem_plan.total_peak_gb:.3f}GB != "
            f"engine allocation {allocated_gb:.3f}GB"
        )


@requires_c
def test_plan_association_sizes_against_the_real_chunk(monkeypatch):
    """plan_association must price the chunk the run will allocate, not 20,000.

    Measured at n=50000, snps=500000 on the FUSED path: the batch quote at a
    20,000-SNP chunk and one buffer is 228.0GB,
    while the chunk ``LmmChunkPlan.plan`` really plans (24,940 SNPs over two
    buffers, narrowed to 19,940 here) quotes 236.0GB. A machine with 240GB
    available sits strictly between the two thresholds once the 10GB safety
    margin applies: the stale default says "fits" (batch), the real chunk says
    "does not fit" (streaming). At trunk, ``runner.py`` priced the batch
    phase without the planned chunk width or buffer count and picked batch
    here; that flips the execution mode a machine near this line gets, in
    the direction that silently under-estimates memory.
    """
    use_fake_psutil(monkeypatch, available=240e9)

    plan = plan_association(
        50_000,
        500_000,
        config=LmmConfig(lmm_mode=1),
        n_cvt=1,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    ).summary

    assert plan.mode == "streaming", (
        f"expected streaming (the real chunk needs 236.0GB, which does not "
        f"clear the margin against 240GB available), got {plan.mode!r} "
        f"({plan.reason})"
    )


@requires_c
def test_plan_association_mem_budget_narrows_the_chunk(monkeypatch):
    """--mem-budget must narrow the chunk plan_association prices.

    A tight ``mem_budget`` should shrink the chunk plan feeds into the memory
    estimate, in turn shrinking the estimated total. At trunk, ``mem_budget``
    never reached the mode selector at all.
    """
    use_fake_psutil(monkeypatch, available=240e9)

    unbudgeted = plan_association(
        50_000,
        500_000,
        config=LmmConfig(lmm_mode=1),
        n_cvt=1,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    ).summary
    budgeted = plan_association(
        50_000,
        500_000,
        config=LmmConfig(lmm_mode=1, mem_budget=1.0),
        n_cvt=1,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    ).summary

    # 236.0GB does not clear the 10GB safety margin against 240GB.
    assert unbudgeted.mode == "streaming"
    assert "exceeds 240.0GB capacity" in unbudgeted.reason
    # The 1GB ceiling cannot hold the 20GB eigenvector matrix even at a
    # one-SNP chunk, so auto must not select an impossible batch plan.
    assert budgeted.mode == "streaming"
    assert "exceeds 1.0GB capacity" in budgeted.reason


def test_plan_association_keeps_wide_chunks_when_u_exceeds_the_chunk_budget(
    monkeypatch,
):
    use_fake_psutil(monkeypatch, available=500e9)

    chunks = plan_association(
        100_000,
        50_000,
        config=LmmConfig(lmm_mode=1),
        backend="numpy",
        n_cvt=1,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    ).conservative_chunks

    assert chunks.chunk_size >= 1_000, chunks


def test_chunk_plan_honors_mem_budget_bytes():
    """LmmChunkPlan.plan must narrow the chunk when given mem_budget_bytes.

    compute_chunk_size_numpy already accepted mem_budget_bytes, but
    LmmChunkPlan.plan (the single sizing decision the engine allocates from
    and the preflight prices from) had no parameter to pass it through, so
    every production caller's chunk plan ignored --mem-budget.
    """
    dispatch = DispatchPath.FUSED
    n_samples, n_snps, n_cvt = 50_000, 500_000, 1

    auto = _plan(n_samples, n_snps, n_cvt, dispatch)
    budgeted = _plan(n_samples, n_snps, n_cvt, dispatch, mem_budget_bytes=int(1e9))

    assert budgeted.chunk_size < auto.chunk_size


def test_pipeline_memory_plan_honors_mem_budget(monkeypatch):
    """The pipeline preflight prices the budget-aware chunk geometry."""
    from jamma.lmm import accel

    monkeypatch.setattr(accel, "_accel", None)
    n_samples, n_snps, n_cvt = 30, 200, 1
    mem_budget = 12e-6
    execution = plan_association(
        n_samples,
        n_snps,
        config=LmmConfig(mem_budget=mem_budget),
        backend="numpy",
        n_cvt=n_cvt,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    )
    planned = execution.price(eigen=None)

    assert execution.conservative_chunks.chunk_size == 1
    assert planned.compute_chunk_size == execution.conservative_chunks.chunk_size
    assert planned.total_peak_gb > mem_budget
    with pytest.raises(MemoryError, match="exceeds budget"):
        memory.require(
            planned.total_peak_gb,
            64.0,
            "fallback association",
            budget_gb=mem_budget,
        )


def test_chunk_engine_requests_budget_aware_geometry(monkeypatch):
    """The final chunk engine requests the width allowed by mem_budget."""
    from jamma.genotype.snp_stats import SnpSelection
    from jamma.genotype.variants import SnpMeta
    from jamma.lmm import accel
    from jamma.lmm.chunk_runner_numpy import (
        PhenotypeChunkJob,
        run_lmm_chunk_source_numpy_group,
    )
    from jamma.lmm.genotype_source import PreparedGenotypes, SampleBasis
    from jamma.lmm.prepare_common import NullFit, RotatedBasis
    from jamma.lmm.schema import LmmConfig

    monkeypatch.setattr(accel, "_accel", None)
    n_samples, n_snps, n_cvt = 30, 200, 1
    mem_budget = 12e-6
    basis = RotatedBasis(
        eigenvalues=np.ones(n_samples),
        U=np.eye(n_samples),
        W=np.ones((n_samples, n_cvt)),
        UtW=np.ones((n_samples, n_cvt)),
    )
    fit = NullFit(
        Uty=np.ones(n_samples),
        logl_H0=-1.0,
        Hi_eval_null=np.ones(n_samples),
        pve=None,
        pve_se=None,
    )
    requested: list[int] = []

    class GeometryObserved(Exception):
        pass

    def observe_geometry(chunk_size: int):
        requested.append(chunk_size)
        raise GeometryObserved

    indices = np.arange(n_snps, dtype=np.intp)
    genotypes = PreparedGenotypes(
        snp_meta=SnpMeta(
            chr=np.full(n_snps, "1"),
            rs=np.array([f"rs{i}" for i in indices]),
            pos=indices,
            a1=np.full(n_snps, "A"),
            a0=np.full(n_snps, "G"),
        ),
        selection=SnpSelection(
            indices=indices,
            local_indices=indices,
            mask=np.ones(n_snps, dtype=bool),
            filtered_afs=np.zeros(n_snps),
            filtered_miss=np.zeros(n_snps, dtype=int),
            filtered_means=np.zeros(n_snps),
        ),
        n_unexpected=0,
        analyzed_sample_count=n_samples,
        sample_basis=SampleBasis(np.arange(n_samples), n_samples),
        chunk_factory=observe_geometry,
    )

    exec_plan = plan_association(
        n_samples,
        n_snps,
        config=LmmConfig(mem_budget=mem_budget),
        backend="numpy",
        n_cvt=n_cvt,
        genotype_encoding=GenotypeEncoding.HARD_CALLS,
    )
    with pytest.raises(GeometryObserved):
        run_lmm_chunk_source_numpy_group(
            genotypes=genotypes,
            basis=basis,
            jobs=(PhenotypeChunkJob(fit, lambda _arrays, _start, _end: None),),
            config=LmmConfig(
                lmm_mode=1,
                mem_budget=mem_budget,
                show_progress=False,
            ),
            dispatch=exec_plan.dispatch,
            chunks=exec_plan.conservative_chunks.narrow(n_snps),
            workspace=exec_plan.workspace,
        )

    assert exec_plan.conservative_chunks.chunk_size == 1
    assert requested == [exec_plan.conservative_chunks.chunk_size]


def test_chunk_plan_splits_small_inputs_for_pipelining(monkeypatch):
    """A budget that fits every SNP in one chunk still splits a native
    run into enough chunks to overlap rotation with compute, while a plan the
    budget already splits past the pipeline threshold, a run with more
    samples than the cut is measured to help, or a controllable BLAS, is left
    alone."""
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)

    plan = _plan(1_410, 12_226, 1, DispatchPath.FUSED, blas_controllable=False)
    assert plan.n_chunks == 16
    assert plan.chunk_size == 765
    assert plan.use_pipeline

    fallback = _plan(
        1_410, 12_226, 1, DispatchPath.NUMPY_FALLBACK, blas_controllable=False
    )
    assert fallback.n_chunks == 1
    assert not fallback.use_pipeline

    at_bound = _plan(10_000, 5_000, 1, DispatchPath.FUSED, blas_controllable=False)
    assert at_bound.n_chunks == 16
    assert at_bound.use_pipeline

    past_bound = _plan(30_000, 5_000, 1, DispatchPath.FUSED, blas_controllable=False)
    assert past_bound.n_chunks == 1
    assert not past_bound.use_pipeline

    controllable = _plan(1_410, 12_226, 1, DispatchPath.FUSED, blas_controllable=True)
    assert controllable.n_chunks == 1
    assert not controllable.use_pipeline

    budgeted = _plan(
        1_410,
        12_226,
        1,
        DispatchPath.FUSED,
        mem_budget_bytes=int(1e6),
        blas_controllable=False,
    )
    assert budgeted.n_chunks > 16
    assert budgeted.chunk_size == compute_chunk_size_numpy(
        1_410,
        12_226,
        1,
        dispatch=DispatchPath.FUSED,
        mem_budget_bytes=int(1e6),
        pipeline_buffers=2,
    )


def test_chunk_plan_keeps_the_chunk_floor_on_tiny_inputs(monkeypatch):
    monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)

    plan = _plan(30, 400, 1, DispatchPath.FUSED, blas_controllable=False)
    assert plan.chunk_size == 100
    assert plan.n_chunks == 4
    assert not plan.use_pipeline
