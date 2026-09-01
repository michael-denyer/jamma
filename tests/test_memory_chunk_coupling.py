"""Regression tests verifying memory estimation threads chunk size and n_cvt.

These tests assert on observable outputs — estimated GB totals and whether
the live preflight gate (``memory_preflight``) raises MemoryError — rather
than on internal call counts of ``plan_lmm_chunks`` / ``_uab_iab_gb``. This
follows CLAUDE.md: assert observable behavior, not delegation plumbing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.core import memory
from jamma.core.memory import (
    _uab_iab_gb,
    estimate_lmm_memory,
    estimate_streaming_memory,
)
from jamma.lmm.association_plan import DEFAULT_STATS_CHUNK, plan_association
from jamma.lmm.chunk_sizing import (
    LmmChunkPlan,
    compute_chunk_size_numpy,
    lmm_extra_bytes_per_snp,
    plan_lmm_chunks,
)
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.pab import n_index
from jamma.lmm.schema import LmmMode
from jamma.pipeline_config import PipelineConfig
from jamma.pipeline_memory import memory_preflight
from tests.fakes import use_fake_psutil

pytestmark = pytest.mark.tier0


def _streaming_preflight(
    n_valid: int, n_snps: int, n_cvt: int, lmm_mode: LmmMode = 1
) -> None:
    """Drive the live preflight gate the way the pipeline does."""
    config = PipelineConfig(bfile=Path("unused"), lmm_mode=lmm_mode)
    plan = plan_association(
        n_valid,
        n_snps,
        requested="numpy-streaming",
        n_cvt=n_cvt,
        lmm_mode=lmm_mode,
    )
    memory_preflight(config, plan)


def test_chunk_size_varies_with_scale():
    """The sizer returns different values as the sample count scales."""
    small = compute_chunk_size_numpy(1_410, 12_000, dispatch=DispatchPath.FUSED)
    large = compute_chunk_size_numpy(100_000, 500_000, dispatch=DispatchPath.FUSED)
    assert small != large, "Chunk size should vary with scale"


def test_memory_estimate_uses_computed_chunk():
    """Memory estimates differ when using computed chunk sizes at different scales."""
    small_chunk = compute_chunk_size_numpy(1_410, 12_000, dispatch=DispatchPath.FUSED)
    large_chunk = compute_chunk_size_numpy(
        100_000, 500_000, dispatch=DispatchPath.FUSED
    )

    est_small = estimate_streaming_memory(1410, compute_chunk_size=small_chunk)
    est_large = estimate_streaming_memory(100_000, compute_chunk_size=large_chunk)

    assert est_small.total_peak_gb != est_large.total_peak_gb


def test_uab_iab_gb_scales_with_n_cvt():
    """Observable invariant: Uab/Iab buffers grow with n_cvt.

    The runner's preflight underestimate bug (jamma-ca6p) was possible
    precisely because callers defaulted to n_cvt=1. If a future refactor
    silently re-introduces that default, this test will catch it via the
    observable GB output, not by inspecting call arguments.
    """
    chunk = 10_000
    gb_n1 = _uab_iab_gb(n_samples=1000, chunk_size=chunk, n_cvt=1)
    gb_n5 = _uab_iab_gb(n_samples=1000, chunk_size=chunk, n_cvt=5)
    gb_n10 = _uab_iab_gb(n_samples=1000, chunk_size=chunk, n_cvt=10)

    assert gb_n1 < gb_n5 < gb_n10, (
        f"Uab/Iab must grow with n_cvt, got n_cvt=1:{gb_n1} < 5:{gb_n5} < 10:{gb_n10}"
    )


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


def test_preflight_raises_when_n_cvt_inflates_past_available(monkeypatch):
    """Regression for jamma-ca6p: the preflight gate must thread n_cvt.

    Every C dispatch path now holds no per-SNP batch array (the general
    workspace forms Uab on the fly), so n_cvt no longer inflates a C-path
    preflight. The NumPy fallback still materialises the full Uab batch,
    which grows with n_cvt, so this pins the bug there instead: n_cvt=1
    fits comfortably in the pinned budget; n_cvt=90 inflates the full Uab
    batch past it. If the preflight silently defaults n_cvt back to 1, both
    calls succeed — only a correctly threaded n_cvt produces the asymmetric
    pass/raise this asserts.
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

    # With n_cvt=1, peak should be well under 1.0GB available.
    _streaming_preflight(n_samples, n_snps, n_cvt=1, lmm_mode=2)

    # With n_cvt=90, the full Uab batch inflates past it.
    with pytest.raises(MemoryError):
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


# n_cvt=1 selects FUSED/FUSED_SCORE_WS/FUSED_LRT_WS depending on lmm_mode;
# n_cvt>=2 selects FUSED_GENERAL for every mode; accel=False always selects
# NUMPY_FALLBACK regardless of n_cvt/lmm_mode. One representative lmm_mode
# per path, matching select_dispatch_path's own resolution table.
_DISPATCH_CASES = [
    pytest.param(1, 1, True, DispatchPath.FUSED, id="fused"),
    pytest.param(2, 1, True, DispatchPath.FUSED_GENERAL, id="fused_general"),
    pytest.param(2, 2, True, DispatchPath.FUSED_GENERAL, id="fused_general_lrt"),
    pytest.param(1, 3, True, DispatchPath.FUSED_SCORE_WS, id="fused_score_ws"),
    pytest.param(1, 2, True, DispatchPath.FUSED_LRT_WS, id="fused_lrt_ws"),
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
            (``geno_buf`` / ``BedSource``'s chunk) the chunk source hands
            ``prepare()``. True for the streaming comparison, whose
            ``lmm_chunk_gb`` field prices it explicitly. False for the
            batch-gate comparison: ``estimate_lmm_memory`` prices the whole
            genotype matrix once as its separate ``genotypes_gb`` term and
            never claims to price the per-chunk raw block inside
            ``lmm_batch_gb``, so including it here would fault the batch
            gate for a term it was never designed to hold.
    """
    utg_bytes = plan.chunk_size * n_samples * 8 * plan.n_buffers
    extra_bytes = 0
    if dispatch is DispatchPath.NUMPY_FALLBACK:
        idx = n_index(n_cvt)
        uab_batch_bytes = plan.chunk_size * n_samples * idx * 8
        iab_batch_bytes = plan.chunk_size * (n_cvt + 2) * idx * 8
        extra_bytes = (uab_batch_bytes + iab_batch_bytes) * plan.n_buffers
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
) -> float:
    """The streaming preflight's real priced LMM-phase total.

    Calls ``estimate_streaming_memory`` the way ``pipeline_memory.plan_memory``
    builds it (same ``pipeline_buffers``/``compute_chunk_size``/``uab_iab_gb``
    from the same ``_compute_chunk``), and reads its own ``peak_lmm_gb``
    field rather than recomputing the LMM-phase formula here, so a
    regression in either ``plan_memory`` or ``estimate_streaming_memory``
    itself is visible. ``_compute_chunk`` derives its dispatch from the
    real loaded ``jamma.lmm.accel._accel``, so ``accel`` is pinned here to
    match the case under test rather than whatever extension state this
    test process happens to have loaded.
    """
    from jamma.lmm import accel as accel_module

    monkeypatch.setattr(accel_module, "_accel", object() if accel else None)
    execution = plan_association(
        n_samples,
        n_snps,
        requested="numpy-streaming",
        n_cvt=n_cvt,
        lmm_mode=lmm_mode,
    )
    chunk_plan = execution.conservative_chunks
    uab_iab_gb = (
        chunk_plan.chunk_size
        * lmm_extra_bytes_per_snp(
            n_samples,
            n_cvt,
            execution.dispatch,
            n_buffers=chunk_plan.n_buffers,
        )
        / 1e9
    )
    est = estimate_streaming_memory(
        n_samples,
        chunk_size=DEFAULT_STATS_CHUNK,
        n_cvt=n_cvt,
        pipeline_buffers=chunk_plan.n_buffers,
        compute_chunk_size=chunk_plan.chunk_size,
        uab_iab_gb=uab_iab_gb,
    )
    return est.peak_lmm_gb


def _streaming_lmm_phase_non_buffer_terms_gb(
    n_samples: int, compute_chunk_size: int, n_grid: int = 50
) -> float:
    """The streaming LMM phase's terms outside ``_ChunkEngine``'s buffers.

    ``peak_lmm_gb`` is ``eigenvectors_gb + lmm_chunk_gb + rotation_buffer_gb
    + grid_reml_gb + uab_iab_gb``; the last three are what
    ``_engine_allocation_gb`` reproduces from the chunk engine's own buffer
    shapes. ``eigenvectors_gb`` (the persistent U matrix) and
    ``grid_reml_gb`` (the REML grid-search scratch, sized at
    ``compute_chunk_size`` like the other per-chunk buffers) are real memory
    the LMM phase holds too, just not part of ``_ChunkEngine``'s buffers, so
    the byte-exact comparison needs them added back on. Uses the same plain
    geometric helpers ``estimate_streaming_memory`` itself calls
    (``square_matrix_gb``, ``array_gb``), not the pricing logic under test.
    """
    from jamma.core.eigen_plan import array_gb, square_matrix_gb

    return square_matrix_gb(n_samples) + array_gb(n_grid, compute_chunk_size)


def _priced_batch_lmm_phase_gb(
    n_samples: int, n_snps: int, n_cvt: int, dispatch: DispatchPath
) -> tuple[float, LmmChunkPlan]:
    """The batch gate's real priced LMM-batch total (``estimate_lmm_memory``).

    Plans the chunk the way ``runner_numpy.run_lmm_association_numpy``'s
    ``check_memory`` gate does, then reads ``lmm_batch_gb`` off the real
    ``MemoryBreakdown`` ``estimate_lmm_memory`` returns.
    """
    chunk_plan = plan_lmm_chunks(n_samples, n_snps, n_cvt, dispatch)
    est = estimate_lmm_memory(
        n_samples,
        n_snps,
        lmm_batch_size=chunk_plan.chunk_size,
        n_cvt=n_cvt,
        n_buffers=chunk_plan.n_buffers,
    )
    return est.lmm_batch_gb, chunk_plan


class TestChunkPlanMatchesEngine:
    """One LmmChunkPlan, computed once: the engine allocates from it, and the
    preflight prices from it. These pin that the two routes cannot diverge.
    """

    @pytest.mark.parametrize("n_cvt,lmm_mode,accel,dispatch", _DISPATCH_CASES)
    def test_plan_chunk_size_matches_engine(
        self, monkeypatch, n_cvt, lmm_mode, accel, dispatch
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
        # actually derives for (n_cvt, lmm_mode, accel), or this case is
        # testing an unreachable combination.
        assert (
            select_dispatch_path(
                n_cvt, parse_lmm_mode(lmm_mode), accel=accel, log_choices=False
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

    @pytest.mark.parametrize("n_cvt,lmm_mode,accel,dispatch", _DISPATCH_CASES)
    def test_streaming_preflight_priced_bytes_match_engine_allocation(
        self, monkeypatch, n_cvt, lmm_mode, accel, dispatch
    ):
        """estimate_streaming_memory's real peak_lmm_gb equals the engine's
        real buffer allocation, to the byte, across every dispatch path.

        Regression for the P6 finding (a per-SNP batch buffer priced at one
        buffer while the engine allocates n_buffers) and for the
        coordinator-flagged Gap A (pipeline_buffers hardcoded to 2 in
        plan_memory's streaming branch regardless of whether the plan
        actually pipelines). Both would surface here because this drives
        the real estimate_streaming_memory entry point end to end rather
        than recomputing its formula.
        """
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)

        n_samples = 50_000
        n_snps = 500_000

        plan = plan_lmm_chunks(n_samples, n_snps, n_cvt, dispatch)
        priced_gb = _priced_streaming_lmm_phase_gb(
            monkeypatch, n_samples, n_snps, n_cvt, lmm_mode, accel
        )
        allocated_gb = _engine_allocation_gb(
            n_samples, n_cvt, plan, dispatch, include_raw_block=True
        ) + _streaming_lmm_phase_non_buffer_terms_gb(n_samples, plan.chunk_size)

        assert priced_gb == pytest.approx(allocated_gb, rel=1e-9), (
            f"{dispatch}: priced {priced_gb:.3f}GB != allocated {allocated_gb:.3f}GB"
        )

    @pytest.mark.parametrize("n_cvt,lmm_mode,accel,dispatch", _DISPATCH_CASES)
    def test_batch_gate_priced_bytes_are_at_least_engine_allocation(
        self, monkeypatch, n_cvt, lmm_mode, accel, dispatch
    ):
        """estimate_lmm_memory's real lmm_batch_gb never under-prices the
        engine's real buffer allocation, across every dispatch path.

        Regression for the coordinator-flagged Gap B: estimate_lmm_memory
        had no n_buffers concept, so a pipelined batch run (n_buffers=2)
        was priced at one buffer's worth. estimate_lmm_memory is the
        documented "generic estimate" for the full-materialization path
        (it prices the full Uab+Iab batch shape regardless of the real
        dispatch, unlike the streaming estimator's dispatch-aware
        uab_iab_gb), so it can price above the real allocation; it must
        never price below it.
        """
        del accel, lmm_mode  # dispatch alone determines pricing here
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 64.0)

        n_samples = 50_000
        n_snps = 500_000

        priced_gb, plan = _priced_batch_lmm_phase_gb(n_samples, n_snps, n_cvt, dispatch)
        allocated_gb = _engine_allocation_gb(
            n_samples, n_cvt, plan, dispatch, include_raw_block=False
        )

        assert priced_gb >= allocated_gb - 1e-9, (
            f"{dispatch}: priced {priced_gb:.3f}GB < allocated {allocated_gb:.3f}GB"
        )

    def test_batch_gate_priced_bytes_scale_with_pipelining(self, monkeypatch):
        """Direct regression for Gap B: the batch gate's priced total must
        change when the plan pipelines, not stay pinned to one buffer.

        Forces a pipelining case (n_chunks >= _MIN_PIPELINE_CHUNKS, a
        use_split dispatch) by pinning a small RAM budget so the sizer picks
        many small chunks, then compares the gate's real
        estimate_lmm_memory(n_buffers=1) against n_buffers=plan.n_buffers:
        before the fix these were identical regardless of plan.n_buffers.
        """
        monkeypatch.setattr(memory, "available_ram_gb", lambda: 8.0)

        n_samples = 50_000
        n_snps = 500_000
        n_cvt = 2

        # n_cvt >= 2, mode 1 -> FUSED_GENERAL (use_split=True); dispatch is
        # passed directly below, so no lmm_mode is needed.
        dispatch = DispatchPath.FUSED_GENERAL
        plan = plan_lmm_chunks(n_samples, n_snps, n_cvt, dispatch)
        assert plan.use_pipeline, "this case must pipeline for the regression to bite"
        assert plan.n_buffers == 2

        est_one_buffer = estimate_lmm_memory(
            n_samples, n_snps, lmm_batch_size=plan.chunk_size, n_cvt=n_cvt, n_buffers=1
        )
        est_real, _ = _priced_batch_lmm_phase_gb(n_samples, n_snps, n_cvt, dispatch)

        assert est_real == pytest.approx(2 * est_one_buffer.lmm_batch_gb, rel=1e-9)

    def test_plan_memory_priced_bytes_scale_with_non_pipelining(self, monkeypatch):
        """Direct regression for Gap A, calling plan_memory itself.

        pipeline_memory.plan_memory's streaming branch passed
        pipeline_buffers=2 to estimate_streaming_memory unconditionally,
        regardless of whether the chunk plan it just computed actually
        pipelines. NUMPY_FALLBACK never pipelines (plan.n_buffers is always
        1), so at parameters where the LMM chunk-loop term dominates
        total_peak_gb (small n_samples keeps the O(n^2) eigendecomp and
        kinship terms negligible beside the Uab/Iab extra), the hardcoded 2
        must have inflated total_peak_gb above what a real n_buffers=1 run
        needs. Calls plan_memory directly, not a hand-built equivalent, so
        the hardcoded literal this regression is about is the thing under
        test.
        """
        from jamma.lmm import accel

        monkeypatch.setattr(memory, "available_ram_gb", lambda: 8.0)
        monkeypatch.setattr(accel, "_accel", None)  # force NUMPY_FALLBACK

        n_samples = 2_000
        n_snps = 300_000
        n_cvt = 4
        lmm_mode = 1

        dispatch = DispatchPath.NUMPY_FALLBACK
        plan = plan_lmm_chunks(n_samples, n_snps, n_cvt, dispatch)
        assert not plan.use_pipeline, "this case must not pipeline (plan.n_buffers=1)"
        assert plan.n_buffers == 1

        exec_plan = plan_association(
            n_samples,
            n_snps,
            requested="numpy-streaming",
            n_cvt=n_cvt,
            lmm_mode=lmm_mode,
        )
        mem_plan = exec_plan.price()

        # Reference: the same estimate built with pipeline_buffers hardcoded
        # to 2, the pre-fix behavior, to prove the two would have disagreed.
        chunk_plan = exec_plan.conservative_chunks
        uab_iab_gb = (
            chunk_plan.chunk_size
            * lmm_extra_bytes_per_snp(
                n_samples,
                n_cvt,
                exec_plan.dispatch,
                n_buffers=chunk_plan.n_buffers,
            )
            / 1e9
        )
        est_hardcoded_two = estimate_streaming_memory(
            n_samples,
            chunk_size=DEFAULT_STATS_CHUNK,
            n_cvt=n_cvt,
            pipeline_buffers=2,
            compute_chunk_size=chunk_plan.chunk_size,
            uab_iab_gb=uab_iab_gb,
        )

        assert mem_plan.total_peak_gb < est_hardcoded_two.total_peak_gb, (
            "plan_memory's real total must be below what pipeline_buffers=2 "
            "would have priced for a plan that does not pipeline"
        )
        allocated_gb = _engine_allocation_gb(
            n_samples, n_cvt, plan, dispatch, include_raw_block=True
        ) + _streaming_lmm_phase_non_buffer_terms_gb(n_samples, plan.chunk_size)
        assert mem_plan.total_peak_gb == pytest.approx(allocated_gb, rel=1e-9), (
            f"plan_memory total {mem_plan.total_peak_gb:.3f}GB != "
            f"engine allocation {allocated_gb:.3f}GB"
        )

    def test_run_lmm_association_numpy_threads_real_n_buffers(self, monkeypatch):
        """Call-site regression: run_lmm_association_numpy's check_memory gate
        must call estimate_lmm_memory with the plan's real n_buffers, not a
        default that silently reverts to 1.

        estimate_lmm_memory gaining an n_buffers parameter (with a default of
        1) does not by itself guarantee any caller passes the real value —
        the parameter could be dropped from a call site in a later edit and
        every existing test would still pass, because n_buffers=1 is a valid,
        merely wrong, default. This drives the real
        run_lmm_association_numpy(check_memory=True) entry point end to end
        and records the exact kwargs its internal estimate_lmm_memory call
        carries, rather than asserting on a GB total that a coincidentally
        matching wrong number could also produce.

        compute_chunk_size_numpy is forced to a constant so a small, fast
        synthetic dataset can still produce n_chunks >= _MIN_PIPELINE_CHUNKS
        (the real auto-scaling budget floor is 2GB per chunk, which no
        unit-test-sized matrix clears without this).
        """
        from jamma.lmm import association_plan, runner_numpy
        from jamma.lmm.schema import LmmConfig

        n_samples = 30
        n_snps = 400
        forced_chunk_size = 50  # n_snps / this == 8 == _MIN_PIPELINE_CHUNKS

        monkeypatch.setattr(
            "jamma.lmm.chunk_sizing.compute_chunk_size_numpy",
            lambda *args, **kwargs: forced_chunk_size,
        )

        rng = np.random.default_rng(0)
        genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
        phenotypes = rng.standard_normal(n_samples)
        kinship = np.corrcoef(genotypes) + np.eye(n_samples) * 0.1
        kinship = (kinship + kinship.T) / 2
        snp_info = [
            {"chr": "1", "rs": f"rs{i}", "pos": i, "a1": "A", "a0": "T"}
            for i in range(n_snps)
        ]

        recorded_calls: list[dict] = []
        real_estimate_lmm_memory = association_plan.estimate_lmm_memory

        def _recording_estimate_lmm_memory(*args, **kwargs):
            recorded_calls.append(kwargs)
            return real_estimate_lmm_memory(*args, **kwargs)

        monkeypatch.setattr(
            association_plan, "estimate_lmm_memory", _recording_estimate_lmm_memory
        )

        result = runner_numpy.run_lmm_association_numpy(
            genotypes=genotypes,
            phenotypes=phenotypes,
            kinship=kinship,
            snp_info=snp_info,
            config=LmmConfig(lmm_mode=1, check_memory=True, show_progress=False),
        )

        assert result.n_tested == n_snps
        assert len(recorded_calls) == 1, (
            f"expected exactly one estimate_lmm_memory call, got {len(recorded_calls)}"
        )
        call = recorded_calls[0]
        assert call["n_buffers"] == 2, (
            f"expected n_buffers=2 (the plan's real live-buffer count), "
            f"got {call.get('n_buffers')!r}"
        )
        assert call["lmm_batch_size"] == forced_chunk_size, (
            f"expected lmm_batch_size={forced_chunk_size} (the plan's chunk "
            f"size), got {call.get('lmm_batch_size')!r}"
        )


def test_plan_association_sizes_against_the_real_chunk(monkeypatch):
    """plan_association must price the chunk the run will allocate, not 20,000.

    At n=50000, snps=500000, ``estimate_lmm_memory``'s ``lmm_batch_size=20_000``
    default estimates 276.0GB; the chunk ``plan_lmm_chunks`` actually plans for
    this dispatch path allocates enough per-SNP state that the real estimate is
    500.0GB. A machine with 288GB available sits strictly between the two
    thresholds: the stale default says "fits" (batch), the real chunk says "does
    not fit" (streaming). At trunk, ``runner.py`` called ``estimate_lmm_memory``
    without ``lmm_batch_size``/``n_buffers`` and picked batch here; that flips
    the execution mode a machine near this line gets, in the direction that
    silently under-estimates memory.
    """
    use_fake_psutil(monkeypatch, available=288e9)

    plan = plan_association(50_000, 500_000, n_cvt=1, lmm_mode=1).summary

    assert plan.mode == "streaming", (
        f"expected streaming (the real chunk needs ~500GB > 288GB available), "
        f"got {plan.mode!r} ({plan.reason})"
    )


def test_plan_association_mem_budget_narrows_the_chunk(monkeypatch):
    """--mem-budget must narrow the chunk plan_association prices.

    A tight ``mem_budget`` should shrink the chunk plan feeds into the memory
    estimate, in turn shrinking the estimated total. At trunk, ``mem_budget``
    never reached the mode selector at all.
    """
    use_fake_psutil(monkeypatch, available=288e9)

    unbudgeted = plan_association(50_000, 500_000, n_cvt=1, lmm_mode=1).summary
    budgeted = plan_association(
        50_000, 500_000, n_cvt=1, lmm_mode=1, mem_budget=1.0
    ).summary

    # Unbudgeted: the real chunk needs ~500GB, exceeding 288GB -> streaming.
    assert unbudgeted.mode == "streaming"
    assert "500.0GB" in unbudgeted.reason
    # A 1GB chunk budget shrinks the chunk the estimate is priced against so
    # much the run fits comfortably -> batch. This proves mem_budget reached
    # the chunk sizer, rather than only vetoing the run afterward (its only
    # other reach, through memory_preflight's _reject_if_over_budget).
    assert budgeted.mode == "batch"
    assert "500.0GB" not in budgeted.reason


def test_plan_lmm_chunks_honors_mem_budget_bytes():
    """plan_lmm_chunks must narrow the chunk when given mem_budget_bytes.

    compute_chunk_size_numpy already accepted mem_budget_bytes, but
    plan_lmm_chunks (the single sizing decision the engine allocates from
    and the preflight prices from) had no parameter to pass it through, so
    every production caller's chunk plan ignored --mem-budget.
    """
    dispatch = DispatchPath.FUSED
    n_samples, n_snps, n_cvt = 50_000, 500_000, 1

    auto = plan_lmm_chunks(n_samples, n_snps, n_cvt, dispatch)
    budgeted = plan_lmm_chunks(
        n_samples, n_snps, n_cvt, dispatch, mem_budget_bytes=int(1e9)
    )

    assert budgeted.chunk_size < auto.chunk_size


def test_pipeline_memory_plan_honors_mem_budget(monkeypatch):
    """The pipeline preflight prices the budget-aware chunk geometry."""
    from jamma.lmm import accel

    monkeypatch.setattr(accel, "_accel", None)
    n_samples, n_snps, n_cvt = 30, 200, 1
    mem_budget = 12e-6
    expected = plan_lmm_chunks(
        n_samples,
        n_snps,
        n_cvt,
        DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(mem_budget * 1e9),
    )

    execution = plan_association(
        n_samples,
        n_snps,
        requested="numpy",
        n_cvt=n_cvt,
        mem_budget=mem_budget,
    )
    planned = execution.price()

    assert expected.chunk_size == 100
    assert planned.compute_chunk_size == expected.chunk_size


def test_chunk_engine_requests_budget_aware_geometry(monkeypatch):
    """The final chunk engine requests the width allowed by mem_budget."""
    from jamma.core.snp_stats import SnpSelection
    from jamma.lmm import accel
    from jamma.lmm.chunk_runner_numpy import run_lmm_chunk_source_numpy
    from jamma.lmm.genotype_source import PreparedGenotypes
    from jamma.lmm.prepare_common import PreparedLmmRun
    from jamma.lmm.schema import LmmConfig, SnpMeta

    monkeypatch.setattr(accel, "_accel", None)
    n_samples, n_snps, n_cvt = 30, 200, 1
    mem_budget = 12e-6
    expected = plan_lmm_chunks(
        n_samples,
        n_snps,
        n_cvt,
        DispatchPath.NUMPY_FALLBACK,
        mem_budget_bytes=int(mem_budget * 1e9),
    )
    prepared = PreparedLmmRun(
        eigenvalues=np.ones(n_samples),
        U=np.eye(n_samples),
        UtW=np.ones((n_samples, n_cvt)),
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
        chunk_factory=observe_geometry,
    )

    exec_plan = plan_association(
        n_samples,
        n_snps,
        requested="numpy",
        n_cvt=n_cvt,
        mem_budget=mem_budget,
    )
    with pytest.raises(GeometryObserved):
        run_lmm_chunk_source_numpy(
            genotypes=genotypes,
            chunk_sink=lambda _arrays, _start, _end: None,
            dispatch=exec_plan.dispatch,
            chunks=exec_plan.tighten_after_filter(n_snps),
            prepared=prepared,
            config=LmmConfig(
                lmm_mode=1,
                mem_budget=mem_budget,
                show_progress=False,
            ),
        )

    assert expected.chunk_size == 100
    assert requested == [expected.chunk_size]
