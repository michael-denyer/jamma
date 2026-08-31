"""Focused contracts for executable association planning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from jamma.lmm.association_plan import (
    ExecutableAssociationPlan,
    ExecutionPlan,
)
from jamma.lmm.chunk_sizing import LmmChunkPlan, tighten_lmm_chunks
from jamma.lmm.dispatch import DispatchPath

pytestmark = pytest.mark.tier0


def test_plan_is_frozen_and_tightening_returns_a_chunk_plan() -> None:
    conservative = LmmChunkPlan(100, 10, 2, True)
    plan = ExecutableAssociationPlan(
        summary=ExecutionPlan("batch", "test"),
        dispatch=DispatchPath.FUSED,
        conservative_chunks=conservative,
        n_samples=1_000,
        n_snps_before_filter=1_000,
        n_cvt=1,
        mem_budget_gb=None,
    )
    tightened = plan.tighten_after_filter(500)

    assert isinstance(tightened, LmmChunkPlan)
    with pytest.raises(FrozenInstanceError):
        plan.n_samples = 2_000  # type: ignore[misc]


def test_tightening_only_decreases_width_and_preserves_policy() -> None:
    plan = ExecutableAssociationPlan(
        summary=ExecutionPlan("streaming", "test"),
        dispatch=DispatchPath.FUSED_GENERAL,
        conservative_chunks=LmmChunkPlan(100, 10, 2, True),
        n_samples=1_000,
        n_snps_before_filter=1_000,
        n_cvt=2,
        mem_budget_gb=8.0,
    )

    tightened = plan.tighten_after_filter(250)

    # Tightening narrows geometry only; mode and dispatch stay on the plan.
    assert plan.summary.mode == "streaming"
    assert plan.dispatch is DispatchPath.FUSED_GENERAL
    assert tightened == LmmChunkPlan(100, 3, 1, False)


def test_tightening_never_turns_pipeline_on() -> None:
    conservative = LmmChunkPlan(100, 20, 1, False)

    tightened = tighten_lmm_chunks(conservative, 1_500)

    assert tightened.chunk_size == 100
    assert tightened.n_chunks == 15
    assert not tightened.use_pipeline
    assert tightened.n_buffers == 1


def test_tightening_empty_plan_keeps_positive_width() -> None:
    tightened = tighten_lmm_chunks(LmmChunkPlan(100, 10, 2, True), 0)

    assert tightened == LmmChunkPlan(1, 0, 1, False)
