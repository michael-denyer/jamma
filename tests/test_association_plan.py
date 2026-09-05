"""Focused contracts for executable association planning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from jamma.lmm.association_plan import (
    ExecutableAssociationPlan,
    ExecutionPlan,
)
from jamma.lmm.chunk_sizing import LmmChunkPlan
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.workspace import WorkspaceSpec

pytestmark = pytest.mark.tier0


def _workspace(dispatch: DispatchPath, n_cvt: int) -> WorkspaceSpec:
    return WorkspaceSpec.build(dispatch, 1, 1_000, 1_000, n_cvt, 50, 20, 1)


def test_plan_is_frozen_and_tightening_returns_a_chunk_plan() -> None:
    conservative = LmmChunkPlan(100, 10, 2, True)
    plan = ExecutableAssociationPlan(
        summary=ExecutionPlan("batch", "test"),
        dispatch=DispatchPath.FUSED,
        conservative_chunks=conservative,
        n_samples=1_000,
        n_input_samples=1_000,
        n_snps_before_filter=1_000,
        n_cvt=1,
        mem_budget_gb=None,
        workspace=_workspace(DispatchPath.FUSED, 1),
    )
    tightened = plan.conservative_chunks.narrow(500)

    assert isinstance(tightened, LmmChunkPlan)
    with pytest.raises(FrozenInstanceError):
        plan.n_samples = 2_000  # type: ignore[misc]


def test_tightening_only_decreases_width_and_preserves_policy() -> None:
    plan = ExecutableAssociationPlan(
        summary=ExecutionPlan("streaming", "test"),
        dispatch=DispatchPath.FUSED_GENERAL,
        conservative_chunks=LmmChunkPlan(100, 10, 2, True),
        n_samples=1_000,
        n_input_samples=1_000,
        n_snps_before_filter=1_000,
        n_cvt=2,
        mem_budget_gb=8.0,
        workspace=_workspace(DispatchPath.FUSED_GENERAL, 2),
    )

    tightened = plan.conservative_chunks.narrow(250)

    # Tightening narrows geometry only; mode and dispatch stay on the plan.
    assert plan.summary.mode == "streaming"
    assert plan.dispatch is DispatchPath.FUSED_GENERAL
    assert tightened == LmmChunkPlan(100, 3, 1, False)


def test_tightening_never_turns_pipeline_on() -> None:
    conservative = LmmChunkPlan(100, 20, 1, False)

    tightened = conservative.narrow(1_500)

    assert tightened.chunk_size == 100
    assert tightened.n_chunks == 15
    assert not tightened.use_pipeline
    assert tightened.n_buffers == 1


def test_tightening_empty_plan_keeps_positive_width() -> None:
    tightened = LmmChunkPlan(100, 10, 2, True).narrow(0)

    assert tightened == LmmChunkPlan(1, 0, 1, False)
