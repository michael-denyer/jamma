"""Behavior regressions for native workspace pricing and storage selection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jamma.core import memory
from jamma.core.snp_stats import SnpSelection
from jamma.lmm import association_plan
from jamma.lmm.association_plan import ExecutableAssociationPlan, ExecutionPlan
from jamma.lmm.chunk_sizing import LmmChunkPlan
from jamma.lmm.dispatch import DispatchPath
from jamma.lmm.genotype_source import PreparedGenotypes, SampleBasis
from jamma.lmm.prepare_common import PreparedCovariates
from jamma.lmm.runner_numpy import (
    LmmRunSpec,
    PreparedPhenotypeSpec,
    run_lmm_association_group_prepared,
)
from jamma.lmm.schema import DEFAULT_LMM_CONFIG, SnpMeta
from jamma.lmm.workspace import WorkspaceSpec
from tests.conftest import requires_c

pytestmark = pytest.mark.tier0


@requires_c
def test_general_mode4_prices_known_thread_workspace() -> None:
    spec = WorkspaceSpec.build(
        DispatchPath.FUSED_GENERAL, 4, 1_000, 1_000, 100, 50, 20, 18
    )

    # Review reproduction: these four arrays alone occupy 0.889484064 GB.
    idx = 5_253
    known = (18 * idx * 1_000 + 18 * 102 * 1_000 + 18 * 102 * idx + 5_151 * 1_000) * 8
    assert spec.fixed_bytes >= known


@requires_c
def test_workspace_grid_resolution_changes_quote() -> None:
    low = WorkspaceSpec.build(DispatchPath.FUSED_GENERAL, 1, 2_000, 2_000, 4, 20, 20, 2)
    high = WorkspaceSpec.build(
        DispatchPath.FUSED_GENERAL, 1, 2_000, 2_000, 4, 80, 20, 2
    )

    assert high.fixed_bytes > low.fixed_bytes


def test_fallback_grid_prices_pab_and_tensordot_shapes() -> None:
    n_samples, n_cvt, n_grid = 1_000, 100, 80
    spec = WorkspaceSpec.build(
        DispatchPath.NUMPY_FALLBACK,
        4,
        n_samples,
        n_samples,
        n_cvt,
        n_grid,
        20,
        1,
    )
    idx = 5_253
    rows = 102
    known_per_snp = n_grid * (rows * idx + idx) * 8
    known_fixed = 2 * n_grid * n_samples * 8

    assert spec.fixed_bytes >= known_fixed
    assert spec.bytes_per_snp >= known_per_snp


def test_fallback_custom_grid_reduces_chunk_width(monkeypatch) -> None:
    monkeypatch.setattr(association_plan.accel, "available", lambda: False)
    monkeypatch.setattr(association_plan.memory, "available_ram_gb", lambda: 64.0)
    monkeypatch.setattr(association_plan, "is_blas_controllable", lambda: True)

    low = association_plan.plan_association(2_000, 100_000, n_cvt=8, n_grid=20)
    high = association_plan.plan_association(2_000, 100_000, n_cvt=8, n_grid=200)

    assert high.conservative_chunks.chunk_size < low.conservative_chunks.chunk_size


def test_phenotype_group_is_bounded_by_live_native_workspaces(monkeypatch) -> None:
    """Planner keeps only the largest phenotype group whose kernels fit."""
    monkeypatch.setattr(
        association_plan.memory, "fits", lambda need, have: need <= have
    )
    workspace = WorkspaceSpec(
        dispatch=DispatchPath.FUSED,
        lmm_mode=1,
        n_samples=10,
        n_input_samples=10,
        n_cvt=1,
        n_grid=50,
        n_refine=20,
        max_threads=1,
        persistent_bytes=1_000_000_000,
        per_thread_bytes=0,
        transient_per_thread_bytes=0,
        bytes_per_snp=40,
    )
    plan = ExecutableAssociationPlan(
        summary=ExecutionPlan("batch", "group pricing test"),
        dispatch=DispatchPath.FUSED,
        conservative_chunks=LmmChunkPlan(1, 1, 1, False),
        n_samples=10,
        n_input_samples=10,
        n_snps_before_filter=1,
        n_cvt=1,
        mem_budget_gb=None,
        workspace=workspace,
    )

    grouped = association_plan._select_phenotype_group(
        plan, n_phenotypes=4, available_gb=2.5
    )

    assert grouped.phenotype_group_size == 2
    assert grouped.price().total_peak_gb <= 2.5
    assert replace(grouped, phenotype_group_size=3).price().total_peak_gb > 2.5


def test_phenotype_group_preserves_single_phenotype_chunk_width(monkeypatch) -> None:
    """Grouping yields before it shrinks the already feasible chunk width."""
    monkeypatch.setattr(
        association_plan.memory, "fits", lambda need, have: need <= have
    )
    workspace = WorkspaceSpec(
        dispatch=DispatchPath.FUSED,
        lmm_mode=1,
        n_samples=10,
        n_input_samples=10,
        n_cvt=1,
        n_grid=50,
        n_refine=20,
        max_threads=1,
        persistent_bytes=60_000_000,
        per_thread_bytes=0,
        transient_per_thread_bytes=0,
        bytes_per_snp=1_000_000,
    )
    plan = ExecutableAssociationPlan(
        summary=ExecutionPlan("batch", "group geometry test"),
        dispatch=DispatchPath.FUSED,
        conservative_chunks=LmmChunkPlan(100, 1, 1, False),
        n_samples=10,
        n_input_samples=10,
        n_snps_before_filter=100,
        n_cvt=1,
        mem_budget_gb=None,
        workspace=workspace,
    )
    group_two = replace(plan, phenotype_group_size=2)
    group_three = replace(plan, phenotype_group_size=3)
    available_gb = (
        group_two.price().total_peak_gb + group_three.price().total_peak_gb
    ) / 2

    assert (
        replace(
            group_three,
            conservative_chunks=group_three.conservative_chunks.cap_width(100, 1),
        )
        .price()
        .total_peak_gb
        < available_gb
    )
    grouped = association_plan._select_phenotype_group(
        plan, n_phenotypes=3, available_gb=available_gb
    )
    assert grouped.phenotype_group_size == 2
    assert grouped.conservative_chunks.chunk_size == 100


def test_fallback_group_reuses_sequential_compute_scratch(monkeypatch) -> None:
    """Fallback grouping does not multiply scratch used one job at a time."""
    monkeypatch.setattr(association_plan.accel, "available", lambda: False)
    monkeypatch.setattr(association_plan.memory, "available_ram_gb", lambda: 4.0)
    monkeypatch.setattr(association_plan, "is_blas_controllable", lambda: True)

    single = association_plan.plan_association(
        1_000, 10_000, n_cvt=4, lmm_mode=4, n_phenotypes=1
    )
    grouped = association_plan.plan_association(
        1_000, 10_000, n_cvt=4, lmm_mode=4, n_phenotypes=3
    )

    assert grouped.phenotype_group_size == 3
    added = grouped.price().total_peak_gb - single.price().total_peak_gb
    assert added < 3 * grouped.workspace.fixed_bytes / 1e9


def test_grouped_runner_rejects_more_jobs_than_priced(monkeypatch, tmp_path) -> None:
    """The allocation boundary enforces the planner's live-kernel limit."""
    monkeypatch.setattr(association_plan.accel, "available", lambda: False)
    monkeypatch.setattr(association_plan.memory, "available_ram_gb", lambda: 4.0)
    monkeypatch.setattr(association_plan, "is_blas_controllable", lambda: True)
    execution = association_plan.plan_association(4, 1, n_phenotypes=1)
    empty_int = np.array([], dtype=np.intp)
    empty_float = np.array([], dtype=np.float64)
    empty_str = np.array([], dtype=str)
    genotypes = PreparedGenotypes(
        snp_meta=SnpMeta(empty_str, empty_str, empty_int, empty_str, empty_str),
        selection=SnpSelection(
            empty_int,
            empty_int,
            np.array([], dtype=bool),
            empty_float,
            empty_int,
            empty_float,
        ),
        n_unexpected=0,
        analyzed_sample_count=4,
        sample_basis=SampleBasis(np.arange(4), 4),
        chunk_factory=lambda _size: iter(()),
    )
    runs = tuple(
        PreparedPhenotypeSpec(np.arange(4, dtype=float), Path(tmp_path / f"{i}.txt"))
        for i in range(2)
    )
    covariates = PreparedCovariates(np.ones((4, 1)), 1, np.ones((4, 1)))

    with pytest.raises(ValueError, match=r"exceeds.*priced capacity"):
        run_lmm_association_group_prepared(
            genotypes,
            LmmRunSpec(config=DEFAULT_LMM_CONFIG, execution=execution),
            runs,
            eigenvalues=np.ones(4),
            eigenvectors=np.eye(4),
            prepared_covariates=covariates,
        )


def test_auto_uses_user_budget_and_allows_streaming_fallback(monkeypatch) -> None:
    monkeypatch.setattr(association_plan.accel, "available", lambda: False)
    monkeypatch.setattr(association_plan.memory, "available_ram_gb", lambda: 64.0)
    monkeypatch.setattr(association_plan, "is_blas_controllable", lambda: True)

    plan = association_plan.plan_association(
        50_000, 500_000, requested="auto", mem_budget=4.0
    )

    assert plan.summary.mode == "streaming"
    assert plan.dispatch is DispatchPath.NUMPY_FALLBACK


def test_streaming_chunk_converges_on_full_quote_under_physical_ram(
    monkeypatch,
) -> None:
    monkeypatch.setattr(association_plan.accel, "available", lambda: False)
    monkeypatch.setattr(association_plan.memory, "available_ram_gb", lambda: 8.0)
    monkeypatch.setattr(association_plan, "is_blas_controllable", lambda: True)

    plan = association_plan.plan_association(
        2_000, 1_000_000, n_cvt=4, mem_budget=1_000.0
    )
    quote = plan.price()

    assert plan.summary.mode == "streaming"
    assert plan.conservative_chunks.chunk_size < 200_000
    assert (
        plan.conservative_chunks.n_chunks
        == (1_000_000 + plan.conservative_chunks.chunk_size - 1)
        // plan.conservative_chunks.chunk_size
    )
    memory.require(
        quote.total_peak_gb,
        8.0,
        "planned fallback association",
        budget_gb=plan.mem_budget_gb,
    )
    wider_size = plan.conservative_chunks.chunk_size + 1
    wider_chunks = replace(
        plan.conservative_chunks,
        chunk_size=wider_size,
        n_chunks=(1_000_000 + wider_size - 1) // wider_size,
    )
    wider_quote = replace(plan, conservative_chunks=wider_chunks).price()
    with pytest.raises(MemoryError):
        memory.require(
            wider_quote.total_peak_gb,
            8.0,
            "one-SNP-wider fallback association",
            budget_gb=plan.mem_budget_gb,
        )


def test_streaming_chunk_converges_on_user_budget_below_ram(monkeypatch) -> None:
    monkeypatch.setattr(association_plan.accel, "available", lambda: False)
    monkeypatch.setattr(association_plan.memory, "available_ram_gb", lambda: 64.0)
    monkeypatch.setattr(association_plan, "is_blas_controllable", lambda: True)

    plan = association_plan.plan_association(
        2_000,
        1_000_000,
        requested="numpy-streaming",
        n_cvt=4,
        mem_budget=3.0,
    )
    quote = plan.price()

    assert plan.conservative_chunks.chunk_size > 1
    assert quote.total_peak_gb <= 3.0
    memory.require(quote.total_peak_gb, 64.0, budget_gb=3.0)


@requires_c
def test_workspace_thread_capacity_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(association_plan.accel, "available", lambda: True)
    monkeypatch.setattr(association_plan.accel, "HAS_OPENMP", True)
    monkeypatch.setattr(association_plan.memory, "available_ram_gb", lambda: 64.0)
    monkeypatch.setattr(association_plan, "is_blas_controllable", lambda: True)
    monkeypatch.setattr(
        association_plan, "get_c_extension_thread_count", lambda *_args: 7
    )

    plan = association_plan.plan_association(1_000, 2_000, n_cvt=3, lmm_mode=4)

    assert plan.workspace.max_threads == 7


@requires_c
def test_native_sizing_query_is_the_workspace_source() -> None:
    from jamma.lmm import accel

    native = accel.require().workspace_sizes_c(1_000, 100, 50, 4, 18)
    spec = WorkspaceSpec.build(
        DispatchPath.FUSED_GENERAL, 4, 1_000, 1_000, 100, 50, 20, 18
    )

    assert native == (
        spec.persistent_bytes,
        spec.per_thread_bytes,
        spec.transient_per_thread_bytes,
        spec.bytes_per_snp,
    )


@requires_c
def test_native_sizing_counts_retained_run_invariants() -> None:
    """The query includes Python arrays live beside each native workspace."""
    from jamma.lmm import accel

    n_samples = 100_000
    ncvt1_persistent, *_ = accel.require().workspace_sizes_c(n_samples, 1, 50, 1, 1)
    # eigenvalues, UtW, Uty, Hi_eval_null, w, and three invariant Uab rows.
    assert ncvt1_persistent >= 8 * n_samples * 8

    n_cvt = 100
    general_persistent, *_ = accel.require().workspace_sizes_c(
        n_samples, n_cvt, 50, 1, 1
    )
    rows = n_cvt + 2
    index = (n_cvt + 3) * rows // 2
    invariant_columns = index - rows
    # Original+owned eigenvalues, original+transposed UtW, Uty, Hi_eval_null,
    # and the retained invariant SoA.
    retained_doubles = (4 + 2 * n_cvt + invariant_columns) * n_samples
    assert general_persistent >= retained_doubles * 8


@requires_c
def test_native_sizing_query_counts_max_covariate_transport_and_rejects_beyond() -> (
    None
):
    from jamma.lmm import accel

    persistent, *_ = accel.require().workspace_sizes_c(1_000, 100, 50, 4, 18)
    entry_count = 101 * 102 * 103 // 6
    simultaneous_entry_copies = 2 * entry_count * 16
    assert persistent > simultaneous_entry_copies

    with pytest.raises(ValueError, match="invalid workspace sizing dimensions"):
        accel.require().workspace_sizes_c(1_000, 101, 50, 4, 18)
