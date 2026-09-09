"""Allocation and process-wide BLAS lifetimes of real LOCO solve batches."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterator

import numpy as np
import pytest

from jamma.core import memory
from jamma.core import threading as core_threading
from jamma.core.eigen_plan import plan_eigen_driver
from jamma.kinship.loco import loco_retained_set
from jamma.lmm.eigen import _eigendecompose_kinship, center_kinship
from jamma.lmm.loco_config import LocoConfig
from jamma.lmm.loco_eigen import _computed_eigen_pairs
from jamma.lmm.loco_workers import plan_loco_workers, solve_eigen_pairs
from tests.fakes.blas import fake_blas_controller

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize(
    "exit_kind", ["complete", "close", "solve_error", "input_error"]
)
def test_batch_restores_blas_before_consumer_or_exception(monkeypatch, exit_kind):
    active = [11]
    transitions: list[tuple[str, int]] = []
    controller = fake_blas_controller(active, transitions)
    monkeypatch.setattr(core_threading, "is_blas_controllable", lambda: True)
    monkeypatch.setattr(
        core_threading,
        "threadpool_limits",
        lambda *, limits, user_api: controller(limits),
    )
    observed: list[int] = []
    completed: list[int] = []

    def inputs() -> Iterator[tuple[str, np.ndarray]]:
        yield "1", np.eye(8)
        if exit_kind == "input_error":
            raise RuntimeError("input failed")
        yield "2", np.ones((8, 7)) if exit_kind == "solve_error" else np.eye(8)

    def solve(K):
        observed.append(active[0])
        result = _eigendecompose_kinship(
            K, n_threads=2, check_memory=False, show_progress=False
        )
        completed.append(active[0])
        return result

    pairs = solve_eigen_pairs(inputs(), solve, workers=2, n_threads=2)
    try:
        if exit_kind == "input_error":
            with pytest.raises(RuntimeError, match="input failed"):
                next(pairs)
        else:
            first = next(pairs)
            assert first[0] == "1"
            assert active == [11]
            if exit_kind == "solve_error":
                with pytest.raises(ValueError, match="must be square"):
                    next(pairs)
            elif exit_kind == "complete":
                assert [name for name, _, _ in pairs] == ["2"]
    finally:
        pairs.close()
    assert active == [11]
    if exit_kind == "input_error":
        assert observed == []
        assert transitions == []
        return
    assert observed
    assert set(observed) == {2}
    assert set(completed) <= {2}
    assert transitions[0] == ("enter", 2)
    assert transitions[-1] == ("restore", 11)


def test_reused_stream_buffer_is_copied_and_batches_stay_bounded(monkeypatch):
    monkeypatch.setenv("JLINALG_NO_VENDOR_LAPACK", "1")
    names = [str(i) for i in range(7)]
    rng = np.random.default_rng(38)
    matrices = []
    for _ in names:
        A = rng.normal(size=(16, 16))
        matrices.append(np.ascontiguousarray(A @ A.T))
    pulled: list[str] = []
    inputs_alive: list[weakref.ReferenceType[np.ndarray]] = []
    barrier = threading.Barrier(3)
    calls = 0
    lock = threading.Lock()

    def stream():
        buffer = np.empty((16, 16))
        for name, matrix in zip(names, matrices, strict=True):
            buffer[:] = matrix
            pulled.append(name)
            yield name, buffer

    def solve(K, **kwargs):
        nonlocal calls
        inputs_alive.append(weakref.ref(K))
        with lock:
            calls += 1
            index = calls
        if index <= 6:
            barrier.wait(timeout=5)
        return _eigendecompose_kinship(K, **kwargs)

    # The NumPy driver has a separate output, so no worker input may survive
    # into association once the batch has completed.
    plan = plan_eigen_driver(
        16,
        100,
        has_dsyevd=False,
        has_dsyevr=False,
        no_vendor=True,
        inplace_eligible=False,
    )
    pairs = _computed_eigen_pairs(
        stream(),
        names,
        valid_mask=np.ones(16, dtype=bool),
        n_valid=16,
        pre_subset=True,
        all_samples_valid=True,
        partitions={name: np.arange(1) for name in names},
        check_memory=False,
        show_progress=False,
        loco=LocoConfig(),
        cache_write=None,
        eigen_plan=plan,
        mem_budget=None,
        workers=3,
        solve=solve,
    )
    try:
        for i, (name, values, U) in enumerate(pairs):
            assert name == names[i]
            assert len(pulled) == min(3 * (i // 3 + 1), len(names))
            expected = matrices[i].copy()
            center_kinship(expected)
            np.testing.assert_allclose(expected @ U, U * values, atol=1e-10)
            assert all(ref() is None for ref in inputs_alive)
    finally:
        pairs.close()


@pytest.mark.parametrize(
    "has_dsyevd,has_dsyevr,no_vendor,inplace",
    [
        (True, True, False, True),
        (True, True, False, False),
        (False, True, False, False),
        (False, False, True, False),
    ],
)
def test_worker_accounting_uses_driver_peak_and_analysed_output_shape(
    has_dsyevd,
    has_dsyevr,
    no_vendor,
    inplace,
):
    retained = loco_retained_set(10_000, 10_000, 10_000)
    eigen = plan_eigen_driver(
        1000,
        100,
        has_dsyevd=has_dsyevd,
        has_dsyevr=has_dsyevr,
        no_vendor=no_vendor,
        inplace_eligible=inplace,
    )
    plan = plan_loco_workers(
        3,
        n_chr=3,
        n_samples=1000,
        retained=retained,
        eigen_plan=eigen,
        available_gb=100,
        budget_gb=4.3,
        association_gb=1.0,
        cores=8,
    )
    assert plan.workers == 3
    # The other two completed eigenvectors are 0.008 GB each, not the
    # full-output stream's 0.8 GB matrices or their driver workspaces.
    assert plan.consumer_gb == pytest.approx(1.016)


def test_worker_plan_preserves_strict_ram_tie_and_inclusive_user_budget():
    retained = loco_retained_set(1000, 1000, 1000)
    eigen = plan_eigen_driver(
        1000,
        100,
        has_dsyevd=True,
        has_dsyevr=True,
        no_vendor=False,
        inplace_eligible=True,
    )
    peak = retained.while_consuming_gb + 2 * eigen.required_gb

    def planned(available_gb, budget_gb):
        return plan_loco_workers(
            2,
            n_chr=3,
            n_samples=1000,
            retained=retained,
            eigen_plan=eigen,
            association_gb=0.0,
            cores=8,
            available_gb=available_gb,
            budget_gb=budget_gb,
        )

    assert planned(peak + memory.margin_gb(peak), None).workers == 1
    assert planned(100, peak).workers == 2
