"""LOCO worker tests: the worker planner and the concurrent eigen generator.

Related LOCO test files:
- test_loco_numpy.py: NumPy LOCO parity and the pass planner
- test_loco_eigen_cache.py: LOCO eigen cache write/read round-trip
"""

from __future__ import annotations

import dataclasses
import signal
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future
from unittest.mock import patch

import numpy as np
import pytest

from jamma.core.threading import get_physical_core_count
from jamma.genotype.dataset import GenotypeDataset
from jamma.io import read_fam_phenotypes
from jamma.lmm import loco_workers
from jamma.lmm.assoc_output import AssocResult
from jamma.lmm.loco import LocoConfig, run_lmm_loco
from jamma.lmm.loco_eigen import _computed_eigen_pairs
from jamma.lmm.loco_workers import plan_loco_workers, solve_eigen_pairs
from jamma.lmm.schema import LmmConfig
from tests.fixture_paths import LOCO
from tests.support import require_fixture


@pytest.mark.tier0
def test_worker_budget_charges_each_owned_input_once():
    from jamma.kinship.loco import loco_retained_set
    from jamma.lmm.eigen_plan import plan_eigen_driver

    retained = loco_retained_set(10_000, 10_000, 10_000)
    eigen = plan_eigen_driver(
        10_000,
        100,
        has_dsyevd=True,
        has_dsyevr=True,
        forced_numpy=False,
        inplace_blocker=None,
    )
    plan = plan_loco_workers(
        2,
        n_chr=3,
        retained=retained,
        eigen_plan=eigen,
        available_gb=100,
        budget_gb=9,
        association_gb=0.8,
        cores=18,
    )

    assert plan.workers == 2
    # Two DSYEVD-inplace peaks of 2.4 GB, each already holding its 0.8 GB input.
    assert plan.consumer_gb == pytest.approx(2 * 2.400880032)


@pytest.mark.tier0
@pytest.mark.parametrize(
    "requested,n_chr,cores,available,budget,association,expected",
    [
        # The retained set is 3.2 GB and each in-flight DSYEVD-inplace solve
        # 2.4 GB, so W workers peak at 3.2 + 2.4 * W when association (0.8)
        # is under one driver peak: 5.6, 8.0, 10.4, ... A 5.0 GB association
        # replaces one driver peak: 3.2 + 2.4 * (W - 1) + 5.0.
        (6, 22, 8, 8.8, None, 0.8, 1),  # 8.0 + 0.8 margin is not under 8.8
        (1, 22, 8, 100, None, 0.8, 1),  # requested one stays one
        (6, 3, 8, 100, None, 0.8, 3),  # chromosome cap
        (6, 22, 4, 100, None, 0.8, 4),  # core cap
        (6, 22, 8, 100, 9, 0.8, 2),  # 8.0 within a 9 GB budget, 10.4 is not
        (6, 22, 8, 1, None, 0.8, 1),  # nothing fits; the floor is one
        (6, 22, 8, 100, 9, 5.0, 1),  # 3.2 + 2.4 + 5.0 = 10.6 over budget
        (6, 22, 8, 100, 9, 6.0, 1),
        (10**12, 3, 8, 100, 9, 0.8, 2),  # absurd requests do not hang
    ],
)
def test_worker_plan_respects_memory_and_execution_caps(
    requested,
    n_chr,
    cores,
    available,
    budget,
    association,
    expected,
):
    from jamma.kinship.loco import loco_retained_set
    from jamma.lmm.eigen_plan import plan_eigen_driver

    retained = loco_retained_set(10_000, 10_000, 10_000)
    eigen = plan_eigen_driver(
        10_000,
        100,
        has_dsyevd=True,
        has_dsyevr=True,
        forced_numpy=False,
        inplace_blocker=None,
    )
    plan = plan_loco_workers(
        requested,
        n_chr=n_chr,
        cores=cores,
        retained=retained,
        eigen_plan=eigen,
        available_gb=available,
        budget_gb=budget,
        association_gb=association,
    )
    assert plan.workers == expected
    assert plan.consumer_gb >= association


# ---------------------------------------------------------------------------
# The concurrent generator, driven through its ``solve`` parameter
# ---------------------------------------------------------------------------

# Chromosome name -> matrix order. The order is the only thing a ``solve``
# stand-in can see of the chromosome it was handed, so it doubles as its tag.
_ORDER_BY_CHR = {"1": 8, "2": 9, "3": 10}


def _synthetic_stream() -> Iterator[tuple[str, np.ndarray]]:
    """One symmetric matrix per chromosome, each a distinct order."""
    rng = np.random.default_rng(0)
    for chr_name, n in _ORDER_BY_CHR.items():
        A = rng.standard_normal((n, n))
        yield chr_name, np.ascontiguousarray(A @ A.T / n)


def _computed_pairs(*, workers: int, solve):
    from jamma.lmm.eigen import plan_eigen_driver_for_machine

    eigen_plan = plan_eigen_driver_for_machine(
        8, 100, budget_gb=None, inplace_blocker=None
    )
    return _computed_eigen_pairs(
        _synthetic_stream(),
        {c: np.arange(5) for c in _ORDER_BY_CHR},
        subset_rows=None,
        config=LmmConfig(check_memory=False, show_progress=False),
        loco=LocoConfig(),
        cache_write=None,
        eigen_plan=eigen_plan,
        workers=workers,
        solve=solve,
    )


@pytest.mark.tier0
def test_computed_eigen_pairs_overlaps_workers_and_keeps_chromosome_order(monkeypatch):
    """Three workers solve three chromosomes at once; pairs still come out in order.

    Every solve waits at a three-party barrier, so the test only passes when
    all three run at the same time: sequential solves would each time out
    there. Chromosome 1 then waits for chromosome 3's future to finish, so its pair
    coming out first proves the generator resolves futures in chromosome
    order rather than completion order.
    """
    barrier = threading.Barrier(3)
    chr3_done = threading.Event()
    idents: list[int] = []

    class TrackedFuture(Future):
        def set_result(self, result) -> None:
            super().set_result(result)
            # Signaling inside solve would release chromosome 1 before this
            # future finishes, leaving the completion order scheduler-dependent.
            if result[0] == "3":
                chr3_done.set()

    monkeypatch.setattr(loco_workers, "Future", TrackedFuture)

    def solve(K: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        idents.append(threading.get_ident())
        barrier.wait(timeout=10)
        n = len(K)
        if n == 8:
            assert chr3_done.wait(10)
        return np.full(n, float(n)), np.eye(n)

    pairs = list(_computed_pairs(workers=3, solve=solve))

    assert [chr_name for chr_name, _, _ in pairs] == ["1", "2", "3"]
    assert [len(eigenvalues) for _, eigenvalues, _ in pairs] == [8, 9, 10]
    assert len(set(idents)) == 3
    assert threading.get_ident() not in idents


@pytest.mark.tier0
def test_computed_eigen_pairs_propagates_worker_failure_and_shuts_down():
    """A failing solve surfaces in chromosome order and the pool is gone after."""

    def solve(K: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        n = len(K)
        if n == _ORDER_BY_CHR["2"]:
            raise RuntimeError("chromosome 2 failed")
        return np.zeros(n), np.eye(n)

    pairs = _computed_pairs(workers=3, solve=solve)

    chr_name, _, _ = next(pairs)
    assert chr_name == "1"
    assert _eigen_worker_threads()
    with pytest.raises(RuntimeError, match="chromosome 2 failed"):
        next(pairs)

    assert not _eigen_worker_threads()
    pairs.close()


def _eigen_worker_threads() -> list[threading.Thread]:
    """Live LOCO eigen worker threads, by the name solve_eigen_pairs gives them."""
    return [t for t in threading.enumerate() if t.name.startswith("loco-eigen-")]


@pytest.mark.tier0
def test_keyboard_interrupt_propagates_before_the_in_flight_solve_finishes():
    """Ctrl-C during a solve unwinds at once instead of joining the worker.

    GEMMA dies on SIGINT mid-DSYEVD; JAMMA did the same before #359 because the
    solve ran on a daemon thread. The solve here blocks on ``release`` and a
    helper thread sends SIGINT to the main thread once the solve has started.
    The interrupt must propagate while ``release`` is still clear; only then is
    the solve released, and every thread the test started is joined so nothing
    outlives it to disturb the thread accounting of later tests.
    """
    assert threading.current_thread() is threading.main_thread()
    main_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()

    def solve(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        entered.set()
        assert release.wait(30)
        return np.zeros(len(K)), np.eye(len(K))

    def interrupt() -> None:
        assert entered.wait(10)
        time.sleep(0.05)
        signal.pthread_kill(main_thread, signal.SIGINT)

    pairs = solve_eigen_pairs(iter([("1", np.eye(4))]), solve, workers=1, n_threads=1)
    helper = threading.Thread(target=interrupt, daemon=True)
    helper.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            next(pairs)
        assert not release.is_set(), "interrupt waited for the in-flight solve"
    finally:
        release.set()
        helper.join(10)
        for worker in _eigen_worker_threads():
            worker.join(10)
    assert not helper.is_alive()
    assert not _eigen_worker_threads()


# ---------------------------------------------------------------------------
# run_lmm_loco wiring
# ---------------------------------------------------------------------------


def _loco_run(workers: int, monkeypatch):
    monkeypatch.setenv("JAMMA_LOCO_WORKERS", str(workers))
    return run_lmm_loco(
        dataset=GenotypeDataset.open_plink(LOCO.bfile),
        phenotypes=read_fam_phenotypes(LOCO.fam),
        config=LmmConfig(check_memory=False, show_progress=False),
    )


@pytest.mark.tier1
def test_run_lmm_loco_logs_planned_workers(monkeypatch):
    """JAMMA_LOCO_WORKERS=4 on the 3-chromosome fixture logs the planned count.

    The plan caps at the chromosome count and the physical cores; the fixture
    is far too small for memory to bind.
    """
    require_fixture(LOCO.bed, LOCO.fam)
    import jamma.lmm.loco as loco_module

    logged: list[str] = []
    original_info = loco_module.logger.info

    def capture_info(msg, *args, **kwargs):
        logged.append(str(msg))
        return original_info(msg, *args, **kwargs)

    # loguru does not integrate with pytest caplog; capture via the logger sink
    with patch.object(loco_module.logger, "info", side_effect=capture_info):
        result = _loco_run(4, monkeypatch)

    expected = min(3, get_physical_core_count())
    line = next(m for m in logged if m.startswith("LOCO workers:"))
    assert line.startswith(f"LOCO workers: {expected} (requested 4; 3 chromosomes; ")
    assert result.n_tested > 0


@pytest.mark.tier1
def test_run_lmm_loco_workers_match_sequential_to_rounding(monkeypatch):
    """Three workers and one produce the same associations, field by field.

    Floats are compared to 1e-8 relative, not bit for bit. Each solve keeps
    its BLAS thread count, but OpenBLAS and MKL split a solve across one
    shared pool, and concurrent callers change where that split lands, so
    reduction order and the last bits move: run 34404442767 on the
    OpenBLAS-ILP64 leg differed in ``beta`` where the PR run had not. On
    Accelerate each solve is single-threaded and the arrays are identical.
    """
    require_fixture(LOCO.bed, LOCO.fam)

    sequential = _loco_run(1, monkeypatch)
    concurrent = _loco_run(3, monkeypatch)

    assert sequential.n_tested == concurrent.n_tested > 0
    assert sequential.pve == pytest.approx(concurrent.pve, rel=1e-8)
    assert sequential.pve_se == pytest.approx(concurrent.pve_se, rel=1e-8)
    for field in dataclasses.fields(AssocResult):
        one = [getattr(r, field.name) for r in sequential.associations]
        three = [getattr(r, field.name) for r in concurrent.associations]
        if all(v is None or isinstance(v, float) for v in one):
            as_float = lambda vs: np.array(  # noqa: E731
                [np.nan if v is None else v for v in vs], dtype=float
            )
            np.testing.assert_allclose(
                as_float(one), as_float(three), rtol=1e-8, atol=0, err_msg=field.name
            )
        else:
            assert one == three, field.name
