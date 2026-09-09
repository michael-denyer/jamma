"""LOCO worker tests: the worker planner and the concurrent eigen generator.

Related LOCO test files:
- test_loco_numpy.py: NumPy LOCO parity and the pass planner
- test_loco_eigen_cache.py: LOCO eigen cache write/read round-trip
"""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pytest

from jamma.core.eigen_plan import EigenDriverPlan
from jamma.core.threading import get_physical_core_count
from jamma.io import read_fam_phenotypes
from jamma.kinship.loco import LocoRetainedSet
from jamma.lmm.loco import LocoConfig, run_lmm_loco
from jamma.lmm.loco_eigen import _computed_eigen_pairs, plan_loco_workers
from jamma.lmm.schema import LmmConfig
from jamma.lmm.stats import AssocResult
from tests.conftest import require_fixture
from tests.fixture_paths import LOCO


def _worker_plan_inputs() -> tuple[LocoRetainedSet, EigenDriverPlan]:
    """A 3 GB retained set and a 1 GB eigen driver: each worker costs 2 GB.

    One 1 GB matrix and no disk buffer make the retained set 3 GB. A worker
    owns one 1 GB copy of the streamed K_loco plus the driver's 1 GB peak.
    """
    retained = LocoRetainedSet(matrix_gb=1.0, chunk_buffer_gb=0.0)
    eigen_plan = EigenDriverPlan(
        driver="DSYEVD-inplace",
        use_inplace=True,
        use_dsyevr=False,
        no_vendor=False,
        required_gb=1.0,
        pre_fallback_gb=1.0,
        dsyevr_peak_gb=2.0,
        inplace_peak_gb=1.0,
    )
    return retained, eigen_plan


@pytest.mark.tier0
def test_plan_loco_workers_memory_clamps_to_two():
    """8.8 GB free: headroom 8.0, minus the 3 GB retained set, is two 2 GB workers.

    ``fits(3 + 2 * 2 = 7, 8.8)`` holds (7 + 0.7 < 8.8), so the tie check
    leaves 2 alone. The request of 6, the 22 chromosomes and 8 cores are all
    above that, so memory is the binding cap.
    """
    retained, eigen_plan = _worker_plan_inputs()
    plan = plan_loco_workers(
        6,
        n_chr=22,
        retained=retained,
        eigen_plan=eigen_plan,
        available_gb=8.8,
        budget_gb=None,
        association_gb=0.0,
        cores=8,
    )

    assert plan.workers == 2
    assert plan.memory_allows == 2
    assert plan.cores == 8
    assert plan.consumer_gb == 4.0


@pytest.mark.tier0
def test_plan_loco_workers_requested_one_is_sequential():
    """Requested 1 stays 1 and reserves the driver's peak alone, no copy."""
    retained, eigen_plan = _worker_plan_inputs()
    plan = plan_loco_workers(
        1,
        n_chr=22,
        retained=retained,
        eigen_plan=eigen_plan,
        available_gb=100.0,
        budget_gb=None,
        association_gb=0.0,
        cores=8,
    )

    assert plan.workers == 1
    assert plan.consumer_gb == 1.0


@pytest.mark.tier0
def test_plan_loco_workers_caps_at_chromosome_count():
    """Requested 6 with 3 chromosomes is 3: no worker would ever get a fourth."""
    retained, eigen_plan = _worker_plan_inputs()
    plan = plan_loco_workers(
        6,
        n_chr=3,
        retained=retained,
        eigen_plan=eigen_plan,
        available_gb=100.0,
        budget_gb=None,
        association_gb=0.0,
        cores=8,
    )

    assert plan.workers == 3
    assert plan.memory_allows > 3


@pytest.mark.tier0
def test_plan_loco_workers_caps_at_physical_cores():
    """Requested 6 on 4 cores is 4: Accelerate runs one core per solve."""
    retained, eigen_plan = _worker_plan_inputs()
    plan = plan_loco_workers(
        6,
        n_chr=22,
        retained=retained,
        eigen_plan=eigen_plan,
        available_gb=100.0,
        budget_gb=None,
        association_gb=0.0,
        cores=4,
    )

    assert plan.workers == 4


@pytest.mark.tier0
def test_plan_loco_workers_honours_budget_ceiling():
    """A 9 GB budget under 100 GB free: (9 - 3) / 2 = 3 workers, not 43."""
    retained, eigen_plan = _worker_plan_inputs()
    plan = plan_loco_workers(
        6,
        n_chr=22,
        retained=retained,
        eigen_plan=eigen_plan,
        available_gb=100.0,
        budget_gb=9.0,
        association_gb=0.0,
        cores=8,
    )

    assert plan.workers == 3
    assert plan.memory_allows == 3


@pytest.mark.tier0
def test_plan_loco_workers_never_below_one():
    """1 GB free cannot hold even the retained set; the plan is still 1 worker.

    The stream's own gate vetoes the run; the worker planner only decides how
    many chromosomes overlap, and one is the sequential path.
    """
    retained, eigen_plan = _worker_plan_inputs()
    plan = plan_loco_workers(
        6,
        n_chr=22,
        retained=retained,
        eigen_plan=eigen_plan,
        available_gb=1.0,
        budget_gb=None,
        association_gb=0.0,
        cores=8,
    )

    assert plan.workers == 1
    assert plan.memory_allows == 1
    assert plan.consumer_gb == 1.0


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
    _, eigen_plan = _worker_plan_inputs()
    chr_names = list(_ORDER_BY_CHR)
    return _computed_eigen_pairs(
        _synthetic_stream(),
        chr_names,
        valid_mask=np.ones(8, dtype=bool),
        n_valid=8,
        pre_subset=False,
        all_samples_valid=True,
        partitions={c: np.arange(5) for c in chr_names},
        check_memory=False,
        show_progress=False,
        loco=LocoConfig(),
        cache_write=None,
        eigen_plan=eigen_plan,
        mem_budget=None,
        workers=workers,
        solve=solve,
    )


@pytest.mark.tier0
def test_computed_eigen_pairs_overlaps_workers_and_keeps_chromosome_order():
    """Three workers solve three chromosomes at once; pairs still come out in order.

    Every solve waits at a three-party barrier, so the test only passes when
    all three run at the same time: sequential solves would each time out
    there. Chromosome 1 then sleeps longest and finishes last, so its pair
    coming out first proves the generator resolves futures in chromosome
    order rather than completion order.
    """
    barrier = threading.Barrier(3)
    idents: list[int] = []

    def solve(K: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        idents.append(threading.get_ident())
        barrier.wait(timeout=10)
        n = len(K)
        time.sleep((11 - n) * 0.05)
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

    threads_before = threading.active_count()
    pairs = _computed_pairs(workers=3, solve=solve)

    chr_name, _, _ = next(pairs)
    assert chr_name == "1"
    with pytest.raises(RuntimeError, match="chromosome 2 failed"):
        next(pairs)

    assert threading.active_count() == threads_before
    pairs.close()


# ---------------------------------------------------------------------------
# run_lmm_loco wiring
# ---------------------------------------------------------------------------


def _loco_run(workers: int, monkeypatch):
    monkeypatch.setenv("JAMMA_LOCO_WORKERS", str(workers))
    return run_lmm_loco(
        bed_path=LOCO.bfile,
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


@pytest.mark.tier0
def test_plan_loco_workers_prices_the_association_pass_over_the_solves_in_flight():
    """While chromosome c's association runs, the other workers still hold copies.

    Same machine as ``test_plan_loco_workers_memory_clamps_to_two`` (headroom
    8.0, retained 3, 2 GB per worker), but a 5 GB association pass. Two
    workers would need 3 + max(2 * 2, 5 + 1 * 2) = 10 GB, which does not fit;
    one worker needs 3 + max(1, 5) = 8 GB, the sequential figure, which does.
    At 13.3 GB free three workers need 3 + max(6, 5 + 2 * 2) = 12 GB, and 12
    plus its 1.2 GB margin is under 13.3, where 13.2 would tie and fail.
    """
    retained, eigen_plan = _worker_plan_inputs()
    plan = plan_loco_workers(
        6,
        n_chr=22,
        retained=retained,
        eigen_plan=eigen_plan,
        available_gb=8.8,
        budget_gb=None,
        association_gb=5.0,
        cores=8,
    )

    assert plan.workers == 1
    assert plan.memory_allows == 1
    assert plan.consumer_gb == 1.0

    roomier = plan_loco_workers(
        6,
        n_chr=22,
        retained=retained,
        eigen_plan=eigen_plan,
        available_gb=13.3,
        budget_gb=None,
        association_gb=5.0,
        cores=8,
    )

    assert roomier.workers == 3
    assert roomier.consumer_gb == 9.0
