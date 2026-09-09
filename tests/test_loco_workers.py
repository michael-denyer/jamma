"""LOCO worker tests: the worker planner and the concurrent eigen generator.

Related LOCO test files:
- test_loco_numpy.py: NumPy LOCO parity and the pass planner
- test_loco_eigen_cache.py: LOCO eigen cache write/read round-trip
"""

from __future__ import annotations

import pytest

from jamma.core.eigen_plan import EigenDriverPlan
from jamma.kinship.loco import LocoRetainedSet
from jamma.lmm.loco_eigen import plan_loco_workers


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
        cores=8,
    )

    assert plan.workers == 1
    assert plan.memory_allows == 1
    assert plan.consumer_gb == 1.0
