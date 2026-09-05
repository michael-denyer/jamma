"""Bit-exactness lever for the chunk plan.

``LmmChunkPlan.plan`` and ``LmmChunkPlan.narrow`` decide every run's chunk
width, and under MKL two different chunk plans are not bit-identical in the
rotation's last bits, so a plan that drifts moves results in a way no
tolerance test sees. This pins the planner's answer over a grid of shapes,
dispatch paths, budgets, caps, and both BLAS-controllability states as one
sha256 over the canonical table, the same way the kinship digest lever pins
``-gk``. A refactor that preserves the decision leaves the digest alone; a
policy change updates it deliberately, in the same commit as the measured
reason.

Regenerate the digest with ``uv run python tests/test_chunk_plan_digest.py``.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import pytest

from jamma.lmm.chunk_sizing import LmmChunkPlan
from jamma.lmm.dispatch import DispatchPath

pytestmark = pytest.mark.tier0

# R2 removes the 100-SNP floor when it exceeds the variable memory budget.
# Against 346c3eb6, 1,280 of 7,560 rows change, all at the 1 MB budget;
# every changed width decreases. The 2 GB and 40 GB rows are unchanged.
EXPECTED_DIGEST = "1959a9a1959f0614e367034eff7e3256ddd2386ffd2d23a01b71c4922dc2be62"
EXPECTED_ROWS = 7560

N_SAMPLES = (30, 1_410, 5_000, 10_000, 10_001, 30_000, 100_000)
N_SNPS = (100, 12_226, 500_000)
N_CVT = (1, 4)
DISPATCH = (DispatchPath.FUSED, DispatchPath.FUSED_GENERAL, DispatchPath.NUMPY_FALLBACK)
BUDGET_BYTES = (int(2e9), int(40e9), int(1e6))
MAX_CHUNK = (None, 1_000)
BLAS_CONTROLLABLE = (False, True)


def _row(plan: LmmChunkPlan) -> list[int | bool]:
    return [plan.chunk_size, plan.n_chunks, plan.n_buffers, plan.use_pipeline]


def plan_table() -> list[list]:
    """Every plan and its narrowings over the grid, in a fixed order."""
    rows: list[list] = []
    for (
        n_samples,
        n_snps,
        n_cvt,
        dispatch,
        budget,
        max_chunk,
        blas,
    ) in itertools.product(
        N_SAMPLES, N_SNPS, N_CVT, DISPATCH, BUDGET_BYTES, MAX_CHUNK, BLAS_CONTROLLABLE
    ):
        plan = LmmChunkPlan.plan(
            n_samples,
            n_snps,
            n_cvt,
            dispatch,
            budget_bytes=budget,
            blas_controllable=blas,
            max_chunk_size=max_chunk,
        )
        key = [n_samples, n_snps, n_cvt, dispatch.name, budget, max_chunk, blas]
        rows.append([*key, "plan", *_row(plan)])
        for n_filtered in (n_snps, n_snps // 2, 1, 0):
            rows.append([*key, f"narrow:{n_filtered}", *_row(plan.narrow(n_filtered))])
    return rows


def plan_digest(rows: list[list]) -> str:
    canonical = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_chunk_plan_digest_is_unchanged() -> None:
    rows = plan_table()
    assert len(rows) == EXPECTED_ROWS
    assert plan_digest(rows) == EXPECTED_DIGEST, (
        "the chunk plan changed for at least one input; if that is deliberate, "
        "record the measurement and update EXPECTED_DIGEST via "
        "`uv run python tests/test_chunk_plan_digest.py`"
    )


@pytest.mark.parametrize(
    "n_samples, n_snps, dispatch, budget, blas, expected",
    [
        # The 16-chunk cut: one chunk by budget, cut to 16 and pipelined.
        (1_410, 12_226, DispatchPath.FUSED, int(2e9), False, (765, 16, 2, True)),
        # Same shape, controllable BLAS: no cut, one sequential chunk.
        (1_410, 12_226, DispatchPath.FUSED, int(2e9), True, (12_226, 1, 1, False)),
        # Past the 10,000-sample bound: no cut.
        (30_000, 5_000, DispatchPath.FUSED, int(2e9), False, (5_000, 1, 1, False)),
        # The NumPy fallback never pipelines.
        (
            1_410,
            12_226,
            DispatchPath.NUMPY_FALLBACK,
            int(2e9),
            False,
            (12_226, 1, 1, False),
        ),
        # A tight budget splits past the threshold on its own; the pipelined
        # re-size halves the budget across two live buffers.
        (100_000, 500_000, DispatchPath.FUSED, int(2e9), True, (1_250, 400, 2, True)),
        # A throughput floor must not exceed the budget: 100 * 1410 * 8
        # is 1,128,000 bytes; 88 SNPs occupy 992,640 bytes and fit 1 MB.
        (1_410, 100, DispatchPath.FUSED, int(1e6), False, (88, 2, 1, False)),
    ],
)
def test_chunk_plan_spot_rows(n_samples, n_snps, dispatch, budget, blas, expected):
    plan = LmmChunkPlan.plan(
        n_samples, n_snps, 1, dispatch, budget_bytes=budget, blas_controllable=blas
    )
    assert tuple(_row(plan)) == expected


def test_narrow_only_decreases_width_and_only_switches_pipelining_off() -> None:
    plan = LmmChunkPlan(chunk_size=765, n_chunks=16, n_buffers=2, use_pipeline=True)

    assert plan.narrow(12_226) == plan
    assert plan.narrow(5_000) == LmmChunkPlan(765, 7, 1, False)
    assert plan.narrow(1) == LmmChunkPlan(1, 1, 1, False)
    assert plan.narrow(0) == LmmChunkPlan(1, 0, 1, False)
    with pytest.raises(ValueError, match="n_filtered"):
        plan.narrow(-1)


if __name__ == "__main__":
    table = plan_table()
    print(f"rows={len(table)} digest={plan_digest(table)}")
