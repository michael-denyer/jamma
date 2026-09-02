"""Bit-exactness lever for the memory estimates and the memory gates.

Every memory estimate and every gate decision that decides whether a run
proceeds, which eigendecomposition driver runs, and how many LOCO passes
the kinship loop makes is pinned here as one sha256 over a canonical table,
the same way ``test_chunk_plan_digest.py`` pins the chunk planner. A
refactor that preserves the estimate and the decision leaves the digest
alone; a policy change (a different margin, a different inequality) updates
it deliberately, in the same commit as the measured reason.

The grid includes exact-tie rows (``required + margin == available``) so a
change to the strictness of any inequality shows up as a digest change.

The kinship phase budgets the dsyrk backend's scratch, which is zero on the
native path and a pure function of ``n`` on the NumPy fallback. The table
is always priced under the fallback so the digest is the same on every
machine, whichever backend it built.

Regenerate the digest with ``uv run python tests/test_memory_ledger_digest.py``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from unittest.mock import patch

import pytest

from jamma import jlinalg
from jamma.core import memory
from jamma.core.eigen_plan import plan_eigen_driver
from jamma.core.memory import (
    estimate_lmm_memory,
    estimate_streaming_memory,
    margin_gb,
)
from jamma.kinship.loco import _decide_loco_passes

pytestmark = pytest.mark.tier0

# Recorded from master ebc07b6, before the ledger and the gate were reshaped,
# under the forced NumPy dsyrk backend the table always prices with.
EXPECTED_DIGEST = "e943012ebe93a6e4b38144b222eb4f0c87fd78d18c9a19e811fb201ab845898a"
EXPECTED_ROWS = 2438

N_SAMPLES = (30, 1_410, 5_000, 10_001, 50_000, 200_000)
CHUNK_SIZE = (10_000, 1_000)
N_CVT = (1, 4)
PIPELINE_BUFFERS = (1, 2)
COMPUTE_CHUNK = (None, 765)
EIGEN_PEAK_GB = (None, 12.5)
UAB_IAB_GB = (None, 0.7)

N_SNPS = (100, 500_000)
LMM_BATCH = (20_000, 765)
N_BUFFERS = (1, 2)

PEAKS_GB = (0.0, 0.5, 9.99, 50.0, 99.999, 100.0, 100.001, 1_000.0)
AVAILABLE_GB = (0.001, 1.0, 8.0, 40.0, 64.0, 110.0, 1_000.0)
BUDGET_GB = (None, 1.0, 64.0)
FLAGS = (False, True)
N_CHR = (0, 1, 22)
MAX_BATCH_CHRS = (None, 3)


def _f(x: float) -> str:
    return repr(float(x))


def _tie(required_gb: float) -> float:
    return required_gb + margin_gb(required_gb)


def _streaming_rows() -> list[list]:
    rows: list[list] = []
    for n, chunk, n_cvt, buffers, compute, eigen_peak, uab in itertools.product(
        N_SAMPLES,
        CHUNK_SIZE,
        N_CVT,
        PIPELINE_BUFFERS,
        COMPUTE_CHUNK,
        EIGEN_PEAK_GB,
        UAB_IAB_GB,
    ):
        ledger = estimate_streaming_memory(
            n,
            chunk_size=chunk,
            n_cvt=n_cvt,
            pipeline_buffers=buffers,
            compute_chunk_size=compute,
            eigendecomp_peak_gb=eigen_peak,
            uab_iab_gb=uab,
        )
        rows.append(
            [
                "streaming",
                n,
                chunk,
                n_cvt,
                buffers,
                compute,
                eigen_peak,
                uab,
                _f(ledger.kinship_gb),
                _f(ledger.eigen_gb),
                _f(ledger.lmm_gb),
                _f(ledger.peak_gb),
            ]
        )
    return rows


def _batch_rows() -> list[list]:
    rows: list[list] = []
    for n, n_snps, batch, n_cvt, buffers in itertools.product(
        N_SAMPLES, N_SNPS, LMM_BATCH, N_CVT, N_BUFFERS
    ):
        batch_gb = estimate_lmm_memory(
            n, n_snps, lmm_batch_size=batch, n_cvt=n_cvt, n_buffers=buffers
        )
        rows.append(["batch", n, n_snps, batch, n_cvt, buffers, _f(batch_gb)])
    return rows


def _gate_rows() -> list[list]:
    rows: list[list] = []
    for peak in PEAKS_GB:
        rows.append(["margin", _f(peak), _f(margin_gb(peak))])
    for required, available in itertools.product(PEAKS_GB, AVAILABLE_GB):
        rows.append(
            ["fits", _f(required), _f(available), memory.fits(required, available)]
        )
    for required in PEAKS_GB:
        tie = _tie(required)
        rows.append(["fits:tie", _f(required), _f(tie), memory.fits(required, tie)])
    for required, available, budget in itertools.product(
        PEAKS_GB, AVAILABLE_GB, BUDGET_GB
    ):
        rows.append(
            [
                "require",
                _f(required),
                _f(available),
                budget,
                _require_outcome(required, available, budget),
            ]
        )
    for required in PEAKS_GB:
        rows.append(
            [
                "require:tie",
                _f(required),
                _require_outcome(required, _tie(required), None),
            ]
        )
    return rows


def _require_outcome(required: float, available: float, budget: float | None) -> str:
    try:
        memory.require(required, available, "op", budget_gb=budget)
    except MemoryError as exc:
        return str(exc)
    return "ok"


def _eigen_driver_rows() -> list[list]:
    rows: list[list] = []
    for n, available, has_dsyevd, has_dsyevr, no_vendor, inplace in itertools.product(
        N_SAMPLES, AVAILABLE_GB, FLAGS, FLAGS, FLAGS, FLAGS
    ):
        rows.append(
            _eigen_driver_row(n, available, has_dsyevd, has_dsyevr, no_vendor, inplace)
        )
    for n, inplace in itertools.product(N_SAMPLES, FLAGS):
        probe = plan_eigen_driver(
            n,
            1e12,
            has_dsyevd=True,
            has_dsyevr=True,
            no_vendor=False,
            inplace_eligible=inplace,
        )
        rows.append(
            _eigen_driver_row(
                n, _tie(probe.required_gb), True, True, False, inplace, tag="eigen:tie"
            )
        )
    return rows


def _eigen_driver_row(
    n, available, has_dsyevd, has_dsyevr, no_vendor, inplace, tag="eigen"
):
    plan = plan_eigen_driver(
        n,
        available,
        has_dsyevd=has_dsyevd,
        has_dsyevr=has_dsyevr,
        no_vendor=no_vendor,
        inplace_eligible=inplace,
    )
    return [
        tag,
        n,
        _f(available),
        has_dsyevd,
        has_dsyevr,
        no_vendor,
        inplace,
        plan.driver,
        plan.use_inplace,
        plan.use_dsyevr,
        plan.no_vendor,
        _f(plan.required_gb),
        _f(plan.pre_fallback_gb),
    ]


def _loco_rows() -> list[list]:
    rows: list[list] = []
    for n_mat, n_chr, chunk, available, max_batch in itertools.product(
        N_SAMPLES, N_CHR, CHUNK_SIZE, AVAILABLE_GB, MAX_BATCH_CHRS
    ):
        for n_samples in (n_mat, n_mat + 7):
            rows.append(_loco_row(n_mat, n_samples, n_chr, chunk, available, max_batch))
    for n_mat, n_chr in itertools.product(N_SAMPLES, N_CHR):
        probe = _decide_loco_passes(
            n_mat, n_mat, n_chr, 10_000, 1e12, max_batch_chrs=None
        )
        rows.append(
            _loco_row(
                n_mat,
                n_mat,
                n_chr,
                10_000,
                _tie(probe.single_pass_gb),
                None,
                tag="loco:tie",
            )
        )
    return rows


def _loco_row(n_mat, n_samples, n_chr, chunk, available, max_batch, tag="loco"):
    plan = _decide_loco_passes(
        n_mat, n_samples, n_chr, chunk, available, max_batch_chrs=max_batch
    )
    return [
        tag,
        n_mat,
        n_samples,
        n_chr,
        chunk,
        _f(available),
        max_batch,
        plan.single_pass,
        plan.batch_size,
        _f(plan.single_pass_gb),
        _f(plan.min_required_gb),
        _f(plan.eigendecomp_min_gb),
    ]


def ledger_table() -> list[list]:
    """Every estimate and gate decision over the grid, in a fixed order."""
    with patch.object(
        jlinalg,
        "_dsyrk_backend",
        jlinalg._dsyrk_numpy_impl,
        # allow-patch: forces the dispatch fallback so the kinship rows price
        # the same scratch on every machine. _dsyrk_backend is resolved from
        # blas_has_dsyrk at import time, so toggling that flag would not
        # redirect dispatch.
    ):
        return [
            *_streaming_rows(),
            *_batch_rows(),
            *_gate_rows(),
            *_eigen_driver_rows(),
            *_loco_rows(),
        ]


def ledger_digest(rows: list[list]) -> str:
    canonical = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_memory_ledger_digest_is_unchanged() -> None:
    rows = ledger_table()
    assert len(rows) == EXPECTED_ROWS
    assert ledger_digest(rows) == EXPECTED_DIGEST, (
        "an estimate or a gate decision changed for at least one input; if "
        "that is deliberate, record the measurement and update EXPECTED_DIGEST "
        "via `uv run python tests/test_memory_ledger_digest.py`"
    )


if __name__ == "__main__":
    table = ledger_table()
    print(f"rows={len(table)} digest={ledger_digest(table)}")
