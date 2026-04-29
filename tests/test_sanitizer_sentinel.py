"""ASAN/UBSAN sentinel — opt-in heap-OOB probe.

This test deliberately calls a JAMMA_SENTINEL_UB-gated function in
_lmm_accel that performs a one-byte out-of-bounds heap read. When run
under ASAN, the process aborts with a heap-buffer-overflow trace; the
asan-sentinel-meta-test job in .github/workflows/sanitizers.yml then
asserts the trace appeared in the captured pytest log.

GATING: This test runs ONLY when JAMMA_SANITIZE_EXPECT_FAIL=1 is set in
the environment. On a normal pytest invocation (default CI matrix, local
dev), the @pytest.mark.skipif decorator silently skips. Without this
gating, the test would fail every default CI run because the sentinel
function is either not compiled in (default build) or, if compiled in,
aborts the process under ASAN — both of which are loud failures the
default suite must NOT see.

The asan-sentinel-meta-test job in sanitizers.yml is the ONLY caller
that should set JAMMA_SANITIZE_EXPECT_FAIL=1.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("JAMMA_SANITIZE_EXPECT_FAIL") != "1",
    reason=(
        "Sentinel test runs only under the asan-sentinel-meta-test workflow "
        "job (JAMMA_SANITIZE_EXPECT_FAIL=1 + JAMMA_SENTINEL_UB=1 + "
        "JAMMA_SANITIZE=address). On a normal pytest invocation, this test "
        "is skipped — the workflow explicitly opts in via env. See "
        "docs/TESTING.md §1.10."
    ),
)
@pytest.mark.tier0
def test_sentinel_heap_oob_under_asan() -> None:
    """Calls jamma_sentinel_oob() — under ASAN this aborts the process."""
    from jamma.lmm import _lmm_accel

    # If JAMMA_SENTINEL_UB was NOT set at compile time, the symbol won't
    # exist — that's a configuration error in the workflow YAML.
    assert hasattr(_lmm_accel, "jamma_sentinel_oob"), (
        "jamma_sentinel_oob symbol missing — _lmm_accel was built without "
        "-DJAMMA_SENTINEL_UB. Check the workflow's compile step env."
    )

    # This call performs a deliberate one-byte heap-OOB read. Under ASAN
    # the process aborts here with a heap-buffer-overflow trace pointing
    # at _lmm_accel.c. Without ASAN, the call may or may not crash
    # depending on heap layout — but the workflow always sets
    # JAMMA_SANITIZE=address, so ASAN is always active when this runs.
    _lmm_accel.jamma_sentinel_oob()

    # If we reach this line, ASAN did NOT catch the OOB. The asserter
    # step in the workflow (`grep heap-buffer-overflow sentinel-run.log`)
    # will fail and surface the issue.
    pytest.fail(
        "jamma_sentinel_oob() returned normally — ASAN did not catch the "
        "out-of-bounds read. Sanitizer wiring may be broken."
    )
