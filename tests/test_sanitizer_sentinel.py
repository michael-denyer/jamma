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
    """Calls jamma_sentinel_oob() — under ASAN this aborts the process.

    Prints checkpoint markers to stderr with explicit flushes so that if
    the process aborts mid-test, the workflow log still shows how far we
    got. ASan's own trace (heap-buffer-overflow, frame in _lmm_accel.c)
    is the success signal the asan-sentinel-meta-test job greps for.
    """
    import sys

    print("CHECKPOINT-1: about to import _lmm_accel", file=sys.stderr, flush=True)
    from jamma.lmm import _lmm_accel

    print(
        f"CHECKPOINT-2: _lmm_accel imported from {_lmm_accel.__file__!r}",
        file=sys.stderr,
        flush=True,
    )

    has_sentinel = hasattr(_lmm_accel, "jamma_sentinel_oob")
    print(
        f"CHECKPOINT-3: hasattr(_lmm_accel, 'jamma_sentinel_oob') = {has_sentinel}",
        file=sys.stderr,
        flush=True,
    )
    assert has_sentinel, (
        "jamma_sentinel_oob symbol missing — _lmm_accel was built without "
        "-DJAMMA_SENTINEL_UB. Check the workflow's compile step env."
    )

    print(
        "CHECKPOINT-4: about to call jamma_sentinel_oob() — ASan must abort",
        file=sys.stderr,
        flush=True,
    )
    # Deliberate one-byte heap-OOB read. Under ASAN the process aborts
    # here with a heap-buffer-overflow trace pointing at _lmm_accel.c.
    result = _lmm_accel.jamma_sentinel_oob()

    print(
        f"CHECKPOINT-5: jamma_sentinel_oob returned {result!r} — ASan did NOT catch",
        file=sys.stderr,
        flush=True,
    )
    pytest.fail(
        "jamma_sentinel_oob() returned normally — ASAN did not catch the "
        "out-of-bounds read. Sanitizer wiring may be broken."
    )
