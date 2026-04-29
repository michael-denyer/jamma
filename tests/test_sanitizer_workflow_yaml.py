"""Tests for .github/workflows/sanitizers.yml structure.

Asserts the workflow declares the cron schedule, timeout, env vars,
artifact upload, and SHA-pinned actions that Phase 116.1's success
criteria require. Catches accidental edits that drop a critical clause.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "sanitizers.yml"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(_WORKFLOW.read_text())


def _on(workflow):
    """PyYAML normalises the unquoted ``on`` key to the boolean True."""
    return workflow.get("on") or workflow.get(True)


def test_workflow_file_exists():
    assert _WORKFLOW.exists()


def test_cron_is_wednesday_six_utc(workflow):
    on = _on(workflow)
    crons = [s["cron"] for s in on["schedule"]]
    assert "0 6 * * 3" in crons


def test_workflow_dispatch_enabled(workflow):
    on = _on(workflow)
    assert "workflow_dispatch" in on


def test_main_job_has_thirty_minute_timeout(workflow):
    job = workflow["jobs"]["asan-ubsan"]
    assert job["timeout-minutes"] == 30


def test_main_job_sets_force_numpy_fallback(workflow):
    job_env = workflow["jobs"]["asan-ubsan"]["env"]
    assert job_env.get("JAMMA_FORCE_NUMPY_FALLBACK") == "1"
    assert job_env.get("JAMMA_SANITIZE") == "address,undefined"


def test_main_job_asan_options_include_required_flags(workflow):
    opts = workflow["jobs"]["asan-ubsan"]["env"]["ASAN_OPTIONS"]
    for required in [
        "detect_leaks=1",
        "abort_on_error=1",
        "strict_string_checks=1",
        "allocator_may_return_null=1",
    ]:
        assert required in opts, f"{required} missing from ASAN_OPTIONS"


def test_lsan_options_carries_suppressions_path(workflow):
    """The leak-suppressions file is LSAN-format; only LSAN_OPTIONS may
    reference it. ASAN_OPTIONS aborts with 'failed to parse suppressions'
    if it tries to read leak: lines as ASan patterns (Pitfall-9, run
    25108053479)."""
    asan_opts = workflow["jobs"]["asan-ubsan"]["env"]["ASAN_OPTIONS"]
    lsan_opts = workflow["jobs"]["asan-ubsan"]["env"]["LSAN_OPTIONS"]
    assert "suppressions=" not in asan_opts, (
        "ASAN_OPTIONS must NOT carry suppressions=... — file is LSAN-format "
        "and ASan aborts when parsing it as ASan suppressions"
    )
    assert "suppressions=" in lsan_opts, (
        "LSAN_OPTIONS missing the suppressions=... pointer to "
        "scripts/asan-suppressions.txt"
    )
    assert "asan-suppressions.txt" in lsan_opts


def test_main_job_ubsan_options_halt_and_stacktrace(workflow):
    opts = workflow["jobs"]["asan-ubsan"]["env"]["UBSAN_OPTIONS"]
    assert "halt_on_error=1" in opts
    assert "print_stacktrace=1" in opts


def test_artifact_upload_runs_on_failure(workflow):
    """``if: always()`` ensures logs survive even when pytest exits non-zero."""
    for job_name in ("asan-ubsan", "asan-sentinel-meta-test"):
        job = workflow["jobs"][job_name]
        upload_steps = [
            s for s in job["steps"] if "upload-artifact" in str(s.get("uses", ""))
        ]
        assert upload_steps, f"{job_name}: no upload-artifact step"
        assert any(s.get("if") == "always()" for s in upload_steps), (
            f"{job_name}: upload-artifact step missing `if: always()`"
        )


def test_actions_pinned_to_sha(workflow):
    """All ``uses:`` lines must be 40-char SHA, not @vN tag — CLAUDE.md."""
    sha40 = re.compile(r"@[0-9a-f]{40}\b")
    for job_name, job in workflow["jobs"].items():
        for step in job["steps"]:
            uses = step.get("uses")
            if uses:
                assert sha40.search(uses), f"{job_name}: action not SHA-pinned: {uses}"


def test_sentinel_meta_test_job_exists_and_sets_sentinel_macro(workflow):
    job = workflow["jobs"]["asan-sentinel-meta-test"]
    assert job["env"]["JAMMA_SENTINEL_UB"] == "1"
    assert job["env"]["JAMMA_SANITIZE_EXPECT_FAIL"] == "1"


def test_sentinel_assert_step_greps_for_sanitizer_signature(workflow):
    """The asserter step body must look for at least one ASan/UBSAN
    out-of-bounds signature. Either sanitizer can catch the deliberate
    sentinel OOB — UBSAN's bounds check (`runtime error: ... out of
    bounds` / `insufficient space`) typically fires before ASan's
    redzone check (`AddressSanitizer:.*heap-buffer-overflow`) for the
    `buf[5] on malloc(4)` construct used by jamma_sentinel_oob (run
    25109515712). Both count as "sanitizer wired".
    """
    job = workflow["jobs"]["asan-sentinel-meta-test"]
    assertion_steps = [
        s
        for s in job["steps"]
        if (
            "AddressSanitizer:.*heap-buffer-overflow" in str(s.get("run", ""))
            or "runtime error:" in str(s.get("run", ""))
        )
    ]
    assert assertion_steps, (
        "no step asserts on a sanitizer out-of-bounds signature "
        "(ASan heap-buffer-overflow or UBSAN runtime error)"
    )


def test_sentinel_asserter_checks_source_line_attribution(workflow):
    """The asserter step must also confirm the trace mentions _lmm_accel.c."""
    job = workflow["jobs"]["asan-sentinel-meta-test"]
    assertion_steps = [
        s for s in job["steps"] if "_lmm_accel.c" in str(s.get("run", ""))
    ]
    assert assertion_steps, "no step checks for _lmm_accel.c source-line attribution"
