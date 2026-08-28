"""Tests for the composite-action CI setup ladder.

Asserts every workflow job routes through ``.github/actions/setup-jamma``
instead of copying the checkout-uv-python-sync-compile ladder inline. The
one documented exception is ``fingerprint.yml``, which must build both
sides (PR head and merge base) itself in one job and so cannot delegate
to a single composite-action call; its two ``uv sync`` invocations still
carry ``--locked``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"

# fingerprint.yml rebuilds from nothing on both the PR head and the merge
# base within one job, so it cannot route through one setup-jamma call
# and is the one file allowed a bare `uv sync`.
_ALLOWED_BARE_UV_SYNC = frozenset({"fingerprint.yml"})


def _iter_run_lines(step: dict) -> list[str]:
    run = step.get("run")
    if not run:
        return []
    return run.splitlines()


def _load_workflow(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def workflow_paths() -> list[Path]:
    paths = sorted(_WORKFLOWS_DIR.glob("*.yml"))
    assert paths, f"no workflow files found under {_WORKFLOWS_DIR}"
    return paths


def test_no_job_step_runs_uv_sync_outside_the_composite_action(workflow_paths):
    violations = []
    for path in workflow_paths:
        if path.name in _ALLOWED_BARE_UV_SYNC:
            continue
        workflow = _load_workflow(path)
        for job_name, job in workflow.get("jobs", {}).items():
            for step in job.get("steps", []):
                for line in _iter_run_lines(step):
                    if "uv sync" in line:
                        violations.append(f"{path.name}:{job_name}: {line.strip()}")
    assert not violations, (
        "uv sync must run only inside .github/actions/setup-jamma, or in "
        "fingerprint.yml's documented per-side rebuild:\n" + "\n".join(violations)
    )


def test_fingerprint_bare_uv_sync_is_locked(workflow_paths):
    path = _WORKFLOWS_DIR / "fingerprint.yml"
    workflow = _load_workflow(path)
    sync_lines = [
        line
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        for line in _iter_run_lines(step)
        if "uv sync" in line
    ]
    assert sync_lines, (
        "fingerprint.yml no longer runs uv sync — update the exception list"
    )
    for line in sync_lines:
        assert "--locked" in line, (
            f"fingerprint.yml: uv sync not locked: {line.strip()}"
        )


def test_setup_jamma_action_exists():
    action = _REPO_ROOT / ".github" / "actions" / "setup-jamma" / "action.yml"
    assert action.exists()
    spec = _load_workflow(action)
    assert spec["runs"]["using"] == "composite"
    step_names = {step.get("run", "") for step in spec["runs"]["steps"]}
    assert any("uv sync --locked" in run for run in step_names)


def test_triage_issue_action_exists():
    action = _REPO_ROOT / ".github" / "actions" / "triage-issue" / "action.yml"
    assert action.exists()
    spec = _load_workflow(action)
    assert spec["runs"]["using"] == "composite"
    for key in ("title", "label", "body"):
        assert key in spec["inputs"]


def test_setup_jamma_used_exactly_eight_times(workflow_paths):
    count = 0
    for path in workflow_paths:
        workflow = _load_workflow(path)
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                if step.get("uses") == "./.github/actions/setup-jamma":
                    count += 1
    assert count == 8


def test_flaky_detect_and_sanitizers_use_triage_issue_action():
    for filename, expected_calls in (
        ("flaky-detect.yml", 1),
        ("sanitizers.yml", 1),
    ):
        workflow = _load_workflow(_WORKFLOWS_DIR / filename)
        calls = sum(
            1
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            if step.get("uses") == "./.github/actions/triage-issue"
        )
        assert calls == expected_calls, (
            f"{filename}: expected {expected_calls} triage-issue call(s), found {calls}"
        )


def test_no_github_script_triage_block_remains():
    """Only the shared triage-issue action may still reference
    actions/github-script; the two calling workflows must not."""
    for filename in ("flaky-detect.yml", "sanitizers.yml"):
        workflow = _load_workflow(_WORKFLOWS_DIR / filename)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                uses = str(step.get("uses", ""))
                assert "github-script" not in uses, (
                    f"{filename}: still calls actions/github-script directly "
                    "instead of ./.github/actions/triage-issue"
                )
