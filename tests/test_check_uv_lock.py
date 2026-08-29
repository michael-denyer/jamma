"""Tests for scripts/check-uv-lock.sh.

The hook's job is to fail when uv.lock drifts from pyproject.toml. Its
other job, easy to lose, is to let the reader see *why* uv refused. These
cover both by putting a fake ``uv`` on PATH.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check-uv-lock.sh"


def _run_with_fake_uv(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Run the hook with a fake ``uv`` earliest on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "uv"
    fake.write_text(f"#!/bin/sh\n{body}\n")
    fake.chmod(0o755)

    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return subprocess.run(
        ["sh", str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_passes_when_uv_reports_lock_in_sync(tmp_path: Path) -> None:
    result = _run_with_fake_uv(tmp_path, "exit 0")
    assert result.returncode == 0, result.stderr


def test_fails_when_uv_reports_drift(tmp_path: Path) -> None:
    result = _run_with_fake_uv(tmp_path, "exit 1")
    assert result.returncode == 1
    assert "Run: uv lock" in result.stdout


def test_uv_own_error_reaches_the_reader(tmp_path: Path) -> None:
    """uv's stderr must not be discarded.

    When uv fails for a reason other than drift, a swallowed stderr leaves
    only "run uv lock", which is the wrong instruction and unfixable
    without rerunning uv by hand.
    """
    result = _run_with_fake_uv(
        tmp_path, "echo 'error: failed to parse pyproject.toml' >&2\nexit 2"
    )
    assert result.returncode == 1
    assert "failed to parse pyproject.toml" in result.stderr
