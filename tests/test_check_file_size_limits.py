"""Behavior tests for the tracked handwritten-code size gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import install_lint_script

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_file_size_limits.py"


def _run(
    tmp_path: Path,
    tracked: dict[str, str],
    untracked: dict[str, str] | None = None,
    *,
    max_lines: int = 3,
    warn_lines: int = 2,
    warn_function_lines: int = 150,
):
    script = install_lint_script(_SCRIPT, tmp_path / "scripts")

    for relative, content in {**tracked, **(untracked or {})}.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    subprocess.run(
        ["git", "-C", str(tmp_path), "init"],
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", *tracked],
        capture_output=True,
        text=True,
        check=True,
    )
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--max-lines",
            str(max_lines),
            "--warn-lines",
            str(warn_lines),
            "--warn-function-lines",
            str(warn_function_lines),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_tracked_code_above_limit_fails_and_untracked_code_is_ignored(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        tracked={"src/large.py": "line\n" * 4},
        untracked={"src/untracked.py": "line\n" * 5},
    )

    assert result.returncode == 1
    assert "src/large.py: 4 lines" in result.stderr
    assert "untracked.py" not in result.stderr


def test_file_at_limit_passes_and_reports_the_warning_band(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        tracked={
            "src/at_limit.c": "line\n" * 3,
            "tests/near_limit.py": "line\n" * 2,
        },
    )

    assert result.returncode == 0
    assert "src/at_limit.c: 3 lines" in result.stderr
    assert "tests/near_limit.py: 2 lines" in result.stderr
    assert "exceed" not in result.stderr


def test_non_code_files_do_not_enter_the_gate(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        tracked={
            "docs/large.md": "line\n" * 4,
            "src/small.py": "line\n",
        },
    )

    assert result.returncode == 0
    assert "large.md" not in result.stderr


def test_long_python_function_warns_without_failing(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        tracked={
            "src/long_function.py": (
                "def long_function():\n"
                "    value = 1\n"
                "    value += 1\n"
                "    return value\n"
            )
        },
        max_lines=10,
        warn_lines=10,
        warn_function_lines=4,
    )

    assert result.returncode == 0
    assert "long_function spans 4 lines" in result.stderr
