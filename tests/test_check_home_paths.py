"""Tests for scripts/check_home_paths.py.

The lint rejects a home directory path in a tracked file. Covers the two
path forms, the forms that name no account (``~/`` and placeholders), files
that are not UTF-8, the ``allow-home-path:`` escape hatch, and the committed
tree itself.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import install_lint_script

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_home_paths.py"


def _run(tmp_path: Path, rel_path: str, content: bytes) -> subprocess.CompletedProcess:
    """Copy the script into tmp_path/scripts/, write one target file, run it."""
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")

    dst = tmp_path / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(content)

    return subprocess.run(
        [sys.executable, str(script_copy), str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "line,found",
    [
        ("## Command Line Input = /Users/alice/.local/bin/gemma -lmm 1", "alice"),
        ('  "cwd": "/home/bob/references/tiny",', "bob"),
        ("binary = C:/Users/carol.d/bin/gemma.exe", "carol.d"),
        ("data = /mnt/c/Users/dave/project/tiny.bed", "dave"),
        ("HOME=/home/erin", "erin"),
    ],
)
def test_home_directory_is_detected(tmp_path, line, found):
    result = _run(tmp_path, "gemma.log.txt", f"first line\n{line}\n".encode())
    assert result.returncode == 1
    assert "gemma.log.txt:2:" in result.stderr
    assert found in result.stderr


@pytest.mark.parametrize(
    "line",
    [
        "## Command Line Input = ~/.local/bin/gemma -lmm 1",
        "## Command Line Input = gemma -lmm 1",
        "Fails on /Users/<name>/ and /home/<name>/.",
        'export GEMMA="/home/$USER/bin/gemma"',
    ],
)
def test_path_that_names_no_account_passes(tmp_path, line):
    result = _run(tmp_path, "notes.md", f"{line}\n".encode())
    assert result.returncode == 0, result.stderr


def test_home_directory_in_a_file_that_is_not_utf8_is_detected(tmp_path):
    """Generated artifacts are not all text, and a path inside one still leaks."""
    content = b"\xff\xfe\x00binary\n/Users/alice/build/gemma\n"
    result = _run(tmp_path, "blob.bin", content)
    assert result.returncode == 1
    assert "blob.bin:2:" in result.stderr


def test_allow_marker_skips_the_line(tmp_path):
    content = (
        b"runner = /home/runner/work  # allow-home-path: CI image layout\n"
        b"mine = /home/alice/work\n"
    )
    result = _run(tmp_path, "ci.yml", content)
    assert result.returncode == 1
    assert "ci.yml:2:" in result.stderr
    assert "ci.yml:1:" not in result.stderr


def test_missing_file_fails_with_a_report_instead_of_a_traceback(tmp_path):
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")

    result = subprocess.run(
        [sys.executable, str(script_copy), str(tmp_path / "absent.txt")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Files could not be read, so they were not checked:" in result.stderr
    assert "absent.txt" in result.stderr
    assert "Traceback" not in result.stderr


def test_committed_tree_has_no_home_directory():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
