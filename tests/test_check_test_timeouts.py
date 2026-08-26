"""Tests for scripts/check_test_timeouts.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import install_lint_script

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_test_timeouts.py"


def _run(tmp_path: Path, test_file_contents: str) -> subprocess.CompletedProcess:
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_stub.py").write_text(test_file_contents)
    return subprocess.run(
        [sys.executable, str(script_copy)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),  # script uses __file__, cwd doesn't matter, but harmless
    )


@pytest.mark.tier0
def test_short_timeout_passes(tmp_path):
    result = _run(tmp_path, "@pytest.mark.timeout(30)\ndef test_x(): pass\n")
    assert result.returncode == 0, result.stderr


@pytest.mark.tier0
def test_long_timeout_without_justification_fails(tmp_path):
    result = _run(tmp_path, "@pytest.mark.timeout(600)\ndef test_x(): pass\n")
    assert result.returncode == 1
    assert "600" in result.stderr


@pytest.mark.tier0
def test_long_timeout_with_same_line_justified_passes(tmp_path):
    result = _run(
        tmp_path,
        "@pytest.mark.timeout(600)  # justified: scale test\ndef test_x(): pass\n",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.tier0
def test_long_timeout_with_preceding_justified_passes(tmp_path):
    result = _run(
        tmp_path,
        (
            "# justified: 120k-sample eigendecomp legitimately takes minutes\n"
            "@pytest.mark.timeout(1800)\n"
            "def test_x(): pass\n"
        ),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.tier0
def test_threshold_boundary_is_exactly_120s(tmp_path):
    """120 passes, 121 fails — documents the boundary."""
    pass_result = _run(
        tmp_path / "a", "@pytest.mark.timeout(120)\ndef test_x(): pass\n"
    )
    assert pass_result.returncode == 0, pass_result.stderr
    fail_result = _run(
        tmp_path / "b", "@pytest.mark.timeout(121)\ndef test_x(): pass\n"
    )
    assert fail_result.returncode == 1


@pytest.mark.tier0
def test_justified_too_far_above_does_not_count(tmp_path):
    """Justification must be within 3 lines above — further is too
    decoupled and could be unrelated."""
    result = _run(
        tmp_path,
        (
            "# justified: unrelated context\n"
            "\n"
            "\n"
            "\n"
            "\n"
            "@pytest.mark.timeout(600)\n"
            "def test_x(): pass\n"
        ),
    )
    assert result.returncode == 1


@pytest.mark.tier0
def test_unreadable_file_fails_instead_of_passing_silently(tmp_path):
    """A test file the lint cannot decode must fail the gate, not be skipped.

    Asserts the report text, not just the exit code. See the same test in
    tests/test_check_quiet_flags.py for why: a traceback also exits 1.
    """
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_stub.py").write_bytes(
        b"@pytest.mark.timeout(600)\n\xff\xfe not utf-8\n"
    )

    result = subprocess.run(
        [sys.executable, str(script_copy)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Traceback" not in result.stderr, (
        f"the gate must report, not crash:\n{result.stderr}"
    )
    assert "Files could not be read, so they were not checked:" in result.stderr
    assert "  tests/test_stub.py: UnicodeDecodeError:" in result.stderr
    assert (
        "1 file(s) skipped. A skipped file is an unchecked file, and this "
        "gate reports success only when every target was read."
    ) in result.stderr


@pytest.mark.tier0
def test_missing_tests_dir_fails_instead_of_passing_silently(tmp_path):
    """A repo root with no tests/ directory must fail the gate, not report
    success over a tree it never looked at (the #188 lesson: a gate that
    reports 0 for an absent target cannot tell 'nothing to check' from
    'the tree moved and nobody noticed').
    """
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")
    # Deliberately no tests/ directory created under tmp_path.

    result = subprocess.run(
        [sys.executable, str(script_copy)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )

    assert result.returncode == 1
    assert "does not exist" in result.stderr
    assert "tests" in result.stderr


@pytest.mark.tier0
def test_unexpected_argv_fails_instead_of_being_ignored(tmp_path):
    """An argument must fail the gate, not be silently ignored.

    pre-commit wires this hook with pass_filenames: false, so it is never
    handed a file list; passing one anyway is unexpected and, before this
    test, `python3 scripts/check_test_timeouts.py some_file` reported
    green regardless of what was in the real tests/ tree.
    """
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_stub.py").write_text(
        "@pytest.mark.timeout(600)\ndef test_x(): pass\n"
    )

    result = subprocess.run(
        [sys.executable, str(script_copy), "some_file"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
    )

    assert result.returncode == 1
    assert "takes no arguments" in result.stderr


@pytest.mark.tier0
def test_unreadable_file_does_not_hide_violations_elsewhere(tmp_path):
    """One undecodable file must not stop the other files being checked.

    Reading the batch and collecting the failures, rather than raising on the
    first, is what keeps the violation in test_other.py visible.
    """
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_stub.py").write_bytes(b"\xff\xfe not utf-8\n")
    (tests_dir / "test_other.py").write_text(
        "@pytest.mark.timeout(999)\ndef test_x(): pass\n"
    )

    result = subprocess.run(
        [sys.executable, str(script_copy)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "  tests/test_stub.py: UnicodeDecodeError:" in result.stderr
    assert "Unjustified long test timeouts detected:" in result.stderr
    assert "tests/test_other.py:1: @pytest.mark.timeout(999)" in result.stderr
