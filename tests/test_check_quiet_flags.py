"""Tests for scripts/check_quiet_flags.py.

The lint enforces CLAUDE.md's "No Quiet Flags Anywhere" rule and the
hook-skip ban. Covers positive cases (flags must be caught), expected
negatives (documentation mentions, unrelated `-q` uses), and the
``# allow-quiet:`` escape hatch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import install_lint_script

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_quiet_flags.py"


def _run(tmp_path: Path, rel_path: str, content: str) -> subprocess.CompletedProcess:
    """Copy the script into tmp_path/scripts/, write one target file, run it."""
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")

    dst = tmp_path / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content)

    return subprocess.run(
        [sys.executable, str(script_copy), str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_long_quiet_flag_is_detected(tmp_path):
    result = _run(tmp_path, "foo.sh", "pip install numpy --quiet\n")
    assert result.returncode == 1
    assert "--quiet" in result.stderr


def test_silent_flag_is_detected(tmp_path):
    result = _run(tmp_path, "foo.py", 'subprocess.run(["curl", "--silent", url])\n')
    assert result.returncode == 1
    assert "--silent" in result.stderr


@pytest.mark.parametrize(
    "cmd",
    ["pip", "uv", "apt-get", "pytest", "ruff", "npm", "curl"],
)
def test_short_q_after_known_command_is_detected(tmp_path, cmd):
    result = _run(tmp_path, "foo.sh", f"{cmd} install something -q\n")
    assert result.returncode == 1
    assert "-q" in result.stderr


def test_short_q_in_unrelated_context_is_not_flagged(tmp_path):
    """`-q` on a non-command line (not preceded by a known tool) is not
    flagged — too noisy, and CLAUDE.md's concern is CI tool silencing,
    not arbitrary `-q` strings."""
    result = _run(tmp_path, "foo.py", 'config = {"mode": "-q"}\n')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "flag_line,marker",
    [
        ("git commit -m x --no-verify", "--no-verify"),
        ("git commit --no-gpg-sign -m x", "--no-gpg-sign"),
        ("git -c commit.gpgsign=false commit -m x", "-c commit.gpgsign=false"),
    ],
)
def test_hook_skip_flags_are_detected(tmp_path, flag_line, marker):
    result = _run(tmp_path, "deploy.sh", f"{flag_line}\n")
    assert result.returncode == 1
    assert marker in result.stderr


def test_flag_in_comment_is_ignored(tmp_path):
    """A documentation mention like `# do NOT pass --quiet` is not a
    real invocation. The lint strips trailing comments before matching."""
    result = _run(
        tmp_path, "note.py", "# reminder: never add --quiet to this command\n"
    )
    assert result.returncode == 0, result.stderr


def test_allow_quiet_escape_hatch(tmp_path):
    """A line with `# allow-quiet:` opts out. Reserved for documented
    exceptions."""
    result = _run(
        tmp_path,
        "diff_lint.sh",
        "ruff check --quiet diff/  # allow-quiet: exit code IS the signal\n",
    )
    assert result.returncode == 0, result.stderr


def test_multiple_violations_on_one_line_all_reported(tmp_path):
    result = _run(tmp_path, "bad.sh", "git commit --no-verify --no-gpg-sign -m foo\n")
    assert result.returncode == 1
    assert "--no-verify" in result.stderr
    assert "--no-gpg-sign" in result.stderr


def test_clean_file_passes(tmp_path):
    result = _run(tmp_path, "good.sh", "pip install numpy\npytest tests/\n")
    assert result.returncode == 0, result.stderr


def test_shebang_line_is_ignored(tmp_path):
    """The shebang `#!/usr/bin/env ...` is not a comment for our purposes,
    but also not a place where banned flags would appear. Explicitly
    verify nothing weird happens."""
    result = _run(tmp_path, "s.py", "#!/usr/bin/env python3\nprint('ok')\n")
    assert result.returncode == 0, result.stderr


def test_unreadable_file_fails_instead_of_passing_silently(tmp_path):
    """A file the lint cannot decode must fail the gate, not be skipped.

    Skipping means a quiet flag inside that file passes the check while the
    hook reports success.

    The assertions pin the whole message, not just the exit code. An earlier
    refactor let `read_lines` raise out of `main`, which kept the exit code
    and the words "could not be read" while replacing the report with a
    traceback. A test that checked only those two things passed anyway. The
    legible report is the point of this gate, so it is what gets asserted.
    """
    script_copy = install_lint_script(_SCRIPT, tmp_path / "scripts")

    dst = tmp_path / "foo.sh"
    dst.write_bytes(b"pip install numpy --quiet\n\xff\xfe not utf-8\n")

    result = subprocess.run(
        [sys.executable, str(script_copy), str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Traceback" not in result.stderr, (
        f"the gate must report, not crash:\n{result.stderr}"
    )
    assert "Files could not be read, so they were not checked:" in result.stderr
    assert "  foo.sh: UnicodeDecodeError:" in result.stderr
    assert (
        "1 file(s) skipped. A skipped file is an unchecked file, and this "
        "gate reports success only when every target was read."
    ) in result.stderr
