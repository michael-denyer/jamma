"""Tests for scripts/check-doc-anchors.py.

The lint exists because 53 anchors in docs/CODEMAP.md rotted unnoticed before
7.2.0. A lint that passes on a clean tree but cannot detect the rot it was
written for is worse than none, so most of these drive it against a fake repo
carrying a known-stale anchor and assert that it fails and says why.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check-doc-anchors.py"

# alpha() is on line 4 and Beta on line 8. The tests below anchor at those.
_MODULE = '''"""A stub module."""


def alpha() -> int:
    return 1


class Beta:
    pass
'''


def _run(tmp_path: Path, doc_body: str) -> subprocess.CompletedProcess[str]:
    """Build a miniature git repo around ``doc_body`` and run the lint inside it.

    A real repo with the doc committed, because the lint enumerates its inputs
    with ``git ls-files``. That is what keeps it from carrying a third copy of the
    ignore list that .markdownlint-cli2.jsonc and lychee.toml both hold, and it
    means these tests exercise the same enumeration the pre-commit hook does.
    """
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_SCRIPT, scripts_dir / _SCRIPT.name)

    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "stub.py").write_text(_MODULE)

    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "MAP.md").write_text(doc_body)

    for args in (
        ["init"],
        # A committer identity is not configured in CI's checkout, and `add`
        # alone does not need one; ls-files reads the index, so no commit.
        ["add", "docs/MAP.md", "src/stub.py"],
    ):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            capture_output=True,
            text=True,
            check=True,
        )

    return subprocess.run(
        [sys.executable, str(scripts_dir / _SCRIPT.name)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_anchor_on_the_right_line_passes(tmp_path: Path) -> None:
    result = _run(tmp_path, "| `alpha()` | [stub.py:4](../src/stub.py#L4) |\n")
    assert result.returncode == 0, result.stdout + result.stderr


def test_anchor_on_the_wrong_line_fails_and_names_the_real_one(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, "| `alpha()` | [stub.py:9](../src/stub.py#L9) |\n")
    assert result.returncode == 1
    assert "`alpha` is at ../src/stub.py#L4, not #L9" in result.stdout


def test_symbol_absent_from_the_file_fails(tmp_path: Path) -> None:
    result = _run(tmp_path, "| `gamma()` | [stub.py:4](../src/stub.py#L4) |\n")
    assert result.returncode == 1
    assert "`gamma` is not defined anywhere" in result.stdout


def test_line_past_end_of_file_fails(tmp_path: Path) -> None:
    result = _run(tmp_path, "| `alpha()` | [stub.py:900](../src/stub.py#L900) |\n")
    assert result.returncode == 1
    assert "past the end of the file" in result.stdout


def test_missing_target_file_fails(tmp_path: Path) -> None:
    result = _run(tmp_path, "| `alpha()` | [gone.py:4](../src/gone.py#L4) |\n")
    assert result.returncode == 1
    assert "does not exist" in result.stdout


def test_class_anchor_resolves(tmp_path: Path) -> None:
    result = _run(tmp_path, "| `Beta` | [stub.py:8](../src/stub.py#L8) |\n")
    assert result.returncode == 0, result.stdout + result.stderr


def test_unnamed_anchor_must_still_land_on_a_definition(tmp_path: Path) -> None:
    """Quick-Navigation rows label the link with a filename, not a symbol.

    There is no symbol to match, so the weaker check applies: the line has to
    define something. Line 5 is ``return 1``.
    """
    result = _run(tmp_path, "| Entry point | [stub.py:5](../src/stub.py#L5) |\n")
    assert result.returncode == 1
    assert "is not a definition" in result.stdout


def test_unnamed_anchor_on_a_definition_passes(tmp_path: Path) -> None:
    result = _run(tmp_path, "| Entry point | [stub.py:4](../src/stub.py#L4) |\n")
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_filename_in_backticks_is_not_read_as_a_symbol(tmp_path: Path) -> None:
    """``stub.py`` would otherwise be mistaken for a symbol named ``py``."""
    result = _run(
        tmp_path,
        "| `stub.py` rows | [`alpha`](../src/stub.py#L4) |\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_keyword_in_backticks_is_not_read_as_a_symbol(tmp_path: Path) -> None:
    """A row saying "not a bare ``assert``" names no symbol.

    Falling back to the weaker definition check is right here; guessing
    ``assert`` and reporting it missing is a false positive.
    """
    result = _run(
        tmp_path,
        "| uses a guard, not bare `assert` | [stub.py:4](../src/stub.py#L4) |\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_real_repository_docs_are_clean() -> None:
    """The lint must pass on the tree it ships with, or it cannot gate."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
