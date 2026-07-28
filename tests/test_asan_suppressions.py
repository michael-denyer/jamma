"""Tests for scripts/asan-suppressions.txt.

Asserts the suppression file contains ONLY documented third-party noise
patterns and never a `leak:jamma_*` / `leak:_jlinalg*` / `leak:_lmm_accel*`
suppression. A blanket in-repo suppression would silently swallow real
bugs in our C code — see scripts/asan-suppressions.txt header comment.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SUPPRESSIONS = _REPO_ROOT / "scripts" / "asan-suppressions.txt"


def test_suppression_file_exists():
    assert _SUPPRESSIONS.exists(), (
        f"{_SUPPRESSIONS} missing — sanitizers.yml workflow references it"
    )


def test_no_in_repo_suppressions():
    """No `leak:jamma_*`, `leak:_jlinalg*`, or `leak:_lmm_accel*` entries.

    A suppression matching JAMMA-internal symbols defeats the workflow's
    primary purpose (catching leaks in our C code). Caught at lint time.
    """
    forbidden = re.compile(r"^leak:(jamma|_jlinalg|_lmm_accel)")
    violations = []
    for lineno, line in enumerate(_SUPPRESSIONS.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if forbidden.match(stripped):
            violations.append(f"line {lineno}: {stripped!r}")
    assert not violations, "in-repo LSAN suppressions found:\n  " + "\n  ".join(
        violations
    )


def test_no_blanket_wildcard_suppressions():
    """`leak:*` or `leak:Py*` would suppress everything — never allowed."""
    blanket = re.compile(r"^leak:\*$|^leak:Py\*$")
    for lineno, line in enumerate(_SUPPRESSIONS.read_text().splitlines(), 1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            assert not blanket.match(stripped), (
                f"line {lineno}: blanket suppression {stripped!r} forbidden"
            )


def test_every_group_has_source_citation():
    """Each leak: entry must inherit a citation from the most recent comment
    block above it (https://, GH#, CPython#, NumPy#, or TODO).

    A "group" is the run of `#` lines preceding the first leak: in a cluster.
    Subsequent leak: entries in the same cluster inherit that block's
    citation — walking backward past intermediate leak: lines is allowed
    until either (a) a comment block, or (b) the file's start, is reached.

    TODO is allowed because the loguru entry uses it (see header note);
    the first sanitizer-workflow run will resolve TODOs into real URLs.
    """
    citation = re.compile(r"https?://|GH#|CPython#|NumPy#|TODO")
    lines = _SUPPRESSIONS.read_text().splitlines()
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped.startswith("leak:"):
            continue
        # Walk backward through any combination of leak: entries and blanks
        # until we hit a contiguous run of `#` lines — that's this entry's
        # group comment block.
        block_lines: list[str] = []
        cursor = lineno - 2  # 0-based index of line ABOVE the leak: entry
        # Skip past intermediate leak: lines and blanks first.
        while cursor >= 0:
            prev = lines[cursor].strip()
            if prev.startswith("#"):
                break
            cursor -= 1
        # Now collect contiguous `#` lines.
        while cursor >= 0:
            prev = lines[cursor].strip()
            if not prev.startswith("#"):
                break
            block_lines.insert(0, prev)
            cursor -= 1
        block_text = "\n".join(block_lines)
        assert citation.search(block_text), (
            f"line {lineno} {stripped!r} has no upstream citation in its "
            f"preceding comment block:\n{block_text or '<no comments>'}"
        )


def test_every_non_comment_line_is_well_formed():
    """Every non-blank, non-comment line must be `leak:<symbol>` with a
    plausible C symbol body."""
    valid = re.compile(r"^leak:[A-Za-z_][A-Za-z0-9_]*$")
    for lineno, line in enumerate(_SUPPRESSIONS.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert valid.match(stripped), (
            f"line {lineno}: malformed entry {stripped!r} (expected leak:<symbol>)"
        )
