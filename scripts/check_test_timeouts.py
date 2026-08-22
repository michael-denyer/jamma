#!/usr/bin/env python3
"""Flag unjustified long ``@pytest.mark.timeout(N)`` values.

CLAUDE.md sets ``--timeout=30`` as the default in ``pyproject.toml``
addopts. Individual tests may legitimately need more (tier2 scale tests,
benchmark warm-ups), but a timeout value above THRESHOLD_S without a
visible justification usually means someone bumped it to paper over a
hang rather than fixing the underlying slow code.

Rule: ``@pytest.mark.timeout(N)`` with ``N > THRESHOLD_S`` is a violation
unless the same line OR one of the ~3 lines above it contains the word
``justified`` in a comment. (Searching nearby rather than only the same
line accommodates ruff-format line splitting.)

Usage:
  python3 scripts/check_test_timeouts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _lint_common import (
    allowed,
    display_path,
    read_batch,
    repo_root,
    report,
    report_unreadable,
)

THRESHOLD_S: int = 120
TEST_DIR = Path("tests")

# The lint's own test file unavoidably embeds long-timeout strings as
# subprocess input; skip it to avoid circular false positives.
SKIP_FILES: frozenset[str] = frozenset(
    {"test_check_test_timeouts.py"},
)

TIMEOUT_PATTERN = re.compile(r"@pytest\.mark\.timeout\(\s*(\d+(?:\.\d+)?)\s*[,\)]")
JUSTIFIED_PATTERN = re.compile(r"#\s*justified\b", re.IGNORECASE)
NEIGHBORHOOD_LINES = 3


def scan_line(line: str) -> float | None:
    """Return the timeout on this line when it exceeds the threshold."""
    match = TIMEOUT_PATTERN.search(line)
    if match is None:
        return None
    value = float(match.group(1))
    return value if value > THRESHOLD_S else None


def main() -> int:
    root = repo_root()
    test_dir = root / TEST_DIR
    if not test_dir.is_dir():
        return 0

    targets = [p for p in sorted(test_dir.rglob("*.py")) if p.name not in SKIP_FILES]
    lines_by_path, unreadable = read_batch(targets, root=root)

    violations: list[str] = []
    for path, lines in lines_by_path.items():
        rel = display_path(path, root)
        for i, line in enumerate(lines):
            value = scan_line(line)
            if value is None:
                continue
            if allowed(lines, i, JUSTIFIED_PATTERN, window=NEIGHBORHOOD_LINES):
                continue
            violations.append(
                f"{rel}:{i + 1}: @pytest.mark.timeout({value:g}) "
                f"exceeds {THRESHOLD_S}s threshold without a nearby "
                "'# justified: <reason>' comment"
            )

    skipped = report_unreadable(unreadable)
    found = report(
        "Unjustified long test timeouts detected:",
        violations,
        f"{len(violations)} violation(s). Long timeouts often mask "
        "hangs or hide slow code that should be optimized. If the "
        "timeout is legitimately needed (scale test, benchmark), add "
        f"'# justified: <reason>' on or within {NEIGHBORHOOD_LINES} "
        "lines above the decorator.",
    )
    return max(skipped, found)


if __name__ == "__main__":
    sys.exit(main())
