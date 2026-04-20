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
  python3 scripts/check-test-timeouts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    test_dir = repo_root / TEST_DIR
    if not test_dir.is_dir():
        return 0

    violations: list[str] = []
    for path in test_dir.rglob("*.py"):
        if path.name in SKIP_FILES:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, 1):
            match = TIMEOUT_PATTERN.search(line)
            if not match:
                continue
            value = float(match.group(1))
            if value <= THRESHOLD_S:
                continue
            # Look at the line itself + up to NEIGHBORHOOD_LINES above.
            start = max(0, lineno - 1 - NEIGHBORHOOD_LINES)
            neighborhood = lines[start:lineno]
            if any(JUSTIFIED_PATTERN.search(candidate) for candidate in neighborhood):
                continue
            rel = path.relative_to(repo_root).as_posix()
            violations.append(
                f"{rel}:{lineno}: @pytest.mark.timeout({value:g}) "
                f"exceeds {THRESHOLD_S}s threshold without a nearby "
                "'# justified: <reason>' comment"
            )

    if violations:
        print("Unjustified long test timeouts detected:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            f"\n{len(violations)} violation(s). Long timeouts often mask "
            "hangs or hide slow code that should be optimized. If the "
            "timeout is legitimately needed (scale test, benchmark), add "
            f"'# justified: <reason>' on or within {NEIGHBORHOOD_LINES} "
            "lines above the decorator.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
