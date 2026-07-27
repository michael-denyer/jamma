#!/usr/bin/env python3
"""Enforce per-subsystem coverage thresholds on top of the global gate.

The global ``slipcover --fail-under 80`` in CI is a coarse signal. This
script parses slipcover's JSON output and applies stricter, subsystem-
specific thresholds for the high-risk areas (LMM, jlinalg, kinship) so
that a regression in critical numerical code can't be masked by
over-tested utility modules.

Subsystem thresholds are intentionally below the *current* coverage of
each subsystem so the gate enforces a floor without becoming a ratchet
that drives test-padding for its own sake. Bump them after meaningful
test additions, not after every measurement.

Usage::

    slipcover --source src/jamma --json coverage.json -m pytest ...
    python scripts/check_subsystem_coverage.py coverage.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict

# Per-subsystem floor (percent line coverage).
#
# Floors, not ratchets: set BELOW the LOWEST observed coverage across both
# CI (Linux) and dev (macOS) so the gate alarms on regressions without
# false-failing on platform-specific BLAS-path differences.
#
# Reference measurements (2026-04-29):
#                       Linux-CI   macOS-Accelerate
#   src/jamma/lmm/      84.2%      82.3%
#   src/jamma/jlinalg/  21.8%      33.6%   (vendor-LAPACK paths differ by platform)
#   src/jamma/kinship/  55.8%      52.7%
#   src/jamma/io/       84.5%      84.5%
#
# Floors are set below the minimum of each pair. Bumping is deliberate.
SUBSYSTEM_THRESHOLDS: tuple[tuple[str, float], ...] = (
    # ``prefix`` is matched against file paths reported by slipcover
    # (relative to the project root, forward slashes).
    ("src/jamma/lmm/", 80.0),
    ("src/jamma/jlinalg/", 18.0),
    ("src/jamma/kinship/", 50.0),
    ("src/jamma/io/", 80.0),
)


class _FileCoverage(TypedDict, total=False):
    """The slipcover per-file fields this script reads.

    total=False because slipcover omits either list when it is empty.
    """

    executed_lines: list[int]
    missing_lines: list[int]


def _coverage_for_prefix(
    files: dict[str, _FileCoverage], prefix: str
) -> tuple[int, int]:
    """Aggregate executed/missing line counts across files under ``prefix``."""
    executed = 0
    missing = 0
    matched = False
    for path, stats in files.items():
        if not path.startswith(prefix):
            continue
        matched = True
        executed_lines = stats.get("executed_lines") or []
        missing_lines = stats.get("missing_lines") or []
        executed += len(executed_lines)
        missing += len(missing_lines)
    if not matched:
        return (0, 0)
    return (executed, missing)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write(
            "usage: check_subsystem_coverage.py <coverage.json>\n"
            "Generate the input with `slipcover --json coverage.json`.\n"
        )
        return 2
    json_path = Path(argv[0])
    if not json_path.exists():
        sys.stderr.write(f"missing coverage report: {json_path}\n")
        return 2

    with json_path.open() as f:
        report = json.load(f)
    files = report.get("files") or {}
    if not files:
        sys.stderr.write(
            "coverage report has no 'files' section — slipcover schema "
            "drift? aborting.\n"
        )
        return 2

    failures: list[str] = []
    for prefix, threshold in SUBSYSTEM_THRESHOLDS:
        executed, missing = _coverage_for_prefix(files, prefix)
        total = executed + missing
        if total == 0:
            failures.append(
                f"  {prefix!r}: no source files in coverage report — "
                f"check that --source includes this subsystem."
            )
            continue
        pct = 100.0 * executed / total
        status = "OK " if pct >= threshold else "FAIL"
        line = (
            f"  [{status}] {prefix:<28} {pct:5.1f}%  "
            f"(threshold {threshold:.1f}%, {executed}/{total} lines)"
        )
        sys.stdout.write(line + "\n")
        if pct < threshold:
            failures.append(line.strip())

    if failures:
        sys.stderr.write(
            "\nSubsystem coverage check FAILED:\n"
            + "\n".join(f"  - {line}" for line in failures)
            + "\n"
        )
        return 1
    sys.stdout.write("\nAll subsystems meet their coverage thresholds.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
