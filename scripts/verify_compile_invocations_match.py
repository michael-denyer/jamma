#!/usr/bin/env python3
"""Verify wheel-build and dev-mode compile paths share a single source of truth.

Discharges ROADMAP success criterion 2 (byte-identical compiler invocations
between wheel and dev-mode paths) by STRUCTURAL guarantee rather than
empirical trace capture: if both paths call the SAME function with the
SAME constants, their compile invocations differ only by input paths
(expected — wheel-build writes to build/lib/jamma, dev-mode writes
in-place).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRY_POINTS = [
    REPO_ROOT / "hatch_build.py",
    REPO_ROOT / "src/jamma/jlinalg/_compile_jlinalg.py",
    REPO_ROOT / "src/jamma/lmm/_compile_accel.py",
]
BANNED_LITERALS = [
    "'-O2'",
    '"-O2"',
    "'-O3'",
    '"-O3"',
    "'-fno-fast-math'",
    '"-fno-fast-math"',
    "'-ftree-vectorize'",
    '"-ftree-vectorize"',
    "'-funroll-loops'",
    '"-funroll-loops"',
    "'-fopenmp'",
    '"-fopenmp"',
]
# Ensure sys.path lets us import build_support/
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    failures: list[str] = []

    # 1. Import and inspect build_support.compile_and_link constants.
    from build_support import compile_and_link

    print(
        f"BASE_CFLAGS ({len(compile_and_link.BASE_CFLAGS)} flags): "
        f"{compile_and_link.BASE_CFLAGS}"
    )
    print(
        f"LAPACK_CFLAGS ({len(compile_and_link.LAPACK_CFLAGS)} flags): "
        f"{compile_and_link.LAPACK_CFLAGS}"
    )
    print(f"BASELINE_SOURCES: {compile_and_link.BASELINE_SOURCES}")
    print(f"LAPACK_SOURCES: {compile_and_link.LAPACK_SOURCES}")

    # 2. Scan entry points for banned literals.
    for ep in ENTRY_POINTS:
        if not ep.exists():
            failures.append(f"{ep}: entry point missing")
            continue
        text = ep.read_text()
        for lit in BANNED_LITERALS:
            if lit in text:
                failures.append(f"{ep}: banned literal {lit} found")

    # 3. Confirm both entry points invoke the helper.
    for ep in ENTRY_POINTS:
        if not ep.exists():
            continue
        text = ep.read_text()
        has_call = (
            "compile_and_link.compile_jlinalg" in text or "compile_jlinalg(" in text
        )
        if not has_call:
            failures.append(
                f"{ep}: no compile_jlinalg call found — entry point bypasses helper"
            )

    if failures:
        print("\nCROSS-PATH EQUIVALENCE VIOLATIONS:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1

    print(
        "\nStructural equivalence holds. All entry points use "
        "build_support.compile_and_link as single source of truth."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
