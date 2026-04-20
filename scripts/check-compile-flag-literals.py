#!/usr/bin/env python3
"""Ban bare compile-flag literals outside jamma._build_support.

Single source of truth for compiler flags is
`src/jamma/_build_support/compile_and_link.py`.
Any flag literal in the four compile entry points means someone duplicated a
flag again — that's exactly the footgun Phase 123 is preventing.

Scope: this is a drift-catcher for honest copy-paste, NOT defense-in-depth.
Known bypasses (documented as xfail tests in
tests/test_check_compile_flag_literals.py): explicit string concat
(``"-O" + "3"``), f-string interpolation (``f"-O{level}"``), implicit
adjacent-string concat (``"-O" "3"``). Anyone deliberately evading the lint
wanted to; that's a code-review problem, not a regex problem.

Deliberate exception: a line ending with ``# allow-compile-flag-literal``
(plus optional rationale) is skipped. Reserved for the one case where
divergence is intentional — ``-march=native`` in ``_compile_accel.py``
must NOT move to ``_build_support`` because wheels target the lowest
common denominator and dev builds target the local CPU. The escape
hatch keeps the lint strict for everything else.

Target files:
  - hatch_build.py
  - src/jamma/jlinalg/_compile_jlinalg.py
  - src/jamma/lmm/_compile_accel.py
  - src/jamma/core/recompile.py  (runtime module must stay clean)

Flag set (the ones we've actually seen duplicated, plus portability
footguns like ``-march=native`` that MUST stay dev-only per
CLAUDE.md):
  -O0 / -O1 / -O2 / -O3 / -ftree-vectorize / -fno-fast-math /
  -fno-math-errno / -fno-trapping-math / -fno-finite-math-only /
  -funroll-loops / -fopenmp / -march=native / -mtune=native /
  -std=c99 / -std=c11 / -std=c17 / -shared / -pthread

Usage:
  python3 scripts/check-compile-flag-literals.py
  # exits 0 if clean, exits 1 with violations printed to stderr
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FLAGS: set[str] = {
    "-O0",
    "-O1",
    "-O2",
    "-O3",
    "-ftree-vectorize",
    "-fno-fast-math",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-fno-finite-math-only",
    "-funroll-loops",
    "-fopenmp",
    # Portability footguns — dev-only in _compile_accel.py, never in wheels.
    "-march=native",
    "-mtune=native",
    # C std flags: leak between dev/wheel builds if hardcoded.
    "-std=c99",
    "-std=c11",
    "-std=c17",
    # Link-step flags that belong in LINK_FLAGS_BY_PLATFORM.
    "-shared",
    "-pthread",
}

TARGETS: list[str] = [
    "hatch_build.py",
    "src/jamma/jlinalg/_compile_jlinalg.py",
    "src/jamma/lmm/_compile_accel.py",
    "src/jamma/core/recompile.py",
]

# Match a flag literal inside single or double quotes: '-O3' or "-O3".
# Covers compile-style flags starting with -O, -f, -W, -m (march/mtune),
# -s (shared, std), or -p (pthread). The FLAGS allowlist below is the
# real filter — this pattern is just a cheap pre-filter to skip
# obviously-not-a-flag strings. The leading `-` immediately after the
# opening quote keeps paths like "/usr/lib/-O3-test/foo" from matching.
FLAG_PATTERN = re.compile(r"""["'](-[OfWmsp][A-Za-z0-9_=-]*)["']""")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    violations: list[str] = []

    for target in TARGETS:
        path = repo_root / target
        if not path.exists():
            # Missing file is a separate problem — the cleanup plan may have
            # deleted something the lint expected. Surface it as a violation.
            violations.append(
                f"{target}: target file missing — "
                f"check-compile-flag-literals.py needs updating"
            )
            continue

        source_lines = path.read_text().splitlines()
        for lineno, line in enumerate(source_lines, 1):
            # Skip pure-comment lines (but not inline comments — a literal on
            # a code line followed by `# comment` is still a violation).
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Deliberate-divergence escape hatch. Accept the marker inline
            # on the same line, or on the immediately preceding comment
            # line (useful when the literal is long enough that ruff-format
            # would split the inline comment to its own line).
            if "allow-compile-flag-literal" in line:
                continue
            if lineno >= 2:
                prev = source_lines[lineno - 2].lstrip()
                if prev.startswith("#") and "allow-compile-flag-literal" in prev:
                    continue
            for match in FLAG_PATTERN.finditer(line):
                flag = match.group(1)
                if flag in FLAGS:
                    violations.append(
                        f"{target}:{lineno}: bare compile-flag literal {flag!r} — "
                        f"add flags to src/jamma/_build_support/"
                        f"compile_and_link.py instead"
                    )

    if violations:
        print("Compile-flag drift detected:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            f"\n{len(violations)} violation(s). See docstring in "
            f"scripts/check-compile-flag-literals.py for rationale.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
