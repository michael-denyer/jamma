#!/usr/bin/env python3
"""Ban bare compile-flag literals outside build_support/.

Single source of truth for compiler flags is `build_support/compile_and_link.py`.
Any flag literal in the four compile entry points means someone duplicated a
flag again — that's exactly the footgun Phase 123 is preventing.

Target files:
  - hatch_build.py
  - src/jamma/jlinalg/_compile_jlinalg.py
  - src/jamma/lmm/_compile_accel.py
  - src/jamma/core/recompile.py  (per D-04 — runtime module must stay clean)

Flag set (the ones we've actually seen duplicated):
  -O0 / -O1 / -O2 / -O3 / -ftree-vectorize / -fno-fast-math /
  -fno-math-errno / -fno-trapping-math / -fno-finite-math-only /
  -funroll-loops / -fopenmp

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
}

TARGETS: list[str] = [
    "hatch_build.py",
    "src/jamma/jlinalg/_compile_jlinalg.py",
    "src/jamma/lmm/_compile_accel.py",
    "src/jamma/core/recompile.py",
]

# Match a flag literal inside single or double quotes: '-O3' or "-O3".
# We specifically look for compile-style flags (start with -O, -f, or -W).
# The broader pattern avoids false positives for strings like "-version" or
# shell flags that happen to start with "-f" (e.g. "-file").
FLAG_PATTERN = re.compile(r"""["'](-[OfW][A-Za-z0-9_=-]*)["']""")


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

        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            # Skip pure-comment lines (but not inline comments — a literal on
            # a code line followed by `# comment` is still a violation).
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            for match in FLAG_PATTERN.finditer(line):
                flag = match.group(1)
                if flag in FLAGS:
                    violations.append(
                        f"{target}:{lineno}: bare compile-flag literal {flag!r} — "
                        f"add flags to build_support/compile_and_link.py instead"
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
