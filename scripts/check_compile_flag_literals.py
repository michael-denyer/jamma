#!/usr/bin/env python3
"""Ban bare compile-flag literals outside jamma._build_support.

Single source of truth for compiler flags is
`src/jamma/_build_support/build_models.py`.
Any flag literal in the four compile entry points means someone duplicated a
flag again — that's exactly the footgun this lint prevents.

Scope: this is a drift-catcher for honest copy-paste, NOT defense-in-depth.
Known bypasses (documented as xfail tests in
tests/test_check_compile_flag_literals.py): explicit string concat
(``"-O" + "3"``), f-string interpolation (``f"-O{level}"``), implicit
adjacent-string concat (``"-O" "3"``). Anyone deliberately evading the lint
wanted to; that's a code-review problem, not a regex problem.

Deliberate exception: a line ending with ``# allow-compile-flag-literal``
(plus optional rationale) is skipped. No entry point uses it today —
``-march=native`` lives in ``LMM_ACCEL_SPEC.dev_extra_cflags`` in
``_build_support/build_models.py``, applied only on the dev rebuild path
and never on the wheel path, so wheels stay portable without any entry point
needing its own escape hatch. The hatch remains for a future case where
divergence between entry points is genuinely intentional.

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
  -std=c99 / -std=c11 / -std=c17 / -shared / -pthread /
  -fsanitize=address / -fsanitize=undefined /
  -fsanitize=address,undefined / -fno-omit-frame-pointer / -shared-libasan

Usage:
  python3 scripts/check_compile_flag_literals.py
  # exits 0 if clean, exits 1 with violations printed to stderr
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
    # Sanitizer flags. Must NEVER appear in the four entry
    # points; they are assembled by
    # jamma._build_support.compile_and_link.apply_sanitizer_overrides()
    # and reach hatch_build.py / _compile_jlinalg.py / _compile_accel.py /
    # core/recompile.py via the existing extra_cflags / extra_link_flags /
    # extra_lapack_cflags machinery.
    # The FLAG_PATTERN regex above already covers -f and -s prefixes, so no
    # regex change is needed for these additions — only the FLAGS set.
    "-fsanitize=address",
    "-fsanitize=undefined",
    "-fsanitize=address,undefined",  # combined form — most-likely copy-paste shape
    "-fno-omit-frame-pointer",
    "-shared-libasan",  # only used if someone switches to clang for ASAN
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
# Comma is in the body class so the combined sanitizer form
# "-fsanitize=address,undefined" matches as a single flag.
FLAG_PATTERN = re.compile(r"""["'](-[OfWmsp][A-Za-z0-9_=,-]*)["']""")

ALLOW_MARKER = "allow-compile-flag-literal"


def scan_line(line: str) -> list[str]:
    """Return the banned flag literals quoted on one line of source.

    Pure-comment lines are exempt: documentation and rationale mention
    ``-O3`` constantly. An inline comment after a literal is not, because
    the literal on that line is still real.
    """
    if line.lstrip().startswith("#"):
        return []
    return [m.group(1) for m in FLAG_PATTERN.finditer(line) if m.group(1) in FLAGS]


def main() -> int:
    root = repo_root()
    violations: list[str] = []

    present: list[Path] = []
    for target in TARGETS:
        path = root / target
        if path.exists():
            present.append(path)
        else:
            # Missing file is a separate problem — the cleanup plan may have
            # deleted something the lint expected. Surface it as a violation.
            violations.append(
                f"{target}: target file missing — "
                f"check_compile_flag_literals.py needs updating"
            )

    lines_by_path, unreadable = read_batch(present, root=root)
    for path, lines in lines_by_path.items():
        target = display_path(path, root)
        for i, line in enumerate(lines):
            # Deliberate-divergence escape hatch. Inline on the offending
            # line, or on the comment line immediately above it — ruff-format
            # splits a long inline comment up there. Only a *comment* line
            # above counts; an inline marker covers its own line alone.
            above_is_comment = i > 0 and lines[i - 1].lstrip().startswith("#")
            if allowed(lines, i, ALLOW_MARKER, window=1 if above_is_comment else 0):
                continue
            for flag in scan_line(line):
                violations.append(
                    f"{target}:{i + 1}: bare compile-flag literal {flag!r} — "
                    f"add flags to src/jamma/_build_support/"
                    f"build_models.py instead"
                )

    skipped = report_unreadable(unreadable)
    found = report(
        "Compile-flag drift detected:",
        violations,
        f"{len(violations)} violation(s). See docstring in "
        f"scripts/check_compile_flag_literals.py for rationale.",
    )
    return max(skipped, found)


if __name__ == "__main__":
    sys.exit(main())
