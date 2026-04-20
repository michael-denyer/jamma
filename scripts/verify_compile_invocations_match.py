#!/usr/bin/env python3
"""Verify wheel-build and dev-mode compile paths share a single source of truth.

Structural guarantee (not empirical trace capture): if the three ENTRY_POINTS
all call ``jamma._build_support.compile_and_link.compile_jlinalg`` with the
shared constants, their compile invocations differ only by input paths (wheel
writes to build/lib/jamma, dev-mode writes in-place).

Note on scope: ``src/jamma/core/recompile.py`` is deliberately EXCLUDED from
ENTRY_POINTS. It is a thin import-retry shim that delegates to
``_compile_jlinalg`` / ``_compile_accel`` rather than invoking the helper
directly, so it cannot satisfy the ``compile_jlinalg(`` assertion below.
The complementary ``check-compile-flag-literals.py`` lint DOES cover
recompile.py for bare flag literals.
"""

from __future__ import annotations

import importlib.util
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


def _load_compile_and_link():
    """Load compile_and_link by file path so we don't trigger
    jamma/__init__.py (which needs the package installed for its
    importlib.metadata.version("jamma") call — not true under pre-commit's
    isolated Python).
    """
    path = REPO_ROOT / "src/jamma/_build_support/compile_and_link.py"
    mod_name = "_verify_compile_and_link"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    # Register on sys.modules BEFORE exec_module so @dataclass can look the
    # module up in sys.modules while the class is being processed (Python
    # 3.14+ raises AttributeError otherwise).
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    failures: list[str] = []

    # 1. Import and inspect compile_and_link constants.
    compile_and_link = _load_compile_and_link()

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
        "jamma._build_support.compile_and_link as single source of truth."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
