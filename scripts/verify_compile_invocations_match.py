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

import ast
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


def _load_compile_and_link(path: Path):
    """Load compile_and_link by file path so we don't trigger
    jamma/__init__.py (which needs the package installed for its
    importlib.metadata.version("jamma") call — not true under pre-commit's
    isolated Python).
    """
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


def _has_compile_jlinalg_call(source: str) -> bool:
    """Return True iff ``source`` has a real call to ``compile_jlinalg``.

    Uses AST inspection rather than substring matching, so the following
    do NOT count as calls:

      - a mention inside a comment (e.g. ``# compile_jlinalg(x)``)
      - a mention inside a string literal or docstring
      - a local ``def compile_jlinalg(...):`` with the same name

    A call matches if the callee is ``compile_jlinalg`` (bare name) or
    ``<anything>.compile_jlinalg`` (attribute access, e.g.
    ``compile_and_link.compile_jlinalg(...)``).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "compile_jlinalg":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "compile_jlinalg":
            return True
    return False


def check(
    build_support_path: Path,
    entry_points: list[Path],
    banned_literals: list[str] = BANNED_LITERALS,
) -> tuple[int, list[str]]:
    """Run the equivalence check. Returns (exit_code, failures).

    Parameterized so tests can point it at synthetic trees.
    """
    failures: list[str] = []

    # 1. Import and inspect compile_and_link constants. The import must
    # succeed — if it fails the helper tree is broken and downstream
    # checks are meaningless.
    compile_and_link = _load_compile_and_link(build_support_path)

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

    # 2. Scan entry points for banned literals and confirm a real call.
    for ep in entry_points:
        if not ep.exists():
            failures.append(f"{ep}: entry point missing")
            continue
        text = ep.read_text()
        for lit in banned_literals:
            if lit in text:
                failures.append(f"{ep}: banned literal {lit} found")
        if not _has_compile_jlinalg_call(text):
            failures.append(
                f"{ep}: no compile_jlinalg call found — entry point bypasses helper"
            )

    if failures:
        print("\nCROSS-PATH EQUIVALENCE VIOLATIONS:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1, failures

    print(
        "\nStructural equivalence holds. All entry points use "
        "jamma._build_support.compile_and_link as single source of truth."
    )
    return 0, failures


def main() -> int:
    build_support_path = REPO_ROOT / "src/jamma/_build_support/compile_and_link.py"
    exit_code, _ = check(build_support_path, ENTRY_POINTS)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
