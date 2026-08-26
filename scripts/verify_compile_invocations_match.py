#!/usr/bin/env python3
"""Verify every compile entry point routes through the shared build driver.

Structural guarantee (not empirical trace capture): if the three ENTRY_POINTS
all reach ``jamma._build_support.compile_and_link.run_build``, their compile
invocations differ only by the ``BuildSpec`` they pass and the package
directory they build in — the flags, source lists, and the twelve preflight
steps all live in one place.

This lint used to also scan the entry points for bare compile-flag literals.
That duplicated ``check_compile_flag_literals.py`` (which owns the check and
carries the ``# allow-compile-flag-literal`` escape hatch), so the scan was
dropped; the AST "calls ``run_build``" check is all that remains here.

``src/jamma/core/recompile.py`` is deliberately EXCLUDED from ENTRY_POINTS. It
is an import-retry shim that delegates to ``_compile_jlinalg`` / ``_compile_accel``
rather than driving ``run_build`` itself.
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


def _has_run_build_call(source: str) -> bool:
    """Return True iff ``source`` has a real call to ``run_build``.

    Uses AST inspection rather than substring matching, so the following
    do NOT count as calls:

      - a mention inside a comment (e.g. ``# run_build(x)``)
      - a mention inside a string literal or docstring
      - a local ``def run_build(...):`` with the same name

    A call matches if the callee is ``run_build`` (bare name) or
    ``<anything>.run_build`` (attribute access).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "run_build":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "run_build":
            return True
    return False


def check(
    build_support_path: Path,
    entry_points: list[Path],
) -> tuple[int, list[str]]:
    """Run the equivalence check. Returns (exit_code, failures).

    Parameterized so tests can point it at synthetic trees.
    """
    failures: list[str] = []

    # 1. Import compile_and_link and confirm the shared driver is present. The
    # import must succeed — if it fails the helper tree is broken and downstream
    # checks are meaningless.
    compile_and_link = _load_compile_and_link(build_support_path)
    if not hasattr(compile_and_link, "run_build"):
        failures.append(f"{build_support_path}: run_build not found in helper")

    # 2. Confirm every entry point routes through run_build.
    for ep in entry_points:
        if not ep.exists():
            failures.append(f"{ep}: entry point missing")
            continue
        if not _has_run_build_call(ep.read_text()):
            failures.append(
                f"{ep}: no run_build call found — entry point bypasses the "
                "shared build driver"
            )

    if failures:
        print("\nCROSS-PATH EQUIVALENCE VIOLATIONS:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1, failures

    print(
        "Structural equivalence holds. All entry points route through "
        "jamma._build_support.compile_and_link.run_build."
    )
    return 0, failures


def main() -> int:
    build_support_path = REPO_ROOT / "src/jamma/_build_support/compile_and_link.py"
    exit_code, _ = check(build_support_path, ENTRY_POINTS)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
