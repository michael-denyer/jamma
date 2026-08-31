#!/usr/bin/env python3
"""Verify every compile entry point routes through the shared build driver.

Structural guarantee (not empirical trace capture): if the three ENTRY_POINTS
all reach ``jamma._build_support.compile_and_link.run_build`` — directly, or
through ``compile_and_link.compile_extension``, which is itself one
``run_build`` call — their compile invocations differ only by the
``BuildSpec`` they pass and the package directory they build in. The flags,
source lists, and the twelve preflight steps all live in one place.

This lint used to also scan the entry points for bare compile-flag literals.
That duplicated ``check_compile_flag_literals.py`` (which owns the check and
carries the ``# allow-compile-flag-literal`` escape hatch), so the scan was
dropped. The remaining AST check resolves only bindings imported from the
shared facade, plus the equivalent isolated-build loader binding; a same-named
local function cannot satisfy it.

``src/jamma/core/recompile.py`` is deliberately EXCLUDED from ENTRY_POINTS. It
is an import-retry shim that delegates to ``compile_and_link.compile_extension``
rather than driving ``run_build`` itself.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
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
    package_name = "_verify_build_support"
    for loaded in tuple(sys.modules):
        if loaded == package_name or loaded.startswith(f"{package_name}."):
            sys.modules.pop(loaded)
    package = types.ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    sys.modules[package_name] = package

    for sibling in ("build_models", "build_execution"):
        sibling_path = path.with_name(f"{sibling}.py")
        if not sibling_path.exists():
            continue
        sibling_name = f"{package_name}.{sibling}"
        sibling_spec = importlib.util.spec_from_file_location(
            sibling_name, str(sibling_path)
        )
        if sibling_spec is None or sibling_spec.loader is None:
            raise ImportError(f"could not load {sibling_path}")
        sibling_module = importlib.util.module_from_spec(sibling_spec)
        sys.modules[sibling_name] = sibling_module
        sibling_spec.loader.exec_module(sibling_module)

    mod_name = f"{package_name}.compile_and_link"
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


_DRIVER_NAMES = frozenset({"run_build", "compile_extension"})
_FACADE_MODULE = "jamma._build_support.compile_and_link"


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> set[str]:
    """Return simple names rebound by a top-level assignment."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}


def _loads_isolated_facade(value: ast.expr) -> bool:
    """Recognize hatch_build.py's file-path load of compile_and_link.py."""
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return False
    if value.func.id != "_load_build_support_module":
        return False
    strings = {
        arg.value
        for arg in value.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    return "compile_and_link.py" in strings


def _has_run_build_call(source: str) -> bool:
    """Return whether ``source`` calls a driver bound from the shared facade.

    Recognized bindings are exact imports from
    ``jamma._build_support.compile_and_link`` and hatch_build.py's equivalent
    file-path-loaded module. Uses AST inspection rather than substring or
    callee-name matching, so the following do NOT count:

      - a mention inside a comment (e.g. ``# run_build(x)``)
      - a mention inside a string literal or docstring
      - a local ``def run_build(...):`` that calls itself
      - ``some_unrelated_object.compile_extension(...)``
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    module_bindings: set[str] = set()
    driver_bindings: set[str] = set()

    # Resolve top-level facade bindings. Rebinding a verified name invalidates
    # it unless the new value is itself another verified facade binding.
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                local = alias.asname or alias.name
                module_bindings.discard(local)
                driver_bindings.discard(local)
                if statement.module == _FACADE_MODULE and alias.name in _DRIVER_NAMES:
                    driver_bindings.add(local)
            continue

        if isinstance(statement, ast.Import):
            for alias in statement.names:
                local = alias.asname or alias.name.split(".")[0]
                module_bindings.discard(local)
                driver_bindings.discard(local)
                if alias.name == _FACADE_MODULE and alias.asname:
                    module_bindings.add(local)
            continue

        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            assigned = _assigned_names(statement)
            module_bindings.difference_update(assigned)
            driver_bindings.difference_update(assigned)
            value = statement.value
            if value is None:
                continue
            if _loads_isolated_facade(value):
                module_bindings.update(assigned)
                continue
            if (
                isinstance(value, ast.Attribute)
                and value.attr in _DRIVER_NAMES
                and isinstance(value.value, ast.Name)
                and value.value.id in module_bindings
            ):
                driver_bindings.update(assigned)
            continue

        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_bindings.discard(statement.name)
            driver_bindings.discard(statement.name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in driver_bindings:
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _DRIVER_NAMES
            and isinstance(func.value, ast.Name)
            and func.value.id in module_bindings
        ):
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
