#!/usr/bin/env python3
"""Ban patching numerical functions and internal jamma collaborators in tests.

Enforces the boundary catalogue in docs/TESTING.md §2.2: ``@patch`` and
``MagicMock`` are allowed only at OS / hardware / process / external-UI
boundaries. Patching ``numpy.linalg.*``, BLAS routines, or jamma's own
numerical functions hides interface drift and silently masks real
numerical behaviour.

The gate walks each test module's ``ast`` and collects every patch call:

* ``patch("dotted.target")``, ``mock.patch(...)``, ``mocker.patch(...)``
* ``patch.object(<expr>, "attr")``, ``mocker.patch.object(...)``
* ``monkeypatch.setattr("dotted.target", ...)``
* ``monkeypatch.setattr(<expr>, "attr", ...)``

An ``<expr>`` first argument is resolved through the module's import
table, so ``patch.object(cn, "calc_pab")`` after
``import jamma.lmm.compute_numpy as cn`` names the same target as
``patch("jamma.lmm.compute_numpy.calc_pab")``. The target is then
canonicalised through the *source* module's import table: a test that
patches ``jamma.lmm.eigen.jlinalg.eigh`` is patching ``jamma.jlinalg.eigh``
at one of its import sites, and ``jamma.pipeline.eigendecompose_kinship``
is ``jamma.lmm.eigen.eigendecompose_kinship``. The policy table keys on
the canonical name, so it is about the thing being replaced, not the
spelling a test happened to reach it by.

Policy: a numerical module in `FORBIDDEN` may not be patched, as a whole
or attribute by attribute. ALL_CAPS attributes are configuration knobs
(``_CF_MAX_ITER``, ``_SAMPLED_SYMMETRY_THRESHOLD``) and are allowed.
The attributes in `SEAMS` are the documented capability seams a test drives
to select a dispatch path (``compute_numpy._accel`` to reach the NumPy
path, the ``jlinalg.blas_*`` detection flags) and are allowed.

Escape hatch: ``# allow-patch: <reason>`` on any line of the call, or in
the comment block directly above it (or above the ``with`` header the call
is an item of). Use it where delegation is the contract under
test (a dispatch spy) and say so in the reason.

Usage::

    python3 scripts/check_forbidden_patches.py            # repo-wide
    python3 scripts/check_forbidden_patches.py f1 f2      # specific files
    python3 scripts/check_forbidden_patches.py --list     # every resolved site
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from _lint_common import (
    LintReadError,
    allowed,
    display_path,
    repo_root,
    report,
    report_unreadable,
)

# Canonical module prefixes a test may not patch, with the attributes that
# are documented seams rather than numerical functions.
FORBIDDEN: tuple[tuple[str, str], ...] = (
    ("numpy.linalg", "Patches a NumPy numerical function. Use real synthetic data."),
    ("numpy.matmul", "Patches a NumPy numerical function. Use real synthetic data."),
    (
        "scipy",
        "Patches scipy. scipy is the test-only reference; patching it makes "
        "the reference circular.",
    ),
    ("jamma.jlinalg", "Patches a jlinalg numerical wrapper. Use real synthetic data."),
    ("jamma.lmm.likelihood", "Patches a likelihood function. Use synthetic data."),
    (
        "jamma.lmm.likelihood_numpy",
        "Patches a likelihood function. Use synthetic data.",
    ),
    (
        "jamma.lmm.compute_numpy",
        "Patches a compute function. Set _accel to None to reach the NumPy path.",
    ),
    ("jamma.lmm.uab", "Patches a Uab/Iab kernel. Use synthetic data."),
    ("jamma.lmm.special", "Patches a special function. Drive it with inputs."),
    (
        "jamma.lmm.prepare_common",
        "Patches null-model preparation. Drive it with inputs.",
    ),
    ("jamma.lmm.eigen", "Patches eigendecomposition. Use synthetic data."),
    ("jamma.kinship.compute", "Patches kinship computation. Use synthetic data."),
)

# Canonical attribute names that select a dispatch path rather than compute
# a number. Toggling them is how a test reaches the other path.
SEAMS: frozenset[str] = frozenset(
    {
        "jamma.lmm.compute_numpy._accel",
        "jamma.jlinalg.blas_has_dsyevd",
        "jamma.jlinalg.blas_has_dsyevr",
        "jamma.jlinalg.blas_has_dsyrk",
        "jamma.jlinalg.blas_has_dgemm",
        "jamma.jlinalg.blas_is_ilp64",
        "jamma.jlinalg.blas_backend",
        "jamma.jlinalg.set_n_threads",
        "jamma.jlinalg.get_n_threads",
    }
)

ALLOW_RE = re.compile(r"#\s*allow-patch\s*:\s*\S")
DEFAULT_GLOBS: tuple[str, ...] = ("tests/**/*.py",)


@dataclass(frozen=True)
class PatchSite:
    """One patch call in a test file, with its target resolved."""

    path: Path
    line: int
    end_line: int
    form: str
    target: str
    marker_from: int
    """First line an ``allow-patch`` marker may sit on: the line above the
    call, or above the ``with`` header the call is an item of."""

    @property
    def resolved(self) -> bool:
        return not self.target.startswith("?")


def _imports(tree: ast.AST, package: str) -> dict[str, str]:
    """Map every name an ``import`` binds, at any scope, to its dotted origin."""
    table: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                table[bound] = alias.name if alias.asname else bound
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".") if package else []
                parts = parts[: len(parts) - node.level + 1]
                base = ".".join(p for p in (*parts, base) if p)
            for alias in node.names:
                table[alias.asname or alias.name] = f"{base}.{alias.name}"
    return table


def _dotted(node: ast.expr, table: dict[str, str]) -> str | None:
    """Resolve a Name/Attribute chain through ``table``; None if the root is local."""
    if isinstance(node, ast.Name):
        return table.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, table)
        return None if base is None else f"{base}.{node.attr}"
    return None


class Canonicaliser:
    """Rewrite a dotted target through the import tables of the source tree.

    ``jamma.lmm.eigen.jlinalg.eigh`` splits into the longest module prefix
    on disk (``jamma.lmm.eigen``) and the rest; the first remaining name is
    looked up in that module's import table (``jlinalg`` -> ``jamma.jlinalg``)
    and the walk repeats until the name stops changing.
    """

    def __init__(self, src_root: Path) -> None:
        self.src_root = src_root
        self._tables: dict[str, dict[str, str] | None] = {}

    def _module_file(self, dotted: str) -> Path | None:
        rel = Path(*dotted.split("."))
        for candidate in (
            self.src_root / rel.with_suffix(".py"),
            self.src_root / rel / "__init__.py",
        ):
            if candidate.is_file():
                return candidate
        return None

    def _table(self, module: str) -> dict[str, str] | None:
        if module not in self._tables:
            path = self._module_file(module)
            if path is None:
                self._tables[module] = None
            else:
                tree = ast.parse(path.read_text(encoding="utf-8"))
                package = (
                    module if path.name == "__init__.py" else module.rpartition(".")[0]
                )
                self._tables[module] = _imports(tree, package)
        return self._tables[module]

    def canonical(self, dotted: str) -> str:
        seen: set[str] = set()
        while dotted not in seen:
            seen.add(dotted)
            parts = dotted.split(".")
            for cut in range(len(parts) - 1, 0, -1):
                table = self._table(".".join(parts[:cut]))
                if table is None:
                    continue
                origin = table.get(parts[cut])
                if origin is not None:
                    dotted = ".".join((origin, *parts[cut + 1 :]))
                break
        return dotted


def _is_patch_call(func: ast.expr) -> str | None:
    """Return the call form if ``func`` is a patch-like callable."""
    if isinstance(func, ast.Name) and func.id == "patch":
        return "patch"
    if isinstance(func, ast.Attribute):
        if func.attr == "patch":
            return "patch"
        if func.attr == "object" and _is_patch_call(func.value) == "patch":
            return "patch.object"
        if func.attr == "setattr":
            return "setattr"
    return None


def _target_of(call: ast.Call, form: str, table: dict[str, str]) -> str | None:
    """Return the dotted target of a patch call, ``?name.attr`` if unresolvable."""
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value if "." in first.value or form != "setattr" else None
    if form == "patch":
        return None
    if len(call.args) < 2:
        return None
    attr = call.args[1]
    if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
        return None
    base = _dotted(first, table)
    if base is None:
        return f"?{ast.unparse(first)}.{attr.value}"
    return f"{base}.{attr.value}"


def collect_sites(path: Path, source: str, canon: Canonicaliser) -> list[PatchSite]:
    """Every patch call in one test module, targets canonicalised."""
    tree = ast.parse(source, filename=str(path))
    table = _imports(tree, "")
    headers = [
        (node.lineno, node.body[0].lineno - 1)
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
    ]
    sites: list[PatchSite] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        form = _is_patch_call(node.func)
        if form is None:
            continue
        target = _target_of(node, form, table)
        if target is None:
            continue
        if not target.startswith("?"):
            target = canon.canonical(target)
        enclosing = [top for top, last in headers if top <= node.lineno <= last]
        marker_from = min([node.lineno, *enclosing]) - 1
        sites.append(
            PatchSite(
                path,
                node.lineno,
                node.end_lineno or node.lineno,
                form,
                target,
                marker_from,
            )
        )
    return sites


def classify(target: str) -> str | None:
    """Return the reason a canonical target is forbidden, or None."""
    if target in SEAMS or target.rsplit(".", 1)[-1].isupper():
        return None
    for prefix, reason in FORBIDDEN:
        if target == prefix or target.startswith(prefix + "."):
            return reason
    return None


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _iter_target_files(args: list[str], root: Path) -> list[Path]:
    if args:
        return [Path(a) for a in args if Path(a).suffix == ".py"]
    files: list[Path] = []
    for pattern in DEFAULT_GLOBS:
        files.extend(root.glob(pattern))
    return sorted(files)


def scan(
    files: list[Path], root: Path, src_root: Path
) -> tuple[list[PatchSite], list[str], list[str]]:
    """Return (all sites, violation lines, unreadable lines)."""
    canon = Canonicaliser(src_root)
    sites: list[PatchSite] = []
    violations: list[str] = []
    unreadable: list[str] = []
    for path in files:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unreadable.append(LintReadError(path, exc).entry(root))
            continue
        lines = source.splitlines()
        rel = display_path(path, root)
        for site in collect_sites(path, source, canon):
            sites.append(site)
            reason = classify(site.target)
            if reason is None:
                continue
            first = site.marker_from - 1
            while (
                first > 0
                and _is_comment(lines[first])
                and _is_comment(lines[first - 1])
            ):
                first -= 1
            if allowed(
                lines, site.end_line - 1, ALLOW_RE, window=site.end_line - 1 - first
            ):
                continue
            violations.append(
                f"{rel}:{site.line}: {site.form} {site.target}\n    -> {reason}"
            )
    return sites, violations, unreadable


def main(argv: list[str]) -> int:
    root = repo_root()
    listing = "--list" in argv
    args = [a for a in argv if a != "--list"]
    files = _iter_target_files(args, root)
    if args and not files:
        sys.stderr.write(
            "[check_forbidden_patches] no .py files in args; "
            "falling back to repo-wide scan.\n"
        )
        files = _iter_target_files([], root)

    sites, violations, unreadable = scan(files, root, root / "src")
    if listing:
        for site in sites:
            rel = display_path(site.path, root)
            print(f"{rel}:{site.line}: {site.form} {site.target}")
        return 0

    skipped = report_unreadable(unreadable)
    found = report(
        "Forbidden patch targets found in tests "
        "(see docs/TESTING.md §2.2 boundary catalogue):",
        violations,
        "Drive the code with inputs instead. Where delegation is the contract "
        "under test, add `# allow-patch: <reason>` on the call.",
    )
    return max(skipped, found)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
