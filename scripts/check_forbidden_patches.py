#!/usr/bin/env python3
"""Ban patching numerical functions and internal jamma collaborators in tests.

Enforces the boundary catalogue in docs/TESTING.md §2.2: ``@patch`` and
``MagicMock`` are allowed only at OS / hardware / process / external-UI
boundaries. Patching ``numpy.linalg.*``, BLAS routines, or jamma's own
internal functions hides interface drift and silently masks real numerical
behaviour.

The gate covers four invocation forms (the regexes that follow are anchored
on the literal call prefix):

* ``patch("dotted.path.to.target")``                — ``unittest.mock.patch``
* ``patch.object(<module-or-class>, "<attr>")``     — module-attribute form
* ``mocker.patch("dotted.path.to.target")``         — pytest-mock
* ``monkeypatch.setattr("dotted.path.to.target")``  — string-target form

The module-form ``monkeypatch.setattr(<module>, "<attr>", ...)`` is also
covered when the module reference is one of the documented forbidden
aliases (``compute_numpy``, ``cn``, ``likelihood``, ``jlinalg``, ``jl``,
``kinship_compute``, ``kc``) and the attribute name is a lowercase
function-shaped identifier. Feature-flag constants (``_AVAILABLE`` /
``_ENABLED`` suffixes) remain allowed because tests legitimately toggle
them to drive dispatch paths. Add ``# allow-patch: <reason>`` for the
rare case where patching the function reference itself (e.g. setting it
to ``None`` to force the NumPy fallback, or to a sentinel that asserts
on call) is the right test design.

Exceptions: a line carrying ``# allow-patch: <reason>`` is skipped. The
comment may appear on the line of the matched call or on any continuation
line up to the matching close paren — multi-line patch calls work the same
way single-line ones do.

Usage::

    python3 scripts/check_forbidden_patches.py        # repo-wide
    python3 scripts/check_forbidden_patches.py f1 f2  # specific files
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

from _lint_common import (
    allowed,
    display_path,
    read_batch,
    repo_root,
    report,
    report_unreadable,
)

# Forbidden targets: each entry is (target_regex_after_open_paren, reason).
# Each target is paired with every invocation prefix to build the full
# pattern table below.
TARGETS: tuple[tuple[str, str], ...] = (
    (
        r'["\']numpy\.linalg\.',
        "Patches a NumPy numerical function. Use real synthetic data, or "
        "patch at the jamma import site (jamma.lmm.eigen.np.linalg.eigh) "
        "if testing routing logic.",
    ),
    (
        r'["\']numpy\.matmul',
        "Patches a NumPy numerical function. Use real synthetic data.",
    ),
    (
        r'["\']scipy\.',
        "Patches scipy. scipy is a test-only reference; patching it makes "
        "the reference circular. Use real scipy.stats values.",
    ),
    (
        # Functions in jamma.lmm.likelihood (not feature-flag constants).
        r'["\']jamma\.lmm\.likelihood\.(?![A-Z_]+\b)[a-z_]',
        "Patches a jamma LMM likelihood function. Use small synthetic data "
        "with known expected values.",
    ),
    (
        # Functions in jamma.lmm.compute_numpy (not _AVAILABLE/_ENABLED flags).
        r'["\']jamma\.lmm\.compute_numpy\.(?![A-Z_]+\b)'
        r"[_a-z][a-z_]*\b(?<!_AVAILABLE)(?<!_ENABLED)",
        "Patches a jamma compute function. Use real synthetic data. "
        "(Toggling _C_*_AVAILABLE flags is allowed — those are dispatch "
        "booleans, not functions.)",
    ),
    (
        r'["\']jamma\.jlinalg\.(eigh|dgemm|dsyrk)\b',
        "Patches a jlinalg numerical wrapper. Use real synthetic data.",
    ),
    (
        r'["\']jamma\.kinship\.compute\.(?![A-Z_]+\b)[a-z_]',
        "Patches kinship computation. Use real small synthetic data.",
    ),
)

# Invocation prefixes for STRING-target forms. Each is paired with every
# TARGETS entry to build the cartesian-product pattern table.
STRING_INVOCATIONS: tuple[str, ...] = (
    r"patch\(\s*",
    r"mocker\.patch\(\s*",
    r"monkeypatch\.setattr\(\s*",
)

# patch.object(<module>, "<attr>") forms — first arg is a module reference,
# not a string. We match common module-reference paths to forbidden targets.
# Each entry is (full_pattern, reason).
PATCH_OBJECT_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"patch\.object\(\s*(?:np|numpy)\.linalg\b",
        "patch.object on numpy.linalg — patches a NumPy numerical function.",
    ),
    (
        r"patch\.object\(\s*scipy\b",
        "patch.object on scipy — circular reference, use real scipy.stats.",
    ),
    (
        r"patch\.object\(\s*jamma\.lmm\.likelihood\b",
        "patch.object on jamma.lmm.likelihood — use synthetic data.",
    ),
    (
        r"patch\.object\(\s*jamma\.lmm\.compute_numpy\s*,\s*"
        r'["\'](?![A-Z_]+\b)[_a-z]',
        "patch.object on a jamma.lmm.compute_numpy function (not a flag).",
    ),
    (
        r'patch\.object\(\s*jamma\.jlinalg\b\s*,\s*["\'](eigh|dgemm|dsyrk)\b',
        "patch.object on a jlinalg numerical wrapper.",
    ),
    (
        r"patch\.object\(\s*jamma\.kinship\.compute\b",
        "patch.object on kinship.compute — use synthetic data.",
    ),
)

# Aliases tests use to refer to forbidden modules. Used by the
# ``monkeypatch.setattr(<alias>, "<attr>", ...)`` rule below.
_FORBIDDEN_MODULE_ALIASES: tuple[str, ...] = (
    "compute_numpy",  # `import jamma.lmm.compute_numpy as compute_numpy`
    "cn",  # `import jamma.lmm.compute_numpy as cn`
    "likelihood",  # `from jamma.lmm import likelihood`
    "lik",  # alias
    "jlinalg",  # `from jamma import jlinalg`
    "jl",  # alias
    "kinship_compute",  # `from jamma.kinship import compute as kinship_compute`
    "kc",  # alias
)

# monkeypatch.setattr(<alias>, "<func_name>", ...) — module-form bypass.
# Function-shaped attribute names are flagged. Excludes _AVAILABLE and
# _ENABLED feature-flag constants (which legitimately drive dispatch paths
# in tests). For the rare legitimate function-form patch (e.g. forcing the
# NumPy fallback by setting ``_compute_score_batch_c = None``), add an
# ``# allow-patch: <reason>`` comment on the call.
_MODULE_FORM_PATTERN = (
    r"monkeypatch\.setattr\(\s*(?:" + "|".join(_FORBIDDEN_MODULE_ALIASES) + r")"
    r'\s*,\s*["\'](?![A-Z_]+\b)_?[a-z][a-z_0-9]*\b(?<!_AVAILABLE)(?<!_ENABLED)'
)
PATCH_OBJECT_PATTERNS = (
    *PATCH_OBJECT_PATTERNS,
    (
        _MODULE_FORM_PATTERN,
        "monkeypatch.setattr on a function attribute of a forbidden module "
        "(jamma.lmm.compute_numpy / likelihood / jlinalg / kinship.compute). "
        "Toggle _C_*_AVAILABLE flags instead, or add `# allow-patch: <reason>` "
        "if forcing dispatch fallback or sentinel-on-call is intentional.",
    ),
)


def _build_patterns() -> tuple[tuple[re.Pattern[str], str], ...]:
    items: list[tuple[re.Pattern[str], str]] = []
    for target_re, reason in TARGETS:
        for invocation in STRING_INVOCATIONS:
            items.append((re.compile(invocation + target_re), reason))
    for full, reason in PATCH_OBJECT_PATTERNS:
        items.append((re.compile(full), reason))
    return tuple(items)


FORBIDDEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = _build_patterns()

# Allow-list comment marker. A line carrying this comment is skipped.
ALLOW_RE = re.compile(r"#\s*allow-patch\s*:\s*\S")

DEFAULT_GLOBS: tuple[str, ...] = ("tests/**/*.py",)


def _iter_target_files(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args if Path(a).suffix == ".py"]
    root = repo_root()
    files: list[Path] = []
    for pattern in DEFAULT_GLOBS:
        files.extend(root.glob(pattern))
    return sorted(files)


def _end_of_call(lines: Sequence[str], start: int) -> int:
    """Return the index of the last physical line of the call on ``start``.

    Multi-line ``patch(...)`` calls span a paren-balanced region and the
    allow comment may sit on any line of it, so count parens until the call
    closes (or EOF).

    The paren counter is naive about parens inside string literals — that
    would only matter if a forbidden target string contained a stray ``(``
    or ``)``, which they don't.
    """
    depth = lines[start].count("(") - lines[start].count(")")
    end = start
    while depth > 0 and end + 1 < len(lines):
        end += 1
        depth += lines[end].count("(") - lines[end].count(")")
    return end


def scan_line(line: str) -> list[tuple[str, str]]:
    """Return ``(matched_text, reason)`` for every forbidden patch on a line."""
    findings: list[tuple[str, str]] = []
    for pattern, reason in FORBIDDEN_PATTERNS:
        match = pattern.search(line)
        if match is not None:
            findings.append((match.group(0), reason))
    return findings


def main(argv: list[str]) -> int:
    files = _iter_target_files(argv)
    root = repo_root()

    # If args were passed but none had a .py suffix, fall back to repo-wide
    # scanning rather than passing vacuously. Pre-commit can hand the hook a
    # non-.py-only batch (e.g. only docs staged); we'd rather scan the tree
    # than silently green-light it.
    if argv and not files:
        sys.stderr.write(
            "[check_forbidden_patches] no .py files in args; "
            "falling back to repo-wide scan.\n"
        )
        files = _iter_target_files([])

    lines_by_path, unreadable = read_batch(files, root=root)

    violations: list[str] = []
    for path, lines in lines_by_path.items():
        rel = display_path(path, root)
        for i, line in enumerate(lines):
            findings = scan_line(line)
            if not findings:
                continue
            # The marker may sit on any physical line of a multi-line call,
            # so ask from the call's last line back to its first.
            last = _end_of_call(lines, i)
            if allowed(lines, last, ALLOW_RE, window=last - i):
                continue
            violations.extend(
                f"{rel}:{i + 1}: {snippet!r}\n    -> {reason}"
                for snippet, reason in findings
            )

    skipped = report_unreadable(unreadable)
    found = report(
        "Forbidden patch targets found in tests "
        "(see docs/TESTING.md \u00a72.2 boundary catalogue):",
        violations,
        "If the patch is genuinely necessary (e.g. short-circuiting for "
        "warning-routing tests), add `# allow-patch: <reason>` to the line.",
    )
    return max(skipped, found)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
