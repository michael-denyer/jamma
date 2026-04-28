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

The module-form ``monkeypatch.setattr(<module>, "<attr>", ...)`` is *not*
flagged because tests legitimately use it to toggle internal feature-flag
constants (e.g. ``_C_ACCEL_AVAILABLE``) — those are dispatch booleans, not
numerical functions. Use ``patch.object(<module>, "<func>", ...)`` to do
the same and you'll trip the gate, which is the desired behaviour.

Exceptions: a line carrying ``# allow-patch: <reason>`` is skipped. The
comment may appear on the line of the matched call or on any continuation
line up to the matching close paren — multi-line patch calls work the same
way single-line ones do.

Usage::

    python3 scripts/check-forbidden-patches.py        # repo-wide
    python3 scripts/check-forbidden-patches.py f1 f2  # specific files
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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
    repo_root = Path(__file__).resolve().parent.parent
    files: list[Path] = []
    for pattern in DEFAULT_GLOBS:
        files.extend(repo_root.glob(pattern))
    return sorted(files)


class _ScanError(Exception):
    """Raised when a target file cannot be read.

    The gate must surface this — silently treating a read failure as
    "no findings" would hide both the broken file and the fact that the
    gate skipped it.
    """


def _allow_window_for_match(lines: list[str], match_line: int) -> range:
    """Return the range of physical lines on which an allow-patch comment
    applies to a match starting at ``match_line``.

    Multi-line ``patch(...)`` calls span a paren-balanced region; the
    comment may appear on any line in that region. We approximate by
    counting parens until we close the open call (or hit EOF).

    The paren counter is naive about parens inside string literals — that
    would only matter if a forbidden target string contained a stray ``(``
    or ``)``, which they don't.
    """
    line = lines[match_line]
    depth = line.count("(") - line.count(")")
    end = match_line
    while depth > 0 and end + 1 < len(lines):
        end += 1
        depth += lines[end].count("(") - lines[end].count(")")
    return range(match_line, end + 1)


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return [(line_number, matched_text, reason), ...] for forbidden patches.

    Raises ``_ScanError`` if the file cannot be read.
    """
    findings: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _ScanError(f"{path}: {exc}") from exc
    lines = text.splitlines()
    for lineno, line in enumerate(lines):
        for pattern, reason in FORBIDDEN_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            allow_window = _allow_window_for_match(lines, lineno)
            if any(ALLOW_RE.search(lines[i]) for i in allow_window):
                continue
            findings.append((lineno + 1, match.group(0), reason))
    return findings


def main(argv: list[str]) -> int:
    files = _iter_target_files(argv)
    repo_root = Path(__file__).resolve().parent.parent

    # If args were passed but none had a .py suffix, fall back to repo-wide
    # scanning rather than passing vacuously. Pre-commit can hand the hook a
    # non-.py-only batch (e.g. only docs staged); we'd rather scan the tree
    # than silently green-light it.
    if argv and not files:
        sys.stderr.write(
            "[check-forbidden-patches] no .py files in args; "
            "falling back to repo-wide scan.\n"
        )
        files = _iter_target_files([])

    failures: list[str] = []
    read_errors: list[str] = []
    for path in files:
        try:
            scan_results = _scan_file(path)
        except _ScanError as exc:
            read_errors.append(str(exc))
            continue
        for lineno, snippet, reason in scan_results:
            try:
                rel = path.relative_to(repo_root)
            except ValueError:
                rel = path
            failures.append(f"{rel}:{lineno}: {snippet!r}\n    -> {reason}")

    if read_errors:
        sys.stderr.write(
            "Forbidden-patches gate could not read the following files. "
            "The gate is non-functional until these are resolved:\n\n"
        )
        sys.stderr.write("\n".join(read_errors))
        sys.stderr.write("\n")
        return 1

    if failures:
        sys.stderr.write(
            "Forbidden patch targets found in tests "
            "(see docs/TESTING.md §2.2 boundary catalogue):\n\n"
        )
        sys.stderr.write("\n\n".join(failures))
        sys.stderr.write(
            "\n\nIf the patch is genuinely necessary (e.g. short-circuiting "
            "for warning-routing tests), add `# allow-patch: <reason>` to "
            "the line.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
