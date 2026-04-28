#!/usr/bin/env python3
"""Ban patching numerical functions and internal jamma collaborators in tests.

Enforces the boundary catalogue in docs/TESTING.md §2.2: ``@patch`` and
``MagicMock`` are allowed only at OS / hardware / process / external-UI
boundaries. Patching ``numpy.linalg.*``, BLAS routines, or jamma's own
internal functions hides interface drift and silently masks real numerical
behaviour.

This hook scans tests/ for forbidden patch targets and fails with a list
of offending sites.

Exceptions (lines with ``# allow-patch: <reason>`` are skipped) exist for
the rare legitimate case — e.g. patching at the jamma import site of a
numpy function for short-circuit testing of warning routing.

Usage:
  python3 scripts/check-forbidden-patches.py        # repo-wide
  python3 scripts/check-forbidden-patches.py f1 f2  # specific files
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Patch targets that should never appear in test code. The boundary
# catalogue in docs/TESTING.md §2.2 lists what IS allowed; anything else
# is an interface-drift hazard.
FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r'patch\(\s*["\']numpy\.linalg\.',
        "Patches a NumPy numerical function. Use real synthetic data, or "
        "patch at the jamma import site (jamma.lmm.eigen.np.linalg.eigh) "
        "if testing routing logic.",
    ),
    (
        r'patch\(\s*["\']numpy\.matmul',
        "Patches a NumPy numerical function. Use real synthetic data.",
    ),
    (
        r'patch\(\s*["\']scipy\.',
        "Patches scipy. scipy is a test-only reference; patching it makes "
        "the reference circular. Use real scipy.stats values.",
    ),
    # Patches against jamma's own compute / likelihood / kinship modules.
    # Exclude feature-flag constants (anything ending in _AVAILABLE /
    # _ENABLED / _DISABLED) — those are dispatch booleans, legitimate to
    # toggle in tests. The negative lookahead ``(?![A-Z_]*_AVAILABLE)`` etc.
    # only fires when the patched symbol is a function (lowercase first
    # char or underscore + lowercase).
    (
        r'patch\(\s*["\']jamma\.lmm\.likelihood\.(?![A-Z_]+\b)[a-z_]',
        "Patches a jamma LMM likelihood function. Use small synthetic data "
        "with known expected values.",
    ),
    (
        r'patch\(\s*["\']jamma\.lmm\.compute_numpy\.(?![A-Z_]+\b)[_a-z][a-z_]*\b(?<!_AVAILABLE)(?<!_ENABLED)',
        "Patches a jamma compute function. Use real synthetic data. "
        "(Toggling _C_*_AVAILABLE flags is allowed — those are dispatch "
        "booleans, not functions.)",
    ),
    (
        r'patch\(\s*["\']jamma\.jlinalg\.(eigh|dgemm|dsyrk)\b',
        "Patches a jlinalg numerical wrapper. Use real synthetic data.",
    ),
    (
        r'patch\(\s*["\']jamma\.kinship\.compute\.(?![A-Z_]+\b)[a-z_]',
        "Patches kinship computation. Use real small synthetic data.",
    ),
)

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


def _scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return [(line_number, matched_text, reason), ...] for forbidden patches.

    Raises ``_ScanError`` if the file cannot be read. Callers must surface
    this — see class docstring.
    """
    findings: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _ScanError(f"{path}: {exc}") from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_RE.search(line):
            continue
        for pattern, reason in FORBIDDEN_PATTERNS:
            match = re.search(pattern, line)
            if match:
                findings.append((lineno, match.group(0), reason))
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
