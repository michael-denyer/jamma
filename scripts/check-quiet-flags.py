#!/usr/bin/env python3
"""Ban ``--quiet``/``-q``/``--silent`` and hook-skip flags project-wide.

CLAUDE.md rule "No Quiet Flags Anywhere": when a tool fails silently, you
can't diagnose the problem — you just see exit code 1 with no output. This
hook catches the same class of drift that `check-compile-flag-literals.py`
catches for compile flags. Scope: CI workflows, shell scripts, pre-commit
config, notebook ``%pip`` cells, and Python subprocess invocations.

Two banned categories:

  1. Quiet flags: ``--quiet``, ``-q``, ``--silent``, ``-s`` (when clearly
     a silencing flag, not ``-s`` for sort or similar). We match
     ``--quiet`` and ``--silent`` as the unambiguous forms, plus ``-q``
     when it follows a known command (pip, uv, apt-get, pytest, ruff).

  2. Hook-skip / signing-bypass flags: ``--no-verify`` (git commit),
     ``--no-gpg-sign``, ``-c commit.gpgsign=false``. These bypass the
     safety net — forbidden unless the user explicitly authorizes.

Exceptions (lines with ``# allow-quiet: <reason>`` are skipped). Reserved
for genuine use cases (e.g. ``ruff check --quiet`` in a diff-only context
where exit code IS the output).

Usage:
  python3 scripts/check-quiet-flags.py            # repo-wide
  python3 scripts/check-quiet-flags.py file1 f2   # specific files
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Files the hook scans when given no explicit args. Repo-wide sweep covers
# CI, pre-commit config, shell scripts, and Python source.
DEFAULT_GLOBS: tuple[str, ...] = (
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    ".pre-commit-config.yaml",
    "scripts/*.sh",
    "scripts/*.py",
    "src/**/*.py",
    "tests/**/*.py",
    "hatch_build.py",
)

# Paths skipped entirely: vendor, cache, build artifacts, and this
# script itself (which unavoidably mentions the banned flags in its own
# documentation and pattern strings).
SKIP_PREFIXES: tuple[str, ...] = (
    ".venv/",
    ".planning/",
    ".beads/",
    ".claude/",
    "dist/",
    "build/",
    "__pycache__/",
    "scripts/check-quiet-flags.py",
    "tests/test_check_quiet_flags.py",
)

# Commands where ``-q`` unambiguously means "quiet". Extend as new tools
# land in CI. Ordering matters for the regex — longest match first.
COMMANDS_WITH_Q_AS_QUIET: tuple[str, ...] = (
    "pip",
    "uv",
    "apt-get",
    "apt",
    "pytest",
    "ruff",
    "prek",
    "pre-commit",
    "npm",
    "curl",
    "wget",
    "git",
)

# --quiet / --silent: unambiguous silencing flags.
LONG_QUIET_PATTERN = re.compile(r"(?<![A-Za-z0-9_])--(quiet|silent)(?![A-Za-z0-9_-])")

# -q / -s following a known command on the same line. Multiline shell
# constructs fall through; that's acceptable since pre-commit hooks run
# per-file and most invocations fit one logical line.
_CMD_ALT = "|".join(COMMANDS_WITH_Q_AS_QUIET)
SHORT_Q_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_/])(?:" + _CMD_ALT + r")\b"
    r"[^\n]*?(?<![A-Za-z0-9_])-q(?![A-Za-z0-9_=])",
)

# git commit --no-verify, --no-gpg-sign, -c commit.gpgsign=false.
HOOK_SKIP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("--no-verify", re.compile(r"(?<![A-Za-z0-9_])--no-verify(?![A-Za-z0-9_-])")),
    ("--no-gpg-sign", re.compile(r"(?<![A-Za-z0-9_])--no-gpg-sign(?![A-Za-z0-9_-])")),
    (
        "-c commit.gpgsign=false",
        re.compile(r"-c\s+commit\.gpgsign\s*=\s*false", re.IGNORECASE),
    ),
)

ALLOW_MARKER = "allow-quiet"


def _iter_target_files(repo_root: Path, argv_files: list[str]) -> list[Path]:
    if argv_files:
        return [Path(f).resolve() for f in argv_files]
    out: list[Path] = []
    for pattern in DEFAULT_GLOBS:
        out.extend(repo_root.glob(pattern))
    # De-dup while preserving order; drop skipped prefixes.
    seen: set[Path] = set()
    result: list[Path] = []
    for p in out:
        if p.is_absolute() and p.is_relative_to(repo_root):
            rel = p.relative_to(repo_root).as_posix()
        else:
            rel = p.as_posix()
        if any(rel.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            result.append(p)
    return result


def _check_line(line: str) -> list[str]:
    """Return list of violation descriptions found on a single line."""
    if ALLOW_MARKER in line:
        return []
    violations: list[str] = []

    # Strip Python/YAML/shell comments so an in-comment mention like
    # "# do not pass --quiet" doesn't trip the lint. Best-effort — a `#`
    # inside a quoted string still counts, but our patterns look for
    # flag forms that shouldn't appear in log strings typically.
    code_part = _strip_trailing_comment(line)

    if m := LONG_QUIET_PATTERN.search(code_part):
        violations.append(f"banned flag {m.group(0)!r}")
    if m := SHORT_Q_PATTERN.search(code_part):
        violations.append(f"banned short-form quiet flag in {m.group(0)!r}")
    for label, pattern in HOOK_SKIP_PATTERNS:
        if pattern.search(code_part):
            violations.append(f"banned hook-skip flag {label!r}")
    return violations


def _strip_trailing_comment(line: str) -> str:
    """Return the line with a trailing ``#``-comment removed.

    Naive: does not parse strings, so a ``#`` inside a quoted literal also
    truncates. That's acceptable — the lint errs on the side of false
    negatives inside string literals, which is the right bias (a banned
    flag inside a printed help string is documentation, not an invocation).
    """
    # Skip shebangs (#!/usr/bin/env ...).
    if line.lstrip().startswith("#!"):
        return ""
    # Find a `#` not preceded by a backslash. Good-enough heuristic.
    idx = line.find("#")
    while idx > 0 and line[idx - 1] == "\\":
        idx = line.find("#", idx + 1)
    if idx < 0:
        return line
    return line[:idx]


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    files = _iter_target_files(repo_root, argv)
    if not files:
        return 0

    violations: list[str] = []
    unreadable: list[str] = []
    for path in files:
        rel_path = (
            path.relative_to(repo_root).as_posix()
            if path.is_absolute() and path.is_relative_to(repo_root)
            else str(path)
        )
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            unreadable.append(f"{rel_path}: {type(exc).__name__}: {exc}")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for v in _check_line(line):
                rel = (
                    path.relative_to(repo_root).as_posix()
                    if path.is_absolute() and path.is_relative_to(repo_root)
                    else str(path)
                )
                violations.append(f"{rel}:{lineno}: {v}")

    if unreadable:
        print("Files could not be read, so they were not checked:", file=sys.stderr)
        for u in unreadable:
            print(f"  {u}", file=sys.stderr)
        print(
            f"\n{len(unreadable)} file(s) skipped. A skipped file is an "
            "unchecked file, and this gate reports success only when every "
            "target was read.",
            file=sys.stderr,
        )

    if violations:
        print("Quiet / hook-skip flag drift detected:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(
            f"\n{len(violations)} violation(s). CLAUDE.md bans --quiet/-q/"
            "--silent project-wide: logs exist to be read. Hook-skip flags "
            "(--no-verify, --no-gpg-sign, -c commit.gpgsign=false) need "
            "explicit user authorization. Add '# allow-quiet: <reason>' on "
            "the line for a documented exception.",
            file=sys.stderr,
        )
        return 1
    return 1 if unreadable else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
