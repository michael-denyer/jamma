#!/usr/bin/env python3
"""Reject a home directory path in any tracked file.

``/Users/<name>/`` and ``/home/<name>/`` name the account that produced a
file. A generated artifact picks one up whenever its generator records a
resolved path: GEMMA copies ``argv[0]`` into the ``Command Line Input`` line
of every log, and ``provenance.json`` records the command that ran.

Every tracked file is checked. One that is not UTF-8 is decoded with
replacement characters, which leaves an ASCII path intact. The path counts
wherever it sits in a longer one and with or without a trailing slash. A
form that names no account passes: ``~/`` and a placeholder such as
``<name>`` or ``$USER``.

Exceptions: a line that carries ``allow-home-path: <reason>`` is skipped.
Reserved for a path that is the same on every machine, such as a CI image
layout.

Usage:
  python3 scripts/check_home_paths.py            # repo-wide
  python3 scripts/check_home_paths.py file1 f2   # specific files
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
    tracked_files,
)

# The lint's own test feeds it home directories as fixtures. It is tracked,
# so `tracked_files` cannot drop it.
SELF_EXCLUDE: frozenset[str] = frozenset({"tests/test_check_home_paths.py"})

# A login name is letters, digits, `.`, `_` and `-`, which is what leaves
# `<name>` and `$USER` alone.
HOME_PATH_PATTERN = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+")

ALLOW_MARKER = "allow-home-path"


def _iter_target_files(argv_files: list[str]) -> list[Path]:
    if argv_files:
        return [Path(f).resolve() for f in argv_files]
    root = repo_root()
    return [
        path
        for path in tracked_files()
        if path.relative_to(root).as_posix() not in SELF_EXCLUDE
        # The index still lists a file deleted from the worktree without
        # `git rm`. Reading it would fail the gate for an unrelated reason.
        and path.is_file()
    ]


def main(argv: list[str]) -> int:
    root = repo_root()
    lines_by_path, unreadable = read_batch(
        _iter_target_files(argv), root=root, errors="replace"
    )

    violations: list[str] = []
    for path, lines in lines_by_path.items():
        rel = display_path(path, root)
        for i, line in enumerate(lines):
            matches = HOME_PATH_PATTERN.findall(line)
            if matches and not allowed(lines, i, ALLOW_MARKER):
                violations.extend(
                    f"{rel}:{i + 1}: home directory path {match!r}" for match in matches
                )

    skipped = report_unreadable(unreadable)
    found = report(
        "Home directory path in a tracked file:",
        violations,
        f"{len(violations)} violation(s). A tracked file must not name the "
        "account that produced it. Record the path relative to the "
        "repository, or relative to home as '~/', and fix the generator if "
        "one wrote it. Add 'allow-home-path: <reason>' on the line for a "
        "path that is the same on every machine.",
    )
    return max(skipped, found)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
