"""Shared plumbing for the ``scripts/check_*.py`` lints.

Five lints gate this repo, and each one had carried its own copy of the
same five steps: find the repo root, enumerate the files to look at, read
one, decide whether an opt-out comment covers a finding, and print the
violations with an exit code. The pattern tables differ. The plumbing did
not, so it drifted — two of the five printed a different shape of report
for the same kind of failure, and the ``repo_root = Path(__file__)...``
line existed five times.

The lints are named with underscores rather than hyphens for one reason:
``check-quiet-flags.py`` is not an importable module name, so there was
nowhere for this file to be imported *from*. A lint invoked as
``python3 scripts/check_quiet_flags.py`` gets ``scripts/`` as
``sys.path[0]``, which is what makes ``import _lint_common`` resolve.

Stdlib only, and deliberately so: the pre-commit entries run these through
``language: system``, which means the interpreter on the shebang line, not
the project venv.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO


class LintReadError(Exception):
    """A file the lint was asked to check could not be read or decoded.

    Distinct from the violations a lint reports. A violation means the tree
    is wrong; this means the lint never looked. Both fail the gate, because
    a gate that skips a file it cannot read reports success over unchecked
    code.
    """


def repo_root() -> Path:
    """Return the repository root, derived from this module's own location."""
    return Path(__file__).resolve().parent.parent


def tracked_files(*pathspecs: str, root: Path | None = None) -> list[Path]:
    """Return absolute paths of the git-tracked files matching ``pathspecs``.

    ``git ls-files`` is the enumeration, not a filesystem walk plus a
    hand-maintained ignore list. Anything gitignored is excluded because git
    does not track it, which is one fewer copy of that list to keep in step
    with ``.markdownlint-cli2.jsonc``, ``lychee.toml``, and the other lints.
    Staged-but-uncommitted files are included, because ``ls-files`` reads the
    index — which is what a pre-commit gate wants.

    Pathspecs are git's, not pathlib's. ``*`` crosses ``/``, so ``tests/*.py``
    reaches ``tests/fakes/x.py``; writing ``tests/**/*.py`` would instead
    *require* an intervening directory and silently miss ``tests/x.py``.

    Args:
        pathspecs: git pathspecs. Passing none enumerates the whole tree.
        root: Repository to enumerate. Defaults to `repo_root()`.

    Returns:
        Sorted absolute paths.

    Raises:
        RuntimeError: If git cannot list the tree. Falling back to a
            filesystem walk would silently start checking uncommitted and
            vendored files.
    """
    base = repo_root() if root is None else root
    result = subprocess.run(
        ["git", "-C", str(base), "ls-files", "-z", *pathspecs],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed in {base} (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return sorted(base / name for name in result.stdout.split("\0") if name)


def read_lines(path: Path) -> list[str]:
    """Return the file's lines, without terminators.

    Args:
        path: File to read as UTF-8.

    Raises:
        LintReadError: If the file cannot be read or decoded. Returning an
            empty list instead would let the lint pass over a file it never
            checked, which is the failure mode these gates exist to prevent.
    """
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise LintReadError(
            f"{path}: could not be read: {type(exc).__name__}: {exc}"
        ) from exc


def allowed(
    lines: Sequence[str],
    i: int,
    marker: str | re.Pattern[str],
    *,
    window: int = 0,
) -> bool:
    """Return whether an opt-out ``marker`` covers the finding on ``lines[i]``.

    Every lint here has an escape hatch comment, and every one of them has to
    look slightly above the offending line as well as on it, because
    ruff-format splits a long line and carries its trailing comment up with
    it.

    Args:
        lines: The file's lines.
        i: Zero-based index of the line the finding is on.
        marker: Substring, or a compiled pattern to `re.Pattern.search`.
        window: How many lines above ``i`` the marker may also appear on.
            The default of 0 checks line ``i`` alone.
    """
    start = max(0, i - window)
    span = lines[start : i + 1]
    if isinstance(marker, str):
        return any(marker in line for line in span)
    return any(marker.search(line) for line in span)


def report(
    header: str,
    violations: Sequence[str],
    footer: str,
    *,
    stream: TextIO = sys.stderr,
) -> int:
    """Print a violation report and return the exit status for it.

    Prints nothing and returns 0 when ``violations`` is empty, so a lint can
    end with ``return report(...)`` unconditionally.

    Args:
        header: One line naming the class of problem.
        violations: One entry per finding. Each is indented two spaces.
        footer: What to do about it. Preceded by a blank line.
        stream: Where to write. Defaults to stderr; `check_doc_anchors`
            passes stdout because it has always reported there.

    Returns:
        1 if there were violations, 0 otherwise.
    """
    if not violations:
        return 0
    print(header, file=stream)
    for violation in violations:
        print(f"  {violation}", file=stream)
    print(f"\n{footer}", file=stream)
    return 1
