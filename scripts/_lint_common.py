"""Shared plumbing for the ``scripts/check_*.py`` lints.

Five lints gate this repo, and each one had carried its own copy of the
same five steps: find the repo root, enumerate the files to look at, read
one, decide whether an opt-out comment covers a finding, and print the
violations with an exit code. The pattern tables differ. The plumbing did
not, so it drifted -- two of the five printed a different shape of report
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
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TextIO

# What a lint prints when it could not read one of its targets. Kept here
# rather than in each lint because the message is the point: a gate that
# skips a file reports success over code it never checked, and the reader
# has to be told which file and why. #171 added it deliberately, after
# four gates were found reporting success for files they never opened.
# Do not let it decay back into a traceback: a traceback exits 1 too,
# which is why the tests pin this text and not just the exit code.
UNREADABLE_HEADER = "Files could not be read, so they were not checked:"


class LintReadError(Exception):
    """A file the lint was asked to check could not be read or decoded.

    Distinct from the violations a lint reports. A violation means the tree
    is wrong; this means the lint never looked. Both fail the gate, because
    a gate that skips a file it cannot read reports success over unchecked
    code.

    Attributes:
        path: The file that could not be read.
        cause: The underlying OSError or UnicodeDecodeError.
    """

    def __init__(self, path: Path, cause: Exception) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"{path}: {type(cause).__name__}: {cause}")

    def entry(self, root: Path | None = None) -> str:
        """One report line: the path as the reader knows it, and the reason."""
        return (
            f"{display_path(self.path, root)}: "
            f"{type(self.cause).__name__}: {self.cause}"
        )


def repo_root() -> Path:
    """Return the repository root, derived from this module's own location."""
    return Path(__file__).resolve().parent.parent


def display_path(path: Path, root: Path | None = None) -> str:
    """Return the path as a violation should name it.

    Repo-relative when the file is inside the tree, absolute otherwise, so
    a lint handed an outside path by argv still says which file it meant.
    """
    base = repo_root() if root is None else root
    if path.is_absolute() and path.is_relative_to(base):
        return path.relative_to(base).as_posix()
    return str(path)


def tracked_files(*pathspecs: str, root: Path | None = None) -> list[Path]:
    """Return absolute paths of the git-tracked files matching ``pathspecs``.

    ``git ls-files`` is the enumeration, not a filesystem walk plus a
    hand-maintained ignore list. Anything gitignored is excluded because git
    does not track it, which is one fewer copy of that list to keep in step
    with ``.markdownlint-cli2.jsonc``, ``lychee.toml``, and the other lints.
    Staged-but-uncommitted files are included, because ``ls-files`` reads the
    index, which is what a pre-commit gate wants.

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


def read_lines(path: Path, *, errors: str = "strict") -> list[str]:
    """Return the file's lines, without terminators.

    Args:
        path: File to read as UTF-8.
        errors: Decode policy. ``"replace"`` is for targets a lint merely
            points at rather than owns, where undecodable bytes are not the
            lint's business.

    Raises:
        LintReadError: If the file cannot be read, or cannot be decoded
            under a strict policy. Returning an empty list instead would
            let the lint pass over a file it never checked, which is the
            failure mode these gates exist to prevent.
    """
    try:
        return path.read_text(encoding="utf-8", errors=errors).splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise LintReadError(path, exc) from exc


def read_batch(
    paths: Iterable[Path], *, root: Path | None = None
) -> tuple[dict[Path, list[str]], list[str]]:
    """Read every path, collecting the failures instead of raising on the first.

    This is where `LintReadError` is caught, once, so a lint's ``main`` keeps
    its pattern table and its ``scan_line`` and nothing else. Raising out of a
    lint would end the run in a traceback, which tells the reader far less
    than a list of the files the gate could not check.

    Args:
        paths: Files to read, in the order they should be reported.
        root: Repository the paths are named relative to.

    Returns:
        The lines of every file that could be read, keyed by path and in
        input order, and a report line for each file that could not.
    """
    lines_by_path: dict[Path, list[str]] = {}
    unreadable: list[str] = []
    for path in paths:
        try:
            lines_by_path[path] = read_lines(path)
        except LintReadError as exc:
            unreadable.append(exc.entry(root))
    return lines_by_path, unreadable


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


def report_unreadable(unreadable: Sequence[str]) -> int:
    """Print the files a lint could not read and return the exit status.

    Always fails when there is anything to report. A skipped file is an
    unchecked file, so a gate that stayed green here would be claiming a
    result it does not have.
    """
    return report(
        UNREADABLE_HEADER,
        unreadable,
        f"{len(unreadable)} file(s) skipped. A skipped file is an unchecked "
        "file, and this gate reports success only when every target was read.",
    )
