#!/usr/bin/env python3
"""Keep tracked handwritten code below the repository's decomposition limit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _lint_common import read_batch, repo_root, report, report_unreadable, tracked_files

CODE_PATHS: tuple[str, ...] = ("*.py", "*.pyi", "*.c", "*.h", "*.sh")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-lines",
        type=int,
        default=1_000,
        help="fail when a tracked code file exceeds this many lines",
    )
    parser.add_argument(
        "--warn-lines",
        type=int,
        default=850,
        help="report non-failing files at or above this many lines",
    )
    return parser


def _display(path: Path, root: Path, n_lines: int) -> str:
    return f"{path.relative_to(root).as_posix()}: {n_lines} lines"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warn_lines < 1 or args.max_lines < args.warn_lines:
        _parser().error("require 1 <= --warn-lines <= --max-lines")

    root = repo_root()
    lines_by_path, unreadable = read_batch(
        tracked_files(*CODE_PATHS, root=root), root=root
    )
    if unreadable:
        return report_unreadable(unreadable)

    sizes = sorted(
        ((path, len(lines)) for path, lines in lines_by_path.items()),
        key=lambda item: (-item[1], item[0].as_posix()),
    )
    violations = [
        _display(path, root, n_lines)
        for path, n_lines in sizes
        if n_lines > args.max_lines
    ]
    warnings = [
        _display(path, root, n_lines)
        for path, n_lines in sizes
        if args.warn_lines <= n_lines <= args.max_lines
    ]

    if warnings:
        print(
            f"Tracked code files at or above {args.warn_lines} lines:",
            file=sys.stderr,
        )
        for warning in warnings:
            print(f"  {warning}", file=sys.stderr)

    return report(
        f"Tracked code files exceed {args.max_lines} lines:",
        violations,
        "Split each file along a real module seam before adding more code.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
