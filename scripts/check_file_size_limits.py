#!/usr/bin/env python3
"""Keep tracked handwritten code below the repository's decomposition limit."""

from __future__ import annotations

import argparse
import ast
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
    parser.add_argument(
        "--warn-function-lines",
        type=int,
        default=150,
        help="report non-failing Python functions at or above this many lines",
    )
    return parser


def _display(path: Path, root: Path, n_lines: int) -> str:
    return f"{path.relative_to(root).as_posix()}: {n_lines} lines"


def _long_functions(
    lines_by_path: dict[Path, list[str]], root: Path, warn_lines: int
) -> list[str]:
    findings: list[tuple[Path, int, str, int]] = []
    for path, lines in lines_by_path.items():
        if path.suffix not in {".py", ".pyi"}:
            continue
        try:
            tree = ast.parse("\n".join(lines), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.end_lineno is None:
                continue
            span = node.end_lineno - node.lineno + 1
            if span >= warn_lines:
                findings.append((path, node.lineno, node.name, span))
    findings.sort(key=lambda item: (-item[3], item[0].as_posix(), item[1]))
    return [
        f"{path.relative_to(root).as_posix()}:{line}: {name} spans {span} lines"
        for path, line, name, span in findings
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.warn_lines < 1 or args.max_lines < args.warn_lines:
        _parser().error("require 1 <= --warn-lines <= --max-lines")
    if args.warn_function_lines < 1:
        _parser().error("require 1 <= --warn-function-lines")

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
    function_warnings = _long_functions(lines_by_path, root, args.warn_function_lines)

    if warnings:
        print(
            f"Tracked code files at or above {args.warn_lines} lines:",
            file=sys.stderr,
        )
        for warning in warnings:
            print(f"  {warning}", file=sys.stderr)

    if function_warnings:
        print(
            f"Tracked Python functions at or above {args.warn_function_lines} lines:",
            file=sys.stderr,
        )
        for warning in function_warnings:
            print(f"  {warning}", file=sys.stderr)

    return report(
        f"Tracked code files exceed {args.max_lines} lines:",
        violations,
        "Split each file along a real module seam before adding more code.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
