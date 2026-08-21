#!/usr/bin/env python3
"""Verify every ``path#Lnnn`` anchor in the docs still points at what it names.

``docs/CODEMAP.md`` navigates the tree with most of the 94 links of the form
``[memory.py:327](../src/jamma/core/memory.py#L327)``. Nothing kept those line
numbers honest, so they rotted: at 7.2.0 fifty-three of them were wrong, some by
hundreds of lines (``MemoryBreakdown`` was listed at 154 and sat at 271). A
reader following one lands in the middle of an unrelated function and has no
signal that the map is stale. Neither lychee nor markdownlint catches this,
because the *file* resolves; only the line is wrong.

Three checks, in increasing strength:

1. The target file exists.
2. The line number is within the file.
3. The line is the definition it claims to be. When the surrounding table row
   names a symbol in backticks, that exact symbol must be defined on that line.
   When it names no symbol (the Quick Navigation tables link a bare filename),
   the line must at least define *something*, since every anchor in these docs
   points at a ``def``, ``class``, or module-level constant.

Python targets resolve through ``ast``, so the check is exact rather than a
grep. C targets fall back to a definition-shaped regex. Targets that are neither
get checks 1 and 2 only, because "definition" means nothing in a ``.toml``.

What a passing run does *not* prove
-----------------------------------

Check 3 has to work out which symbol an anchor means, and for CODEMAP's tables
that is a guess. Those rows label the link ``file.py:123`` and put the symbol in
a separate column, so ``_wanted_symbol`` takes the first plausible backticked
name in the row. A row naming two symbols can therefore be checked against the
wrong one, which passes when that other symbol happens to sit on the anchored
line and fails misleadingly when it does not.

So a green run means "no anchor is provably wrong", not "every anchor is
provably right". Closing the gap means labelling each link with its own symbol,
``[compute_kinship](../src/jamma/pipeline_kinship.py#L33)``, at which point the
label alone is authoritative and the row never has to be consulted. That is a
change to roughly 120 links across the docs and has not been made.

Usage:
  python3 scripts/check_doc_anchors.py
"""

from __future__ import annotations

import ast
import keyword
import re
import sys
from pathlib import Path

from _lint_common import read_lines, repo_root, report, tracked_files

# LICENSE.md is upstream GPL boilerplate. Everything else the other two doc
# checkers ignore (.venv, .planning, .beads, .claude, .code-review-graph,
# node_modules, dist, build, target, CLAUDE.md, AGENTS.md) is gitignored, so
# git excludes it by construction and there is no list here to drift from
# .markdownlint-cli2.jsonc's "ignores" or lychee.toml's "exclude_path".
SKIP_FILES: frozenset[str] = frozenset({"LICENSE.md"})

PY_SUFFIXES: frozenset[str] = frozenset({".py", ".pyi"})
C_SUFFIXES: frozenset[str] = frozenset({".c", ".h"})

LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+?)#L(\d+)\)")
# A backticked identifier, optionally written with a trailing "()".
BACKTICKED = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)(?:\(\))?`")
# Anything that reads as a path or filename rather than a symbol.
FILENAME_LIKE = re.compile(r"/|\.(?:py|pyi|c|h|md|toml|yml|yaml|sh|json|txt)$")
C_DEFINITION = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_ *]*\b(?P<fn>[A-Za-z_][A-Za-z0-9_]*)\s*\("
    r"|\s*#\s*define\s+(?P<macro>[A-Za-z_][A-Za-z0-9_]*)\b"
    r"|\s*(?:typedef\s+)?struct\s+(?P<struct>[A-Za-z_][A-Za-z0-9_]*))"
)


def _python_definitions(source: str, path: Path) -> dict[int, set[str]]:
    """Map each definition line to the names defined there."""
    out: dict[int, set[str]] = {}
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.setdefault(node.lineno, set()).add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.setdefault(node.lineno, set()).add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.setdefault(node.lineno, set()).add(node.target.id)
    return out


def _c_definition_at(line: str) -> str | None:
    match = C_DEFINITION.match(line)
    if match is None:
        return None
    return match.group("fn") or match.group("macro") or match.group("struct")


def _symbol_candidates(text: str) -> list[str]:
    """Backticked names in ``text`` that could plausibly be a code symbol.

    Filenames and language keywords are dropped. A row like "uses
    ``raise RuntimeError``, not bare ``assert``" names no symbol at all, and
    saying so is better than guessing ``assert``.
    """
    out = []
    for raw in BACKTICKED.findall(text):
        if FILENAME_LIKE.search(raw) or keyword.iskeyword(raw):
            continue
        out.append(raw)
    return out


def _wanted_symbol(label: str, row: str) -> str | None:
    """The symbol an anchor claims to point at, if the prose names one.

    The link label wins when it names one, because that is the tightest binding
    between the link and a name. CODEMAP's tables instead label links
    ``file.py:123`` and put the symbol in the row's Component column, so fall
    back to the row.

    Taking the first candidate from the row is a guess, not a derivation. A row
    that names two symbols may be checked against the wrong one. See "What a
    passing run does not prove" in the module docstring.
    """
    for source in (label, row):
        candidates = _symbol_candidates(source)
        if candidates:
            return candidates[0].split(".")[-1]
    return None


def _markdown_files() -> list[Path]:
    """Every committed markdown file this repo is responsible for."""
    return [p for p in tracked_files("*.md") if p.name not in SKIP_FILES]


def _check_anchor(
    doc: Path, doc_line: int, label: str, rel: str, want: int, row: str, root: Path
) -> str | None:
    """Return a problem description, or None when the anchor is sound."""
    where = f"{doc.relative_to(root)}:{doc_line}"
    target = (doc.parent / rel).resolve()
    if not target.exists():
        return f"{where}: {rel} does not exist"

    lines = read_lines(target)
    if want > len(lines):
        return (
            f"{where}: {rel}#L{want} is past the end of the file ({len(lines)} lines)"
        )

    # Both languages reduce to the same two views: what is defined on the
    # anchored line, and where each name is actually defined.
    if target.suffix in PY_SUFFIXES:
        by_line = _python_definitions("\n".join(lines), target)
    elif target.suffix in C_SUFFIXES:
        by_line = {}
        for n, text in enumerate(lines, start=1):
            name = _c_definition_at(text)
            if name is not None:
                by_line.setdefault(n, set()).add(name)
    else:
        # Not source. Existence and range are all we can honestly assert.
        return None

    names_here = by_line.get(want, set())
    first_line_of: dict[str, int] = {}
    for line_no in sorted(by_line):
        for name in by_line[line_no]:
            first_line_of.setdefault(name, line_no)

    symbol = _wanted_symbol(label, row)
    if symbol is not None:
        if symbol in names_here:
            return None
        real = first_line_of.get(symbol)
        if real is None:
            return f"{where}: `{symbol}` is not defined anywhere in {rel}"
        return f"{where}: `{symbol}` is at {rel}#L{real}, not #L{want}"

    if not names_here:
        return (
            f"{where}: {rel}#L{want} is not a definition "
            f"({lines[want - 1].strip()[:50]!r})"
        )
    return None


def main() -> int:
    root = repo_root()
    problems: list[str] = []
    anchors = 0

    for doc in _markdown_files():
        for doc_line, row in enumerate(read_lines(doc), start=1):
            for label, rel, anchor in LINK.findall(row):
                anchors += 1
                problem = _check_anchor(
                    doc, doc_line, label, rel, int(anchor), row, root
                )
                if problem is not None:
                    problems.append(problem)

    if problems:
        # stdout, not stderr: this lint has always reported there, and its
        # tests read result.stdout.
        return report(
            f"Stale documentation anchors ({len(problems)} of {anchors} checked):\n",
            problems,
            "Each anchor should point at the line where its symbol is defined.\n"
            "Fix the line number, or the symbol name, or drop the anchor.",
            stream=sys.stdout,
        )

    print(f"check_doc_anchors: {anchors} anchors OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
