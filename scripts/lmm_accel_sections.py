"""Map the section and static-function structure of ``_lmm_accel.c``.

The file is organised into banner-delimited sections that are the natural
family seams for a split, but most of its static helpers are referenced from
more than one section. Which helpers those are decides what a shared core must
hold, and re-deriving that by hand after every extraction step is both slow and
easy to get wrong.

Run it for a section-by-section summary::

    uv run python scripts/lmm_accel_sections.py

``--cross`` lists only the statics referenced from outside their own section,
most-shared first, which is the worklist for the shared core::

    uv run python scripts/lmm_accel_sections.py --cross

``--json`` emits the same data as a machine-readable object.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent / "src/jamma/lmm/_lmm_accel.c"

_BANNER = re.compile(r"^/\* ={10,}\s*$")
_STATIC_HEAD = re.compile(r"^static\s+(?:inline\s+)?(?:const\s+)?[\w\s*]+?(\w+)\s*\(")
_TYPEDEF_END = re.compile(r"^\}\s*(\w+_t)\s*;")
_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")


@dataclass
class Section:
    index: int
    start: int
    title: str
    end: int = 0


@dataclass
class Func:
    name: str
    line: int
    end: int
    section: int
    ref_sections: set[int] = field(default_factory=set)
    ref_count: int = 0

    @property
    def crosses(self) -> bool:
        return bool(self.ref_sections - {self.section})


def _sections(lines: list[str]) -> list[Section]:
    """Locate banner-delimited sections; the preamble is section 0."""
    found: list[Section] = [Section(index=0, start=1, title="(preamble)")]
    for i, line in enumerate(lines):
        if not _BANNER.match(line):
            continue
        title = ""
        for follower in lines[i + 1 : i + 3]:
            stripped = follower.lstrip(" *").rstrip()
            if stripped and not stripped.startswith("="):
                title = stripped
                break
        found.append(Section(index=len(found), start=i + 1, title=title))
    for current, following in itertools.pairwise(found):
        current.end = following.start - 1
    found[-1].end = len(lines)
    return found


def _section_of(line: int, sections: list[Section]) -> int:
    for section in sections:
        if section.start <= line <= section.end:
            return section.index
    return 0


_SIGNATURE_SPAN = 40


def _functions(lines: list[str], sections: list[Section]) -> list[Func]:
    """Find static function definitions by brace-matching from each head.

    A forward declaration reaches ``;`` before ``{`` and is skipped. Only the
    hunt for the opening brace is span-limited; the body walk is not, or every
    function longer than the limit would be silently dropped.
    """
    found: list[Func] = []
    i = 0
    while i < len(lines):
        match = _STATIC_HEAD.match(lines[i])
        if not match:
            i += 1
            continue

        open_at = None
        for j in range(i, min(len(lines), i + _SIGNATURE_SPAN)):
            brace = lines[j].find("{")
            semi = lines[j].find(";")
            if semi != -1 and (brace == -1 or semi < brace):
                break
            if brace != -1:
                open_at = j
                break
        if open_at is None:
            i += 1
            continue

        depth = 0
        for j in range(open_at, len(lines)):
            depth += lines[j].count("{") - lines[j].count("}")
            if depth == 0:
                found.append(
                    Func(
                        name=match.group(1),
                        line=i + 1,
                        end=j + 1,
                        section=_section_of(i + 1, sections),
                    )
                )
                i = j + 1
                break
        else:
            i += 1
    return found


def _typedefs(lines: list[str]) -> list[str]:
    return sorted({m.group(1) for line in lines if (m := _TYPEDEF_END.match(line))})


def _resolve_refs(lines: list[str], funcs: list[Func], sections: list[Section]) -> None:
    """Attribute every identifier occurrence outside a function's own body."""
    by_name = {f.name: f for f in funcs}
    for lineno, line in enumerate(lines, start=1):
        code = line.split("/*")[0].split("//")[0]
        for ident in _IDENT.findall(code):
            func = by_name.get(ident)
            if func is None or func.line <= lineno <= func.end:
                continue
            func.ref_sections.add(_section_of(lineno, sections))
            func.ref_count += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cross", action="store_true", help="only crossing statics")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    lines = args.source.read_text().splitlines()
    sections = _sections(lines)
    funcs = _functions(lines, sections)
    _resolve_refs(lines, funcs, sections)
    crossing = sorted(
        (f for f in funcs if f.crosses), key=lambda f: (-f.ref_count, f.name)
    )

    if args.json:
        json.dump(
            {
                "source": str(args.source),
                "lines": len(lines),
                "sections": [
                    {"index": s.index, "start": s.start, "end": s.end, "title": s.title}
                    for s in sections
                ],
                "typedefs": _typedefs(lines),
                "statics": [
                    {
                        "name": f.name,
                        "line": f.line,
                        "end": f.end,
                        "section": f.section,
                        "ref_count": f.ref_count,
                        "ref_sections": sorted(f.ref_sections),
                        "crosses": f.crosses,
                    }
                    for f in funcs
                ],
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    if args.cross:
        print(f"{len(crossing)} of {len(funcs)} statics referenced across sections\n")
        for f in crossing:
            others = sorted(f.ref_sections - {f.section})
            print(
                f"  {f.name:<44} s{f.section:<3} {f.ref_count:>3} refs "
                f"from s{','.join(map(str, others))}"
            )
        return 0

    print(f"{args.source}: {len(lines)} lines, {len(sections)} sections")
    print(f"{len(funcs)} static functions, {len(crossing)} referenced across sections")
    print(f"{len(_typedefs(lines))} typedefs: {', '.join(_typedefs(lines))}\n")
    for section in sections:
        owned = [f for f in funcs if f.section == section.index]
        shared = sum(1 for f in owned if f.crosses)
        print(
            f"s{section.index:<3} {section.start:>6}-{section.end:<6} "
            f"{len(owned):>3} statics ({shared} shared)  {section.title[:58]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
