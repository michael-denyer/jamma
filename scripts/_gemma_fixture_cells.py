#!/usr/bin/env python3
"""Derive generate_gemma_fixtures.sh's GEMMA cell table from MANIFEST.toml.

Every regenerable GEMMA-produced fixture records its provenance as a
`generation_cmd` string on its `.log.txt` entry in tests/fixtures/MANIFEST.toml
(written by the run that produced the committed file). This script is the
single source for that table, so the two cannot drift apart: reads each
`generation_cmd`, rewrites its path-like arguments back into the `%ROOT%` /
`%OUTDIR%` tokens generate_gemma_fixtures.sh expands, and prints one
pipe-separated `name|outdir|prefix|args` line per fixture, in the same
"name|outdir|prefix|args" shape the shell script's CELLS table used to carry
by hand.

Excluded, and left to the shell script's own logic:
  - Fixtures whose `generation_cmd` does not start with the literal `gemma `
    binary invocation (a legacy `/data/legacy` or `/data/input` source tree,
    or a stray `jamma` binary path some early fixture was produced with).
    Those are historical audit records for fixtures this script cannot
    regenerate, not GEMMA cells.
  - The three `gemma_loco_chr{1,2,3}` fixtures. Their `.log.txt` files carry
    no `generation_cmd` at all (LOCO fixtures were never re-run through the
    generic pipeline, only through generate_loco_synthetic.py), so there is
    nothing here to loop over; the shell script keeps a hand-written loop for
    those three.

Usage:
  python3 scripts/_gemma_fixture_cells.py [manifest_path]
"""

from __future__ import annotations

import shlex
import sys
import tomllib
from pathlib import Path

_EXCLUDED_SOURCE_MARKERS = ("/data/legacy", "/data/input")


def cells_from_manifest(manifest_path: Path) -> list[str]:
    """Return one pipe-separated CELLS line per regenerable GEMMA fixture.

    Args:
        manifest_path: Path to tests/fixtures/MANIFEST.toml.

    Returns:
        Lines in `name|outdir|prefix|args` form, sorted by fixture path, with
        `%ROOT%` and `%OUTDIR%` tokens restored in place of the literal paths
        `generation_cmd` recorded.
    """
    with manifest_path.open("rb") as f:
        manifest = tomllib.load(f)

    lines = []
    for fixture_path in sorted(manifest["file"]):
        entry = manifest["file"][fixture_path]
        cmd = entry.get("generation_cmd")
        if not cmd or not cmd.startswith("gemma "):
            continue
        if any(marker in cmd for marker in _EXCLUDED_SOURCE_MARKERS):
            continue

        outdir = fixture_path.rsplit("/", 1)[0]
        tokens = shlex.split(cmd)[1:]  # drop the leading "gemma"
        prefix = tokens[tokens.index("-o") + 1]

        rewritten = []
        for tok in tokens:
            if tok.startswith("/data/"):
                tok = tok[len("/data/") :]
            if tok == outdir:
                tok = "%OUTDIR%"
            elif tok.startswith(outdir + "/"):
                tok = "%OUTDIR%/" + tok[len(outdir) + 1 :]
            elif "/" in tok and not tok.startswith("%"):
                tok = "%ROOT%/" + tok
            rewritten.append(tok)

        lines.append(f"{prefix}|{outdir}|{prefix}|{' '.join(rewritten)}")
    return lines


def main() -> int:
    default_manifest = Path("tests/fixtures/MANIFEST.toml")
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_manifest
    for line in cells_from_manifest(manifest_path):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
