"""A local run of ``generate_gemma_fixtures.sh`` records a portable command.

GEMMA copies argv[0] and every argument into the ``Command Line Input`` line
of its log. ``scripts/check_fixture_manifest.py --write`` copies that line into
``generation_cmd`` in ``tests/fixtures/MANIFEST.toml``, and
``scripts/_gemma_fixture_cells.py`` derives the generator's cell table from
``generation_cmd``. So the command the generator runs has to map back to the
row that produced it. A command that names the binary's directory or the
checkout does not: the row disappears from the table without an error.

The test runs the real script against a compiled stand-in for GEMMA, because
a script stand-in cannot see argv[0]. The kernel replaces it with the script's
path when it starts the interpreter.
"""

from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = _REPO_ROOT / "scripts"
_MANIFEST = _REPO_ROOT / "tests" / "fixtures" / "MANIFEST.toml"

# Rows with %ROOT% paths, %OUTDIR% paths, and the one row with no -outdir.
_ONLY = "gemma_[acs]*"

_ARGV_RECORDER_C = """
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    FILE *out = fopen(getenv("ARGV_RECORD"), "a");
    if (!out) return 1;
    for (int i = 0; i < argc; i++) fprintf(out, "%s%s", i ? " " : "", argv[i]);
    fputc('\\n', out);
    return fclose(out);
}
"""


def _load_cells():
    """Import the cell-table script from ``scripts/``, which is not a package."""
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        import _gemma_fixture_cells
    finally:
        if sys.path and sys.path[0] == str(_SCRIPT_DIR):
            sys.path.pop(0)
    return _gemma_fixture_cells


def _commands_run_locally(tmp_path: Path) -> list[str]:
    """Run the generator's local runner and return each command GEMMA saw."""
    source = tmp_path / "argv_recorder.c"
    source.write_text(_ARGV_RECORDER_C)
    # Not named gemma: a real binary is often gemma-0.98.5-linux-static-AMD64.
    binary = tmp_path / "gemma-0.98.5-stand-in"
    subprocess.run(["cc", "-o", str(binary), str(source)], check=True)

    record = tmp_path / "argv.txt"
    outroot = tmp_path / "outroot"
    outroot.mkdir()
    subprocess.run(
        [
            "bash",
            str(_SCRIPT_DIR / "generate_gemma_fixtures.sh"),
            "--gemma-path",
            str(binary),
            "--only",
            _ONLY,
            "--outroot",
            str(outroot),
        ],
        check=True,
        cwd=_REPO_ROOT,
        # The script reads the manifest through `uv run`. Without UV_NO_SYNC
        # that would re-sync the environment the suite is running in.
        env={**os.environ, "ARGV_RECORD": str(record), "UV_NO_SYNC": "1"},
    )
    return record.read_text().splitlines()


def test_local_run_maps_back_to_the_rows_that_produced_it(tmp_path: Path) -> None:
    cells = _load_cells()
    rows = [
        row
        for row in cells.cells_from_manifest(_MANIFEST)
        if fnmatch.fnmatchcase(row.split("|", 1)[0], _ONLY)
    ]
    assert len(rows) == 6, f"{_ONLY!r} no longer selects the six expected rows: {rows}"

    by_prefix = {}
    for command in _commands_run_locally(tmp_path):
        tokens = shlex.split(command)
        by_prefix[tokens[tokens.index("-o") + 1]] = command

    regenerated = tmp_path / "MANIFEST.toml"
    with regenerated.open("w") as f:
        for row in rows:
            _name, outdir, prefix, _args = row.split("|")
            f.write(f'[file."{outdir}/{prefix}.log.txt"]\n')
            f.write(f'generation_cmd = "{by_prefix[prefix]}"\n\n')

    assert cells.cells_from_manifest(regenerated) == rows
