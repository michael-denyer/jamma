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

A row also has to run at all. GEMMA must start in the directory the row's
paths are relative to, and the LOCO kinship step must import the tests package
when the generator starts it as a script.
"""

from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
import sys
from pathlib import Path

import _gemma_fixture_cells as cells
import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = _REPO_ROOT / "scripts"
_MANIFEST = _REPO_ROOT / "tests" / "fixtures" / "MANIFEST.toml"

# Rows with %ROOT% paths, %OUTDIR% paths, no -outdir, and the one row GEMMA
# runs from inside its fixture directory. The LOCO rows end in a digit and stay
# out: they need the JAMMA kinship step first.
_ONLY = "gemma_*[!0-9]"
_ROWS_SELECTED = 7

_ARGV_RECORDER_C = """
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char cwd[4096];
    FILE *out = fopen(getenv("ARGV_RECORD"), "a");
    if (!out || !getcwd(cwd, sizeof cwd)) return 1;
    fprintf(out, "%s\t", cwd);
    for (int i = 0; i < argc; i++) fprintf(out, "%s%s", i ? " " : "", argv[i]);
    fputc('\\n', out);
    return fclose(out);
}
"""


def _selected_rows() -> list[str]:
    rows = [
        row
        for row in cells.cells_from_manifest(_MANIFEST)
        if fnmatch.fnmatchcase(row.split("|", 1)[0], _ONLY)
    ]
    assert len(rows) == _ROWS_SELECTED, f"{_ONLY!r} selects {rows}"
    return rows


@pytest.fixture(scope="module")
def local_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, str]]:
    """Run the generator's local runner once.

    Returns:
        Per output prefix, the directory GEMMA started in, relative to the
        data root, and the command it saw.
    """
    tmp_path = tmp_path_factory.mktemp("local_run")
    source = tmp_path / "argv_recorder.c"
    source.write_text(_ARGV_RECORDER_C)
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
    runs = {}
    for line in record.read_text().splitlines():
        cwd, command = line.split("\t")
        tokens = shlex.split(command)
        prefix = tokens[tokens.index("-o") + 1]
        runs[prefix] = (Path(cwd).relative_to(outroot.resolve()), command)
    return runs


def test_local_run_maps_back_to_the_rows_that_produced_it(
    local_run: dict[str, tuple[Path, str]], tmp_path: Path
) -> None:
    rows = _selected_rows()
    assert len(local_run) == len(rows), local_run

    regenerated = tmp_path / "MANIFEST.toml"
    with regenerated.open("w") as f:
        for row in rows:
            _name, outdir, prefix, _workdir, _args = row.split("|")
            f.write(f'[file."{outdir}/{prefix}.log.txt"]\n')
            f.write(f'generation_cmd = "{local_run[prefix][1]}"\n\n')

    assert cells.cells_from_manifest(regenerated) == rows


def test_a_command_with_bare_file_names_runs_in_its_fixture_directory(
    local_run: dict[str, tuple[Path, str]],
) -> None:
    """gemma_lrt's recorded command names `test`, not a path from the root."""
    started_in = {prefix: cwd for prefix, (cwd, _command) in local_run.items()}

    assert started_in.pop("gemma_lrt") == Path("tests/fixtures/gemma_synthetic")
    assert set(started_in.values()) == {Path()}


def test_loco_kinship_step_runs_as_a_script(tmp_path: Path) -> None:
    """The generator starts it as `python scripts/generate_loco_synthetic.py`.

    Python then puts scripts/ on sys.path, not the repository root, and the
    step imports from the tests package.
    """
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "generate_loco_synthetic.py"),
            "--loco-kinship",
            str(_REPO_ROOT / "tests" / "fixtures" / "gemma_loco" / "test"),
            str(tmp_path),
        ],
        check=True,
        cwd=_REPO_ROOT,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )

    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == sorted(
        name
        for chrom in (1, 2, 3)
        for name in (f"loco_chr{chrom}_kinship.cXX.txt", f"chr{chrom}_snps.txt")
    )
