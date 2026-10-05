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
from typing import NamedTuple

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

# Records where it started and what it was called with. Like GEMMA, it writes
# to ./output when the command names no -outdir.
_GEMMA_STAND_IN_C = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char cwd[4096], path[4096];
    const char *prefix = "", *kinds[] = {"assoc", "log"};
    int has_outdir = 0;
    FILE *out = fopen(getenv("ARGV_RECORD"), "a");
    if (!out || !getcwd(cwd, sizeof cwd)) return 1;
    fprintf(out, "%s\\t", cwd);
    for (int i = 0; i < argc; i++) {
        fprintf(out, "%s%s", i ? " " : "", argv[i]);
        if (!strcmp(argv[i], "-o") && i + 1 < argc) prefix = argv[i + 1];
        if (!strcmp(argv[i], "-outdir")) has_outdir = 1;
    }
    fputc('\\n', out);
    if (fclose(out)) return 1;
    if (has_outdir) return 0;
    mkdir("output", 0777);
    for (int i = 0; i < 2; i++) {
        snprintf(path, sizeof path, "output/%s.%s.txt", prefix, kinds[i]);
        if (!(out = fopen(path, "w")) || fclose(out)) return 1;
    }
    return 0;
}
"""


class LocalRun(NamedTuple):
    """One run of the generator's local runner, keyed by output prefix."""

    outroot: Path
    started_in: dict[str, Path]  # relative to outroot
    command: dict[str, str]


@pytest.fixture(scope="module")
def local_run(tmp_path_factory: pytest.TempPathFactory) -> LocalRun:
    tmp_path = tmp_path_factory.mktemp("local_run")
    source = tmp_path / "gemma_stand_in.c"
    source.write_text(_GEMMA_STAND_IN_C)
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
    run = LocalRun(outroot, {}, {})
    for line in record.read_text().splitlines():
        cwd, command = line.split("\t")
        tokens = shlex.split(command)
        prefix = tokens[tokens.index("-o") + 1]
        run.started_in[prefix] = Path(cwd).relative_to(outroot.resolve())
        run.command[prefix] = command
    return run


def test_local_run_maps_back_to_the_rows_that_produced_it(
    local_run: LocalRun, tmp_path: Path
) -> None:
    rows = [
        row
        for row in cells.cells_from_manifest(_MANIFEST)
        if fnmatch.fnmatchcase(row.split("|", 1)[0], _ONLY)
    ]
    assert len(rows) == 7, f"{_ONLY!r} no longer selects seven rows: {rows}"
    assert len(local_run.command) == len(rows), local_run.command

    regenerated = tmp_path / "MANIFEST.toml"
    with regenerated.open("w") as f:
        for row in rows:
            _name, outdir, prefix, _workdir, _args = row.split("|")
            f.write(f'[file."{outdir}/{prefix}.log.txt"]\n')
            f.write(f'generation_cmd = "{local_run.command[prefix]}"\n\n')

    assert cells.cells_from_manifest(regenerated) == rows


def test_a_command_with_bare_file_names_runs_in_its_fixture_directory(
    local_run: LocalRun,
) -> None:
    """gemma_lrt's recorded command names `test`, not a path from the root."""
    started_in = dict(local_run.started_in)

    assert started_in.pop("gemma_lrt") == Path("tests/fixtures/gemma_synthetic")
    assert set(started_in.values()) == {Path()}


def test_output_left_in_dot_output_moves_beside_the_fixture(
    local_run: LocalRun,
) -> None:
    """./output is under the directory GEMMA started in, not always the root."""
    fixtures = local_run.outroot / "tests" / "fixtures"

    for directory, prefix in (
        ("gemma_synthetic", "gemma_lrt"),
        ("gemma_covariate", "gemma_covariate"),
    ):
        for kind in ("assoc", "log"):
            assert (fixtures / directory / f"{prefix}.{kind}.txt").is_file()
    assert not list(local_run.outroot.rglob("output"))


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


def test_listing_from_another_uv_project_leaves_it_untouched(tmp_path: Path) -> None:
    """``uv run`` resolves its project from the working directory unless told."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "other"\nversion = "0"\nrequires-python = ">=3.11"\n'
    )

    subprocess.run(
        ["bash", str(_SCRIPT_DIR / "generate_gemma_fixtures.sh"), "--list"],
        check=True,
        cwd=tmp_path,
        env={**os.environ, "UV_NO_SYNC": "1"},
    )

    assert list(tmp_path.iterdir()) == [pyproject]
