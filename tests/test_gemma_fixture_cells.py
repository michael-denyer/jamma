"""Tests for scripts/_gemma_fixture_cells.py.

``scripts/generate_gemma_fixtures.sh`` regenerates every fixture the cell
table lists. The references under ``tests/fixtures/mathematical_*`` also
record a ``generation_cmd`` that starts with ``gemma``, but
``scripts/mathematical_validation.py`` generates them from inputs it writes
itself, so the shell script must not treat them as cells.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = _REPO_ROOT / "scripts"
_MANIFEST = _REPO_ROOT / "tests" / "fixtures" / "MANIFEST.toml"
_PYTHON_GENERATED = "tests/fixtures/mathematical_"


def _load_cells():
    """Import the cell table from ``scripts/``, which is not a package."""
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        import _gemma_fixture_cells
    finally:
        if sys.path and sys.path[0] == str(_SCRIPT_DIR):
            sys.path.pop(0)
    return _gemma_fixture_cells


def test_python_generated_reference_is_not_a_cell(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.toml"
    manifest.write_text(
        '[file."tests/fixtures/gemma_x/x.log.txt"]\n'
        'generation_cmd = "gemma -bfile data/x -lmm 1'
        ' -outdir tests/fixtures/gemma_x -o x"\n'
        '[file."tests/fixtures/mathematical_validation/tiny/gemma.log.txt"]\n'
        'generation_cmd = "gemma -bfile tiny -k kinship.txt -lmm 1'
        ' -outdir . -o gemma"\n'
    )

    cells = _load_cells().cells_from_manifest(manifest)

    assert cells == [
        "x|tests/fixtures/gemma_x|x|.|-bfile %ROOT%/data/x -lmm 1 -outdir %OUTDIR% -o x"
    ]


def test_committed_mathematical_references_stay_out_of_the_cell_table() -> None:
    with _MANIFEST.open("rb") as stream:
        entries = tomllib.load(stream)["file"]
    recorded_as_gemma = [
        path
        for path, entry in entries.items()
        if path.startswith(_PYTHON_GENERATED)
        and entry.get("generation_cmd", "").startswith("gemma ")
    ]
    # Without these entries the assertion below could not fail.
    assert recorded_as_gemma

    cells = _load_cells().cells_from_manifest(_MANIFEST)

    outdirs = [cell.split("|")[1] for cell in cells]
    assert cells
    assert not [outdir for outdir in outdirs if outdir.startswith(_PYTHON_GENERATED)]
