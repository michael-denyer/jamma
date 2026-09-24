"""JAMMA on BGEN against GEMMA on BIMBAM holding the same dosages.

``tests/fixtures/bgen_parity`` holds a BGEN built from GEMMA's mouse_hs1940
example, with fractional dosages and missing samples on a share of variants,
and GEMMA 0.98.5's ``-gk 1`` and ``-lmm 1..4`` outputs on a BIMBAM file of
the dosages JAMMA decodes from it. Both tools therefore start from identical
doubles, so every difference comes from the analysis, and the documented
``ToleranceConfig`` bounds apply unrelaxed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from jamma.cli import main
from jamma.kinship.io import read_kinship_matrix
from jamma.validation import ToleranceConfig, compare_assoc_results
from jamma.validation.compare import compare_kinship_matrices, load_gemma_assoc
from tests.fixture_paths import BGEN_PARITY
from tests.support import require_fixture, requires_c

pytestmark = [pytest.mark.tier1, requires_c]

runner = CliRunner()


def _jamma(args: list[str]) -> None:
    result = runner.invoke(
        main,
        [
            "-bgen",
            str(BGEN_PARITY.bgen),
            "-p",
            str(BGEN_PARITY.phenotypes),
            *args,
        ],
    )
    assert result.exit_code == 0, result.output


def test_kinship_matches_gemma(tmp_path: Path):
    require_fixture(*BGEN_PARITY.paths)
    _jamma(["-gk", "1", "-o", "k", "-outdir", str(tmp_path)])

    comparison = compare_kinship_matrices(
        read_kinship_matrix(tmp_path / "k.cXX.npy"),
        np.loadtxt(BGEN_PARITY.kinship),
    )
    assert comparison.passed, comparison.message


@pytest.mark.parametrize(
    ("lmm_mode", "run"), [(1, "wald"), (2, "lrt"), (3, "score"), (4, "all")]
)
def test_association_matches_gemma(tmp_path: Path, lmm_mode: int, run: str):
    require_fixture(*BGEN_PARITY.paths)
    # A copy, so reading the text kinship leaves no .npy sidecar in fixtures.
    kinship = tmp_path / BGEN_PARITY.kinship.name
    shutil.copyfile(BGEN_PARITY.kinship, kinship)
    _jamma(
        ["-k", str(kinship), "-lmm", str(lmm_mode), "-o", run, "-outdir", str(tmp_path)]
    )

    expected = load_gemma_assoc(BGEN_PARITY.assoc[run])
    comparison = compare_assoc_results(
        load_gemma_assoc(tmp_path / f"{run}.assoc.txt"), expected, ToleranceConfig()
    )
    assert comparison.passed, f"-lmm {lmm_mode} on BGEN vs GEMMA:\n{comparison}"
