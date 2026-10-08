"""``show_progress=False`` reaches the eigendecomposition progress bar."""

from __future__ import annotations

import io
from pathlib import Path

import progressbar
import pytest

from jamma.gwas import gwas
from jamma.lmm import LmmConfig, run_lmm_association_numpy
from jamma.pipeline import PipelineConfig
from jamma.pipeline_kinship import compute_kinship
from tests.fixture_paths import LOCO

pytestmark = pytest.mark.tier0


@pytest.fixture
def bar_output(monkeypatch) -> io.StringIO:
    # progressbar2 swaps ``fd=sys.stdout`` for the stream it saved at import,
    # so that saved stream is the one every bar writes to.
    stream = io.StringIO()
    monkeypatch.setattr(progressbar.utils.streams, "original_stdout", stream)
    return stream


def test_gwas_draws_the_eigendecomp_bar_when_progress_is_on(
    bar_output: io.StringIO, tmp_path: Path
) -> None:
    gwas(LOCO.bfile, output_dir=tmp_path, check_memory=False, no_telemetry=True)

    assert "Eigendecomp" in bar_output.getvalue()


def test_gwas_draws_no_eigendecomp_bar_when_progress_is_off(
    bar_output: io.StringIO, tmp_path: Path
) -> None:
    gwas(
        LOCO.bfile,
        output_dir=tmp_path,
        show_progress=False,
        check_memory=False,
        no_telemetry=True,
    )

    assert "Eigendecomp" not in bar_output.getvalue()


def test_kinship_write_eigen_draws_no_bar_when_progress_is_off(
    bar_output: io.StringIO, tmp_path: Path
) -> None:
    config = PipelineConfig(
        bfile=LOCO.bfile,
        output_dir=tmp_path,
        write_eigen=True,
        show_progress=False,
        check_memory=False,
    )

    compute_kinship(config, 1)

    assert "Eigendecomp" not in bar_output.getvalue()


def test_runner_draws_no_eigendecomp_bar_when_progress_is_off(
    bar_output: io.StringIO, synthetic_data
) -> None:
    genotypes, kinship, phenotypes, snp_info = synthetic_data

    run_lmm_association_numpy(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        config=LmmConfig(lmm_mode=1, show_progress=False),
    )

    assert "Eigendecomp" not in bar_output.getvalue()
