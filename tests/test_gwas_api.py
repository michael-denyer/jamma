"""Tests for the top-level gwas() API function."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

import pytest

from jamma.gwas import gwas
from jamma.pipeline import PipelineConfig, PipelineResult
from tests.fixture_paths import LOCO, MOUSE

BFILE = MOUSE.bfile
KINSHIP_FILE = MOUSE.kinship
COVARIATE_FILE = MOUSE.covariates
SYNTHETIC_BFILE = LOCO.bfile


@pytest.mark.slow
@pytest.mark.tier1
def test_gwas_basic(tmp_path: Path) -> None:
    """gwas() with pre-computed kinship returns a PipelineResult and writes output."""
    result = gwas(
        BFILE,
        kinship_file=KINSHIP_FILE,
        output_dir=tmp_path,
        show_progress=False,
        check_memory=False,
    )

    assert isinstance(result, PipelineResult)
    assert result.n_samples > 0
    assert result.n_snps_tested > 0
    assert result.timing.kinship_s >= 0
    assert result.timing.lmm_s >= 0
    assert result.timing.total_s > 0

    # Check output file exists and is non-empty
    assoc_file = tmp_path / "result.assoc.txt"
    assert assoc_file.exists()
    lines = assoc_file.read_text().strip().splitlines()
    assert len(lines) > 1  # Header + at least one data line


@pytest.mark.tier0
def test_gwas_missing_bfile() -> None:
    """gwas() raises FileNotFoundError for nonexistent bfile."""
    with pytest.raises(FileNotFoundError):
        gwas("/nonexistent/path", check_memory=False, show_progress=False)


@pytest.mark.tier0
def test_gwas_invalid_lmm_mode() -> None:
    """gwas() raises ValueError for invalid lmm_mode."""
    with pytest.raises(ValueError, match="lmm_mode must be"):
        gwas(BFILE, lmm_mode=99, check_memory=False, show_progress=False)


@pytest.mark.tier0
def test_gwas_import_from_top_level() -> None:
    """gwas and PipelineResult are importable from the top-level jamma package."""
    from jamma import PipelineResult as PR
    from jamma import gwas as g

    assert callable(g)
    assert PR is PipelineResult


@pytest.mark.tier0
def test_gwas_keywords_are_exactly_the_pipeline_config_fields() -> None:
    """Every PipelineConfig knob is a gwas() keyword, and nothing else is.

    The API mirrors the config by hand, so a field added to one and not the
    other is the drift this pins. ``hwe`` is the one renamed keyword (it is
    GEMMA's flag name; the field says what it thresholds).
    """
    params = set(inspect.signature(gwas).parameters)
    params = (params - {"hwe"}) | {"hwe_threshold"}
    fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    assert params == fields


@pytest.mark.tier1
def test_gwas_loco_numpy_backend(tmp_path: Path) -> None:
    """gwas() with loco=True, backend='numpy' completes end-to-end."""
    result = gwas(
        SYNTHETIC_BFILE,
        loco=True,
        backend="numpy",
        output_dir=tmp_path,
        show_progress=False,
        check_memory=False,
    )

    assert isinstance(result, PipelineResult)
    assert result.n_samples > 0
    assert result.n_snps_tested > 0

    # Output file should exist and contain results
    assoc_file = tmp_path / "result.assoc.txt"
    assert assoc_file.exists()
    lines = assoc_file.read_text().strip().splitlines()
    assert len(lines) > 1  # Header + data
