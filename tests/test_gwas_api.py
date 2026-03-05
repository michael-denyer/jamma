"""Tests for the top-level gwas() API function."""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma.gwas import GWASResult, gwas

# Fixture paths for mouse_hs1940 dataset
FIXTURES = Path(__file__).parent / "fixtures" / "mouse_hs1940"
BFILE = FIXTURES / "mouse_hs1940"
KINSHIP_FILE = FIXTURES / "mouse_hs1940_kinship.cXX.txt"
COVARIATE_FILE = FIXTURES / "covariates.txt"

# Fixture paths for gemma_loco dataset (3 chromosomes — required for LOCO tests)
SYNTHETIC_DIR = Path(__file__).parent / "fixtures" / "gemma_loco"
SYNTHETIC_BFILE = SYNTHETIC_DIR / "test"


@pytest.mark.slow
@pytest.mark.tier1
@pytest.mark.requires_jax
def test_gwas_basic(tmp_path: Path) -> None:
    """gwas() with pre-computed kinship returns valid GWASResult and writes output."""
    result = gwas(
        BFILE,
        kinship_file=KINSHIP_FILE,
        output_dir=tmp_path,
        show_progress=False,
        check_memory=False,
    )

    assert isinstance(result, GWASResult)
    assert result.n_samples > 0
    assert result.n_snps_tested > 0
    assert "kinship_s" in result.timing
    assert "lmm_s" in result.timing
    assert "total_s" in result.timing
    assert result.timing["total_s"] > 0

    # Check output file exists and is non-empty
    assoc_file = tmp_path / "result.assoc.txt"
    assert assoc_file.exists()
    lines = assoc_file.read_text().strip().splitlines()
    assert len(lines) > 1  # Header + at least one data line


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_gwas_custom_prefix(tmp_path: Path) -> None:
    """gwas() writes output with the specified prefix."""
    result = gwas(
        BFILE,
        kinship_file=KINSHIP_FILE,
        output_dir=tmp_path,
        output_prefix="custom",
        show_progress=False,
        check_memory=False,
    )

    assert isinstance(result, GWASResult)
    assert (tmp_path / "custom.assoc.txt").exists()


@pytest.mark.slow
@pytest.mark.tier1
@pytest.mark.requires_jax
def test_gwas_save_kinship(tmp_path: Path) -> None:
    """gwas() computes and saves kinship when save_kinship=True."""
    result = gwas(
        BFILE,
        save_kinship=True,
        output_dir=tmp_path,
        show_progress=False,
        check_memory=False,
    )

    # Kinship file should exist and be non-empty (default is binary .npy)
    kinship_path = tmp_path / "result.cXX.npy"
    assert kinship_path.exists()
    assert kinship_path.stat().st_size > 0

    # Full pipeline should have completed
    assert (tmp_path / "result.assoc.txt").exists()
    assert result.n_samples > 0


@pytest.mark.slow
@pytest.mark.tier1
@pytest.mark.requires_jax
def test_gwas_with_precomputed_kinship(tmp_path: Path) -> None:
    """Loading pre-computed kinship skips computation (just file read)."""
    result = gwas(
        BFILE,
        kinship_file=KINSHIP_FILE,
        output_dir=tmp_path,
        show_progress=False,
        check_memory=False,
    )

    # File read is <1s normally but I/O contention under pytest-xdist
    # can push it higher — use generous threshold to avoid flake
    assert result.timing["kinship_s"] < 10.0
    assert result.n_samples > 0


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
    """gwas and GWASResult are importable from the top-level jamma package."""
    from jamma import GWASResult as GR
    from jamma import gwas as g

    assert callable(g)
    assert hasattr(GR, "__dataclass_fields__")


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

    assert isinstance(result, GWASResult)
    assert result.n_samples > 0
    assert result.n_snps_tested > 0

    # Output file should exist and contain results
    assoc_file = tmp_path / "result.assoc.txt"
    assert assoc_file.exists()
    lines = assoc_file.read_text().strip().splitlines()
    assert len(lines) > 1  # Header + data
