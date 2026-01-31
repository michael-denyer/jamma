"""Tests for GEMMA-Next CLI."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from gemma_next.cli import app

runner = CliRunner()

# Path to example PLINK data
EXAMPLE_BFILE = Path(__file__).parent.parent / "legacy" / "example" / "mouse_hs1940"


def test_cli_help():
    """Test that --help shows usage with expected options."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "-outdir" in result.output
    assert "-o" in result.output
    assert "gk" in result.output


def test_cli_version():
    """Test that --version shows version number."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_gk_help():
    """Test that gk --help shows -bfile option."""
    result = runner.invoke(app, ["gk", "--help"])
    assert result.exit_code == 0
    assert "-bfile" in result.output
    assert "-gk" in result.output


def test_cli_gk_loads_data(tmp_path: Path):
    """Test that gk command loads PLINK data and creates output directory."""
    outdir = tmp_path / "output"

    result = runner.invoke(
        app, ["-outdir", str(outdir), "gk", "-bfile", str(EXAMPLE_BFILE)]
    )

    assert result.exit_code == 0
    assert outdir.exists()
    assert "1940 samples" in result.output
    assert "12226 SNPs" in result.output


def test_cli_gk_log_file(tmp_path: Path):
    """Test that gk command creates GEMMA-format log file."""
    outdir = tmp_path / "output"

    result = runner.invoke(
        app, ["-outdir", str(outdir), "gk", "-bfile", str(EXAMPLE_BFILE)]
    )

    assert result.exit_code == 0

    log_path = outdir / "result.log.txt"
    assert log_path.exists()

    log_content = log_path.read_text()
    assert "GEMMA-Next" in log_content
    assert "n_samples = 1940" in log_content
    assert "n_snps = 12226" in log_content
    assert "##" in log_content


def test_cli_gk_invalid_bfile(tmp_path: Path):
    """Test that gk command fails gracefully with nonexistent bfile."""
    outdir = tmp_path / "output"
    fake_bfile = tmp_path / "nonexistent"

    result = runner.invoke(
        app, ["-outdir", str(outdir), "gk", "-bfile", str(fake_bfile)]
    )

    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_cli_gk_custom_outdir(tmp_path: Path):
    """Test that gk command respects custom -outdir."""
    custom_dir = tmp_path / "custom_output_dir"

    result = runner.invoke(
        app, ["-outdir", str(custom_dir), "gk", "-bfile", str(EXAMPLE_BFILE)]
    )

    assert result.exit_code == 0
    assert custom_dir.exists()
    assert (custom_dir / "result.log.txt").exists()


def test_cli_gk_custom_prefix(tmp_path: Path):
    """Test that gk command respects custom -o prefix."""
    outdir = tmp_path / "output"

    result = runner.invoke(
        app,
        ["-outdir", str(outdir), "-o", "myprefix", "gk", "-bfile", str(EXAMPLE_BFILE)],
    )

    assert result.exit_code == 0
    assert (outdir / "myprefix.log.txt").exists()


def test_cli_lmm_placeholder():
    """Test that lmm command shows not implemented message."""
    result = runner.invoke(app, ["lmm", "-bfile", str(EXAMPLE_BFILE)])

    assert result.exit_code == 0
    assert "not yet implemented" in result.output.lower()
