"""Tests for JAMMA CLI."""

from pathlib import Path

from typer.testing import CliRunner

from jamma.cli import app

runner = CliRunner()

# Path to example PLINK data
EXAMPLE_BFILE = Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


def test_cli_help():
    """Test that --help shows usage with expected options."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "-outdir" in result.output
    assert "-o" in result.output
    assert "gk" in result.output


def test_cli_version():
    """Test that --version shows version number."""
    import jamma

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert jamma.__version__ in result.output


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
    assert "100 samples" in result.output
    assert "500 SNPs" in result.output


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
    assert "JAMMA" in log_content
    assert "n_samples = 100" in log_content
    assert "n_snps = 500" in log_content
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


def test_cli_lmm_requires_kinship():
    """Test that lmm command requires -k (kinship) flag."""
    result = runner.invoke(app, ["lmm", "-bfile", str(EXAMPLE_BFILE)])

    assert result.exit_code == 1
    assert "-k" in result.output or "kinship" in result.output.lower()


def test_cli_lmm_help():
    """Test that lmm --help shows all required options."""
    result = runner.invoke(app, ["lmm", "--help"])

    assert result.exit_code == 0
    assert "-bfile" in result.output
    assert "-k" in result.output
    assert "-lmm" in result.output


def test_cli_lmm_mode_2_not_implemented():
    """Test that lmm mode 2 shows not implemented message."""
    result = runner.invoke(
        app, ["lmm", "-bfile", str(EXAMPLE_BFILE), "-k", "fake.txt", "-lmm", "2"]
    )

    assert result.exit_code == 1
    assert "not yet implemented" in result.output.lower()


def test_cli_gk_maf_miss_flags(tmp_path: Path):
    """Test that gk command accepts -maf and -miss flags."""
    outdir = tmp_path / "output"

    # Run with MAF and missing filters
    result = runner.invoke(
        app,
        [
            "-outdir",
            str(outdir),
            "gk",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-maf",
            "0.05",
            "-miss",
            "0.1",
        ],
    )

    assert result.exit_code == 0
    assert "Filtering" in result.output
    assert "MAF >= 0.05" in result.output

    # Verify log file contains filter parameters
    log_path = outdir / "result.log.txt"
    log_content = log_path.read_text()
    assert "maf_threshold = 0.05" in log_content
    assert "miss_threshold = 0.1" in log_content


def test_cli_gk_help_shows_filter_flags():
    """Test that gk --help shows -maf and -miss options."""
    result = runner.invoke(app, ["gk", "--help"])
    assert result.exit_code == 0
    assert "-maf" in result.output
    assert "-miss" in result.output


def test_cli_lmm_help_shows_filter_flags():
    """Test that lmm --help shows -maf and -miss options."""
    result = runner.invoke(app, ["lmm", "--help"])
    assert result.exit_code == 0
    assert "-maf" in result.output
    assert "-miss" in result.output
