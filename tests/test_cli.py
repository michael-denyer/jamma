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


def test_cli_lmm_help_shows_covariate_flag():
    """Test that lmm --help shows -c option."""
    result = runner.invoke(app, ["lmm", "--help"])
    assert result.exit_code == 0
    assert "-c" in result.output
    assert "Covariate" in result.output


def test_lmm_with_covariate_file(tmp_path: Path):
    """Test that lmm command accepts -c option and runs with covariates."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # First, create kinship matrix
    result = runner.invoke(
        app, ["-outdir", str(kinship_dir), "gk", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.txt"
    assert kinship_file.exists()

    # Create covariate file with correct sample count (100 samples)
    cov_file = tmp_path / "covariates.txt"
    with open(cov_file, "w") as f:
        for _ in range(100):
            f.write("1 0.5\n")  # Intercept + one covariate

    # Run LMM with covariates
    result = runner.invoke(
        app,
        [
            "-outdir",
            str(outdir),
            "lmm",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-k",
            str(kinship_file),
            "-c",
            str(cov_file),
        ],
    )

    assert result.exit_code == 0
    assert "Loading covariates" in result.output
    assert "Loaded 2 covariates" in result.output


def test_lmm_covariate_file_not_found(tmp_path: Path):
    """Test that lmm command fails gracefully when covariate file not found."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # First, create kinship matrix
    result = runner.invoke(
        app, ["-outdir", str(kinship_dir), "gk", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.txt"

    # Run LMM with nonexistent covariate file
    fake_cov = tmp_path / "nonexistent_covariates.txt"
    result = runner.invoke(
        app,
        [
            "-outdir",
            str(outdir),
            "lmm",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-k",
            str(kinship_file),
            "-c",
            str(fake_cov),
        ],
    )

    assert result.exit_code == 1
    assert "Covariate file not found" in result.output


def test_lmm_covariate_sample_mismatch(tmp_path: Path):
    """Test that lmm command fails when covariate row count mismatches samples."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # First, create kinship matrix
    result = runner.invoke(
        app, ["-outdir", str(kinship_dir), "gk", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.txt"

    # Create covariate file with WRONG sample count (50 instead of 100)
    cov_file = tmp_path / "bad_covariates.txt"
    with open(cov_file, "w") as f:
        for _ in range(50):  # Wrong number of rows
            f.write("1 0.5\n")

    # Run LMM with mismatched covariates
    result = runner.invoke(
        app,
        [
            "-outdir",
            str(outdir),
            "lmm",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-k",
            str(kinship_file),
            "-c",
            str(cov_file),
        ],
    )

    assert result.exit_code == 1
    assert "50 rows" in result.output
    assert "100 samples" in result.output


def test_lmm_covariate_intercept_warning(tmp_path: Path):
    """Test that lmm command warns when covariate file lacks intercept column."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # First, create kinship matrix
    result = runner.invoke(
        app, ["-outdir", str(kinship_dir), "gk", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.txt"

    # Create covariate file WITHOUT intercept (first column not all 1s)
    cov_file = tmp_path / "no_intercept.txt"
    with open(cov_file, "w") as f:
        for i in range(100):
            # First column varies (age), NOT intercept
            f.write(f"{20 + i % 50} 0.5\n")

    # Run LMM - should succeed but warn
    result = runner.invoke(
        app,
        [
            "-outdir",
            str(outdir),
            "lmm",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-k",
            str(kinship_file),
            "-c",
            str(cov_file),
        ],
    )

    assert result.exit_code == 0
    assert "Warning" in result.output
    assert "intercept" in result.output.lower()
