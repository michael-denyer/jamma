"""Tests for JAMMA CLI."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from jamma.cli import main

runner = CliRunner()

# Path to example PLINK data
EXAMPLE_BFILE = Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


@pytest.mark.parametrize(
    "flag",
    [
        "-bfile",
        "-gk",
        "-lmm",
        "-k",
        "-c",
        "-o",
        "-outdir",
        "-maf",
        "-miss",
        "-loco",
        "-eigen",
        "-n",
        "-d",
        "-u",
        "-hwe",
        "-snps",
        "-ksnps",
    ],
)
def test_cli_help_shows_flag(flag: str):
    """All CLI flags appear in --help output."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert flag in result.output


@pytest.mark.parametrize(
    "flag,description_fragment",
    [
        ("-c", "Covariate"),
        ("-n", "Phenotype column"),
    ],
)
def test_cli_help_shows_description(flag: str, description_fragment: str):
    """Flag descriptions contain expected text."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert flag in result.output
    assert description_fragment in result.output


def test_cli_version():
    """Test that --version shows version number."""
    import jamma

    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert jamma.__version__ in result.output


def test_cli_gk_loads_data(tmp_path: Path):
    """Test that gk command loads PLINK data and creates output directory."""
    outdir = tmp_path / "output"

    result = runner.invoke(
        main, ["-outdir", str(outdir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )

    assert result.exit_code == 0
    assert outdir.exists()
    assert "100 samples" in result.output
    assert "500 SNPs" in result.output


def test_cli_gk_log_file(tmp_path: Path):
    """Test that gk command creates GEMMA-format log file."""
    outdir = tmp_path / "output"

    result = runner.invoke(
        main, ["-outdir", str(outdir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
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
        main, ["-outdir", str(outdir), "-gk", "1", "-bfile", str(fake_bfile)]
    )

    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_cli_gk_custom_outdir(tmp_path: Path):
    """Test that gk command respects custom -outdir."""
    custom_dir = tmp_path / "custom_output_dir"

    result = runner.invoke(
        main, ["-outdir", str(custom_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )

    assert result.exit_code == 0
    assert custom_dir.exists()
    assert (custom_dir / "result.log.txt").exists()


def test_cli_gk_custom_prefix(tmp_path: Path):
    """Test that gk command respects custom -o prefix."""
    outdir = tmp_path / "output"

    result = runner.invoke(
        main,
        [
            "-outdir",
            str(outdir),
            "-o",
            "myprefix",
            "-gk",
            "1",
            "-bfile",
            str(EXAMPLE_BFILE),
        ],
    )

    assert result.exit_code == 0
    assert (outdir / "myprefix.log.txt").exists()


def test_cli_lmm_requires_kinship():
    """Test that lmm command requires -k (kinship) flag."""
    result = runner.invoke(main, ["-lmm", "1", "-bfile", str(EXAMPLE_BFILE)])

    assert result.exit_code == 1
    assert "-k" in result.output or "kinship" in result.output.lower()


def test_cli_lmm_mode_2_accepted():
    """Test that lmm mode 2 (LRT) is accepted and doesn't show 'not implemented'."""
    result = runner.invoke(
        main, ["-lmm", "2", "-bfile", str(EXAMPLE_BFILE), "-k", "fake.txt"]
    )

    # Mode 2 is now implemented - fails on kinship file, not 'not implemented'
    assert result.exit_code == 1
    assert "not yet implemented" not in result.output.lower()
    assert "kinship matrix file not found" in result.output.lower()


def test_cli_gk_mode_2_succeeds(tmp_path: Path):
    """Test that -gk 2 computes standardized kinship and writes output."""
    outdir = tmp_path / "output"
    result = runner.invoke(
        main, ["-outdir", str(outdir), "-gk", "2", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0, f"gk mode 2 failed: {result.output}"
    assert "standardized" in result.output.lower()

    # Verify output file was created
    kinship_path = outdir / "result.cXX.txt"
    assert kinship_path.exists(), "Kinship output file should exist"

    # Verify it has content (100 lines for 100 samples)
    lines = kinship_path.read_text().strip().split("\n")
    assert len(lines) == 100, f"Expected 100 lines, got {len(lines)}"


def test_cli_gk_maf_miss_flags(tmp_path: Path):
    """Test that gk command accepts -maf and -miss flags."""
    outdir = tmp_path / "output"

    # Run with MAF and missing filters
    result = runner.invoke(
        main,
        [
            "-outdir",
            str(outdir),
            "-gk",
            "1",
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


def test_lmm_jax_default_mode4(tmp_path: Path):
    """Test that default JAX backend works with -lmm 4."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # Create kinship matrix
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.txt"

    # Run LMM mode 4 with JAX (default)
    result = runner.invoke(
        main,
        [
            "-outdir",
            str(outdir),
            "-lmm",
            "4",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-k",
            str(kinship_file),
            "--no-check-memory",
        ],
    )

    assert result.exit_code == 0
    assoc_path = outdir / "result.assoc.txt"
    assert assoc_path.exists()

    # Verify all-tests output has expected columns
    content = assoc_path.read_text()
    header = content.split("\n")[0]
    assert "p_wald" in header
    assert "p_lrt" in header
    assert "p_score" in header


def test_lmm_with_covariate_file(tmp_path: Path):
    """Test that lmm command accepts -c option and runs with covariates."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # First, create kinship matrix
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
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
        main,
        [
            "-outdir",
            str(outdir),
            "-lmm",
            "1",
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
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.txt"

    # Run LMM with nonexistent covariate file
    fake_cov = tmp_path / "nonexistent_covariates.txt"
    result = runner.invoke(
        main,
        [
            "-outdir",
            str(outdir),
            "-lmm",
            "1",
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
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
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
        main,
        [
            "-outdir",
            str(outdir),
            "-lmm",
            "1",
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
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
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
        main,
        [
            "-outdir",
            str(outdir),
            "-lmm",
            "1",
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


def test_cli_gk_ksnps_missing_file_error(tmp_path: Path):
    """CLI gk command exits gracefully when -ksnps file doesn't exist."""
    result = runner.invoke(
        main,
        [
            "-outdir",
            str(tmp_path),
            "-gk",
            "1",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-ksnps",
            str(tmp_path / "nonexistent.txt"),
        ],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_cli_gk_lmm_mutually_exclusive():
    """Providing both -gk and -lmm should fail with a usage error."""
    result = runner.invoke(
        main, ["-bfile", str(EXAMPLE_BFILE), "-gk", "1", "-lmm", "1"]
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_cli_requires_gk_or_lmm():
    """Providing -bfile without -gk or -lmm should fail with a usage error."""
    result = runner.invoke(main, ["-bfile", str(EXAMPLE_BFILE)])
    assert result.exit_code == 2
    assert "One of -gk or -lmm is required" in result.output


@pytest.mark.parametrize(
    "flag,value",
    [
        ("-gxe", "file.txt"),
        ("-vc", "1"),
        ("-mk", "file.txt"),
        ("-mvlmm", "1"),
    ],
)
def test_cli_unimplemented_flags_error(flag: str, value: str):
    """Unimplemented flags should produce a clear error message."""
    result = runner.invoke(
        main, ["-bfile", str(EXAMPLE_BFILE), "-gk", "1", flag, value]
    )
    assert result.exit_code == 1
    assert "not yet implemented" in result.output.lower()


def test_cli_gk_default_no_filtering(tmp_path: Path):
    """Default gk invocation should not apply MAF/miss filtering."""
    outdir = tmp_path / "output"
    result = runner.invoke(
        main, ["-outdir", str(outdir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    assert "Filtering" not in result.output


def test_cli_gk_explicit_maf_applies_filtering(tmp_path: Path):
    """Explicit -maf with -gk should apply filtering."""
    outdir = tmp_path / "output"
    result = runner.invoke(
        main,
        [
            "-outdir",
            str(outdir),
            "-gk",
            "1",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-maf",
            "0.05",
        ],
    )
    assert result.exit_code == 0
    assert "Filtering" in result.output
    assert "MAF >= 0.05" in result.output
