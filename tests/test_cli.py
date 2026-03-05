"""Tests for JAMMA CLI."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from jamma.cli import main

runner = CliRunner()

# Path to example PLINK data
EXAMPLE_BFILE = Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


@pytest.mark.tier1
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
        "--profile-dir",
    ],
)
def test_cli_help_shows_flag(flag: str):
    """All CLI flags appear in --help output."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert flag in result.output


@pytest.mark.tier1
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


@pytest.mark.tier1
def test_cli_version():
    """Test that --version shows version number."""
    import jamma

    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert jamma.__version__ in result.output


@pytest.mark.tier1
@pytest.mark.requires_jax
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


@pytest.mark.tier1
@pytest.mark.requires_jax
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


@pytest.mark.tier1
def test_cli_gk_invalid_bfile(tmp_path: Path):
    """Test that gk command fails gracefully with nonexistent bfile."""
    outdir = tmp_path / "output"
    fake_bfile = tmp_path / "nonexistent"

    result = runner.invoke(
        main, ["-outdir", str(outdir), "-gk", "1", "-bfile", str(fake_bfile)]
    )

    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "error" in result.output.lower()


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_cli_gk_custom_outdir(tmp_path: Path):
    """Test that gk command respects custom -outdir."""
    custom_dir = tmp_path / "custom_output_dir"

    result = runner.invoke(
        main, ["-outdir", str(custom_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )

    assert result.exit_code == 0
    assert custom_dir.exists()
    assert (custom_dir / "result.log.txt").exists()


@pytest.mark.tier1
@pytest.mark.requires_jax
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


@pytest.mark.tier1
def test_cli_lmm_requires_kinship():
    """Test that lmm command requires -k (kinship) flag."""
    result = runner.invoke(main, ["-lmm", "1", "-bfile", str(EXAMPLE_BFILE)])

    assert result.exit_code == 1
    assert "-k" in result.output or "kinship" in result.output.lower()


@pytest.mark.tier1
def test_cli_lmm_mode_2_accepted():
    """Test that lmm mode 2 (LRT) is accepted and doesn't show 'not implemented'."""
    result = runner.invoke(
        main, ["-lmm", "2", "-bfile", str(EXAMPLE_BFILE), "-k", "fake.txt"]
    )

    # Mode 2 is now implemented - fails on kinship file, not 'not implemented'
    assert result.exit_code == 1
    assert "not yet implemented" not in result.output.lower()
    assert "kinship matrix file not found" in result.output.lower()


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_cli_gk_mode_2_succeeds(tmp_path: Path):
    """Test that -gk 2 computes standardized kinship and writes output."""
    outdir = tmp_path / "output"
    result = runner.invoke(
        main, ["-outdir", str(outdir), "-gk", "2", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0, f"gk mode 2 failed: {result.output}"
    assert "standardized" in result.output.lower()

    # Verify output file was created (binary .npy by default)
    kinship_path = outdir / "result.cXX.npy"
    assert kinship_path.exists(), "Kinship output file should exist"

    # Verify it has correct shape (100x100)
    import numpy as np

    K = np.load(kinship_path)
    assert K.shape == (100, 100), f"Expected (100, 100) shape, got {K.shape}"


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_cli_gk_legacy_text(tmp_path: Path):
    """--legacy-text flag produces .txt kinship output instead of .npy."""
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
            "--legacy-text",
        ],
    )
    assert result.exit_code == 0
    assert (outdir / "result.cXX.txt").exists()
    assert not (outdir / "result.cXX.npy").exists()


@pytest.mark.tier1
@pytest.mark.requires_jax
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


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_lmm_jax_default_mode4(tmp_path: Path):
    """Test that default JAX backend works with -lmm 4."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # Create kinship matrix
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.npy"

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


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_lmm_with_covariate_file(tmp_path: Path):
    """Test that lmm command accepts -c option and runs with covariates."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # First, create kinship matrix
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.npy"
    assert kinship_file.exists()

    # Create covariate file with correct sample count (100 samples)
    # Column 1 = intercept, column 2 = varying covariate (must not be constant)
    cov_file = tmp_path / "covariates.txt"
    with open(cov_file, "w") as f:
        for i in range(100):
            f.write(f"1 {0.1 * (i % 10):.1f}\n")

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


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_lmm_covariate_file_not_found(tmp_path: Path):
    """Test that lmm command fails gracefully when covariate file not found."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # First, create kinship matrix
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.npy"

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


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_lmm_covariate_sample_mismatch(tmp_path: Path):
    """Test that lmm command fails when covariate row count mismatches samples."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # First, create kinship matrix
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.npy"

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


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_lmm_covariate_intercept_warning(tmp_path: Path):
    """Test that lmm command warns when covariate file lacks intercept column."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # First, create kinship matrix
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.npy"

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


@pytest.mark.tier1
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


@pytest.mark.tier1
def test_cli_gk_lmm_mutually_exclusive():
    """Providing both -gk and -lmm should fail with a usage error."""
    result = runner.invoke(
        main, ["-bfile", str(EXAMPLE_BFILE), "-gk", "1", "-lmm", "1"]
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


@pytest.mark.tier1
def test_cli_requires_gk_or_lmm():
    """Providing -bfile without -gk or -lmm should fail with a usage error."""
    result = runner.invoke(main, ["-bfile", str(EXAMPLE_BFILE)])
    assert result.exit_code == 2
    assert "One of -gk or -lmm is required" in result.output


@pytest.mark.tier1
@pytest.mark.parametrize(
    "flag,value",
    [
        ("-wsnp", "file.txt"),
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


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_cli_gk_default_no_filtering(tmp_path: Path):
    """Default gk invocation should not apply MAF/miss filtering."""
    outdir = tmp_path / "output"
    result = runner.invoke(
        main, ["-outdir", str(outdir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    assert "Filtering" not in result.output


@pytest.mark.tier1
@pytest.mark.requires_jax
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


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_lmin_lmax_flags(tmp_path: Path):
    """CLI accepts -lmin and -lmax flags and runs successfully."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # Create kinship matrix
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.npy"

    # Run LMM with custom lambda bounds
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
            "-lmin",
            "1e-3",
            "-lmax",
            "1e3",
            "--no-check-memory",
        ],
    )
    assert result.exit_code == 0
    assoc_path = outdir / "result.assoc.txt"
    assert assoc_path.exists()


@pytest.mark.tier1
def test_lmin_validation():
    """CLI rejects invalid -lmin values."""
    # lmin = 0 should fail
    result = runner.invoke(
        main,
        ["-bfile", str(EXAMPLE_BFILE), "-lmm", "1", "-k", "fake.txt", "-lmin", "0"],
    )
    assert result.exit_code == 2
    assert "-lmin must be > 0" in result.output

    # lmin = -1 should fail
    result = runner.invoke(
        main,
        [
            "-bfile",
            str(EXAMPLE_BFILE),
            "-lmm",
            "1",
            "-k",
            "fake.txt",
            "-lmin",
            "-1",
        ],
    )
    assert result.exit_code == 2
    assert "-lmin must be > 0" in result.output


@pytest.mark.tier1
def test_lmax_validation():
    """CLI rejects -lmax less than or equal to -lmin."""
    result = runner.invoke(
        main,
        [
            "-bfile",
            str(EXAMPLE_BFILE),
            "-lmm",
            "1",
            "-k",
            "fake.txt",
            "-lmin",
            "1e-3",
            "-lmax",
            "1e-4",
        ],
    )
    assert result.exit_code == 2
    assert "-lmax must be > -lmin" in result.output


@pytest.mark.tier1
def test_cli_help_shows_lmin_lmax():
    """CLI --help shows -lmin and -lmax flags with defaults."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "-lmin" in result.output
    assert "-lmax" in result.output
    assert "1e-5" in result.output
    assert "1e5" in result.output


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_widv_flag(tmp_path: Path):
    """CLI accepts -widv flag and applies weights to kinship."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # Create kinship matrix first
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.npy"

    # Create weight file with 100 samples (all 1.0 -- should not change results)
    weight_file = tmp_path / "weights.txt"
    with open(weight_file, "w") as f:
        for _ in range(100):
            f.write("1.0\n")

    # Run LMM with -widv
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
            "-widv",
            str(weight_file),
            "--no-check-memory",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert (outdir / "result.assoc.txt").exists()


@pytest.mark.tier1
def test_wsnp_not_implemented():
    """CLI rejects -wsnp with 'not yet implemented' error."""
    weight_file = "dummy_snp_weights.txt"
    result = runner.invoke(
        main,
        ["-bfile", str(EXAMPLE_BFILE), "-gk", "1", "-wsnp", weight_file],
    )
    assert result.exit_code == 1
    assert "not yet implemented" in result.output.lower()


@pytest.mark.tier1
def test_cli_help_shows_widv():
    """CLI --help shows -widv flag."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "-widv" in result.output
    assert "weight" in result.output.lower()


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_cat_flag(tmp_path: Path):
    """CLI accepts -cat flag and applies categorical encoding to covariates."""
    outdir = tmp_path / "output"
    kinship_dir = tmp_path / "kinship_out"

    # Create kinship matrix
    result = runner.invoke(
        main, ["-outdir", str(kinship_dir), "-gk", "1", "-bfile", str(EXAMPLE_BFILE)]
    )
    assert result.exit_code == 0
    kinship_file = kinship_dir / "result.cXX.npy"

    # Create covariate file: intercept + varying continuous + categorical(1,2,1,2,...)
    cov_file = tmp_path / "covariates.txt"
    with open(cov_file, "w") as f:
        for i in range(100):
            cat_val = 1 + (i % 2)  # alternating 1, 2
            f.write(f"1 {0.1 * (i % 10):.1f} {cat_val}\n")

    # Run LMM with -cat "3" to encode column 3 as categorical
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
            "-cat",
            "3",
            "--no-check-memory",
        ],
    )
    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert "Categorical encoding applied" in result.output
    assert (outdir / "result.assoc.txt").exists()


@pytest.mark.tier1
def test_cat_requires_covariate_file(tmp_path: Path):
    """CLI rejects -cat without -c (covariate file)."""
    # Use a dummy file for -k (validation fails on -cat before kinship load)
    dummy_kinship = tmp_path / "dummy_k.txt"
    dummy_kinship.write_text("0")

    result = runner.invoke(
        main,
        [
            "-bfile",
            str(EXAMPLE_BFILE),
            "-lmm",
            "1",
            "-k",
            str(dummy_kinship),
            "-cat",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "-cat requires -c" in result.output


@pytest.mark.tier1
def test_cli_help_shows_cat():
    """CLI --help shows -cat flag."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "-cat" in result.output
    assert "Categorical" in result.output


# Path to pre-computed GEMMA kinship for synthetic data (avoids -gk step in NumPy CI)
KINSHIP_FILE = (
    Path(__file__).parent / "fixtures" / "gemma_synthetic" / "gemma_kinship.cXX.txt"
)


# ===========================================================================
# Multi-phenotype -n parsing tests (Phase 59-02)
# ===========================================================================


@pytest.mark.tier1
class TestMultiNParsing:
    """Tests for CLI -n multi-value parsing."""

    def test_multi_n_space_separated(self, tmp_path: Path) -> None:
        """CLI -n '1 2 3' parses to phenotype_columns=[1, 2, 3]."""
        from unittest.mock import patch

        outdir = tmp_path / "output"
        with patch("jamma.cli.PipelineRunner") as mock_runner_cls:
            mock_runner_cls.return_value.run.return_value = _mock_pipeline_result(
                outdir
            )
            result = runner.invoke(
                main,
                [
                    "-bfile",
                    str(EXAMPLE_BFILE),
                    "-lmm",
                    "1",
                    "-k",
                    str(KINSHIP_FILE),
                    "-n",
                    "1 2 3",
                    "-outdir",
                    str(outdir),
                    "--no-check-memory",
                ],
            )
            assert result.exit_code == 0, result.output
            config = mock_runner_cls.call_args[0][0]
            assert config.phenotype_columns == [1, 2, 3]

    def test_multi_n_comma_separated(self, tmp_path: Path) -> None:
        """CLI -n '1,2,3' parses to phenotype_columns=[1, 2, 3]."""
        from unittest.mock import patch

        outdir = tmp_path / "output"
        with patch("jamma.cli.PipelineRunner") as mock_runner_cls:
            mock_runner_cls.return_value.run.return_value = _mock_pipeline_result(
                outdir
            )
            result = runner.invoke(
                main,
                [
                    "-bfile",
                    str(EXAMPLE_BFILE),
                    "-lmm",
                    "1",
                    "-k",
                    str(KINSHIP_FILE),
                    "-n",
                    "1,2,3",
                    "-outdir",
                    str(outdir),
                    "--no-check-memory",
                ],
            )
            assert result.exit_code == 0, result.output
            config = mock_runner_cls.call_args[0][0]
            assert config.phenotype_columns == [1, 2, 3]

    def test_single_n_backward_compat(self, tmp_path: Path) -> None:
        """CLI -n 1 still works as before."""
        from unittest.mock import patch

        outdir = tmp_path / "output"
        with patch("jamma.cli.PipelineRunner") as mock_runner_cls:
            mock_runner_cls.return_value.run.return_value = _mock_pipeline_result(
                outdir
            )
            result = runner.invoke(
                main,
                [
                    "-bfile",
                    str(EXAMPLE_BFILE),
                    "-lmm",
                    "1",
                    "-k",
                    str(KINSHIP_FILE),
                    "-n",
                    "1",
                    "-outdir",
                    str(outdir),
                    "--no-check-memory",
                ],
            )
            assert result.exit_code == 0, result.output
            config = mock_runner_cls.call_args[0][0]
            assert config.phenotype_columns == [1]

    def test_duplicate_n_error(self) -> None:
        """CLI -n '1 1 3' produces a clear error about duplicates."""
        result = runner.invoke(
            main,
            [
                "-bfile",
                str(EXAMPLE_BFILE),
                "-lmm",
                "1",
                "-k",
                str(KINSHIP_FILE),
                "-n",
                "1 1 3",
            ],
        )
        assert result.exit_code != 0
        assert "duplicate" in result.output.lower()

    def test_invalid_n_error(self) -> None:
        """CLI -n 'abc' produces a clear error."""
        result = runner.invoke(
            main,
            [
                "-bfile",
                str(EXAMPLE_BFILE),
                "-lmm",
                "1",
                "-k",
                str(KINSHIP_FILE),
                "-n",
                "abc",
            ],
        )
        assert result.exit_code != 0
        assert "integer" in result.output.lower()

    def test_empty_n_error(self) -> None:
        """CLI -n '' produces a clear error."""
        result = runner.invoke(
            main,
            [
                "-bfile",
                str(EXAMPLE_BFILE),
                "-lmm",
                "1",
                "-k",
                str(KINSHIP_FILE),
                "-n",
                "",
            ],
        )
        assert result.exit_code != 0

    def test_multi_n_with_gk_error(self) -> None:
        """CLI -n '1 2' -gk 1 produces a clear error."""
        result = runner.invoke(
            main,
            [
                "-bfile",
                str(EXAMPLE_BFILE),
                "-gk",
                "1",
                "-n",
                "1 2",
            ],
        )
        assert result.exit_code != 0
        assert "not supported" in result.output.lower()


def _mock_pipeline_result(outdir: Path):
    """Create a minimal mock PipelineResult for CLI tests."""
    from jamma.pipeline import PipelineResult

    outdir.mkdir(parents=True, exist_ok=True)
    assoc_path = outdir / "result.assoc.txt"
    assoc_path.write_text("chr\trs\tps\tn_miss\tn_obs\n")
    return PipelineResult(
        associations=[],
        n_samples=100,
        n_snps_tested=500,
        assoc_path=assoc_path,
        assoc_paths=[assoc_path],
        timing={"total_s": 1.0, "load_s": 0.1, "lmm_s": 0.9},
        backend="numpy",
        n_covariates=1,
    )


@pytest.mark.tier1
def test_lmm_numpy_backend(tmp_path: Path):
    """CLI with --backend numpy runs LMM mode 1 end-to-end without JAX.

    Uses the pre-computed GEMMA kinship fixture so -gk is not required.
    Not marked @requires_jax so this runs in NumPy-only CI.
    """
    outdir = tmp_path / "output"

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
            str(KINSHIP_FILE),
            "--backend",
            "numpy",
            "--no-check-memory",
        ],
    )

    assert result.exit_code == 0, f"CLI NumPy backend failed:\n{result.output}"
    assoc_path = outdir / "result.assoc.txt"
    assert assoc_path.exists(), "Association output file should exist"
    lines = assoc_path.read_text().strip().split("\n")
    assert len(lines) > 1, "Association file should have more than just the header line"
