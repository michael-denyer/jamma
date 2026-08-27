"""Tests for JAMMA CLI."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from jamma.cli import main
from tests.fakes import FakePipelineRunnerFactory

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
        "--no-telemetry",
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
@pytest.mark.parametrize("flag", ["-wsnp", "-gxe", "-mk", "-mvlmm"])
def test_cli_rejects_gemma_flags_jamma_does_not_implement(flag: str):
    """GEMMA flags with no JAMMA implementation are unknown options, not stubs.

    GEMMA's -vc is not in the list: click reads it as the short flags -v -c,
    so it fails on the covariate file instead of as an unknown option.
    """
    result = runner.invoke(main, ["-bfile", str(EXAMPLE_BFILE), "-gk", "1", flag, "1"])
    assert result.exit_code == 2
    assert "no such option" in result.output.lower()
    assert "not yet implemented" not in result.output.lower()


@pytest.mark.tier1
def test_lmin_validation():
    """CLI rejects invalid -lmin values."""
    # lmin = 0 should fail
    result = runner.invoke(
        main,
        ["-bfile", str(EXAMPLE_BFILE), "-lmm", "1", "-k", "fake.txt", "-lmin", "0"],
    )
    assert result.exit_code == 2
    assert "l_min must be positive" in result.output

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
    assert "l_min must be positive" in result.output


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
    assert "l_max (0.0001) must be greater than l_min (0.001)" in result.output


@pytest.mark.tier1
def test_invalid_lmm_mode_reports_cli_error():
    """A knob rejected at config construction reads as a usage error.

    PipelineConfig validates its knobs in __post_init__, so -lmm 99 raises
    before the runner starts. The construction must sit inside the handler
    that turns ValueError into a usage error, or the user gets a traceback.
    """
    result = runner.invoke(
        main,
        ["-bfile", str(EXAMPLE_BFILE), "-lmm", "99", "-k", "fake.txt"],
    )
    assert result.exit_code == 2
    assert "lmm_mode must be" in result.output
    assert "Traceback" not in result.output


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
def test_gk_mode_outside_1_2_is_a_usage_error():
    """-gk 3 is rejected by the option itself, before any file is read."""
    result = runner.invoke(main, ["-bfile", str(EXAMPLE_BFILE), "-gk", "3"])
    assert result.exit_code == 2
    assert "-gk" in result.output
    assert "1<=x<=2" in result.output


@pytest.mark.tier1
def test_gk2_with_loco_reports_cli_error(tmp_path: Path):
    """The -gk 2 with -loco guard moved into compute_kinship; still a CLI error."""
    result = runner.invoke(
        main,
        ["-outdir", str(tmp_path), "-bfile", str(EXAMPLE_BFILE), "-gk", "2", "-loco"],
    )
    assert result.exit_code == 1
    assert "-gk 2 (standardized) is not supported with -loco" in result.output


@pytest.mark.tier1
def test_cli_help_shows_widv():
    """CLI --help shows -widv flag."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "-widv" in result.output
    assert "weight" in result.output.lower()


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
    assert result.exit_code == 2
    assert "-cat requires -c" in result.output


@pytest.mark.tier1
def test_cat_comma_separated_reaches_pipeline_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-cat '1,3' parses like -n: commas and spaces both separate indices."""
    factory = FakePipelineRunnerFactory(result=_mock_pipeline_result(tmp_path / "out"))
    monkeypatch.setattr("jamma.cli.PipelineRunner", factory)
    cov = tmp_path / "cov.txt"
    cov.write_text("1 0 2\n1 1 3\n")
    result = runner.invoke(
        main,
        [
            "-bfile",
            str(EXAMPLE_BFILE),
            "-lmm",
            "1",
            "-k",
            str(KINSHIP_FILE),
            "-c",
            str(cov),
            "-cat",
            "1,3",
            "--no-check-memory",
        ],
    )
    assert result.exit_code == 0, result.output
    assert factory.last_config.cat_columns == [1, 3]


@pytest.mark.tier1
@pytest.mark.parametrize("value", ["x", ""])
def test_cat_rejects_non_integer_and_empty(value: str) -> None:
    """-cat with a non-integer or nothing is a usage error naming the flag."""
    result = runner.invoke(
        main,
        ["-bfile", str(EXAMPLE_BFILE), "-lmm", "1", "-k", "k.txt", "-cat", value],
    )
    assert result.exit_code == 2
    assert "-cat" in result.output


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
# Multi-phenotype -n parsing tests
# ===========================================================================


@pytest.mark.tier1
class TestMultiNParsing:
    """Tests for CLI -n multi-value parsing."""

    def test_multi_n_space_separated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI -n '1 2 3' parses to phenotype_columns=[1, 2, 3]."""
        outdir = tmp_path / "output"
        factory = FakePipelineRunnerFactory(result=_mock_pipeline_result(outdir))
        monkeypatch.setattr("jamma.cli.PipelineRunner", factory)

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
        assert factory.last_config.phenotype_columns == [1, 2, 3]

    def test_multi_n_comma_separated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI -n '1,2,3' parses to phenotype_columns=[1, 2, 3]."""
        outdir = tmp_path / "output"
        factory = FakePipelineRunnerFactory(result=_mock_pipeline_result(outdir))
        monkeypatch.setattr("jamma.cli.PipelineRunner", factory)

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
        assert factory.last_config.phenotype_columns == [1, 2, 3]

    def test_single_n_backward_compat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI -n 1 still works as before."""
        outdir = tmp_path / "output"
        factory = FakePipelineRunnerFactory(result=_mock_pipeline_result(outdir))
        monkeypatch.setattr("jamma.cli.PipelineRunner", factory)

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
        assert factory.last_config.phenotype_columns == [1]

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
        n_covariates=1,
    )


@pytest.mark.tier1
@pytest.mark.slow
def test_cli_gk_end_to_end(tmp_path: Path):
    """CLI -gk 1 computes kinship and writes output file."""
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
            "-o",
            "result",
        ],
    )

    assert result.exit_code == 0, f"CLI gk failed:\n{result.output}"
    # Default output is binary .npy format (not legacy text .cXX.txt)
    kinship_path = outdir / "result.cXX.npy"
    assert kinship_path.exists(), "Kinship output file should exist"
    assert kinship_path.stat().st_size > 0, "Kinship output file should be non-empty"


@pytest.mark.tier1
@pytest.mark.slow
def test_cli_lmm_with_covariates(tmp_path: Path):
    """CLI -lmm 1 -c <covariate_file> runs end-to-end with covariates."""
    import numpy as np

    outdir = tmp_path / "output"

    # Create GEMMA-format covariate file: no header, whitespace-delimited,
    # first column = intercept (1.0), second column = random covariate.
    rng = np.random.default_rng(42)
    n_samples = 100  # gemma_synthetic test dataset sample count
    intercept = np.ones(n_samples)
    covariate = rng.standard_normal(n_samples)
    cov_path = tmp_path / "covariates.txt"
    with open(cov_path, "w") as f:
        for i in range(n_samples):
            f.write(f"{intercept[i]:.1f}\t{covariate[i]:.6f}\n")

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
            "-c",
            str(cov_path),
            "-o",
            "result",
            "--no-check-memory",
        ],
    )

    assert result.exit_code == 0, f"CLI lmm with covariates failed:\n{result.output}"
    assoc_path = outdir / "result.assoc.txt"
    assert assoc_path.exists(), "Association output file should exist"
    lines = assoc_path.read_text().strip().split("\n")
    assert len(lines) > 1, "Association file should have header + data lines"


@pytest.mark.tier1
def test_lmm_numpy_backend(tmp_path: Path):
    """CLI with --backend numpy runs LMM mode 1 end-to-end.

    Uses pre-computed GEMMA kinship so -gk is not required.
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


@pytest.mark.tier1
def test_cli_backend_numpy_streaming_accepted():
    """CLI accepts --backend numpy-streaming."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "numpy-streaming" in result.output


@pytest.mark.tier1
def test_cli_backend_numpy_streaming_wires_to_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """--backend numpy-streaming is passed through to PipelineConfig."""
    factory = FakePipelineRunnerFactory(result=_mock_pipeline_result(tmp_path / "out"))
    monkeypatch.setattr("jamma.cli.PipelineRunner", factory)

    runner.invoke(
        main,
        [
            "-lmm",
            "1",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-k",
            str(KINSHIP_FILE),
            "--backend",
            "numpy-streaming",
            "--no-check-memory",
        ],
    )

    assert len(factory.runners) == 1
    assert factory.last_config.backend == "numpy-streaming"


@pytest.mark.tier1
def test_cli_eigen_dir_without_loco_errors():
    """--eigen-dir is rejected outside -loco mode."""
    result = runner.invoke(
        main,
        [
            "-lmm",
            "1",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-k",
            str(KINSHIP_FILE),
            "--eigen-dir",
            "some/dir",
            "--no-check-memory",
        ],
    )

    assert result.exit_code != 0
    assert "--eigen-dir is only supported with -loco mode" in result.output


@pytest.mark.tier1
def test_cli_loco_eigen_defaults_eigen_dir_to_outdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """-loco -eigen without --eigen-dir defaults eigen_dir to the output dir."""
    factory = FakePipelineRunnerFactory(result=_mock_pipeline_result(tmp_path / "out"))
    monkeypatch.setattr("jamma.cli.PipelineRunner", factory)

    runner.invoke(
        main,
        [
            "-outdir",
            str(tmp_path / "out"),
            "-lmm",
            "1",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-loco",
            "-eigen",
            "--no-check-memory",
        ],
    )

    assert len(factory.runners) == 1
    # CLI default fired: without --eigen-dir, eigen_dir tracks the output dir
    # (it would be None if the default never ran).
    assert factory.last_config.eigen_dir is not None
    assert factory.last_config.eigen_dir == factory.last_config.output_dir


@pytest.mark.tier1
def test_cli_legacy_text_wires_to_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """--legacy-text is forwarded through to PipelineConfig."""
    factory = FakePipelineRunnerFactory(result=_mock_pipeline_result(tmp_path / "out"))
    monkeypatch.setattr("jamma.cli.PipelineRunner", factory)

    runner.invoke(
        main,
        [
            "-outdir",
            str(tmp_path / "out"),
            "-lmm",
            "1",
            "-bfile",
            str(EXAMPLE_BFILE),
            "-loco",
            "--legacy-text",
            "--no-check-memory",
        ],
    )

    assert len(factory.runners) == 1
    assert factory.last_config.legacy_text is True


@pytest.mark.tier1
def test_output_prefix_with_separator_reports_a_usage_error():
    """`-o a/b` must read as a usage error, not a Python traceback.

    OutputConfig rejects a prefix containing a path separator. Building it
    outside the CLI's error handling let that ValueError reach the user raw.
    """
    result = runner.invoke(main, ["-lmm", "1", "-o", "a/b", "-bfile", "nope"])
    assert result.exit_code == 2
    assert "path separators" in result.output
    assert "Traceback" not in result.output
