"""Tests for kinship I/O and CLI integration.

These tests verify the GEMMA-format kinship output and CLI end-to-end workflow.
"""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from jamma.cli import app
from jamma.kinship import write_kinship_matrix


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def example_plink_path() -> Path:
    """Path prefix for example PLINK files."""
    return Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


class TestWriteKinshipFormat:
    """Tests for write_kinship_matrix format compliance."""

    def test_write_kinship_format(self, tmp_path):
        """Verify tab-separated output with precision 10."""
        K = np.array([[0.123456789012345, 0.987654321098765], [0.987654321098765, 0.5]])
        output_path = tmp_path / "test.cXX.txt"

        write_kinship_matrix(K, output_path)

        content = output_path.read_text()
        lines = content.strip().split("\n")

        # Check format: tab-separated
        assert "\t" in lines[0]

        # Check 2 rows for 2x2 matrix
        assert len(lines) == 2

        # Parse first row values
        values = lines[0].split("\t")
        assert len(values) == 2

        # Check precision (10 significant figures, trailing zeros may be dropped)
        # 0.123456789012345 rounds to 10 sig figs as 0.123456789
        assert values[0] == "0.123456789"
        # Verify numeric equivalence to original
        assert np.isclose(float(values[0]), 0.123456789012345, rtol=1e-9)

    def test_write_kinship_creates_directory(self, tmp_path):
        """Parent directories should be created if they don't exist."""
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        output_path = tmp_path / "nested" / "dir" / "test.cXX.txt"

        write_kinship_matrix(K, output_path)

        assert output_path.exists()

    def test_write_kinship_symmetric_matrix(self, tmp_path):
        """Symmetric matrix should write correctly."""
        K = np.array(
            [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]], dtype=np.float64
        )
        output_path = tmp_path / "test.cXX.txt"

        write_kinship_matrix(K, output_path)

        # Read back and verify
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Parse and verify symmetry
        K_read = np.array(
            [[float(v) for v in line.split("\t")] for line in lines], dtype=np.float64
        )
        assert np.allclose(K, K_read)
        assert np.allclose(K_read, K_read.T)

    def test_write_kinship_no_header(self, tmp_path):
        """Output should have no header row."""
        K = np.array([[0.5, 0.1], [0.1, 0.5]])
        output_path = tmp_path / "test.cXX.txt"

        write_kinship_matrix(K, output_path)

        first_line = output_path.read_text().split("\n")[0]
        # First line should be numeric values, not header
        values = first_line.split("\t")
        for v in values:
            float(v)  # Should not raise

    def test_write_kinship_large_values(self, tmp_path):
        """Large values should use scientific notation correctly."""
        K = np.array([[1234567890.123, 0.0], [0.0, 1.0]])
        output_path = tmp_path / "test.cXX.txt"

        write_kinship_matrix(K, output_path)

        content = output_path.read_text()
        lines = content.strip().split("\n")
        values = lines[0].split("\t")
        # Should be able to parse back
        assert np.isclose(float(values[0]), 1234567890.123, rtol=1e-9)

    def test_write_kinship_small_values(self, tmp_path):
        """Small values should preserve precision."""
        K = np.array([[1e-10, 0.0], [0.0, 1.0]])
        output_path = tmp_path / "test.cXX.txt"

        write_kinship_matrix(K, output_path)

        content = output_path.read_text()
        lines = content.strip().split("\n")
        values = lines[0].split("\t")
        # Should be able to parse back
        assert np.isclose(float(values[0]), 1e-10, rtol=1e-9)


class TestKinshipRoundtrip:
    """Tests for write-then-read consistency."""

    def test_kinship_roundtrip(self, tmp_path):
        """Written kinship should load back correctly."""
        # Create a realistic kinship matrix
        rng = np.random.default_rng(42)
        n = 10
        X = rng.random((n, 50))
        K = X @ X.T / 50  # Simple kinship-like matrix

        output_path = tmp_path / "test.cXX.txt"
        write_kinship_matrix(K, output_path)

        # Read back
        lines = output_path.read_text().strip().split("\n")
        K_read = np.array(
            [[float(v) for v in line.split("\t")] for line in lines], dtype=np.float64
        )

        # Should match original within precision limits
        # 10 significant figures means about 1e-9 relative tolerance
        assert np.allclose(K, K_read, rtol=1e-9)


class TestCLIIntegration:
    """Tests for CLI gk command integration."""

    def test_cli_gk_creates_kinship_file(self, runner, tmp_path, example_plink_path):
        """Test that gk command creates kinship file."""
        result = runner.invoke(
            app,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "gk",
                "-bfile",
                str(example_plink_path),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        kinship_file = tmp_path / "test.cXX.txt"
        assert kinship_file.exists(), "Kinship file not created"

    def test_cli_gk_creates_log_file(self, runner, tmp_path, example_plink_path):
        """Test that gk command creates log file."""
        result = runner.invoke(
            app,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "gk",
                "-bfile",
                str(example_plink_path),
            ],
        )

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        log_file = tmp_path / "test.log.txt"
        assert log_file.exists(), "Log file not created"

    def test_cli_gk_kinship_file_format(self, runner, tmp_path, example_plink_path):
        """Test that kinship file has correct format."""
        result = runner.invoke(
            app,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "gk",
                "-bfile",
                str(example_plink_path),
            ],
        )

        assert result.exit_code == 0
        kinship_file = tmp_path / "test.cXX.txt"

        # Check file has expected number of lines (100 samples)
        lines = kinship_file.read_text().strip().split("\n")
        assert len(lines) == 100, f"Expected 100 lines, got {len(lines)}"

        # Check first line has expected number of columns
        first_line_values = lines[0].split("\t")
        assert len(first_line_values) == 100, "Expected 100 columns"

    def test_cli_gk_log_contains_kinship_file(
        self, runner, tmp_path, example_plink_path
    ):
        """Test that log file mentions kinship output."""
        result = runner.invoke(
            app,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "gk",
                "-bfile",
                str(example_plink_path),
            ],
        )

        assert result.exit_code == 0
        log_file = tmp_path / "test.log.txt"
        log_content = log_file.read_text()

        # Log should contain kinship file path
        assert "kinship_file" in log_content

    def test_cli_gk_invalid_bfile(self, runner, tmp_path):
        """Test error handling for non-existent PLINK file."""
        result = runner.invoke(
            app,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "gk",
                "-bfile",
                str(tmp_path / "nonexistent"),
            ],
        )

        assert result.exit_code != 0

    def test_cli_gk_output_shows_timing(self, runner, tmp_path, example_plink_path):
        """Test that CLI output shows timing information."""
        result = runner.invoke(
            app,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "gk",
                "-bfile",
                str(example_plink_path),
            ],
        )

        assert result.exit_code == 0
        assert "computed in" in result.stdout.lower()
