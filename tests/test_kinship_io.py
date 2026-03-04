"""Tests for kinship I/O and CLI integration.

These tests verify the GEMMA-format kinship output and CLI end-to-end workflow.
"""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from jamma.cli import main
from jamma.kinship import write_kinship_matrix
from jamma.kinship.io import write_loco_kinship_matrices


@pytest.fixture
def runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def example_plink_path() -> Path:
    """Path prefix for example PLINK files."""
    return Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


@pytest.mark.tier0
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

    def test_savetxt_matches_python_loop(self, tmp_path):
        """np.savetxt output must be byte-identical to the old Python loop.

        Guards against format drift if np.savetxt ever changes its %.10g
        rendering. Uses a 50x50 random matrix for non-trivial coverage.
        """
        rng = np.random.default_rng(12345)
        K = rng.standard_normal((50, 50))
        K = (K + K.T) / 2  # Make symmetric like a real kinship matrix

        # Write with np.savetxt (current implementation)
        savetxt_path = tmp_path / "savetxt.cXX.txt"
        write_kinship_matrix(K, savetxt_path)

        # Write with old Python loop
        loop_path = tmp_path / "loop.cXX.txt"
        with open(loop_path, "w") as f:
            for i in range(K.shape[0]):
                values = [f"{K[i, j]:.10g}" for j in range(K.shape[1])]
                f.write("\t".join(values) + "\n")

        assert savetxt_path.read_bytes() == loop_path.read_bytes()


@pytest.mark.tier0
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


@pytest.mark.tier0
class TestReadKinshipValidation:
    """Tests for read_kinship_matrix error paths (non-square, asymmetric, mismatch)."""

    def test_read_kinship_non_square_raises(self, tmp_path):
        """Non-square matrix file raises ValueError mentioning 'square'."""
        from jamma.kinship import read_kinship_matrix

        path = tmp_path / "nonsquare.cXX.txt"
        K_rect = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]], dtype=np.float64)
        np.savetxt(path, K_rect, fmt="%.10g", delimiter="\t")

        with pytest.raises(ValueError, match="square"):
            read_kinship_matrix(path)

    def test_read_kinship_non_symmetric_raises(self, tmp_path):
        """Asymmetric matrix raises ValueError mentioning 'symmetric'."""
        from jamma.kinship import read_kinship_matrix

        path = tmp_path / "asymm.cXX.txt"
        K_asym = np.array(
            [[1.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.2, 0.3, 1.0]], dtype=np.float64
        )
        np.savetxt(path, K_asym, fmt="%.10g", delimiter="\t")

        with pytest.raises(ValueError, match="symmetric"):
            read_kinship_matrix(path)

    def test_read_kinship_dimension_mismatch_raises(self, tmp_path):
        """Correct symmetric 5x5 matrix with n_samples=10 raises dimension error."""
        from jamma.kinship import read_kinship_matrix

        path = tmp_path / "dim_mismatch.cXX.txt"
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        K = (A @ A.T) / 5
        K = (K + K.T) / 2
        np.savetxt(path, K, fmt="%.10g", delimiter="\t")

        with pytest.raises(ValueError, match="dimension"):
            read_kinship_matrix(path, n_samples=10)

    def test_read_kinship_valid_roundtrip(self, tmp_path):
        """Valid symmetric PSD matrix round-trips through write/read correctly."""
        from jamma.kinship import read_kinship_matrix

        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        K = (A @ A.T) / 5
        K = (K + K.T) / 2  # Ensure exact symmetry

        path = tmp_path / "valid.cXX.txt"
        write_kinship_matrix(K, path)

        K_loaded = read_kinship_matrix(path, n_samples=5)

        assert K_loaded.shape == (5, 5)
        np.testing.assert_allclose(K_loaded, K, rtol=1e-9)


@pytest.mark.tier0
class TestWriteLocoKinshipMatrices:
    """Tests for write_loco_kinship_matrices."""

    def _make_kinship(self, n: int = 4, seed: int = 0) -> np.ndarray:
        """Return a small symmetric PSD kinship matrix."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        K = (A @ A.T) / n
        return (K + K.T) / 2

    def test_writes_one_file_per_chromosome(self, tmp_path):
        """One output file is written for each (chr, K) pair."""
        loco_iter = (
            ("1", self._make_kinship(seed=1)),
            ("2", self._make_kinship(seed=2)),
            ("X", self._make_kinship(seed=3)),
        )

        paths = write_loco_kinship_matrices(loco_iter, tmp_path)

        assert len(paths) == 3

    def test_output_filenames_use_expected_pattern(self, tmp_path):
        """Files are named {prefix}.loco.cXX.chr{chr}.txt."""
        loco_iter = [
            ("1", self._make_kinship(seed=0)),
            ("22", self._make_kinship(seed=1)),
        ]

        paths = write_loco_kinship_matrices(loco_iter, tmp_path, prefix="study")

        names = {p.name for p in paths}
        assert "study.loco.cXX.chr1.txt" in names
        assert "study.loco.cXX.chr22.txt" in names

    def test_default_prefix_is_result(self, tmp_path):
        """Default prefix is 'result'."""
        loco_iter = [("3", self._make_kinship())]

        paths = write_loco_kinship_matrices(loco_iter, tmp_path)

        assert paths[0].name == "result.loco.cXX.chr3.txt"

    def test_output_directory_created_if_missing(self, tmp_path):
        """Output directory (and parents) are created if they don't exist."""
        output_dir = tmp_path / "nested" / "loco"
        loco_iter = [("1", self._make_kinship())]

        write_loco_kinship_matrices(loco_iter, output_dir)

        assert output_dir.exists()

    def test_matrix_content_roundtrips(self, tmp_path):
        """Written kinship matrices load back with GEMMA-format precision."""
        from jamma.kinship.io import read_kinship_matrix

        K1 = self._make_kinship(n=5, seed=10)
        K2 = self._make_kinship(n=5, seed=20)
        loco_iter = [("1", K1.copy()), ("2", K2.copy())]

        paths = write_loco_kinship_matrices(loco_iter, tmp_path)

        for path, K_expected in zip(paths, [K1, K2], strict=True):
            K_loaded = read_kinship_matrix(path)
            np.testing.assert_allclose(K_loaded, K_expected, rtol=1e-9)

    def test_returns_paths_in_iterator_order(self, tmp_path):
        """Returned paths preserve the same order as the input iterator."""
        chromosomes = ["5", "3", "1"]
        loco_iter = [(c, self._make_kinship(seed=i)) for i, c in enumerate(chromosomes)]

        paths = write_loco_kinship_matrices(loco_iter, tmp_path)

        for path, chr_name in zip(paths, chromosomes, strict=True):
            assert f"chr{chr_name}" in path.name

    def test_empty_iterator_returns_empty_list(self, tmp_path):
        """Empty iterator produces an empty result list."""
        paths = write_loco_kinship_matrices(iter([]), tmp_path)

        assert paths == []


@pytest.mark.tier1
class TestCLIIntegration:
    """Tests for CLI gk command integration."""

    def test_cli_gk_creates_kinship_file(self, runner, tmp_path, example_plink_path):
        """Test that gk command creates kinship file."""
        result = runner.invoke(
            main,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "-gk",
                "1",
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
            main,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "-gk",
                "1",
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
            main,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "-gk",
                "1",
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
            main,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "-gk",
                "1",
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
            main,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "-gk",
                "1",
                "-bfile",
                str(tmp_path / "nonexistent"),
            ],
        )

        assert result.exit_code != 0

    def test_cli_gk_output_shows_timing(self, runner, tmp_path, example_plink_path):
        """Test that CLI output shows timing information."""
        result = runner.invoke(
            main,
            [
                "-outdir",
                str(tmp_path),
                "-o",
                "test",
                "-gk",
                "1",
                "-bfile",
                str(example_plink_path),
            ],
        )

        assert result.exit_code == 0
        assert "computed in" in result.stdout.lower()
