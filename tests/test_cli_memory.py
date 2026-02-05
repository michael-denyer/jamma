"""Tests for CLI memory pre-flight checks.

Uses subprocess to test memory checks without requiring specific machine
memory sizes.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from jamma.cli import app

runner = CliRunner()

# Test fixture path
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gemma_synthetic"
PLINK_PREFIX = FIXTURE_DIR / "test"
KINSHIP_FILE = FIXTURE_DIR / "gemma_kinship.cXX.txt"


class TestCliMemoryCheck:
    """Tests for CLI lmm command memory pre-flight checks."""

    def test_no_check_memory_bypasses_check(self, tmp_path):
        """--no-check-memory should skip pre-flight check."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "jamma",
                "-o",
                str(tmp_path / "result"),
                "lmm",
                "-bfile",
                str(PLINK_PREFIX),
                "-k",
                str(KINSHIP_FILE),
                "--no-check-memory",
            ],
            capture_output=True,
            text=True,
        )

        # Should not contain "Checking memory requirements"
        assert "Checking memory requirements" not in result.stdout
        # Should succeed (exit code 0) on small test data
        assert result.returncode == 0

    def test_mem_budget_exceeded_fails(self, tmp_path):
        """--mem-budget should fail if estimate exceeds budget."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "jamma",
                "-o",
                str(tmp_path / "result"),
                "lmm",
                "-bfile",
                str(PLINK_PREFIX),
                "-k",
                str(KINSHIP_FILE),
                "--mem-budget",
                "0.0000001",  # Impossibly small: 0.1 bytes
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        output = result.stderr.lower() + result.stdout.lower()
        assert "exceeds budget" in output

    def test_memory_check_enabled_by_default(self, tmp_path):
        """Memory check should run by default."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "jamma",
                "-o",
                str(tmp_path / "result"),
                "lmm",
                "-bfile",
                str(PLINK_PREFIX),
                "-k",
                str(KINSHIP_FILE),
            ],
            capture_output=True,
            text=True,
        )

        # Should contain memory check message
        assert (
            "Checking memory requirements" in result.stdout
            or "Memory estimate" in result.stdout
        )

    def test_memory_check_reports_estimate(self, tmp_path):
        """Memory check should report estimated and available memory."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "jamma",
                "-o",
                str(tmp_path / "result"),
                "lmm",
                "-bfile",
                str(PLINK_PREFIX),
                "-k",
                str(KINSHIP_FILE),
            ],
            capture_output=True,
            text=True,
        )

        # Should contain memory estimate info
        output = result.stdout + result.stderr
        assert "required" in output.lower() or "available" in output.lower()

    def test_check_memory_flag_explicit_enable(self, tmp_path):
        """--check-memory should explicitly enable memory check."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "jamma",
                "-o",
                str(tmp_path / "result"),
                "lmm",
                "-bfile",
                str(PLINK_PREFIX),
                "-k",
                str(KINSHIP_FILE),
                "--check-memory",
            ],
            capture_output=True,
            text=True,
        )

        # Should contain memory check message
        output = result.stdout + result.stderr
        assert "Memory" in output

    def test_fixtures_exist(self):
        """Verify test fixtures exist."""
        assert PLINK_PREFIX.with_suffix(".bed").exists()
        assert KINSHIP_FILE.exists()


@pytest.mark.tier0
class TestCliMemoryCheckUnit:
    """Unit tests for memory check logic (no subprocess)."""

    def test_estimate_called_before_load(self):
        """Memory estimate should be computable from metadata alone."""
        from jamma.core.memory import estimate_streaming_memory
        from jamma.io import get_plink_metadata

        # This simulates what CLI does: get dimensions, then estimate
        meta = get_plink_metadata(PLINK_PREFIX)
        est = estimate_streaming_memory(
            n_samples=meta["n_samples"],
            n_snps=meta["n_snps"],
        )

        assert est.total_peak_gb >= 0
        assert est.available_gb >= 0

    def test_metadata_does_not_load_genotypes(self):
        """get_plink_metadata should only read dimensions, not genotypes."""
        from jamma.io import get_plink_metadata

        # This should be fast and low-memory
        meta = get_plink_metadata(PLINK_PREFIX)

        assert "n_samples" in meta
        assert "n_snps" in meta
        assert meta["n_samples"] == 100
        assert meta["n_snps"] == 500


@pytest.mark.tier0
class TestCliStreamingRunner:
    """Tests for CLI lmm command JAX streaming runner integration."""

    def test_cli_jax_default_uses_streaming_runner(self, tmp_path):
        """Verify CLI calls run_lmm_association_streaming."""
        with patch("jamma.cli.run_lmm_association_streaming") as mock_stream:
            mock_stream.return_value = []

            runner.invoke(
                app,
                [
                    "-outdir",
                    str(tmp_path),
                    "-o",
                    "lmm_test",
                    "lmm",
                    "-bfile",
                    str(PLINK_PREFIX),
                    "-k",
                    str(KINSHIP_FILE),
                    "--no-check-memory",
                ],
            )

            assert mock_stream.called
            call_kwargs = mock_stream.call_args.kwargs
            assert "output_path" in call_kwargs
            assert call_kwargs["output_path"] is not None
            assert str(call_kwargs["output_path"]).endswith(".assoc.txt")
            # JAX path should pass raw data, not filtered
            assert call_kwargs["check_memory"] is False
            assert call_kwargs["snp_info"] is None
