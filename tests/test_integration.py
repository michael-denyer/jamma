"""Integration tests for JAMMA.

These tests verify that all components work together correctly:
- CLI invokes I/O and logging properly
- Data flows through the full pipeline
- Output files match expected format
- All modules import without errors
- JAX computations work with real data
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from typer.testing import CliRunner

from jamma.cli import app

runner = CliRunner()

# Path to example PLINK data
EXAMPLE_BFILE = Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


class TestGkWorkflow:
    """Integration tests for the gk (kinship) workflow."""

    def test_gk_workflow_creates_output(self, tmp_path: Path) -> None:
        """Run CLI gk command and verify output directory and log file created."""
        outdir = tmp_path / "output"

        result = runner.invoke(
            app, ["-outdir", str(outdir), "gk", "-bfile", str(EXAMPLE_BFILE)]
        )

        assert result.exit_code == 0
        assert outdir.exists(), "Output directory should be created"
        assert (outdir / "result.log.txt").exists(), "Log file should be created"

    def test_gk_workflow_log_contents(self, tmp_path: Path) -> None:
        """Run CLI gk command and verify log file contents match GEMMA format."""
        outdir = tmp_path / "output"

        result = runner.invoke(
            app, ["-outdir", str(outdir), "gk", "-bfile", str(EXAMPLE_BFILE)]
        )

        assert result.exit_code == 0

        log_path = outdir / "result.log.txt"
        log_content = log_path.read_text()

        # Verify GEMMA-style format with ## prefixes
        assert log_content.startswith("##"), "Log should start with ## prefix"
        assert "## JAMMA Version" in log_content, "Log should contain version"
        assert "## Date" in log_content, "Log should contain date"
        assert "## Command Line Input" in log_content, "Log should contain command line"
        assert "## total time" in log_content, "Log should contain timing"
        assert "n_samples = 100" in log_content, "Log should contain sample count"
        assert "n_snps = 500" in log_content, "Log should contain SNP count"


class TestPlinkToValidationRoundtrip:
    """Integration tests for data loading, processing, and validation."""

    def test_plink_to_kinship_roundtrip(
        self, sample_plink_data: Path, tmp_path: Path
    ) -> None:
        """Load PLINK data, create kinship matrix, save/reload, and compare."""
        from jamma.io import load_plink_binary
        from jamma.validation import compare_kinship_matrices

        # Load PLINK data
        plink_data = load_plink_binary(sample_plink_data)

        # Create synthetic kinship matrix: K = X @ X.T / p
        # Use a small subset for speed (first 100 samples, 1000 SNPs)
        X = plink_data.genotypes[:100, :1000].astype(np.float64)

        # Handle missing values by mean imputation for this test
        col_means = np.nanmean(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Center genotypes
        X_centered = X - np.mean(X, axis=0)

        # Compute kinship
        p = X.shape[1]
        K_original = X_centered @ X_centered.T / p

        # Save in GEMMA format
        kinship_path = tmp_path / "test.cXX.txt"
        np.savetxt(kinship_path, K_original, fmt="%.15e")

        # Reload
        from jamma.validation import load_gemma_kinship

        K_loaded = load_gemma_kinship(kinship_path)

        # Compare with tolerance
        result = compare_kinship_matrices(K_original, K_loaded)
        assert result.passed, f"Kinship roundtrip failed: {result.message}"


class TestFullModuleImports:
    """Integration tests for module imports."""

    def test_jamma_package_imports(self) -> None:
        """Verify main package imports without errors."""
        import jamma

        assert hasattr(jamma, "__version__")
        # Version should be a valid semver string (e.g., "0.1.0", "1.0.0-dev")
        assert isinstance(jamma.__version__, str)
        assert len(jamma.__version__) > 0

    def test_io_module_imports(self) -> None:
        """Verify I/O module imports without errors."""
        from jamma.io import PlinkData, load_plink_binary

        assert PlinkData is not None
        assert load_plink_binary is not None

    def test_core_module_imports(self) -> None:
        """Verify core module imports without errors."""
        from jamma.core import (
            OutputConfig,
            configure_jax,
            get_jax_info,
            verify_jax_installation,
        )

        assert OutputConfig is not None
        assert configure_jax is not None
        assert get_jax_info is not None
        assert verify_jax_installation is not None

    def test_utils_module_imports(self) -> None:
        """Verify utils module imports without errors."""
        from jamma.utils import setup_logging, write_gemma_log

        assert setup_logging is not None
        assert write_gemma_log is not None

    def test_validation_module_imports(self) -> None:
        """Verify validation module imports without errors."""
        from jamma.validation import (
            ComparisonResult,
            ToleranceConfig,
            compare_arrays,
            compare_kinship_matrices,
            load_gemma_kinship,
        )

        assert ComparisonResult is not None
        assert ToleranceConfig is not None
        assert compare_arrays is not None
        assert compare_kinship_matrices is not None
        assert load_gemma_kinship is not None


class TestJaxWithPlinkData:
    """Integration tests for JAX operations with real PLINK data."""

    def test_jax_with_plink_genotypes(self, sample_plink_data: Path) -> None:
        """Load PLINK data, convert to JAX, and run computations."""
        from jamma.core import configure_jax
        from jamma.io import load_plink_binary

        # Ensure JAX is configured for 64-bit
        configure_jax(enable_x64=True)

        # Load PLINK data
        plink_data = load_plink_binary(sample_plink_data)

        # Convert subset to JAX array (first 100 samples, 100 SNPs for speed)
        genotypes_np = plink_data.genotypes[:100, :100].astype(np.float64)

        # Handle NaN for JAX (replace with column mean)
        col_means = np.nanmean(genotypes_np, axis=0)
        nan_mask = np.isnan(genotypes_np)
        genotypes_np[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        genotypes_jax = jnp.array(genotypes_np)

        # Run basic statistics
        mean_per_snp = jnp.mean(genotypes_jax, axis=0)
        std_per_snp = jnp.std(genotypes_jax, axis=0)

        # Verify results are valid
        assert mean_per_snp.shape == (100,), "Mean should have one value per SNP"
        assert std_per_snp.shape == (100,), "Std should have one value per SNP"
        assert jnp.all(jnp.isfinite(mean_per_snp)), "Mean values should be finite"
        assert jnp.all(jnp.isfinite(std_per_snp)), "Std values should be finite"

        # Verify means are in reasonable range for genotypes (0-2)
        assert jnp.all(mean_per_snp >= 0), "Genotype means should be >= 0"
        assert jnp.all(mean_per_snp <= 2), "Genotype means should be <= 2"

    def test_jax_matrix_operations_with_plink(self, sample_plink_data: Path) -> None:
        """Verify JAX matrix operations work correctly with real genotype data."""
        from jamma.core import configure_jax
        from jamma.io import load_plink_binary

        configure_jax(enable_x64=True)

        # Load and prepare data
        plink_data = load_plink_binary(sample_plink_data)
        genotypes_np = plink_data.genotypes[:50, :100].astype(np.float64)

        # Handle NaN
        col_means = np.nanmean(genotypes_np, axis=0)
        nan_mask = np.isnan(genotypes_np)
        genotypes_np[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        X = jnp.array(genotypes_np)

        # Compute kinship-like matrix: K = X @ X.T / p
        p = X.shape[1]
        K = X @ X.T / p

        # Verify shape and properties
        assert K.shape == (50, 50), "Kinship should be n_samples x n_samples"
        assert jnp.allclose(K, K.T), "Kinship should be symmetric"
        assert jnp.all(jnp.isfinite(K)), "Kinship values should be finite"
