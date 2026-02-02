"""Tests for NumPy LMM runner incremental writing.

Verifies that run_lmm_association() with output_path writes results
incrementally to disk instead of accumulating in memory.
"""

import numpy as np
import pytest

from jamma.lmm import run_lmm_association, write_assoc_results


@pytest.fixture
def small_gwas_data():
    """Create small GWAS dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    n_snps = 50

    # Genotypes: 0, 1, 2 with some variation
    genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_snps)).astype(float)

    # Phenotypes: continuous
    phenotypes = np.random.randn(n_samples)

    # Kinship: identity-ish matrix (simple for testing)
    kinship = (
        np.eye(n_samples) * 0.5 + 0.5 * np.random.randn(n_samples, n_samples) * 0.01
    )
    kinship = (kinship + kinship.T) / 2  # Make symmetric

    # SNP info
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
        for i in range(n_snps)
    ]

    return {
        "genotypes": genotypes,
        "phenotypes": phenotypes,
        "kinship": kinship,
        "snp_info": snp_info,
    }


class TestRunLmmAssociationOutputPath:
    """Tests for output_path parameter in run_lmm_association()."""

    def test_output_path_none_returns_list(self, small_gwas_data):
        """When output_path=None, should return list of results."""
        results = run_lmm_association(
            genotypes=small_gwas_data["genotypes"],
            phenotypes=small_gwas_data["phenotypes"],
            kinship=small_gwas_data["kinship"],
            snp_info=small_gwas_data["snp_info"],
            output_path=None,
        )

        assert isinstance(results, list)
        assert len(results) > 0

    def test_output_path_writes_to_file(self, small_gwas_data, tmp_path):
        """When output_path provided, should write results to file."""
        output_file = tmp_path / "test_results.assoc.txt"

        results = run_lmm_association(
            genotypes=small_gwas_data["genotypes"],
            phenotypes=small_gwas_data["phenotypes"],
            kinship=small_gwas_data["kinship"],
            snp_info=small_gwas_data["snp_info"],
            output_path=output_file,
        )

        assert results == []  # Returns empty when writing to file
        assert output_file.exists()

        # File should have content
        content = output_file.read_text()
        lines = content.strip().split("\n")
        assert len(lines) > 1  # Header + data

    def test_output_path_returns_empty_list(self, small_gwas_data, tmp_path):
        """When output_path provided, should return empty list."""
        output_file = tmp_path / "test_results.assoc.txt"

        results = run_lmm_association(
            genotypes=small_gwas_data["genotypes"],
            phenotypes=small_gwas_data["phenotypes"],
            kinship=small_gwas_data["kinship"],
            snp_info=small_gwas_data["snp_info"],
            output_path=output_file,
        )

        assert results == []

    def test_output_format_matches_batch_writer(self, small_gwas_data, tmp_path):
        """Output from incremental writing should match batch write_assoc_results()."""
        incremental_file = tmp_path / "incremental.assoc.txt"
        batch_file = tmp_path / "batch.assoc.txt"

        # Get results in memory
        results = run_lmm_association(
            genotypes=small_gwas_data["genotypes"],
            phenotypes=small_gwas_data["phenotypes"],
            kinship=small_gwas_data["kinship"],
            snp_info=small_gwas_data["snp_info"],
            output_path=None,
        )

        # Write batch
        write_assoc_results(results, batch_file)

        # Write incremental
        run_lmm_association(
            genotypes=small_gwas_data["genotypes"],
            phenotypes=small_gwas_data["phenotypes"],
            kinship=small_gwas_data["kinship"],
            snp_info=small_gwas_data["snp_info"],
            output_path=incremental_file,
        )

        # Compare files
        incremental_content = incremental_file.read_text()
        batch_content = batch_file.read_text()

        assert incremental_content == batch_content

    def test_output_path_with_covariates(self, small_gwas_data, tmp_path):
        """output_path should work with covariates."""
        output_file = tmp_path / "test_with_cov.assoc.txt"

        # Add covariates (intercept + one covariate)
        n_samples = small_gwas_data["genotypes"].shape[0]
        covariates = np.column_stack(
            [
                np.ones(n_samples),  # Intercept
                np.random.randn(n_samples),  # Random covariate
            ]
        )

        results = run_lmm_association(
            genotypes=small_gwas_data["genotypes"],
            phenotypes=small_gwas_data["phenotypes"],
            kinship=small_gwas_data["kinship"],
            snp_info=small_gwas_data["snp_info"],
            covariates=covariates,
            output_path=output_file,
        )

        assert results == []
        assert output_file.exists()

    def test_backward_compatibility_no_output_path(self, small_gwas_data):
        """Existing calls without output_path should still work."""
        # This call pattern should work unchanged
        results = run_lmm_association(
            small_gwas_data["genotypes"],
            small_gwas_data["phenotypes"],
            small_gwas_data["kinship"],
            small_gwas_data["snp_info"],
        )

        assert isinstance(results, list)
