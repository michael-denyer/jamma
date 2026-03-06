"""Pytest fixtures for JAMMA test suite."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest


def load_phenotypes_from_fam(fam_path: Path) -> np.ndarray:
    """Load phenotypes from FAM file (column 6, 0-indexed column 5).

    Handles both GEMMA's missing-phenotype marker (-9) and literal 'NA'
    strings. Returns float64 array with NaN for missing values.

    Args:
        fam_path: Path to .fam PLINK file.

    Returns:
        Array of phenotype values (float64), with -9 and NA replaced by NaN.
    """
    from jamma.core.constants import PHENOTYPE_MISSING

    data = np.loadtxt(fam_path, usecols=5, dtype=str)
    missing = np.isin(data, [str(int(PHENOTYPE_MISSING)), "NA"])
    pheno = np.where(missing, "0", data).astype(np.float64)
    pheno[missing] = np.nan
    return pheno


if TYPE_CHECKING:
    from jamma.validation import ToleranceConfig

# =============================================================================
# Test Tier System
# =============================================================================
#
# JAMMA uses a three-tier test system to balance CI speed with thorough validation:
#
# tier0 - Fast Unit Tests (<5s each)
#   - Pure computation tests (no I/O, no GEMMA reference)
#   - Mocked external dependencies
#   - Run on every commit in CI
#   - Example: test_eigenvalue_thresholding, test_pab_computation
#   - Run: pytest -m tier0
#
# tier1 - Parity Tests (<60s each)
#   - Validates numerical output against GEMMA reference data
#   - Uses fixture files in tests/fixtures/
#   - Run on PRs and merges
#   - Example: test_assoc_matches_gemma, test_kinship_matches_gemma
#   - Run: pytest -m tier1
#
# tier2 - Scale Tests (memory/time intensive)
#   - Large sample counts (10k+ samples)
#   - Memory-constrained scenarios
#   - Run manually or in nightly CI with large VMs
#   - Example: test_streaming_large_dataset, test_memory_estimation_accuracy
#   - Run: pytest -m tier2
#
# @pytest.mark.slow is independent (not tied to a specific tier).
#
# Quick reference:
#   pytest -m tier0           # Fast tests only (~30s total)
#   pytest -m "tier0 or tier1"  # All fast + parity tests
#   pytest -m "not tier2"     # Exclude slow/memory tests
#   pytest                    # All tests
# =============================================================================


def pytest_runtest_setup(item):
    """Auto-skip tests marked requires_jax when JAX is not installed."""
    if any(m.name == "requires_jax" for m in item.iter_markers()):
        try:
            import jax  # noqa: F401
        except ImportError:
            pytest.skip("JAX not installed — install with: pip install jamma[jax]")


@pytest.fixture(scope="session", autouse=True)
def _configure_jax_for_tests():
    """Ensure JAX 64-bit precision is configured for all tests.

    Previously, importing jamma.kinship.compute triggered module-level
    configure_jax(). Now that side effects are removed, this fixture
    provides the same guarantee explicitly.

    When JAX is not installed (NumPy-only environment), this fixture
    silently passes — tests that need JAX will fail with their own
    ImportError, providing clear error messages.
    """
    try:
        from jamma.core.jax_config import ensure_jax_configured

        ensure_jax_configured()
    except ImportError:
        pass  # JAX not installed; NumPy backend tests run without configuration


@pytest.fixture
def sample_plink_data() -> Path:
    """Return path prefix for sample PLINK data from test fixtures.

    Returns:
        Path prefix for gemma_synthetic PLINK files (without .bed/.bim/.fam extension)
    """
    return Path(__file__).parent / "fixtures" / "gemma_synthetic" / "test"


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory for test results.

    Args:
        tmp_path: pytest's temporary path fixture

    Returns:
        Path to output directory
    """
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def tolerance_config() -> ToleranceConfig:
    """Default tolerance configuration for numerical comparisons.

    Returns:
        ToleranceConfig with default tolerance values for different comparison types
    """
    from jamma.validation import ToleranceConfig

    return ToleranceConfig()


@pytest.fixture(scope="session")
def validation_pipeline_data():
    """Run LMM pipeline once on gemma_synthetic data for validation tests.

    Returns dict with keys: jamma_results, reference_results, comparison.

    Session-scoped: all validation tests share one pipeline run, avoiding
    the cost of running the full association pipeline 3x.

    Returns None if reference data is not available (tests should skip).
    """

    fixture_root = Path(__file__).parent / "fixtures"
    example_data = fixture_root / "gemma_synthetic" / "test"
    reference_assoc = fixture_root / "gemma_synthetic" / "gemma_assoc.assoc.txt"

    if not reference_assoc.exists():
        return None

    try:
        import jax  # noqa: F401
    except ImportError:
        return None  # JAX not installed; validation tests will skip

    from jamma.io import load_plink_binary
    from jamma.kinship.io import read_kinship_matrix
    from jamma.lmm.runner_jax import run_lmm_association_jax
    from jamma.validation import (
        ToleranceConfig,
        compare_assoc_results,
        load_gemma_assoc,
    )

    reference_kinship = fixture_root / "gemma_synthetic" / "gemma_kinship.cXX.txt"

    plink_data = load_plink_binary(example_data)
    kinship = read_kinship_matrix(reference_kinship)
    reference_results = load_gemma_assoc(reference_assoc)

    phenotypes = load_phenotypes_from_fam(example_data.with_suffix(".fam"))

    # Build SNP info
    snp_info = [
        {
            "chr": str(plink_data.chromosome[i]),
            "rs": plink_data.sid[i],
            "pos": plink_data.bp_position[i],
            "a1": plink_data.allele_1[i],
            "a0": plink_data.allele_2[i],
            "maf": 0.0,
            "n_miss": 0,
        }
        for i in range(plink_data.n_snps)
    ]

    run_result = run_lmm_association_jax(
        genotypes=plink_data.genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        show_progress=False,
        check_memory=False,
    )
    jamma_results = run_result.associations

    jax_tolerances = ToleranceConfig(lambda_rtol=5e-5)
    comparison = compare_assoc_results(
        jamma_results, reference_results, config=jax_tolerances
    )

    result = {
        "jamma_results": jamma_results,
        "reference_results": reference_results,
        "comparison": comparison,
    }

    return result


def _build_synthetic_covariate_data(
    n_cvt: int,
    n_samples: int = 200,
    n_snps: int = 50,
    seed: int = 42,
) -> dict:
    """Build synthetic rotated data for C extension testing.

    Generates eigenvalues, rotated covariates (UtW), phenotype (Uty),
    genotypes (UtG), and computes Uab_batch for the given n_cvt.

    Args:
        n_cvt: Number of covariates.
        n_samples: Number of samples.
        n_snps: Number of SNPs.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys: eigenvalues, UtW, Uty, UtG, Uab_batch,
        n_samples, n_snps, n_cvt.
    """
    from jamma.lmm.likelihood import compute_Uab

    rng = np.random.default_rng(seed)

    eigenvalues = np.sort(rng.uniform(0.1, 2.0, n_samples))[::-1]  # descending
    UtW = np.abs(rng.standard_normal((n_samples, n_cvt))) + 0.5
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Compute Uab for each SNP
    n_index = (n_cvt + 3) * (n_cvt + 2) // 2
    Uab_batch = np.zeros((n_snps, n_samples, n_index), dtype=np.float64)
    for i in range(n_snps):
        Uab_batch[i] = compute_Uab(UtW, Uty, UtG[:, i])

    return {
        "eigenvalues": eigenvalues,
        "UtW": UtW,
        "Uty": Uty,
        "UtG": UtG,
        "Uab_batch": Uab_batch,
        "n_samples": n_samples,
        "n_snps": n_snps,
        "n_cvt": n_cvt,
    }


@pytest.fixture
def synthetic_covariate_data_ncvt2() -> dict:
    """Synthetic data with 2 covariates for C extension testing.

    200 samples, 50 SNPs, 2 covariates. Returns dict with
    eigenvalues, UtW, Uty, UtG, Uab_batch, n_samples, n_snps, n_cvt.
    """
    return _build_synthetic_covariate_data(n_cvt=2, seed=42)


@pytest.fixture
def synthetic_covariate_data_ncvt4() -> dict:
    """Synthetic data with 4 covariates for C extension testing.

    200 samples, 50 SNPs, 4 covariates. Returns dict with
    eigenvalues, UtW, Uty, UtG, Uab_batch, n_samples, n_snps, n_cvt.
    """
    return _build_synthetic_covariate_data(n_cvt=4, seed=99)
